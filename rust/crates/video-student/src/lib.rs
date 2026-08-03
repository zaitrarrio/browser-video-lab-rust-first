use burn::{nn::{Gelu, Initializer, LayerNorm, LayerNormConfig, Linear, LinearConfig}, prelude::*};
use video_contract::StudentSpec;

pub mod quant;

/// Query rows per attention tile. 1024 is measured, not guessed: swept on an
/// RTX 5090 (CUDA backend, seq=1600, heads=16, 24 layers, 40 steps × accum 8)
/// the peak-memory/throughput pairs were 256 → 25638 MiB / 107.8 s, 512 → 24646
/// MiB / 100.7 s, 1024 → 24006 MiB / 100.4 s, and ≥1600 (i.e. untiled) → 25638
/// MiB / 103.3 s. Small tiles lose on both axes — the score matmul goes
/// launch-latency-bound and the extra allocations defeat the pool's reuse — so
/// the useful range is "as few tiles as still bounds the live matrix", and 1024
/// also leaves every seq ≤ 1024 on the untiled path, where tiling measured
/// slightly *worse* (seq=800: 13414 MiB tiled vs 13158 MiB untiled).
pub const DEFAULT_ATTENTION_BLOCK: usize = 1024;

/// The tile width is deliberately *not* part of `StudentSpec`: it changes no
/// output, only how the same arithmetic is scheduled, so baking it into the
/// serialized model contract would mean checkpoints disagreeing about something
/// that is a property of the device you happen to be running on. The env
/// override exists so a block-size sweep is a re-run and not a re-build; an
/// unparseable or zero value falls back to the default rather than failing a
/// long training run over a typo'd variable.
pub fn attention_block() -> usize {
    use std::sync::OnceLock;
    static BLOCK: OnceLock<usize> = OnceLock::new();
    *BLOCK.get_or_init(|| {
        std::env::var("VIDEO_ATTENTION_BLOCK").ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|v| *v > 0)
            .unwrap_or(DEFAULT_ATTENTION_BLOCK)
    })
}

/// Scaled-dot-product attention over blocks of query rows.
///
/// The single-shot form materialises the full `[b, heads, seq, seq]` f32
/// probability matrix: at the training geometry that is 164 MB *per layer*, and
/// autodiff holds all 24 of them for the backward pass — 3.9 GB for one sample,
/// which is the wall the trainer hits before it can carry a real batch
/// dimension. Softmax along dim 3 normalises each query row independently, so a
/// block of query rows can be computed and normalised without ever seeing the
/// other rows' scores: tiling is exact, not an approximation, and the only
/// numerical difference is matmul tile ordering (well under 1e-5).
///
/// This is query-block tiling, not flash attention — Burn 0.21 has no fused
/// kernel, so autodiff still tapes every tile's intermediates and the taped
/// total is unchanged. What tiling buys is the *transient* peak and allocator
/// reuse, and that is worth far less than the arithmetic suggests: measured on
/// an RTX 5090, peak GPU memory went 25638 → 24006 MiB (−6.4%) and throughput
/// 3.10 → 3.19 samples/s (+2.9%). Do not read the 164 MB/layer figure above as
/// headroom recovered. Subtracting the fixed 7910 MiB of weights, gradients,
/// accumulator and Adam state (measured with a seq=16 cache), one sample still
/// holds ~16.1 GB of activations, of which ~14.5 GB is attention — because
/// `softmax` tapes roughly three-and-a-half `[b, heads, seq, seq]` tensors per
/// layer (pre-softmax scores, `exp`, the normalised result) no matter how the
/// rows are sliced. A true batch of 2 would need ~40 GB and still does not fit
/// in 32 GB; the thing that would actually unlock a batch dimension is a fused
/// softmax whose backward recomputes rather than stores, not a wider tile.
///
/// `seq <= block` (the browser's own geometry, and every test spec) takes the
/// original single-shot path so nothing pays for slicing it cannot use.
pub fn tiled_attention<B: Backend>(q: Tensor<B, 4>, k: Tensor<B, 4>, v: Tensor<B, 4>, scale: f64, block: usize) -> Tensor<B, 4> {
    let [b, heads, seq, head_dim] = q.dims();
    let kt = k.swap_dims(2, 3); // [b, heads, head_dim, seq] — hoisted: every tile scores against all of K
    if block == 0 || seq <= block {
        return burn::tensor::activation::softmax(q.matmul(kt) / scale, 3).matmul(v);
    }
    let mut out = Vec::with_capacity(seq.div_ceil(block));
    let mut start = 0;
    while start < seq {
        let end = (start + block).min(seq); // last tile is short whenever block ∤ seq
        let rows = q.clone().slice([0..b, 0..heads, start..end, 0..head_dim]);
        let probs = burn::tensor::activation::softmax(rows.matmul(kt.clone()) / scale, 3); // [b, heads, end-start, seq]
        out.push(probs.matmul(v.clone()));
        start = end;
    }
    Tensor::cat(out, 2)
}

#[derive(Module, Debug)]
pub struct MixerBlock<B: Backend> {
    norm: LayerNorm<B>, q: Linear<B>, k: Linear<B>, v: Linear<B>, proj: Linear<B>,
    norm_mlp: LayerNorm<B>, up: Linear<B>, down: Linear<B>, activation: Gelu,
    /// adaLN-zero modulation: `width → 6·width`, split into shift/scale/gate for
    /// the attention and MLP halves. `None` is the historical architecture, where
    /// conditioning is added once at the stem and never reaches a block directly.
    ///
    /// Zero-initialised on purpose, which is the "zero" in adaLN-zero: at step 0
    /// both gates are 0, so every block is an exact identity and the network starts
    /// as the residual stream alone. That is what keeps adding 6·width·width fresh
    /// parameters per block from destabilising early training — the alternative,
    /// random init, multiplies the residual by an arbitrary scale in all 24 blocks
    /// before a single gradient has been applied.
    ada: Option<Linear<B>>,
}
impl<B: Backend> MixerBlock<B> {
    fn new(width: usize, mlp_ratio: usize, per_block_conditioning: bool, device: &B::Device) -> Self { Self {
        norm: LayerNormConfig::new(width).init(device), q: LinearConfig::new(width,width).init(device),
        k: LinearConfig::new(width,width).init(device), v: LinearConfig::new(width,width).init(device),
        proj: LinearConfig::new(width,width).init(device), norm_mlp: LayerNormConfig::new(width).init(device),
        up: LinearConfig::new(width,width*mlp_ratio).init(device), down: LinearConfig::new(width*mlp_ratio,width).init(device), activation:Gelu::new(),
        ada: per_block_conditioning.then(|| LinearConfig::new(width, 6*width)
            .with_initializer(Initializer::Zeros).init(device)),
    }}
    // Multi-head scaled-dot-product attention. `heads` and `block` come from the
    // caller so the block stays weightless-of-config (`block` is the query-tile
    // width; see `tiled_attention`). Attention is bidirectional within the
    // chunk by design (rust/README.md): causality lives at the streaming level,
    // not in an intra-chunk mask, which keeps browser attention cheap.
    fn forward(&self, x: Tensor<B,3>, cond: Tensor<B,3>, heads: usize, block: usize) -> Tensor<B,3> {
        let [b,seq,width]=x.dims(); let head_dim=width/heads; let scale=(head_dim as f64).sqrt();
        // Six [b,1,width] modulation terms from the conditioning vector, broadcast
        // over tokens. Without `ada` these are the identity (shift 0, scale 1,
        // gate 1), so this path reduces exactly to the historical block.
        let m6 = self.ada.as_ref().map(|ada| {
            let p = ada.forward(cond).reshape([b,1,6,width]);
            let take = |i: usize| p.clone().slice([0..b, 0..1, i..i+1, 0..width]).reshape([b,1,width]);
            [take(0),take(1),take(2),take(3),take(4),take(5)]
        });
        let modulate = |t: Tensor<B,3>, shift: Option<Tensor<B,3>>, scale: Option<Tensor<B,3>>| match (shift,scale) {
            (Some(sh),Some(sc)) => t*(sc+1.0)+sh,   // scale is an offset from 1, so zero-init is identity
            _ => t,
        };
        let gate = |t: Tensor<B,3>, g: Option<Tensor<B,3>>| match g { Some(g)=>t*g, None=>t };

        let n=modulate(self.norm.forward(x.clone()), m6.as_ref().map(|m|m[0].clone()), m6.as_ref().map(|m|m[1].clone()));
        let split=|t: Tensor<B,3>| t.reshape([b,seq,heads,head_dim]).swap_dims(1,2); // [b, heads, seq, head_dim]
        let q=split(self.q.forward(n.clone())); let k=split(self.k.forward(n.clone())); let v=split(self.v.forward(n));
        let context=tiled_attention(q,k,v,scale,block).swap_dims(1,2).reshape([b,seq,width]);
        // Gate is applied to the branch, not the residual: at zero-init the block
        // returns `x` unchanged rather than annihilating the stream.
        let x=x+gate(self.proj.forward(context), m6.as_ref().map(|m|m[2].clone()));
        let m=modulate(self.norm_mlp.forward(x.clone()), m6.as_ref().map(|m|m[3].clone()), m6.as_ref().map(|m|m[4].clone()));
        x+gate(self.down.forward(self.activation.forward(self.up.forward(m))), m6.as_ref().map(|m|m[5].clone()))
    }
}

#[derive(Module, Debug)]
pub struct BrowserVideoStudent<B: Backend> {
    input: Linear<B>, text: Linear<B>, time: Linear<B>, blocks: Vec<MixerBlock<B>>, norm: LayerNorm<B>, output: Linear<B>, spec: StudentSpec,
}
impl<B: Backend> BrowserVideoStudent<B> {
    pub fn new(spec: StudentSpec, device:&B::Device)->Self { spec.validate().expect("valid spec"); let token=spec.latent_channels*spec.patch_volume(); Self {
        input:LinearConfig::new(token,spec.width).init(device),text:LinearConfig::new(spec.text_width,spec.width).init(device),time:LinearConfig::new(1,spec.width).init(device),
        blocks:(0..spec.layers).map(|_|MixerBlock::new(spec.width,spec.mlp_ratio,spec.per_block_conditioning,device)).collect(),norm:LayerNormConfig::new(spec.width).init(device),output:LinearConfig::new(spec.width,token).init(device),spec,
    }}
    // Patchify `[b,c,t,h,w]` into `t·(h/ph)·(w/pw)` tokens of `c·ph·pw` (Wan uses
    // patch_size [1,2,2]), run the mixer, then unpatchify back to full latent
    // resolution. Patchify is what makes the student's token count — and its
    // relation grams — match a real Wan teacher; the flatten it replaces did not.
    pub fn forward(&self, latents:Tensor<B,5>, timestep:Tensor<B,2>, prompt:Tensor<B,3>)->(Tensor<B,5>,Vec<Tensor<B,3>>){
        let [b,c,t,h,w]=latents.dims();let [pt,ph,pw]=self.spec.patch_size; // pt==1 (validated)
        assert!(t%pt==0&&h%ph==0&&w%pw==0,"latent {t}x{h}x{w} not divisible by patch_size {pt}x{ph}x{pw}");
        let (hp,wp)=(h/ph,w/pw);let tokens=t*hp*wp;let pv=c*ph*pw;
        // Patchify staying within the NdArray 6-dim ceiling: fold b·t, split h/w in
        // one reshape, group spatial patch cells (c,ph,pw) into the token vector.
        // [b,t,c,hp,ph,wp,pw]→permute→[b·t,hp,wp,c,ph,pw]→[b, t·hp·wp, c·ph·pw]
        let mut x=latents.swap_dims(1,2).reshape([b*t,c,hp,ph,wp,pw]).permute([0,2,4,1,3,5]).reshape([b,tokens,pv]);
        let cond=self.text.forward(prompt.mean_dim(1))+self.time.forward(timestep/1000.0).unsqueeze_dim(1);x=self.input.forward(x)+cond.clone();
        let mut hidden=Vec::with_capacity(self.blocks.len());let tile=attention_block();for block in &self.blocks{x=block.forward(x,cond.clone(),self.spec.heads,tile);hidden.push(x.clone())}
        // Inverse patchify back to [b,c,t,h,w] (same 6-dim ceiling).
        let y=self.output.forward(self.norm.forward(x)).reshape([b*t,hp,wp,c,ph,pw]).permute([0,3,1,4,2,5]).reshape([b,t,c,h,w]).swap_dims(1,2);(y,hidden)
    }
}

pub fn relation<B:Backend>(x:Tensor<B,3>)->Tensor<B,3>{let norm=x.clone().powf_scalar(2.0).sum_dim(2).sqrt().clamp_min(1e-6);let x=x/norm;x.clone().matmul(x.swap_dims(1,2))}
pub fn temporal_difference<B:Backend>(x:Tensor<B,5>)->Tensor<B,5>{let d=x.dims();let t=d[2];x.clone().slice([0..d[0],0..d[1],1..t,0..d[3],0..d[4]])-x.slice([0..d[0],0..d[1],0..t-1,0..d[3],0..d[4]])}

/// Mean-pool the two spatial axes of a `[b,c,t,h,w]` latent by 2, or return it
/// unchanged when either axis is odd.
///
/// Mean pooling is a low-pass filter, which is the point. `docs/VALIDATION-ROUND-6.md`
/// measures the student at 0.88-0.94 cosine in the highest spatial-frequency band
/// and ~0.48 at half magnitude in the 0.10-0.25 band, and that band carries only
/// 7-13% of the target's energy — so an unweighted MSE over the full-resolution
/// latent barely asks for it. An MSE term on pooled copies re-weights the objective
/// toward exactly the scales that are missing.
///
/// Written as reshape-and-mean rather than `avg_pool2d` because the latent is rank 5
/// while Burn's pooling takes rank 4, and because reducing one axis at a time keeps
/// every intermediate at rank 6 — the NdArray ceiling the patchify in `forward` is
/// already written around.
/// Nearest-neighbour 2x upsample of the two spatial axes — the inverse pairing for
/// `spatial_pool2` in a Laplacian band.
///
/// `repeat_dim` on an inserted axis, then fold it back, so every intermediate stays
/// at rank 6 for the same reason `spatial_pool2` does.
pub fn spatial_upsample2<B: Backend>(x: Tensor<B, 5>) -> Tensor<B, 5> {
    let [b, c, t, h, w] = x.dims();
    let x = x.reshape([b, c, t, h, 1, w]).repeat_dim(4, 2).reshape([b, c, t, h * 2, w]);
    x.reshape([b, c, t, h * 2, w, 1]).repeat_dim(5, 2).reshape([b, c, t, h * 2, w * 2])
}

/// One octave of a Laplacian pyramid: the content `spatial_pool2` keeps minus the
/// content one further pooling keeps, i.e. a **band-pass** rather than a low-pass.
///
/// This exists because the low-pass version did not work.
/// `docs/VALIDATION-ROUND-7.md` measures a multi-scale (pooled MSE) term moving the
/// 0.00–0.10 band from 0.684 to 0.773 and leaving the missing 0.10–0.25 band at
/// 0.466 against a control's 0.470 — because a pooled MSE contains *everything*
/// below its cutoff, and the 0.00–0.10 band carries 31% of the target's energy
/// against the target band's 12%, so it still dominates 2.6:1 inside the new term.
///
/// Subtracting the next octave removes exactly that. In a 40x40 latent, one
/// `spatial_pool2` keeps normalised radial frequency below ~0.354 and two keep
/// below ~0.177, so `band(x, 1)` isolates roughly 0.177–0.354 and `band(x, 2)`
/// roughly 0.088–0.177 — between them bracketing the band that is missing, and
/// excluding the one that is not.
pub fn laplacian_band<B: Backend>(x: Tensor<B, 5>, octave: usize) -> Tensor<B, 5> {
    let mut lo = x;
    for _ in 0..octave { lo = spatial_pool2(lo) }
    let coarser = spatial_pool2(lo.clone());
    // Odd axis: `spatial_pool2` is the identity, so the band would be all-zero.
    // Return it as such rather than pretending there is signal there.
    if coarser.dims() == lo.dims() { return lo.zeros_like() }
    lo - spatial_upsample2(coarser)
}

pub fn spatial_pool2<B: Backend>(x: Tensor<B, 5>) -> Tensor<B, 5> {
    let [b, c, t, h, w] = x.dims();
    if h % 2 != 0 || w % 2 != 0 { return x }
    // Split h into (h/2, 2) with the pair index fast-varying, average it away, then
    // the same for w. Row-major makes both reshapes pure re-indexing.
    let x = x.reshape([b, c, t, h / 2, 2, w]).mean_dim(4).reshape([b, c, t, h / 2, w]);
    x.reshape([b, c, t, h / 2, w / 2, 2]).mean_dim(5).reshape([b, c, t, h / 2, w / 2])
}

/// Serialises the tests that build a `BrowserVideoStudent`.
///
/// Burn's `Backend::seed` sets a *process-global* RNG, so a test that seeds and
/// then constructs a model is only correct if nothing else draws from that RNG in
/// between. `quant.rs`'s round-trip test does exactly that — seed 11, build,
/// seed 999, build — and cargo runs the crate's tests on several threads, so any
/// other model-constructing test landing between the two lines silently changes
/// what it is comparing against. It failed intermittently, passed 6/6 in
/// isolation, and reproduced under workspace-wide parallelism. Every test that
/// constructs a model takes this lock; poisoning is ignored because a panic in one
/// test should surface as that test's failure, not as a cascade.
#[cfg(test)]
pub(crate) static MODEL_RNG: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::{ndarray::NdArrayDevice, NdArray};
    use burn::tensor::backend::Backend;
    type Cpu = NdArray<f32>;

    // Multi-head attention must use every head: changing spec.heads (width fixed)
    // changes head_dim, the softmax scale, and the per-head partition, so the
    // output must differ from the single-head configuration. A block that ignored
    // `heads` (the prior bug) would produce identical output for both.
    #[test]
    fn heads_are_actually_used() {
        let _rng = crate::MODEL_RNG.lock().unwrap_or_else(|e| e.into_inner());
        let device = NdArrayDevice::default();
        // 4×4 latent → 2×2 = 4 tokens after patch_size [1,2,2]; multi-head only
        // matters with more than one token, so keep post-patch tokens > 1.
        let base = StudentSpec { latent_channels: 2, text_width: 4, width: 16, layers: 1, heads: 1, mlp_ratio: 2, max_tokens: 64, patch_size: [1, 2, 2], per_block_conditioning: false };
        let multi = StudentSpec { heads: 4, ..base.clone() };
        // Same seed → identical initial weights; only `heads` differs.
        <Cpu as Backend>::seed(&device, 7);
        let m1 = BrowserVideoStudent::<Cpu>::new(base, &device);
        <Cpu as Backend>::seed(&device, 7);
        let m4 = BrowserVideoStudent::<Cpu>::new(multi, &device);
        let latents = Tensor::<Cpu, 1>::from_floats([0.3f32; 2 * 1 * 4 * 4].as_slice(), &device).reshape([1, 2, 1, 4, 4]);
        let ts = Tensor::<Cpu, 1>::from_floats([500.0f32].as_slice(), &device).reshape([1, 1]);
        let prompt = Tensor::<Cpu, 1>::from_floats([0.1f32; 3 * 4].as_slice(), &device).reshape([1, 3, 4]);
        let (y1, _) = m1.forward(latents.clone(), ts.clone(), prompt.clone());
        let (y4, _) = m4.forward(latents, ts, prompt);
        let diff: f32 = (y1 - y4).abs().sum().into_scalar();
        assert!(diff > 1e-4, "1-head and 4-head outputs identical (diff={diff}) — heads not used");
    }

    // Tiling attention over query blocks must be a pure scheduling change: the
    // softmax is per-row, so splitting the rows cannot alter the maths, only the
    // order matmul tiles are summed in. A block that got this wrong (normalising
    // over the tile instead of the full key axis, or misplacing a tile in the
    // concatenation) would still train to *something*, so the guard has to be a
    // numerical identity against the single-shot path, not a smoke test. `seq`
    // here is 16 tokens against a block of 5 — four tiles, the last one short,
    // which is the ragged case `block ∤ seq` that off-by-one bugs live in.
    #[test]
    // A wrong reshape here would not fail loudly — it would silently train the
    // multi-scale term against a permuted target, which is worse than not having
    // the term at all. Pin it against hand-computed 2x2 means.
    #[test]
    fn spatial_pool2_averages_2x2_blocks() {
        type B = burn::backend::NdArray<f32>;
        let d = Default::default();
        // one [1,1,1,2,4] latent: two 2x2 blocks with means 3.5 and 5.5
        let x = Tensor::<B, 1>::from_floats([1., 2., 5., 6., 3., 4., 7., 8.].as_slice(), &d)
            .reshape([1, 1, 1, 2, 4]);
        let y = spatial_pool2(x);
        assert_eq!(y.dims(), [1, 1, 1, 1, 2]);
        let v = y.into_data().convert::<f32>().into_vec::<f32>().unwrap();
        assert!((v[0] - 2.5).abs() < 1e-6, "got {v:?}");
        assert!((v[1] - 6.5).abs() < 1e-6, "got {v:?}");
    }

    // Odd axes must be left alone rather than truncated: the loss loop uses an
    // unchanged shape as its signal to stop adding levels.
    #[test]
    fn spatial_pool2_is_identity_on_odd_axes() {
        type B = burn::backend::NdArray<f32>;
        let d = Default::default();
        let x = Tensor::<B, 1>::from_floats([1., 2., 3.].as_slice(), &d).reshape([1, 1, 1, 1, 3]);
        assert_eq!(spatial_pool2(x).dims(), [1, 1, 1, 1, 3]);
    }

    // Pins the octave semantics, which are easy to get off by one: `band(x, k)`
    // isolates what survives k poolings but not k+1, so octave 0 is the *highest*
    // frequencies and each further octave is one scale coarser.
    #[test]
    fn laplacian_band_is_band_pass_and_octave_indexed() {
        type B = burn::backend::NdArray<f32>;
        let d = Default::default();
        let mk = |f: &dyn Fn(usize, usize) -> f32| {
            let mut v = vec![0.0f32; 64];
            for y in 0..8 { for x in 0..8 { v[y * 8 + x] = f(x, y) } }
            Tensor::<B, 1>::from_floats(v.as_slice(), &d).reshape([1, 1, 1, 8, 8])
        };
        let peak = |t: Tensor<B, 5>| t.abs().max().into_data().convert::<f32>().into_vec::<f32>().unwrap()[0];

        // Pure DC has no band energy at any octave.
        for k in 0..2 {
            let e = peak(laplacian_band(mk(&|_, _| 3.0), k));
            assert!(e < 1e-6, "constant field, octave {k}: expected 0, got {e}");
        }
        // Period 2 (per-cell checkerboard) is the finest scale: octave 0 only.
        let fine = |x: usize, y: usize| if (x + y) % 2 == 0 { 1.0 } else { -1.0 };
        assert!(peak(laplacian_band(mk(&fine), 0)) > 0.5, "period-2 must live in octave 0");
        assert!(peak(laplacian_band(mk(&fine), 1)) < 1e-6, "period-2 must not reach octave 1");
        // Period 4 (2x2 blocks) is one scale coarser: octave 1, and invisible to 0
        // because one pooling reproduces it exactly.
        let coarse = |x: usize, y: usize| if ((x / 2) + (y / 2)) % 2 == 0 { 1.0 } else { -1.0 };
        assert!(peak(laplacian_band(mk(&coarse), 0)) < 1e-6, "period-4 must not leak into octave 0");
        assert!(peak(laplacian_band(mk(&coarse), 1)) > 0.5, "period-4 must live in octave 1");
    }

    #[test]
    fn spatial_upsample2_inverts_shape_and_repeats() {
        type B = burn::backend::NdArray<f32>;
        let d = Default::default();
        let x = Tensor::<B, 1>::from_floats([1., 2.].as_slice(), &d).reshape([1, 1, 1, 1, 2]);
        let y = spatial_upsample2(x);
        assert_eq!(y.dims(), [1, 1, 1, 2, 4]);
        let v = y.into_data().convert::<f32>().into_vec::<f32>().unwrap();
        assert_eq!(v, vec![1., 1., 2., 2., 1., 1., 2., 2.]);
    }

    fn tiled_attention_matches_single_shot() {
        let device = NdArrayDevice::default();
        <Cpu as Backend>::seed(&device, 11);
        let (width, heads, seq) = (16usize, 4usize, 16usize);
        let block = MixerBlock::<Cpu>::new(width, 2, false, &device);
        let zero = Tensor::<Cpu, 3>::zeros([1, 1, width], &device); // unused: `ada` is None here
        // Structured, non-degenerate input: a constant tensor makes every query
        // row identical and would pass even with the rows scrambled.
        let values: Vec<f32> = (0..seq * width).map(|i| ((i as f32) * 0.37).sin()).collect();
        let x = Tensor::<Cpu, 1>::from_floats(values.as_slice(), &device).reshape([1, seq, width]);
        // `usize::MAX` can never be exceeded by `seq`, so it selects the original
        // single-shot path by the same `seq <= block` branch production uses.
        let single = block.forward(x.clone(), zero.clone(), heads, usize::MAX);
        let tiled = block.forward(x, zero, heads, 5);
        let max_diff: f32 = (single - tiled).abs().max().into_scalar();
        assert!(max_diff < 1e-5, "tiled attention diverges from single-shot (max abs diff={max_diff})");
    }

    // adaLN-zero has one property that has to hold exactly or the flag is a
    // liability rather than an experiment: at initialisation every block must be
    // an identity, because both gates are zero. Round 1's architecture is the
    // control in this A/B, so the treatment must not start from a different
    // random function — if the gates were randomly initialised, all 24 blocks
    // would rescale the residual stream before a single gradient was applied and
    // any difference in the loss curve would be unattributable.
    //
    // The check is direct: with the flag on, every hidden state at step 0 equals
    // the stem output, so all of them are equal to each other. With the flag off
    // they are not, because each block does real work from the start.
    #[test]
    fn per_block_conditioning_is_identity_at_initialisation() {
        let _rng = crate::MODEL_RNG.lock().unwrap_or_else(|e| e.into_inner());
        let device = NdArrayDevice::default();
        let off = StudentSpec { latent_channels: 2, text_width: 4, width: 16, layers: 3, heads: 2, mlp_ratio: 2, max_tokens: 64, patch_size: [1, 2, 2], per_block_conditioning: false };
        let on = StudentSpec { per_block_conditioning: true, ..off.clone() };
        let latents = Tensor::<Cpu, 1>::from_floats([0.3f32; 2 * 1 * 4 * 4].as_slice(), &device).reshape([1, 2, 1, 4, 4]);
        let ts = Tensor::<Cpu, 1>::from_floats([500.0f32].as_slice(), &device).reshape([1, 1]);
        let prompt = Tensor::<Cpu, 1>::from_floats([0.1f32; 3 * 4].as_slice(), &device).reshape([1, 3, 4]);

        <Cpu as Backend>::seed(&device, 5);
        let (_, hidden_on) = BrowserVideoStudent::<Cpu>::new(on, &device).forward(latents.clone(), ts.clone(), prompt.clone());
        let spread = |h: &Vec<Tensor<Cpu, 3>>| -> f32 {
            (h.last().unwrap().clone() - h.first().unwrap().clone()).abs().max().into_scalar()
        };
        assert!(spread(&hidden_on) < 1e-6,
            "zero-init gates must make every block an identity; deepest hidden differs from the first by {}",
            spread(&hidden_on));

        <Cpu as Backend>::seed(&device, 5);
        let (_, hidden_off) = BrowserVideoStudent::<Cpu>::new(off, &device).forward(latents, ts, prompt);
        assert!(spread(&hidden_off) > 1e-4,
            "control must NOT be an identity, else this test proves nothing about the treatment");
    }

    // The modulation has to be reachable from sigma specifically — that is the
    // entire point of the flag. Round 1 measured parity that was flat across
    // sigma and suspected the single stem injection was a ceiling; if `ada` were
    // wired to the prompt alone, or the timestep never reached it, the flag would
    // look enabled and change nothing about sigma sensitivity.
    #[test]
    fn per_block_conditioning_carries_sigma_into_every_block() {
        let _rng = crate::MODEL_RNG.lock().unwrap_or_else(|e| e.into_inner());
        let device = NdArrayDevice::default();
        let spec = StudentSpec { latent_channels: 2, text_width: 4, width: 16, layers: 3, heads: 2, mlp_ratio: 2, max_tokens: 64, patch_size: [1, 2, 2], per_block_conditioning: true };
        <Cpu as Backend>::seed(&device, 5);
        let model = BrowserVideoStudent::<Cpu>::new(spec, &device);
        // Perturb the ada weights off zero — at init the gates are closed by
        // construction, so a sigma test there would trivially pass.
        let model = model.map(&mut ShiftAda);
        let latents = Tensor::<Cpu, 1>::from_floats([0.3f32; 2 * 1 * 4 * 4].as_slice(), &device).reshape([1, 2, 1, 4, 4]);
        let prompt = Tensor::<Cpu, 1>::from_floats([0.1f32; 3 * 4].as_slice(), &device).reshape([1, 3, 4]);
        let at = |t: f32| {
            let ts = Tensor::<Cpu, 1>::from_floats([t].as_slice(), &device).reshape([1, 1]);
            model.forward(latents.clone(), ts, prompt.clone()).1.last().unwrap().clone()
        };
        let diff: f32 = (at(50.0) - at(950.0)).abs().max().into_scalar();
        assert!(diff > 1e-5, "deepest hidden state is insensitive to sigma (max diff={diff})");
    }

    /// Nudges every parameter off zero so the closed adaLN gates open.
    struct ShiftAda;
    impl<B: Backend> burn::module::ModuleMapper<B> for ShiftAda {
        fn map_float<const D: usize>(&mut self, param: burn::module::Param<Tensor<B, D>>) -> burn::module::Param<Tensor<B, D>> {
            let (id, tensor, mapper) = param.consume();
            burn::module::Param::from_mapped_value(id, tensor + 0.05, mapper)
        }
    }

    // Prompt conditioning must actually reach the output: the same model and
    // latents with two different prompt embeddings must denoise differently.
    // If the prompt were ignored (as the demo did before Phase 3), the outputs
    // would be identical.
    #[test]
    fn prompt_conditions_the_output() {
        let _rng = crate::MODEL_RNG.lock().unwrap_or_else(|e| e.into_inner());
        let device = NdArrayDevice::default();
        let spec = StudentSpec { latent_channels: 2, text_width: 4, width: 16, layers: 1, heads: 2, mlp_ratio: 2, max_tokens: 64, patch_size: [1, 2, 2], per_block_conditioning: false };
        <Cpu as Backend>::seed(&device, 3);
        let model = BrowserVideoStudent::<Cpu>::new(spec, &device);
        let latents = Tensor::<Cpu, 1>::from_floats([0.3f32; 2 * 1 * 4 * 4].as_slice(), &device).reshape([1, 2, 1, 4, 4]);
        let ts = Tensor::<Cpu, 1>::from_floats([500.0f32].as_slice(), &device).reshape([1, 1]);
        let prompt_a = Tensor::<Cpu, 1>::from_floats([0.1f32; 3 * 4].as_slice(), &device).reshape([1, 3, 4]);
        let prompt_b = Tensor::<Cpu, 1>::from_floats([-0.7f32; 3 * 4].as_slice(), &device).reshape([1, 3, 4]);
        let (ya, _) = model.forward(latents.clone(), ts.clone(), prompt_a);
        let (yb, _) = model.forward(latents, ts, prompt_b);
        let diff: f32 = (ya - yb).abs().sum().into_scalar();
        assert!(diff > 1e-4, "output identical for different prompts (diff={diff}) — prompt ignored");
    }
}

