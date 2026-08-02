use burn::{nn::{Gelu, LayerNorm, LayerNormConfig, Linear, LinearConfig}, prelude::*};
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
}
impl<B: Backend> MixerBlock<B> {
    fn new(width: usize, mlp_ratio: usize, device: &B::Device) -> Self { Self {
        norm: LayerNormConfig::new(width).init(device), q: LinearConfig::new(width,width).init(device),
        k: LinearConfig::new(width,width).init(device), v: LinearConfig::new(width,width).init(device),
        proj: LinearConfig::new(width,width).init(device), norm_mlp: LayerNormConfig::new(width).init(device),
        up: LinearConfig::new(width,width*mlp_ratio).init(device), down: LinearConfig::new(width*mlp_ratio,width).init(device), activation:Gelu::new(),
    }}
    // Multi-head scaled-dot-product attention. `heads` and `block` come from the
    // caller so the block stays weightless-of-config (`block` is the query-tile
    // width; see `tiled_attention`). Attention is bidirectional within the
    // chunk by design (rust/README.md): causality lives at the streaming level,
    // not in an intra-chunk mask, which keeps browser attention cheap.
    fn forward(&self, x: Tensor<B,3>, heads: usize, block: usize) -> Tensor<B,3> {
        let [b,seq,width]=x.dims(); let head_dim=width/heads; let scale=(head_dim as f64).sqrt();
        let n=self.norm.forward(x.clone());
        let split=|t: Tensor<B,3>| t.reshape([b,seq,heads,head_dim]).swap_dims(1,2); // [b, heads, seq, head_dim]
        let q=split(self.q.forward(n.clone())); let k=split(self.k.forward(n.clone())); let v=split(self.v.forward(n));
        let context=tiled_attention(q,k,v,scale,block).swap_dims(1,2).reshape([b,seq,width]);
        let x=x+self.proj.forward(context); let m=self.norm_mlp.forward(x.clone()); x+self.down.forward(self.activation.forward(self.up.forward(m)))
    }
}

#[derive(Module, Debug)]
pub struct BrowserVideoStudent<B: Backend> {
    input: Linear<B>, text: Linear<B>, time: Linear<B>, blocks: Vec<MixerBlock<B>>, norm: LayerNorm<B>, output: Linear<B>, spec: StudentSpec,
}
impl<B: Backend> BrowserVideoStudent<B> {
    pub fn new(spec: StudentSpec, device:&B::Device)->Self { spec.validate().expect("valid spec"); let token=spec.latent_channels*spec.patch_volume(); Self {
        input:LinearConfig::new(token,spec.width).init(device),text:LinearConfig::new(spec.text_width,spec.width).init(device),time:LinearConfig::new(1,spec.width).init(device),
        blocks:(0..spec.layers).map(|_|MixerBlock::new(spec.width,spec.mlp_ratio,device)).collect(),norm:LayerNormConfig::new(spec.width).init(device),output:LinearConfig::new(spec.width,token).init(device),spec,
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
        let cond=self.text.forward(prompt.mean_dim(1))+self.time.forward(timestep/1000.0).unsqueeze_dim(1);x=self.input.forward(x)+cond;
        let mut hidden=Vec::with_capacity(self.blocks.len());let tile=attention_block();for block in &self.blocks{x=block.forward(x,self.spec.heads,tile);hidden.push(x.clone())}
        // Inverse patchify back to [b,c,t,h,w] (same 6-dim ceiling).
        let y=self.output.forward(self.norm.forward(x)).reshape([b*t,hp,wp,c,ph,pw]).permute([0,3,1,4,2,5]).reshape([b,t,c,h,w]).swap_dims(1,2);(y,hidden)
    }
}

pub fn relation<B:Backend>(x:Tensor<B,3>)->Tensor<B,3>{let norm=x.clone().powf_scalar(2.0).sum_dim(2).sqrt().clamp_min(1e-6);let x=x/norm;x.clone().matmul(x.swap_dims(1,2))}
pub fn temporal_difference<B:Backend>(x:Tensor<B,5>)->Tensor<B,5>{let d=x.dims();let t=d[2];x.clone().slice([0..d[0],0..d[1],1..t,0..d[3],0..d[4]])-x.slice([0..d[0],0..d[1],0..t-1,0..d[3],0..d[4]])}

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
        let device = NdArrayDevice::default();
        // 4×4 latent → 2×2 = 4 tokens after patch_size [1,2,2]; multi-head only
        // matters with more than one token, so keep post-patch tokens > 1.
        let base = StudentSpec { latent_channels: 2, text_width: 4, width: 16, layers: 1, heads: 1, mlp_ratio: 2, max_tokens: 64, patch_size: [1, 2, 2] };
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
    fn tiled_attention_matches_single_shot() {
        let device = NdArrayDevice::default();
        <Cpu as Backend>::seed(&device, 11);
        let (width, heads, seq) = (16usize, 4usize, 16usize);
        let block = MixerBlock::<Cpu>::new(width, 2, &device);
        // Structured, non-degenerate input: a constant tensor makes every query
        // row identical and would pass even with the rows scrambled.
        let values: Vec<f32> = (0..seq * width).map(|i| ((i as f32) * 0.37).sin()).collect();
        let x = Tensor::<Cpu, 1>::from_floats(values.as_slice(), &device).reshape([1, seq, width]);
        // `usize::MAX` can never be exceeded by `seq`, so it selects the original
        // single-shot path by the same `seq <= block` branch production uses.
        let single = block.forward(x.clone(), heads, usize::MAX);
        let tiled = block.forward(x, heads, 5);
        let max_diff: f32 = (single - tiled).abs().max().into_scalar();
        assert!(max_diff < 1e-5, "tiled attention diverges from single-shot (max abs diff={max_diff})");
    }

    // Prompt conditioning must actually reach the output: the same model and
    // latents with two different prompt embeddings must denoise differently.
    // If the prompt were ignored (as the demo did before Phase 3), the outputs
    // would be identical.
    #[test]
    fn prompt_conditions_the_output() {
        let device = NdArrayDevice::default();
        let spec = StudentSpec { latent_channels: 2, text_width: 4, width: 16, layers: 1, heads: 2, mlp_ratio: 2, max_tokens: 64, patch_size: [1, 2, 2] };
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

