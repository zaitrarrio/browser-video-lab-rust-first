use burn::{nn::{Gelu, LayerNorm, LayerNormConfig, Linear, LinearConfig}, prelude::*};
use video_contract::StudentSpec;

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
    // Multi-head scaled-dot-product attention. `heads` comes from the spec so the
    // block stays weightless-of-config. Attention is bidirectional within the
    // chunk by design (rust/README.md): causality lives at the streaming level,
    // not in an intra-chunk mask, which keeps browser attention cheap.
    fn forward(&self, x: Tensor<B,3>, heads: usize) -> Tensor<B,3> {
        let [b,seq,width]=x.dims(); let head_dim=width/heads; let scale=(head_dim as f64).sqrt();
        let n=self.norm.forward(x.clone());
        let split=|t: Tensor<B,3>| t.reshape([b,seq,heads,head_dim]).swap_dims(1,2); // [b, heads, seq, head_dim]
        let q=split(self.q.forward(n.clone())); let k=split(self.k.forward(n.clone())); let v=split(self.v.forward(n));
        let attention=burn::tensor::activation::softmax(q.matmul(k.swap_dims(2,3))/scale,3); // [b, heads, seq, seq]
        let context=attention.matmul(v).swap_dims(1,2).reshape([b,seq,width]);
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
        let mut hidden=Vec::with_capacity(self.blocks.len());for block in &self.blocks{x=block.forward(x,self.spec.heads);hidden.push(x.clone())}
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
}

