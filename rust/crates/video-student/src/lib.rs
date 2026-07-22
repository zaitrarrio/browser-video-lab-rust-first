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
    fn forward(&self, x: Tensor<B,3>) -> Tensor<B,3> {
        let n=self.norm.forward(x.clone()); let scale=(n.dims()[2] as f64).sqrt();
        let attention=burn::tensor::activation::softmax(self.q.forward(n.clone()).matmul(self.k.forward(n.clone()).swap_dims(1,2))/scale,2);
        let x=x+self.proj.forward(attention.matmul(self.v.forward(n))); let m=self.norm_mlp.forward(x.clone()); x+self.down.forward(self.activation.forward(self.up.forward(m)))
    }
}

#[derive(Module, Debug)]
pub struct BrowserVideoStudent<B: Backend> {
    input: Linear<B>, text: Linear<B>, time: Linear<B>, blocks: Vec<MixerBlock<B>>, norm: LayerNorm<B>, output: Linear<B>, spec: StudentSpec,
}
impl<B: Backend> BrowserVideoStudent<B> {
    pub fn new(spec: StudentSpec, device:&B::Device)->Self { spec.validate().expect("valid spec"); Self {
        input:LinearConfig::new(spec.latent_channels,spec.width).init(device),text:LinearConfig::new(spec.text_width,spec.width).init(device),time:LinearConfig::new(1,spec.width).init(device),
        blocks:(0..spec.layers).map(|_|MixerBlock::new(spec.width,spec.mlp_ratio,device)).collect(),norm:LayerNormConfig::new(spec.width).init(device),output:LinearConfig::new(spec.width,spec.latent_channels).init(device),spec,
    }}
    pub fn forward(&self, latents:Tensor<B,5>, timestep:Tensor<B,2>, prompt:Tensor<B,3>)->(Tensor<B,5>,Vec<Tensor<B,3>>){
        let [b,c,t,h,w]=latents.dims();let mut x=latents.swap_dims(1,4).reshape([b,t*h*w,c]);let cond=self.text.forward(prompt.mean_dim(1)).unsqueeze_dim(1)+self.time.forward(timestep/1000.0).unsqueeze_dim(1);x=self.input.forward(x)+cond;let mut hidden=Vec::with_capacity(self.blocks.len());for block in &self.blocks{x=block.forward(x);hidden.push(x.clone())}let y=self.output.forward(self.norm.forward(x)).reshape([b,t,h,w,c]).swap_dims(1,4);(y,hidden)
    }
}

pub fn relation<B:Backend>(x:Tensor<B,3>)->Tensor<B,3>{let norm=x.clone().powf_scalar(2.0).sum_dim(2).sqrt().clamp_min(1e-6);let x=x/norm;x.clone().matmul(x.swap_dims(1,2))}
pub fn temporal_difference<B:Backend>(x:Tensor<B,5>)->Tensor<B,5>{let t=x.dims()[2];x.clone().slice([0..x.dims()[0],0..x.dims()[1],1..t,0..x.dims()[3],0..x.dims()[4]])-x.slice([0..x.dims()[0],0..x.dims()[1],0..t-1,0..x.dims()[3],0..x.dims()[4]])}

