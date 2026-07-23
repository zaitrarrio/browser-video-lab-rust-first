import {AutoTokenizer} from '@huggingface/transformers';
import {manifest,normal,ort,rgbaFromNchw,session} from './common';
// Runs the distilled latent-video student (`denoiser.onnx`) on ONNX Runtime Web
// over WebGPU. Graph contract mirrors python `export.py`:
//   noisy_latents[1,C,T,H,W], timestep[1], prompt_embeds[1,S,text_width] -> noise_pred[1,C,T,H,W].
// Prompt conditioning uses a browser-sized text encoder (umt5-small, 512-dim):
// tokenize -> text_encoder.onnx -> last_hidden_state, fed straight into the
// denoiser. The student is distilled with text_width matching this encoder, so
// its input Linear is the learned projection into the model width. When no
// text_encoder is shipped the embeds are seeded so runs stay deterministic and
// the tab still works before real weights exist. No VAE is exported, so — like
// the Rust/WASM student — the first three latent channels are shown directly.
type Latent={channels:number;frames:number;seq:number;text_width:number};
export class OnnxStudentRuntime{private m!:Awaited<ReturnType<typeof manifest>>;private denoiser!:ort.InferenceSession;private text?:ort.InferenceSession;private tokenizer:any;
 async load(url:string,progress:(s:string)=>void){this.m=await manifest(url);if(this.m.models.text_encoder){progress('Loading text encoder…');this.tokenizer=await AutoTokenizer.from_pretrained(this.m.models.tokenizer);this.text=await session(this.m.models.text_encoder)}progress('Loading student denoiser…');this.denoiser=await session(this.m.models.denoiser);progress(this.text?'ONNX student ready · prompt-conditioned':'ONNX student ready · seeded embeds')}
 private async embed(prompt:string,L:Latent,seed:number){if(this.text&&this.tokenizer){const tok=await this.tokenizer(prompt,{padding:'max_length',max_length:L.seq,truncation:true});const feeds:Record<string,ort.Tensor>={input_ids:new ort.Tensor('int64',BigInt64Array.from(Array.from(tok.input_ids.data as ArrayLike<number>),BigInt),[1,L.seq])};if(tok.attention_mask)feeds.attention_mask=new ort.Tensor('int64',BigInt64Array.from(Array.from(tok.attention_mask.data as ArrayLike<number>),BigInt),[1,L.seq]);const out=await this.text.run(feeds);return out.last_hidden_state??Object.values(out)[0]}
  return new ort.Tensor('float32',normal(L.seq*L.text_width,(seed^0x9e3779b9)>>>0),[1,L.seq,L.text_width])}
 async run(prompt:string,steps:number,seed:number,onFrame:(x:{rgba:Uint8ClampedArray,width:number,height:number})=>void,signal:AbortSignal){const L:Latent=this.m.latent??{channels:4,frames:1,seq:8,text_width:64};const w=this.m.width,h=this.m.height,C=L.channels,T=L.frames,train=this.m.scheduler?.trainSteps??1000;
  let lat=normal(C*T*h*w,seed);const prompt_embeds=await this.embed(prompt,L,seed);const n=Math.max(1,steps),rate=1/n;
  for(let i=0;i<n;i++){if(signal.aborted)throw new DOMException('Stopped','AbortError');const t=Math.round((train-1)*(1-i/n));const out=await this.denoiser.run({noisy_latents:new ort.Tensor('float32',lat,[1,C,T,h,w]),timestep:new ort.Tensor('float32',new Float32Array([t]),[1]),prompt_embeds});const eps=(out.noise_pred??Object.values(out)[0]).data as Float32Array;for(let j=0;j<lat.length;j++)lat[j]-=eps[j]*rate}
  if(signal.aborted)throw new DOMException('Stopped','AbortError');const plane=h*w,frame0=new Float32Array(3*plane);for(let c=0;c<3&&c<C;c++)frame0.set(lat.subarray(c*T*plane,c*T*plane+plane),c*plane);onFrame({rgba:rgbaFromNchw(frame0,w,h),width:w,height:h})}
}
