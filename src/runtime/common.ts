import * as ort from "onnxruntime-web/webgpu";
import {AutoTokenizer} from "@huggingface/transformers";
export type Manifest={kind:"sd-turbo"|"longlive"|"memflow"|"student";width:number;height:number;latentScale:number;models:Record<string,string>;io?:Record<string,string>;scheduler?:{trainSteps:number};memory?:{capacity:number;topK:number;keySize:number};latent?:{channels:number;frames:number;seq:number;text_width:number}};
export async function manifest(url:string){const r=await fetch(url);if(!r.ok)throw new Error(`Manifest ${r.status}: ${url}`);const m=await r.json() as Manifest;const base=new URL(url,location.href);for(const k of Object.keys(m.models))m.models[k]=new URL(m.models[k],base).href;return m}
export async function session(url:string){return ort.InferenceSession.create(url,{executionProviders:["webgpu"],graphOptimizationLevel:"all",enableMemPattern:false})}
export function normal(size:number,seed:number){let s=seed>>>0;const out=new Float32Array(size);for(let i=0;i<size;i+=2){s=(1664525*s+1013904223)>>>0;const u=(s+1)/4294967297;s=(1664525*s+1013904223)>>>0;const v=(s+1)/4294967297;const r=Math.sqrt(-2*Math.log(u));out[i]=r*Math.cos(2*Math.PI*v);if(i+1<size)out[i+1]=r*Math.sin(2*Math.PI*v)}return out}
// A browser text encoder (umt5-small): its tokenizer plus the ONNX session.
export type TextEncoder={tokenizer:any;session:ort.InferenceSession};
// Load the encoder a manifest references, or undefined when none is shipped.
export async function loadTextEncoder(m:Manifest):Promise<TextEncoder|undefined>{if(!m.models.text_encoder)return undefined;const tokenizer=await AutoTokenizer.from_pretrained(m.models.tokenizer);const s=await session(m.models.text_encoder);return{tokenizer,session:s}}
// Deterministic 32-bit FNV-1a hash so a prompt string can seed a reproducible
// fallback embedding — different prompts then produce different output even
// before a real encoder is shipped.
export function hashPrompt(prompt:string){let h=0x811c9dc5>>>0;for(let i=0;i<prompt.length;i++){h^=prompt.charCodeAt(i);h=Math.imul(h,0x01000193)>>>0}return h>>>0}
// Encode a prompt into a flat `[seq, textWidth]` embedding. Uses the real
// encoder when one is present and its hidden width matches the model
// (`data.length === seq*textWidth`); otherwise a prompt-seeded fallback so the
// tab still runs and the prompt still influences the result.
export async function encodePrompt(enc:TextEncoder|undefined,prompt:string,seq:number,textWidth:number,seed:number):Promise<{data:Float32Array;source:"encoder"|"seeded"}>{
  if(enc){try{const tok=await enc.tokenizer(prompt,{padding:"max_length",max_length:seq,truncation:true});const feeds:Record<string,ort.Tensor>={input_ids:new ort.Tensor("int64",BigInt64Array.from(Array.from(tok.input_ids.data as ArrayLike<number>),BigInt),[1,seq])};if(tok.attention_mask)feeds.attention_mask=new ort.Tensor("int64",BigInt64Array.from(Array.from(tok.attention_mask.data as ArrayLike<number>),BigInt),[1,seq]);const out=await enc.session.run(feeds);const t=out.last_hidden_state??Object.values(out)[0];const data=t.data as Float32Array;if(data.length===seq*textWidth)return{data,source:"encoder"}}catch{/* fall through to seeded */}}
  return{data:normal(seq*textWidth,seed),source:"seeded"};
}
export function rgbaFromNchw(data:Float32Array,w:number,h:number){const o=new Uint8ClampedArray(w*h*4),plane=w*h;for(let i=0;i<plane;i++){o[i*4]=255*Math.max(0,Math.min(1,data[i]/2+.5));o[i*4+1]=255*Math.max(0,Math.min(1,data[plane+i]/2+.5));o[i*4+2]=255*Math.max(0,Math.min(1,data[2*plane+i]/2+.5));o[i*4+3]=255}return o}
export {ort};
