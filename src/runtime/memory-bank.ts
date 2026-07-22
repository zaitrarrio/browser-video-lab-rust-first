import {ort} from './common';
export type MemoryEntry={key:Float32Array;value:ort.Tensor;age:number};
export class AdaptiveMemoryBank{
 private entries:MemoryEntry[]=[];private clock=0;
 constructor(readonly capacity=16,readonly topK=4){}
 get size(){return this.entries.length}
 clear(){this.entries=[];this.clock=0}
 add(key:Float32Array,value:ort.Tensor){const norm=normalize(key);this.entries.push({key:norm,value,age:this.clock++});if(this.entries.length>this.capacity)this.entries.shift()}
 retrieve(query:Float32Array){const q=normalize(query);return this.entries.map(e=>({entry:e,score:dot(q,e.key)})).sort((a,b)=>b.score-a.score||b.entry.age-a.entry.age).slice(0,this.topK)}
}
export function normalize(x:Float32Array){let sum=0;for(const v of x)sum+=v*v;const n=Math.sqrt(sum)||1;return Float32Array.from(x,v=>v/n)}
export function dot(a:Float32Array,b:Float32Array){let s=0;for(let i=0;i<Math.min(a.length,b.length);i++)s+=a[i]*b[i];return s}
export function concatMemory(items:ort.Tensor[]){if(!items.length)return undefined;const first=items[0],dims=[...first.dims];if(dims.length<2)throw new Error('Memory value must have a token axis');dims[1]=items.reduce((n,t)=>n+t.dims[1],0);const C=(first.data as Float32Array).constructor as Float32ArrayConstructor;const data=new C(items.reduce((n,t)=>n+(t.data as Float32Array).length,0));let offset=0;for(const t of items){data.set(t.data as Float32Array,offset);offset+=(t.data as Float32Array).length}return new ort.Tensor(first.type as 'float32',data,dims)}
