import "./style.css";
type Mode="sd"|"longlive"|"memflow";
let mode:Mode="sd";
const root=document.querySelector<HTMLDivElement>("#app")!;

function render(){root.innerHTML=`<main><h1>Browser Video Lab</h1><div class="muted">WebGPU-first SD-Turbo, LongLive, and adaptive-memory MemFlow inference</div><nav><button data-mode="sd" class="${mode==='sd'?'active':''}">SD-Turbo</button><button data-mode="longlive" class="${mode==='longlive'?'active':''}">LongLive</button><button data-mode="memflow" class="${mode==='memflow'?'active':''}">MemFlow</button></nav><section class="card"><label>Model manifest URL<input id="manifest" style="width:100%" value="/models/${mode}/manifest.json"></label><label><div style="margin-top:14px">Prompt</div><textarea id="prompt">A cinematic tracking shot of a silver robot walking through Austin at sunset</textarea></label><div class="grid"><label>Seed<input id="seed" type="number" value="42" style="width:100%"></label><label>${mode==='sd'?'Steps<input id="steps" type="number" value="4" min="1" max="12" style="width:100%">':'Chunks<input id="steps" type="number" value="4" min="1" max="32" style="width:100%">'}</label></div><div class="actions"><button id="load">Load model</button><button id="run">Generate</button><button id="stop">Stop</button></div><div class="status" id="status">WebGPU: ${'gpu' in navigator?'available':'unavailable'}</div><canvas id="output" width="512" height="512"></canvas></section></main>`;
 document.querySelectorAll<HTMLButtonElement>("[data-mode]").forEach(b=>b.onclick=()=>{mode=b.dataset.mode as Mode;render()});
 const worker=new Worker(new URL("./worker.ts",import.meta.url),{type:"module"});
 const status=document.querySelector<HTMLDivElement>("#status")!, canvas=document.querySelector<HTMLCanvasElement>("#output")!;
 worker.onmessage=(e)=>{const m=e.data;if(m.type==='status')status.textContent=m.message;if(m.type==='frame'){canvas.width=m.width;canvas.height=m.height;canvas.getContext('2d')!.putImageData(new ImageData(new Uint8ClampedArray(m.rgba),m.width,m.height),0,0)}};
 const payload=()=>({mode,manifestUrl:(document.querySelector<HTMLInputElement>('#manifest')!).value,prompt:(document.querySelector<HTMLTextAreaElement>('#prompt')!).value,seed:+(document.querySelector<HTMLInputElement>('#seed')!).value,steps:+(document.querySelector<HTMLInputElement>('#steps')!).value});
 document.querySelector<HTMLButtonElement>('#load')!.onclick=()=>worker.postMessage({type:'load',...payload()});
 document.querySelector<HTMLButtonElement>('#run')!.onclick=()=>worker.postMessage({type:'run',...payload()});
 document.querySelector<HTMLButtonElement>('#stop')!.onclick=()=>worker.postMessage({type:'stop'});
}
render();
