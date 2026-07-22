"""One-time bridge from an official LongLive teacher to framework-neutral Safetensors shards."""
from __future__ import annotations
import argparse,importlib,json
from pathlib import Path
import torch
from safetensors.torch import save_file

def resolve(spec):
    module,name=spec.split(':',1);return getattr(importlib.import_module(module),name)
def main():
    p=argparse.ArgumentParser();p.add_argument('--adapter',required=True,help='module:function returning the official frozen teacher');p.add_argument('--dataset',required=True);p.add_argument('--output',required=True);p.add_argument('--limit',type=int,default=0);p.add_argument('--scheduler',default='longlive');a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True);teacher=resolve(a.adapter)({}).cuda().eval();files=sorted(Path(a.dataset).glob('*.pt'));files=files[:a.limit or None];shards=[];shapes={}
    with torch.no_grad():
        for i,file in enumerate(files):
            item=torch.load(file,map_location='cpu',weights_only=True);lat=item['latents'].cuda();prompt=item['prompt_embeds'].cuda();time=item.get('timestep',torch.randint(0,1000,(lat.shape[0],))).cuda();noisy=item.get('noisy_latents',lat+torch.randn_like(lat)*(time.float()/1000).view(-1,1,1,1,1));result=teacher(noisy,time,prompt);tensors={'noisy_latents':noisy.cpu().contiguous(),'timestep':time.cpu().contiguous(),'prompt_embeds':prompt.cpu().contiguous(),'teacher_noise_pred':result['noise_pred'].cpu().contiguous()}
            for layer,h in enumerate(result.get('hidden_states',[])):
                h=torch.nn.functional.normalize(h.float(),dim=-1);tensors[f'teacher_relation.{layer}']=(h@h.transpose(-1,-2)).half().cpu().contiguous()
            name=f'shard-{i:06d}.safetensors';save_file(tensors,out/name);shards.append(name)
            for key,value in tensors.items():shapes[key]={'name':key,'shape':list(value.shape),'dtype':str(value.dtype).replace('torch.','').upper()}
    manifest={'format_version':1,'teacher':'LongLive-1.3B','scheduler':a.scheduler,'shards':shards,'tensors':list(shapes.values()),'hidden_relation_layers':list(range(len(result.get('hidden_states',[])))) if files else []};(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':main()
