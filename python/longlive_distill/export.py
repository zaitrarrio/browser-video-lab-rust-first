from __future__ import annotations
import argparse
from pathlib import Path
import torch,yaml
from .model import build_student
class Export(torch.nn.Module):
    def __init__(self,m):super().__init__();self.m=m
    def forward(self,noisy_latents,timestep,prompt_embeds):return self.m(noisy_latents,timestep,prompt_embeds)["noise_pred"]
def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);a=p.parse_args();cfg=yaml.safe_load(Path(a.config).read_text());m=build_student(cfg['student']['model']);state=torch.load(a.checkpoint,map_location='cpu',weights_only=True);m.load_state_dict(state['model']);m.eval();shape=cfg['data']['latent_shape'];text=cfg['data']['text_shape'];Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    torch.onnx.export(Export(m),(torch.randn(*shape),torch.zeros(shape[0]),torch.randn(*text)),a.output,input_names=['noisy_latents','timestep','prompt_embeds'],output_names=['noise_pred'],opset_version=18,dynamic_axes={'noisy_latents':{2:'frames'},'noise_pred':{2:'frames'}})
if __name__=='__main__':main()
