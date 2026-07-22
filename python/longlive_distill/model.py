from __future__ import annotations
import torch
from torch import nn

class CausalBlock(nn.Module):
    def __init__(self,width:int,heads:int,mlp_ratio:float):
        super().__init__();self.n1=nn.LayerNorm(width);self.attn=nn.MultiheadAttention(width,heads,batch_first=True);self.n2=nn.LayerNorm(width);hidden=int(width*mlp_ratio);self.mlp=nn.Sequential(nn.Linear(width,hidden),nn.GELU(),nn.Linear(hidden,width))
    def forward(self,x):
        n=x.shape[1];mask=torch.ones(n,n,device=x.device,dtype=torch.bool).triu(1);y,_=self.attn(self.n1(x),self.n1(x),self.n1(x),attn_mask=mask,need_weights=False);x=x+y;return x+self.mlp(self.n2(x))

class BrowserCausalVideoStudent(nn.Module):
    """Small causal latent-video transformer with an ONNX-friendly public contract."""
    def __init__(self,latent_channels=4,text_width=256,width=1024,layers=24,heads=16,mlp_ratio=4.0):
        super().__init__();self.latent_channels=latent_channels;self.width=width;self.input=nn.Linear(latent_channels,width);self.text=nn.Linear(text_width,width);self.time=nn.Sequential(nn.Linear(1,width),nn.SiLU(),nn.Linear(width,width));self.blocks=nn.ModuleList([CausalBlock(width,heads,mlp_ratio) for _ in range(layers)]);self.norm=nn.LayerNorm(width);self.output=nn.Linear(width,latent_channels)
    def forward(self,noisy_latents,timestep,prompt_embeds):
        b,c,t,h,w=noisy_latents.shape;x=noisy_latents.permute(0,2,3,4,1).reshape(b,t*h*w,c);cond=self.text(prompt_embeds.mean(1)).unsqueeze(1)+self.time(timestep.reshape(b,1).float()/1000).unsqueeze(1);x=self.input(x)+cond;hidden=[]
        for block in self.blocks:x=block(x);hidden.append(x)
        out=self.output(self.norm(x)).reshape(b,t,h,w,c).permute(0,4,1,2,3);return {"noise_pred":out,"hidden_states":hidden}

def build_student(cfg:dict):return BrowserCausalVideoStudent(**cfg)
def build_toy_teacher(cfg:dict):return BrowserCausalVideoStudent(**cfg)
