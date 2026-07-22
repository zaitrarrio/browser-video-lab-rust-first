#!/usr/bin/env python3
"""Compress Wan 2.1 T2V 1.3B with Pruna Smash and run an optional smoke test."""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
import torch
from diffusers import WanPipeline

def args():
    p=argparse.ArgumentParser()
    p.add_argument("--model",default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    p.add_argument("--output",default="artifacts/wan21-t2v-1.3b-smashed")
    p.add_argument("--compiler",action="append",default=[])
    p.add_argument("--kernel",action="append",default=[])
    p.add_argument("--token",default=os.getenv("PRUNA_TOKEN"))
    p.add_argument("--dtype",choices=["bf16","fp16"],default="bf16")
    p.add_argument("--smoke-test",action="store_true")
    p.add_argument("--prompt",default="A silver robot walks through Austin at sunset")
    return p.parse_args()

def main():
    a=args(); dtype=torch.bfloat16 if a.dtype=="bf16" else torch.float16
    if not torch.cuda.is_available(): raise SystemExit("CUDA GPU required for Wan compression")
    from pruna import SmashConfig, smash
    pipe=WanPipeline.from_pretrained(a.model,torch_dtype=dtype).to("cuda")
    cfg=SmashConfig()
    cfg["compilers"]=a.compiler or ["torch_compile"]
    if a.kernel: cfg["kernels"]=a.kernel
    started=time.perf_counter()
    try: smashed=smash(model=pipe,token=a.token,smash_config=cfg)
    except TypeError: smashed=smash(model=pipe,api_key=a.token,smash_config=cfg)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);smashed.save_model(str(out))
    metadata={"source":a.model,"dtype":a.dtype,"compilers":cfg["compilers"],"kernels":a.kernel,"seconds":time.perf_counter()-started,"gpu":torch.cuda.get_device_name()}
    (out/"compression.json").write_text(json.dumps(metadata,indent=2)+"\n")
    if a.smoke_test:
        result=smashed(prompt=a.prompt,num_frames=17,num_inference_steps=4,height=256,width=256)
        from diffusers.utils import export_to_video
        export_to_video(result.frames[0],str(out/"smoke-test.mp4"),fps=8)
    print(json.dumps(metadata,indent=2))
if __name__=="__main__": main()
