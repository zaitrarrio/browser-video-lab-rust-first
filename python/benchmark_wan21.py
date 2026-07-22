#!/usr/bin/env python3
import argparse,time,torch
from pruna.engine.PrunaModel import PrunaModel
from diffusers.utils import export_to_video
p=argparse.ArgumentParser();p.add_argument("model");p.add_argument("--output",default="wan-benchmark.mp4");p.add_argument("--prompt",default="A silver robot walks through Austin at sunset");a=p.parse_args()
m=PrunaModel.load_model(a.model);torch.cuda.synchronize();t=time.perf_counter();r=m(prompt=a.prompt,num_frames=33,num_inference_steps=8,height=480,width=832);torch.cuda.synchronize();elapsed=time.perf_counter()-t
export_to_video(r.frames[0],a.output,fps=16);print({"seconds":elapsed,"frames":33,"generation_fps":33/elapsed})
