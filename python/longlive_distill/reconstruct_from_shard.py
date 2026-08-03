"""Reconstruct x0 from a cache shard using the *teacher's* own velocity, and decode it.

The convention check that no round has run. `video-train sample`'s module doc
states the flow-matching identities the whole pipeline rests on:

    x_s = (1 - s)*x0 + s*eps        s in [0,1]
    v   = eps - x0                  what the teacher predicts
    =>  x0 = x_s - s*v

So a cache shard already contains everything needed to rebuild the clip it was
drawn from — no student involved. If that reconstruction decodes to the reference
video, the noising parameterisation, the sigma/timestep mapping and the decode are
all mutually consistent, and any failure to produce coherent video is the
student's. If it does *not*, the student has been learning a field its own sampler
could never integrate, and every parity number ever recorded is against the wrong
target.

Caveat worth stating: caches built with `--guidance g > 0` store the CFG-combined
velocity, not the exact v for that (x0, eps) pair, so the reconstruction is
approximate by construction. It should still be obviously the clip.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors.torch import load_file


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shard", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model-id", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    a = p.parse_args()

    t = load_file(a.shard)
    x_s = t["noisy_latents"].float()
    v = t["teacher_noise_pred"].float()
    sigma = float(t["timestep"].float().reshape(-1)[0]) / 1000.0
    x0 = x_s - sigma * v

    print(f"sigma={sigma:.4f}")
    for name, ten in (("noisy x_s", x_s), ("teacher v", v), ("recovered x0", x0)):
        print(f"  {name:14s} std={ten.std():.4f} mean={ten.mean():+.4f}")

    from diffusers import AutoencoderKLWan

    from .decode_latents import decode_to_video

    vae = AutoencoderKLWan.from_pretrained(a.model_id, subfolder="vae", torch_dtype=torch.float32).to("cuda")
    decode_to_video(vae, x0, Path(a.output))
    print(f"wrote {a.output}")


if __name__ == "__main__":
    main()
