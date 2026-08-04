"""What would the Wan2.1 VAE cost in a browser tab?

`docs/DEPLOYMENT-BUDGET.md` prices the *sampler* at 243 s per 0.8-second clip on a
laptop dGPU and notes that the output is not pixels: turning a latent into video
needs the VAE, which today only runs server-side in `decode_latents.py`. Every
strategy that puts video in a tab needs a wgpu decoder, so its cost is on the
critical path regardless of which one is chosen — and nobody has measured it.

Porting the VAE to Burn to find out would be days of work. This gets the answer
without that, by measuring the two things that are portable:

* **FLOPs**, counted exactly with `torch.utils.flop_counter`, which is a property
  of the architecture and not of the framework running it.
* **Parameters**, which set the download and the memory footprint.

and then projecting onto the throughput the *same Burn/cubecl/WGPU stack* was
measured achieving on a mid-range laptop GPU (0.20 TFLOPS at 1600 tokens,
`video-native`). That projection is the honest part to distrust: convolutions and
transformers stress a backend differently, so treat it as an order of magnitude,
not a number.

The comparison that matters is against the sampler's 32 steps. A decode is paid
**once** per clip; a denoising step is paid 32 times.
"""
from __future__ import annotations

import argparse
import time

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    p.add_argument("--frames", type=int, default=4, help="latent frames")
    p.add_argument("--size", type=int, default=40, help="latent side")
    p.add_argument("--iters", type=int, default=3)
    # Measured for the Burn/cubecl/WGPU stack on an AMD Radeon Pro 5500M.
    p.add_argument("--browser-tflops", type=float, default=0.20)
    p.add_argument("--sampler-step-s", type=float, default=7.604,
                   help="measured s/step for validation-320-adaln on that same GPU")
    p.add_argument("--steps", type=int, default=32)
    a = p.parse_args()

    from diffusers import AutoencoderKLWan

    vae = AutoencoderKLWan.from_pretrained(a.model_id, subfolder="vae", torch_dtype=torch.float32).to("cuda").eval()
    params = sum(q.numel() for q in vae.parameters())
    dec_params = sum(q.numel() for q in vae.decoder.parameters()) if hasattr(vae, "decoder") else float("nan")
    print(f"VAE params total {params/1e6:.1f}M   decoder {dec_params/1e6:.1f}M")
    print(f"  download at f32 {params*4/2**20:6.1f} MB | int8 {params/2**20:6.1f} MB | int4 {params/2/2**20:6.1f} MB")

    z = torch.randn(1, 16, a.frames, a.size, a.size, device="cuda")

    # Exact FLOP count, architecture-only, framework-independent.
    from torch.utils.flop_counter import FlopCounterMode
    counter = FlopCounterMode(display=False)
    with counter, torch.no_grad():
        vae.decode(z, return_dict=False)
    flops = counter.get_total_flops()

    with torch.no_grad():
        for _ in range(2):
            vae.decode(z, return_dict=False)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(a.iters):
            out = vae.decode(z, return_dict=False)[0]
        torch.cuda.synchronize()
        cuda_s = (time.perf_counter() - t0) / a.iters

    print(f"\ndecode {tuple(z.shape)} -> {tuple(out.shape)}")
    print(f"  {flops/1e12:8.2f} TFLOP per decode   (counted, not estimated)")
    print(f"  {cuda_s:8.3f} s on this CUDA GPU with torch  -> {flops/cuda_s/1e12:.1f} TFLOPS achieved")

    proj = flops / (a.browser_tflops * 1e12)
    sampler = a.sampler_step_s * a.steps
    print(f"\nprojected onto the browser stack at {a.browser_tflops} TFLOPS:")
    print(f"  VAE decode        {proj:8.1f} s   (paid once)")
    print(f"  sampler {a.steps} steps  {sampler:8.1f} s   (paid {a.steps}x)")
    print(f"  total             {proj+sampler:8.1f} s   VAE is {proj/(proj+sampler)*100:.1f}% of it")
    print(f"\n  the VAE is worth {proj/a.sampler_step_s:.1f} denoising steps")


if __name__ == "__main__":
    main()
