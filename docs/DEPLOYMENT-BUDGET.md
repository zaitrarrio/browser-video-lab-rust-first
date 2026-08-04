# What a generation actually costs in a tab

`WHITEPAPER.md` §9 says "No benchmark numbers are reported because none have been
earned." This document earns one, and it is uncomfortable: **on a laptop dGPU, one
0.8-second clip takes about four minutes, and the output is not pixels.**

The strategy risk this measures is that nothing in the distillation pipeline
reduces the number of function evaluations. Distillation here buys a 3.4×
parameter reduction and an 8× download reduction; it leaves the 32-step sampler
untouched, and step count is the dominant term in the deployment cost.

## Method

`video-native` runs the same Burn/cubecl/WGPU stack `video-web` compiles to wasm,
on a real GPU. It now separates model construction, cubecl's first-call shader
compilation, and the steady-state forward, because a single-shot run conflates all
three — an early reading of this was 3× pessimistic for exactly that reason.

Hardware: **AMD Radeon Pro 5500M**, 4 GB, a mid-range laptop discrete GPU, chosen
deliberately over the rented 5090 as the closer proxy for a browser. Its ~4 TFLOPS
fp32 is a vendor figure, *not* independently measured here — unlike the 5090's
239.3 TFLOPS bf16 in `PERF-ROUND-2.md` §1, which was measured with torch.

Geometry is the trained one: `[1,16,4,40,40]`, 1600 tokens, 13 pixel frames at
320×320 — 0.8 s of video at 16 fps.

## The measurement

| spec | width · layers | params | s/step | achieved | 32 steps |
| --- | --- | --- | --- | --- | --- |
| `browser-demo` | 192 · 4 | ~2M | 0.117 | 0.10 TFLOPS | **3.7 s** |
| `probe-79m` | 640 · 16 | 118M | 2.003 | 0.18 TFLOPS | **64.1 s** |
| `validation-320-adaln` | 1152 · 24 | 574M | 7.604 | 0.20 TFLOPS | **243.3 s** |

Per-step time tracks FLOPs almost exactly — 0.10 to 0.20 TFLOPS across a 130×
range of model size — so this is compute-bound at roughly **5% of the card**, the
same order as every other cubecl number in this project.

## Three things the table does not say

**1. The shipped demo is a toy.** `browser-demo.json` is width 192, 4 layers,
`text_width` 64 — about 2M parameters. It is the only configuration that has ever
run in a browser. The 383M student has not.

**2. The browser path does one frame, not video.** `video-web::generate` takes a
`side` and builds `[1,C,1,side,side]`. At the real student and 400 tokens that is
0.728 s/step and **23.3 s** for 32 steps — but it is a single latent frame, and the
temporal-difference term the student was trained with is never exercised. The
source says so plainly; this is a documented gap, not a hidden one.

**3. There is no decoder.** The RGBA the browser returns false-colours three of
sixteen latent channels. Turning a latent into pixels needs the Wan2.1 VAE, which
runs server-side in `decode_latents.py` and does not exist for wasm. So the 243 s
above buys a debug view. A browser VAE is a second model, unbudgeted here.

## The gap, and what could close it

| target | needed |
| --- | --- |
| 30 s per clip | 8.1× |
| 5 s per clip | **48.7×** |
| 1 s per clip | 243× |

| lever | worth | status |
| --- | --- | --- |
| steps 32 → 4 | **8.0×** | **not implemented** — needs a different objective |
| kernel efficiency 0.20 → 1.2 TFLOPS | **6.0×** | the cubecl gap, measured here at ~5% of peak |
| model 574M → 118M | 3.8× | measured, but costs quality |
| int4 quantization | **1.0×** | download only — 1.53 GB → 191 MB, no arithmetic win |

Steps and kernels together are 48×, which lands at ~5 s per clip. Neither exists
today.

**The int4 row is the one most likely to be misread.** §3.6 of the whitepaper is
right that int4 is 8× smaller than f32, and that solves the *download*. It does
not touch the arithmetic: the runtime dequantizes onto a fresh model and the
matmuls run in f32. Quantization is a bandwidth and footprint win, not a latency
one, and no wgpu int4 matmul path exists to make it otherwise.

## Why this is a strategy risk and not a tuning problem

Step reduction is not a knob on this pipeline. The student is trained to regress
the teacher's velocity field pointwise, which is a *32-step* field; running it in
4 steps does not approximate it. Few-step generation is a different objective —
consistency distillation, distribution matching (DMD), self-forcing — and adopting
one changes the cache, the loss and the evaluation together.

So the two halves of the project are currently inconsistent: the training
objective produces a many-step model, and the deployment target needs a few-step
one. That inconsistency survives every bug fixed in validation rounds 4–7, and it
is worth deciding on *before* spending GPU-days on a schedule the browser cannot
run.

## What this does not show

* **One GPU, native, not wasm.** WebGPU through wasm adds indirection and loses
  the native shader cache between sessions, so these are optimistic lower bounds
  for a tab, not upper ones.
* **The ~4 TFLOPS peak is a datasheet figure**, so "5% of the card" is
  approximate. The *relative* numbers and the FLOP-proportionality are measured.
* **No VAE cost is included anywhere**, because no browser VAE exists to measure.
* **Nothing here is about quality.** A model that ran in 5 s and could not draw
  would be no better than round 7's.
