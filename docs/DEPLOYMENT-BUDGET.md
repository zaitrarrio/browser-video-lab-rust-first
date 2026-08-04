# What a generation actually costs in a tab

`WHITEPAPER.md` §9 says "No benchmark numbers are reported because none have been
earned." This document earns one, and it is uncomfortable: **on a laptop dGPU, one
0.8-second clip takes roughly six minutes, and the output is not pixels.**

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
| `validation-320-adaln` | 1152 · 24 | 574M | 7.5–15.8 | 0.10–0.20 TFLOPS | **240–500 s** |

**The production row is a range, and that is a finding rather than sloppiness.**
Six repeats at 1600 tokens gave 7.5, 7.5, 11.0, 12.0, 12.8 and 15.8 s/step — a 2×
swing — while 400 tokens gave 0.73, 0.75, 0.78, 0.81, tight. The 574M student's f32
weights alone are **2.14 GB of a 4 GB card**, so at 1600 tokens it is at the edge of
VRAM and the timing goes erratic. A first draft of this document quoted the best
observation, 7.604 s, as if it were typical; the median is nearer 11.5 s and the
honest projection is **~370 s**, not 243.

Token count is where the cliff is:

| tokens | s/step | GFLOP | achieved | vs 400 tokens |
| --- | --- | --- | --- | --- |
| 400 | 0.776 | 323 | **0.42 TFLOPS** | 1.00× |
| 800 | 3.958 | 682 | 0.17 TFLOPS | 5.10× time for 2.11× FLOPs |
| 1600 | ~11.5 | 1506 | 0.13 TFLOPS | ~15× time for 4.66× FLOPs |

Efficiency falls 0.42 → 0.13 TFLOPS as tokens rise, and time grows three times
faster than FLOPs do. On a 4 GB GPU this workload is not compute-bound at the top
end — it is memory-bound, and 400 tokens is the last point that behaves.

### The VAE, measured

`vae_budget.py` counts the decode exactly with `torch.utils.flop_counter` rather
than estimating it:

| | |
| --- | --- |
| parameters | 126.9M total, 73.3M decoder |
| download | 484 MB f32 · **121 MB int8** · 60.5 MB int4 |
| decode `[1,16,4,40,40]` → `[1,3,13,320,320]` | **11.49 TFLOP** |
| on the 5090 with torch | 0.227 s (50.6 TFLOPS achieved) |
| projected at the browser stack's 0.20 TFLOPS | **~57 s** |

So a decode is worth **7.6 denoising steps**. It is not today's bottleneck — 57 s
against ~370 s — but it is paid once where steps are paid 32 times, which means
**step distillation moves the bottleneck onto the VAE rather than removing it.** At
4 steps the split becomes ~46 s of sampler against ~57 s of decode.

Caveat: that projection applies a throughput measured on the DiT's matmuls to a 3D
convolution stack. Convolutions stress a backend differently, cubecl's conv3d
support is unverified here, and torch itself gets only 50.6 TFLOPS on the VAE
against 91 on the student. Treat ~57 s as an order of magnitude.

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
runs server-side in `decode_latents.py` and does not exist for wasm. The sampler
time above buys a debug view; the decode is now costed above, but not ported.

## The gap, and what could close it

Against ~370 s of sampler plus ~57 s of decode, call it **~430 s**:

| target | needed |
| --- | --- |
| 30 s per clip | 14× |
| 5 s per clip | **86×** |
| 1 s per clip | 430× |

| lever | worth | status |
| --- | --- | --- |
| **1600 → 400 tokens** (spatial-16 VAE teacher) | **~15× on the sampler** | measured; also the only point that fits 4 GB. See `STRATEGY-OPTIONS.md`. |
| steps 32 → 4 | 8.0× on the sampler | **not implemented** — needs a different objective |
| kernel efficiency | ~3–6× | the cubecl gap |
| model 574M → 118M | 3.8× | measured, costs quality |
| int4 quantization | **1.0×** | download only — no arithmetic win |

Note these do not multiply cleanly, because the decode does not shrink with steps
or tokens. Tokens and steps together take the sampler from ~370 s to ~3 s and leave
the VAE's ~57 s untouched — at which point the decoder *is* the deliverable's cost
and nothing else matters until it is addressed.

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
