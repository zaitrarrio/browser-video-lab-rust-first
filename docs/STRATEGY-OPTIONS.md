# Three strategies, evaluated

Companion to [`DEPLOYMENT-BUDGET.md`](DEPLOYMENT-BUDGET.md), which prices the
current design at **~430 s per 0.8-second clip** on a 4 GB laptop dGPU (~370 s of
sampler, ~57 s of VAE) against a 5-second target — an **86× gap**. This evaluates
three candidate directions against that number and against the quality failure of
[`VALIDATION-ROUND-7.md`](VALIDATION-ROUND-7.md).

The short version: **two of the three converge on the same teacher swap, and the
third is aimed at a problem this project does not have yet.**

## 1. The VAE — measured, and it changes the ordering

| | |
| --- | --- |
| parameters | 126.9M (73.3M decoder) |
| download | 484 MB f32 · 121 MB int8 · **60.5 MB int4** |
| decode `[1,16,4,40,40]` → `[1,3,13,320,320]` | **11.49 TFLOP**, counted |
| projected on the browser stack | **~57 s**, ≈ 7.6 denoising steps |

The decoder is **not** today's bottleneck. But it is paid **once** where a step is
paid 32 times, so it does not shrink when the sampler does:

| configuration | sampler | VAE | total |
| --- | --- | --- | --- |
| today (1600 tokens, 32 steps) | ~370 s | 57 s | ~430 s |
| 400 tokens, 32 steps | ~25 s | 57 s | **82 s** |
| 400 tokens, 4 steps | ~3 s | 57 s | **60 s — 95% decoder** |

**Every path leads to the decoder.** Two clips into any optimisation programme, the
VAE is the deliverable's cost. It is also the one component required by *all* the
strategies below, and the only piece of work that is never wasted.

Download is fine — 60.5 MB int4 on top of the student's 191 MB.

## 2. Image-to-video — and the teacher swap it implies

**What it fixes, and it is precisely the measured failure.** Round 5 localised the
defect to high σ, where `x_σ` carries almost no clip identity and the model must
synthesise structure from the prompt alone. Round 7 showed it cannot, because it
has no positional encoding — but even with that fixed, high-σ synthesis is the
hardest thing being asked. **Conditioning on a first frame supplies the structure
instead of demanding it be invented.** The model animates rather than composes.

It also relieves the umT5-XXL → umt5-small mismatch (strategy risk #1), because
the image carries most of the conditioning.

**The teacher.** Wan2.1-1.3B is text-to-video only; its I2V sibling is 14B, which
breaks the free-tier economics `WHITEPAPER.md` §3.1 is built on.
**Wan2.2-TI2V-5B** is Apache-2.0, dense 5B, and natively text-*and*-image-to-video.
`TEACHER-OPTIONS.md` already analysed it as Plan C.

**And it happens to be the biggest measured speedup available.** Wan2.2-TI2V's VAE
is spatial-16 against Wan2.1's spatial-8, so 320×320 becomes a 20×20 latent, and a
4-frame clip is **400 tokens rather than 1600**. Measured on the laptop GPU with
the production student at that geometry:

| geometry | tokens | s/step | 32 steps | stable? |
| --- | --- | --- | --- | --- |
| Wan2.1 (spatial-8) | 1600 | 7.5–15.8 | 240–500 s | **no**, 2× swing |
| Wan2.2-TI2V (spatial-16) | 400 | 0.78 | **25 s** | yes, ±5% |

That is ~15× on the sampler *and* it is the difference between fitting in 4 GB and
thrashing. `DEPLOYMENT-BUDGET.md` shows efficiency collapsing 0.42 → 0.13 TFLOPS
between 400 and 1600 tokens; 400 is the last point that behaves.

**This reverses a recorded recommendation, in a specific scope.**
`PERF-ROUND-1.md` §5 concluded "Plan C should not be undertaken as a *performance*
measure — its speed argument is 1.63× for a teacher swap." That measurement stands:
it was taken on an **A100/5090-class card, training, at large `--accum`**, where
batching amortises per-kernel overhead and memory is ample. The deployment regime
is the opposite — **batch 1, inference, 4 GB** — and there the same token reduction
is worth ~15×. Both numbers are right in their own regime; §5's advice needs a
scope note rather than a correction.

**Costs, honestly:**

* A 5B teacher against 1.3B — cache generation gets roughly 4× slower, and
  `TEACHER-OPTIONS.md` §Plan C already flags that 5B does not fit the free-tier CPU
  path at all.
* `latent_channels` 16 → 48 through the spec, the contract, the quantiser and the
  browser runtime.
* A different VAE, whose decode cost is **unmeasured** — it decodes from a smaller
  latent to the same pixels, so it may well be *more* expensive per decode, not
  less. This should be measured before committing.
* It is a different product. "Type a prompt, get video" becomes "supply an image".

## 3. MemFlow — right idea, wrong layer, wrong problem

[arXiv 2512.14699](https://arxiv.org/abs/2512.14699), Kling AI Research. Narrative
Adaptive Memory retrieves historically relevant frames by cross-attending the
current text prompt against past visual tokens; Sparse Memory Activation restricts
attention to the top-k of those. Reported at **18.7 FPS on a single H100**, against
a memory-free baseline at 20.3 — a 7.9% cost for 60-second narrative consistency.

**It solves a problem this project does not have.** MemFlow is about coherence
across a *minute* of video. This project cannot yet produce a coherent *0.8
seconds*. Adding a memory bank to a model that does not draw buys nothing.

**Its speed claim does not transfer.** 18.7 FPS is on an H100 at roughly 990
TFLOPS. The browser stack measures **0.42 TFLOPS** at 400 tokens — about 2,400×
less compute. MemFlow is also explicitly an *overhead* on top of its base, not a
saving.

**But its substrate is the interesting part.** MemFlow is compatible with "any
streaming video generation model with KV cache", and its baseline is **LongLive** —
which is this project's original Plan A teacher and the source of the
`longlive_distill` package name. That family is *causal, streaming, and few-step
per chunk*, which is exactly the property `DEPLOYMENT-BUDGET.md` says the current
32-step objective lacks.

So the useful reading is: **adopt the substrate, not the paper.** A streaming
few-step generator is what fixes the NFE problem; MemFlow becomes relevant only
once 5 seconds works and 60 is wanted. Filing it as a future concern rather than a
current option.

## What this adds up to

1. **Port the VAE to wgpu, and measure it first.** Required by every strategy, on
   the critical path of all of them, and currently the thing that 95% of an
   optimised pipeline's time would go to. If a wgpu conv3d decode turns out to cost
   200 s rather than 57, several options die and it is better to know now.
2. **Measure the Wan2.2-TI2V VAE before committing to the teacher swap.** The
   sampler win is measured and large; the decode side is assumed and could offset
   it.
3. **Then the teacher swap**, which buys the token reduction and image
   conditioning in one move and is already scoped in `TEACHER-OPTIONS.md`.
4. **Positional encoding regardless** — round 7's defect is orthogonal to all of
   this, and a fast model that cannot draw is worth nothing.
5. **MemFlow: not now.**

## What this does not show

* **The Wan2.2 numbers are the *student* at Wan2.2's geometry**, not the Wan2.2
  teacher. Token count is what was measured; nothing here validates that a student
  distilled from a spatial-16 teacher reaches the same quality — a 4× coarser
  latent is a real loss of spatial detail, and `TEACHER-OPTIONS.md` should be
  re-read on that point.
* **The ~57 s VAE projection applies a matmul throughput to a convolution stack.**
  cubecl's conv3d support is unverified here. Order of magnitude only.
* **MemFlow was evaluated from its abstract and two summaries**, not the full
  paper — the PDF did not parse. Base model, parameter count and steps-per-chunk
  are unconfirmed, so the "substrate" reading is inference from its baseline
  comparison rather than something read directly.
