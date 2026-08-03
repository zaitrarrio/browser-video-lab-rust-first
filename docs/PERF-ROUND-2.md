# Performance round 2 — the trainer is far off the card, and one protocol bug

> **Read §6a first.** A measurement-protocol error means the Burn figures in §3,
> §5 and §6 are too low, and the torch/Burn ratios are overstated. Corrected where
> a corrected number exists; flagged where one does not.

One measurement session on a rented RTX 5090 (`$0.335/h`, Florida), run alongside
the overfit probe ([`OVERFIT-PROBE.md`](OVERFIT-PROBE.md)). Six results, in the
order they were found, because each one motivated the next:

1. The trainer reaches **3–8% of the card's bf16 matmul throughput** (§1).
2. That idle capacity is **not** recoverable by scheduling — co-locating a second
   trainer *lowered* aggregate throughput (§2).
3. At a **fixed parameter count**, efficiency rises monotonically with width:
   4.1% of peak at width 640, 12.0% at width 2304, a **1.72×** wall-clock win
   from a spec-file edit (§3).
4. **cubecl's matmul autotuning had never been compiled in.** Restoring it is
   worth **+41.6%** and changes no code. Kernel *fusion* is a regression (§5).
5. **The same model in PyTorch is 3.5× faster at production geometry and 6.5×
   faster at the probe's**, and two thirds of that gap is *not* about fused
   attention (§6).
6. But the Rust trainer is a **65 MB binary that links no CUDA libraries at
   all**, against a 7.0 GB torch environment (§7).

## 1. What the card actually does

`torch` on the same GPU, as the reference for what the silicon delivers:

| benchmark | idle card | measured while the trainer ran |
| --- | --- | --- |
| matmul 8192³ bf16 | **239.3 TFLOPS** | 113.5 |
| matmul 8192³ f32 | 69.7 TFLOPS | 37.0 |
| 512 MB device copy | 1523.5 GB/s | 708 |

**Take the reference on an idle card.** The right-hand column was taken with the
probe training in the background and is roughly half the real figure — which is
itself the cleanest evidence for §2. Every percentage in this document is against
**239.3 TFLOPS**; an earlier draft quoted the contended 113.5 and so overstated
every efficiency by 2.1×.

Against the trainer, with FLOPs counted as
`3 · (2·layers·(4+2·mlp)·width²·tokens + layers·4·tokens²·width)` per sample:

| run | samples/s | achieved | % of bf16 peak |
| --- | --- | --- | --- |
| 79M probe (w640 · 16L), cuda/bf16 | 7.28 | 7.8 TFLOPS | **3.3%** |
| 383M (w1152 · 24L), cuda/bf16 | 4.09 | 18.5 TFLOPS | **7.7%** |
| 383M, wgpu/f32 (round 1's 2.04) | 2.04 | 9.2 TFLOPS | 3.8% |

**Correction to `PERF-ROUND-1.md` §1.** That section attributes the missing time
to *"cubecl's wgpu backend has no tensor-core path"*, quoting ~9.6 of ~105
TFLOPS. The CUDA backend is 1.6–2.5× faster in wall clock but sits at the same
order of inefficiency. **Switching backends did not buy the arithmetic**, and any
plan assuming CUDA closed that gap is wrong.

*Method note.* Throughput here is the delta in `state.json`'s accumulated
`train_seconds` across a discarded warm-up chunk and a timed chunk, which excludes
cubecl's runtime kernel compilation and model construction. An earlier reading of
9.96 samples/s for the 79M probe came from differencing step numbers in a live
log that only prints every 50 steps — ±17% over a 300-step window. Two 1152×24
measurements taken the exact way, in different sessions, agree to 0.5%.

## 2. Occupancy is not the lever

At 237 W of a 575 W TGP and 5.3 GB of 32 GB, the obvious hypothesis is that one
job cannot fill the card and two would. Measured over five minutes with both
trainers live:

| | solo | co-located |
| --- | --- | --- |
| 79M steps/s | 1.245 | 0.498 (**40%**) |
| 383M steps/s | ~0.51 | 0.329 (64%) |
| aggregate | 22.6 TFLOPS (383M alone) | **16.2 TFLOPS** |
| power | 237 W | 292 W of 575 W |

**Adding an entire second training job made aggregate throughput worse than
running the larger job alone**, and moved power by 55 W. The spare watts are not
addressable by scheduling. Do not re-attempt co-location as a throughput measure.

## 3. Efficiency is a function of width, at constant parameters

Every config holds `layers · width² ≈ 31.85M` — all ~574M parameters including
the constant adaLN-zero term — at `head_dim = 64`. Only the matmul aspect ratio
moves.

| width | layers | heads | samples/s | vs shipped | GFLOP/sample | TFLOPS | % of peak |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 640 | 78 | 10 | 1.856 | 0.46× | 5214 | 9.7 | 4.1% |
| 768 | 54 | 12 | 2.367 | 0.58× | 4943 | 11.7 | 4.9% |
| 1152 | 24 | 18 | 4.027 | 0.99× | 4519 | 18.2 | 7.6% |
| **1152** | **24** | **16** | **4.072** | **1.00×** | 4519 | 18.4 | 7.7% |
| 1536 | 14 | 24 | 5.244 | 1.29× | 4466 | 23.4 | 9.8% |
| 2048 | 8 | 32 | 6.181 | 1.52× | 4369 | 27.0 | 11.3% |
| 2304 | 6 | 36 | **6.992** | **1.72×** | 4094 | 28.6 | **12.0%** |

Monotonic across a 3.6× range of widths: **3.77× the wall clock for the same
parameters**, of which 2.96× is efficiency and 1.27× is doing less work
(attention is `layers · 4 · tokens² · width`, so at fixed `layers · width²` it
falls as `1/width`).

**The protocol reproduces round 1.** The `w1152 · 24L · 16H` row is the shipped
`validation-320` geometry at 4.072 samples/s against `PERF-ROUND-1.md` §4's 4.103
for cuda/bf16/accum 8 — 0.8% apart, different box, different cache.

**This sweep predates §5 and was run without autotune.** Autotune's whole job is
picking kernels per shape, so it is exactly the thing that could shrink or widen a
shape-dependent gap. Re-run it before quoting the 1.72× alongside §5's +41.6%.

## 4. What §3 does NOT show, and it is the whole risk

**Nothing about quality.** A 6-layer width-2304 DiT is not interchangeable with a
24-layer width-1152 one just because the parameter counts match — depth is not a
free variable in a transformer, and round 3 has just finished demonstrating that
at low budget *architecture dominates everything else*
([`VALIDATION-ROUND-3.md`](VALIDATION-ROUND-3.md)). A 1.72× speedup that costs
more than 1.72× in sample-efficiency is a loss. **1536 · 14 or 2048 · 8 is the
defensible middle**, and the quality A/B is what `scripts/overfit-probe.sh` is
built to run.

Other limits: one measurement per config, 30 timed steps, one box (round 1 used a
median of three); peak VRAM not recorded per config; `head_dim` held at 64 rather
than the shipped 72 (the 1152 row was run both ways and differs by 1.1%); all at
1600 tokens, and the attention term scales with `tokens²`.

## 5. Autotuning was compiled out; fusion is a regression

`burn-cuda`'s own default features are `["std", "fusion", "autotune", …]`. The
workspace pins burn with `default-features = false` — deliberately, so `train`,
`dataset` and the C `libsqlite3-sys` they drag in cannot reach
`wasm32-unknown-unknown` — and `burn/cuda` enables `burn-cuda` *without*
`burn-cuda/default`. `cargo tree -p video-train --features cuda` confirms
`burn-fusion` absent. **Autotuning and fusion had never been compiled into the
CUDA trainer**, and `video-train` is never in the wasm graph, so excluding them
bought nothing. `Cuda<F, I>` is itself
`#[cfg(feature = "fusion")] Fusion<CubeBackend<…>>`, so this is a manifest change
and no code change.

| features | shape | samples/s | vs baseline | % of peak |
| --- | --- | --- | --- | --- |
| `cuda` (baseline) | 1152 · 24 | 4.092 | — | 7.7% |
| `cuda` + **autotune** | 1152 · 24 | **5.795** | **+41.6%** | **10.9%** |
| `cuda` + autotune + fusion | 1152 · 24 | 5.313 | +29.8% | 10.0% |
| `cuda` (baseline) | 640 · 16 | 7.283 | — | 3.3% |
| `cuda` + **autotune** | 640 · 16 | **9.025** | **+23.9%** | 4.1% |
| `cuda` + autotune + fusion | 640 · 16 | 7.461 | +2.4% | 3.3% |

**Autotune is a large free win; fusion is a net loss on top of it** — 8.3% slower
at 1152×24, 17% at 640×16. The plausible reading is that Burn's fusion layer pays
graph bookkeeping to remove elementwise traffic autodiff tapes anyway, and may
constrain what autotune can pick. `cuda` now carries `burn/autotune`;
`cuda-fusion` stays reachable only so the negative result is reproducible.

## 6. The same model in PyTorch

`python/bench_torch_student.py` reimplements `video-student` — same patchify, same
adaLN-zero blocks, same three-term loss (output MSE + 0.25 temporal + 0.05
relation grams over two layers), same AdamW at lr 2e-5 / wd 0.01 / clip 1.0, same
accum 8, all in bf16. Structural parity is checked, not assumed: torch reports
574.6M parameters against the Rust `approximate_parameters` formula's 574.0M, the
difference being the biases and LayerNorms that formula omits.

Three attention arms, because *"torch is faster"* and *"torch has a fused
attention kernel Burn 0.21 lacks"* are different claims with different
consequences. The `naive` arm materializes the same `[b, heads, N, N]` matrix
`tiled_attention` does.

**Production geometry, 1152 · 24 · 16H, 1600 tokens:**

| implementation | attention | samples/s | TFLOPS | % of peak | peak mem |
| --- | --- | --- | --- | --- | --- |
| Burn, no autotune | tiled | 4.092 | 18.5 | 7.7% | — |
| Burn, **autotune** | tiled | 5.795 | 26.2 | 10.9% | — |
| torch | naive | 12.352 | 55.8 | 23.3% | 8.1 GiB |
| torch | **SDPA** | **20.163** | 91.1 | 38.1% | 6.3 GiB |
| torch | SDPA + `compile` | **21.259** | 96.1 | **40.2%** | 6.1 GiB |

**Probe geometry, 640 · 16 · 8H:**

| implementation | attention | samples/s | TFLOPS | % of peak | peak mem |
| --- | --- | --- | --- | --- | --- |
| Burn, **autotune** | tiled | 9.025 | 9.7 | 4.1% | — |
| torch | naive | 47.241 | 50.5 | 21.1% | 2.3 GiB |
| torch | **SDPA** | **58.264** | 62.3 | 26.0% | 1.7 GiB |

Against the best Burn configuration available today:

| | 1152 · 24 | 640 · 16 |
| --- | --- | --- |
| torch naive (same attention algorithm) | **2.13×** | **5.23×** |
| torch SDPA | **3.48×** | **6.46×** |
| torch SDPA + compile | **3.67×** | — |

**Two thirds of the gap is not attention.** The `naive` arm uses the identical
algorithm and identical materialized matrix, and is still 2.13× ahead at
production geometry — so a fused attention kernel in cubecl, the fix
`PERF-ROUND-1.md` §6 has been pointing at since round 1, would recover roughly a
third of the difference and leave the rest.

**The gap widens as the model shrinks.** 6.46× at width 640 against 3.48× at
1152, consistent with §3: Burn's per-kernel overhead dominates at small matmuls.
Anything narrow — a browser-sized student, an ablation, a probe — is where Burn
costs the most.

**Memory is the other half.** torch+SDPA holds the whole accum-8 step in 6.3 GiB
of 32 GB. `perf/tiled-attention` concluded that a batch dimension needs "a fused
softmax whose backward recomputes rather than stores"; SDPA is that kernel, and it
turns the batch ceiling from a research problem into a flag.

**What it reprices.** 3.2M sample-views at accum 8:

| trainer | wall clock | cost at $0.335/h |
| --- | --- | --- |
| Burn, no autotune | 9.1 d | $73 |
| Burn, autotune | 6.4 d | $51 |
| torch SDPA | **1.84 d** | **$15** |
| torch SDPA + compile | 1.74 d | $14 |

*Caveats.* The torch bench runs on preallocated synthetic tensors with no shard
decode; round 1 measured shard-load at 0.2–0.3% of wall clock, so this is not
material, but it is not zero. AdamW over bf16 parameters matches what Burn does
but is not what anyone should ship — f32 master weights would cost some
throughput and memory back.

## 6a. CORRECTION — the Burn side of §3, §5 and §6 is measured too low

The 26,513-step probe run reports **22,013 steps in 9000.4 s of `train_seconds`,
i.e. 19.567 samples/s** at 640×16 with autotune. §5 benchmarked that same shape at
**9.025**. The live number is **2.17× higher**, and the live number is the honest
one — it is a single process, a single `train_seconds` reading, and two and a half
hours long.

**The protocol bug: the warm-up chunk and the timed chunk are separate
processes.** `--resume` starts a fresh binary, and cubecl's JIT and autotune caches
live in-process. So the "discarded warm-up" discarded nothing — every timed chunk
paid full kernel-compilation and autotune-benchmarking cost inside its own
measured window. That also explains the size of the effect: 2.17× on the autotune
arms (tuning benchmarks are expensive) against ~1.37× on the baseline arms (JIT
only).

What this invalidates:

* **§5's absolute figures.** Autotune's true value is larger than +41.6%, because
  the autotune arm paid tuning cost that the baseline arm did not. The sign and
  ordering hold; the magnitudes do not.
* **§6's Burn column, and therefore every torch/Burn ratio.** At 640×16 the
  corrected ratio is **58.264 / 19.567 = 2.98×**, not 6.46×. The 1152×24 Burn
  figure has **not** been re-measured — no long single-process run exists at that
  shape — so 3.48× is unquantified until one does.

**The leak is additive, not multiplicative**, and estimating it the other way is a
mistake worth naming because a first draft of this section made it. Startup is a
roughly fixed per-process cost, so it inflates a *fast* config's measured time by a
larger fraction than a slow one's. Solving `t = S + N/r` at the one shape where
both numbers are known:

| | 640×16, autotune |
| --- | --- |
| timed chunk actually took | 26.59 s |
| real work in 30 steps (from the 22,013-step run) | 12.27 s |
| **leaked startup S** | **14.33 s** |

Carrying that same S to 1152×24, whose timed chunk took 41.42 s, gives ~8.86
samples/s and a torch/Burn ratio of **≈2.28×**:

| assumption for 1152×24 | torch/Burn |
| --- | --- |
| as published | 3.48× |
| constant *multiplier* (wrong model) | 1.60× |
| **constant *startup* (defensible)** | **≈2.28×** |

So the honest bracket is **2.3–3.5×**, and on the 3.2M-view schedule that is 4.2
GPU-days against torch's 1.84 — a difference of about **$19**, not the $36 §8
quotes.

* **§3's width sweep is affected in a known direction.** It ran without autotune,
  so only JIT warm-up leaked in and S is smaller — but a fixed S penalises the
  *fast* rows more than the slow ones, so the sweep **understates** the wide/shallow
  advantage. The true benefit is larger than 1.72×, not smaller.

**Do not quote §5 or §6 until they are re-measured**, either as one long single-
process run per config, or as a two-point slope across two different step counts
in fresh processes — `rate = (N₂ − N₁) / (t₂ − t₁)` — which cancels a constant
per-process startup instead of pretending it was warmed away.

## 7. What the Rust trainer buys, measured

The counter-argument to §6 is the environment, and it is real:

| | Rust `video-train` | torch |
| --- | --- | --- |
| artifact | **65.5 MB** single binary | 7.0 GB venv |
| of which CUDA libs | **none** | 4.2 GB (`nvidia/*` wheels) |
| dynamic links | libc, libm, libgcc, vdso | Python + ~40 shared objects |
| host requirement | NVIDIA driver | driver + Python + toolchain |

`ldd` on the trainer lists **five** entries and not one of them is CUDA — cubecl
opens the driver API at runtime. A training box needs a kernel driver and a
65 MB file.

This matters more than it looks, because of how the pipeline splits. Teacher-cache
generation needs torch and the 30 GB of Wan2.1 weights; **training does not** — it
consumes the framework-neutral shard cache, which is the whole point of that
design. So the expensive, repeated, multi-day half of the pipeline is exactly the
half that can run on a minimal box, and a torch trainer would put a 7 GB
environment back onto it.

The honest limit of that argument: the shard cache is 19 GB for 1536 shards, so a
"lean" training box still stages a large dataset, and if one box does both jobs it
needs torch regardless. The saving is real only when cache generation and training
are separated — which is what the framework-neutral cache was built for, and which
is how a long run would actually be operated.

## 8. So: Rust+CUDA, PyTorch, or TensorRT?

**The trainer is already Rust+CUDA.** What remains in Python is teacher-cache
generation, the dataset build, and VAE decode. Round 1 §3 measured the teacher at
92% of TGP and listed micro-optimising it under "what not to re-attempt"; here the
entire 1536-shard cache took about seven minutes. Porting Wan2.1's DiT, VAE and
umt5 to Burn is a very large rewrite aimed at the part that is not the bottleneck,
and it would *not* remove the torch dependency from cache generation unless all
three were ported.

**TensorRT cannot train.** It builds inference engines; there is no backward pass,
so it is structurally inapplicable to the multi-GPU-day number that motivates all
of this. Its plausible targets are teacher-cache generation (92% TGP already) and
VAE decode (seconds per clip), and it can never touch the deliverable itself,
which ships to WebGPU where an NVIDIA-native engine cannot go.

**So the real choice is Burn-vs-torch for the trainer only**, and it is a genuine
trade rather than a rout:

* **Take the free 41.6% now** (§5, done) and re-run §3's width sweep with autotune
  before spending anything on shape. Both are manifest/JSON edits.
* **Burn's remaining cost is 3.5× at production geometry and 6.5× at probe
  geometry** — roughly 4.5 GPU-days and $36 on the deliverable as scoped, plus a
  multiple on every ablation, which is where the width and depth questions in §4
  have to be answered.
* **Rust buys a 65 MB dependency-free trainer**, which is worth real money and
  real reliability on rented boxes — four of the five machines rented for this
  session were unusable, and a smaller install surface is fewer ways to fail.
* **The browser student stays Rust/WGPU regardless.** Nothing in §6 touches it;
  that is the project's thesis, and TensorRT and torch are both irrelevant to it.

The measurement that would settle it is **a fused attention kernel in cubecl**.
It recovers ~1/3 of the gap (§6's naive arm bounds this), and if the remaining
~2.1× on the dense path is acceptable for the environment win, Burn stays and the
question is closed. That is a bounded piece of work with a known payoff, and it
should be scheduled *after* the probe answers whether this pipeline produces video
at all — a 3.5× speedup on a pipeline that does not work is worth nothing.

## 9. Still the biggest unexploited lever

Even the best Burn row leaves **89% of the card unused**, and torch leaves 60%.
Fused attention with recompute-on-backward is what `PERF-ROUND-1.md` §6 said it
was — at 1600 tokens the Rust trainer still materializes a
`[b, heads, 1600, 1600]` probability matrix and autodiff still tapes it. §6 now
puts a number on what fixing it is worth, and a number on what it would leave.
