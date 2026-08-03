# Performance round 2 — the card is 90% idle, and the lever is the model's shape

One measurement session on a rented RTX 5090, run alongside the overfit probe
([`OVERFIT-PROBE.md`](OVERFIT-PROBE.md)). Four results, in the order they were
found, because each one motivated the next:

1. The trainer reaches **7–16% of the card's bf16 matmul throughput**, and the
   CUDA backend did not fix what round 1 blamed on wgpu.
2. The idle capacity is **not** recoverable by running more jobs — co-locating a
   second trainer lowered aggregate throughput.
3. At a **fixed parameter count**, arithmetic efficiency rises monotonically with
   width: 8.5% of peak at width 640, 25.2% at width 2304. That is a **1.72×**
   wall-clock win over the shipped geometry, available from a spec-file edit.
4. **cubecl's matmul autotuning had never been compiled in.** `default-features
   = false` on the workspace's burn pin — correct, and there for the wasm graph —
   also stripped `autotune` from the native CUDA trainer. Restoring it is worth
   **+41.6%** at production geometry and changes no code. Kernel *fusion*, turned
   on alongside it, is a **regression** (§5).

Sections 1–3 were all measured **before** finding 4, so every absolute figure in
them is a no-autotune figure. The relative conclusions survive; the absolute ones
are superseded by §5.

## 1. What the card actually does

`torch` on the same GPU, as the reference for what the silicon delivers when a
tensor-core path is used:

| benchmark | result |
| --- | --- |
| matmul 8192³ bf16 | **113.5 TFLOPS** |
| matmul 8192³ f32 | 37.0 TFLOPS |
| 512 MB device-to-device copy | 708 GB/s |

Against the trainer, computing FLOPs as
`3 · (2·layers·(4+2·mlp)·width²·tokens + layers·4·tokens²·width)` per sample:

| run | samples/s | achieved | % of bf16 peak |
| --- | --- | --- | --- |
| 79M probe (w640 · 16L), cuda/bf16 | 7.28 | 7.8 TFLOPS | **6.9%** |
| 383M (w1152 · 24L), cuda/bf16 | 4.09 | 18.5 TFLOPS | **16.3%** |
| 383M, wgpu/f32 (round 1's 2.04) | 2.04 | 9.2 TFLOPS | 8.1% |

(An earlier reading of 9.96 samples/s for the 79M probe was taken by differencing
step numbers out of a live run's log. The trainer logs every 50 steps, so over a
300-step window that carries ±17% quantisation error. Every figure in this
document instead differences `state.json`'s accumulated `train_seconds` across
two chunks, which is exact; the two 1152×24 measurements taken that way, in
different sessions, agree to 0.5%.)

**Correction to `PERF-ROUND-1.md` §1.** That section attributes the missing time
to *"cubecl's wgpu backend has no tensor-core path"*, quoting ~9.6 of ~105
TFLOPS. The CUDA backend is 1.6–2.5× faster in wall clock but sits at the same
order of inefficiency — 16% of peak, not 90%. Whatever the wgpu tensor-core
story is, **switching backends did not buy the arithmetic**, and any future plan
that assumes CUDA closed that gap is wrong.

## 2. Occupancy is not the lever

At 237 W of a 575 W TGP and 5.3 GB of 32 GB, the obvious hypothesis is that one
job cannot fill the card and two would. Measured, over five minutes with both
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

Every config below holds `layers · width² ≈ 31.85M` — so all of them are ~574M
parameters including the constant adaLN-zero term — with `head_dim = 64`
throughout. The only thing moving is the aspect ratio of the matmuls. Protocol:
an 8-step warm-up chunk discarded to absorb cubecl's runtime kernel compilation,
then 30 timed steps at accum 8; the figure is the delta in `state.json`'s
accumulated `train_seconds`, which excludes both the JIT and model construction.

| width | layers | heads | samples/s | vs shipped | GFLOP/sample | TFLOPS | % of peak |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 640 | 78 | 10 | 1.856 | 0.46× | 5214 | 9.7 | 8.5% |
| 768 | 54 | 12 | 2.367 | 0.58× | 4943 | 11.7 | 10.3% |
| 1152 | 24 | 18 | 4.027 | 0.99× | 4519 | 18.2 | 16.0% |
| **1152** | **24** | **16** | **4.072** | **1.00×** | 4519 | 18.4 | 16.2% |
| 1536 | 14 | 24 | 5.244 | 1.29× | 4466 | 23.4 | 20.6% |
| 2048 | 8 | 32 | 6.181 | 1.52× | 4369 | 27.0 | 23.8% |
| 2304 | 6 | 36 | **6.992** | **1.72×** | 4094 | 28.6 | **25.2%** |

Monotonic across a 3.6× range of widths. End to end, **3.77× the wall-clock
throughput for the same number of parameters**, of which 2.96× is efficiency and
1.27× is doing less work (attention is `layers · 4 · tokens² · width`, so at fixed
`layers · width²` it falls as `1/width`).

**The protocol reproduces round 1.** The `w1152 · 24L · 16H` row is the shipped
`validation-320` geometry and measures 4.072 samples/s against
`PERF-ROUND-1.md` §4's 4.103 for cuda/bf16/accum 8 — 0.8% apart, on a different
box with a different cache. The sweep is measuring what round 1 measured.

### What it reprices

3.2M sample-views, the schedule §4 costed:

| shape | 3.2M views @ accum 8 |
| --- | --- |
| 1152 · 24 (shipped) | 9.1 d |
| 1536 · 14 | 7.1 d |
| 2048 · 8 | 6.0 d |
| 2304 · 6 | **5.3 d** |

Round 1's headline was **7.4 GPU-days** at accum 32. The same 1.72× puts that at
**~4.3 GPU-days** — a larger single win than anything in round 1 except the move
to CUDA, and it costs one JSON file. It also composes with `--accum`, which is a
separate 2.35×.

## 4. What this does NOT show, and it is the whole risk

**Nothing about quality.** This is a throughput measurement and only a throughput
measurement. A 6-layer, width-2304 DiT is not interchangeable with a 24-layer,
width-1152 one just because they have the same parameter count — depth is not a
free variable in a transformer, and round 3 has just finished demonstrating that
at low budget *architecture dominates everything else*
([`VALIDATION-ROUND-3.md`](VALIDATION-ROUND-3.md)). A 1.72× speedup that costs
more than 1.72× in sample-efficiency is a loss.

So the honest recommendation is **not** "ship width 2304". It is:

* **1536 · 14 (1.29×) or 2048 · 8 (1.52×) is the defensible middle** — a real
  speedup without betting the model on six layers.
* **The quality A/B is now the blocking question**, and the probe harness in
  `scripts/overfit-probe.sh` is built to run exactly it: same cache, same floor,
  one boolean's worth of difference, decoded to an MP4 at the end.

Other limits, stated plainly:

* One measurement per config, 30 timed steps, one box. Round 1 used a median of
  three repeats with drift checks at both ends of the table; this does not.
* Peak VRAM was not recorded per config. Wide/shallow should use *less* attention
  memory (fewer layers × the `[b,heads,N,N]` tape), which may interact with the
  batch ceiling `perf/tiled-attention` could not move — untested.
* `head_dim` was held at 64, which is not what the shipped spec uses (1152/16 =
  72). The 1152 row was run both ways and they differ by 1.1%, so head_dim is not
  driving the trend.
* All at 1600 tokens. The attention term scales with `tokens²`, so the balance
  between the dense and attention paths — and therefore the shape of this curve —
  will move at other resolutions.

## 5. Autotuning was compiled out; fusion is a regression

Everything above says the trainer is far off the silicon, which raises the
strategic question — leave Burn for PyTorch, or for hand-written CUDA? Before
answering it, check the build.

`burn-cuda`'s own default features are `["std", "fusion", "autotune", …]`. The
workspace pins burn with `default-features = false` — deliberately, so `train`
and `dataset` and the C `libsqlite3-sys` they drag in cannot reach
`wasm32-unknown-unknown` — and `burn/cuda` enables `burn-cuda` *without*
`burn-cuda/default`. `cargo tree -p video-train --features cuda` confirms
`burn-fusion` is absent from the graph. **Matmul autotuning and kernel fusion
have never been compiled into the CUDA trainer**, and `video-train` is never in
the wasm graph, so nothing was gained by excluding them.

`Cuda<F, I>` is itself `#[cfg(feature = "fusion")] Fusion<CubeBackend<…>>`, so
this is a manifest change and no code change. Same protocol as §3 — 8 warm-up
steps discarded (which for the autotune arms absorbs the tuning benchmarks
themselves), then 30 timed steps at accum 8:

| features | shape | samples/s | vs baseline | % of peak |
| --- | --- | --- | --- | --- |
| `cuda` (baseline) | 1152 · 24 | 4.092 | — | 16.3% |
| `cuda` + **autotune** | 1152 · 24 | **5.795** | **+41.6%** | **23.1%** |
| `cuda` + autotune + fusion | 1152 · 24 | 5.313 | +29.8% | 21.2% |
| `cuda` (baseline) | 640 · 16 | 7.283 | — | 6.9% |
| `cuda` + **autotune** | 640 · 16 | **9.025** | **+23.9%** | 8.5% |
| `cuda` + autotune + fusion | 640 · 16 | 7.461 | +2.4% | 7.0% |

**Autotune is a large free win. Fusion is a net loss on top of it** — 8.3% slower
at 1152×24 and 17% slower at 640×16. The plausible reading is that Burn's fusion
layer pays graph-construction and dispatch bookkeeping to eliminate elementwise
traffic that autodiff is going to tape anyway, so the saving never materialises;
it may also be constraining what autotune is allowed to pick. `cuda` now carries
`burn/autotune`; `cuda-fusion` remains reachable purely so the negative result
stays reproducible.

**This reprices §1 and §3.** At production geometry the trainer is at 23.1% of
peak, not 16.3%. The 3.2M-view schedule goes from 9.1 days to **6.4 days** at
accum 8, and round 1's 7.4-day headline to **~5.2 days** — again, for a manifest
line. §3's width sweep was run without autotune and needs redoing before its
1.72× can be quoted alongside this: autotune's entire job is picking good kernels
per shape, so it is exactly the thing that could shrink — or widen — a
shape-dependent gap.

## 6. So: Rust+CUDA, or PyTorch, or TensorRT?

The question this section was written to answer, with the numbers above.

**The trainer is already Rust+CUDA.** What remains in Python is teacher-cache
generation, the dataset build and the VAE decode. Round 1 §3 measured the teacher
at 92% of TGP and listed micro-optimising it under "what not to re-attempt"; in
this session the whole 1536-shard cache took about seven minutes. Porting Wan2.1's
DiT, VAE and umt5 to Burn would be a very large rewrite aimed squarely at the part
that is not the bottleneck.

**TensorRT cannot train.** It builds inference engines; there is no backward pass,
so it is structurally inapplicable to the multi-GPU-day number that motivates all
of this. Its plausible targets here are teacher-cache generation (92% TGP already)
and VAE decode (seconds per clip). And it can never touch the deliverable itself:
the student ships to WebGPU/WASM, where an NVIDIA-native engine cannot go.

**The 16%-of-peak figure that motivated the question was a build flag**, worth
+41.6% of the gap for one manifest line. That does not close the remaining
distance to torch's 113.5 TFLOPS — the trainer is at 23.1% — but it changes the
decision. The honest position:

* **Stay in Rust.** The PyTorch-free trainer is a deliberate property of this
  project (`task rust:train:smoke` exists to prove it), and the case for
  abandoning it rested on a number that was 41% pessimistic.
* **Re-run §3's width sweep with autotune on** before spending anything on shape.
* **A torch-vs-Burn head-to-head on the same student is still worth measuring**,
  because 23% of peak is not 50% — but it is now an optimisation question, not an
  architecture question, and it should wait until the probe has said whether this
  pipeline produces video at all.

## 7. Still the biggest unexploited lever

Even the best row here leaves **75% of the card unused**. Fused attention with
recompute-on-backward remains what `PERF-ROUND-1.md` §6 said it was: at 1600
tokens the trainer still materializes a `[b, heads, 1600, 1600]` probability
matrix and autodiff still tapes it. Nothing in this round changes that
conclusion — it only shows that the dense path, too, is running at a quarter of
the silicon.
