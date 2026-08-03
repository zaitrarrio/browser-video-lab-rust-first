# Performance round 2 — the gap to torch is not cubecl's kernels

One measurement session on a rented RTX 5090 (`$0.335/h`, Florida), run alongside
the overfit probe ([`OVERFIT-PROBE.md`](OVERFIT-PROBE.md)). What was learned, in
the order it was found, because each result motivated the next:

1. The trainer reaches a small fraction of the card's bf16 matmul throughput (§1).
2. That idle capacity is **not** recoverable by scheduling — co-locating a second
   trainer *lowered* aggregate throughput (§2).
3. At a **fixed parameter count**, efficiency rises monotonically with width — a
   **1.72×** wall-clock win from a spec-file edit (§3).
4. **cubecl's matmul autotuning had never been compiled in**, and restoring it
   changes no code. Kernel *fusion*, on top of it, is a regression (§5).
5. **The first measurement protocol was wrong** and under-reported Burn; §5a has
   the fix, two independent validations of it, and which tables it invalidates.
6. **Running the identical Rust model on LibTorch's kernels buys 3%.** The 2.28×
   torch advantage at production geometry is 1.03× kernels, 1.35× framework
   overhead, 1.63× fused attention — and the fused-attention kernel **already
   exists in `burn-cubecl`**; `burn-autodiff` discards it (§6).
7. The Rust trainer is a **65 MB binary linking no CUDA libraries at all**,
   against a 7.0 GB torch environment (§7).

**Bottom line: do not adopt `burn-tch`, and do not port the trainer to PyTorch on
current evidence.** Make attention's backward reachable in Burn instead — that is
1.63× and it keeps the binary.

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

**Autotune is a large free win; fusion is a net loss on top of it.** The sign and
ordering are trustworthy; the magnitudes in that table are **not** — see §5a. The
plausible reading of fusion is that Burn's fusion layer pays graph bookkeeping to
remove elementwise traffic autodiff tapes anyway, and may constrain what autotune
can pick. `cuda` now carries `burn/autotune`; `cuda-fusion` stays reachable only so
the negative result is reproducible.

## 5a. The measurement protocol, and the bug in the first one

**The bug.** §3 and §5 ran a "warm-up" chunk and a "timed" chunk as *separate
processes*, subtracting `state.json`'s accumulated `train_seconds`. That cancels
nothing: `--resume` starts a fresh binary and cubecl's JIT and autotune caches are
per-process, so every timed chunk paid full kernel-compilation and tuning cost
inside its own measured window.

**The fix.** Model a process as `t(N) = S + N/r` and take two fresh runs at
different step counts:

```
r = (N₂ − N₁) / (t₂ − t₁)          N₁ = 40, N₂ = 200
```

which removes a constant per-process startup instead of pretending a previous
process warmed it away.

**Two independent checks that it works.**

| check | slope protocol | reference | agreement |
| --- | --- | --- | --- |
| burn-cuda 640×16 | 19.618 | 19.567, from a 22,013-step single-process run | **0.26%** |
| burn-cuda 1152×24 | 8.849 | 8.86, predicted analytically from S | **0.12%** |

The fitted startup is **13.9 s for cubecl** on both shapes — constant, as the model
assumes — and **0.2–0.4 s for LibTorch**, whose kernels are precompiled and need no
JIT or autotune. That difference is itself the cleanest confirmation of what was
leaking.

**Consequences.** §3's width sweep and §5's table both still carry the leak. A
constant `S` inflates a *fast* config's measured time by a larger fraction, so both
**understate** the fast rows: the true width advantage in §3 is larger than 1.72×,
and autotune's true value is larger than +41.6%. Neither has been re-run; the
corrected `cuda` figures at 640×16 and 1152×24 in §6 are the only Burn numbers in
this document measured properly.

## 6. Burn against PyTorch, and against LibTorch

`python/bench_torch_student.py` reimplements `video-student` — same patchify, same
adaLN-zero blocks, same three-term loss, same AdamW at lr 2e-5 / wd 0.01 / clip
1.0, same accum 8, bf16 throughout. Structural parity is checked, not assumed:
torch reports 574.6M parameters against the Rust `approximate_parameters`
formula's 574.0M, the difference being the biases and LayerNorms it omits.

`--backend tch` runs the **identical Rust model** on LibTorch's ATen kernels —
same trainer, same loss, same optimizer, same checkpoints, one dispatch arm. That
is what separates *"cubecl's kernels are slow"* from *"Burn is slow"*, which no
comparison against Python could distinguish.

All Burn figures below use the two-point slope protocol of §5a. torch is timed
in-process after 3 warm-up iterations.

| implementation | 640 · 16 | vs burn-cuda | 1152 · 24 | vs burn-cuda | % of peak |
| --- | --- | --- | --- | --- | --- |
| **burn-cuda** (autotune) | 19.618 | 1.00× | **8.849** | 1.00× | 16.7% |
| **burn-tch** (ATen kernels) | 22.624 | 1.15× | **9.151** | **1.03×** | 17.3% |
| torch, naive attention | 47.241 | 2.41× | 12.352 | 1.40× | 23.3% |
| torch, SDPA | 58.264 | 2.97× | **20.163** | **2.28×** | 38.1% |
| torch, SDPA + `compile` | — | — | 21.259 | 2.40× | 40.2% |

### The gap is not the kernels

**burn-tch buys 3% at production geometry.** Handing Burn the exact kernels torch
uses recovers almost nothing, which retires the hypothesis this whole section was
written to test. cubecl's kernels are competitive with ATen's for this workload —
that is a real vindication of Burn's compiler, and it is the opposite of what §1's
"3–8% of peak" implied.

The 2.28× therefore decomposes cleanly, and every factor is measured:

| factor | step | multiplier |
| --- | --- | --- |
| kernel quality | burn-cuda → burn-tch | **1.03×** |
| framework overhead | burn-tch → torch naive | **1.35×** |
| fused attention | torch naive → torch SDPA | **1.63×** |
| | product | 2.26× (measured total 2.28×) |

So the money is in the last two: 1.35× of graph construction, autodiff taping and
dispatch that torch does more cheaply, and 1.63× of fused attention.

### The fused-attention 1.63× is reachable without leaving Burn

`burn-cubecl` **already has a flash-attention kernel** (`cubek-attention`, with its
own autotuner, at `kernel/attention/{base,tune}.rs`), and `burn-tch` maps Burn's
`attention` op to `tch::Tensor::scaled_dot_product_attention`. Neither is
reachable in training:

```rust
// burn-autodiff-0.21.0/src/ops/module.rs:1876
fn attention(...) -> ... {
    attention_fallback::<Self>(query, key, value, mask, attn_bias, options)
}
```

Unconditional — no `B::has_attention_backward()` escape hatch of the kind
`ctc_loss` twelve lines below it has. **Burn cannot train with fused attention on
any backend today, because autodiff throws the kernel away.** (Note also that the
cubecl kernel falls back whenever `options.scale.is_some()`, which this model
would set.)

That reframes `perf/tiled-attention`'s conclusion. It said a batch dimension needs
"a fused softmax whose backward recomputes rather than stores" and treated that as
a kernel-writing project. The forward kernel already exists in-tree; what is
missing is a backward for it in `burn-autodiff`. That is a much smaller, more
targeted piece of work, and it is worth **1.63×** plus whatever memory it frees.

### What it reprices

3.2M sample-views at accum 8:

| trainer | wall clock | cost at $0.335/h |
| --- | --- | --- |
| burn-cuda, autotune | **4.19 d** | $34 |
| torch SDPA | **1.84 d** | $15 |

*Caveats.* The torch bench runs on preallocated synthetic tensors with no shard
decode; round 1 measured shard-load at 0.2–0.3% of wall clock. AdamW over bf16
parameters matches what Burn does but is not what anyone should ship. The
`burn-tch` build required `LIBTORCH_BYPASS_VERSION_CHECK=1` because `torch-sys
0.22` pins libtorch 2.9.0 and the box has 2.11.0; it built, ran, and tracked the
CUDA backend's loss curve to ~0.5% at step 1, but an ABI mismatch that silently
changes numerics is not fully excluded.

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
generation, the dataset build and VAE decode. Round 1 §3 measured the teacher at
92% of TGP and listed micro-optimising it under "what not to re-attempt"; here the
entire 1536-shard cache took about seven minutes. Porting Wan2.1's DiT, VAE and
umt5 to Burn is a very large rewrite aimed at the part that is not the bottleneck.

**TensorRT cannot train.** It builds inference engines; there is no backward pass,
so it is structurally inapplicable to the multi-GPU-day number that motivates all
of this. Its plausible targets are teacher-cache generation (already saturated)
and VAE decode (seconds per clip), and it can never touch the deliverable, which
ships to WebGPU where an NVIDIA-native engine cannot go.

**`burn-tch` is a measured no.** 1.03× at production geometry, for a ~2 GB libtorch
dependency that destroys the 65 MB binary of §7. Its value was diagnostic, not
practical: it proves the gap is not cubecl's kernels.

**A PyTorch trainer is a weaker case than §6's headline suggests.** 2.28× is real,
and worth 2.35 GPU-days and about $19 on the deliverable as scoped. But 1.63× of
it is fused attention that Burn already has a kernel for, and giving up the
dependency-free binary — on rented boxes where four of five machines this session
were unusable — is not obviously worth the remaining 1.4×.

**The recommendation, in order:**

1. **Make `attention`'s backward reachable in Burn.** `burn-cubecl` ships the
   forward flash kernel with its own autotuner; `burn-autodiff` discards it with an
   unconditional `attention_fallback`. Worth **1.63×** plus the activation memory
   that `perf/tiled-attention` could not free, and it keeps everything about the
   Rust story intact. This is now the single highest-value engineering item.
2. **Re-run §3 and §5 under the §5a protocol.** Both understate the fast
   configurations. Cheap, and §3's 1.72× shape win may be larger than recorded.
3. **Revisit PyTorch only if (1) lands and the residual still hurts.** At that
   point the question is a 1.4× framework-overhead gap against a 65 MB binary,
   which is a legitimate judgement call rather than a rout.
4. **The browser student stays Rust/WGPU regardless.** Nothing here touches it.

And none of this is urgent: [`VALIDATION-ROUND-4.md`](VALIDATION-ROUND-4.md) shows
the pipeline does not yet produce coherent video. A 2× faster trainer for a model
that cannot draw is worth nothing.

## 9. Still the biggest unexploited lever

Even the best Burn row leaves **89% of the card unused**, and torch leaves 60%.
Fused attention with recompute-on-backward is what `PERF-ROUND-1.md` §6 said it
was — at 1600 tokens the Rust trainer still materializes a
`[b, heads, 1600, 1600]` probability matrix and autodiff still tapes it. §6 now
puts a number on what fixing it is worth, and a number on what it would leave.
