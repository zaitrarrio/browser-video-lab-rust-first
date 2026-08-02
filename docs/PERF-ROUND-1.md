# Performance round 1 — where the time actually goes

Three trainer experiments, one teacher-cache ablation, and a verification pass over
the merged result. The through-line is uncomfortable and worth stating first:
**every strong prior about where the time went was wrong**, in both directions, and
only measurement settled it. Two of the results below exist specifically so nobody
re-attempts the thing that did not work.

Hardware: one rented RTX 5090 (32 GB, 575 W TGP) per run. Cache-lever numbers are
the median of 3 timed repeats after a discarded warm-up, with a `base` run at both
ends of the table as a drift check (0.3% apart). Trainer-branch numbers are quoted
from their own commits and were not re-measured except where stated.

## 1. The trainer branches

### `perf/pipeline-stalls` — the host-stall theory, refuted

The trainer reached 8.8% of the card and the standing theory was host-side stalls:
24 blocking device→host syncs per optimizer step, two hidden-state readbacks, and
single-threaded shard decode. All three were removed, and `--profile` was added to
find out whether they had ever mattered. On wgpu:

| phase | share | per sample |
| --- | --- | --- |
| fwd+bwd | 97.1% | 473.32 ms |
| optim | 2.5% | 11.99 ms |
| shard-load | 0.3% | 1.28 ms |
| upload | 0.1% | 0.61 ms |
| readback | 0.0% | 0.06 ms |

Removing every stall moved throughput 2.036 → 2.061 samples/s (**+1.2%**). The
missing time is arithmetic running at ~9.6 of ~105 TFLOPS, because cubecl's wgpu
backend has no tensor-core path. The changes were kept because they are strictly
less work on the critical path and because they are what makes the 97.1% *mean*
something, not because they were fast.

### `perf/half-precision` — memory win, not a speed win

`--precision {f32,f16,bf16}` across train/sample/eval. On CUDA:

| precision | samples/s | peak MB | final loss |
| --- | --- | --- | --- |
| f32 | 5.48 | 15302 | 1.4966 |
| f16 | 6.02 | 8064 | 1.85 |
| bf16 | 5.71 | 8064 | 1.5093 |

Memory halves; throughput barely moves. The real win is the ceiling: at a 4×40×40
latent, f32 OOMs on 32 GB while f16 completes in 13.8 GB. Checkpoints stay f32
regardless (`FullPrecisionSettings`), so a half-precision run still produces a
record the f32 trainer and `video-web` load unchanged.

### `perf/tiled-attention` — a negative result, deliberately recorded

Query-block tiling (exact, not approximate) against the `[b,heads,seq,seq]`
probability matrix: peak memory 25638 → 24006 MiB (**−6.4%**), throughput
3.10 → 3.19 samples/s (**+2.9%**). Burn 0.21 has no fused kernel, so autodiff still
tapes every tile and only the transient peak moves. One sample still holds ~16.1 GB
of activations, ~14.5 GB of it attention; a batch of 2 would need ~40 GB.

**This does not unlock a batch dimension, and a wider tile will not change that.**
What would is a fused softmax whose backward recomputes rather than stores.

## 2. Verifying the merge

All three merged into `main` (pipeline-stalls fast-forward; half-precision with
three conflict hunks; tiled-attention clean). 17/17 workspace tests pass. The merge
commit explicitly listed what it could not verify without a GPU. That verification
has now run — on a **real** 32-shard teacher cache (`wan21-flow-shift3`, latents
from `make_wan_dataset.py`), `validation-320` spec, 40 steps × accum 8, seed 42.

| run | steps | first → last | non-finite | result |
| --- | --- | --- | --- | --- |
| f32 | 40 | 2.6851 → 2.1578 | 0 | ok |
| bf16 | 40 | 2.6777 → 1.9764 | 0 | ok |
| bf16 (rerun) | 40 | 2.6777 → 1.9764 | 0 | bit-reproducible |
| f16 | 12 | 2.6853 → — | 5 | **hard-failed** |
| f16 `--grad-clip 1e30` | 16 | 2.6853 → — | 5 | **hard-failed** |
| f32 `--grad-clip 1e30` | 40 | 2.6851 → 1.9998 | 0 | ok |

No `TypeMismatch` on any path, which is what the merge risked: half-precision's
`read_f32` had to be reapplied to the *single consolidated readback* that
pipeline-stalls introduced, since that readback is now the only one on the hot path
and would otherwise have failed on every step of every half-precision run.

**Correction to `perf/half-precision`.** That commit attributes f16 instability to
`clip_by_norm` reducing `sum(g²)` in f16, and states the same run at
`--grad-clip 1e30` "descends smoothly to 1.4948". On a real teacher cache it does
not: f16 goes non-finite and hard-fails with clipping inert, and the diagnostic
names **hidden layer 0** — a forward-pass overflow, not a gradient-norm artifact.
The commit's *conclusion* (bf16 is the usable half format, f16 is not) stands and is
in fact stronger than written; its causal explanation does not.

`--profile` after the merge, on CUDA:

| phase | share | per sample |
| --- | --- | --- |
| fwd+bwd | 93.1% | 154.73 ms |
| optim | 6.5% | 10.78 ms |
| upload | 0.3% | 0.47 ms |
| shard-load | 0.2% | 0.25 ms |
| readback | 0.0% | 0.02 ms |

Same conclusion as the wgpu profile — arithmetic dominates, host stalls are nil.
`optim` is proportionally larger only because fwd+bwd got faster. The profiled run's
loss trajectory is identical to the unprofiled one.

**Residual growth is depth-driven.** Over 40 steps `layerlast_norm` climbs
2195 → 22588 while `layer0_norm` stays flat (~1100–1650). That is exactly the
distinction the hidden-state logging was added to make, and it points *away* from
the time/text conditioning injected before block 0 as the cause.

## 3. Teacher-cache creation

Seven proposed accelerations for `cache_teacher.py`, ablated one at a time against
base. 4 clips × 16 draws, guidance 5.0, `--relation-layers 2`, latent
`[1,16,4,32,32]`.

| lever | shards/s | vs base | mean power | changes cache? |
| --- | --- | --- | --- | --- |
| base | 6.13 | — | 528 W | — |
| **bf16 teacher** | **16.10** | **+163%** | 429 W | yes — mean 6.9%, max 3.45 |
| batch draws | 7.22 | +18% | 571 W | no — mean 6e-4 |
| no grams | 6.18 | +0.8% | 541 W | identical |
| no hook capture | 6.18 | +0.8% | 543 W | identical |
| async writes | 6.16 | +0.5% | 542 W | identical |
| batch CFG cond+uncond | 5.65 | **−8%** | 495 W | no — max 2.5e-3 |
| all combined | 20.30 | +231% | 522 W | yes — mean 6.8% |

**The GPU was never idle.** Base already draws 528 W of 575 W TGP. The premise that
batch-1 teacher forwards waste the card is false at this geometry — one 1.3B-DiT
forward over 1024 tokens is already substantial work — which is why batching the
draws returns 18% rather than the several-fold win it was predicted to.

Saturation sweep (bf16, batched, no grams), shards/s by draw batch:
1 → 13.95, 2 → 18.74, 4 → 19.12, 8 → 20.11, 16 → 20.29. **The knee is 4–8**; batch
16 doubles VRAM for +0.9%.

Dataset generation (`make_wan_dataset.py`) was measured single-shot, without
warm-up or repeats, so these are weaker evidence: base 1.91 s/clip, passing
precomputed embeddings instead of re-encoding 1.78 s/clip (+6%), batching clips
1.51 s/clip (+21%).

## 4. The 18-GPU-day estimate, re-measured

Round 1 costed the deliverable at **~18 GPU-days** from 2.04 samples/s (wgpu, f32,
batch 8, 1600 tokens) against a 3.2M sample-view schedule. That number is what
everything above is ultimately trying to move, so it was re-measured at exactly
that geometry — same `validation-320` spec, same `[1,16,4,40,40]` latent, a real
f32 teacher cache built at 320×320.

| backend / precision / batch | samples/s | vs 2.04 | 3.2M views |
| --- | --- | --- | --- |
| wgpu f32, accum 8 (round 1) | 2.04 | — | 18.2 d |
| cuda f32, accum 8 | 3.265 | 1.60× | 11.4 d |
| cuda f32, accum 32 | 3.668 | 1.80× | 10.1 d |
| cuda bf16, accum 8 | 4.103 | 2.01× | 9.0 d |
| **cuda bf16, accum 32** | **5.000** | **2.45×** | **7.4 d** |

**CUDA alone is 1.60×, not the ~3× an earlier cross-geometry comparison implied.**
That estimate came from dividing a 1600-token wgpu figure by a 1024-token CUDA one
and should not have been quoted as a backend multiplier.

**bf16's value here is throughput, not memory.** Its gain grows with batch — +10.7%
at accum 8, +18.6% at 16, +28.8% at 32, and +36.3% at accum 32 once warm-up is
amortised — which is what a memory-bound workload looks like: more batch means more
attention traffic, so halving the element width matters more. Note this contradicts
the +4% measured in `perf/half-precision`, which was taken at 1024 tokens; the
benefit is geometry-dependent and grows with token count.

**f32 does not OOM at this geometry.** `perf/half-precision` reports f32 OOMing
outright on 32 GB at 4×40×40 while half precision fits in 13.8 GB. It does not
reproduce: f32 completed at accum 8, 16 and 32, sitting at ~24 GB of 32 GB at the
largest. This also resolves the tension with round 1, which ran f32 at exactly this
geometry to produce the 2.04 baseline. bf16 therefore does *not* buy batch headroom
here, and any argument for it resting on the memory ceiling is wrong.

So the honest position: CUDA + bf16 + a larger batch takes an ~18-day run to
**~7.4 days** (~$87 at $0.49/h against round 1's $215). Real, banked, and still not
an afternoon — the remaining factor has to come from attention.

## 5. Token count is not the lever the O(N²) argument implies

`TEACHER-OPTIONS.md` makes the strongest technical case for Plan C on its VAE:
Wan2.2-TI2V is `z_dim 48, spatial 16` against Wan2.1's `16, 8`, so the same pixels
become 4× fewer latent positions, and since attention is O(N²) the table there
scores it at **1/256** the attention cost. Adopting it is a teacher swap — 5B
weights, ~44 GB, a new adapter, and `latent_channels 16 → 48` through the spec, the
contract and the browser decode — so the payoff was measured first, using the
teacher already in hand: resolution alone sweeps token count against the Wan2.1 VAE,
and 320×320 at Wan2.2's compression *is* 400 tokens.

Throughput at accum 8 (cuda bf16, 20 steps): 400 → 3.810, 784 → 4.000,
1600 → 3.077 samples/s, and 3136 OOMs. **4× the tokens costs ~1.24×, and 400 tokens
is slower than 784.** The profile at the two ends says why: fwd+bwd goes
146.09 → 637.51 ms/sample for 7.84× the tokens — 4.36×, *sub-linear* — while `optim`
stays flat at 23.3 → 26.2 ms/sample, being O(parameters) and consuming 13.7% of a
400-token step against 3.9% of a 3136-token one.

That first pass understates the case, because token reduction's real benefit is the
batch it buys. Sweeping batch at both geometries under one protocol:

| accum | 400 tokens | 1600 tokens | ratio |
| --- | --- | --- | --- |
| 8 | 2.500 | 2.162 | 1.16× |
| 32 | 5.818 | 4.103 | 1.42× |
| 64 | 7.619 | 4.741 | 1.61× |
| 128 | **8.258** | **5.079** | **1.63×** |

**Best against best, 4× fewer tokens is worth 1.63× — not 16×, and not 256×.** The
O(N²) term is real but is not what the wall clock is made of at this model size: at
1152 hidden width, attention only overtakes the dense path around N ≈ 1152 tokens,
and below that the run is bound by parameter math, the optimizer step, and kernel
launch latency rather than by sequence length.

**Batch is the bigger and cheaper lever.** Raising accum 8 → 128 is worth 2.35× at
1600 tokens and 3.30× at 400, costs nothing but a flag, and needs no teacher. At
production geometry accum 128 beats accum 32 by 24% under matched conditions.

So Plan C should not be undertaken as a *performance* measure. Its licence and
quality arguments stand on their own; its speed argument is 1.63× for a teacher
swap, against 1.24× available immediately from a flag.

**Where the machine is actually idle.** At 99% reported utilisation the card drew
313 W of 575 W. Utilisation counts kernel residency, not work: the trainer is
memory-bandwidth-bound, consistent with `perf/tiled-attention` finding ~14.5 GB of
~16.1 GB per-sample activations in unfused attention. The spare resource is
arithmetic, so capability that is compute-dense and memory-light — wider layers,
per-block adaLN-zero σ conditioning, prompt cross-attention — is close to free here,
while anything that adds tokens or activations is not.

## 6. What not to re-attempt

* **Host-stall removal in the trainer.** Measured at 0.5% of wall clock. Done, and
  the remaining time is arithmetic.
* **Wider attention tiles to get a batch dimension.** The tape, not the transient
  peak, is the limit. Needs a fused softmax with recompute-on-backward.
* **f16 for training.** It hard-fails on real data, and disabling gradient clipping
  does not save it. `train` now refuses it.
* **bf16 for batch headroom.** f32 does not OOM at production geometry; take bf16
  for throughput, which is real and grows with batch.
* **The Wan2.2 spatial-16 VAE as a speed lever.** Measured at 1.63×, not the 16-256×
  the O(N²) argument implies (section 5). Raise `--accum` instead.
* **Batching CFG cond+uncond in the teacher pass.** Measured regression.
* **Micro-optimising the teacher loop for GPU utilisation.** It is at 92% of TGP.

## 7. Still unmeasured

* **A same-box wgpu baseline.** Section 4 compares CUDA against round 1's *recorded*
  2.04 samples/s — same GPU model, same geometry, different machine and cache. Two
  attempts to measure both arms on one box failed the same way: every `nvidia/cuda`
  image sets `NVIDIA_DRIVER_CAPABILITIES=compute,utility`, Vast does not let the
  instance env override it, so the only Vulkan ICD is llvmpipe and wgpu silently
  runs on the CPU (GPU at 0%, 5.7 W). Needs a base image that requests graphics
  capability.
* **Whether bf16 *teacher* targets degrade the student.** (Distinct from trainer
  bf16 in section 4, which is measured and safe.) The 6.9% mean deviation from
  fp32 targets is not a rounding artifact and nothing here shows it is harmless.
  Build two small caches, train a short run on each, compare against the
  trivial-predictor floor in `parity_baseline.py` before committing a large cache.
* **Whether raising `--accum` past 128 keeps paying.** 128 was the largest tried and
  had not clearly plateaued at 1600 tokens.
* **The cache levers at production geometry.** Everything above is at 1024 tokens.
  Gram cost scales with tokens², so `--relation-layers` matters far more at
  2304 tokens than the +0.8% measured here.

## Method notes

Three measurement bugs were found and fixed while producing the table above; they
are recorded because each would have produced a confident wrong answer:

* Noise generation inside the micro-batch loop regenerated all draws and discarded
  all but one, inflating every non-batched config.
* Warm-up landed entirely on whichever config ran first.
* Peak VRAM read from `nvidia-smi` is cumulative allocator reserve, not per-config —
  it made a no-grams run look memory-hungrier than base.

`nvidia-smi` utilisation is not used as a saturation signal anywhere here. It
reports the fraction of time a kernel was resident, not how much of the machine that
kernel used; power against TGP and the throughput-vs-batch knee are used instead.
