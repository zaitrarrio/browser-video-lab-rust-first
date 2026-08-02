# Validation round 1 — can a teacher cache actually produce the Rust student?

The question this round exists to answer: **can a framework-neutral teacher cache
train the Burn student to produce coherent video?** Every prior round proved
plumbing — that shapes line up, that chunked training resumes, that a record
round-trips into the browser. None of them could answer this one, and three
structural reasons why are recorded below.

Hardware: one rented RTX 5090 (Vast, `$0.49/h`). Teacher: `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`
(Apache-2.0 — Plan B of `TEACHER-OPTIONS.md`).

## Why no previous cache could have worked

### 1. The cached latents were noise

`make_dataset.py` writes `torch.randn` for `latents` in *both* of its modes, and
`cache_teacher.py` then builds `x_t` from them. With `x₀ ~ N(0, I)`, every cached
input is off the video manifold at every noise level, so the teacher is only ever
queried at points a sampler passes through in its first step and never again. The
entire low-σ end of the trajectory — where all image structure is decided — is
unsupervised. A student that matched such a cache perfectly would still emit
noise when sampled. This is not a quality problem; it is a "the experiment cannot
succeed" problem, and it applies to every cache built so far, including the
Kaggle one.

**Fixed by** `make_wan_dataset.py`: run the Wan pipeline to a finished latent with
`output_type="latent"` and cache *that* as `x₀`. No video corpus is needed — the
teacher supplies its own data — and each clip comes with a reference MP4 to
compare against later.

### 2. The noising did not match the teacher's parameterization

`cache_teacher.py` built `x_t = x₀ + σ·ε`. Wan2.1 is a flow-matching model
trained on `x_σ = (1-σ)·x₀ + σ·ε`, predicting the velocity `ε - x₀`. The old form
never reaches pure noise at σ=1 and never reaches clean data at σ=0, so the
teacher was being asked for predictions at inputs whose noise level did not match
the timestep it was told. Both ends of the trajectory were supervised at the
wrong point.

**Fixed by** `--noising flow` (now the default) plus `--shift`, which draws σ
through the same warp the inference schedule uses (`flow_shift=3.0`), so the
cache covers exactly the σ values a sampler visits.

### 3. Nothing could turn a latent into pixels

`video-web` false-colours three of the sixteen latent channels and calls the
result a frame. That is a debug view — no arrangement of latent channels is an
image — and it meant "does this produce coherent video" was not a question anyone
could answer by looking. There was no VAE decode anywhere in the tree, and no
sampler outside the browser.

**Fixed by** `video-train sample` (integrates the student's velocity field down
the same shifted σ schedule and writes safetensors) and `decode_latents.py`
(undoes the pipeline's latent normalization and decodes with the Wan VAE).

## The shortest decisive test

Not "train the 390M student properly" — that is thousands of GPU-hours and it
cannot fail *informatively*. The cheapest falsifiable test of the architecture is
an **overfit probe**: give the student a handful of clips and enough supervision
to memorize them. If a 383M student cannot reproduce eight clips it has seen
thousands of times, the architecture, the loss, or the sampler convention is
broken, and no amount of data would have helped. If it can, the loop is validated
and what remains is data and compute — a budget question, not a correctness one.

Concretely:

| | |
|---|---|
| clips | 8 captions, distinct dominant colours (`data/prompts-validation.txt`) |
| geometry | 320×320, 13 frames → latent `[1,16,4,40,40]`, 1600 tokens post-patchify |
| cache | 8 clips × 192 (noise, σ) draws = **1536 shards**, 4.3 GB |
| targets | CFG-guided velocity `v_uncond + 5·(v_cond − v_uncond)` |
| relation grams | none — see below |
| student | `rust/config/validation-320.json`, the production 390M shape at this geometry |

**Resolution was chosen empirically, not assumed.** A sweep of the teacher itself
at 256², 320², 384², 480² and 480×832 showed Wan2.1-1.3B is incoherent at 256²
(below its training resolution) and coherent from 320² up. Since the student is
being asked to reproduce the teacher, the teacher's own coherence floor bounds
the test — 256² would have produced a meaningless comparison against a smeared
reference. 320² is the cheapest resolution where the reference is worth matching.

**Relation grams were dropped for this round** (`--relation-layers -1`, new). A
gram is `tokens²` fp16 — 10.6 MB per layer against ~3 MB for everything else in a
shard — so cacheing them costs roughly an order of magnitude in distinct (noise,
σ) draws for the same disk. Draw count, not step count, is what bounds
supervision here (`TEACHER-OPTIONS.md`), and the output term is what decides
whether sampling works at all. The relation term is a separate hypothesis and
deserves its own arm.

## Reading the parity number

`video-train eval` reports cosine and relative L2 against the teacher on held-out
shards (a fresh-seed cache of the same clips the student never trained on). That
number is meaningless without a floor, because in flow matching the target is
correlated with the model's own input:

```
<x_σ, v> = σ·|ε|² − (1−σ)·|x₀|²
```

so a model that learned nothing but "echo your input" already scores well above
zero. Measured on the held-out cache (`parity_baseline.py`):

| trivial predictor | cosine | rel L2 |
|---|---|---|
| echo the noisy input | +0.197 | 1.034 |
| best-scaled echo (least squares) | **+0.431** | 0.879 |
| mean teacher target | +0.278 | 0.960 |

**+0.431 is the floor.** Anything at or below it is consistent with the student
having learned nothing at all.

## Results

### The loop closes

Every stage ran on real data, end to end, for the first time:

```
8 captions ─▶ Wan2.1 pipeline ─▶ real latents [1,16,4,40,40] + reference MP4s
          ─▶ 1536 CFG-guided flow-matching shards (4.3 GB, validate-cache OK)
          ─▶ Burn/WGPU student, 383M params, on the 5090
          ─▶ video-train sample ─▶ latents ─▶ Wan VAE ─▶ MP4
```

No PyTorch in the middle two stages, and no synthetic stand-in anywhere. The
teacher-cache contract survived contact with a real teacher: `validate-cache`
passed on all 1536 shards, and the patchified student's token count (1600) lined
up with the Wan teacher's without adjustment.

### Teacher parity

Held-out shards (fresh-seed draws of the same 8 clips, never trained on), scored
against the trivial floor:

| checkpoint | batch | lr | cosine | rel L2 |
|---|---|---|---|---|
| *trivial floor (best-scaled echo)* | — | — | *+0.453* | *0.840* |
| step 1000 | 1 | 5e-5 | +0.460 | 0.918 |
| step 2000 | 1 | 5e-5 | +0.526 | 0.850 |
| step 2400 | 8 | **1e-4** | **+0.399** | **1.060** |
| step 2000 + 400×8 | 8 | 2e-5 | **+0.584** | **0.804** |

The student clears the floor and improves monotonically — **except** at lr 1e-4,
where it fell *below* the floor while the deepest hidden-state norm inflated from
10.3k to 13.9k. That reproduces, on a real cache, the instability behind the
`step-349` investigation and confirms the lr default of 2e-5 was the right call.
Batching is a real win at a safe lr; batching *plus* a raised lr is not.

### Sampling

The sampler integrates, and produces a latent with plausible statistics rather
than noise or a collapse (std 0.84 against the teacher clips' 1.18). Decoded,
it is a smooth colour field in roughly the right palette with **no spatial
structure** — see `artifacts/validation-round-1/compare-b8.png`, student left,
nearest teacher clip right.

Two measurements explain why, and both point the same way:

- **The student under-predicts velocity magnitude by ~35%**, uniformly across
  σ (`|pred|/|teacher|` = 0.63–0.67 in every band). An Euler integrator that
  travels 65% of the required distance at every step cannot arrive: the sampled
  latent's std/teacher ratio, 0.71, is what that shortfall compounds to. This is
  ordinary regression shrinkage from an undertrained model, not a convention bug.
- **Parity is flat across σ** (0.53 at σ<0.2 rising to 0.60 at σ>0.8). The
  student has learned a σ-averaged approximation of the field rather than a
  σ-specific one. It is not catastrophically worse at low σ, which would have
  indicated broken conditioning; it is uniformly mediocre, which indicates
  undertraining.

Prompt conditioning does not yet steer the sample: conditioned on clip 0, the
result is nearest to clip 4 (+0.32) and only +0.16 to clip 0, though the spread
across clips (0.51) shows the model has learned *some* clip-specific structure.

### Throughput — the number that sizes everything else

**2.04 samples/s** (400 steps × batch 8 in 1567 s), f32, Burn/WGPU, no fused
attention, at 1600 tokens on an RTX 5090. A conventional DiT distillation
schedule — say 100k steps at batch 32, 3.2M sample-views — is therefore **~18
GPU-days on this box**, about $215 at $0.49/h. That is the real cost of the
deliverable, and it is a budget question rather than a correctness one.

The obvious levers, none of them started: bf16 (Burn/WGPU runs f32 today), a
fused attention kernel (the `[1,16,1600,1600]` probability matrix is
materialized), and the Wan2.2 spatial-16 VAE, which `TEACHER-OPTIONS.md` already
identifies as a 4× token reduction and would cut attention 16×.

## Outcome

**The architecture is validated structurally and the loop is validated
functionally end to end. Coherent video was not produced, and this round could
not have produced it.**

What is now known that was not before:

1. A real teacher cache trains the Rust student. The contract, the patchify
   geometry, the relation-free loss, the chunked trainer, the record round-trip,
   the sampler and the decode all work on real Wan2.1 data.
2. The student learns something real — parity clears the trivial floor and
   improves with batching — but at 5,200 sample-views it is far short of what a
   383M diffusion student needs, and the velocity-magnitude shortfall says
   exactly that.
3. Three latent bugs that would have invalidated any amount of training are
   fixed: noise latents, the noising parameterization, and the missing decode.
4. Two real defects were found and fixed in the trainer: no batching at all
   (batch was hardcoded to 1), and — reconfirmed — that lr 1e-4 destabilizes this
   architecture.
5. The cost of the actual deliverable is now measured, not guessed: ~18 GPU-days
   on a 5090 for a conventional schedule, before any of the three speed levers.

### What round 2 should test, in order

1. **Spend the compute.** The single highest-information action is a long run at
   batch ≥ 8 and lr 2e-5 on a larger cache. Everything below is an optimization
   of that, and none of it should be done first.
2. **bf16 + fused attention** before, not after, the long run — a 3–5× throughput
   win changes what is affordable.
3. **Per-block timestep conditioning.** The student's only σ input is
   `Linear(1 → width)` added once at the stem, where every DiT in this family
   uses adaLN-zero modulation in every block. Flat-across-σ parity is consistent
   with that being a ceiling, though this round cannot separate it from
   undertraining. Worth an A/B once (2) makes runs cheap.
4. **Prompt conditioning.** Mean-pooling the caption to one added vector, with no
   cross-attention, is the weakest link after σ. The clip-selection result says
   it is not yet steering anything.

