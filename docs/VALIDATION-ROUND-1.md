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

_(filled in below once the run completed — see "Outcome".)_
