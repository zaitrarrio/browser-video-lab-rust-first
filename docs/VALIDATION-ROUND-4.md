# Validation round 4 — teacher parity does not transfer to sampling

The overfit probe from [`OVERFIT-PROBE.md`](OVERFIT-PROBE.md) ran. It answers the
question it was built to answer, and the answer is **no** — but it fails in a
place no previous round could see, and it eliminates most of the explanations.

**Headline:** the student agrees with the teacher's velocity field at **0.8391
cosine against a 0.4634 floor** — nearly three times round 1's margin — and
integrating that same field from noise still produces no spatial structure at all.

## Setup

79M probe student (`probe-79m.json`: width 640, 16 layers, 8 heads, adaLN-zero
per-block conditioning), 4 clips at 320×320 × 13 frames, latent `[1,16,4,40,40]`,
1600 tokens. Teacher cache built in **f32** (not bf16) at `shift 3.0`,
`guidance 5.0`, 1536 shards; eval cache 128 shards at `shift 1.0` from a fresh
seed on the same clips.

**26,513 optimizer steps × accum 8 = 212,104 sample-views**, cuda/bf16, lr 2e-5,
in 2h30m of a rented 5090. That is **31×** round 3's 6,400 views and **62×** round
1's 400 optimizer steps. Undertraining is no longer a live explanation at this
model size.

## Three things that are now verified, not assumed

**1. The decode path is exact.** Decoding a dataset item's own latent through
`decode_latents.py` produces a file **md5-identical** to the reference MP4 written
during the dataset build. Round 1's "smooth colour field" was a real observation
about a real student, not a decode artifact.

**2. The flow convention is correct.** `reconstruct_from_shard.py` takes a cache
shard and rebuilds `x0 = x_s - s·v` from the *teacher's* stored velocity, no
student involved. At **σ = 0.796** — an input that is 80% noise — the recovered
latent decodes to an unmistakable red car on a coastal road, the clip it was drawn
from. The noising parameterisation, the σ↔timestep mapping, the CFG combination
and the VAE decode are all mutually consistent. **The target the student is
trained against is the right target.**

**3. The student learned that target well.**

| | round 1 | round 3 (treatment) | **round 4** |
| --- | --- | --- | --- |
| held-out cosine | +0.584 | +0.6508 | **+0.8391** |
| trivial floor | +0.453 | +0.5085 | **+0.4634** |
| margin over floor | +0.131 | +0.142 | **+0.3757** |
| `pred_norm / teacher_norm` | 0.63–0.67 | — | **0.88–0.90** |

Round 1 diagnosed a 35% velocity-magnitude shortfall and called it "ordinary
regression shrinkage from an undertrained model". That was right: 62× the
optimizer steps takes it from 0.65 to ~0.89.

## And it still does not produce a picture

The sampled latent, conditioned on clip 0's prompt embedding and integrated from
pure noise down the same shifted σ schedule the cache was built with:

```
sample  std=0.8073  mean=+0.0671  min=-2.940  max=+3.109
teacher clips std=1.2222
  cos=+0.1782  clip-00000  A red sports car driving fast along a coastal road
  cos=-0.1640  clip-00001  A large orange bonfire burning at night
  cos=+0.1976  clip-00002  Ocean waves crashing onto a rocky shore
  cos=+0.0746  clip-00003  A yellow hot air balloon rising over green fields
best=clip-00002  spread over clips=+0.3617
```

Decoded, it is **blocky patch-scale texture with no spatial structure** —
distinguishable from round 1's smooth colour field, and no closer to video.

**Prompt conditioning still does not steer the sample.** Conditioned on clip 0, it
lands nearest clip 2. Round 1 found exactly this (conditioned on clip 0, nearest
clip 4) and filed it as recommendation 4: the caption is mean-pooled into a single
vector and added, with no cross-attention. 212k sample-views did not fix it,
which is now evidence that it is architectural rather than undertrained.

## The sampler is not the cause

Four configurations, same weights, same prompt, same seed:

| steps | shift | sample std | cos to clip 0 | best |
| --- | --- | --- | --- | --- |
| 32 | 3.0 | 0.8073 | +0.178 | clip 2 |
| 128 | 3.0 | 0.8228 | +0.174 | clip 2 |
| 32 | 1.0 | 0.8092 | +0.181 | clip 2 |
| 128 | 1.0 | 0.8231 | +0.174 | clip 2 |

**4× the integration steps and a different schedule warp move nothing.** This is
not Euler discretisation error, and it is not the `shift` warp. The student's
field integrates to the same wrong place however carefully it is followed.

## What is left, and the test that separates it

The gap between *0.84 cosine on cache draws* and *no structure when integrated* is
the whole finding. Two explanations survive, and they call for opposite work:

**A — Off-manifold drift (exposure bias).** Every shard in the cache is
`(1-σ)·x0 + σ·ε` for a **real** teacher latent `x0`. That is a thin tube around
the data manifold. Sampling starts at pure noise and follows the student's own
predictions, leaving that tube on the first step and never returning; the cache
carries no supervision there, so 0.84 cosine inside the tube says nothing about
behaviour outside it. Round 1 noticed a version of this — *"queried at points a
sampler passes through in its first step and never again"* — and fixed the σ
**coverage**; covering the σ values is not the same as covering the **states**.

**B — The student never learned global structure.** The output head is per-token
`Linear(width → c·ph·pw)`, so each token independently emits one 2×2 latent patch.
Output that is locally plausible and globally incoherent is exactly what tokens
that have not agreed with each other look like — i.e. attention is not carrying
structure to the output, and the model would be blocky even on an on-manifold
input.

**These are distinguishable with one measurement.** Feed the student an
on-manifold `x_s` straight from the eval cache, form `x0 = x_s - σ·v_student` from
its *own* prediction, and decode — the same operation
`reconstruct_from_shard.py` just ran with the teacher's velocity.

* If that decodes to a recognisable clip → the model is sound and the defect is
  **trajectory drift** (A). The fix is on-policy states in the cache: label the
  student's own sampling trajectory with the teacher, which is a real change to
  `cache_teacher.py` because it puts the student in the loop.
* If it is blocky → the model never learned global structure (B), and the work is
  architectural: prompt cross-attention, depth, capacity.

This needs a small `denoise` subcommand on `video-train` (or an `--emit-pred` flag
on `eval`) to write the student's prediction out. **It is the next thing to
build**, and it is perhaps an hour of work against a question that currently
splits the roadmap in half.

## What this round does not show

* **Only the 79M probe arm ran.** The 383M `validation-320-adaln` arm was not
  trained. A capacity explanation for (B) is therefore not excluded — though the
  probe was sized specifically so that failure to memorise **four** clips it has
  seen 53,000 times each would not be a capacity story.
* **One seed, one prompt sampled.** The four sampler configs share a seed.
* **No held-out clips.** By design: the eval is fresh noise draws on the trained
  clips, so nothing here speaks to generalization.
* **The relation and temporal loss terms were not ablated.** `w_temporal 0.25`
  and `w_feature 0.05` were left at their defaults throughout.

## Cost

About 2h30m of RTX 5090 at $0.335/h for the training run, plus roughly an hour of
measurement and setup, on a box that took **five rentals to find** — two machines
could not reach huggingface.co at all and one sustained 0.45 MB/s. Total spend for
the round is under $5.
