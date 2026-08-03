# Validation round 5 — the student fails at high σ, on inputs the cache covers

[`VALIDATION-ROUND-4.md`](VALIDATION-ROUND-4.md) ended with two surviving
explanations for a student at 0.84 teacher-parity whose samples have no spatial
structure, and named the one measurement that separates them:

* **A — off-manifold drift.** The cache only contains `(1-σ)·x0 + σ·ε` for real
  `x0`; sampling leaves that tube immediately, so parity inside it says nothing.
* **B — no global structure.** The model would be incoherent even on a perfect
  input.

`video-train denoise` is that measurement. It runs the student on a shard's
stored `x_σ` — on-manifold by construction — and inverts the same identity the
teacher control uses, `x0 = x_σ - σ·v`, so the two differ only in whose velocity
produced them.

**Result: B, and sharply localised.** The student fails at high σ on exactly the
points the cache covers. Drift is not the primary defect.

## Two σ values that look decisive and are not

This is most of the work, and getting it wrong would have produced a confident
wrong answer in either direction.

**Low σ proves nothing.** At σ = 0.315 the student's reconstruction decodes to an
unmistakable rocky shore, x0 cosine 0.972 against the teacher. But `x0 = x_σ −
0.315·v` is dominated by `x_σ`, which already carries 68% of the clean signal —
and decoding the **raw noisy input** at that σ, with no model involved at all,
produces the same recognisable shore. The student is being credited for its
input.

**σ ≈ 1 proves nothing either.** At σ = 0.998 the student's reconstruction is
flat teal with no structure — and so is the **teacher's**, flat green. One Euler
step from pure noise cannot produce an image for anyone; that is what the other
thirty-one steps are for.

The informative band is where the teacher succeeds in one step and the student
can be asked to match it.

## The measurement

Eval cache (fresh seed, never trained on), one high-σ shard per clip, plus the
two traps for contrast:

| shard | clip | σ | v cosine | x0 cosine | x0 std (student / teacher) | teacher decodes to | student decodes to |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 000000 | 0 | 0.796 | 0.755 | 0.645 | 0.537 / 0.938 | **car on a coastal road, sharp** | flat blocky grey-green |
| 000033 | 1 | 0.819 | 0.792 | 0.786 | 1.009 / 1.321 | **bonfire, flames resolved** | faint centred orange glow |
| 000097 | 3 | 0.882 | 0.867 | 0.842 | 1.182 / 1.386 | balloon over fields | uniform green |
| 000064 | 2 | 0.998 | 0.937 | 0.895 | 1.030 / 1.230 | *flat green — fails too* | flat teal |
| 000080 | 2 | 0.315 | 0.828 | 0.972 | 0.875 / 0.910 | shore | shore — *but so does raw `x_σ`* |

**At σ ≈ 0.8 the teacher recovers the clip in a single step and the student does
not.** Same input, same identity, same decode. That is a student defect on an
on-manifold point, which is what rules out drift as the explanation for round 4.

**What the student does get is colour, not structure.** The bonfire shard returns
a dark frame with a centred orange glow; the fields shard returns uniform green;
the car-on-road shard returns flat grey-green. The palette is clip-appropriate
every time and the geometry is absent every time. So the prompt is carrying
*some* identity — enough to pick a colour distribution — and none of the
structure that would make it a picture.

**The magnitude shortfall is specifically a reconstruction shortfall.** `x0`
standard deviation runs 0.57–0.85 of the teacher's across the high-σ rows, worst
at the shard where the failure is most complete.

**Parity is worst in the band that matters.** Velocity cosine is 0.937 at
σ = 0.998 and 0.867 at σ = 0.046, but dips to 0.755 at σ = 0.796. Both ends of
the schedule are easy for different reasons — near σ = 1 the target is nearly the
input, near σ = 0 there is little left to predict — and the student is weakest in
between, which is exactly where a sampler decides global layout. Five points is
not a curve, but it is consistent across four clips.

## What this eliminates

Together with round 4, the following are now measured out rather than argued
about:

* **The decode** — md5-identical through two paths.
* **The flow convention** — the teacher's own velocity reconstructs the clip.
* **The sampler** — 4× the steps and a different warp move nothing.
* **Undertraining, at this model size** — 212,104 sample-views on four clips.
* **Off-manifold drift as the primary cause** — the failure reproduces on
  on-manifold inputs.

## What follows

The defect is the student's ability to synthesise global structure at high σ, and
the two candidates are capacity and conditioning.

1. **Prompt cross-attention** — round 1's recommendation 4, never tested. The
   mechanism now has direct support: at σ ≈ 0.8 the input carries almost no clip
   identity, so *the prompt is the only thing that can say which clip to build* —
   and the prompt is a mean-pooled caption added as one vector. The observed
   behaviour, right palette and no geometry, is what a conditioning path with
   enough bandwidth for a colour and not for a layout would produce.
2. **The 383M arm** — `validation-320-adaln` has still never been trained. The
   probe was deliberately sized so failure would not be a capacity story, but
   79M is 79M and this does not exclude it.
3. **Then, and only then, the performance work.** [`PERF-ROUND-2.md`](PERF-ROUND-2.md)
   has 1.63× waiting behind a fused-attention backward in `burn-autodiff`. None
   of it matters until the pipeline draws.

## What this round does not show

* **One checkpoint, one seed, five shards.** Four clips, but a single training
  run.
* **σ was not swept systematically.** The five points come from whatever draws
  the eval cache happened to contain per clip; a real sweep would build shards at
  chosen σ.
* **Nothing about the 383M architecture**, or about generalization — the eval
  cache is fresh draws on trained clips by design.
* **The teacher's one-step reconstruction is itself CFG-combined** and therefore
  approximate (`reconstruct_from_shard.py` says so); it is a strong control, not
  ground truth.
