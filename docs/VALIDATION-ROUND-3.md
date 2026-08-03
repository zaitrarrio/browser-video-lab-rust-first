# Validation round 3 — the conditioning was a ceiling, not just undertraining

Round 1 ([`VALIDATION-ROUND-1.md`](VALIDATION-ROUND-1.md)) found held-out parity that
cleared the trivial floor but was **flat across σ**, and named the suspect: the
student's only σ input is a single `Linear(1 → width)` added once at the stem, where
every DiT in this family modulates every block. It could not separate that from
undertraining, and left the A/B as its recommendation 3 — *worth an A/B once runs
are cheap*. Runs are now cheap ([`PERF-ROUND-1.md`](PERF-ROUND-1.md)), so this is
that A/B.

**Result: the hypothesis holds.** Per-block adaLN-zero conditioning does not merely
train better — it moves parity in exactly the place and direction round 1 predicted,
and the control, trained identically, does not clear the trivial-predictor floor at
all.

## Setup

Control and treatment differ by one boolean. Same seed (42), same training cache,
same schedule — 200 steps × accum 32 at lr 2e-5, cuda/bf16, 1600 tokens — and the
same binary. `per_block_conditioning` adds `Linear(width → 6·width)` per block,
zero-initialised, so at step 0 both arms are the *same function*.

8 clips at 320×320 from `make_wan_dataset.py`; 6 into the training cache, **2 held
out** and never trained on. Evaluated twice, on two independently built held-out
caches: one at `--shift 3.0` (the inference warp, which concentrates draws at high
σ) and one at `--shift 1.0` (uniform σ, built specifically because the first
evaluation's low-σ bands rested on 6 and 12 shards).

## Parity by σ band

Uniform-σ cache, 128 shards — the authoritative table, since every band is sampled:

| σ band | n | control | per-block | delta |
| --- | --- | --- | --- | --- |
| 0.00–0.25 | 43 | 0.4757 | **0.7198** | +0.2441 |
| 0.25–0.50 | 32 | 0.4248 | **0.7021** | +0.2773 |
| 0.50–0.75 | 24 | 0.3981 | 0.5479 | +0.1498 |
| 0.75–1.01 | 29 | 0.5135 | 0.5772 | +0.0637 |
| **weighted** | 128 | **0.4570** | **0.6508** | +0.1938 |

The shift-3 cache (96 shards) gives the same shape from a different draw:
+0.2446 / +0.2849 / +0.1220 / +0.0740, aggregate 0.4680 → 0.5910. The two low-σ
deltas move by less than 0.01 between the two caches despite the sample count going
from 6 and 12 to 43 and 32, so the effect is not a small-n artifact.

**The delta decreases monotonically with σ.** That is the predicted signature, not a
uniform improvement: at high σ the target is nearly the model's own input and
echoing already scores well, so there is little to gain; at low σ, where all the
structure is decided, the student has to actually know its noise level.

## The floor is what makes this decisive

`parity_baseline.py` on the same 128 held-out shards:

| predictor | cosine |
| --- | --- |
| echo (return the noisy input) | −0.1578 |
| scaled (best per-shard `a·x_s`) | +0.4985 |
| mean (dataset-mean target) | **+0.5085** |

**The control is below the floor.** 0.4570 against 0.5085 — after 6,400 sample-views
it has not learned anything a constant predictor does not already do. On the shift-3
cache its aggregate was 0.4680 against that cache's mean-predictor floor of 0.4680:
equal to four decimals. Trained identically, the treatment reaches 0.6508 and clears
the best floor by +0.14.

That is the part that upgrades this from "an improvement" to "a ceiling". Both arms
are equally undertrained — the comparison is between two equally-undertrained σ
profiles — and only one of them is learning anything at all.

## Cost

Wall clock 1204 s (control) → 1388 s (treatment), **+15%**, for `6·width·width`
extra parameters per block. Consistent with `PERF-ROUND-1.md` §5: the trainer is
memory-bandwidth-bound at 313 W of 575 W, so compute-dense, memory-light capability
is close to free. Final training loss 0.6226 → 0.5417.

## What this does not show

* **Neither arm is trained.** 6,400 sample-views against round 1's ~5,200; both are
  far short of a real schedule. This compares architectures, not models.
* **One seed, two held-out clips.** The σ profile is consistent across two
  independently built eval caches, which is reassuring, but nothing here is a
  multi-seed result.
* **No sampled video.** Parity is agreement with a teacher's velocity field, not
  coherent output. Round 1's decode path exists and was not run here.
* **The floor moves with the σ distribution.** Best floor is +0.4985 (scaled) on the
  uniform cache and +0.4883 on the shift-3 one, so a parity number is only readable
  against the cache it came from.

## What follows

Round 1 ordered its recommendations *spend the compute first, everything else is an
optimization of that*. This result reorders them: a long run on the control
architecture would have spent ~7 GPU-days training something that, at this budget,
does not beat predicting the mean. **The long run should use
`per_block_conditioning: true`.**

Recommendation 4 — prompt conditioning, currently a mean-pooled caption added as one
vector with no cross-attention — is the same class of defect and the same class of
cheap fix, and is now the obvious next A/B.
