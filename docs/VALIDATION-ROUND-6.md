# Validation round 6 — the student is missing one spatial frequency band

Round 5 localised the failure to high σ and left capacity and conditioning as the
candidates, each costing hours of GPU to test. This round tests neither. Every
result below comes from analysing the *targets* and one already-trained
checkpoint — about twenty minutes of GPU in total, most of it VAE decodes — and
the outcome is more specific than either candidate.

**The student's velocity is missing the 0.10–0.25 spatial-frequency band: cosine
≈ 0.48 and half the magnitude, consistently across clips. That band holds only
7–13% of the target's energy, so an unweighted MSE has almost no incentive to fit
it — and it is the band that carries object-scale structure.**

## Four cheap tests, and what each one killed

### 1. Is the loss neglecting high σ? No.

`analyze_targets.py`, no GPU. On the shift-3.0 training cache:

| σ band | share of gradient budget | echo residual |
| --- | --- | --- |
| 0.00–0.20 | 5.2% | 0.624 |
| 0.40–0.60 | 12.3% | 0.971 |
| 0.80–1.00 | **53.6%** | 0.835 |

σ ≥ 0.5 holds **80.5%** of the total sum-of-squares. The high-σ target is *not*
mostly recoverable by scaling the input — 83.5% of its norm is not. So the
trainer is neither ignoring high σ nor being fed a trivial target there.
**Hypothesis refuted.**

### 2. Is prompt conditioning the bottleneck? No.

`video-train denoise --prompt` holds `x_σ` fixed and swaps only the prompt — one
forward pass. On shard 0 (clip 0, σ = 0.796):

| prompt | velocity cosine | x0 cosine |
| --- | --- | --- |
| **clip 0 (correct)** | **0.755** | **0.645** |
| clip 1 | 0.446 | 0.157 |
| clip 2 | 0.675 | 0.505 |
| clip 3 | 0.465 | 0.248 |

The prompt steers strongly and the correct one wins. Round 5 inferred conditioning
was carrying identity from clip-appropriate *colour*, but those were different
shards, so the colour could have come from `x_σ`; this holds the input fixed and
settles it. **The model knows which clip and still cannot render it.**

### 3. Is the target even learnable by memorisation? Mostly.

For an overfit probe the objective has a closed form: eliminating ε from
`x_σ = (1-σ)x0 + σε` and `v = ε - x0` gives `v = (x_σ - x0)/σ`. A student that
memorised four latents could compute the target exactly. `closed_form_target.py`
measures how far the stored targets sit from it:

| cache | cos(stored, closed form) |
| --- | --- |
| guidance 5.0 (the training cache) | 0.861 |
| guidance 0 | 0.898 |

So a perfect memoriser tops out near **0.90**, not 1.0 — about 10% of the target
is teacher-intrinsic (the dataset's `x0` is itself a 30-step sampler output, so
the teacher's velocity at a re-noised point is not exactly the closed form) and
roughly 4% more is CFG. The student sits at 0.755 against that ~0.90 ceiling.

### 4. Is a cosine of 0.755 simply not enough? No — the *kind* of error matters.

Degrade the **teacher's own** velocity with random orthogonal noise until it hits
the student's exact cosine, then reconstruct and decode:

| velocity | cosine | reconstruction |
| --- | --- | --- |
| teacher + random error | 0.755 | **car visible**, noisy but unmistakable |
| teacher + random error | 0.920 | car sharp |
| **the actual student** | 0.755 | flat blocky grey-green, nothing |

**The student is worse than random error of the same magnitude.** Its error is
therefore structured, not isotropic — which is what sends this round looking at
where in the signal it lives. (Rescaling the student's velocity by 1.36× to match
the teacher's norm does not help: the structure is not there to amplify.)

## Where the error lives

`scripts` spectrum decomposition, radial spatial frequency normalised to [0,1],
three clips at their highest available σ:

| band | clip 0, σ 0.796 | clip 1, σ 0.819 | clip 3, σ 0.882 | teacher energy |
| --- | --- | --- | --- | --- |
| 0.00–0.10 | 0.700 | 0.876 | 0.891 | 31–56% |
| **0.10–0.25** | **0.470** | **0.482** | **0.497** | **8–13%** |
| 0.25–0.50 | 0.718 | 0.625 | 0.796 | 14–21% |
| 0.50–1.01 | 0.902 | 0.882 | 0.941 | 23–36% |

Magnitude ratio `|v_student| / |v_teacher|` in that same band: **0.533, 0.495,
0.523**.

Three different clips and three different σ produce cosines of 0.470, 0.482,
0.497 and magnitudes of 0.53, 0.50, 0.52. That is not a fluke; it is a hole in
the model at a specific scale.

**The student fits the noise and misses the structure.** It reaches 0.88–0.94
cosine in the highest band — which at σ ≈ 0.8 is essentially ε, the easiest part
of the target and 23–36% of its energy — and collapses to ~0.48 at half magnitude
in the band an object occupies. In a 40×40 latent, radial frequency 0.10–0.25 is
a wavelength of roughly 5 to 14 cells: features spanning a seventh to a third of
the frame. A car. A balloon. A bonfire.

This explains every observation on the table: the correct palette (the DC and
lowest band are partly fit), the absent geometry (the object band is not), the
blockiness (per-token output with no agreement at object scale), and why uniform
rescaling cannot help (it would amplify the bands that are already right).

## What follows — and it is a loss change, not an architecture change

The object band carries **7–13% of the target's energy**. Unweighted MSE
therefore spends roughly nine tenths of its gradient on bands the student already
fits to 0.88–0.94. The model is not failing to optimise the objective; it is
optimising an objective that barely mentions the thing we care about.

1. **A frequency-weighted or multi-scale loss** is the direct fix, and it is a
   change to `video-train`'s loss rather than to `video-student`'s architecture —
   cheap to write and cheap to A/B on the cache and floor that already exist. My
   first hypothesis, that the loss mis-spends its budget, was refuted in its
   original form (§1) and returns here in a corrected one: not *which σ*, but
   *which spatial frequency*.
2. **Then re-test capacity.** If a reweighted loss closes the 0.10–0.25 band and
   sampling still fails, the 383M arm becomes the question — 6.7 h, ~$2.
3. **The performance work stays behind both.**
   [`PERF-ROUND-2.md`](PERF-ROUND-2.md) has 1.63× waiting behind a
   fused-attention backward, and it is worth nothing until the pipeline draws.

## What this round does not show

* **One checkpoint, three shards, high σ only.** The band structure has not been
  swept across σ, and every number comes from the single 79M probe run.
* **The proposed fix is untested.** That a band is under-weighted in the loss does
  not prove that up-weighting it will be learnable, or that it will not cost the
  bands that currently work.
* **Nothing here is about generalization** — still four clips, still fresh draws
  on trained clips.
* **The spectrum tool is a scratch script**, not committed with the same care as
  `analyze_targets.py` and `closed_form_target.py`.
