# The hours-not-days probe — plan

Three validation rounds have ended at a parity number. Parity is agreement with
the teacher's velocity field on shards the teacher itself drew; it is not video.
The one time anything was decoded (round 1) the result was *a smooth colour
field in roughly the right palette with no spatial structure*, and rounds 2 and
3 did not decode at all.

The deliverable — a 383M browser student that generalizes — is priced at
**~7.4 GPU-days** ([`PERF-ROUND-1.md`](PERF-ROUND-1.md) §4) and no amount of
optimization is going to make that an afternoon. So this run does not attempt
it. It answers a strictly smaller question that has never been answered and can
be answered in one session:

> Sample from noise, decode, and look — does anything in this pipeline produce
> a picture?

Runner: [`scripts/overfit-probe.sh`](../scripts/overfit-probe.sh).

## What is made smaller, and what deliberately is not

**Smaller: the task.** 4 clips, and the evaluation is fresh *noise draws* on
those same 4 clips rather than held-out clips. This gives up the generalization
signal on purpose. Memorization is the falsification test the validation plan
already called for: a student that cannot reproduce four clips it has seen
thousands of times has an architecture, loss or sampler problem, and more data
was never the missing ingredient.

**Not smaller: the geometry.** 320×320, 13 frames, `[1,16,4,40,40]`, 1600
tokens — the same as every round and every perf measurement. Two reasons. The
teacher is incoherent below 320×320, so a student scored against a 256×256
reference is scored against smear. And token count is not the lever the O(N²)
argument implies: §5 measured 4× fewer tokens at 1.63×, so shrinking the frame
buys almost nothing and costs the whole comparison.

**Not smaller (in the `full` arm): the architecture.** Round 3 showed per-block
adaLN-zero conditioning is the difference between clearing the trivial floor and
sitting under it. Whatever this probe finds should be about the model that
actually ships.

## The two arms

| arm | spec | params | why run it |
| --- | --- | --- | --- |
| `small` | `rust/config/probe-79m.json` | 79M + 39M adaLN | cheap falsification. ~3–4× the throughput, so ~3–4× the optimizer steps in the same hour. If *this* cannot memorize 4 clips, the defect is not capacity. |
| `full` | `rust/config/validation-320-adaln.json` | 383M + 191M adaLN | the artifact worth having: the deliverable architecture, on the deliverable geometry, with real pixels out. |

Run `small` first. It is the arm that can cheaply say "stop, something upstream
is wrong", and it costs about an hour.

## Budget

RTX 5090, cuda/bf16, 1600 tokens, accum 8. The 383M figure is measured
(`PERF-ROUND-1.md` §4); the 79M figure is an estimate from the FLOP proxies
(`layers·width²` is 0.21×, `heads·layers` is 0.33×) and the script prints the
real one.

| arm | samples/s | 4 h sample-views | optimizer steps | vs round 3 |
| --- | --- | --- | --- | --- |
| `small` | ~12–16 *(est.)* | 170k–230k | 21k–29k | ~30× the views |
| `full` | 4.10 *(measured)* | 59k | 7.4k | 9× the views |

Round 1 trained for 800 optimizer steps; round 3 for 200. Plus ~25 min for the
dataset, the f32 teacher cache and the sampling pass, the whole session is
**~6 GPU-hours, about $5** at $0.78/h.

## Two corrections this plan makes to the standing advice

**Do not raise `--accum` here.** `PERF-ROUND-1.md` §5 is right that batch is the
biggest cheap throughput lever — accum 8 → 128 is worth 2.35× samples/s at this
geometry. But throughput is the wrong metric for a fitting run: at a fixed wall
clock, accum 128 buys 1.6× the samples while dividing the *optimizer step count*
by 16. What fits a model is steps. Accum stays 8, which is also the only setting
at which lr 2e-5 has been shown stable.

**Build the cache in f32, not bf16.** §3 measured the bf16 teacher at +163% and
§7 lists "whether bf16 teacher targets degrade the student" as unmeasured, with
a 6.9% mean deviation from f32 targets. On 1536 shards the f32 path costs about
five extra minutes. For a run whose entire output is a fidelity judgement, that
is the wrong place to accept an unmeasured 7%.

## The gate before any GPU time

`decode_latents.py` has been in the tree since round 1 and has been run
essentially once. Phase `check` decodes the teacher's *own* latent and compares
it against the reference MP4 the dataset build wrote from the same tensor. Same
input, two paths through the denormalize-and-decode code.

If those two files do not match, the decode path is wrong, and every student
result downstream is unreadable — including round 1's colour field, which would
then have been evidence of nothing. It costs a minute. Do not skip it.

## How to read the result

Three readouts, in increasing order of how much they can mislead:

1. **`video-train eval` against `parity_baseline`.** Never the cosine alone —
   the floor moves with the σ distribution of the cache it was computed on.
   Watch `pred_norm/teacher_norm` just as hard: round 1's student predicted
   velocities at 0.63–0.67 of the teacher's norm, and an Euler integrator that
   travels two-thirds of every step cannot arrive whatever its direction
   accuracy. A memorizing student should drive that ratio toward 1.
2. **`compare_samples.py`.** Did the sample land on the clip it was conditioned
   on, or on an average of all four? Round 1, conditioned on clip 0, landed
   nearest clip 4. The four captions have deliberately distinct dominant
   colours so this is visible rather than merely computable.
3. **The MP4.** The only one that answers the question.

## What this cannot show, stated up front

* **Nothing about generalization.** Held-out draws on trained clips is not
  held-out clips. A perfect result here is a licence to spend the 7.4 GPU-days,
  not a substitute for them.
* **One seed, four clips, one prompt sampled.**
* **The `small` arm is not the deliverable.** A win there localizes the problem
  away from capacity; it does not transfer to the 383M student on its own.
* **A negative result is ambiguous in one direction.** Failure to memorize at
  ~25k steps is strong evidence of a defect, but not proof: it is still fewer
  steps than a conventional diffusion schedule. It would tell you where to look,
  not what is broken.
