# Validation round 7 — the band is unreachable, and the reason is in the module

Round 6 found the student missing one spatial-frequency band and proposed a loss
fix: that band carries 7–13% of the target's energy, so an unweighted MSE barely
asks for it. This round tested that, twice, and the fix does not work — but the
*shape* of the failure identifies the real defect, which is not the loss at all.

**The student has no positional encoding. It is permutation-equivariant over
tokens, so it cannot place structure its input does not already locate.** That is
proven by a test in `video-student`, not inferred.

## Four arms, one immovable band

All at 13.1k optimizer steps, seed 42, same cache, differing only in the loss.
Cosine against the teacher's velocity by normalised radial spatial frequency,
shard 0 at σ = 0.796:

| arm | 0.00–0.10 | **0.10–0.25** | 0.25–0.50 | 0.50–1.01 |
| --- | --- | --- | --- | --- |
| control | 0.684 | **0.470** | 0.720 | 0.902 |
| low-pass `--w-multiscale 1` | 0.773 | **0.466** | 0.681 | 0.893 |
| band-pass `--w-band 10` | 0.502 | **0.452** | 0.767 | 0.910 |
| band-pass `--w-band 50` | 0.445 | **0.449** | 0.767 | 0.907 |
| **spread** | **0.328** | **0.021** | 0.086 | 0.017 |

**Loss reweighting has enormous leverage on the 0.00–0.10 band — 0.328 of spread —
and none on the target band.** Across a 50× range of weights on a term built
specifically to isolate it, 0.10–0.25 moves 0.021, and *downward*. `--w-band 50`
and `--w-band 10` differ by 0.003: five times the weight, no change. The band is
not under-weighted. It is not reachable.

Two things also worth recording from the arms themselves:

* **The low-pass term moved the wrong band, exactly as its construction implies.**
  A pooled MSE contains everything below its cutoff, and the 0.00–0.10 band carries
  31% of the target's energy against the target band's 12%, so it still dominated
  2.6:1 inside the new term. It improved what was already fit and paid for it in
  the higher bands.
* **The band-pass term did what it was built to do**, just not where it mattered:
  0.25–0.50 rose 0.720 → 0.767 and stayed there. Octave 1 covers roughly
  0.177–0.354, which overlaps that measurement band, so the mechanism works. The
  part of the term aimed at 0.088–0.177 simply produced nothing.

## Why: the module has no positional encoding

`BrowserVideoStudent` is `input, text, time, blocks, norm, output`. `forward` is
patchify → `Linear` → add a broadcast conditioning vector → attention blocks →
`LayerNorm` → `Linear`. Every one of those is applied per token or is attention.
There is no positional embedding anywhere, and no other symmetry-breaking term.

So the network is permutation-equivariant over tokens:

    output_i = f(x_i, {x_1 … x_N})

Two tokens with the same content produce the same output whatever their
positions. `student_is_permutation_equivariant_because_it_has_no_positional_encoding`
pins this: rolling the latent one patch in space rolls the output identically, max
difference < 1e-4.

That single fact predicts every measurement of the last four rounds:

| observation | explanation |
| --- | --- |
| finest band 0.90 | each token's own content — position-independent |
| coarsest band 0.68 | a global mean — position-independent |
| **0.10–0.25 stuck at 0.47** | object-scale arrangement — needs to know *where* |
| immovable under any loss | not under-weighted; not representable |
| fails at high σ, fine at low σ (round 5) | at low σ the input locates the structure; at high σ it is noise and there is nothing to locate |
| right palette, no geometry (rounds 4–5) | the marginal distribution of patches is learnable; their arrangement is not |
| prompt steers colour, not layout (round 6) | conditioning is a single broadcast vector — it cannot say *where* either |

## What this costs, and what it does not

The two loss experiments were not wasted: they are what turns "the band is
under-weighted" from a plausible story into a refuted one. A single arm would have
been ambiguous; four arms spanning a 50× weight range, with one band moving 0.328
and the target moving 0.021, is not.

But the ordering was wrong. Round 6 ended pointing at capacity or conditioning,
and the module should have been read before either loss was written. Checking
whether a transformer has positional embeddings is a two-minute grep, and it would
have come before ~3 GPU-hours and two implementations.

## Next

1. **Add positional encoding to `video-student`** — learned per-token, or
   sinusoidal/RoPE over `(t, h, w)`. `burn-nn` ships `pos_encoding.rs` and
   `rope_encoding.rs`. Every DiT in this family has one. The permutation test is
   written to fail the moment it arrives, and should be deleted then.
2. **Re-run the probe unchanged** — same cache, same floor, same spectrum readout.
   The prediction is specific: the 0.10–0.25 band moves off 0.47. If it does not,
   this diagnosis is wrong and capacity returns as the candidate.
3. **Then the loss question can be re-asked**, if it still matters. `--w-band`
   stays in the tree; it is measured, it works on the bands it can reach, and it
   costs nothing at weight 0.

## What this round does not show

* **One checkpoint per arm, one shard, one σ.** The band figures come from shard 0
  at σ = 0.796; rounds 5–6 saw the same pattern on two other clips, but the arms
  here were not re-scored across clips.
* **Positional encoding is not yet tested.** That the model *cannot* represent
  position is proven; that adding it *fixes* the band is a prediction.
* **13.1k steps per arm**, against the 26.5k reference. The control at both
  budgets gives the same band cosine to three decimals, so this is unlikely to
  matter, but it was not verified per arm.
* **Nothing about generalization** — still four clips, still fresh draws on
  trained clips.
