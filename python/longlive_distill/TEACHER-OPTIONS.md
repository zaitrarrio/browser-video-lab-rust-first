# Choosing a teacher

`ADAPTER.md` defines the teacher *contract*. This document chooses which teacher
satisfies it, and what has to change either way.

Written after establishing, empirically, that the free-tier hardware cannot run
the obvious plan and that the cache format cannot hold enough data to train the
student. Both plans below fix the second problem; they differ on the first.

## What is true regardless of which teacher wins

**The student is already independent.** `video-web::BrowserModel` takes the spec
JSON and `student.bin`, nothing else. The teacher exists only to manufacture
targets, once. This is not in question and neither plan changes it.

**The relation loss cannot run against a real teacher.** This is a blocker, not a
size problem. Every Wan-family DiT patchifies with `patch_size: [1,2,2]`
(verified in each `transformer/config.json`), so a teacher sees 4× fewer tokens
than the student, which flattens `[B,C,T,H,W] → [B, T·H·W, C]` with no patchify
at all:

```
student gram   [1, 6144, 6144]      4·32·48, unpatchified
teacher gram   [1, 1536, 1536]      Wan2.1: 4·16·24 after [1,2,2]
teacher gram   [1,  384,  384]      Wan2.2-TI2V: 4·8·12, VAE is spatial-16
```

`mse(student_gram, teacher_gram)` cannot broadcast those. It fails on the first
real shard. It passes today only because `synth-cache` generates the teacher side
at the student's own token count — the synthetic path structurally cannot surface
this, exactly as with the latent-channel mismatch below.

Any fix must resample both sides onto a shared token set. Subsampling *K* shared
positions solves the size problem and this one together, which is why it is first
on the list.

**Cache sizes, with the real geometry.** Each `teacher_relation.{N}` gram is
`tokens²` fp16, and `cache_teacher.py` writes `hidden_relation_layers =
list(range(n_layers))` — every layer, 30 for both Wan2.1-1.3B and Wan2.2-TI2V-5B:

| teacher | tokens | gram/layer | shard (30 layers) | shards in ~30 GB |
|---|---|---|---|---|
| Wan2.1-1.3B | 1536 | 4.7 MB | ~141 MB | ~200 |
| Wan2.2-TI2V-5B | 384 | 0.3 MB | ~8.8 MB | ~3,400 |

`video-train/src/lib.rs:143` loads *every shard into RAM eagerly*, so that last
column is a hard ceiling. 200 samples will not train a 383M student.

Three changes, needed by every plan:

1. **Subsample a shared token set before the gram.** Pick *K* (say 512) positions,
   map them across the student and teacher grids, store the indices alongside.
   This is what makes the loss run at all; the size win is a bonus.
2. **Cache 4–6 layers, not all 30.** `linspace_idx` already only consumes
   `min(teacher_layers, student_layers)` of what is written.
3. **Multiple `(noise, timestep)` draws per clip.** This is the one that matters
   most, see below.

Combined, a shard lands near ~1 MB whichever teacher wins. A streaming loader
replacing the eager RAM load at `lib.rs:143` becomes necessary once shard counts
pass a few thousand.

**Timestep draws, not clips, are the unit of supervision.** `cache_teacher.py`
freezes `noisy_latents` and `timestep` into each shard, and
`video-train/src/lib.rs:194` cycles shards by `(step-1) % len`. So step 200,001
shows the model a byte-identical input it has already seen hundreds of times.
The PyTorch reference loop (`train.py:36`) draws fresh noise and a fresh timestep
every step; the cached path cannot. **Effective supervision = shard count**, no
matter how many steps are scheduled. Emitting 8–32 shards per clip at different
`(noise, t)` pairs is what converts a few hundred prompts into a real dataset,
and it costs teacher forward passes rather than video data.

**`student_prompt_embeds` is never written.** `cache_teacher.py` emits only the
teacher's 4096-wide `prompt_embeds`, but `browser-390m-umt5.json` expects
`text_width: 512`. Only the Rust synthetic generator emits both. As it stands a
real cache can only train the 4096-wide student, not the production umt5 one.
Whichever teacher is chosen must also encode prompts with umt5-small.

**The teacher pass is inference-only.** No backprop, no optimizer, a few thousand
forward passes. It does not need an H100; it needs patience. This is why a CPU
run is viable and why the free-tier GPU's `sm_60` incompatibility — fatal for
PyTorch, irrelevant to Burn — need not block anything.

---

## Plan A — LongLive-1.3B as teacher

Keeps the model the project was designed around.

**Feasibility, verified against the `v1.0` branch** (which matches the 1.3B
checkpoint; `main` has moved to 2.0/Wan2.2):

- `flash_attn` and `flash_attn_interface` are both imported under
  `try/except ModuleNotFoundError`, and neither appears in `requirements.txt`.
  `attention()` falls back to `scaled_dot_product_attention` — pure PyTorch.
- No Triton import in `pipeline/causal_inference.py`, `utils/wan_wrapper.py` or
  `wan/modules/causal_model.py`.
- Exactly one hardcoded device call: `wan_wrapper.py:33`,
  `self.text_encoder = self.text_encoder.cuda()`. One-line patch, and the text
  encoder is separated out anyway.

**Weights** — `Efficient-Large-Model/LongLive-1.3B` (`models/longlive_base.pt`
5.68 GB, `models/lora.pt` 2.80 GB) plus `Wan-AI/Wan2.1-T2V-1.3B` as its base
(VAE 0.51 GB, DiT 5.68 GB, umT5-XXL encoder 11.36 GB). Both ungated. Paths are
hardcoded to `wan_models/Wan2.1-T2V-1.3B/`, including a `google/umt5-xxl/`
tokenizer subdirectory. Note Kaggle gives `/kaggle/working` only 21 GB but
`/tmp` 1.1 TB — downloads belong in `/tmp`.

**Licence: CC-BY-NC-SA-4.0.** Non-commercial and share-alike, and that plausibly
follows the student distilled from it. This is a project-level constraint, not a
technical one, and it is the strongest argument against this plan.

### Work

1. `python/longlive_teacher_adapter.py` — build `CausalInferencePipeline`, load
   `state_dict["generator_ema" | "generator"]` with FSDP prefixes stripped, and
   wrap `WanDiffusionWrapper`. **Register forward hooks on the DiT blocks**: the
   wrapper returns `(flow_pred, pred_x0)` and exposes no hidden states, so the
   `hidden_states` half of the `ADAPTER.md` contract must be captured, not read.
2. A prompt→latents generator using
   `pipeline.inference(noise=…, text_prompts=…, return_latents=True)`, writing
   `.pt` files of `latents` + `prompt_embeds`. No video corpus required.
3. A separate one-time prompt-encoding pass: umT5-XXL for the teacher side,
   umt5-small for `student_prompt_embeds`. Run on CPU, cache, never load again.
4. The three cache-format changes above, in `cache_teacher.py` and mirrored in
   `video-contract` / `video-train`.
5. A Kaggle **CPU** kernel to run it, plus a dataset to hold the weights so the
   26 GB is fetched once.

### Risks

- `flow_pred` is a flow-matching velocity, not an ε-prediction. It will be stored
  in a field named `teacher_noise_pred`. Harmless for distillation — the student
  matches whatever is cached — but the name will lie, and `losses.py` should be
  read with that in mind.
- CPU throughput for a 1.3B DiT at 6144 tokens is unmeasured. Must be timed on
  one sample before committing to thousands.
- Most integration work of the two plans: hooks, config plumbing, checkpoint
  key-cleaning, a research-grade repo as a dependency.

---

## Plan B — Wan2.1-T2V-1.3B as teacher, no LongLive

Drops LongLive entirely and distils from the model it is built on.

**Licence: Apache-2.0.** No downstream restriction on the student. This is the
central reason to prefer it.

**Already partially present.** `python/smash_wan21.py` and
`python/benchmark_wan21.py` target `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`, so the
project already carries Wan2.1 plumbing and the `diffusers` dependency.

**Smaller and better supported.** 17.55 GB rather than 26 GB, no research repo to
vendor, `diffusers`-native loading, and hidden states reachable through standard
hooks or `output_hidden_states` rather than bespoke wrappers.

### Work

Same shape as Plan A, minus the LongLive integration:

1. `python/wan21_teacher_adapter.py` — load the Diffusers transformer, hook the
   blocks for `hidden_states`, return the `ADAPTER.md` dict.
2. Prompt→latents via the Diffusers pipeline, or simply sample latents and let
   the timestep sweep provide coverage.
3. The same one-time prompt-encoding pass (both widths).
4. The same three cache-format changes.
5. The same Kaggle CPU kernel, against a smaller download.

### What is lost

LongLive's contribution is *causal, streaming, long-video* generation — frame-level
autoregression with KV-recache. Wan2.1 is the bidirectional short-clip base. The
student in `video-student/src/lib.rs` is a chunk-bidirectional mixer, and the
browser runtime already re-synthesizes its own conditioning, so the causal signal
is arguably not being exploited today. But distilling from Wan2.1 means the
student is taught a short-clip denoiser, and any long-horizon behaviour would
have to come from architecture rather than from the teacher.

---

## Plan C — LongLive-2.0-5B as teacher

The newest and strongest teacher, and the worst fit for free hardware.

**Licence: NVIDIA Open Model License Agreement** (the card reports `other`). This
is the one place C beats A outright — it is not a non-commercial licence, where
1.0's CC-BY-NC-SA is. If the licence is what rules out Plan A, it does not rule
out Plan C. Read the agreement before relying on that.

**Weights** — `Efficient-Large-Model/LongLive-2.0-5B`, `model_bf16.pt` 10.00 GB,
ungated. Its base is **Wan2.2-TI2V-5B** (Apache-2.0), a further 34.18 GB: VAE
2.82 GB, DiT ~19.8 GB across three shards, umT5-XXL 11.36 GB. **~44 GB total** —
roughly 1.7× Plan A and 2.5× Plan B. Fetch into `/tmp`, cache as a dataset.

**flash-attn is still optional.** `wan_5b/modules/attention.py` guards FA2, FA3
*and* FA4 behind `try/except`, and `attention()` falls back to
`scaled_dot_product_attention` exactly as `v1.0` does. The NVFP4 configs target
Blackwell and the FA3 path targets Hopper; neither is reachable from a P100, and
neither is required. The documented environment is PyTorch 2.8.0 + CUDA 12.8
with `torchao==0.13.0`.

**The genuine advantage: `sampling_steps: 4`.** 2.0 is DMD-distilled to four
denoising steps. That does not change cache building — one teacher forward pass
per shard either way — but it makes the *prompt→latents* stage roughly an order
of magnitude cheaper than a ~50-step sampler. On CPU, where that stage dominates,
this is the difference between plausible and not.

**The problem: 5B on CPU.** BF16 weights alone are 10 GB against Kaggle's ~30 GB,
before activations, and a 5B forward pass is ~4× a 1.3B one. Plan A's CPU
throughput is already unmeasured; C multiplies the unknown. Realistically C wants
a rented GPU — where its 4-step sampling and stronger quality make it the obvious
choice, and where the ~44 GB download is a non-issue.

**The strongest technical argument for C is its VAE.** Wan2.2-TI2V-5B is
`z_dim: 48, scale_factor_spatial: 16` against Wan2.1's `z_dim: 16, spatial 8`.
Twice the spatial compression per axis is **4× fewer latent positions** for the
same pixels — a 16×24 latent at 256×384 rather than 32×48. Through the DiT's
`[1,2,2]` patchify that is 384 tokens against Wan2.1's 1536.

For a browser student that is decisive, because attention is O(N²):

| | latent grid | tokens | attention cost |
|---|---|---|---|
| student today | 4×32×48 | 6144 | 1× (baseline) |
| + patchify | 4×16×24 | 1536 | **1/16** |
| + patchify, Wan2.2 VAE | 4×8×12 | 384 | **1/256** |

It also collapses the cache problem: shards drop to ~8.8 MB, roughly 3,400 in
RAM rather than ~200.

The cost is that the student spec moves furthest from what is written today —
48 latent channels on a 16×24 grid. But the spec is four numbers in a JSON file,
and the geometry is the thing that actually decides whether this runs in a
browser.

### Work

As Plan A, against the `main` branch and `wan_5b/` rather than `v1.0` and `wan/`,
plus reconciling the Wan2.2 latent geometry with the student spec.

---

## A shared problem all three plans must resolve

`rust/config/browser-390m-umt5.json` declares `latent_channels: 4`, and the
student's input projection is `Linear(4 → 1152)`. Verified against the published
configs, no teacher produces 4-channel latents:

| VAE | `z_dim` | spatial | temporal |
|---|---|---|---|
| Wan2.1-T2V-1.3B | **16** | 8 | 4 |
| Wan2.2-T2V-A14B | **16** | 8 | 4 |
| Wan2.2-TI2V-5B | **48** | **16** | 4 |

So either the student spec moves to the teacher's channel count, or the cache
carries a projection down to 4. This is invisible today because `synth-cache`
generates whatever the spec asks for — the synthetic path can never surface it.
It will surface on the first real shard, and it is not teacher-specific.

**Related, and worth more than any teacher decision:** the student runs 6144
tokens where its own teacher runs 1536, because it never patchifies. Attention is
O(N²), so it is doing **16× the attention work of the model it is imitating**.
Adopting `patch_size [1,2,2]` cuts tokens 4×, attention 16×, and makes the gram
shapes line up with a Wan teacher for free. See `rust/DELIVERY.md`.

---

## Recommendation

It splits on one question: **is there budget for a few hours of rented GPU?**

| | A · LongLive-1.3B | B · Wan2.1-1.3B | C · LongLive-2.0-5B |
|---|---|---|---|
| Licence | CC-BY-NC-SA | **Apache-2.0** | NVIDIA Open Model |
| Commercial use | no | yes | yes |
| Download | 26 GB | **17.5 GB** | 44 GB |
| Sampling steps | ~50 | ~50 | **4** |
| Free-tier CPU | plausible, unmeasured | **best odds** | unlikely |
| VAE spatial compression | 8 | 8 | **16** |
| Teacher tokens @256×384 | 1536 | 1536 | **384** |
| Shard size (30 layers) | ~141 MB | ~141 MB | **~8.8 MB** |
| Teaches | causal long-video | short-clip denoising | causal long-video, strongest |

**No GPU budget → Plan B.** Smallest download, permissive licence, `diffusers`
rather than a research repo, and Wan2.1 plumbing already in the tree. You give up
the causal long-video signal — which the current chunk-bidirectional student may
not be exploiting anyway.

**Some GPU budget → Plan C.** It dominates A on every axis that matters: a
licence permitting commercial use where A's forbids it, a stronger teacher,
4-step sampling that makes latent generation an order of magnitude cheaper, and —
the argument that emerged last and matters most — a spatial-16 VAE that makes the
browser student **256× cheaper in attention** and shrinks shards 16×. Its only
real cost is hardware, and hardware is exactly what the budget buys.

Note the geometry argument is partly separable from the teacher: Wan2.2's VAE
could in principle encode latents for a student distilled from another teacher.
That is unexplored, and would want checking before it is relied on.

**Plan A is hard to recommend.** C supersedes it on licence and quality; B beats
it on cost and simplicity. It wins only if 1.0 specifically is the research
subject.

Either way the cache-format work is identical and is the real prerequisite. It
can be built and tested against the existing synthetic generator before any
teacher exists — and should be, because it is the only part guaranteed not to be
wasted whichever teacher wins.

## Verification

Order matters — each step is checkable without the next one existing.

1. **Cache format, no teacher needed.** Extend `synth-cache` to emit
   token-subsampled grams and multiple `(noise, t)` draws per clip. Confirm
   `video-train` trains against it, that `resumed_chunks_match_a_single_run`
   still passes, and measure real RAM against shard count.
2. **Teacher smoke, one sample.** Run the chosen adapter on a single prompt on
   CPU and time it. Assert the output dict matches `ADAPTER.md` and that shapes
   line up with `REQUIRED_SAMPLE_TENSORS`. This is the go/no-go for CPU
   throughput.
3. **End-to-end, small.** ~32 prompts × 8 timesteps = 256 shards. Push to Kaggle
   as `…-teacher-cache`, run a 1,000-step chunk, confirm a cache HIT and a
   falling loss.
4. **Compare against synthetic.** Same step count on the synthetic cache. If the
   real-teacher loss curve is not clearly better, something is wrong upstream.
5. **Browser.** `task rust:weights` after `task rust:wasm` (that order — the wasm
   step drops `browser-demo.json` in and will otherwise overwrite the trained
   spec), then confirm the demo reports `trained weights` rather than
   `random init`.

Note for step 5: the demo cannot currently be prompted. `video-web::generate`
synthesizes prompt embeddings from an LCG and `src/runtime/rust-video.ts` ignores
the prompt string. Wiring a real umt5-small encoder into the browser is separate
work, and no amount of teacher quality substitutes for it.

---

## Plan B — implemented (Phase 2)

The cache-format work and the Plan B teacher path are now in the tree, CPU-only:

- `python/wan21_teacher_adapter.py` — `build_teacher({...})` loads the real
  `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` transformer and hooks its blocks for
  `hidden_states` (post-patchify tokens, so the grams line up with the patchified
  Burn student). `build_teacher({"toy": true})` is a tiny CPU stand-in that
  patchifies with the same geometry, exercising the whole path with no download.
- `longlive_distill/make_dataset.py` — builds the `.pt` items (latents,
  `prompt_embeds`, `student_prompt_embeds`). `--synthetic` needs no downloads;
  the default encodes a captions file with umT5-XXL (teacher, 4096) and
  umt5-small (student, 512).
- `longlive_distill/cache_teacher.py` — now adapter-driven and device-flexible,
  with `--draws-per-clip`, `--relation-layers` (cap), and `student_prompt_embeds`
  passthrough: the three TEACHER-OPTIONS cache-format changes.

**Prove it end to end on CPU (no GPU, no weights):**

```bash
task teacher:cache:smoke   # synthetic dataset -> toy teacher cache -> Rust validate-cache
task teacher:test          # Python contract tests
```

The produced cache trains the Burn student directly:

```bash
cargo run --manifest-path rust/Cargo.toml -p video-train -- \
  train --spec rust/config/browser-smoke.json --cache artifacts/wan-cache \
  --output artifacts/wan-train --backend ndarray --steps 12 --lr 0.01
```

**Real run (rented CPU / Kaggle CPU kernel).** Install `requirements-wan.txt`
(`diffusers`, `transformers`), then:

```bash
# 1. encode captions -> .pt items (both encoder widths)
python -m longlive_distill.make_dataset --captions prompts.txt \
  --output data/wan-latents --latent-shape 16 4 32 48
# 2. cache the frozen Wan2.1 teacher on CPU (8 draws/clip, 6 relation layers)
task teacher:cache:wan21 DATASET=data/wan-latents
# 3. train the Rust student on data/teacher-cache (spec: browser-390m-umt5.json)
```

**Confirm before the real run:** time one teacher forward pass on a single sample
(CPU throughput for a 1.3B DiT is the one unmeasured risk), and confirm the
`diffusers` forward kwargs the adapter assumes (annotated in
`wan21_teacher_adapter.py`) against the pinned release. Token subsampling (a size
optimization) stays deferred: patchify already aligns the grams.
