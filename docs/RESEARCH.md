# Framework-Neutral Distillation for In-Browser Generative Video — Research Notes

*Internal research document. Every quantitative claim is tied to a file, a config, or
a published model card, and marked as a measurement, a design target, or an estimate.
Where the system has not yet produced a validated result, it says so.*

---

## 1. Problem

Modern text-to-video diffusion models are large and runtime-heavy. `Wan2.1-T2V-1.3B`
is ~17.5 GB of weights (DiT + VAE + an 11.4 GB umT5-XXL text encoder) and expects a
CUDA/PyTorch stack; `LongLive-1.3B` adds a causal-streaming wrapper on top. None of
this runs in a web browser tab: there is no PyTorch, GPU memory is a fraction of
installed VRAM, and the largest text encoder alone exceeds a reasonable download.

The goal of this project is narrow and concrete: **produce a single generative-video
model that runs entirely inside a browser tab, driven by a typed prompt, with no
server inference and no ONNX Runtime** — and to do so on free-tier hardware, with a
reproducible pipeline, from a frozen open teacher.

The interesting part is not the browser runtime (that is engineering). It is the
*training architecture* that makes a browser-native student trainable at all when the
teacher and the student share neither a framework, a width, a text encoder, nor an
inference regime.

## 2. Core idea: the teacher is a one-time target factory

The organizing decision is that **the teacher and the student are decoupled by a
framework-neutral cache**, not by a live training loop.

- The teacher (PyTorch/CUDA, `Wan2.1-1.3B`) runs **once**, in inference only, over a
  set of prompts × sampled `(noise, timestep)` draws. For each draw it emits a
  *supervision shard*: the noisy latent, the timestep, the teacher's denoising
  prediction, both text-encoder embeddings, and a set of hidden-state **relation
  matrices** (see §4). Shards are plain `safetensors`.
- The student (Rust / [Burn](https://burn.dev) / WGPU) trains against those shards
  with **no PyTorch in the loop, ever**. Once the cache exists, all student
  development, training, quantization, and deployment is Rust-only.

`python/longlive_distill/TEACHER-OPTIONS.md` states the invariant plainly: *"The
student is already independent. `video-web::BrowserModel` takes the spec JSON and
`student.bin`, nothing else. The teacher exists only to manufacture targets, once."*

This is what turns an intractable cross-stack training problem into two independent,
cheap problems: an **inference-only** teacher pass (patience, not an H100 — it can run
on CPU) and a **PyTorch-free** student loop (a compiled Rust binary that runs on the
free-tier GPU whose old `sm_60` CUDA is fatal to modern PyTorch but irrelevant to
Burn).

## 3. The student

A compact causal-latent video denoiser, `BrowserVideoStudent` (`rust/crates/video-student`):

| Property | Value | Source |
|---|---|---|
| Parameters | **382,804,992** (~383M) | `rust/config/browser-390m*.json`, `task rust:estimate` |
| Width / layers / heads / MLP | 1152 / 24 / 16 / 4× | spec JSON |
| Latent channels | 16 (Wan2.1 VAE `z_dim`) | reconciled from a bad `4` — `EVALUATION.md` gap 4 |
| Text width | **512 (umt5-small)**, not 4096 | `browser-384m-umt5.yaml` |
| Patchify | `[1,2,2]` → 1536 tokens at `[1,16,4,32,48]` | `EVALUATION.md` gap 2 |
| Sizes | F32 1.53 GB · int8 383 MB · **int4 191 MB (8×)** | `EVALUATION.md`, quantizer |

The student is **not** the teacher's architecture. It is a chunk-bidirectional mixer,
not a causal DiT; it is 512-wide in text conditioning, not 4096; and it patchifies to
land on the teacher's token geometry rather than sharing it. Everything in §4 exists to
bridge those mismatches.

## 4. The distillation objective

`video-train/src/lib.rs` computes three losses per step and combines them with fixed
weights (`w_output 1.0`, `w_temporal 0.25`, `w_feature 0.05`; the umt5 config nudges
temporal to `0.3`). AdamW, `weight_decay 0.01`, gradient clipping at norm `1.0`,
`lr 2e-5`.

1. **Output (denoising-response) matching** — `mse(student_pred, teacher_pred)`. The
   student learns to reproduce the frozen teacher's prediction for the exact cached
   `(noisy latent, timestep, prompt)`. This is the primary signal.

2. **Temporal-difference matching** — `mse(Δt student, Δt teacher)` on frame-to-frame
   differences (skipped when `T = 1`). It supervises motion/temporal structure
   directly rather than leaving it implicit in per-frame output loss.

3. **Width-independent hidden-relation matching** — the load-bearing novelty. For a
   set of layers, the student's hidden states `H ∈ ℝ^{tokens×width}` are reduced to a
   **relation (Gram) matrix** `R = H Hᵀ ∈ ℝ^{tokens×tokens}`, and matched against the
   teacher's cached relation matrix with MSE, averaged over `linspace`-selected layer
   pairs. Because `R` is indexed by *tokens*, not *width*, **a 512-wide student can
   match a 4096-wide teacher's internal structure** — the widths cancel. This is what
   makes cross-width, cross-architecture distillation coherent.

### The geometry constraint that makes (3) work

The relation loss only broadcasts if student and teacher produce the **same token
count**. Every Wan-family DiT patchifies `[1,2,2]`, so an un-patchified student at
`[1,16,4,32,48]` would emit `6144` tokens against a teacher's `1536` and the Gram MSE
would fail on the first real shard. Patchify (§3) aligns them at `1536×1536`. This was
the single most important correctness fix, and `EVALUATION.md` names the reason it was
invisible for so long: the synthetic cache generator produced the teacher side *at the
student's own token count*, so **green CI proved plumbing, not correctness**. The
project's guiding maxim is now "synthetic ≠ real geometry."

## 5. Cross-encoder conditioning (typing in the browser)

The teacher conditions on umT5-**XXL** (4096-dim, 11.4 GB) — unshippable to a tab. The
student instead conditions on umt5-**small** (512-dim), *the same encoder the browser
runtimes already tokenize with*. The trick: **the student's `text` input `Linear`
(512 → width) is the learned projection** — no extra module. During caching, each shard
carries *both* embeddings of the *same caption* (`prompt_embeds [1,128,4096]` for the
teacher, `student_prompt_embeds [1,128,512]` for the student); the trainer routes each
to its side. The student learns to reproduce a T5-XXL-conditioned teacher response from
a umt5-small-conditioned input.

The accepted cost is stated openly: umt5-small is a much weaker encoder, so prompt
adherence will trail the teacher. That is the price of running entirely in-browser.

## 6. Supervision economics: draws, not clips

Because the cache freezes `(noisy_latent, timestep)` into each shard and the trainer
cycles shards by `(step-1) % len`, **effective supervision equals shard count, not step
count** — step 200,001 re-shows a byte-identical input seen hundreds of times. The
consequence, from `TEACHER-OPTIONS.md`: the unit of supervision is a *timestep draw*,
not a *clip*. Emitting 8–32 `(noise, timestep)` draws per prompt turns a few hundred
prompts into a real dataset, **at the cost of teacher forward passes rather than video
data** — no video corpus is required. Combined with caching only 4–6 relation layers
(not all 30) and a bounded lazy `ShardCache` (replacing an eager full-RAM load), a shard
lands near ~1 MB and the dataset scales.

## 7. Deployment: Rust → WASM → WebGPU, quantized

`video-web` exposes the student via `wasm-bindgen`: `new BrowserModel(specJson)`,
`await model.prepare()` (acquires the WebGPU adapter, instantiates the Burn model),
`model.generate(seed, steps, side)`. `task rust:wasm` compiles it with
`wasm-pack --target web`; there is **no ONNX Runtime** on this path.

Weights ship as a quantized bundle: `video-cli quantize` reads the trained Burn record
and writes `weights.q{4,8}` + an ordered `index.json`; `video-web` dequantizes onto a
fresh model **in module order, with no tensor names**, so producer and consumer cannot
drift. int4 is 8× smaller than F32 (1.53 GB → ~191 MB). A unit test proves the bundle
reconstructs the model's forward pass.

## 8. The role of Pruna Smash

Pruna [Smash](https://docs.pruna.ai) sits on the **teacher side**, and only there.
`python/smash_wan21.py` compresses `Wan2.1-T2V-1.3B` with a native-CUDA compiler stack
(`torch_compile` by default; optional kernels such as `flash_attn3` when the GPU and
Pruna build support them) and persists the result; `python/benchmark_wan21.py` times it.

Its function in *this* pipeline is to make the **expensive part cheaper**: the cache is
manufactured by thousands of frozen-teacher forward passes, and a Smashed teacher runs
those faster and cheaper on CUDA. Two boundaries are deliberate and documented:

- **Pruna optimizes the teacher, never the student.** *"Smash compiler outputs are not
  WebGPU models. Keep native Pruna artifacts server-side."* (`README.md`,
  "Production cautions"). The browser artifact is produced by Burn quantization (§7),
  not by Pruna.
- **It is CUDA-only, and therefore optional on the free tier.** The Kaggle CPU
  cache kernel (`kaggle/cache_chunk.py`) *deliberately does not install Pruna* — *"the
  native CUDA compressor, which this CPU cache job never touches."* Pruna is the path
  you switch on when a GPU teacher pass is available and throughput matters.

So the system uses **two complementary compression regimes for two different models**:
Pruna Smash (native CUDA) to accelerate the *teacher* that builds the cache, and Burn
int4/int8 (WebGPU) to shrink the *student* that ships. They never touch the same tensor.

## 9. Free-tier training system

Distillation runs on free hardware as two GitHub Actions pipelines over Kaggle:

- **`cache.yml` → Kaggle CPU** (`kaggle/cache_chunk.py`, driven by
  `scripts/kaggle-cache.mjs`): encodes captions with both encoders, runs the frozen
  Wan2.1 teacher over each `(clip, draw)`, and versions the shards as the
  `…-teacher-cache` dataset. CPU because the teacher pass is inference-only.
- **`train.yml` → Kaggle GPU** (`kaggle/run_chunk.py`, driven by
  `scripts/kaggle-orchestrate.mjs`): trains the student in **resumable chunks**.
  `--resume` restores weights, AdamW moments, the step counter, *and* the shard cursor,
  so N chunks land where one long run would (asserted by
  `resumed_chunks_match_a_single_run`). A content-addressed toolchain cache keyed on a
  hash of `rust/**` avoids re-compiling Burn/CUDA on every run; multi-GB checkpoints are
  versioned back *from inside* the kernel, so CI only ever downloads a few-KB
  `state.json`. On completion, `student.bin` is published to a rolling `weights-latest`
  release.

The whole path — cache on CPU, train on GPU, quantize, bundle — is CI-driven and needs
no browser and no local GPU.

## 10. Honest status

Per `EVALUATION.md`, the system is **architecturally complete and operationally
early**. Phases 0–3 landed: patchify + channel reconciliation (real geometry), the
lazy shard cache + per-clip draws, int8/int4 quantization end-to-end, and genuine
prompt conditioning (a unit test proves different prompts → different output). Phase 2
(the real Wan2.1 CPU teacher cache) is implemented and the CI pipeline to run it exists.

What does **not** yet exist is a *validated, quality-assessed trained student from a
real teacher cache*. The most recent GPU run diverged (feature/Gram loss → NaN at step
349) purely from a learning rate left at `1e-4`; the reference config specifies `2e-5`,
and that default has been corrected. No claim of output quality is made here — only that
each mechanism is in place and unit-tested, and the first end-to-end real run is now
unblocked.

## 11. Why the approach generalizes

Nothing above is specific to video except losses (1) and (2). The transferable pattern
is:

> **Manufacture a framework-neutral supervision cache from a frozen teacher once;
> distil a small, framework-native student against it with a width-independent relation
> loss; quantize to int4; ship as WASM/WebGPU.**

- **Image diffusion** — the closest transfer. Swap the temporal loss out; keep output +
  relation matching. Distil SD/SDXL/Flux into a tiny browser image student; the repo
  already carries an SD-Turbo WebGPU runtime and an ONNX student path.
- **Text / LLMs** — cache teacher logits and hidden relations over a prompt set, distil
  a small student with the same width-independent relation matching, quantize Q4, run
  in-browser (WebLLM-style). The diffusion-specific losses drop; the *cache-once + Gram
  + quantized-WASM* spine carries over, as does the cross-encoder trick for shrinking
  the tokenizer/embedding side.

The claim is not that this beats task-specific distillation on quality. It is that a
single, reproducible, free-tier, framework-neutral spine takes a heavyweight open
generative teacher to a typed-prompt, in-browser, sub-200 MB student — and that the same
spine points at image and text with the video-only pieces removed.

## References

See `docs/WHITEPAPER.md` §References for the prior-work bibliography (knowledge
distillation, relational/feature KD, diffusion distillation, video diffusion teachers,
browser/edge ML, and Pruna).
