# Evaluation: findings, next steps, gains, applications

A synthesis of what `rust/DELIVERY.md` and `python/longlive_distill/TEACHER-OPTIONS.md`
establish, an assessment of how solid those findings are, the implementation order they
imply, the size and speed each step buys, and where the result can be applied. Every
claim below was re-checked against the source, not taken from the prose.

## 1. What the findings actually say — and whether they hold

The two documents describe a project that is **architecturally finished and
operationally unstarted**. The recurring student — spec JSON plus a Burn record — is
independent of the teacher and already runs in the browser (`video-web::BrowserModel`,
`src/runtime/rust-video.ts`). What does not exist yet is any path from a *real* teacher
to a *shippable* download. Five gaps stand between the two, and a sixth fact explains why
none of them are visible in CI.

### The meta-finding: the synthetic path masks every real-geometry bug

`synth-cache` generates the teacher side at the **student's own** token count, channel
count, and layer count. So the relation loss broadcasts, the latent projection fits, and
the tests pass — structurally, none of the real-world mismatches below *can* surface
until the first real shard. This is the single most important observation in the two
docs, because it means "green CI" currently proves plumbing, not correctness. Every
gap below is a place where synthetic ≠ real.

### The five gaps (all verified against source)

| # | Gap | Evidence (re-verified) | Axis | Blocks |
|---|---|---|---|---|
| 1 | **Quantize is wired to nothing.** | `video-cli` `quantize()` opens with `SafeTensors::deserialize` and `continue`s on any non-F32 tensor; `video-train` writes Burn `BinFileRecorder`/`NamedMpkFileRecorder`, never safetensors. `rust-video.ts:61` fetches only `student.bin`; no `q4`/`q8`/`index.json` reference exists in `lib.rs` or `rust-video.ts`. | Size | The deliverable. 1.53 GB → 191 MB is gated here. |
| 2 | **No patchify.** | `BrowserVideoStudent::forward` (`video-student/src/lib.rs:41`) does `reshape([b, t·h·w, c])` — flatten, no `patch_size`. At `[1,4,4,32,48]` = 6144 tokens vs a Wan teacher's 1536. | Speed **and** trainability | Playback cost (O(N²)) **and** the relation-loss gram shape match. |
| 3 | **Cache can't hold enough supervision.** | `video-train/src/lib.rs:143` loads every shard into RAM eagerly; `cache_teacher.py` writes one `(noise, timestep)` per clip and `hidden_relation_layers = range(n_layers)` = all 30. Training cycles shards by `(step-1) % len`, so effective supervision = shard count, not step count. | Data / trainability | A 383M student cannot learn from ~200 byte-identical inputs. |
| 4 | **Latent-channel mismatch.** | `browser-390m-umt5.json` declares `latent_channels: 4`; input projection is `Linear(4 → 1152)`. No published VAE emits 4 channels — Wan2.1 `z_dim 16`, Wan2.2-TI2V `z_dim 48`. | Correctness | The first real shard. Invisible to `synth-cache`. |
| 5 | **Prompt is discarded.** | `BrowserModel::generate` builds prompt embeddings from an `Lcg`; `rust-video.ts:75` names the arg `_prompt`. | Product | "A model you can type at" — a different deliverable from "a model that denoises". |

**Assessment.** These findings are unusually trustworthy: each is tied to a file and line,
cross-checked against the *published* `transformer/config.json` geometry, and honest about
what the synthetic path hides. The param count (`382,804,992`) and the size table
(fp32 1.53 GB / int8 383 MB / int4 191 MB) are consistent with the spec
(`width 1152, layers 24, mlp_ratio 4`). I found no overstated claim. The one thing the
docs under-weight is that gaps 2 and 4 are *the same class of bug* as the meta-finding —
geometry the synthetic generator invents to match itself — so they should be fixed
together, behind one "real-geometry" gate.

### Status — Phase 0 landed on this branch

The teacher-free, GPU-free gate is implemented:

- **Gap 2 (patchify) — done.** `BrowserVideoStudent::forward` now patchifies
  `patch_size` (default `[1,2,2]`); a 4×32×48 latent runs 1536 tokens, not 6144
  (16× less attention), and the hidden relation grams now match a Wan teacher's.
- **Gap 4 (channels) — done.** `latent_channels` reconciled to real VAE z-dims:
  16-ch Wan2.1 is the default on every build/demo path; a 48-ch Wan2.2 config
  ships as a documented (GPU-teacher) option. The param count stays ~383M.
- **Gap 3 (cache scale) — done for the Rust path.** The eager RAM load is replaced
  by a bounded lazy `ShardCache`; `synth-cache --draws-per-clip` multiplies
  supervision per clip. Token subsampling is intentionally deferred (patchify
  already aligns the grams, making it a size optimization, not a correctness fix).

**Phase 1 (quantization, gap 1) — done.** `video-cli quantize` now reads the
trained Burn record (`student.bin`/`.mpk`) and writes an int8/int4 bundle
(`weights.q{bits}` + ordered `index.json`); `video-web::prepare_with_quantized`
dequantizes it onto a fresh model in module order (no tensor names, so producer
and consumer can't drift), and `rust-video.ts` prefers a quantized bundle over
`student.bin`. int4 is 8× smaller than F32 — 1.53 GB → **~191 MB** for the 390M
spec. A unit test proves the bundle reconstructs the model's forward pass; a CI
smoke proves the record→bundle wiring.

**Phase 3 (prompt conditioning, gap 5) — done.** `BrowserModel::generate` no
longer synthesizes prompt embeddings from an LCG and `rust-video.ts` no longer
discards `_prompt`: the runtime encodes the prompt (a real umt5-small ONNX
encoder when a `rust-video/text-encoder.json` manifest ships one, else a
deterministic prompt-seeded embedding) and hands the `[seq, text_width]` tensor
to the WASM student. The tokenize→encode path is shared with the ONNX student
runtime, so typing a different prompt changes the output even before a real
encoder is shipped. A unit test proves the student is genuinely conditioned on
the prompt (different embeddings → different output).

Remaining: **Phase 2** — a one-time teacher cache to make the weights *trained*
rather than random (Plan B/Wan2.1 under the no-GPU constraint). After that the
browser student is small, fast, promptable, and actually distilled.

### The teacher decision

`TEACHER-OPTIONS.md` reduces cleanly to one question — **is there budget for a few hours
of rented GPU?**

- **No GPU → Plan B (Wan2.1-1.3B).** Apache-2.0, 17.5 GB, `diffusers`-native, plumbing
  already in-tree. Gives up the causal long-video signal the current chunk-bidirectional
  student may not exploit anyway.
- **Some GPU → Plan C (LongLive-2.0-5B).** Commercial-friendly licence, 4-step DMD
  sampling (≈10× cheaper latent generation), and — decisively — a **spatial-16 VAE** that
  makes the browser student 256× cheaper in attention and shrinks shards 16×.
- **Plan A (LongLive-1.3B)** is hard to justify: C beats it on licence and quality, B on
  cost and simplicity.

This recommendation is sound. The critical insight is buried at the end: **the
cache-format work (gaps 2–4) is identical across all three plans and is the real
prerequisite** — it can be built and tested against the synthetic generator *before any
teacher is downloaded*, and is the only work guaranteed not to be wasted.

## 2. Next implementation steps (dependency-ordered)

Ordered so each step is checkable before the next exists, and so no step waits on the
teacher decision until it must.

**Phase 0 — Real-geometry gate (no teacher, no GPU). The unblock-everything phase.**
1. **Add `patch_size [1,2,2]` to the student** (gap 2). Patchify on input, unpatchify on
   output. 6144 → 1536 tokens. This simultaneously fixes the relation-loss shape mismatch
   with a Wan teacher — the highest-leverage single change in the repo.
2. **Reconcile latent channels** (gap 4). Move the spec to the teacher's `z_dim`
   (16 for B, 48 for C) *or* carry a projection-to-4 in the cache. Decide with the teacher.
3. **Cache-format v2** (gap 3), tested against an extended `synth-cache`:
   - subsample *K* (~512) shared token positions before the gram, store indices;
   - cache 4–6 layers via `linspace_idx`, not all 30;
   - emit 8–32 `(noise, t)` draws per clip;
   - replace the eager RAM load at `video-train/src/lib.rs:143` with a streaming loader.
   Gate: `resumed_chunks_match_a_single_run` still passes; RAM stays flat vs shard count.

**Phase 1 — Close the size axis (no teacher).**
4. **Quantize producer** (gap 1a): export the Burn record to F32 safetensors *or* teach
   `quantize()` to read `.bin`/`.mpk` directly.
5. **Quantize consumer** (gap 1b): `prepare_with_quantized(index_json, weights)` in
   `video-web` (keep int8, dequantize per-matmul); make `rust-video.ts` prefer `q8`/`q4`
   over `student.bin`. After this the 191 MB path is real end-to-end.

**Phase 2 — Teacher, once.**
6. Pick B or C by the GPU-budget question. Build the adapter (`ADAPTER.md` contract),
   the prompt→latents pass, and the **two-width** prompt-encoding pass (umT5-XXL for the
   teacher, **umt5-small for `student_prompt_embeds`** — currently never written).
7. Teacher smoke on **one** CPU sample; time it. Go/no-go for throughput.
8. End-to-end small: ~32 prompts × 8 timesteps = 256 shards; 1,000-step run; confirm
   cache HIT and falling loss; compare the curve against synthetic (must be clearly
   better, or something upstream is wrong).

**Phase 3 — Product.**
9. **Wire the prompt** (gap 5): ship a umt5-small encoder reachable from the browser,
   feed real `last_hidden_state` into the student, stop discarding `_prompt`. This is a
   scoping decision, not polish — it defines whether the deliverable is promptable.

Phases 0 and 1 need no teacher and no GPU. That is roughly half the remaining work, and
all of it is de-riskable today.

## 3. Potential gains (quantified)

**Size (Phase 1).** Per-tensor symmetric quantization on 382.8M params:

| format | size | vs fp32 |
|---|---|---|
| fp32 (`student.bin` today) | 1.53 GB | 1× |
| int8 | 383 MB | 4× |
| int4 | 191 MB | **8×** |

The gap between "not a browser download" and "a browser download" is exactly this step.

**Speed (Phase 0 + teacher choice).** Attention is O(N²):

| configuration | tokens | attention cost |
|---|---|---|
| student today (flatten) | 6144 | 1× |
| + `[1,2,2]` patchify | 1536 | **1/16** |
| + patchify, Wan2.2 spatial-16 VAE (Plan C) | 384 | **1/256** |

Patchify alone is a **16× compute cut** at some spatial-detail cost — the exact trade
every Wan-family DiT already makes — and it is free in the sense that it also *enables*
real-teacher distillation. Plan C's VAE compounds it to 256×.

**Data (Phase 0).** Multiple `(noise, t)` draws per clip convert a few hundred prompts
into thousands of independent supervision points at the cost of teacher forward passes,
not video data — the difference between a student that can and cannot converge.

**Combined:** a ~191 MB, 384-token, promptable browser video student — two orders of
magnitude cheaper in attention and eight times smaller than what exists today — with no
step blocked by anything but work.

## 4. Potential applications

- **Zero-server generative video in the browser.** The whole point: WebGPU inference,
  weights cached by content-hash in a service worker / OPFS, no inference backend to run
  or pay for. Viable for demos, edu, on-device creative tools, and privacy-sensitive use
  where prompts never leave the tab.
- **Reproducible, PyTorch-free model supply chain.** The Rust student trains and deploys
  without PyTorch; PyTorch is isolated to the one-time teacher cache and the optional
  Pruna/Wan side-track. `models.yml` already content-addresses the bundle (SHA-256
  manifest) — a template for auditable, cache-friendly model distribution.
- **Cross-architecture distillation harness.** The teacher-cache contract
  (`noisy_latents, timestep, prompt_embeds, teacher_noise_pred`, optional relation grams)
  is teacher-agnostic. Once the token-subsampled relation loss lands, the same harness
  can distil *any* DiT-family teacher into a small browser student — the Wan2.2-VAE-for-
  another-teacher idea in `TEACHER-OPTIONS.md` is one unexplored instance.
- **Edge / commercial deployment (Plan C).** The NVIDIA Open Model licence on
  LongLive-2.0-5B and Apache-2.0 on Wan2.1 both permit commercial use, so a distilled
  student under a chosen licence is a shippable component, not just a research artifact.
- **A worked case study in synthetic-test blind spots.** The meta-finding — that a
  self-consistent synthetic generator hides every real-geometry bug — generalizes to any
  ML pipeline validated against fixtures it also generates. Worth citing beyond this repo.

## 5. Bottom line

The findings are accurate, source-grounded, and honest about their own blind spot. The
architecture is done; the recurring pipeline is not. **Phase 0 — patchify, channel
reconciliation, and cache-format v2 — is now implemented on this branch:** it was
teacher-free, GPU-free, and fixed the trainability blocker and the largest speed lever at
once, so none of it can be wasted by a later teacher decision. Quantization (Phase 1) is
next and makes the artifact shippable; the teacher (Phase 2) makes it *good*; the prompt
(Phase 3) decides whether it is a product.
