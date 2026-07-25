# From trained weights to a browser download

Three different things in this repo are easy to mistake for one another, because
all three are called "compression" somewhere. They act on different objects, at
different stages, for different reasons.

| | changes | acts on | produces | needed for |
|---|---|---|---|---|
| **Distillation** — `video-train` | the architecture: 1.3B/5B → 383M | teacher → new student | `student.bin` | the whole project |
| **Quantization** — `video-cli quantize` | the number format: F32 → int8/int4 | the trained student | `weights.q8` + `index.json` | a shippable download |
| **Smash** — `python/smash_wan21.py` | execution: `torch_compile`, kernels | the *teacher* | a faster CUDA pipeline | server-side speed only |

## Smash is a side track

`smash_wan21.py` compresses `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` with Pruna and
exits immediately without CUDA: `raise SystemExit("CUDA GPU required for Wan
compression")`. It produces nothing the browser can load, and it is not part of
this path.

Its two real uses:

- **Baseline.** `benchmark_wan21.py` establishes what the full model costs on a
  server — the number the browser student is trying to approximate.
- **Cache-building speedup**, but only if the teacher is Wan2.1. See
  `python/longlive_distill/TEACHER-OPTIONS.md`; Plan B's teacher is exactly the
  model this script already targets, so a rented-GPU cache build could smash the
  teacher first and cut the cost of thousands of forward passes. Note that
  distilling from a compressed teacher bakes its error into the student —
  `torch_compile` is lossless, quantized kernels are not.

## Quantization is on the critical path, not polish

`StudentSpec::approximate_parameters` gives **382,804,992** parameters for
`browser-390m-umt5.json`. At `FullPrecisionSettings` that is:

| format | size |
|---|---|
| fp32 — what `student.bin` is today | **~1.53 GB** |
| int8 | ~383 MB |
| int4 | ~191 MB |

1.53 GB is not a browser download. Quantization is what makes the deliverable
deliverable, and the gap between it and int4 is a factor of eight.

`quantize()` in `video-cli/src/main.rs` is per-tensor symmetric: one `scale` per
tensor, no zero point, `qmax` 127 or 7. Int4 packs two values per byte, each
shifted by +8 into a nibble. It writes `weights.q{bits}` plus an `index.json` of
`{name, shape, scale, offset, length, bits}`.

## Both ends of that step are missing

**No producer.** `quantize()` opens its input with `SafeTensors::deserialize` and
skips every tensor that is not F32. `video-train` writes `student.bin` with Burn's
`BinFileRecorder` and `student.mpk` with `NamedMpkFileRecorder` — neither is
safetensors. Nothing in the repo converts between them, so the trained student
cannot currently be fed to the quantizer at all. (`video-train` does depend on
`safetensors`, but only to *read* the teacher cache.)

**No consumer.** `video-web` exposes `BrowserModel::new(spec_json)` and
`prepare_with_weights(weights: Uint8Array)`, and `src/runtime/rust-video.ts:61`
fetches exactly one artifact — `rust-video/student.bin` — falling back to random
init if it 404s. There is no reference to `q8`, `q4`, `index.json` or `weights.q`
anywhere in `video-web/src/lib.rs` or `rust-video.ts`. `task models:bundle` copies
`artifacts/student-q8` and `student-q4` into the bundle when they exist, and
`models:manifest` hashes them, but nothing ever loads them.

So `video-cli quantize` today is a correct utility wired to nothing at either end.

Closing the gap needs two pieces of work, independent of each other and of the
teacher decision:

1. **Producer** — either export the Burn record to safetensors F32, or teach
   `quantize()` to read a Burn `.bin`/`.mpk` directly.
2. **Consumer** — a `prepare_with_quantized(index_json, weights)` path in
   `video-web` that dequantizes per tensor on load (or, better, keeps int8 and
   dequantizes per matmul), plus the fetch logic in `rust-video.ts` to prefer
   `q4`/`q8` over `student.bin`.

## The student does 16× the attention work of its own teacher

Download size is one half of "will this run in a browser". The other half is
tokens, and there is a large, free win sitting here.

`BrowserVideoStudent::forward` flattens `[B,C,T,H,W] → [B, T·H·W, C]` — no
patchify. At `latent_shape [1,4,4,32,48]` that is **6144 tokens**. Every
Wan-family DiT, teacher and otherwise, uses `patch_size: [1,2,2]` (verified in
each published `transformer/config.json`), so the *teacher* processes the same
content as **1536 tokens**.

Attention is O(N²):

| | tokens | attention cost |
|---|---|---|
| student today | 6144 | 1× |
| student with `[1,2,2]` patchify | 1536 | **1/16** |
| + a spatial-16 VAE (Wan2.2-TI2V) | 384 | **1/256** |

Adopting patchify cuts tokens 4× and attention 16×, at some cost in spatial
detail — the same trade every DiT in this family already makes. It also makes the
relation-loss gram shapes line up with a Wan teacher, which they currently do not:
see `python/longlive_distill/TEACHER-OPTIONS.md`.

This is independent of teacher choice, of quantization, and of the cache work. It
is the single largest playback lever available, and nothing depends on it landing
first.

## The ONNX track is a different browser runtime

`task onnx:student` runs `longlive_distill.export` against a **PyTorch**
checkpoint and writes `public/models/onnx/denoiser.onnx` for the ONNX Runtime Web
demo. That is a separate runtime from the Burn/WASM/WebGPU one, fed by a separate
trainer (`python/longlive_distill/train.py`, not `video-train`). The root README
is explicit: *"The Rust student needs no ONNX export."* Two parallel browser
tracks; do not wire them together by accident.

## Two ordering hazards in the bundle

- `task rust:wasm` writes `browser-demo.json` into `public/rust-video/` as the
  default spec. `task rust:weights` must run **after** it, or the demo spec
  overwrites the trained one and the record will not match the architecture.
- `models:bundle` copies `rust/config/browser-390m.json` (the 4096-wide variant)
  as `student-spec.json`, then overwrites it with `TRAIN_SPEC` only if
  `student.bin` exists. Without trained weights the bundle advertises a spec that
  does not match anything shipped.

## The prompt is not wired up

`BrowserModel::generate` synthesizes its own latents *and* prompt embeddings from
an LCG, and `rust-video.ts:75` names the argument `_prompt` because it is
discarded. The demo cannot be prompted, whatever weights it loads.

Making text conditioning real needs a umt5-small encoder reachable from the
browser — a 512-wide encoder matching `text_width` in the umt5 spec — and that is
neither built nor scoped. No amount of teacher quality substitutes for it, and it
is worth deciding whether the deliverable is "a browser model that denoises" or
"a browser model you can type at", because they are different projects.

## Current state

```
teacher ──smash?──▶ teacher cache ──▶ video-train ──▶ student.bin (1.53 GB)
                    (format work,       (proven on          │
                     teacher TBD)        free P100)         │
                                                      ✗ quantize   — format mismatch
                                                      ✗ load q4/q8 — not implemented
                                                      ✗ prompt     — discarded
                                                            │
                                                     browser ~191 MB
```

Two axes, and they are independent:

- **Size** — 1.53 GB → 191 MB, gated on the quantize producer and consumer above.
- **Speed** — 6144 → 1536 tokens via patchify, or 384 with a spatial-16 VAE.

Neither is blocked by the teacher decision. Both are blocked on work nobody has
started.
