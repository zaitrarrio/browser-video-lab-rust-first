# Browser Video Lab

Five related experiments in one project:

1. **SD-Turbo WebGPU** — browser-side text encoding, denoising and VAE decoding using ONNX Runtime Web.
2. **LongLive WebGPU** — a streaming, causal browser runtime with persistent KV cache for an externally exported LongLive graph.
3. **MemFlow WebGPU** — prompt-conditioned adaptive retrieval over a bounded bank of historical video memories.
4. **Rust student (Burn · WGPU · WASM)** — the Burn latent-video student compiled to WebAssembly, running its denoiser directly on WebGPU with no ONNX Runtime. See [Rust-first edition](#rust-first-edition).
5. **Wan 2.1 + Pruna Smash** — native CUDA compression, persistence and benchmarking for `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`.

The three ONNX browser runtimes do not download multi-gigabyte weights with the repository. Put exported ONNX models under `public/models`, then copy each `manifest.example.json` to `manifest.json` (the interactive UI defaults to the shipped `manifest.example.json` so it loads out of the box). Large ONNX files should be hosted with byte-range support in production. The Rust student needs no ONNX export — build it with `task rust:wasm` (outputs to `public/rust-video/`).

## Browser app

Requirements: Node 20+, Chrome or Edge with WebGPU, and a desktop GPU.

```bash
npm install
npm run dev
```

The SD graph contract is:

- text encoder: `input_ids -> last_hidden_state`
- UNet: `sample, timestep, encoder_hidden_states -> out_sample`
- VAE decoder: `latent_sample -> sample`

The included denoising loop is deliberately scheduler-neutral and useful for integration testing. For production-quality SD-Turbo output, export the scheduler math into the UNet graph or replace the Euler update in `src/runtime/sd-turbo.ts` with the exact scheduler used during export.

## LongLive experimental graph contract

LongLive's official implementation depends on CUDA/Triton and does not currently publish a browser ONNX graph. The runtime here is browser-native, but requires an export/distillation step outside the browser. Its generator accepts:

- `noise`, `prompt_embeds`, `chunk_index`
- zero or more `past.*` KV tensors

It returns a latent `sample` or `latents` plus matching `present.*` tensors. The runtime renames `present.*` to `past.*` for the next causal chunk. Export a fixed-resolution, fixed-window graph first; dynamic KV shapes are not consistently supported across WebGPU implementations.

This is an honest integration boundary: the browser runner is implemented, but the upstream LongLive CUDA checkpoint cannot simply be renamed to ONNX. LongLive 2.0's NVFP4 kernels are NVIDIA-native and are not WebGPU operators.

## MemFlow adaptive-memory contract

This implementation targets Kling's streaming-video MemFlow, not the unrelated CVPR 2024 optical-flow project with the same name. It adds a bounded browser-side memory bank to the causal generator:

1. Mean-pool the current prompt embedding into a retrieval query.
2. Rank historical memory keys by cosine similarity.
3. Feed only the top-K memory values into the next generation chunk.
4. Store the generator's new `memory_key` and `memory_value` outputs.
5. Evict the oldest entry when the configured capacity is reached.

The generator graph accepts `noise`, `prompt_embeds`, `chunk_index`, `memory_values`, and `memory_mask`. It returns `latents` (or `sample`), `memory_key`, and `memory_value`. Export an empty-memory sentinel path so that a zero `memory_mask` is valid on the first chunk. The example manifest defaults to 16 stored entries and top-4 retrieval.

The official MemFlow release is tested on 80 GB NVIDIA GPUs and uses native PyTorch/CUDA dependencies. The browser runtime is complete, but producing its ONNX graph requires distillation/export of a fixed-shape generator; the original checkpoint does not execute directly in ONNX Runtime Web.

## Smash Wan 2.1

Use a Linux CUDA system with enough VRAM. Pruna authentication/package access may be required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r python/requirements-wan.txt
python python/smash_wan21.py --smoke-test
python python/benchmark_wan21.py artifacts/wan21-t2v-1.3b-smashed
```

Pruna API releases have used both `token=` and `api_key=`; the script supports both call signatures. The default compiler follows Pruna's Wan tutorial. Add an available kernel explicitly, for example `--kernel flash_attn3`, only when the installed Pruna build and GPU support it.

## Distill LongLive for the browser

The production recipe targets an approximately 390M-parameter causal student at four latent frames and 256×384-equivalent latent resolution. It combines output/noise matching, temporal-difference matching, and width-independent hidden-relation matching. Training uses precomputed `.pt` shards in `data/longlive-latents`, each containing `latents` and `prompt_embeds` tensors.

LongLive's upstream repository officially uses an AR teacher-forcing stage followed by DMD distillation. This project adds cross-architecture teacher–student compression on top: the 1.3B teacher remains frozen while the smaller browser architecture learns its denoising response and temporal structure. Implement the small adapter described in `python/longlive_distill/ADAPTER.md` against the exact upstream revision being used.

```bash
task setup:python
task distill:smoke
task distill DISTILL_CONFIG=python/longlive_distill/configs/browser-384m.yaml
task distill:export CHECKPOINT=artifacts/longlive-browser-384m/student.pt
```

The initial export is a timestep-conditioned `denoiser.onnx`; it is intentionally not copied over the browser's causal `generator.onnx`. Recommended progression is 390M BF16 teacher matching, then few-step/DMD tuning, then packaging the distilled sampler behind the browser generator contract, INT8 validation, and only then INT4 weight quantization for WebGPU. The smoke configuration uses synthetic tensors solely to validate the optimizer and export plumbing; production training requires real VAE latents and text embeddings.

### Prompt conditioning for the browser (umt5-small)

The teacher conditions on T5-XXL (4096-dim) embeddings, which cannot run in a browser tab. To keep real free-text prompting, the browser student is distilled to condition on **umt5-small** (512-dim) — the same encoder the LongLive/MemFlow runtimes already tokenize with — while the teacher keeps its own 4096-dim conditioning. The student's `text` input Linear (`text_width` → `width`) *is* the learned projection, so no extra module is required; set `text_width: 512` (see `configs/browser-384m-umt5.yaml`).

This couples the data pipeline: each `.pt` shard must carry both `prompt_embeds` (`[1,128,4096]`, teacher) and `student_prompt_embeds` (`[1,128,512]`, umt5-small) for the same caption. `train.py` routes each to its model; shards without `student_prompt_embeds` fall back to the teacher embeddings. Quality caveat: umt5-small is a much weaker text encoder than T5-XXL, so prompt adherence will trail the teacher — the accepted trade-off for running entirely in-browser.

```bash
task distill DISTILL_CONFIG=python/longlive_distill/configs/browser-384m-umt5.yaml
task onnx:student CHECKPOINT=artifacts/longlive-browser-umt5/student.pt DISTILL_CONFIG=python/longlive_distill/configs/browser-384m-umt5.yaml
# Export the matching umt5-small encoder into the same folder (needs `optimum`):
#   optimum-cli export onnx --model google/umt5-small --task feature-extraction public/models/onnx/text_encoder
```

At inference the `onnx-student` runtime tokenizes the prompt, runs `text_encoder/model.onnx`, and feeds `last_hidden_state` straight into `denoiser.onnx`. Until a `text_encoder` is shipped in the manifest it falls back to seeded embeddings so the tab still works.

## Task automation

Install [Task](https://taskfile.dev/), then run `task --list`. The Taskfile covers dependency setup, browser development, all checks, distillation smoke/production/export, Wan Smash and benchmarking, packaging, and reproducible-output cleanup.

## Rust-first edition

The `rust/` workspace implements the PyTorch-free recurring path:

- A versioned Safetensors cache for one-time LongLive teacher supervision
- A Burn causal latent-video student and distillation primitives
- Width-independent hidden-relation and temporal matching
- Native WGPU and WASM/WebGPU model construction
- Shared Q8 and packed Q4 weight bundles
- Parameter estimation and teacher-cache validation tools

The `video-web` crate exposes the student to the browser via `wasm-bindgen`: `new BrowserModel(specJson)`, `await model.prepare()` (acquires the WebGPU adapter and instantiates the Burn model), then `model.generate(seed, steps, side)` runs the denoiser and returns an RGBA frame. `task rust:wasm` compiles it with `wasm-pack --target web` into `public/rust-video/` (plus a compact `student-spec.json`), which the "Rust student" runtime in `src/runtime/rust-video.ts` dynamically imports. The published 390M spec (`rust/config/browser-390m.json`) is intended for native/quantized bundles; the in-tab demo uses the light `rust/config/browser-demo.json`.

> Wasm build note: the workspace pins `burn` to `default-features = false, features = ["std", "wgpu"]` — the `train` feature drags in `libsqlite3-sys` (C), which cannot compile for `wasm32-unknown-unknown`. `getrandom`'s browser backend is enabled via `rust/.cargo/config.toml`.

PyTorch is now isolated to the optional `task teacher:cache` bridge and Pruna's Wan compressor. Once teacher shards exist, student development and deployment can be Rust-only. See `rust/README.md` for the artifact contracts and progression.

## Validate

```bash
npm run typecheck
npm test
npm run build
node scripts/check-models.mjs public/models/*/manifest.example.json
```

## Deploy the demonstration page

The demo builds to a static `dist/` bundle served by a zero-dependency Node server
(`server/index.mjs`) that sets the COOP/COEP headers WebGPU threading requires and
supports HTTP byte-range requests for large model files.

Locally:

```bash
npm run serve          # build, then serve dist/ on http://localhost:8080
npm start              # serve an existing dist/ (set PORT/STATIC_ROOT to override)
```

**Railway.** `railway.json` builds the multi-stage `Dockerfile` (typecheck + test +
build → runtime that serves `dist/`) and health-checks `/healthz`. Deploy with
`railway up`, or let CI handle it:

```bash
railway up            # from a linked project, or
docker build -t browser-video-lab . && docker run -p 8080:8080 browser-video-lab
```

**GitHub Actions.** `.github/workflows/deploy.yml` runs the full check suite
(`typecheck`, `test`, model-manifest validation, `build`) on every push and PR, then
deploys `main` to Railway. Configure these repository settings:

- Secret `RAILWAY_TOKEN` — a Railway project/account token.
- Variables `RAILWAY_SERVICE` (defaults to `browser-video-lab`) and optionally
  `RAILWAY_PUBLIC_URL` for the deployment environment link.

## Releases & WASM artifacts

`.github/workflows/release.yml` versions and releases automatically **when a pull
request is merged into `main`** (or via manual `workflow_dispatch`). It:

1. Picks the bump level — a `release:major` / `release:minor` / `release:patch` PR
   label wins; otherwise the PR title's conventional-commit prefix decides
   (`feat:` → minor, `!:` / `BREAKING CHANGE` → major, everything else → patch).
2. Computes the next semver from the latest `v*` tag (the first release publishes
   the current `package.json` version as-is), bumps `package.json`, and commits it
   back to `main` as `chore(release): vX.Y.Z [skip ci]`.
3. Builds the page, then publishes the ONNX Runtime Web `.wasm` binaries from
   `onnxruntime-web/dist` as **release assets**, alongside `SHA256SUMS.txt` and a
   `wasm-manifest.json` recording the release and `onnxruntime-web` versions.

Consumers can then load weights from a stable release URL and pin
`ort.env.wasm.wasmPaths` to it. No secrets are required — the built-in
`GITHUB_TOKEN` creates the tag and release. If `main` is a protected branch, allow
the `github-actions[bot]` to push the version-bump commit (or supply a PAT).

## Build models in CI & cache weights

`.github/workflows/models.yml` builds the reproducible, PyTorch-free model path
on every change under `rust/**` (and on demand via `workflow_dispatch`). It:

1. Sets up Rust with the `wasm32-unknown-unknown` target and runs the workspace
   tests, the parameter estimate, and one native WGPU smoke forward pass.
2. Compiles the Burn/WGPU student to WebAssembly (`public/rust-video`).
3. Quantizes an F32 Safetensors checkpoint to Q8 then Q4 bundles **when one is
   available** — set the repo variable `CHECKPOINT_URL` (or pass one to
   `workflow_dispatch`). With no checkpoint it builds the runtime and skips
   quantization rather than fabricate untrained weights.
4. Assembles a `model-bundle/` (WASM runtime + student spec + any weight
   bundles) with a `models-manifest.json` recording each file's size and
   SHA-256, uploads it as a build artifact, and attaches it to `v*` tag releases.

**Two caching layers keep it fast:**

- `Swatinem/rust-cache` caches the compiled Burn/WGPU dependency graph (the
  dominant cost), keyed on `rust/Cargo.lock`.
- An `actions/cache` step caches the produced weight bundle (`public/rust-video`
  and the Q8/Q4 artifacts), keyed on a hash of the Rust sources, configs, and
  the checkpoint URL. When nothing that affects the weights changed, the bundle
  is restored instead of rebuilt.

Locally, mirror the bundle step with `task models:bundle` (builds the WASM
module and writes the content-addressed manifest). The SHA-256 manifest is what
lets the browser cache weights by content hash via a service worker or the
Origin Private File System.

## Production cautions

- Configure COOP/COEP headers if using threaded WASM fallbacks.
- Cache weights with a service worker or Origin Private File System.
- Validate model licenses before redistributing weights.
- Browser GPU memory is not the same as installed VRAM; handle device loss and out-of-memory errors.
- Keep native Pruna artifacts server-side. Smash compiler outputs are not WebGPU models.
