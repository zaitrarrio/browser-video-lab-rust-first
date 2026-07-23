# Rust-first video model workspace

This workspace removes PyTorch from student training and deployment. It contains:

- `video-contract`: versioned Safetensors teacher-cache and student configuration contracts.
- `video-student`: Burn causal latent-video student plus relation and temporal distillation primitives.
- `video-cli`: cache validation, parameter estimation, and symmetric INT8 bundle creation.
- `video-native`: native Burn/WGPU forward inference.
- `video-train`: native, PyTorch-free distillation trainer over the teacher cache (CPU/WGPU, optional CUDA).
- `video-web`: WASM wrapper using the same Burn WGPU model; loads trained `student.bin` records when shipped.

The official LongLive teacher remains an optional producer. Run it once to generate framework-neutral teacher shards; all recurring training and deployment can then be Rust-only. A complete teacher-cache shard contains `noisy_latents`, `timestep`, `prompt_embeds`, and `teacher_noise_pred`; optional `teacher_relation.N` tensors preserve intermediate structure without forcing equal hidden widths.

The Burn student performs bidirectional attention within each emitted chunk. Causality is enforced at the streaming level: chunks only receive prior cached context, never future chunks. This avoids large browser attention masks and matches the existing chunked runtime.

## Build

```bash
task setup:rust
task rust:check
task rust:test
task rust:native:smoke
task rust:wasm
```

## Stages

1. Cache official teacher supervision once.
2. Validate it with `video-cli validate-cache`.
3. Train the Burn student in BF16/WGPU or CUDA-capable Burn backend.
4. Perform few-step/DMD tuning before treating the denoiser as a generator.
5. Quantize to Q8, validate quality, then add Q4 only after Q8 passes.
6. Deploy through native Burn WGPU or the WASM wrapper.

## Training in Burn

`video-train` ports `losses.py` (output MSE + temporal-difference MSE + hidden-relation matching against the cache's precomputed `teacher_relation.N` gram matrices) and trains from safetensors shards directly — no PyTorch after the one-time teacher cache. Shards may carry an optional `student_prompt_embeds` tensor (umt5-small, 512-dim) so the student conditions on a browser-runnable encoder while the teacher keeps its own embeddings; without it the loader falls back to `prompt_embeds`.

```bash
task rust:train:smoke                       # synthetic cache → 10 CPU steps → student.bin (plumbing gate)
task rust:train CACHE=data/teacher-cache    # real run; add -- --backend cuda --steps 100000 --ckpt-every 1000
task rust:weights                           # install student.bin + matching spec into public/rust-video
```

`--backend` selects ndarray (CPU), wgpu (Metal/Vulkan/DX12), or cuda (`--features cuda`; suits a Colab GPU session — checkpoint with `--ckpt-every` and continue across sessions with `--resume`, which restores model weights and restarts optimizer state cold). The trainer writes `student.mpk` checkpoints plus a `student.bin` BinFileRecorder record; `video-web` fetches `rust-video/student.bin`, loads it via `prepare_with_weights`, and reports `trained weights` vs `random init` in the demo status line so an untrained model is never passed off as distilled.
