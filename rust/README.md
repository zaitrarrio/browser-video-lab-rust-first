# Rust-first video model workspace

This workspace removes PyTorch from student training and deployment. It contains:

- `video-contract`: versioned Safetensors teacher-cache and student configuration contracts.
- `video-student`: Burn causal latent-video student plus relation and temporal distillation primitives.
- `video-cli`: cache validation, parameter estimation, and symmetric INT8 bundle creation.
- `video-native`: native Burn/WGPU forward inference.
- `video-web`: WASM wrapper using the same Burn WGPU model.

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

The current `video-web` constructor initializes the architecture. Loading Burn records and invoking streaming generation are deliberately separate from the TypeScript ONNX runtime until the distilled checkpoint exists; the artifact contract prevents pretending an untrained model is deployable.
