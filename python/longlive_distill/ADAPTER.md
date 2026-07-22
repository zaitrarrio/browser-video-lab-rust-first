# LongLive teacher adapter

The production config intentionally names `longlive_teacher_adapter:build_teacher`. Create that module in `python/` against the exact upstream LongLive revision and checkpoint you use. It must return a frozen `torch.nn.Module` whose forward signature is:

```python
forward(noisy_latents, timestep, prompt_embeds) -> {
    "noise_pred": Tensor[B,C,T,H,W],
    "hidden_states": list[Tensor[B,N,D]],
}
```

Keeping this adapter local prevents this project from copying or guessing private checkpoint-loading details. The supplied toy teacher exercises the entire optimizer, loss, checkpoint, and export path without an NVIDIA GPU.
