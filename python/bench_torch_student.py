"""A PyTorch reimplementation of the Burn student, purely to price the trainer.

`PERF-ROUND-2.md` measures the Rust/Burn/cubecl trainer at 23.1% of what torch
gets out of the same card on a plain bf16 matmul. That gap is the entire argument
for moving the trainer off Burn, and it cannot be settled by comparing a training
step against a matmul microbenchmark — it needs the *same model, same loss, same
optimizer, same geometry*, in both frameworks.

This file is that reference. It is deliberately not a training tool: it runs a
fixed number of steps on synthetic tensors and reports throughput. Correctness
here means **structural** parity with `video-student` — the same parameter count,
the same shapes — not numerical agreement, since the point is speed.

Mirrors `rust/crates/video-student/src/lib.rs`:

  patchify [b,c,t,h,w] -> [b, t*(h/ph)*(w/pw), c*ph*pw]
  cond  = text(prompt.mean(1)) + time(timestep/1000)          [b,1,width]
  x     = input(tokens) + cond
  block = adaLN-zero(cond) -> attn -> gated residual -> MLP -> gated residual
  out   = output(norm(x)), unpatchified

and the three-term loss in `video-train`: output MSE, temporal-difference MSE at
0.25, and relation-gram MSE at 0.05 over two captured layers.

Three attention arms, because "torch is faster" and "torch has a fused attention
kernel Burn 0.21 lacks" are different claims with different consequences:

  --attn naive   materializes [b,heads,N,N] like `tiled_attention` does
  --attn sdpa    F.scaled_dot_product_attention (flash/mem-efficient backend)
  --attn compile sdpa under torch.compile
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class MixerBlock(nn.Module):
    def __init__(self, width: int, mlp_ratio: int, heads: int, per_block_cond: bool, attn: str):
        super().__init__()
        self.heads, self.attn = heads, attn
        self.norm = nn.LayerNorm(width)
        self.q = nn.Linear(width, width)
        self.k = nn.Linear(width, width)
        self.v = nn.Linear(width, width)
        self.proj = nn.Linear(width, width)
        self.norm_mlp = nn.LayerNorm(width)
        self.up = nn.Linear(width, width * mlp_ratio)
        self.down = nn.Linear(width * mlp_ratio, width)
        # Zero-initialised, as in the Rust module: at step 0 both gates are 0 so
        # the block is an exact identity.
        self.ada = None
        if per_block_cond:
            self.ada = nn.Linear(width, 6 * width)
            nn.init.zeros_(self.ada.weight)
            nn.init.zeros_(self.ada.bias)

    def _attend(self, q, k, v, scale):
        if self.attn == "naive":
            # The shape `tiled_attention` builds and autodiff then tapes.
            return torch.softmax(q @ k.transpose(-2, -1) / scale, dim=-1) @ v
        return F.scaled_dot_product_attention(q, k, v, scale=1.0 / scale)

    def forward(self, x, cond):
        b, seq, width = x.shape
        head_dim = width // self.heads
        scale = head_dim ** 0.5
        m = self.ada(cond).reshape(b, 1, 6, width).unbind(2) if self.ada is not None else None

        def mod(t, i, j):
            return t * (m[j] + 1.0) + m[i] if m is not None else t

        def gate(t, i):
            return t * m[i] if m is not None else t

        n = mod(self.norm(x), 0, 1)
        split = lambda t: t.reshape(b, seq, self.heads, head_dim).transpose(1, 2)
        ctx = self._attend(split(self.q(n)), split(self.k(n)), split(self.v(n)), scale)
        x = x + gate(self.proj(ctx.transpose(1, 2).reshape(b, seq, width)), 2)
        h = mod(self.norm_mlp(x), 3, 4)
        return x + gate(self.down(F.gelu(self.up(h))), 5)


class Student(nn.Module):
    def __init__(self, spec: dict, attn: str):
        super().__init__()
        self.spec = spec
        w, pv = spec["width"], spec["latent_channels"] * spec["patch_size"][1] * spec["patch_size"][2]
        self.input = nn.Linear(pv, w)
        self.text = nn.Linear(spec["text_width"], w)
        self.time = nn.Linear(1, w)
        self.blocks = nn.ModuleList(
            MixerBlock(w, spec["mlp_ratio"], spec["heads"], spec.get("per_block_conditioning", False), attn)
            for _ in range(spec["layers"])
        )
        self.norm = nn.LayerNorm(w)
        self.output = nn.Linear(w, pv)

    def forward(self, latents, timestep, prompt):
        b, c, t, h, wd = latents.shape
        _, ph, pw = self.spec["patch_size"]
        hp, wp = h // ph, wd // pw
        x = (
            latents.transpose(1, 2).reshape(b * t, c, hp, ph, wp, pw)
            .permute(0, 2, 4, 1, 3, 5).reshape(b, t * hp * wp, c * ph * pw)
        )
        cond = self.text(prompt.mean(1)).unsqueeze(1) + self.time(timestep / 1000.0).unsqueeze(1)
        x = self.input(x) + cond
        hidden = []
        for blk in self.blocks:
            x = blk(x, cond)
            hidden.append(x)
        y = (
            self.output(self.norm(x)).reshape(b * t, hp, wp, c, ph, pw)
            .permute(0, 3, 1, 4, 2, 5).reshape(b, t, c, h, wd).transpose(1, 2)
        )
        return y, hidden


def relation(x):
    return (x / x.pow(2).sum(-1, keepdim=True).sqrt().clamp_min(1e-6)) @ (
        x / x.pow(2).sum(-1, keepdim=True).sqrt().clamp_min(1e-6)
    ).transpose(1, 2)


def temporal_difference(x):
    return x[:, :, 1:] - x[:, :, :-1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--width", type=int, default=1152)
    p.add_argument("--layers", type=int, default=24)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--attn", choices=["naive", "sdpa", "compile"], default="sdpa")
    p.add_argument("--accum", type=int, default=8)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--frames", type=int, default=4)
    p.add_argument("--size", type=int, default=40)
    p.add_argument("--relation-layers", type=int, default=2)
    p.add_argument("--dtype", choices=["bf16", "f32"], default="bf16")
    p.add_argument("--params-only", action="store_true")
    a = p.parse_args()

    spec = dict(latent_channels=16, text_width=512, width=a.width, layers=a.layers,
                heads=a.heads, mlp_ratio=4, patch_size=[1, 2, 2], per_block_conditioning=True)
    model = Student(spec, "sdpa" if a.attn == "compile" else a.attn)
    n = sum(q.numel() for q in model.parameters())
    # The Rust `approximate_parameters` omits biases, LayerNorms and the adaLN
    # term; recompute its formula here so the two are comparable on the same basis.
    dense = a.layers * (4 + 2 * 4) * a.width * a.width + 512 * a.width + 2 * 16 * 4 * a.width
    ada = a.layers * 6 * a.width * a.width
    print(f"params total={n/1e6:.1f}M  (rust approximate_parameters={dense/1e6:.1f}M + adaLN {ada/1e6:.1f}M "
          f"= {(dense+ada)/1e6:.1f}M; difference is biases + LayerNorms)")
    if a.params_only:
        return

    dev = torch.device("cuda")
    dt = torch.bfloat16 if a.dtype == "bf16" else torch.float32
    model = model.to(dev, dt)
    if a.attn == "compile":
        model = torch.compile(model)
    # AdamW over bf16 parameters, matching Burn, which instantiates the whole
    # backend at one element type rather than keeping an f32 master copy.
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

    tokens = a.frames * (a.size // 2) ** 2
    mk = lambda: (
        torch.randn(1, 16, a.frames, a.size, a.size, device=dev, dtype=dt),
        torch.rand(1, 1, device=dev, dtype=dt) * 1000,
        torch.randn(1, 128, 512, device=dev, dtype=dt),
        torch.randn(1, 16, a.frames, a.size, a.size, device=dev, dtype=dt),
        [torch.randn(1, tokens, tokens, device=dev, dtype=dt) for _ in range(a.relation_layers)],
    )
    batch = [mk() for _ in range(a.accum)]
    sel = [max(0, a.layers // 2 - 1), a.layers - 1][: a.relation_layers]

    def step():
        opt.zero_grad(set_to_none=True)
        for noisy, ts, prompt, target, grams in batch:
            pred, hidden = model(noisy, ts, prompt)
            loss = F.mse_loss(pred, target)
            loss = loss + 0.25 * F.mse_loss(temporal_difference(pred), temporal_difference(target))
            feat = sum(F.mse_loss(relation(hidden[l]), g) for l, g in zip(sel, grams))
            (loss + 0.05 * feat).div(a.accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    for _ in range(a.warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(a.steps):
        step()
    torch.cuda.synchronize()
    el = time.perf_counter() - t0

    sps = a.steps * a.accum / el
    fwd = 2 * (a.layers * 12 * a.width ** 2) * tokens + a.layers * 4 * tokens ** 2 * a.width
    tf = 3 * fwd * sps / 1e12
    print(f"attn={a.attn} dtype={a.dtype} width={a.width} layers={a.layers} tokens={tokens}")
    print(f"  {a.steps} steps x accum {a.accum} in {el:.2f}s -> {sps:.3f} samples/s")
    print(f"  {tf:.1f} TFLOPS   peak {torch.cuda.max_memory_allocated()/2**30:.1f} GiB")


if __name__ == "__main__":
    main()
