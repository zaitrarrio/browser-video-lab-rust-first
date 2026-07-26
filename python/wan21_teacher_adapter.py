"""Wan2.1-T2V-1.3B teacher adapter (Plan B).

`TEACHER-OPTIONS.md` Plan B distils the browser student from `Wan-AI/Wan2.1-T2V-1.3B`
directly (Apache-2.0, `diffusers`-native, CPU-plausible) instead of LongLive. This
module builds a frozen teacher matching the `ADAPTER.md` contract:

    forward(noisy_latents, timestep, prompt_embeds) -> {
        "noise_pred":     Tensor[B, C, T, H, W],
        "hidden_states":  list[Tensor[B, N, D]],   # post-patchify tokens
    }

The teacher's DiT patchifies with `patch_size [1,2,2]`, so its hidden states — and
therefore the relation grams cached from them — are at the SAME post-patch token
count as the patchified Burn student (`video-student`), which is what lets the
width-independent relation loss line up. See `rust/DELIVERY.md`.

Two builders:

* `build_teacher({"toy": True, ...})` — a tiny CPU module that patchifies exactly
  like the real DiT and the Burn student. It exercises the whole cache/adapter/
  contract path with no download and no GPU (the pattern `ADAPTER.md` recommends).
* `build_teacher({...})` — loads the real Diffusers transformer and hooks its
  blocks. The `diffusers` call surface is annotated where it must be confirmed
  against the pinned release; a shape mismatch raises rather than corrupts a cache.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn


def _patchify(x: torch.Tensor, patch: tuple[int, int, int]) -> torch.Tensor:
    """[B,C,T,H,W] -> [B, T*(H/ph)*(W/pw), C*ph*pw] for temporal patch 1.

    Mirrors `BrowserVideoStudent::forward` (token order t, h/ph, w/pw; channel
    group c, ph, pw) so the two token sequences correspond position-for-position.
    """
    b, c, t, h, w = x.shape
    pt, ph, pw = patch
    if pt != 1:
        raise ValueError("temporal patch must be 1 (matches the student)")
    if h % ph or w % pw:
        raise ValueError(f"latent {h}x{w} not divisible by patch {ph}x{pw}")
    x = x.reshape(b, c, t, h // ph, ph, w // pw, pw)
    x = x.permute(0, 2, 3, 5, 1, 4, 6)  # [b, t, h/ph, w/pw, c, ph, pw]
    return x.reshape(b, t * (h // ph) * (w // pw), c * ph * pw)


def _unpatchify(x: torch.Tensor, shape: tuple[int, int, int, int, int], patch: tuple[int, int, int]) -> torch.Tensor:
    """Inverse of `_patchify`, back to [B,C,T,H,W]."""
    b, c, t, h, w = shape
    _, ph, pw = patch
    x = x.reshape(b, t, h // ph, w // pw, c, ph, pw)
    x = x.permute(0, 4, 1, 2, 5, 3, 6)  # [b, c, t, h/ph, ph, w/pw, pw]
    return x.reshape(b, c, t, h, w)


class ToyWanTeacher(nn.Module):
    """Deterministic CPU stand-in for the Wan2.1 DiT with the real token geometry.

    Not a quality teacher — it validates the cache contract (post-patch grams,
    shapes, dtypes) end to end without weights, exactly as `ADAPTER.md`'s toy
    teacher does for Plan A.
    """

    def __init__(self, latent_channels: int = 16, hidden: int = 64, layers: int = 4,
                 text_width: int = 4096, patch: tuple[int, int, int] = (1, 2, 2)):
        super().__init__()
        self.patch = patch
        token_dim = latent_channels * patch[0] * patch[1] * patch[2]
        self.proj_in = nn.Linear(token_dim, hidden)
        self.text = nn.Linear(text_width, hidden)
        self.time = nn.Linear(1, hidden)
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden))
            for _ in range(layers)
        )
        self.norm = nn.LayerNorm(hidden)
        self.proj_out = nn.Linear(hidden, token_dim)

    def forward(self, noisy_latents: torch.Tensor, timestep: torch.Tensor, prompt_embeds: torch.Tensor) -> dict[str, Any]:
        shape = noisy_latents.shape
        x = _patchify(noisy_latents, self.patch)
        cond = self.text(prompt_embeds.float().mean(1)).unsqueeze(1) + self.time(timestep.reshape(-1, 1).float() / 1000).unsqueeze(1)
        x = self.proj_in(x) + cond
        hidden: list[torch.Tensor] = []
        for block in self.blocks:
            x = x + block(x)
            hidden.append(x)
        y = _unpatchify(self.proj_out(self.norm(x)), shape, self.patch)
        return {"noise_pred": y, "hidden_states": hidden}


class Wan21Teacher(nn.Module):
    """Frozen wrapper around the real Diffusers Wan2.1 transformer.

    Hooks every transformer block to capture its output hidden states — the
    wrapper returns a `sample`, not intermediates, so they must be captured, not
    read. CONFIRM against the pinned `diffusers` (requirements-wan.txt): the class
    name (`WanTransformer3DModel`), the `.blocks` attribute, and the forward
    kwargs (`hidden_states`, `timestep`, `encoder_hidden_states`). A wrong
    assumption trips the shape check in `forward` rather than silently caching
    garbage.
    """

    def __init__(self, model_id: str = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", dtype: torch.dtype = torch.float32,
                 device: str = "cpu", subfolder: str = "transformer"):
        super().__init__()
        from diffusers import WanTransformer3DModel  # local import: only the real path needs diffusers

        self.transformer = WanTransformer3DModel.from_pretrained(model_id, subfolder=subfolder, torch_dtype=dtype)
        self.transformer.eval().requires_grad_(False)
        self.transformer.to(device)
        self._captured: list[torch.Tensor] = []
        blocks = getattr(self.transformer, "blocks", None)
        if blocks is None:
            raise AttributeError("expected `.blocks` on the Wan transformer — confirm the diffusers class")
        for block in blocks:
            block.register_forward_hook(self._hook)

    def _hook(self, _module, _inputs, output):
        tensor = output[0] if isinstance(output, (tuple, list)) else output
        self._captured.append(tensor)

    @torch.no_grad()
    def forward(self, noisy_latents: torch.Tensor, timestep: torch.Tensor, prompt_embeds: torch.Tensor) -> dict[str, Any]:
        self._captured = []
        out = self.transformer(
            hidden_states=noisy_latents,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            return_dict=True,
        )
        noise_pred = getattr(out, "sample", None)
        if noise_pred is None:
            noise_pred = out[0] if isinstance(out, (tuple, list)) else out
        if noise_pred.shape != noisy_latents.shape:
            raise RuntimeError(
                f"teacher noise_pred {tuple(noise_pred.shape)} != input {tuple(noisy_latents.shape)}; "
                "the diffusers forward signature likely differs from the assumed one"
            )
        if not self._captured:
            raise RuntimeError("no hidden states captured — confirm the block hook target")
        return {"noise_pred": noise_pred, "hidden_states": list(self._captured)}


def build_teacher(cfg: dict[str, Any] | None = None) -> nn.Module:
    """Return a frozen teacher. `cfg={"toy": True}` builds the CPU stand-in;
    otherwise the real Wan2.1 transformer (needs the weights + `diffusers`)."""
    cfg = dict(cfg or {})
    if cfg.pop("toy", False):
        return ToyWanTeacher(**cfg).eval().requires_grad_(False)
    return Wan21Teacher(**cfg).eval()
