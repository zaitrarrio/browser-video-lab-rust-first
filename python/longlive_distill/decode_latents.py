"""Decode a latent tensor to an MP4 with the Wan VAE.

The missing last link. Nothing in the tree could turn a latent into pixels: the
browser runtime false-colours three of the sixteen latent channels
(`video-web/src/lib.rs`), which is a debug view, not a decode — no arrangement of
latent channels is an image. Without this step "does the student produce coherent
video" is not a question anyone can answer by looking.

Reads either `latents` (what `video-train sample` writes) from a safetensors file
or a `.pt` dataset item, undoes the DiT-space normalization the Wan pipeline
applies, decodes, and writes frames.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def denormalize(vae, latents: torch.Tensor) -> torch.Tensor:
    """Undo the per-channel normalization the Wan pipeline applies to latents.

    `WanPipeline.__call__` divides by `latents_std` and adds `latents_mean`
    immediately before `vae.decode`; a latent produced or predicted in the DiT's
    space must go through the same step or the decode is scaled wrong — which
    looks like washed-out or blown-out video rather than an obvious failure.
    """
    mean = torch.tensor(vae.config.latents_mean, dtype=latents.dtype, device=latents.device).view(1, -1, 1, 1, 1)
    inv_std = 1.0 / torch.tensor(vae.config.latents_std, dtype=latents.dtype, device=latents.device).view(1, -1, 1, 1, 1)
    return latents * inv_std + mean


def decode_to_video(vae, latents: torch.Tensor, output: Path, fps: int = 16) -> Path:
    device = next(vae.parameters()).device
    latents = latents.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        frames = vae.decode(denormalize(vae, latents), return_dict=False)[0]
    # [B, C, T, H, W] in [-1, 1] -> uint8 [T, H, W, C]
    video = ((frames[0].permute(1, 2, 3, 0).float().clamp(-1, 1) + 1) * 127.5).to(torch.uint8).cpu().numpy()

    import imageio.v3 as iio

    output.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(output, video, fps=fps, codec="libx264")
    # A still is worth having beside the clip: a video player will happily loop
    # eight frames of noise and look busy doing it.
    iio.imwrite(output.with_suffix(".png"), video[0])
    print(f"decoded {video.shape} -> {output}")
    return output


def load_latents(path: Path) -> torch.Tensor:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        tensors = load_file(path)
        for key in ("latents", "noisy_latents", "teacher_noise_pred"):
            if key in tensors:
                return tensors[key]
        raise SystemExit(f"{path} has none of latents/noisy_latents/teacher_noise_pred")
    return torch.load(path, map_location="cpu", weights_only=True)["latents"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--latents", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model-id", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    from diffusers import AutoencoderKLWan

    vae = AutoencoderKLWan.from_pretrained(a.model_id, subfolder="vae", torch_dtype=torch.float32).to(a.device).eval()
    decode_to_video(vae, load_latents(Path(a.latents)), Path(a.output), a.fps)


if __name__ == "__main__":
    main()
