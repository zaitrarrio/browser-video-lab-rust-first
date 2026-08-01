"""Score and render a student sample against the teacher clips it was trained on.

Two questions, answered separately, because they fail independently:

1. **Did the sampler land anywhere?** A student whose velocity field integrates
   to a latent with roughly the right statistics (per-channel scale close to the
   teacher's) has learned the field. One that integrates to something an order of
   magnitude off, or to a near-constant, has not — and that shows up here before
   any decode, where it is unambiguous.
2. **Did it land on a clip?** Cosine of the sampled latent against each teacher
   clip. An overfit probe on N clips should peak clearly on one of them; a flat
   profile across all N means the prompt is not steering the sample.

Then it decodes the sample and the best-matching reference into one side-by-side
MP4, which is the artefact a person can actually judge.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file


def load_sample(path: Path) -> torch.Tensor:
    return load_file(path)["latents"].float()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sample", required=True, help="latents.safetensors from `video-train sample`")
    p.add_argument("--dataset", required=True, help="dir of clip-*.pt written by make_wan_dataset")
    p.add_argument("--output", required=True, help="side-by-side MP4")
    p.add_argument("--model-id", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    p.add_argument("--device", default="cuda")
    p.add_argument("--fps", type=int, default=16)
    a = p.parse_args()

    sample = load_sample(Path(a.sample))
    clips = sorted(Path(a.dataset).glob("clip-*.pt"))
    refs = [torch.load(c, map_location="cpu", weights_only=True) for c in clips]

    print(f"sample  shape={tuple(sample.shape)} mean={sample.mean():+.4f} std={sample.std():.4f} "
          f"min={sample.min():+.3f} max={sample.max():+.3f}")
    ref_std = torch.stack([r["latents"] for r in refs]).std()
    print(f"teacher clips std={ref_std:.4f}  (a sample far off this has not learned the field's scale)")

    scores = []
    for c, r in zip(clips, refs):
        lat = r["latents"].float()
        cos = float((sample.flatten() @ lat.flatten()) / (sample.norm() * lat.norm()).clamp_min(1e-12))
        scores.append((cos, c, lat, r.get("caption", "")))
        print(f"  cos={cos:+.4f}  {c.name}  {r.get('caption','')[:60]}")
    scores.sort(key=lambda s: -s[0])
    best = scores[0]
    spread = best[0] - scores[-1][0]
    print(f"best={best[1].name} cos={best[0]:+.4f}  spread over clips={spread:+.4f}")

    from diffusers import AutoencoderKLWan

    from .decode_latents import denormalize

    vae = AutoencoderKLWan.from_pretrained(a.model_id, subfolder="vae", torch_dtype=torch.float32).to(a.device).eval()

    def frames(latents: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            out = vae.decode(denormalize(vae, latents.to(a.device)), return_dict=False)[0]
        return ((out[0].permute(1, 2, 3, 0).float().clamp(-1, 1) + 1) * 127.5).to(torch.uint8).cpu().numpy()

    left, right = frames(sample), frames(best[2])
    n = min(len(left), len(right))
    grid = np.concatenate([left[:n], right[:n]], axis=2)  # student | nearest teacher clip

    import imageio.v3 as iio

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out, grid, fps=a.fps, codec="libx264")
    iio.imwrite(out.with_suffix(".png"), np.concatenate(list(grid[:: max(1, n // 4)]), axis=0))
    print(f"wrote {out} (left: student, right: {best[1].name})")


if __name__ == "__main__":
    main()
