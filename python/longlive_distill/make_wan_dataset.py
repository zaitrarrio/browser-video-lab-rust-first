"""Build dataset items whose latents are REAL, on-manifold Wan2.1 latents.

Why this exists
---------------
`make_dataset.py` writes `torch.randn` for `latents` in both of its modes, and
says so ("Latents are sampled noise here — the timestep sweep in `cache_teacher`
provides the supervision coverage"). That is enough to exercise shapes and
plumbing, and it is the reason no cache built so far could ever teach a student
to produce coherent video:

    x_t = (1-σ)·x₀ + σ·ε      with x₀ ~ N(0, I)

is pure noise for *every* σ. The teacher is then queried only at points far off
the video manifold, and the whole low-σ end of the trajectory — where a sampler
spends its last and most decisive steps, and where all the structure lives — is
never supervised at all. A student that matches the teacher perfectly on such a
cache still emits noise when sampled.

This module produces x₀ from the teacher's own generative distribution: run the
Wan2.1 pipeline to a finished latent and cache that. No video corpus is needed —
the teacher supplies its own data, which is the standard self-distillation
setup — and, as a side effect, every clip comes with a reference MP4 that the
student's own sample can later be compared against.

Emits, per clip:
  latents                 [1, 16, T', H/8, W/8]  real, in the DiT's normalized space
  prompt_embeds           [1, seq, 4096]         teacher conditioning (umT5-XXL)
  negative_prompt_embeds  [1, seq, 4096]         for CFG-guided cache targets
  student_prompt_embeds   [1, seq, 512]          umt5-small, browser-runnable
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

# Wan's own default negative prompt (the pipeline uses this when none is given).
DEFAULT_NEGATIVE = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, "
    "static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, "
    "extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, "
    "fused fingers, still picture, messy background, three legs, many people in the background, "
    "walking backwards"
)


def read_captions(path: Path, limit: int) -> list[str]:
    lines = [l.strip() for l in path.read_text().splitlines() if l.strip() and not l.lstrip().startswith("#")]
    if not lines:
        raise SystemExit(f"no captions in {path}")
    return lines[:limit] if limit else lines


def encode_student(captions: list[str], model_id: str, seq: int, width: int, device: str):
    """umt5-small embeddings — the encoder the browser can actually run."""
    from transformers import AutoTokenizer, T5EncoderModel

    tok = AutoTokenizer.from_pretrained(model_id)
    enc = T5EncoderModel.from_pretrained(model_id).to(device).eval().requires_grad_(False)
    out = []
    with torch.no_grad():
        for caption in captions:
            ids = tok(caption, padding="max_length", max_length=seq, truncation=True, return_tensors="pt").to(device)
            last = enc(**ids).last_hidden_state
            if last.shape[-1] != width:
                raise ValueError(f"{model_id} hidden width {last.shape[-1]} != expected {width}")
            out.append(last.float().cpu())
    del enc
    torch.cuda.empty_cache()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--captions", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model-id", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    p.add_argument("--student-encoder", default="google/umt5-small")
    p.add_argument("--student-width", type=int, default=512)
    p.add_argument("--student-seq", type=int, default=128)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=256)
    # (frames - 1) must be divisible by the VAE's temporal factor of 4.
    p.add_argument("--frames", type=int, default=13)
    p.add_argument("--steps", type=int, default=30, help="pipeline denoising steps per clip")
    p.add_argument("--guidance", type=float, default=5.0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--reference-dir", default="", help="also decode each clip to an MP4 here (the comparison target)")
    a = p.parse_args()

    if (a.frames - 1) % 4:
        raise SystemExit(f"--frames {a.frames}: (frames-1) must be divisible by 4 for the Wan VAE")

    from diffusers import AutoencoderKLWan, WanPipeline

    dtype = getattr(torch, a.dtype)
    captions = read_captions(Path(a.captions), a.limit)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)

    # The VAE stays fp32: Wan's own model card calls for it, and it is the piece
    # that decides whether a decoded frame is clean or banded.
    vae = AutoencoderKLWan.from_pretrained(a.model_id, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(a.model_id, vae=vae, torch_dtype=dtype).to(a.device)

    student = encode_student(captions, a.student_encoder, a.student_seq, a.student_width, a.device)

    manifest = []
    for i, caption in enumerate(captions):
        with torch.no_grad():
            pos, neg = pipe.encode_prompt(
                prompt=caption, negative_prompt=DEFAULT_NEGATIVE,
                do_classifier_free_guidance=True, device=a.device,
            )
            result = pipe(
                prompt=caption, negative_prompt=DEFAULT_NEGATIVE,
                height=a.height, width=a.width, num_frames=a.frames,
                num_inference_steps=a.steps, guidance_scale=a.guidance,
                generator=torch.Generator(device=a.device).manual_seed(a.seed + i),
                output_type="latent",
            )
        # `output_type="latent"` short-circuits the decode and hands back the
        # pipeline's working latent — already in the DiT's normalized space, which
        # is the space the student must learn. decode_latents.py undoes it.
        latents = result.frames if hasattr(result, "frames") else result[0]
        latents = latents.float().cpu()

        torch.save(
            {
                "latents": latents,
                "prompt_embeds": pos.float().cpu(),
                "negative_prompt_embeds": neg.float().cpu(),
                "student_prompt_embeds": student[i],
                "caption": caption,
            },
            out / f"clip-{i:05d}.pt",
        )
        manifest.append({"index": i, "caption": caption, "latent_shape": list(latents.shape)})
        print(f"[{i + 1}/{len(captions)}] {tuple(latents.shape)}  {caption[:70]}", flush=True)

        if a.reference_dir:
            from .decode_latents import decode_to_video

            ref = Path(a.reference_dir)
            ref.mkdir(parents=True, exist_ok=True)
            decode_to_video(vae, latents, ref / f"clip-{i:05d}.mp4")

    (out / "items.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {len(manifest)} real-latent items to {out}")


if __name__ == "__main__":
    main()
