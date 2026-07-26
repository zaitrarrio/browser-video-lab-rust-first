"""Build the `.pt` dataset that `cache_teacher` consumes.

Each item carries `latents`, `prompt_embeds` (teacher width), and
`student_prompt_embeds` (umt5-small, 512) for the SAME caption — the coupling the
umt5 config documents, so the Burn student conditions on a browser-runnable
encoder while the teacher keeps its own. Two modes:

  --synthetic : sampled latents + random embeddings, no downloads — exercises the
                whole make-dataset → cache → train path on CPU, like the toy
                teacher. This is what the smoke task and the pytest use.
  (default)   : encode a captions file with the teacher encoder (umT5-XXL, 4096)
                and the student encoder (umt5-small, 512) via `transformers`.
                CONFIRM the encoder ids against your Wan2.1 checkpoint; a width
                mismatch asserts rather than silently miscaching.

Latents are sampled noise here — the timestep sweep in `cache_teacher` provides
the supervision coverage (TEACHER-OPTIONS.md). Swap in real VAE latents from the
Diffusers pipeline when you have them.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import torch


def _save(out: Path, i: int, latents, prompt_embeds, student_prompt_embeds):
    torch.save(
        {"latents": latents, "prompt_embeds": prompt_embeds, "student_prompt_embeds": student_prompt_embeds},
        out / f"clip-{i:05d}.pt",
    )


def synthetic(a):
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    c, t, h, w = a.latent_shape
    for i in range(a.count):
        g = torch.Generator().manual_seed(a.seed + i)
        _save(out, i,
              torch.randn(1, c, t, h, w, generator=g),
              torch.randn(1, a.seq, a.teacher_width, generator=g),
              torch.randn(1, a.seq, a.student_width, generator=g))
    print(f"wrote {a.count} synthetic items to {out}")


def encode(a):
    from transformers import AutoTokenizer, T5EncoderModel  # heavy; only the real path needs it

    captions = [line.strip() for line in Path(a.captions).read_text().splitlines() if line.strip()]

    def run(model_id: str, width: int):
        tok = AutoTokenizer.from_pretrained(model_id)
        model = T5EncoderModel.from_pretrained(model_id).eval().requires_grad_(False)
        embeds = []
        with torch.no_grad():
            for caption in captions:
                ids = tok(caption, padding="max_length", max_length=a.seq, truncation=True, return_tensors="pt")
                last = model(**ids).last_hidden_state  # [1, seq, width]
                if last.shape[-1] != width:
                    raise ValueError(f"{model_id} hidden width {last.shape[-1]} != expected {width}")
                embeds.append(last)
        return embeds

    teacher = run(a.teacher_encoder, a.teacher_width)
    student = run(a.student_encoder, a.student_width)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    c, t, h, w = a.latent_shape
    for i, (te, se) in enumerate(zip(teacher, student)):
        g = torch.Generator().manual_seed(a.seed + i)
        _save(out, i, torch.randn(1, c, t, h, w, generator=g), te, se)
    print(f"encoded {len(teacher)} captions to {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--captions", help="one caption per line (real-encoder mode)")
    p.add_argument("--synthetic", action="store_true", help="random embeddings, no downloads")
    p.add_argument("--count", type=int, default=8, help="synthetic items to emit")
    p.add_argument("--seq", type=int, default=128)
    p.add_argument("--teacher-width", type=int, default=4096)
    p.add_argument("--student-width", type=int, default=512)
    p.add_argument("--teacher-encoder", default="google/umt5-xxl")
    p.add_argument("--student-encoder", default="google/umt5-small")
    p.add_argument("--latent-shape", type=int, nargs=4, default=[16, 4, 32, 48], metavar=("C", "T", "H", "W"))
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    (synthetic if a.synthetic or not a.captions else encode)(a)


if __name__ == "__main__":
    main()
