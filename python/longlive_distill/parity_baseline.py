"""Trivial-predictor baselines for `video-train eval`.

A cosine of 0.6 against the teacher sounds like distillation is working. It is
not, on its own, evidence of anything: in flow matching the target velocity
`v = eps - x0` is *correlated with the model's own input* `x_s = (1-s)x0 + s*eps`,
strongly so at high sigma, where

    <x_s, v> = s*|eps|^2 - (1-s)*|x0|^2

is large and positive. A student that learned nothing but "echo your input"
therefore scores well above zero. Any reported parity number has to be read
against that floor, so this computes it from the same shards the student is
scored on.

Baselines, all requiring zero training:
  echo      predict the noisy input unchanged
  scaled    predict a*x_s with the single best a per shard (the least-squares
            projection — the strongest possible input-echoing predictor)
  mean      predict the mean teacher target over the eval shards (what a model
            that ignores its input entirely converges to)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.flatten() @ b.flatten()) / (a.norm() * b.norm()).clamp_min(1e-12))


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()

    root = Path(a.cache)
    manifest = json.loads((root / "manifest.json").read_text())
    shards = manifest["shards"][: a.limit or None]

    targets = [load_file(root / s)["teacher_noise_pred"] for s in shards]
    noisy = [load_file(root / s)["noisy_latents"] for s in shards]
    mean_target = torch.stack(targets).mean(0)

    rows = {"echo": [], "scaled": [], "mean": []}
    for x, t in zip(noisy, targets):
        rows["echo"].append((cosine(x, t), rel_l2(x, t)))
        alpha = float((x.flatten() @ t.flatten()) / x.flatten().pow(2).sum().clamp_min(1e-12))
        rows["scaled"].append((cosine(alpha * x, t), rel_l2(alpha * x, t)))
        rows["mean"].append((cosine(mean_target, t), rel_l2(mean_target, t)))

    print(f"{len(shards)} shards from {manifest['teacher']} ({manifest['scheduler']})")
    for name, vals in rows.items():
        cos = sum(v[0] for v in vals) / len(vals)
        rel = sum(v[1] for v in vals) / len(vals)
        print(f"  {name:7s} cosine={cos:+.4f}  rel_l2={rel:.4f}")
    print("A distilled student must clear the best of these to have learned anything at all.")


if __name__ == "__main__":
    main()
