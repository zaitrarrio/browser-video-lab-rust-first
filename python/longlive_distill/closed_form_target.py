"""Is the overfit task even the task we think it is? A free check on the target.

For a memorisation probe the flow-matching objective has a **closed form**. From

    x_s = (1-s)*x0 + s*eps        v = eps - x0

eliminate eps:

    v = (x_s - x0) / s

So a student that has memorised `x0` for a prompt does not need to learn a
denoiser at all — it needs to store four latents, route them by prompt, and
compute a scaled difference against its own input. That is a trivial function for
a 79M model, and `VALIDATION-ROUND-5.md` shows it failing at high sigma after
212k views, which is only sensible if the target is *not* that closed form.

The suspect is classifier-free guidance. The cache is built with `--guidance 5.0`,
so `teacher_noise_pred` stores

    v_uncond + 5*(v_cond - v_uncond)

which is a legitimate *sampling* field but is not the probability-flow velocity,
and does not satisfy `x0 = x_s - s*v` except approximately. This script measures
exactly how far the stored target sits from the closed form, per sigma band, by
pairing each shard with the true `x0` from the dataset item it was drawn from.

If the cosine is near 1, the target is essentially the closed form and the
student's failure is optimisation or architecture. If it is not, the student has
been asked to fit a field that no memorised-`x0` shortcut can produce — and the
guidance setting, not the model, is the thing to change first.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--dataset", required=True, help="dir of clip-*.pt the cache was built from")
    p.add_argument("--draws-per-clip", type=int, required=True, help="cache is clip-major")
    p.add_argument("--limit", type=int, default=200)
    a = p.parse_args()

    root = Path(a.cache)
    shards = json.loads((root / "manifest.json").read_text())["shards"]
    clips = [torch.load(c, map_location="cpu", weights_only=True)["latents"].double().numpy().ravel()
             for c in sorted(Path(a.dataset).glob("clip-*.pt"))]
    print(f"{len(clips)} clips, {len(shards)} shards, {a.draws_per_clip} draws/clip")

    step = max(1, len(shards) // a.limit)
    rows = []
    for i in range(0, len(shards), step):
        with safe_open(str(root / shards[i]), framework="np") as f:
            x = f.get_tensor("noisy_latents").astype(np.float64).ravel()
            v = f.get_tensor("teacher_noise_pred").astype(np.float64).ravel()
            s = float(f.get_tensor("timestep").astype(np.float64).ravel()[0]) / 1000.0
        x0 = clips[i // a.draws_per_clip]
        if s < 1e-6:
            continue
        ideal = (x - x0) / s
        cos = float(v @ ideal) / max(np.linalg.norm(v) * np.linalg.norm(ideal), 1e-12)
        rows.append((s, cos, float(np.linalg.norm(v) / max(np.linalg.norm(ideal), 1e-12))))

    rows.sort()
    sig = np.array([r[0] for r in rows]); cos = np.array([r[1] for r in rows]); rat = np.array([r[2] for r in rows])
    print(f"\n{'sigma band':>12} {'n':>5} {'cos(stored, closed-form)':>26} {'norm ratio':>12}")
    for lo, hi in zip(np.linspace(0, 1, 6)[:-1], np.linspace(0, 1, 6)[1:]):
        m = (sig >= lo) & (sig < hi if hi < 1.0 else sig <= 1.0)
        if m.any():
            print(f"  {lo:.2f}-{hi:.2f} {m.sum():7d} {cos[m].mean():25.4f} {rat[m].mean():12.3f}")
    print(f"\n{'':>12} {len(rows):5d} {cos.mean():25.4f} {rat.mean():12.3f}   <- overall")
    print("\ncos ~ 1.0  => target IS the closed form; a memorised x0 suffices and the")
    print("              failure is optimisation/architecture.")
    print("cos << 1.0 => guidance has moved the target off the probability flow, and")
    print("              no memorised-x0 shortcut can produce it.")


if __name__ == "__main__":
    main()
