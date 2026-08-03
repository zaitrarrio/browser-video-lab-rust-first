"""Where does the training signal actually live? A cache analysis, no GPU, no model.

`VALIDATION-ROUND-5.md` localises the student's failure to high sigma and offers
two explanations, capacity and conditioning, each of which costs hours of GPU to
test. There is a third candidate that costs nothing to check, because it is a
property of the *targets* rather than of the model:

    x_s = (1-s)*x0 + s*eps        v = eps - x0

The trainer minimises unweighted MSE on `v`. As `s -> 1` the input `x_s -> eps`,
so `v -> x_s/s - x0*(1 + 1/s)`: most of the target is recoverable from the input
by scaling, and only a residual actually requires knowing which clip this is. If
that residual is a small fraction of the norm at high sigma, then MSE — which
weights by the square — spends nearly all of its gradient budget on a component
the model can fit without ever learning any structure, and a model that ignores
structure at high sigma is doing exactly what the loss asked.

Two numbers per sigma band, and neither needs a forward pass:

* `echo_resid` — ||v - a*x_s|| / ||v|| for the least-squares `a`. The share of the
  target that scaling the input cannot explain. This is the *only* part that can
  carry clip identity, and `parity_baseline.py`'s "scaled" floor is its aggregate.
* `mse_share` — that band's share of total sum-of-squares over the cache, i.e. of
  the gradient budget. A band with a large share and a small `echo_resid` is one
  the trainer works hard on and learns nothing structural from.

Run it on the training cache (built at `--shift 3.0`, which deliberately
concentrates draws at high sigma) and on a uniform one to see how much the warp
moves the budget.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open


def shard_stats(path: Path) -> dict:
    # Only two tensors are read. The relation grams dominate shard size (~10 MB at
    # 1600 tokens) and are irrelevant here, so `safe_open` rather than `load_file`
    # turns a 19 GB scan into a fraction of that.
    with safe_open(str(path), framework="np") as f:
        x = f.get_tensor("noisy_latents").astype(np.float64).ravel()
        v = f.get_tensor("teacher_noise_pred").astype(np.float64).ravel()
        t = f.get_tensor("timestep").astype(np.float64).ravel()[0]
    xx = float(x @ x)
    a = float(v @ x) / max(xx, 1e-12)          # least-squares scale
    r = v - a * x
    vn = float(np.sqrt(v @ v))
    return {
        "sigma": t / 1000.0,
        "v_sq": float(v @ v),
        "v_norm": vn,
        "echo_resid": float(np.sqrt(r @ r)) / max(vn, 1e-12),
        "cos_v_x": float(v @ x) / max(np.sqrt(xx) * vn, 1e-12),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--limit", type=int, default=0, help="0 = every shard")
    p.add_argument("--bands", type=int, default=5)
    a = p.parse_args()

    root = Path(a.cache)
    manifest = json.loads((root / "manifest.json").read_text())
    shards = manifest["shards"]
    if a.limit:
        shards = shards[:: max(1, len(shards) // a.limit)][: a.limit]
    print(f"{len(shards)} shards from {manifest.get('teacher')} ({manifest.get('scheduler')})")

    rows = [shard_stats(root / s) for s in shards]
    sig = np.array([r["sigma"] for r in rows])
    vsq = np.array([r["v_sq"] for r in rows])
    res = np.array([r["echo_resid"] for r in rows])
    total = vsq.sum()

    edges = np.linspace(0.0, 1.0, a.bands + 1)
    print(f"\n{'sigma band':>12} {'n':>5} {'mse_share':>10} {'echo_resid':>11} {'cos(v,x)':>9}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (sig >= lo) & (sig < hi if hi < 1.0 else sig <= 1.0)
        if not m.any():
            continue
        cos = np.mean([r["cos_v_x"] for r, k in zip(rows, m) if k])
        print(f"  {lo:.2f}-{hi:.2f} {m.sum():7d} {vsq[m].sum()/total:9.1%} "
              f"{res[m].mean():10.3f} {cos:9.3f}")

    print(f"\n{'':>12} {'total':>5} {'100.0%':>10} {res.mean():10.3f}")
    hi = sig >= 0.5
    if hi.any():
        print(f"\nsigma >= 0.5 holds {hi.mean():.1%} of shards and {vsq[hi].sum()/total:.1%} of the "
              f"gradient budget, at mean echo-residual {res[hi].mean():.3f}")
    lo = sig < 0.5
    if lo.any():
        print(f"sigma <  0.5 holds {lo.mean():.1%} of shards and {vsq[lo].sum()/total:.1%} of the "
              f"gradient budget, at mean echo-residual {res[lo].mean():.3f}")


if __name__ == "__main__":
    main()
