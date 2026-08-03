"""Where in the spatial spectrum does the student's velocity go wrong?

The error-budget test showed the student is *worse* than the teacher's own
velocity degraded with random noise to the same cosine — so its error is not
isotropic. A cosine of 0.755 that is spent entirely on low spatial frequencies
and misses the high ones would look exactly like what round 5 decoded: correct
palette, no geometry.
"""
import sys
import torch
from safetensors.torch import load_file

shard, recon = sys.argv[1], sys.argv[2]
t = load_file(shard)
x = t["noisy_latents"].float()
s = float(t["timestep"].float().reshape(-1)[0]) / 1000.0
vt = t["teacher_noise_pred"].float()
vs = (x - load_file(recon)["latents"].float()) / s


def band(v, lo, hi):
    """Keep an annulus of radial spatial frequency, per (frame, channel)."""
    z = v[0].permute(1, 0, 2, 3)                        # [T,C,H,W]
    F = torch.fft.rfft2(z)
    fy = torch.fft.fftfreq(z.shape[-2])[:, None].abs()
    fx = torch.fft.rfftfreq(z.shape[-1])[None, :].abs()
    r = (fy ** 2 + fx ** 2).sqrt() / 0.7071             # normalise to [0,1]
    return torch.fft.irfft2(F * ((r >= lo) & (r < hi)).to(F.dtype), s=z.shape[-2:])


print(f"sigma={s:.3f}  student vs teacher velocity, by spatial frequency")
print(f"{'band':>12} {'cosine':>9} {'teacher energy':>16} {'|vs|/|vt|':>11}")
tot = float((vt ** 2).sum())
for lo, hi in ((0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 1.01)):
    a, b = band(vs, lo, hi), band(vt, lo, hi)
    cos = float((a * b).sum() / (a.norm() * b.norm() + 1e-12))
    print(f"  {lo:.2f}-{hi:.2f} {cos:9.3f} {float((b ** 2).sum()) / tot:15.1%} {float(a.norm() / b.norm()):11.3f}")
