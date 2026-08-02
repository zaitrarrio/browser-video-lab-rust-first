"""CPU contract test for the Plan B teacher cache.

Exercises make_dataset (synthetic) -> cache_teacher (toy Wan2.1 teacher) and
asserts the shards match the `video-contract` the Burn trainer reads: required
tensors, int64 timesteps, post-patchify relation grams, the layer cap, and the
per-clip draw multiplication. No GPU, no downloads.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[2]
ENV = {**os.environ, "PYTHONPATH": str(ROOT / "python")}


def _run(module: str, *args: str):
    subprocess.run([sys.executable, "-m", module, *args], cwd=ROOT, env=ENV, check=True)


def test_synthetic_dataset_then_toy_cache(tmp_path):
    data, cache = tmp_path / "data", tmp_path / "cache"
    # 16-channel, 2×4×4 latent → post-patch tokens = 2·(4/2)·(4/2) = 8.
    _run("longlive_distill.make_dataset", "--synthetic", "--output", str(data),
         "--count", "3", "--seq", "4", "--teacher-width", "4096", "--student-width", "16",
         "--latent-shape", "16", "2", "4", "4")
    assert len(list(data.glob("*.pt"))) == 3

    _run("longlive_distill.cache_teacher",
         "--adapter", "wan21_teacher_adapter:build_teacher", "--adapter-arg", "toy=true",
         "--dataset", str(data), "--output", str(cache),
         "--draws-per-clip", "2", "--relation-layers", "3")

    manifest = json.loads((cache / "manifest.json").read_text())
    assert manifest["format_version"] == 1
    assert len(manifest["shards"]) == 3 * 2, "clips × draws"
    assert manifest["hidden_relation_layers"] == [0, 1, 2]

    shard = load_file(cache / manifest["shards"][0])
    for key in ("noisy_latents", "timestep", "prompt_embeds", "teacher_noise_pred", "student_prompt_embeds"):
        assert key in shard, f"missing {key}"
    assert shard["timestep"].dtype == torch.int64
    assert tuple(shard["noisy_latents"].shape) == (1, 16, 2, 4, 4)
    assert tuple(shard["teacher_noise_pred"].shape) == (1, 16, 2, 4, 4)
    assert tuple(shard["student_prompt_embeds"].shape) == (1, 4, 16)
    # Relation grams at the post-patchify token count (8), capped at 3 layers.
    assert tuple(shard["teacher_relation.0"].shape) == (1, 8, 8)
    assert "teacher_relation.3" not in shard


def test_draw_batching_reproduces_the_unbatched_cache(tmp_path):
    """`--draw-batch` must be a scheduling change, not a data change.

    Batching draws into one teacher forward is worth ~18% (docs/PERF-ROUND-1.md),
    but it rewrites the loop that decides what every shard contains, and the failure
    mode is silent: a cache whose inputs no longer match its seeds still trains, just
    to the wrong thing. The per-draw generator is seeded from (clip, draw), so the
    inputs must come out bit-identical however they are grouped; only the teacher's
    own output is allowed to move, and only by GEMM reduction order.
    """
    data = tmp_path / "data"
    _run("longlive_distill.make_dataset", "--synthetic", "--output", str(data),
         "--count", "2", "--seq", "4", "--teacher-width", "4096", "--student-width", "16",
         "--latent-shape", "16", "2", "4", "4")

    caches = {}
    for batch in (1, 8):
        out = tmp_path / f"cache-b{batch}"
        _run("longlive_distill.cache_teacher",
             "--adapter", "wan21_teacher_adapter:build_teacher", "--adapter-arg", "toy=true",
             "--dataset", str(data), "--output", str(out),
             "--draws-per-clip", "8", "--draw-batch", str(batch), "--relation-layers", "2")
        caches[batch] = (out, json.loads((out / "manifest.json").read_text()))

    (dir1, m1), (dir8, m8) = caches[1], caches[8]
    assert m1["shards"] == m8["shards"], "batching must not change shard count or order"
    assert m1["scheduler"] == m8["scheduler"]

    for name in m1["shards"]:
        a, b = load_file(dir1 / name), load_file(dir8 / name)
        assert a.keys() == b.keys(), f"{name}: different tensors"
        # Inputs are built on the CPU from the seeded generator and must be exact —
        # if these ever differ, the cache is being fed something other than what its
        # seed says, which no tolerance should paper over.
        for key in ("noisy_latents", "timestep", "prompt_embeds"):
            assert torch.equal(a[key], b[key]), f"{name}/{key} changed under batching"
        # The teacher's output may move at reduction-order level only.
        assert torch.allclose(a["teacher_noise_pred"], b["teacher_noise_pred"], rtol=1e-4, atol=1e-5), \
            f"{name}: teacher_noise_pred moved more than reduction order explains"
        for key in (k for k in a if k.startswith("teacher_relation.")):
            assert torch.allclose(a[key].float(), b[key].float(), rtol=1e-3, atol=1e-3), f"{name}/{key}"


def test_draws_vary_the_timestep(tmp_path):
    data, cache = tmp_path / "data", tmp_path / "cache"
    _run("longlive_distill.make_dataset", "--synthetic", "--output", str(data),
         "--count", "1", "--seq", "4", "--teacher-width", "4096", "--student-width", "16",
         "--latent-shape", "16", "2", "4", "4")
    _run("longlive_distill.cache_teacher",
         "--adapter", "wan21_teacher_adapter:build_teacher", "--adapter-arg", "toy=true",
         "--dataset", str(data), "--output", str(cache), "--draws-per-clip", "4", "--relation-layers", "2")
    manifest = json.loads((cache / "manifest.json").read_text())
    assert len(manifest["shards"]) == 4
    steps = {int(load_file(cache / s)["timestep"].item()) for s in manifest["shards"]}
    assert len(steps) > 1, "draws must vary the (noise, timestep) draw"
