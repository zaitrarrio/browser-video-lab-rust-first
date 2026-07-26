"""Build the teacher cache once, headlessly, as a Kaggle CPU script kernel.

This is the CPU sibling of `run_chunk.py`: where that trains on the free GPU, this
produces the one input that pipeline still needed by hand — the framework-neutral
teacher-cache dataset the trainer mounts. Pushed by `scripts/kaggle-cache.mjs`,
which replaces the CONFIG literal below before every push.

The teacher pass is inference-only (no backprop, no optimizer), so it runs on
Kaggle's always-available CPU tier rather than competing for the GPU quota — which
is why Plan B (Wan2.1-1.3B, Apache-2.0) is the teacher here. Steps:

  encode captions ─▶ Wan2.1 teacher forward ─▶ safetensors shards ─▶ version back

Big downloads (the ~17.5 GB Wan2.1 checkpoint, the umT5-XXL encoder) go to /tmp,
which Kaggle sizes at ~1.1 TB — /kaggle/working is only ~21 GB and is reserved for
the shards that get versioned back as the `…-teacher-cache` dataset.
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

CONFIG = json.loads(r"""{{CONFIG}}""")

WORKING = Path("/kaggle/working")
SCRATCH = Path("/tmp/teacher-build")
REPO = SCRATCH / "repo"
DATA = SCRATCH / "wan-latents"
CACHE = WORKING / "teacher-cache"
HF_HOME = SCRATCH / "hf"
started = time.time()


def sh(cmd, cwd=None, env=None, check=True):
    print(f"$ {cmd}", flush=True)
    merged = {**os.environ, **(env or {})}
    return subprocess.run(cmd, shell=True, cwd=cwd, env=merged, check=check)


def require_internet():
    """Fail immediately, and legibly, on a kernel with no network.

    Cloning, pip, the HF hub, and the dataset push all need it. Kaggle honours
    `enable_internet` only on a phone-verified account, yet reports it back as
    true regardless — so the request succeeding proves nothing, and the first
    real symptom is otherwise ~200 s of DNS retries. (Same guard as run_chunk.py.)
    """
    import socket

    try:
        socket.getaddrinfo("huggingface.co", 443)
    except socket.gaierror as exc:
        raise SystemExit(
            f"this kernel has no internet ({exc}).\n"
            "The metadata requested it and Kaggle accepted, but the setting only applies "
            "to a phone-verified account: kaggle.com → Settings → Phone Verification, then "
            "re-run. Without it the kernel cannot clone, install, download the teacher, or push."
        )


def authenticate():
    """Pin the CLI and prove the injected `KGAT_…` token before doing real work."""
    sh(f"pip install --quiet --upgrade kaggle=={CONFIG['kaggle_cli_version']}")
    sh("kaggle --version", check=False)
    token = CONFIG["api_token"]
    os.environ["KAGGLE_API_TOKEN"] = token
    print(f"KAGGLE_API_TOKEN injected ({len(token)} chars)", flush=True)
    probe = subprocess.run(
        "kaggle kernels list --mine --page-size 1", shell=True, capture_output=True, text=True
    )
    if probe.returncode != 0:
        raise SystemExit(f"KAGGLE_API_TOKEN rejected by the CLI:\n{probe.stderr.strip()}")


def push_dataset(folder: Path, slug: str, title: str, message: str):
    """Create-or-version `slug` from `folder`. Idempotent across both cases."""
    (folder / "dataset-metadata.json").write_text(
        json.dumps({"title": title, "id": slug, "licenses": [{"name": "CC0-1.0"}]}, indent=2)
    )
    versioned = subprocess.run(
        f'kaggle datasets version -p "{folder}" -m "{message}" --dir-mode zip',
        shell=True, capture_output=True, text=True,
    )
    print(versioned.stdout or "", versioned.stderr or "", flush=True)
    if versioned.returncode == 0:
        return
    print(f"version failed for {slug}; attempting first-time create", flush=True)
    sh(f'kaggle datasets create -p "{folder}" --dir-mode zip')


def checkout_repo():
    sh(f'git clone --filter=blob:none "{CONFIG["repo_url"]}" "{REPO}"')
    sh(f'git checkout --detach {CONFIG["commit"]}', cwd=REPO)


def install_deps():
    """Only the teacher pass's needs — torch ships in the Kaggle image already.

    Deliberately not the full requirements-wan.txt: that pulls Pruna (the native
    CUDA compressor), which this CPU cache job never touches.
    """
    sh('pip install --quiet "diffusers>=0.33" "transformers>=4.49" accelerate safetensors ftfy')


def build_dataset() -> Path:
    """Encode captions to the .pt items cache_teacher reads (both encoder widths)."""
    captions = REPO / CONFIG["captions_file"]
    if not captions.exists():
        raise SystemExit(f"captions file {captions} not found in the repo at {CONFIG['commit'][:8]}")
    c, t, h, w = CONFIG["latent_shape"]
    env = {"PYTHONPATH": "python", "HF_HOME": str(HF_HOME)}
    sh(
        f'python -m longlive_distill.make_dataset --captions "{captions}" --output "{DATA}" '
        f'--latent-shape {c} {t} {h} {w} --seq {CONFIG["seq"]} '
        f'--teacher-width {CONFIG["teacher_width"]} --student-width {CONFIG["student_width"]} '
        f'--teacher-encoder {CONFIG["teacher_encoder"]} --student-encoder {CONFIG["student_encoder"]} '
        f'--limit {CONFIG["limit"]}',
        cwd=REPO, env=env,
    )
    return DATA


def build_cache(dataset: Path) -> Path:
    """Run the frozen Wan2.1 teacher over every (clip, draw) into safetensors shards."""
    env = {"PYTHONPATH": "python", "HF_HOME": str(HF_HOME)}
    sh(
        f'python -m longlive_distill.cache_teacher '
        f'--adapter wan21_teacher_adapter:build_teacher '
        f'--adapter-arg model_id="{CONFIG["model_id"]}" '
        f'--dataset "{dataset}" --output "{CACHE}" --device cpu '
        f'--draws-per-clip {CONFIG["draws_per_clip"]} --relation-layers {CONFIG["relation_layers"]} '
        f'--teacher-name "{CONFIG["teacher_name"]}"',
        cwd=REPO, env=env,
    )
    return CACHE


def main():
    require_internet()
    authenticate()
    checkout_repo()
    install_deps()
    dataset = build_dataset()
    cache = build_cache(dataset)

    manifest = json.loads((cache / "manifest.json").read_text())
    push_dataset(cache, CONFIG["teacher_dataset"], CONFIG["teacher_title"],
                 f'{len(manifest["shards"])} shards, {len(manifest["hidden_relation_layers"])} relation layers')

    # Leave only a small summary for the orchestrator to read back.
    state = {
        "shards": len(manifest["shards"]),
        "relation_layers": len(manifest["hidden_relation_layers"]),
        "teacher": manifest["teacher"],
        "teacher_dataset": CONFIG["teacher_dataset"],
        "commit": CONFIG["commit"],
        "kernel_seconds": round(time.time() - started, 1),
    }
    shutil.rmtree(cache, ignore_errors=True)
    shutil.rmtree(SCRATCH, ignore_errors=True)
    (WORKING / "state.json").write_text(json.dumps(state, indent=2) + "\n")
    print(json.dumps(state, indent=2), flush=True)


if __name__ == "__main__":
    sys.exit(main())
