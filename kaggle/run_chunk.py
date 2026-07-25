"""One resumable training chunk, executed headlessly as a Kaggle script kernel.

Pushed by `scripts/kaggle-orchestrate.mjs`, which replaces the CONFIG literal
below before every push. Nothing here is interactive: the kernel restores three
caches, trains until it runs out of steps or wall clock, versions the caches it
changed, and leaves a tiny `state.json` behind for the orchestrator to read.

Cache layers, each skipped entirely on a hit:
  toolchain  a prebuilt `video-train` binary, keyed by a hash of the Rust sources.
             A miss costs ~20 min of cargo; a hit costs one file copy.
  teacher    the framework-neutral safetensors shards. Built once, reused forever.
  ckpt       student weights + AdamW moments + step counter. This is what makes a
             12h session cap irrelevant — every chunk resumes the previous one.

Big artifacts are versioned back to Kaggle datasets from *inside* the kernel so
they never round-trip through the orchestrator; `/kaggle/working` is stripped to
metadata before the run ends, keeping the output pull to a few kilobytes.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

CONFIG = json.loads(r"""{{CONFIG}}""")

WORKING = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
REPO = WORKING / "repo"
RUN = WORKING / "run"
BIN = WORKING / "bin"
started = time.time()


def sh(cmd, cwd=None, env=None, check=True):
    print(f"$ {cmd}", flush=True)
    merged = {**os.environ, **(env or {})}
    return subprocess.run(cmd, shell=True, cwd=cwd, env=merged, check=check)


def require_internet():
    """Fail immediately, and legibly, on a kernel that has no network.

    Every stage needs it: cloning the repo, installing the CLI, reaching the
    secrets backend, pushing the checkpoint. Kaggle accepts `enable_internet`
    in the metadata and reports it back as true regardless, but the setting only
    takes effect on a phone-verified account — so the request succeeding tells
    you nothing, and the first symptom is otherwise ~200s of pip DNS retries.
    """
    import socket

    try:
        socket.getaddrinfo("pypi.org", 443)
    except socket.gaierror as exc:
        raise SystemExit(
            f"this kernel has no internet ({exc}).\n"
            "The metadata requested it and Kaggle accepted, but the setting only applies "
            "to a phone-verified account: kaggle.com → Settings → Phone Verification, then "
            "re-run. Without it the chunk cannot clone, install, authenticate, or push."
        )


def authenticate():
    """The `KGAT_…` token, from this kernel's own secrets. Required, not optional.

    Pinning the CLI here — as CI does — is what makes that a single requirement
    rather than a guess: 2.x authenticates only from KAGGLE_API_TOKEN, 1.x only
    from a legacy username/key pair, and the image's own version is not ours to
    rely on. One pinned CLI, one valid credential shape, one way to fail.
    """
    sh(f"pip install --quiet --upgrade kaggle=={CONFIG['kaggle_cli_version']}")
    sh("kaggle --version", check=False)

    # Injected by the orchestrator, not read from a notebook secret: Kaggle has no
    # API for attaching one, so secrets would mean a manual UI step for every new
    # kernel slug and would keep this pipeline from ever being headless. The token
    # therefore lives in this file's source on Kaggle — private, but cleartext.
    token = CONFIG["api_token"]
    os.environ["KAGGLE_API_TOKEN"] = token
    print(f"KAGGLE_API_TOKEN injected ({len(token)} chars)", flush=True)

    # A well-formed token is not an accepted one. Two seconds now beats learning
    # at the dataset push that a full session of training has nowhere to go.
    probe = subprocess.run(
        "kaggle kernels list --mine --page-size 1", shell=True, capture_output=True, text=True
    )
    if probe.returncode != 0:
        raise SystemExit(f"KAGGLE_API_TOKEN rejected by the CLI:\n{probe.stderr.strip()}")


def push_dataset(folder: Path, slug: str, title: str, message: str):
    """Create-or-version `slug` from `folder`. Idempotent across both cases."""
    owner, name = slug.split("/", 1)
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
    # No such dataset yet — the first chunk of a fresh pipeline creates it.
    print(f"version failed for {slug}; attempting first-time create", flush=True)
    sh(f'kaggle datasets create -p "{folder}" --dir-mode zip')


def source_dir(slug: str) -> Path | None:
    """Kaggle mounts a dataset at /kaggle/input/<name>. Empty dir == cold cache."""
    path = INPUT / slug.split("/", 1)[1]
    return path if path.is_dir() and any(path.iterdir()) else None


def checkout_repo():
    sh(f'git clone --filter=blob:none "{CONFIG["repo_url"]}" "{REPO}"')
    sh(f'git checkout --detach {CONFIG["commit"]}', cwd=REPO)


def restore_toolchain() -> Path:
    """Return a path to a runnable `video-train`, building it only on a key miss."""
    BIN.mkdir(parents=True, exist_ok=True)
    cached = source_dir(CONFIG["toolchain_dataset"])
    key_file = cached / "source-key.txt" if cached else None
    if key_file and key_file.exists() and key_file.read_text().strip() == CONFIG["source_key"]:
        print(f"toolchain cache HIT ({CONFIG['source_key'][:12]})", flush=True)
        shutil.copy2(cached / "video-train", BIN / "video-train")
        (BIN / "video-train").chmod(0o755)
        return BIN / "video-train"

    print(f"toolchain cache MISS ({CONFIG['source_key'][:12]}) — building", flush=True)
    sh("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal")
    cargo = f'{os.path.expanduser("~")}/.cargo/bin/cargo'
    features = f'--features {CONFIG["features"]}' if CONFIG["features"] else ""
    sh(f'{cargo} build --release --locked --manifest-path rust/Cargo.toml -p video-train {features}', cwd=REPO)
    built = REPO / "rust/target/release/video-train"
    shutil.copy2(built, BIN / "video-train")

    staged = WORKING / "toolchain"
    staged.mkdir(parents=True, exist_ok=True)
    shutil.copy2(built, staged / "video-train")
    (staged / "source-key.txt").write_text(CONFIG["source_key"] + "\n")
    push_dataset(staged, CONFIG["toolchain_dataset"], CONFIG["toolchain_title"],
                 f'video-train @ {CONFIG["source_key"][:12]}')
    shutil.rmtree(staged)
    return BIN / "video-train"


def restore_teacher_cache(trainer: Path) -> Path:
    cached = source_dir(CONFIG["teacher_dataset"])
    if cached and (cached / "manifest.json").exists():
        print("teacher cache HIT", flush=True)
        return cached
    if not CONFIG["allow_synthetic_teacher"]:
        raise SystemExit(
            f'teacher cache dataset {CONFIG["teacher_dataset"]} is empty and '
            "allow_synthetic_teacher is false — refusing to train on noise"
        )
    # Explicitly opted-in plumbing mode: random tensors validate the pipeline end
    # to end on real GPU hardware. It produces a running student, never a good one.
    print("teacher cache MISS — synthesizing (PLUMBING ONLY, not a real teacher)", flush=True)
    synth = WORKING / "teacher"
    sh(f'"{trainer}" synth-cache --spec {CONFIG["spec"]} --output "{synth}" '
       f'--shards 8 --frames 2 --height 8 --width 8 --seq 8', cwd=REPO)
    return synth


def restore_checkpoint() -> Path | None:
    """Copy the prior chunk's run dir into working; return the resume target."""
    RUN.mkdir(parents=True, exist_ok=True)
    cached = source_dir(CONFIG["checkpoint_dataset"])
    if not cached or not (cached / "student.mpk").exists():
        print("checkpoint cache MISS — starting from step 0", flush=True)
        return None
    for name in ("student.mpk", "optim.mpk", "state.json"):
        if (cached / name).exists():
            shutil.copy2(cached / name, RUN / name)
    prior = json.loads((RUN / "state.json").read_text()) if (RUN / "state.json").exists() else {}
    print(f'checkpoint cache HIT — resuming at step {prior.get("steps_done", "?")}', flush=True)
    return RUN / "student.mpk"


def main():
    require_internet()
    authenticate()
    checkout_repo()
    trainer = restore_toolchain()
    cache = restore_teacher_cache(trainer)
    resume = restore_checkpoint()

    # Reserve time for the dataset push, so a chunk that fills its budget still
    # gets its checkpoint out. Without this the whole chunk is wasted work.
    budget = max(60, CONFIG["session_seconds"] - int(time.time() - started) - CONFIG["upload_reserve_seconds"])
    args = [
        f'"{trainer}" train',
        f'--spec {CONFIG["spec"]}',
        f'--cache "{cache}"',
        f'--output "{RUN}"',
        f'--backend {CONFIG["backend"]}',
        f'--steps {CONFIG["chunk_steps"]}',
        f'--target-steps {CONFIG["target_steps"]}',
        f'--max-seconds {budget}',
        f'--lr {CONFIG["lr"]}',
        f'--log-every {CONFIG["log_every"]}',
    ]
    if resume:
        args.append(f'--resume "{resume}"')
    sh(" ".join(args), cwd=REPO)

    state = json.loads((RUN / "state.json").read_text())
    push_dataset(RUN, CONFIG["checkpoint_dataset"], CONFIG["checkpoint_title"],
                 f'step {state["steps_done"]}/{state["target_steps"]} loss {state["last_loss"]:.6f}')

    # Leave only metadata in the kernel output: the orchestrator polls this, and
    # pulling multi-GB weights back through CI would dwarf the training itself.
    state["kernel_seconds"] = round(time.time() - started, 1)
    state["commit"] = CONFIG["commit"]
    for path in (REPO, RUN, BIN):
        shutil.rmtree(path, ignore_errors=True)
    (WORKING / "state.json").write_text(json.dumps(state, indent=2) + "\n")
    print(json.dumps(state, indent=2), flush=True)
    if not state["completed"]:
        print(f'chunk finished at step {state["steps_done"]} — schedule another', flush=True)


if __name__ == "__main__":
    sys.exit(main())
