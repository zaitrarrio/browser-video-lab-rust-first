# Free-GPU training pipeline

Distillation runs on Kaggle's free GPU (~30 h/week, 12 h per session) as a chain
of resumable chunks. GitHub Actions schedules them; nothing needs a browser.

Kaggle is the only one of the free tiers that can do this. Colab has no supported
headless submission path, and Hugging Face ZeroGPU caps a free call at ~120 s
with a few minutes of daily quota — fine for inference, not for training.

## How a chunk runs

```
cron ──▶ scripts/kaggle-orchestrate.mjs
           │  hashes rust/** → source key
           │  renders kaggle/run_chunk.py with a CONFIG literal
           ├─▶ kaggle kernels push          (script kernel, GPU + internet)
           ├─▶ kaggle kernels status        (poll to completion)
           └─▶ kaggle kernels output        (state.json only — a few KB)
                                                │
kernel: restore toolchain ─▶ teacher ─▶ ckpt ─▶ train ─▶ version ckpt back
```

Three caches, each a Kaggle dataset, each skipped entirely on a hit:

| dataset | holds | key | cost of a miss |
|---|---|---|---|
| `…-toolchain` | prebuilt `video-train` | sha256 of `rust/**` | ~20 min of cargo |
| `…-teacher-cache` | safetensors shards | built once by hand | a full teacher pass |
| `…-checkpoint` | `student.mpk`, `optim.mpk`, `state.json` | n/a — always resumed | the whole run |

Weights are versioned back from *inside* the kernel, so multi-GB checkpoints
never round-trip through CI. The orchestrator only ever downloads `state.json`.

## Resumption

`video-train` owns the run, not the scheduler. Each chunk takes `--target-steps`
(total) and `--steps` (this chunk), and stops early on `--max-seconds` — always
on a step boundary, always writing a full checkpoint. `--resume student.mpk`
restores weights, **AdamW moments**, the step counter, and the shard cursor, so
two 10k-step chunks land where one 20k-step run would. That equivalence is
asserted in `resumed_chunks_match_a_single_run`, so a regression fails CI rather
than quietly degrading a month-long run.

Past `--target-steps` a chunk is a no-op, so an over-eager cron cannot overtrain.

## One-time setup

1. **Kaggle API token** — kaggle.com → Settings → *API* → create a token. It
   looks like `KGAT_…`. The legacy `key` inside `kaggle.json` is **not** a
   substitute: the 2.x CLI this pipeline pins authenticates only from a
   `KGAT_` token in `KAGGLE_API_TOKEN`, and passing one as `KAGGLE_KEY` fails
   with a bare `401` on `SaveKernel`.
2. **GitHub repo secrets** — add `KAGGLE_API_TOKEN` (the token) and
   `KAGGLE_USERNAME` (only used to namespace the kernel and dataset ids).
3. **Phone verification** — kaggle.com → Settings. Without it a kernel gets no
   internet and no GPU, *regardless of what the kernel metadata asks for*: the
   push is accepted, `enable_internet` reads back as true, and the session then
   fails at the first DNS lookup. `require_internet()` names this on sight.
4. **Teacher cache** — produce it once and upload it as
   `<user>/browser-video-student-chunk-teacher-cache`:

   ```sh
   task teacher:cache TEACHER_ADAPTER=pkg.mod:build_teacher DATASET=data/clips
   kaggle datasets create -p data/teacher-cache --dir-mode zip
   ```

   Without it the pipeline refuses to run unless you explicitly pass
   `allow_synthetic_teacher`, which trains on random tensors: it exercises the
   full GPU path and produces a *running* student, never a good one.

5. **Optional repo variables** — `TRAIN_SPEC`, `CHUNK_STEPS`, `TARGET_STEPS`.

## Where the credential lives

The kernel authenticates from a token **injected into its rendered source**, not
from a Kaggle notebook secret. Kaggle has no API for attaching a secret — no SDK
service, no CLI subcommand, no field on the kernel push — so secrets would mean a
manual browser step for every new kernel slug, which is precisely what this
pipeline exists to avoid.

The cost is that `KAGGLE_API_TOKEN` sits in cleartext in the kernel's source and
in every version's history on Kaggle, readable with `kaggle kernels pull`. The
kernel is pushed `is_private: true`, so today that means you alone — but
**publishing or sharing the kernel publishes the token with it**, and it grants
full API access to the account. If that ever happens, rotate it immediately at
kaggle.com → Settings → *API*.

## Running it

```sh
# Inspect the exact kernel that would be pushed, without pushing it.
node scripts/kaggle-orchestrate.mjs --dry-run

# Fire a chunk by hand.
gh workflow run train.yml -f chunk_steps=20000 -f target_steps=200000
```

The cron fires Mon/Wed/Fri; three ~9 h chunks a week fits inside the free quota
with headroom. When `state.completed` flips true the `promote` job publishes
`student.bin` to the rolling `weights-latest` release, which the model bundle
and demo consume the same way `task rust:weights` does locally.

## Budgeting a run

`session_seconds` (default 11 h) minus `upload_reserve_seconds` (15 min) is the
trainer's wall clock. The reserve exists so a chunk that fills its budget still
has time to push its checkpoint — without it, the session reaper takes the whole
chunk's work. Set `CHUNK_STEPS` above what fits in that window and wall clock
becomes the binding constraint, which is usually what you want: the chunk simply
runs the session out and stops cleanly.
