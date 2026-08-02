# Validation round 2 — spend the compute

Round 1 ([`VALIDATION-ROUND-1.md`](VALIDATION-ROUND-1.md)) closed the loop and
found the student undertrained rather than broken: held-out parity cleared the
trivial floor and improved with batching, but the sampled latent had no spatial
structure and the student under-predicted velocity magnitude by ~35% uniformly
across σ. Its first recommendation was blunt — *spend the compute; everything
else is an optimization of that*. This round does exactly that, on Kaggle's free
GPU rather than a rented one.

## What changed from round 1

| | round 1 | round 2 |
|---|---|---|
| clips | 8 | **128** |
| shards | 1536 (8 × 192) | **2560** (128 × 20) |
| question | can it memorize 8 clips? | can it learn the field? |
| batch | 1, then 8 | **8** throughout |
| hardware | rented RTX 5090, ~40 min | Kaggle free GPU, 11 h/chunk |

The clip count is what matters. Round 1 was an overfit probe by design — dense
(noise, σ) coverage over a handful of clips, which can only distinguish "learned
something" from "learned nothing". 128 clips is still nowhere near a real
distillation corpus, but it is enough that memorization stops being the cheap
explanation for a good parity number.

Geometry is unchanged at 320×320 (`[1,16,4,40,40]`, 1600 tokens post-patchify),
for the reason round 1 established: Wan2.1-1.3B is itself incoherent below 320,
so a smaller reference is not worth matching.

## The cache

`zaitrarriocollier/browser-video-student-chunk-teacher-cache` **v3** — 2560
shards, 7.5 GB, `scheduler: wan21-flow-shift3-cfg5`, `validate-cache` clean.
Real Wan latents, flow-matching noising warped by `shift=3`, CFG-5 guided
velocity targets, no relation grams (see round 1 on why grams cost an order of
magnitude in draw count).

> **v1 and v2 are not this.** v1 is the original cache: `scheduler: wan21`,
> legacy `x₀ + σ·ε` noising, 128 shards, and `torch.randn` latents. Training on
> it reruns the configuration round 1 disproved. If a run ever mounts v1 by
> accident the manifest's `scheduler` string is how to notice — that is what it
> is for.

### Uploading it is not trivial

`kaggle datasets version` uploads top-level files individually — `--dir-mode zip`
only archives *subdirectories* — so 2561 shards go up one at a time and the final
`CreateDatasetVersion` call **504s at the gateway**. Every byte transfers and no
version is committed; roughly an hour is lost with no error until the very end.

The fix is to hand Kaggle a single archive: zip the shards and manifest at the
zip root (`zip -0`, stored — the payload is incompressible float data) and upload
that one file. Kaggle auto-extracts it into the dataset root, so the mounted
layout is identical and `restore_teacher_cache` needs no change. Verified with a
throwaway two-file dataset before committing to the 7.5 GB upload.

## Reading the result

Held-out eval cache: `zaitrarriocollier/browser-video-student-chunk-eval-cache`,
128 shards, one fresh-seed draw per clip at `shift=1` so σ coverage is uniform
rather than warped toward high noise.

Trivial-predictor floor on that cache (`task teacher:baseline`):

| trivial predictor | cosine | rel L2 |
|---|---|---|
| echo the noisy input | −0.101 | 1.209 |
| **best-scaled echo (least squares)** | **+0.443** | **0.850** |
| mean teacher target | +0.213 | 0.977 |

**+0.443 is the floor.** Round 1 finished at +0.584 on 8 clips. A round-2 number
below the floor means the extra clips bought nothing; a number near round 1's on
16× the clips would be the more interesting result, since it would mean the
student is learning the field rather than the clips.

The magnitude ratio matters as much as the cosine: round 1's student predicted
velocities at 0.63–0.67 of the teacher's norm, and an Euler integrator that
travels two-thirds of each step cannot reach the data manifold whatever its
direction accuracy. Watch `pred_norm/teacher_norm` in `video-train eval` output
alongside the cosine.

## Run configuration

```
spec       rust/config/validation-320.json   (390M umt5 student, max_tokens 8192)
backend    cuda (Burn, built in-kernel with --features cuda)
lr         2e-5        raising it is the one thing not to do — see round 1
accum      8           one optimizer step per 8 shards
target     20,000 steps = 160,000 sample-views
session    11 h, checkpoint every 1,000 steps pushed from inside the kernel
```

`TARGET_STEPS` is 20,000 rather than the previous default of 200,000: at accum 8
that default is ~1.6 M sample-views, which against Kaggle's ~30 GPU-h/week runs
for months. The chunk is resumable, so the target is a decision about when to
stop and look, not a limit on the run.

## Result: Kaggle's free GPU cannot run this trainer

Both Burn backends fail on Kaggle, for unrelated reasons, and neither is a
configuration mistake. The accelerator is a **Tesla P100-PCIE-16GB** (Pascal,
sm_60, driver 580.159.04).

| backend | Kaggle P100 (sm_60) | Vast RTX 5090 (sm_120) |
|---|---|---|
| `cuda` | panics every step → loss NaN by step 5 | **works** |
| `wgpu` | no Vulkan adapter | **works** |

**CUDA.** `cubecl-cuda 0.10` panics on every step at `compute/stream.rs:101`:

```
called `Result::unwrap()` on an `Err` value:
couldn't find resource for that handle: Memory page 0 doesn't exist
```

The panics land on a worker thread and are swallowed, so the trainer keeps
running on garbage; the loss is non-finite from step 1 and the sustained-
divergence guard stops the chunk at step 5. Nine minutes, no checkpoint.

This is **not** a generic cubecl bug. The identical binary, cache, spec, lr and
accum ran six clean finite steps on an RTX 5090 (sm_120). The failure is
specific to Pascal-era hardware.

**wgpu.** No adapter at all:

```
No possible adapter available for backend.
NotFound { active_backends: Backends(0x0), requested_backends: Backends(VULKAN),
           supported_backends: Backends(VULKAN | GL) }
```

`report_accelerator()` shows why: no `/etc/vulkan/icd.d`, no
`/usr/share/vulkan/icd.d`, and no `libGLX_nvidia`/`libcuda.so` on the default
library path. The container was granted the nvidia-container-toolkit's compute
capabilities without `graphics`, so there is no Vulkan ICD to find and none can
be installed from inside the kernel — the driver library it would have to point
at is not mounted.

**Choosing a newer GPU is not available over the API.** `ApiSaveKernelRequest`
exposes `enable_gpu`, `enable_tpu` and `enable_internet` and nothing else; there
is no accelerator-type field, so a headless push cannot ask for the T4 (sm_75)
that would plausibly clear the cubecl failure. TPU is moot — Burn has no TPU
backend.

### What this costs the plan

`TEACHER-OPTIONS.md` records the assumption that "the free-tier GPU's sm_60
incompatibility — fatal for PyTorch, irrelevant to Burn" made Kaggle viable for
the Rust student. That assumption is now falsified: sm_60 is fatal for
Burn/cubecl too, just by a different mechanism, and the wgpu escape hatch is
closed by the container image rather than by the GPU.

The free-tier pipeline still works for what it was originally built for — the
**CPU** teacher-cache job (`cache_chunk.py`), which needs no accelerator. It is
only the GPU training half that has no home on Kaggle.

### Options, in the order they cost

1. **Set the kernel's accelerator to T4 in the Kaggle UI once**, then re-push
   over the API and see whether the choice sticks. Requires a manual click and
   is unverified, but it is the only path that keeps training free.
2. **Upgrade Burn past 0.21** and hope cubecl's Pascal support improves. The
   loop is slow — the bug reproduces only on a P100, which means every attempt
   is a Kaggle round trip — and there is no evidence the fix exists upstream.
3. **Train on rented GPU.** Round 1 measured ~18 GPU-days for a conventional
   schedule, about $215 at $0.49/h on the 5090 already in use, before the three
   speed levers round 1 identified (bf16, fused attention, a spatial-16 VAE).

Everything else is ready and waiting: the cache is on Kaggle at v3, the eval
cache and its floor are recorded, the trainer batches, and the sampler and decode
work. Only the hardware is missing.
