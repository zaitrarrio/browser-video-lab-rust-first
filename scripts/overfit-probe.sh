#!/usr/bin/env bash
# The hours-not-days probe: can this pipeline put pixels on a screen at all?
#
# Every round so far has ended at a parity number. Parity is agreement with the
# teacher's velocity field on shards the teacher itself drew; it is not video,
# and round 1's one decode produced a smooth colour field with no spatial
# structure. This script runs the whole chain to an MP4 a person can watch:
#
#   captions -> Wan2.1 latents + reference MP4s
#            -> teacher cache (the same latents, noised at many sigma)
#            -> video-train train        (the Rust/Burn student)
#            -> video-train sample       (integrate from noise)
#            -> compare_samples          (score + side-by-side MP4)
#
# It is deliberately NOT a small version of the deliverable. It trades away
# generalization — 4 clips, and the eval is fresh noise draws on those same 4
# clips, not held-out clips — to buy the one thing the deliverable's ~7 GPU-day
# schedule cannot give in an afternoon: enough optimizer steps for the student
# to actually fit something. If it cannot reproduce four clips it has seen
# thousands of times, more data was never the missing ingredient.
#
# Run phases separately (they are resumable); `all` runs the lot.
#   PHASE=data|check|cache|floor|train|sample|all  bash scripts/overfit-probe.sh
#
# One GPU job at a time on the rented box — concurrent runs have crashed the
# second one.
set -euo pipefail
cd "$(dirname "$0")/.."

PHASE="${PHASE:-all}"
PYTHON="${PYTHON:-.venv/bin/python}"

# --- what the probe is sized at -------------------------------------------
# 320x320 because the teacher is incoherent below it (verified by sweep): a
# student scored against a 256x256 reference is being scored against smear.
# 13 pixel frames is the Wan VAE's (frames-1)%4==0 rule and gives 4 latent
# frames -> [1,16,4,40,40] -> 1600 tokens, the geometry every perf number in
# docs/PERF-ROUND-1.md was taken at.
CLIPS="${CLIPS:-4}"
HEIGHT="${HEIGHT:-320}"
WIDTH="${WIDTH:-320}"
FRAMES="${FRAMES:-13}"
DRAWS="${DRAWS:-384}"          # per clip -> 4*384 = 1536 training shards
EVAL_DRAWS="${EVAL_DRAWS:-32}" # per clip -> 128 eval shards, fresh seed

# ARM=small runs the ~79M probe student; ARM=full runs the 383M deliverable
# architecture with the per-block conditioning round 3 showed is load-bearing.
ARM="${ARM:-small}"
case "$ARM" in
  small) SPEC=rust/config/probe-79m.json;        OUT=artifacts/probe-small ;;
  full)  SPEC=rust/config/validation-320-adaln.json; OUT=artifacts/probe-full ;;
  *) echo "ARM must be small|full" >&2; exit 2 ;;
esac

# accum 8, NOT the accum 128 that maximises samples/s. Throughput is the wrong
# metric here: what fits a model is optimizer steps, and at a fixed wall-clock
# budget accum 128 buys 1.6x the samples/s while dividing the step count by 16.
# lr stays 2e-5 — round 1 measured 1e-4 driving this architecture *below* the
# trivial floor at accum 8, with the deepest hidden norm inflating as it went.
ACCUM="${ACCUM:-8}"
LR="${LR:-2e-5}"
STEPS="${STEPS:-100000}"        # the real budget is MAX_SECONDS; this is a cap
MAX_SECONDS="${MAX_SECONDS:-14400}"
PRECISION="${PRECISION:-bf16}"  # measured safe and bit-reproducible; f16 hard-fails
BACKEND="${BACKEND:-cuda}"

# The cuda backend is behind a cargo feature — without it the binary builds fine
# and then refuses at runtime with "rebuild with --features cuda", which costs a
# launch rather than a compile. wgpu needs no feature.
FEAT=(); [ "$BACKEND" = cuda ] && FEAT=(--features cuda)

DATA=data/probe-latents
REF=artifacts/probe-reference
CACHE=data/probe-cache
EVAL_CACHE=data/probe-eval-cache

export PYTHONPATH=python
run() { echo; echo "=== $* ==="; "$@"; }
want() { [ "$PHASE" = all ] || [ "$PHASE" = "$1" ]; }

# --- 1. clips -------------------------------------------------------------
# The captions are the first CLIPS of data/prompts-validation.txt, chosen to
# have distinct dominant colours so a collapsed or prompt-ignoring student is
# visible by eye rather than only in a cosine.
if want data; then
  run "$PYTHON" -m longlive_distill.make_wan_dataset \
    --captions data/prompts-validation.txt --limit "$CLIPS" \
    --output "$DATA" --reference-dir "$REF" \
    --height "$HEIGHT" --width "$WIDTH" --frames "$FRAMES" \
    --steps 30 --guidance 5.0 --seed 0
fi

# --- 2. the gate nobody has passed yet ------------------------------------
# Decode the teacher's OWN latent and compare it to the reference MP4 the step
# above wrote. Same tensor, two paths: if these differ, the denormalize/decode
# path is wrong and no student result downstream can be read at all. This costs
# a minute and protects the next six hours.
if want check; then
  run "$PYTHON" -m longlive_distill.decode_latents \
    --latents "$DATA/clip-00000.pt" --output artifacts/probe-decode-check.mp4
  echo
  echo "LOOK AT THESE TWO BEFORE SPENDING GPU TIME:"
  echo "  $REF/clip-00000.mp4          (written during dataset build)"
  echo "  artifacts/probe-decode-check.mp4  (re-decoded through decode_latents)"
  echo "They are the same latent. If they do not match, stop here."
fi

# --- 3. teacher cache -----------------------------------------------------
# f32 teacher, not bf16. bf16 is 2.6x faster and deviates from f32 targets by
# ~6.9% mean, and whether that degrades a student is listed as unmeasured in
# PERF-ROUND-1 section 7. At 1536 shards the f32 cache costs ~5 minutes, so
# paying it removes a confound for free. shift 3.0 matches the sampler's warp.
if want cache; then
  run "$PYTHON" -m longlive_distill.cache_teacher \
    --adapter wan21_teacher_adapter:build_teacher \
    --dataset "$DATA" --output "$CACHE" \
    --device cuda --teacher-dtype f32 --draw-batch 8 \
    --draws-per-clip "$DRAWS" --relation-layers 2 \
    --noising flow --shift 3.0 --guidance 5.0 --seed 0

  # Eval cache: same clips, fresh seed, shift 1.0 so sigma coverage is uniform
  # rather than warped toward high noise, where echoing the input already scores
  # well and every parity number flatters itself.
  run "$PYTHON" -m longlive_distill.cache_teacher \
    --adapter wan21_teacher_adapter:build_teacher \
    --dataset "$DATA" --output "$EVAL_CACHE" \
    --device cuda --teacher-dtype f32 --draw-batch 8 \
    --draws-per-clip "$EVAL_DRAWS" --relation-layers 2 \
    --noising flow --shift 1.0 --guidance 5.0 --seed 9001

  # positional, not `--path` — the Taskfile's rust:validate-cache passes a flag
  # this subcommand does not define.
  run cargo run --release --manifest-path rust/Cargo.toml -p video-cli -- \
    validate-cache "$CACHE"
fi

# --- 4. the floor ---------------------------------------------------------
# Read every number below against this and nothing else. A flow-matching target
# correlates with the model's own input, so "echo the input, best scaled"
# already scores ~0.45-0.50 cosine, and round 3's control sat under it.
if want floor; then
  run "$PYTHON" -m longlive_distill.parity_baseline --cache "$EVAL_CACHE"
fi

# --- 5. train -------------------------------------------------------------
if want train; then
  echo "spec=$SPEC accum=$ACCUM lr=$LR budget=${MAX_SECONDS}s"
  resume=""; [ -f "$OUT/student.mpk" ] && resume="--resume $OUT/student.mpk"
  run cargo run --release "${FEAT[@]}" --manifest-path rust/Cargo.toml -p video-train -- train \
    --spec "$SPEC" --cache "$CACHE" --output "$OUT" \
    --backend "$BACKEND" --precision "$PRECISION" \
    --steps "$STEPS" --max-seconds "$MAX_SECONDS" \
    --lr "$LR" --accum "$ACCUM" --log-every 50 --ckpt-every 500 \
    --seed 42 $resume
  cat "$OUT/state.json"
fi

# --- 6. does it produce video ---------------------------------------------
# Three readouts, in increasing order of how much they can lie to you:
#   eval            -> cosine vs the floor, and pred_norm/teacher_norm. Round 1's
#                      student predicted velocities at 0.63-0.67 of the teacher's
#                      norm; an Euler integrator that travels two-thirds of every
#                      step cannot arrive whatever its direction accuracy.
#   compare_samples -> did the sample land on the clip it was conditioned on, or
#                      on some average of all four?
#   the MP4         -> the only one that answers the actual question.
if want sample; then
  W="$OUT/student.mpk"; [ -f "$W" ] || W="$OUT/student.bin"
  run cargo run --release "${FEAT[@]}" --manifest-path rust/Cargo.toml -p video-train -- eval \
    --spec "$SPEC" --weights "$W" --cache "$EVAL_CACHE" \
    --backend "$BACKEND" --precision f32

  # shard-000000 is clip 0's: cache_teacher is clip-major, so the first DRAWS
  # shards all carry clip 0's prompt embedding.
  run cargo run --release "${FEAT[@]}" --manifest-path rust/Cargo.toml -p video-train -- sample \
    --spec "$SPEC" --weights "$W" \
    --prompt "$CACHE/shard-000000.safetensors" \
    --output "$OUT/sample-clip0.safetensors" \
    --backend "$BACKEND" --precision f32 \
    --steps 32 --shift 3.0 --frames 4 --height 40 --width 40 --seed 1

  run "$PYTHON" -m longlive_distill.compare_samples \
    --sample "$OUT/sample-clip0.safetensors" --dataset "$DATA" \
    --output "$OUT/compare-clip0.mp4"
  echo
  echo "watch: $OUT/compare-clip0.mp4   (student left, nearest teacher clip right)"
fi
