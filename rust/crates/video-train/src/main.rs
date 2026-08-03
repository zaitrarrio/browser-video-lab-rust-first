use anyhow::{bail, Result};
use burn::backend::{ndarray::NdArrayDevice, wgpu::WgpuDevice, Autodiff, NdArray, Wgpu};
use burn::tensor::{backend::Backend, bf16, f16, DType};
use clap::{Parser, Subcommand, ValueEnum};
use std::{fs, path::{Path, PathBuf}};
use video_contract::StudentSpec;
use video_train::sample::{denoise, evaluate, sample, DenoiseArgs, SampleArgs};
use video_train::{synth_cache, train, TrainSettings, TrainState};

/// Element type every float tensor in the run is stored and computed in.
///
/// This is the backend's `FloatElem`, not a mixed-precision policy: there is no
/// f32 master copy of the weights and no loss scaling. Half precision therefore
/// halves both the bytes moved per parameter *and* the width every reduction
/// accumulates in — a `mean()` over a whole latent tensor included. f32 stays the
/// default precisely because that second half is not free.
#[derive(Clone, Copy, PartialEq, Eq, Debug, ValueEnum)]
enum Precision {
    /// IEEE binary32. The only precision every backend supports, and the one the
    /// checkpoint format is written in regardless of what the run computed in.
    F32,
    /// IEEE binary16: 11-bit significand, max magnitude 65504. Half the
    /// bandwidth of f32 and the only half format wgpu can do arithmetic in.
    ///
    /// Measured unusable for training the 383M student, and `train` therefore
    /// refuses it (see `require_trainable`).
    ///
    /// An earlier reading blamed `clip_by_norm` reducing `sum(g^2)` in f16 and
    /// reported that `--grad-clip 1e30` cured it. On a real teacher cache it does
    /// not: the run goes non-finite and hard-fails with clipping inert, and the
    /// diagnostic names hidden layer 0 — a forward-pass overflow, not a
    /// gradient-norm artifact. 65504 is simply not enough headroom for this
    /// model's residual stream, whose deepest-layer norm climbs past 22000 in
    /// 40 steps. `docs/PERF-ROUND-1.md` has the runs.
    ///
    /// Still offered for `sample`/`eval`, which do no gradient work and integrate
    /// far fewer steps.
    F16,
    /// bfloat16: 8-bit significand, f32's exponent range. Trades mantissa bits
    /// for the dynamic range that makes f16 overflow, and is CUDA-only here
    /// (no graphics API wgpu targets has bf16 arithmetic). The usable half
    /// format for training: same memory saving as f16, loss curve within 1% of
    /// f32's, and bit-reproducible across repeat runs on an RTX 5090.
    Bf16,
}

impl Precision {
    fn dtype(self) -> DType {
        match self {
            Precision::F32 => DType::F32,
            Precision::F16 => DType::F16,
            Precision::Bf16 => DType::BF16,
        }
    }
    fn name(self) -> &'static str {
        match self {
            Precision::F32 => "f32",
            Precision::F16 => "f16",
            Precision::Bf16 => "bf16",
        }
    }
}

/// Reject a (backend, device, precision) triple the hardware cannot actually run
/// *before* a cache is opened or a weight is allocated.
///
/// Burn will otherwise accept `Wgpu<bf16>` at compile time — `bf16: FloatElement`
/// holds for every cube backend — and only fail once a kernel is dispatched, deep
/// inside cubecl and with no mention of the flag that caused it. The wgpu backend
/// really does lack bf16 arithmetic on every graphics API it targets (Vulkan,
/// Metal, DX12 alike: bf16 is buffer/conversion-only there), so this is a live
/// case and not defensive padding. `supports_dtype` asks the runtime rather than
/// hardcoding a table, so a driver that gains the type stops being rejected
/// without a code change.
///
/// The f32 path returns before probing at all: probing spins up the wgpu adapter,
/// and the default path must behave exactly as it did before this flag existed.
fn require_dtype<B: Backend>(device: &B::Device, backend: &str, precision: Precision) -> Result<()> {
    if precision == Precision::F32 || B::supports_dtype(device, precision.dtype()) {
        return Ok(());
    }
    let usable: Vec<&str> = [Precision::F32, Precision::F16, Precision::Bf16]
        .into_iter()
        .filter(|p| *p == Precision::F32 || B::supports_dtype(device, p.dtype()))
        .map(Precision::name)
        .collect();
    bail!(
        "--backend {backend} cannot compute in {} on this device; it supports {}. \
         This is the device's own answer, not a hardcoded table — a type can be present \
         for storage and conversion and still have no arithmetic behind it, which is \
         exactly bf16's status on every graphics API wgpu targets.",
        precision.name(),
        usable.join(" | "),
    )
}

/// Refuse f16 for `train` specifically.
///
/// Not defensive padding: on a real teacher cache (`wan21-flow-shift3`, 383M
/// student, 40 steps x accum 8) f16 goes non-finite at step 8 and hard-fails at
/// step 12 on the five-in-a-row guard, and does so again with `--grad-clip 1e30`,
/// so the clipper is not what breaks it. A flag whose only outcome is a run that
/// dies a third of the way in is a trap, and the failure costs whatever GPU time
/// it took to get there. bf16 has the same memory saving, stays within 1% of
/// f32's loss curve, and is bit-reproducible.
///
/// `--allow-f16-training` exists so the failure can still be reproduced (it is how
/// the finding in `docs/PERF-ROUND-1.md` was made) without editing the source.
fn require_trainable(precision: Precision, allow: bool) -> Result<()> {
    if precision == Precision::F16 && !allow {
        bail!(
            "--precision f16 is not usable for training this student: it goes non-finite \
             within ~10 steps and hard-fails, with or without gradient clipping (the \
             overflow is in the forward pass, at hidden layer 0 — see \
             docs/PERF-ROUND-1.md). Use --precision bf16 for the same memory saving with \
             a usable loss curve, or pass --allow-f16-training to reproduce the failure."
        );
    }
    Ok(())
}

/// The ndarray backend has no half-precision instantiation to dispatch to:
/// `FloatNdArrayElement` is implemented for f32 and f64 only in burn 0.21, so
/// `NdArray<f16>` is not a type that exists. Failing here with the reason beats a
/// page of trait-bound errors, and beats silently running f32 while the user
/// believes they measured f16.
fn require_ndarray_f32(precision: Precision) -> Result<()> {
    if precision != Precision::F32 {
        bail!(
            "--backend ndarray only supports --precision f32 (burn 0.21 implements \
             FloatNdArrayElement for f32 and f64 only — there is no f16/bf16 ndarray \
             backend to dispatch to). Use --backend wgpu for f16, or --backend cuda \
             for f16/bf16."
        );
    }
    Ok(())
}

/// Expand a (backend × precision) dispatch table around one backend-generic entry
/// point.
///
/// The element type is a *type parameter* of the backend (`Wgpu<f16, i32>`), so
/// every combination is a separate monomorphization and the table has to be
/// written out; the macro is what keeps `train`, `sample` and `eval` from each
/// carrying their own nine-arm copy of it, which is exactly how the three would
/// drift apart. `$run` must be generic over a plain `Backend` — `train` wraps its
/// own `Autodiff` inside `run_train` — so that one table serves all three.
macro_rules! dispatch {
    ($run:ident, $backend:expr, $precision:expr, ($($arg:expr),* $(,)?)) => {{
        let backend: &str = $backend;
        let precision: Precision = $precision;
        match backend {
            "ndarray" => {
                require_ndarray_f32(precision)?;
                $run::<NdArray<f32>>($($arg,)* &NdArrayDevice::default())
            }
            "wgpu" => {
                let device = WgpuDevice::default();
                require_dtype::<Wgpu>(&device, "wgpu", precision)?;
                match precision {
                    Precision::F32 => $run::<Wgpu<f32, i32>>($($arg,)* &device),
                    Precision::F16 => $run::<Wgpu<f16, i32>>($($arg,)* &device),
                    Precision::Bf16 => $run::<Wgpu<bf16, i32>>($($arg,)* &device),
                }
            }
            #[cfg(feature = "cuda")]
            "cuda" => {
                use burn::backend::{cuda::CudaDevice, Cuda};
                let device = CudaDevice::default();
                require_dtype::<Cuda>(&device, "cuda", precision)?;
                match precision {
                    Precision::F32 => $run::<Cuda<f32, i32>>($($arg,)* &device),
                    Precision::F16 => $run::<Cuda<f16, i32>>($($arg,)* &device),
                    Precision::Bf16 => $run::<Cuda<bf16, i32>>($($arg,)* &device),
                }
            }
            #[cfg(not(feature = "cuda"))]
            "cuda" => bail!("rebuild with --features cuda for the CUDA backend"),
            // LibTorch. Same model, same loss, same optimizer — ATen's kernels
            // instead of cubecl's, which is what makes the torch gap in
            // docs/PERF-ROUND-2.md §6 attributable rather than just observed.
            // `LibTorch<E>` takes one element parameter, not the `<F, I>` pair the
            // cubecl backends do.
            #[cfg(feature = "tch")]
            "tch" => {
                use burn::backend::{libtorch::LibTorchDevice, LibTorch};
                let device = LibTorchDevice::Cuda(0);
                require_dtype::<LibTorch>(&device, "tch", precision)?;
                match precision {
                    Precision::F32 => $run::<LibTorch<f32>>($($arg,)* &device),
                    Precision::F16 => $run::<LibTorch<f16>>($($arg,)* &device),
                    Precision::Bf16 => $run::<LibTorch<bf16>>($($arg,)* &device),
                }
            }
            #[cfg(not(feature = "tch"))]
            "tch" => bail!("rebuild with --features tch for the LibTorch backend"),
            other => bail!("unknown backend {other}; use ndarray | wgpu | cuda | tch"),
        }
    }};
}

/// Train under a plain (non-autodiff) backend parameter so one `dispatch!` table
/// can serve `train`, `sample` and `eval`. `Autodiff<B>::Device` is `B::Device`,
/// so the caller passes the same device either way.
fn run_train<B: Backend>(
    spec: StudentSpec,
    cache: &Path,
    out: &Path,
    settings: &TrainSettings,
    device: &B::Device,
) -> Result<(Vec<f32>, TrainState)> {
    train::<Autodiff<B>>(spec, cache, out, settings, device)
}

#[derive(Parser)]
struct App { #[command(subcommand)] command: Command }

#[derive(Subcommand)]
enum Command {
    /// Write a tiny synthetic-but-contract-valid teacher cache (plumbing/CI only — random tensors, no model quality).
    SynthCache {
        #[arg(long)] spec: PathBuf,
        #[arg(long)] output: PathBuf,
        #[arg(long, default_value_t = 4)] shards: usize,
        #[arg(long, default_value_t = 2)] frames: usize,
        #[arg(long, default_value_t = 8)] height: usize,
        #[arg(long, default_value_t = 8)] width: usize,
        #[arg(long, default_value_t = 8)] seq: usize,
        #[arg(long, default_value_t = 64)] teacher_text_width: usize,
        #[arg(long, default_value_t = 2)] relation_layers: usize,
        #[arg(long, default_value_t = 7)] seed: u32,
        /// Independent (noise, timestep) draws emitted per clip. Total shards =
        /// shards × draws-per-clip; effective supervision is the shard count.
        #[arg(long, default_value_t = 1)] draws_per_clip: usize,
    },
    /// Distill the browser student from a teacher cache. PyTorch-free.
    Train {
        #[arg(long)] spec: PathBuf,
        #[arg(long)] cache: PathBuf,
        #[arg(long)] output: PathBuf,
        /// ndarray (CPU), wgpu (Metal/Vulkan/DX12), or cuda (requires --features cuda)
        #[arg(long, default_value = "wgpu")] backend: String,
        /// Float element type for every tensor in the run. f32 (default) is the
        /// historical behaviour; f16/bf16 halve bandwidth and enable tensor cores
        /// but change the numerics — check the loss curve, not just the rate.
        #[arg(long, value_enum, default_value_t = Precision::F32)] precision: Precision,
        #[arg(long, default_value_t = 100)] steps: usize,
        #[arg(long, default_value_t = 1e-4)] lr: f64,
        #[arg(long, default_value_t = 0.01)] weight_decay: f32,
        #[arg(long, default_value_t = 1.0)] grad_clip: f32,
        #[arg(long, default_value_t = 1.0)] w_output: f32,
        #[arg(long, default_value_t = 0.25)] w_temporal: f32,
        #[arg(long, default_value_t = 0.05)] w_feature: f32,
        #[arg(long, default_value_t = 10)] log_every: usize,
        #[arg(long, default_value_t = 0)] ckpt_every: usize,
        #[arg(long, default_value_t = 42)] seed: u64,
        /// Resume a run from a prior student.mpk; optim.mpk and state.json beside it are picked up too.
        #[arg(long)] resume: Option<PathBuf>,
        /// Wall-clock budget for this chunk in seconds (0 = unlimited). Stops on a
        /// step boundary and still checkpoints, so preemptible hosts lose nothing.
        #[arg(long, default_value_t = 0)] max_seconds: u64,
        /// Total steps across every chunk (0 = this chunk is the whole run).
        #[arg(long, default_value_t = 0)] target_steps: usize,
        /// Max decoded shards held resident by the lazy loader (bounds RAM).
        #[arg(long, default_value_t = 128)] shard_cache: usize,
        /// Shards summed per optimizer step — the effective batch size. 1 keeps
        /// the historical one-shard-per-step behaviour exactly.
        #[arg(long, default_value_t = 1)] accum: usize,
        /// Print where the chunk's wall clock went (shard-load / upload / fwd+bwd /
        /// optim / readback). Inserts a device barrier per phase, because a lazy
        /// backend otherwise attributes GPU time to whichever line drains the
        /// queue — so a profiled run is slower than the same run without this.
        /// Use it to find the bottleneck, not to quote throughput.
        #[arg(long, default_value_t = false)] profile: bool,
        /// Run `--precision f16` for training anyway. It hard-fails within ~10
        /// steps; this exists only to reproduce that.
        #[arg(long, default_value_t = false)] allow_f16_training: bool,
    },
    /// Integrate the trained student's velocity field from noise to a clean latent
    /// and write it as safetensors for a VAE decode. This is the step that turns
    /// "the loss went down" into something you can actually look at.
    Sample {
        #[arg(long)] spec: PathBuf,
        #[arg(long)] weights: PathBuf,
        /// Safetensors carrying `student_prompt_embeds` (a cache shard works).
        #[arg(long)] prompt: Option<PathBuf>,
        #[arg(long)] output: PathBuf,
        #[arg(long, default_value = "wgpu")] backend: String,
        /// Float element type for the integration. The record is f32 on disk
        /// whatever this is, so a half-precision sample is a speed/accuracy
        /// choice for *this* run only and never rewrites the weights.
        #[arg(long, value_enum, default_value_t = Precision::F32)] precision: Precision,
        #[arg(long, default_value_t = 32)] steps: usize,
        /// Wan's flow_shift (3.0 for the 1.3B checkpoint); 1.0 disables the warp.
        #[arg(long, default_value_t = 3.0)] shift: f32,
        #[arg(long, default_value_t = 4)] frames: usize,
        #[arg(long, default_value_t = 32)] height: usize,
        #[arg(long, default_value_t = 32)] width: usize,
        #[arg(long, default_value_t = 1)] seed: u32,
    },
    /// Teacher parity on a cache: mean cosine and relative L2 between the
    /// student's prediction and the teacher's on identical inputs. Point it at
    /// held-out shards for a generalization number.
    Eval {
        #[arg(long)] spec: PathBuf,
        #[arg(long)] weights: PathBuf,
        #[arg(long)] cache: PathBuf,
        #[arg(long, default_value = "wgpu")] backend: String,
        /// Float element type for the forward passes. Parity numbers measured in
        /// f16/bf16 include that precision's own error, so compare like with like.
        #[arg(long, value_enum, default_value_t = Precision::F32)] precision: Precision,
        /// Shards to score (0 = all).
        #[arg(long, default_value_t = 0)] limit: usize,
    },
    /// One-step reconstruction `x0 = x_σ - σ·v` from a cache shard, using the
    /// student's own velocity. The shard's `x_σ` is on-manifold by construction,
    /// so decoding the result separates "the model cannot draw" from "the sampler
    /// drifts off the manifold the cache covers" — see docs/VALIDATION-ROUND-4.md.
    Denoise {
        #[arg(long)] spec: PathBuf,
        #[arg(long)] weights: PathBuf,
        #[arg(long)] cache: PathBuf,
        /// Index into the cache manifest. Clip-major, so with N draws per clip,
        /// shard k belongs to clip k / N.
        #[arg(long, default_value_t = 0)] shard: usize,
        #[arg(long)] output: PathBuf,
        /// Also write the teacher's reconstruction from the same shard, through
        /// the same code path. This is the control: the two files then differ
        /// only in whose velocity produced them.
        #[arg(long)] teacher_output: Option<PathBuf>,
        /// Use a different shard's prompt embedding with this shard's x_σ. Same
        /// input, different conditioning — the cheapest possible measurement of
        /// whether the prompt path does anything at all.
        #[arg(long)] prompt: Option<PathBuf>,
        #[arg(long, default_value = "wgpu")] backend: String,
        #[arg(long, value_enum, default_value_t = Precision::F32)] precision: Precision,
    },
}

fn main() -> Result<()> {
    match App::parse().command {
        Command::SynthCache { spec, output, shards, frames, height, width, seq, teacher_text_width, relation_layers, seed, draws_per_clip } => {
            let spec: StudentSpec = serde_json::from_slice(&fs::read(spec)?)?;
            synth_cache(&spec, &output, shards, frames, height, width, seq, teacher_text_width, relation_layers, seed, draws_per_clip)?;
            println!("wrote {} synthetic shards to {}", shards * draws_per_clip, output.display());
        }
        Command::Train { spec, cache, output, backend, precision, steps, lr, weight_decay, grad_clip, w_output, w_temporal, w_feature, log_every, ckpt_every, seed, resume, max_seconds, target_steps, shard_cache, accum, profile, allow_f16_training } => {
            // Before the cache is opened or a weight allocated, as with the other
            // precision guards: the point is to cost nothing rather than to fail
            // ten steps into a GPU run.
            require_trainable(precision, allow_f16_training)?;
            let spec: StudentSpec = serde_json::from_slice(&fs::read(spec)?)?;
            let settings = TrainSettings { steps, lr, weight_decay, grad_clip, w_output, w_temporal, w_feature, log_every, ckpt_every, seed, resume, max_seconds, target_steps, shard_cache, accum, profile };
            let (losses, state) = dispatch!(run_train, &backend, precision, (spec, &cache, &output, &settings))?;
            println!(
                "chunk done · {} steps this chunk · {}/{} total · final loss {:.6} · stopped by {} · {} · artifacts in {}",
                losses.len(),
                state.steps_done,
                if state.target_steps > 0 { state.target_steps.to_string() } else { "-".into() },
                losses.last().copied().unwrap_or(state.last_loss),
                state.stopped_by,
                if state.completed { "COMPLETED" } else { "resume for more" },
                output.display(),
            );
        }
        Command::Sample { spec, weights, prompt, output, backend, precision, steps, shift, frames, height, width, seed } => {
            let spec: StudentSpec = serde_json::from_slice(&fs::read(spec)?)?;
            let args = SampleArgs { spec, weights, prompt, output, steps, shift, frames, height, width, seed };
            dispatch!(sample, &backend, precision, (&args))?;
        }
        Command::Eval { spec, weights, cache, backend, precision, limit } => {
            let spec: StudentSpec = serde_json::from_slice(&fs::read(spec)?)?;
            dispatch!(evaluate, &backend, precision, (&spec, &weights, &cache, limit))?;
        }
        Command::Denoise { spec, weights, cache, shard, output, teacher_output, prompt, backend, precision } => {
            let spec: StudentSpec = serde_json::from_slice(&fs::read(spec)?)?;
            let args = DenoiseArgs { spec, weights, cache, shard, output, teacher_output, prompt };
            dispatch!(denoise, &backend, precision, (&args))?;
        }
    }
    Ok(())
}
