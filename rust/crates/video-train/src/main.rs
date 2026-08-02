use anyhow::{bail, Result};
use burn::backend::{ndarray::NdArrayDevice, wgpu::WgpuDevice, Autodiff, NdArray, Wgpu};
use clap::{Parser, Subcommand};
use std::{fs, path::PathBuf};
use video_contract::StudentSpec;
use video_train::sample::{evaluate, sample, SampleArgs};
use video_train::{synth_cache, train, TrainSettings};

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
        /// Shards to score (0 = all).
        #[arg(long, default_value_t = 0)] limit: usize,
    },
}

fn main() -> Result<()> {
    match App::parse().command {
        Command::SynthCache { spec, output, shards, frames, height, width, seq, teacher_text_width, relation_layers, seed, draws_per_clip } => {
            let spec: StudentSpec = serde_json::from_slice(&fs::read(spec)?)?;
            synth_cache(&spec, &output, shards, frames, height, width, seq, teacher_text_width, relation_layers, seed, draws_per_clip)?;
            println!("wrote {} synthetic shards to {}", shards * draws_per_clip, output.display());
        }
        Command::Train { spec, cache, output, backend, steps, lr, weight_decay, grad_clip, w_output, w_temporal, w_feature, log_every, ckpt_every, seed, resume, max_seconds, target_steps, shard_cache, accum, profile } => {
            let spec: StudentSpec = serde_json::from_slice(&fs::read(spec)?)?;
            let settings = TrainSettings { steps, lr, weight_decay, grad_clip, w_output, w_temporal, w_feature, log_every, ckpt_every, seed, resume, max_seconds, target_steps, shard_cache, accum, profile };
            let (losses, state) = match backend.as_str() {
                "ndarray" => train::<Autodiff<NdArray>>(spec, &cache, &output, &settings, &NdArrayDevice::default())?,
                "wgpu" => train::<Autodiff<Wgpu>>(spec, &cache, &output, &settings, &WgpuDevice::default())?,
                #[cfg(feature = "cuda")]
                "cuda" => train::<Autodiff<burn::backend::Cuda>>(spec, &cache, &output, &settings, &Default::default())?,
                #[cfg(not(feature = "cuda"))]
                "cuda" => bail!("rebuild with --features cuda for the CUDA backend"),
                other => bail!("unknown backend {other}; use ndarray | wgpu | cuda"),
            };
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
        Command::Sample { spec, weights, prompt, output, backend, steps, shift, frames, height, width, seed } => {
            let spec: StudentSpec = serde_json::from_slice(&fs::read(spec)?)?;
            let args = SampleArgs { spec, weights, prompt, output, steps, shift, frames, height, width, seed };
            match backend.as_str() {
                "ndarray" => sample::<NdArray>(&args, &NdArrayDevice::default())?,
                "wgpu" => sample::<Wgpu>(&args, &WgpuDevice::default())?,
                #[cfg(feature = "cuda")]
                "cuda" => sample::<burn::backend::Cuda>(&args, &Default::default())?,
                #[cfg(not(feature = "cuda"))]
                "cuda" => bail!("rebuild with --features cuda for the CUDA backend"),
                other => bail!("unknown backend {other}; use ndarray | wgpu | cuda"),
            }
        }
        Command::Eval { spec, weights, cache, backend, limit } => {
            let spec: StudentSpec = serde_json::from_slice(&fs::read(spec)?)?;
            match backend.as_str() {
                "ndarray" => evaluate::<NdArray>(&spec, &weights, &cache, limit, &NdArrayDevice::default())?,
                "wgpu" => evaluate::<Wgpu>(&spec, &weights, &cache, limit, &WgpuDevice::default())?,
                #[cfg(feature = "cuda")]
                "cuda" => evaluate::<burn::backend::Cuda>(&spec, &weights, &cache, limit, &Default::default())?,
                #[cfg(not(feature = "cuda"))]
                "cuda" => bail!("rebuild with --features cuda for the CUDA backend"),
                other => bail!("unknown backend {other}; use ndarray | wgpu | cuda"),
            };
        }
    }
    Ok(())
}
