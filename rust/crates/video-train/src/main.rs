use anyhow::{bail, Result};
use burn::backend::{ndarray::NdArrayDevice, wgpu::WgpuDevice, Autodiff, NdArray, Wgpu};
use clap::{Parser, Subcommand};
use std::{fs, path::PathBuf};
use video_contract::StudentSpec;
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
        /// Resume model weights from a prior .mpk checkpoint (optimizer state restarts cold).
        #[arg(long)] resume: Option<PathBuf>,
    },
}

fn main() -> Result<()> {
    match App::parse().command {
        Command::SynthCache { spec, output, shards, frames, height, width, seq, teacher_text_width, relation_layers, seed } => {
            let spec: StudentSpec = serde_json::from_slice(&fs::read(spec)?)?;
            synth_cache(&spec, &output, shards, frames, height, width, seq, teacher_text_width, relation_layers, seed)?;
            println!("wrote {shards} synthetic shards to {}", output.display());
        }
        Command::Train { spec, cache, output, backend, steps, lr, weight_decay, grad_clip, w_output, w_temporal, w_feature, log_every, ckpt_every, seed, resume } => {
            let spec: StudentSpec = serde_json::from_slice(&fs::read(spec)?)?;
            let settings = TrainSettings { steps, lr, weight_decay, grad_clip, w_output, w_temporal, w_feature, log_every, ckpt_every, seed, resume };
            let losses = match backend.as_str() {
                "ndarray" => train::<Autodiff<NdArray>>(spec, &cache, &output, &settings, &NdArrayDevice::default())?,
                "wgpu" => train::<Autodiff<Wgpu>>(spec, &cache, &output, &settings, &WgpuDevice::default())?,
                #[cfg(feature = "cuda")]
                "cuda" => train::<Autodiff<burn::backend::Cuda>>(spec, &cache, &output, &settings, &Default::default())?,
                #[cfg(not(feature = "cuda"))]
                "cuda" => bail!("rebuild with --features cuda for the CUDA backend"),
                other => bail!("unknown backend {other}; use ndarray | wgpu | cuda"),
            };
            println!("done · final loss {:.6} · artifacts in {}", losses.last().copied().unwrap_or(f32::NAN), output.display());
        }
    }
    Ok(())
}
