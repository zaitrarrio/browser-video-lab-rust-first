//! Native WGPU forward pass — the same Burn/cubecl stack `video-web` compiles to
//! wasm, on a real GPU, so the browser's per-step cost can be priced without a
//! browser.
//!
//! Timing separates three things that a single-shot run conflates, and that
//! conflation is what made an early reading of this look 3x worse than it is:
//! model construction (random-initialising ~574M parameters on device), cubecl's
//! first-call shader compilation, and the steady-state forward. Only the third is
//! what a sampler pays 32 times.
use anyhow::Result;
use burn::{
    backend::{wgpu::WgpuDevice, Wgpu},
    tensor::{backend::Backend, Distribution, Tensor},
};
use clap::Parser;
use std::{fs, path::PathBuf, time::Instant};

/// Pinned rather than left to inference: `Wgpu` is `Wgpu<F, I>` and calling an
/// associated function like `sync` on the bare alias gives E0283.
type B = Wgpu<f32, i32>;
use video_contract::StudentSpec;
use video_student::BrowserVideoStudent;

#[derive(Parser)]
struct Args {
    #[arg(long)] spec: PathBuf,
    #[arg(long, default_value_t = 4)] frames: usize,
    #[arg(long, default_value_t = 32)] height: usize,
    #[arg(long, default_value_t = 48)] width: usize,
    /// Timed forward passes, after `warmup` discarded ones.
    #[arg(long, default_value_t = 8)] iters: usize,
    #[arg(long, default_value_t = 2)] warmup: usize,
    /// Denoising steps a full generation would run, for the projected total.
    #[arg(long, default_value_t = 32)] steps: usize,
}

fn main() -> Result<()> {
    let a = Args::parse();
    let spec: StudentSpec = serde_json::from_slice(&fs::read(&a.spec)?)?;
    spec.validate()?;
    let device = WgpuDevice::default();

    let t0 = Instant::now();
    let model = BrowserVideoStudent::<B>::new(spec.clone(), &device);
    B::sync(&device);
    let build = t0.elapsed();

    let latents = Tensor::<B, 5>::random(
        [1, spec.latent_channels, a.frames, a.height, a.width],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    let timestep = Tensor::<B, 2>::zeros([1, 1], &device);
    let prompt = Tensor::<B, 3>::zeros([1, 8, spec.text_width], &device);

    let once = || {
        let (out, _) = model.forward(latents.clone(), timestep.clone(), prompt.clone());
        // Force completion: without a sync the queue is still draining and the
        // measurement is of submission, not of work.
        let _ = out.into_data();
    };

    let t1 = Instant::now();
    for _ in 0..a.warmup.max(1) { once() }
    let warm = t1.elapsed();

    let t2 = Instant::now();
    for _ in 0..a.iters.max(1) { once() }
    let per_step = t2.elapsed().as_secs_f64() / a.iters.max(1) as f64;

    let [_pt, ph, pw] = spec.patch_size;
    let tokens = a.frames * (a.height / ph) * (a.width / pw);
    let dense = spec.layers * (4 + 2 * spec.mlp_ratio) * spec.width * spec.width;
    let flops = 2.0 * dense as f64 * tokens as f64
        + spec.layers as f64 * 4.0 * (tokens * tokens) as f64 * spec.width as f64;

    println!("{}", serde_json::json!({
        "spec": a.spec.file_name().and_then(|s| s.to_str()),
        "width": spec.width, "layers": spec.layers, "tokens": tokens,
        "build_s": build.as_secs_f64(),
        "warmup_s": warm.as_secs_f64(),
        "per_step_s": per_step,
        "gflop_per_step": flops / 1e9,
        "achieved_tflops": flops / per_step / 1e12,
        "steps": a.steps,
        "projected_generation_s": per_step * a.steps as f64,
    }));
    Ok(())
}
