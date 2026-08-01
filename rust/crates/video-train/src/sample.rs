//! Sampling and teacher-parity evaluation for a trained student.
//!
//! `video-train train` proved the student can be *fit*; neither the trainer nor
//! `video-web` could until now answer the question the project actually cares
//! about — does integrating the student's own predictions from noise produce a
//! coherent latent? That needs two things the tree lacked: a sampler that uses
//! the *same* flow-matching convention the cache was built with, and a way to
//! get the resulting latents out to a VAE.
//!
//! ## The convention, stated once
//!
//! Wan2.1 is a flow-matching model, not an ε-predictor:
//!
//! ```text
//! x_σ = (1 - σ)·x₀ + σ·ε            σ ∈ [0,1], σ=1 is pure noise
//! v   = dx/dσ = ε - x₀               what the teacher (and so the student) predicts
//! t   = σ · 1000                     the timestep the model is conditioned on
//! ```
//!
//! Integrating backwards from σ=1 to σ=0 with Euler is therefore
//! `x ← x - Δσ·v`, which is exactly the update `video-web::generate` already
//! performs. `cache_teacher.py` must build shards with the same equations or the
//! student learns a field that its own sampler cannot integrate — the failure
//! mode this module exists to make visible.
//!
//! `shift` is Wan's schedule warp (`flow_shift`, 3.0 for the 1.3B checkpoint):
//! it spends more steps at high noise. It maps 0→0 and 1→1, so the endpoints are
//! unchanged; only the spacing moves.
use anyhow::{bail, Context, Result};
use burn::module::Module;
use burn::record::{BinFileRecorder, FullPrecisionSettings, NamedMpkFileRecorder};
use burn::tensor::{backend::Backend, Tensor};
use safetensors::{tensor::TensorView, Dtype};
use std::path::{Path, PathBuf};
use video_contract::{validate_cache, StudentSpec};
use video_student::BrowserVideoStudent;

use crate::{load_shard, Lcg};

/// Wan's shifted flow-matching sigma warp. `shift = 1.0` is the identity.
pub fn shift_sigma(sigma: f32, shift: f32) -> f32 {
    if shift <= 0.0 {
        return sigma;
    }
    shift * sigma / (1.0 + (shift - 1.0) * sigma)
}

/// Descending sigma schedule of `steps + 1` points from 1.0 down to 0.0, warped
/// by `shift`. The extra trailing point is what makes the last Euler step land
/// exactly on σ=0 rather than short of it.
pub fn sigma_schedule(steps: usize, shift: f32) -> Vec<f32> {
    let steps = steps.max(1);
    (0..=steps)
        .map(|i| shift_sigma(1.0 - i as f32 / steps as f32, shift))
        .collect()
}

fn load_student<B: Backend>(spec: &StudentSpec, weights: &Path, device: &B::Device) -> Result<BrowserVideoStudent<B>> {
    let model = BrowserVideoStudent::<B>::new(spec.clone(), device);
    let ext = weights.extension().and_then(|e| e.to_str()).unwrap_or("").to_owned();
    let stem = weights.with_extension("");
    let loaded = match ext.as_str() {
        "mpk" => model.load_file(stem, &NamedMpkFileRecorder::<FullPrecisionSettings>::default(), device),
        "bin" | "" => model.load_file(stem, &BinFileRecorder::<FullPrecisionSettings>::default(), device),
        other => bail!("unknown weights format .{other}; expected .bin or .mpk"),
    }
    .map_err(|e| anyhow::anyhow!("load student record from {}: {e}", weights.display()))?;
    Ok(loaded)
}

/// Read a `[1, seq, width]` prompt embedding out of a safetensors file, preferring
/// `student_prompt_embeds` (the browser-runnable umt5-small side) over the
/// teacher's own `prompt_embeds`. A cache shard is the most convenient carrier:
/// it already holds the exact embedding the student was trained against.
fn read_prompt(path: &Path, text_width: usize) -> Result<(Vec<f32>, usize)> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let st = safetensors::SafeTensors::deserialize(&bytes)?;
    let view = st
        .tensor("student_prompt_embeds")
        .or_else(|_| st.tensor("prompt_embeds"))
        .map_err(|_| anyhow::anyhow!("{} has neither student_prompt_embeds nor prompt_embeds", path.display()))?;
    let shape = view.shape();
    let width = *shape.last().unwrap_or(&0);
    if width != text_width {
        bail!("prompt embedding width {width} != spec.text_width {text_width} (wrong config for this cache?)");
    }
    let values = crate::floats(&view)?;
    Ok((values, shape[shape.len() - 2]))
}

pub struct SampleArgs {
    pub spec: StudentSpec,
    pub weights: PathBuf,
    pub prompt: Option<PathBuf>,
    pub output: PathBuf,
    pub steps: usize,
    pub shift: f32,
    pub frames: usize,
    pub height: usize,
    pub width: usize,
    pub seed: u32,
}

/// Integrate the student's velocity field from pure noise to σ=0 and write the
/// resulting latent as safetensors for a VAE decode.
pub fn sample<B: Backend>(a: &SampleArgs, device: &B::Device) -> Result<()> {
    a.spec.validate()?;
    let [_pt, ph, pw] = a.spec.patch_size;
    if a.height % ph != 0 || a.width % pw != 0 {
        bail!("latent {}x{} not divisible by patch_size {ph}x{pw}", a.height, a.width);
    }
    let model = load_student::<B>(&a.spec, &a.weights, device)?;

    let (prompt_vec, seq) = match &a.prompt {
        Some(p) => read_prompt(p, a.spec.text_width)?,
        // No prompt file: a deterministic seeded embedding, matching what the
        // browser runtime falls back to. Useful for a pure "does it integrate"
        // check, useless for prompt adherence.
        None => {
            let mut rng = Lcg::new(a.seed ^ 0x9e37_79b9);
            (rng.vec(8 * a.spec.text_width), 8)
        }
    };
    let prompt = Tensor::<B, 1>::from_floats(prompt_vec.as_slice(), device).reshape([1, seq, a.spec.text_width]);

    let cells = a.spec.latent_channels * a.frames * a.height * a.width;
    let mut rng = Lcg::new(a.seed);
    let mut latents = Tensor::<B, 1>::from_floats(rng.vec(cells).as_slice(), device)
        .reshape([1, a.spec.latent_channels, a.frames, a.height, a.width]);

    let sigmas = sigma_schedule(a.steps, a.shift);
    for i in 0..a.steps.max(1) {
        let (s, s_next) = (sigmas[i], sigmas[i + 1]);
        let timestep = Tensor::<B, 1>::from_floats([s * 1000.0].as_slice(), device).reshape([1, 1]);
        let (velocity, _hidden) = model.forward(latents.clone(), timestep, prompt.clone());
        // Euler step down the sigma schedule: x ← x - (σ_i - σ_{i+1})·v.
        latents = latents - velocity.mul_scalar(s - s_next);
        println!("{}", serde_json::json!({"step": i + 1, "sigma": s, "sigma_next": s_next}));
    }

    let values: Vec<f32> = latents
        .into_data()
        .to_vec::<f32>()
        .map_err(|e| anyhow::anyhow!("latent readback failed: {e:?}"))?;
    let shape = vec![1, a.spec.latent_channels, a.frames, a.height, a.width];
    let bytes: Vec<u8> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
    if let Some(parent) = a.output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    safetensors::serialize_to_file(
        vec![("latents".to_string(), TensorView::new(Dtype::F32, shape, &bytes)?)],
        None,
        &a.output,
    )?;
    println!("wrote {} to {}", values.len(), a.output.display());
    Ok(())
}

/// Teacher-parity over a cache: how close is the student's prediction to the
/// teacher's on the same inputs?
///
/// Reported per shard and in aggregate:
/// * `cosine` — direction agreement of the predicted velocity field. This is the
///   number that decides whether sampling can work at all; an Euler integrator
///   follows directions, and a student at cosine ≈ 0 is integrating noise.
/// * `rel_l2` — ‖student − teacher‖ / ‖teacher‖, which additionally catches a
///   systematically mis-scaled field that cosine alone would call perfect.
///
/// Run it on shards the student never trained on to get a generalization number
/// rather than a memorization one.
pub fn evaluate<B: Backend>(spec: &StudentSpec, weights: &Path, cache: &Path, limit: usize, device: &B::Device) -> Result<(f32, f32)> {
    spec.validate()?;
    let manifest = validate_cache(cache)?;
    let model = load_student::<B>(spec, weights, device)?;
    let count = if limit == 0 { manifest.shards.len() } else { limit.min(manifest.shards.len()) };
    let (mut cos_sum, mut rel_sum) = (0f32, 0f32);
    for shard in manifest.shards.iter().take(count) {
        let sample = load_shard(&cache.join(shard), &[])?;
        let noisy = crate::t5::<B>(&sample.noisy, device);
        let timestep = Tensor::<B, 1>::from_floats(sample.timestep.as_slice(), device)
            .reshape([sample.timestep.len(), 1]);
        let prompt = crate::t3::<B>(sample.student_prompt.as_ref().unwrap_or(&sample.prompt), device);
        let (pred, _) = model.forward(noisy, timestep, prompt);
        let p: Vec<f32> = pred.into_data().to_vec::<f32>().map_err(|e| anyhow::anyhow!("readback: {e:?}"))?;
        let t = &sample.teacher_pred.0;
        let dot: f64 = p.iter().zip(t).map(|(a, b)| (*a as f64) * (*b as f64)).sum();
        let pn: f64 = p.iter().map(|a| (*a as f64) * (*a as f64)).sum::<f64>().sqrt();
        let tn: f64 = t.iter().map(|b| (*b as f64) * (*b as f64)).sum::<f64>().sqrt();
        let diff: f64 = p.iter().zip(t).map(|(a, b)| ((*a - *b) as f64).powi(2)).sum::<f64>().sqrt();
        let (cos, rel) = ((dot / (pn * tn).max(1e-12)) as f32, (diff / tn.max(1e-12)) as f32);
        // Per-shard, with its sigma: an aggregate cosine hides *where* on the
        // trajectory the student agrees with its teacher, and that is the
        // difference between "undertrained" and "cannot represent its
        // conditioning". A student whose only sigma input is a single Linear at
        // the stem tends to fit the high-sigma end (where the target is nearly
        // the input) and miss the low-sigma end (where all the structure is) —
        // which averages into a mediocre aggregate that looks like slow progress.
        println!("{}", serde_json::json!({
            "shard": shard, "sigma": sample.timestep.first().copied().unwrap_or(0.0) / 1000.0,
            "cosine": cos, "rel_l2": rel,
            "pred_norm": pn as f32, "teacher_norm": tn as f32,
        }));
        cos_sum += cos;
        rel_sum += rel;
    }
    let (cosine, rel_l2) = (cos_sum / count as f32, rel_sum / count as f32);
    println!("{}", serde_json::json!({"shards": count, "cosine": cosine, "rel_l2": rel_l2}));
    Ok((cosine, rel_l2))
}

#[cfg(test)]
mod tests {
    use super::*;

    // The schedule must start at pure noise and land exactly on zero whatever the
    // warp: a last step that stops short of σ=0 leaves residual noise in every
    // sample, and one that starts below 1.0 means the student is asked to denoise
    // from a distribution it was never trained on.
    #[test]
    fn schedule_spans_the_full_range_under_any_shift() {
        for shift in [1.0f32, 3.0, 5.0] {
            let s = sigma_schedule(8, shift);
            assert_eq!(s.len(), 9);
            assert!((s[0] - 1.0).abs() < 1e-6, "shift {shift} does not start at 1.0: {s:?}");
            assert!(s[8].abs() < 1e-6, "shift {shift} does not end at 0.0: {s:?}");
            for w in s.windows(2) {
                assert!(w[0] > w[1], "schedule must decrease strictly: {s:?}");
            }
        }
    }

    // shift > 1 front-loads the steps into the high-noise region — the whole point
    // of Wan's flow_shift. Concretely, the midpoint sigma must sit above the
    // unshifted 0.5.
    #[test]
    fn shift_biases_steps_towards_high_noise() {
        let plain = sigma_schedule(8, 1.0);
        let shifted = sigma_schedule(8, 3.0);
        assert!((plain[4] - 0.5).abs() < 1e-6);
        assert!(shifted[4] > plain[4], "shift=3 should hold higher sigma at the midpoint: {shifted:?}");
    }
}
