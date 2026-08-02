use anyhow::{bail, Context, Result};
use safetensors::SafeTensors;
use serde::{Deserialize, Serialize};
use std::{fs, path::{Path, PathBuf}};

pub const REQUIRED_SAMPLE_TENSORS: &[&str] = &["noisy_latents", "timestep", "prompt_embeds", "teacher_noise_pred"];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorShape { pub name: String, pub shape: Vec<usize>, pub dtype: String }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeacherCacheManifest {
    pub format_version: u32,
    pub teacher: String,
    pub scheduler: String,
    pub shards: Vec<PathBuf>,
    pub tensors: Vec<TensorShape>,
    pub hidden_relation_layers: Vec<usize>,
}

/// Default DiT patch size `[pt, ph, pw]`. Matches every Wan-family DiT
/// (`patch_size: [1,2,2]`). Applied when a spec JSON omits `patch_size`, which
/// keeps every pre-patchify config (and the browser `DEMO_SPEC`) valid.
fn default_patch() -> [usize; 3] { [1, 2, 2] }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StudentSpec {
    pub latent_channels: usize,
    pub text_width: usize,
    pub width: usize,
    pub layers: usize,
    pub heads: usize,
    pub mlp_ratio: usize,
    pub max_tokens: usize,
    /// DiT patch size `[pt, ph, pw]`. Phase 0 requires `pt == 1` (temporal
    /// patching would need a rank-8 patchify); spatial patching cuts the token
    /// count `ph*pw`× and lines the relation grams up with a Wan teacher.
    #[serde(default = "default_patch")]
    pub patch_size: [usize; 3],
    /// Modulate every block with the (σ, prompt) conditioning vector — adaLN-zero,
    /// as every DiT in this family does — instead of adding it once at the stem.
    ///
    /// Default `false` keeps the historical architecture and lets every existing
    /// spec and checkpoint load unchanged, so the two are an A/B rather than a
    /// migration. Round 1 recorded parity that was flat across σ and suspected the
    /// single stem injection of being a ceiling rather than a symptom of
    /// undertraining; this flag is what makes that testable.
    #[serde(default)]
    pub per_block_conditioning: bool,
}

impl StudentSpec {
    pub fn validate(&self) -> Result<()> {
        if self.width % self.heads != 0 { bail!("width must be divisible by heads") }
        if self.layers == 0 || self.max_tokens == 0 { bail!("layers and max_tokens must be non-zero") }
        if self.patch_size.iter().any(|&p| p == 0) { bail!("patch_size entries must be non-zero") }
        if self.patch_size[0] != 1 { bail!("patch_size[0] (temporal) must be 1 in this release") }
        Ok(())
    }
    /// Latent cells folded into one token: `pt * ph * pw`.
    pub fn patch_volume(&self) -> usize { self.patch_size.iter().product() }
    pub fn approximate_parameters(&self) -> usize {
        // input `Linear(latent_channels*patch_volume → width)` and its output
        // mirror both scale with the patch volume.
        self.layers * (4 + 2 * self.mlp_ratio) * self.width * self.width
            + self.text_width * self.width + 2 * self.latent_channels * self.patch_volume() * self.width
    }
}

pub fn validate_cache(root: &Path) -> Result<TeacherCacheManifest> {
    let manifest: TeacherCacheManifest = serde_json::from_slice(&fs::read(root.join("manifest.json"))?)?;
    if manifest.format_version != 1 { bail!("unsupported teacher-cache format") }
    for shard in &manifest.shards {
        let path = root.join(shard); let bytes = fs::read(&path).with_context(|| format!("read {}", path.display()))?;
        let tensors = SafeTensors::deserialize(&bytes)?;
        for name in REQUIRED_SAMPLE_TENSORS { tensors.tensor(name).with_context(|| format!("{name} missing in {}", path.display()))?; }
    }
    Ok(manifest)
}

#[cfg(test)]
mod tests { use super::*; #[test] fn estimates_browser_student(){ let s=StudentSpec{latent_channels:16,text_width:4096,width:1152,layers:24,heads:16,mlp_ratio:4,max_tokens:6144,patch_size:[1,2,2],per_block_conditioning:false}; assert!(s.approximate_parameters()>380_000_000); assert!(s.validate().is_ok()); } }

