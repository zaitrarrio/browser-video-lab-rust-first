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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StudentSpec {
    pub latent_channels: usize,
    pub text_width: usize,
    pub width: usize,
    pub layers: usize,
    pub heads: usize,
    pub mlp_ratio: usize,
    pub max_tokens: usize,
}

impl StudentSpec {
    pub fn validate(&self) -> Result<()> {
        if self.width % self.heads != 0 { bail!("width must be divisible by heads") }
        if self.layers == 0 || self.max_tokens == 0 { bail!("layers and max_tokens must be non-zero") }
        Ok(())
    }
    pub fn approximate_parameters(&self) -> usize {
        self.layers * (4 + 2 * self.mlp_ratio) * self.width * self.width
            + self.text_width * self.width + 2 * self.latent_channels * self.width
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
mod tests { use super::*; #[test] fn estimates_browser_student(){ let s=StudentSpec{latent_channels:4,text_width:4096,width:1152,layers:24,heads:16,mlp_ratio:4,max_tokens:6144}; assert!(s.approximate_parameters()>380_000_000); assert!(s.validate().is_ok()); } }

