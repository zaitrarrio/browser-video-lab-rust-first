use anyhow::{bail, Result};
use burn::backend::{ndarray::NdArrayDevice, NdArray};
use burn::module::Module;
use burn::record::{BinFileRecorder, FullPrecisionSettings, NamedMpkFileRecorder};
use clap::{Parser, Subcommand};
use std::{fs, path::PathBuf};
use video_contract::{validate_cache, StudentSpec};
use video_student::{quant::quantize_module, BrowserVideoStudent};

type Cpu = NdArray<f32>;

#[derive(Parser)]
struct App { #[command(subcommand)] command: Command }

#[derive(Subcommand)]
enum Command {
    /// Validate a teacher cache manifest and its shards.
    ValidateCache { path: PathBuf },
    /// Print the approximate student parameter count for a spec.
    Estimate { spec: PathBuf },
    /// Quantize a trained student record to a browser bundle (`weights.q{bits}` +
    /// `index.json`). Reads the Burn `.bin`/`.mpk` `video-train` writes — not
    /// safetensors — closing the producer half of the deploy path.
    Quantize {
        #[arg(long)] spec: PathBuf,
        #[arg(long)] weights: PathBuf,
        #[arg(long)] output: PathBuf,
        #[arg(long, default_value_t = 8)] bits: u8,
    },
}

fn main() -> Result<()> {
    match App::parse().command {
        Command::ValidateCache { path } => {
            let m = validate_cache(&path)?;
            println!("validated {} shards from {}", m.shards.len(), m.teacher);
        }
        Command::Estimate { spec } => {
            let s: StudentSpec = serde_json::from_slice(&fs::read(spec)?)?;
            s.validate()?;
            println!("{}", s.approximate_parameters());
        }
        Command::Quantize { spec, weights, output, bits } => quantize(spec, weights, output, bits)?,
    }
    Ok(())
}

fn quantize(spec: PathBuf, weights: PathBuf, output: PathBuf, bits: u8) -> Result<()> {
    if bits != 4 && bits != 8 { bail!("bits must be 4 or 8") }
    let spec: StudentSpec = serde_json::from_slice(&fs::read(&spec)?)?;
    spec.validate()?;
    let device = NdArrayDevice::default();
    let model = BrowserVideoStudent::<Cpu>::new(spec, &device);
    // Burn's `load_file` wants the path without the recorder's extension.
    let ext = weights.extension().and_then(|e| e.to_str()).unwrap_or("").to_owned();
    let stem = weights.with_extension("");
    let loaded = match ext.as_str() {
        "mpk" => model.load_file(stem, &NamedMpkFileRecorder::<FullPrecisionSettings>::default(), &device),
        "bin" | "" => model.load_file(stem, &BinFileRecorder::<FullPrecisionSettings>::default(), &device),
        other => bail!("unknown weights format .{other}; expected .bin or .mpk"),
    }
    .map_err(|e| anyhow::anyhow!("load student record from {}: {e}", weights.display()))?;

    let (blob, index) = quantize_module(&loaded, bits);
    fs::create_dir_all(&output)?;
    fs::write(output.join(format!("weights.q{bits}")), &blob)?;
    fs::write(output.join("index.json"), serde_json::to_vec_pretty(&index)?)?;
    println!("wrote {} tensors, {} bytes to {}", index.len(), blob.len(), output.display());
    Ok(())
}
