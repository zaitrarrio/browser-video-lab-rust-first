//! Per-tensor symmetric weight quantization for the browser student, plus the
//! producer/consumer that make it usable end to end.
//!
//! The student trains and saves as a Burn record (`student.bin`/`student.mpk`),
//! ~1.53 GB at F32 for the 390M spec — not a browser download. This module
//! walks the model's float parameters in Burn module order, quantizes each to
//! int8 or int4, and writes a flat weight blob plus an ordered index. The
//! consumer rebuilds the same model from the spec and maps the dequantized
//! tensors back on, in the same order — no tensor names, so the two ends can
//! never drift on naming. int4 is a factor of eight smaller than F32.

use crate::BrowserVideoStudent;
use burn::module::{Module, ModuleMapper, ModuleVisitor, Param};
use burn::tensor::{backend::Backend, Tensor};
use serde::{Deserialize, Serialize};

/// One quantized tensor's metadata. `offset`/`length` index the shared byte
/// blob (`weights.q{bits}`); entries are stored in module-visitation order and
/// consumed in the same order, so `name` is a human-readable ordinal only.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QTensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub scale: f32,
    pub offset: u64,
    pub length: u64,
    pub bits: u8,
}

/// Symmetric quantize one tensor: a single `scale`, no zero point. Returns the
/// scale and the packed bytes — int8 is one byte per value; int4 packs two
/// values per byte, each shifted +8 into a 0..15 nibble.
pub fn quantize_values(values: &[f32], bits: u8) -> (f32, Vec<u8>) {
    let qmax = if bits == 8 { 127.0 } else { 7.0 };
    let max = values.iter().fold(0f32, |m, x| m.max(x.abs()));
    let scale = (max / qmax).max(1e-12);
    let packed = if bits == 8 {
        values.iter().map(|x| (x / scale).round().clamp(-127.0, 127.0) as i8 as u8).collect()
    } else {
        let mut out = Vec::with_capacity(values.len().div_ceil(2));
        for pair in values.chunks(2) {
            let a = ((pair[0] / scale).round().clamp(-7.0, 7.0) as i8 + 8) as u8;
            let b = pair.get(1).map(|x| ((x / scale).round().clamp(-7.0, 7.0) as i8 + 8) as u8).unwrap_or(8);
            out.push((b << 4) | (a & 0x0F));
        }
        out
    };
    (scale, packed)
}

/// Inverse of [`quantize_values`] for one tensor's slice of the blob.
pub fn dequantize_values(entry: &QTensor, blob: &[u8]) -> Vec<f32> {
    let start = entry.offset as usize;
    let len = entry.length as usize;
    let mut out = Vec::with_capacity(len);
    if entry.bits == 8 {
        for i in 0..len {
            out.push((blob[start + i] as i8) as f32 * entry.scale);
        }
    } else {
        for i in 0..len {
            let byte = blob[start + i / 2];
            let nibble = if i % 2 == 0 { byte & 0x0F } else { byte >> 4 };
            out.push((nibble as i8 - 8) as f32 * entry.scale);
        }
    }
    out
}

/// Collects each float parameter's shape and values in module order.
struct Collector {
    tensors: Vec<(Vec<usize>, Vec<f32>)>,
}
impl<B: Backend> ModuleVisitor<B> for Collector {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        let tensor = param.val();
        let shape = tensor.dims().to_vec();
        let values = tensor.to_data().to_vec::<f32>().expect("student params are f32");
        self.tensors.push((shape, values));
    }
}

/// Quantize a trained student to a flat `weights.q{bits}` blob and an ordered
/// index. Run on a CPU backend (`to_data` reads back synchronously).
pub fn quantize_module<B: Backend>(model: &BrowserVideoStudent<B>, bits: u8) -> (Vec<u8>, Vec<QTensor>) {
    let mut collector = Collector { tensors: Vec::new() };
    model.visit(&mut collector);
    let mut blob = Vec::new();
    let mut index = Vec::with_capacity(collector.tensors.len());
    for (i, (shape, values)) in collector.tensors.iter().enumerate() {
        let (scale, packed) = quantize_values(values, bits);
        index.push(QTensor {
            name: format!("p{i:04}"),
            shape: shape.clone(),
            scale,
            offset: blob.len() as u64,
            length: values.len() as u64,
            bits,
        });
        blob.extend_from_slice(&packed);
    }
    (blob, index)
}

/// Replaces each float parameter with its dequantized value, in the same module
/// order the index was written. The incoming model supplies only structure and
/// device; its (random) weights are overwritten.
struct Dequantizer<'a> {
    index: &'a [QTensor],
    blob: &'a [u8],
    cursor: usize,
}
impl<B: Backend> ModuleMapper<B> for Dequantizer<'_> {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let entry = self.index[self.cursor].clone();
        self.cursor += 1;
        param.map(|tensor| {
            let device = tensor.device();
            let dims = tensor.dims();
            let values = dequantize_values(&entry, self.blob);
            Tensor::<B, 1>::from_floats(values.as_slice(), &device).reshape(dims)
        })
    }
}

/// Load a quantized bundle onto a freshly-constructed student. `index` must have
/// been produced by [`quantize_module`] for the same spec.
pub fn dequantize_module<B: Backend>(model: BrowserVideoStudent<B>, index: &[QTensor], blob: &[u8]) -> BrowserVideoStudent<B> {
    let mut dequantizer = Dequantizer { index, blob, cursor: 0 };
    model.map(&mut dequantizer)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::{ndarray::NdArrayDevice, NdArray};
    use burn::tensor::backend::Backend;
    use video_contract::StudentSpec;
    type Cpu = NdArray<f32>;

    fn spec() -> StudentSpec {
        StudentSpec { latent_channels: 2, text_width: 4, width: 16, layers: 2, heads: 2, mlp_ratio: 2, max_tokens: 64, patch_size: [1, 2, 2], per_block_conditioning: false }
    }
    fn inputs(device: &NdArrayDevice) -> (Tensor<Cpu, 5>, Tensor<Cpu, 2>, Tensor<Cpu, 3>) {
        let latents = Tensor::<Cpu, 1>::from_floats([0.4f32; 2 * 1 * 4 * 4].as_slice(), device).reshape([1, 2, 1, 4, 4]);
        let timestep = Tensor::<Cpu, 1>::from_floats([500.0f32].as_slice(), device).reshape([1, 1]);
        let prompt = Tensor::<Cpu, 1>::from_floats([0.1f32; 3 * 4].as_slice(), device).reshape([1, 3, 4]);
        (latents, timestep, prompt)
    }

    // The whole point of Phase 1: a quantized bundle must reconstruct the trained
    // model, not just round-trip bytes. Dequantizing onto a *differently*-seeded
    // fresh model must land its forward pass far closer to the original than that
    // random model was — proving the weights, not the init, drive the output. int4
    // is lossier than int8 but still recovers most of the signal, at half the size.
    #[test]
    fn quantized_bundle_reconstructs_the_model() {
        let _rng = crate::MODEL_RNG.lock().unwrap_or_else(|e| e.into_inner());
        let device = NdArrayDevice::default();
        let (latents, timestep, prompt) = inputs(&device);

        <Cpu as Backend>::seed(&device, 11);
        let trained = BrowserVideoStudent::<Cpu>::new(spec(), &device);
        let (reference, _) = trained.forward(latents.clone(), timestep.clone(), prompt.clone());

        // A different-seed random model: the distance to beat.
        <Cpu as Backend>::seed(&device, 999);
        let (baseline, _) = BrowserVideoStudent::<Cpu>::new(spec(), &device).forward(latents.clone(), timestep.clone(), prompt.clone());
        let baseline_err: f32 = (reference.clone() - baseline).abs().mean().into_scalar();
        assert!(baseline_err > 0.0);

        let mut err = Vec::new();
        for bits in [8u8, 4] {
            let (blob, index) = quantize_module(&trained, bits);
            <Cpu as Backend>::seed(&device, 999);
            let fresh = BrowserVideoStudent::<Cpu>::new(spec(), &device);
            let restored = dequantize_module(fresh, &index, &blob);
            let (out, _) = restored.forward(latents.clone(), timestep.clone(), prompt.clone());
            let e: f32 = (reference.clone() - out).abs().mean().into_scalar();
            assert!(e < 0.2 * baseline_err, "q{bits} did not reconstruct: err {e} vs baseline {baseline_err}");
            err.push(e);
        }
        assert!(err[0] <= err[1] + 1e-6, "int8 should be at least as accurate as int4");

        // int4 packs ~2x denser than int8.
        let (b8, _) = quantize_module(&trained, 8);
        let (b4, _) = quantize_module(&trained, 4);
        assert!(b4.len() <= b8.len() / 2 + 1, "int4 blob should be ~half int8 ({} vs {})", b4.len(), b8.len());
    }
}
