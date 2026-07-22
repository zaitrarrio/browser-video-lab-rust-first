use burn::backend::wgpu::{graphics::WebGpu, init_setup_async, RuntimeOptions, WgpuDevice};
use burn::backend::Wgpu;
use burn::tensor::Tensor;
use video_contract::StudentSpec;
use video_student::BrowserVideoStudent;
use wasm_bindgen::prelude::*;

/// Deterministic Gaussian source mirroring the JS `normal()` LCG so browser
/// runs are reproducible for a given seed.
struct Lcg {
    state: u32,
    spare: Option<f32>,
}
impl Lcg {
    fn new(seed: u32) -> Self {
        Self { state: seed, spare: None }
    }
    fn next_f32(&mut self) -> f32 {
        self.state = self.state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        ((self.state >> 8) as f32) / ((1u32 << 24) as f32)
    }
    fn normal(&mut self) -> f32 {
        if let Some(v) = self.spare.take() {
            return v;
        }
        let u = self.next_f32().max(1e-7);
        let v = self.next_f32();
        let r = (-2.0 * u.ln()).sqrt();
        let two_pi_v = 2.0 * core::f32::consts::PI * v;
        self.spare = Some(r * two_pi_v.sin());
        r * two_pi_v.cos()
    }
}

#[wasm_bindgen]
pub struct BrowserModel {
    spec: StudentSpec,
    model: Option<BrowserVideoStudent<Wgpu>>,
}

#[wasm_bindgen]
impl BrowserModel {
    /// Parse and validate a `StudentSpec`. The model itself is built lazily in
    /// [`prepare`], which must run before [`generate`] because WebGPU adapter
    /// acquisition is asynchronous in the browser.
    #[wasm_bindgen(constructor)]
    pub fn new(spec_json: &str) -> Result<BrowserModel, JsError> {
        let spec: StudentSpec = serde_json::from_str(spec_json)?;
        spec.validate().map_err(|e| JsError::new(&e.to_string()))?;
        Ok(Self { spec, model: None })
    }

    /// Acquire the WebGPU device and instantiate the Burn student. Async because
    /// `navigator.gpu.requestAdapter()` returns a promise.
    pub async fn prepare(&mut self) -> Result<(), JsError> {
        let device = WgpuDevice::default();
        init_setup_async::<WebGpu>(&device, RuntimeOptions::default()).await;
        self.model = Some(BrowserVideoStudent::new(self.spec.clone(), &device));
        Ok(())
    }

    pub fn backend(&self) -> String {
        "burn-wgpu".into()
    }

    pub fn parameters(&self) -> f64 {
        self.spec.approximate_parameters() as f64
    }

    /// Run `steps` denoising iterations on a single `side`×`side` latent frame
    /// seeded by `seed`, then decode the first three latent channels to RGBA.
    /// Returns a `side*side*4` byte buffer (surfaced to JS as a `Uint8Array`).
    pub async fn generate(&self, seed: u32, steps: u32, side: usize) -> Result<Vec<u8>, JsError> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| JsError::new("call prepare() before generate()"))?;
        let device = WgpuDevice::default();
        let channels = self.spec.latent_channels;
        let text_width = self.spec.text_width;
        let seq = 8usize;
        let tokens = side * side;
        if tokens > self.spec.max_tokens {
            return Err(JsError::new(&format!(
                "side={side} yields {tokens} tokens > spec.max_tokens={}",
                self.spec.max_tokens
            )));
        }

        let mut rng = Lcg::new(seed);
        let latent_seed: Vec<f32> = (0..channels * tokens).map(|_| rng.normal()).collect();
        let prompt_seed: Vec<f32> = (0..seq * text_width).map(|_| rng.normal()).collect();

        let mut latents = Tensor::<Wgpu, 1>::from_floats(latent_seed.as_slice(), &device)
            .reshape([1, channels, 1, side, side]);
        let prompt = Tensor::<Wgpu, 1>::from_floats(prompt_seed.as_slice(), &device)
            .reshape([1, seq, text_width]);

        let steps = steps.max(1);
        let rate = 1.0 / steps as f32;
        for i in 0..steps {
            let t = 999.0 * (1.0 - i as f32 / steps as f32);
            let timestep = Tensor::<Wgpu, 1>::from_floats([t].as_slice(), &device).reshape([1, 1]);
            let (pred, _hidden) = model.forward(latents.clone(), timestep, prompt.clone());
            latents = latents - pred.mul_scalar(rate);
        }

        let frame = latents.reshape([channels, tokens]);
        let data = frame
            .into_data_async()
            .await
            .map_err(|e| JsError::new(&format!("readback failed: {e:?}")))?;
        let values = data
            .to_vec::<f32>()
            .map_err(|e| JsError::new(&format!("decode failed: {e:?}")))?;

        let mut rgba = vec![0u8; tokens * 4];
        for p in 0..tokens {
            for ch in 0..3 {
                let raw = if ch < channels { values[ch * tokens + p] } else { 0.0 };
                let norm = (raw * 0.5 + 0.5).clamp(0.0, 1.0);
                rgba[p * 4 + ch] = (norm * 255.0) as u8;
            }
            rgba[p * 4 + 3] = 255;
        }
        Ok(rgba)
    }
}
