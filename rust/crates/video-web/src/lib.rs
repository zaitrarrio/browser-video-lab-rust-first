use burn::backend::Wgpu;use video_contract::StudentSpec;use video_student::BrowserVideoStudent;use wasm_bindgen::prelude::*;
#[wasm_bindgen]
pub struct BrowserModel{model:BrowserVideoStudent<Wgpu>}
#[wasm_bindgen]
impl BrowserModel{
 #[wasm_bindgen(constructor)]pub fn new(spec_json:&str)->Result<BrowserModel,JsError>{let spec:StudentSpec=serde_json::from_str(spec_json)?;spec.validate().map_err(|e|JsError::new(&e.to_string()))?;let device=Default::default();Ok(Self{model:BrowserVideoStudent::new(spec,&device)})}
 pub fn backend(&self)->String{"burn-wgpu".into()}
}

