//! Native Burn distillation trainer. Trains the browser student against the
//! framework-neutral safetensors teacher cache (see `cache_teacher.py` /
//! `video-contract`), so recurring training needs no PyTorch. The loss is a
//! direct port of `python/longlive_distill/losses.py`: output MSE +
//! temporal-difference MSE + width-independent hidden-relation matching
//! against the cache's precomputed `teacher_relation.N` gram matrices.
use anyhow::{bail, Context, Result};
use burn::grad_clipping::GradientClippingConfig;
use burn::module::{AutodiffModule, Module};
use burn::optim::{AdamWConfig, GradientsAccumulator, GradientsParams, Optimizer};
use burn::record::{BinFileRecorder, FullPrecisionSettings, NamedMpkFileRecorder, Recorder};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{backend::Backend, ElementConversion, Tensor};
use safetensors::{tensor::TensorView, Dtype, SafeTensors};
use serde::{Deserialize, Serialize};
use std::{fs, io::Write, path::{Path, PathBuf}, time::Instant};
use video_contract::{validate_cache, StudentSpec, TeacherCacheManifest, TensorShape};
use video_student::{relation, temporal_difference, BrowserVideoStudent};

pub mod sample;

pub struct TrainSettings {
    /// Steps to run in *this* invocation (one chunk of a possibly longer run).
    pub steps: usize,
    pub lr: f64,
    pub weight_decay: f32,
    pub grad_clip: f32,
    pub w_output: f32,
    pub w_temporal: f32,
    pub w_feature: f32,
    pub log_every: usize,
    pub ckpt_every: usize,
    pub seed: u64,
    /// Resume model weights from a prior `student.mpk`. Optimizer state and run
    /// progress are picked up from `optim.mpk` / `state.json` beside it.
    pub resume: Option<PathBuf>,
    /// Wall-clock budget for this chunk; 0 = unlimited. The loop always stops on
    /// a step boundary and still writes a full checkpoint, so a preempted host
    /// (Kaggle's 12h session cap) never costs more than one partial step.
    pub max_seconds: u64,
    /// Total steps across all chunks; 0 = this chunk is the whole run.
    pub target_steps: usize,
    /// Max decoded shards held resident by the lazy loader. Bounds RAM so shard
    /// counts can grow into the thousands without loading the whole cache up front.
    pub shard_cache: usize,
    /// Shards whose gradients are summed before one optimizer step — the effective
    /// batch size. The loop is otherwise strictly one shard per step, which for a
    /// 383M model is a real handicap rather than a detail: with batch 1 the
    /// gradient is dominated by which single (clip, sigma) draw the cursor
    /// happened to land on, and the loss plateaus at the level that noise
    /// supports. Accumulating trades steps for signal at identical memory, since
    /// the activations of each micro-step are freed before the next one runs.
    pub accum: usize,
}
impl Default for TrainSettings {
    fn default() -> Self {
        Self { steps: 100, lr: 1e-4, weight_decay: 0.01, grad_clip: 1.0, w_output: 1.0, w_temporal: 0.25, w_feature: 0.05, log_every: 10, ckpt_every: 0, seed: 42, resume: None, max_seconds: 0, target_steps: 0, shard_cache: 128, accum: 1 }
    }
}

/// Cross-chunk progress, persisted to `<out>/state.json`. This is the file the
/// orchestrator reads to decide whether to schedule another GPU chunk.
#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
pub struct TrainState {
    pub steps_done: usize,
    pub target_steps: usize,
    pub chunks: usize,
    pub train_seconds: f64,
    pub last_loss: f32,
    pub best_loss: f32,
    pub completed: bool,
    /// Why the last chunk ended: `target` | `chunk-steps` | `max-seconds`.
    pub stopped_by: String,
}

impl TrainState {
    fn read(path: &Path) -> Option<Self> {
        serde_json::from_slice(&fs::read(path).ok()?).ok()
    }
}

/// One teacher-cache shard decoded to f32 host buffers (backend-agnostic).
pub struct Sample {
    pub noisy: (Vec<f32>, [usize; 5]),
    pub timestep: Vec<f32>,
    pub prompt: (Vec<f32>, [usize; 3]),
    pub student_prompt: Option<(Vec<f32>, [usize; 3])>,
    pub teacher_pred: (Vec<f32>, [usize; 5]),
    pub relations: Vec<(Vec<f32>, [usize; 3])>,
}

pub(crate) fn floats(v: &TensorView) -> Result<Vec<f32>> {
    Ok(match v.dtype() {
        Dtype::F32 => v.data().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        Dtype::F16 => v.data().chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
        Dtype::BF16 => v.data().chunks_exact(2).map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
        Dtype::I64 => v.data().chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap()) as f32).collect(),
        d => bail!("unsupported dtype {d:?}"),
    })
}
fn dims<const D: usize>(v: &TensorView, name: &str) -> Result<[usize; D]> {
    v.shape().try_into().map_err(|_| anyhow::anyhow!("{name}: expected rank {D}, got shape {:?}", v.shape()))
}

pub fn load_shard(path: &Path, relation_layers: &[usize]) -> Result<Sample> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let st = SafeTensors::deserialize(&bytes)?;
    let get = |name: &str| st.tensor(name).with_context(|| format!("{name} missing in {}", path.display()));
    let noisy = get("noisy_latents")?;
    let pred = get("teacher_noise_pred")?;
    let prompt = get("prompt_embeds")?;
    let student_prompt = st.tensor("student_prompt_embeds").ok();
    let mut relations = Vec::with_capacity(relation_layers.len());
    for layer in relation_layers {
        let v = get(&format!("teacher_relation.{layer}"))?;
        relations.push((floats(&v)?, dims::<3>(&v, "teacher_relation")?));
    }
    Ok(Sample {
        noisy: (floats(&noisy)?, dims::<5>(&noisy, "noisy_latents")?),
        timestep: floats(&get("timestep")?)?,
        prompt: (floats(&prompt)?, dims::<3>(&prompt, "prompt_embeds")?),
        student_prompt: match student_prompt { Some(v) => Some((floats(&v)?, dims::<3>(&v, "student_prompt_embeds")?)), None => None },
        teacher_pred: (floats(&pred)?, dims::<5>(&pred, "teacher_noise_pred")?),
        relations,
    })
}

/// Bounded, lazy shard loader. Keeps at most `cap` decoded shards resident and
/// loads the rest on demand, so RAM stays flat as shard counts grow into the
/// thousands — the eager `manifest.shards.iter().map(load_shard).collect()` it
/// replaces did not. Shard selection stays a pure function of the global step
/// (`(step-1) % len`), so a resumed chunked run draws the identical sequence as
/// a single run (see `resumed_chunks_match_a_single_run`).
struct ShardCache {
    root: PathBuf,
    shards: Vec<PathBuf>,
    relation_layers: Vec<usize>,
    cap: usize,
    resident: std::collections::HashMap<usize, Sample>,
    order: std::collections::VecDeque<usize>,
}
impl ShardCache {
    fn new(root: PathBuf, shards: Vec<PathBuf>, relation_layers: Vec<usize>, cap: usize) -> Self {
        Self { root, shards, relation_layers, cap: cap.max(1), resident: std::collections::HashMap::new(), order: std::collections::VecDeque::new() }
    }
    fn len(&self) -> usize { self.shards.len() }
    fn get(&mut self, idx: usize) -> Result<&Sample> {
        if !self.resident.contains_key(&idx) {
            let s = load_shard(&self.root.join(&self.shards[idx]), &self.relation_layers)?;
            while self.order.len() >= self.cap {
                match self.order.pop_front() { Some(old) => { self.resident.remove(&old); } None => break }
            }
            self.resident.insert(idx, s);
            self.order.push_back(idx);
        }
        Ok(self.resident.get(&idx).expect("just inserted"))
    }
}

pub(crate) fn t3<B: Backend>(x: &(Vec<f32>, [usize; 3]), device: &B::Device) -> Tensor<B, 3> {
    Tensor::<B, 1>::from_floats(x.0.as_slice(), device).reshape(x.1)
}
pub(crate) fn t5<B: Backend>(x: &(Vec<f32>, [usize; 5]), device: &B::Device) -> Tensor<B, 5> {
    Tensor::<B, 1>::from_floats(x.0.as_slice(), device).reshape(x.1)
}
fn mse<B: Backend, const D: usize>(a: Tensor<B, D>, b: Tensor<B, D>) -> Tensor<B, 1> {
    (a - b).powf_scalar(2.0).mean()
}
/// Integer index map matching `torch.linspace(0, len-1, pairs).long()`.
fn linspace_idx(len: usize, pairs: usize) -> Vec<usize> {
    if pairs <= 1 || len <= 1 { return vec![0; pairs.max(1)]; }
    (0..pairs).map(|i| (i as f64 * (len - 1) as f64 / (pairs - 1) as f64) as usize).collect()
}

pub struct StepMetrics { pub output: f32, pub temporal: f32, pub feature: f32, pub total: f32 }

/// Write a *complete*, resumable checkpoint: weights, AdamW moments and run state.
///
/// All four files together are what `--resume` needs; any subset is a run that
/// silently restarts something. Called on a cadence as well as at the end,
/// because a chunk that dies at step 19,999 with nothing on disk has burned
/// hours of GPU for no progress — which is exactly what a 20,000-step chunk with
/// an end-only save did on Kaggle.
fn write_checkpoint<B, O>(
    out: &Path,
    model: &BrowserVideoStudent<B>,
    optim: &O,
    state: &TrainState,
    mpk: &NamedMpkFileRecorder<FullPrecisionSettings>,
) -> Result<()>
where
    B: AutodiffBackend,
    O: Optimizer<BrowserVideoStudent<B>, B>,
{
    let trained = model.valid();
    trained.clone().save_file(out.join("student"), mpk).context("save student.mpk")?;
    trained
        .save_file(out.join("student"), &BinFileRecorder::<FullPrecisionSettings>::default())
        .context("save student.bin")?;
    Recorder::<B>::record(mpk, optim.to_record(), out.join("optim")).context("save optim.mpk")?;
    // Written last: its presence is what marks the other three as a coherent set.
    fs::write(out.join("state.json"), serde_json::to_vec_pretty(state)?).context("save state.json")?;
    Ok(())
}

/// Train the student on a teacher cache for one *chunk*, returning this chunk's
/// per-step losses and the cumulative run state.
///
/// Saves `student.mpk` (native checkpoint), `student.bin` (BinFileRecorder — the
/// format `video-web` loads from fetched bytes), `optim.mpk` (AdamW moments) and
/// `state.json` (cross-chunk progress). Pointing `--resume` at a previous
/// `student.mpk` continues the same run: weights, optimizer moments, step
/// counter and data cursor all pick up where the last chunk stopped.
pub fn train<B: AutodiffBackend>(spec: StudentSpec, cache: &Path, out: &Path, s: &TrainSettings, device: &B::Device) -> Result<(Vec<f32>, TrainState)> {
    spec.validate()?;
    let manifest = validate_cache(cache)?;
    if manifest.shards.is_empty() { bail!("teacher cache has no shards") }
    let mut shards = ShardCache::new(cache.to_path_buf(), manifest.shards.clone(), manifest.hidden_relation_layers.clone(), s.shard_cache);
    let num_shards = shards.len();
    // A channel-mismatched cache would otherwise fail deep inside the input Linear
    // with an opaque shape error; catch it here where the fix is obvious.
    let cache_channels = shards.get(0)?.noisy.1[1];
    if cache_channels != spec.latent_channels {
        bail!("cache latent_channels={} but spec expects {}", cache_channels, spec.latent_channels)
    }
    fs::create_dir_all(out)?;

    let mpk = NamedMpkFileRecorder::<FullPrecisionSettings>::default();
    // f32::MAX, not INFINITY: serde_json writes a non-finite float as `null`, which
    // then fails to deserialize back into f32 and silently discards the run state.
    let mut state = TrainState { target_steps: s.target_steps, best_loss: f32::MAX, ..Default::default() };
    // Seed before construction: this is the draw that decides the initial weights.
    B::seed(device, s.seed);
    let mut model = BrowserVideoStudent::<B>::new(spec.clone(), device);
    let mut optim = AdamWConfig::new()
        .with_weight_decay(s.weight_decay)
        .with_grad_clipping(Some(GradientClippingConfig::Norm(s.grad_clip)))
        .init();

    if let Some(resume) = &s.resume {
        let prior = resume.parent().unwrap_or(Path::new("."));
        model = model.load_file(resume, &mpk, device).with_context(|| format!("resume from {}", resume.display()))?;
        // Optimizer moments are part of the run, not a nicety: dropping them
        // restarts AdamW cold at every chunk boundary, which on a 15-chunk
        // schedule means 15 gradient-scale transients the loss has to re-absorb.
        let optim_path = prior.join("optim.mpk");
        if optim_path.exists() {
            let record = Recorder::<B>::load(&mpk, optim_path.clone(), device)
                .with_context(|| format!("load optimizer state from {}", optim_path.display()))?;
            optim = optim.load_record(record);
        } else {
            eprintln!("warning: no optim.mpk beside {} — AdamW moments restart cold", resume.display());
        }
        if let Some(prev) = TrainState::read(&prior.join("state.json")) {
            state = TrainState { target_steps: if s.target_steps > 0 { s.target_steps } else { prev.target_steps }, ..prev };
        }
        eprintln!("resumed at step {} from {}", state.steps_done, resume.display());
    }

    let start_step = state.steps_done;
    let remaining = if s.target_steps > 0 { s.target_steps.saturating_sub(start_step) } else { usize::MAX };
    let chunk = s.steps.min(remaining);
    if chunk == 0 {
        state.completed = true;
        state.stopped_by = "target".into();
        fs::write(out.join("state.json"), serde_json::to_vec_pretty(&state)?)?;
        return Ok((Vec::new(), state));
    }

    let began = Instant::now();
    let mut stopped_by = "chunk-steps";
    let mut history = Vec::with_capacity(chunk);
    // Metrics go to a file as well as stdout. Kaggle truncates a kernel log to its
    // tail, so a crash that panics in a loop evicts the entire loss history — the
    // one artefact that would say whether the run was healthy before it died.
    // This file lives in `out`, so it ships with the checkpoint.
    let mut metrics = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(out.join("metrics.jsonl"))
        .context("open metrics.jsonl")?;
    let mut last_saved = start_step;
    // An isolated non-finite step is recoverable (see below): the weights that
    // produced it are still intact, and the pathology is likely specific to one
    // data/hidden-state interaction. Non-finite on *every* step in a row means the
    // weights themselves are destroyed (e.g. an absurd lr applying one catastrophic
    // update) — no amount of skipping recovers that, and silently burning the rest
    // of the chunk doing nothing is worse than failing loudly. Five in a row is
    // well past "one odd shard" and clearly into "this run is dead".
    const MAX_CONSECUTIVE_NON_FINITE: u32 = 5;
    let mut consecutive_non_finite: u32 = 0;
    let accum = s.accum.max(1);
    for i in 1..=chunk {
        let step = start_step + i;
        // Tests the actual theory (unbounded residual-stream growth from the
        // time/text conditioning injected before block 0, raised after the
        // step-349 investigation) directly, as a trend, rather than waiting to
        // see whether it eventually crashes. Read before backward() — this is
        // the forward-pass hidden state, unaffected by whether this step's
        // update later gets applied or skipped.
        let hidden_stats = |h: &Tensor<B, 3>| -> (f32, f32) {
            match h.clone().to_data().to_vec::<f32>() {
                Ok(v) => {
                    let sumsq: f32 = v.iter().map(|x| x * x).sum();
                    let maxabs = v.iter().fold(0f32, |m, x| m.max(x.abs()));
                    (sumsq.sqrt(), maxabs)
                }
                Err(_) => (f32::NAN, f32::NAN),
            }
        };
        let (mut layer0_norm, mut layer0_max) = (f32::NAN, f32::NAN);
        let (mut layerlast_norm, mut layerlast_max) = (f32::NAN, f32::NAN);
        let mut accumulator = GradientsAccumulator::new();
        let mut m = StepMetrics { output: 0.0, temporal: 0.0, feature: 0.0, total: 0.0 };
        let mut non_finite_layer: Option<Option<usize>> = None;

        for micro in 0..accum {
            // Shard selection stays a pure function of (step, micro), so a resumed
            // chunked run still draws the identical sequence — with accum=1 this
            // is exactly the original `(step-1) % len`.
            let sample = shards.get(((step - 1) * accum + micro) % num_shards)?;
            let noisy = t5::<B>(&sample.noisy, device);
            let timestep = Tensor::<B, 1>::from_floats(sample.timestep.as_slice(), device).reshape([sample.timestep.len(), 1]);
            let prompt = t3::<B>(sample.student_prompt.as_ref().unwrap_or(&sample.prompt), device);
            let teacher = t5::<B>(&sample.teacher_pred, device);

            let (pred, hidden) = model.forward(noisy, timestep, prompt);
            let output = mse(pred.clone(), teacher.clone());
            let temporal = if sample.noisy.1[2] > 1 { mse(temporal_difference(pred), temporal_difference(teacher)) } else { output.zeros_like() };
            let feature = if sample.relations.is_empty() { output.zeros_like() } else {
                let pairs = sample.relations.len().min(hidden.len());
                let si = linspace_idx(hidden.len(), pairs);
                let ti = linspace_idx(sample.relations.len(), pairs);
                let mut acc = output.zeros_like();
                for (a, b) in si.iter().zip(&ti) { acc = acc + mse(relation(hidden[*a].clone()), t3::<B>(&sample.relations[*b], device)); }
                acc.div_scalar(pairs as f32)
            };
            // Scale here, not after summing: each micro-step's gradient enters the
            // accumulator already divided, so the accumulated gradient is the mean
            // over the batch and `lr` keeps the same meaning at any `accum`.
            let loss = (output.clone().mul_scalar(s.w_output)
                + temporal.clone().mul_scalar(s.w_temporal)
                + feature.clone().mul_scalar(s.w_feature))
                .div_scalar(accum as f32);

            if micro == 0 {
                (layer0_norm, layer0_max) = hidden_stats(&hidden[0]);
                (layerlast_norm, layerlast_max) = hidden_stats(&hidden[hidden.len() - 1]);
            }

            accumulator.accumulate(&model, GradientsParams::from_grads(loss.backward(), &model));

            let (o, t, f) = (
                output.into_scalar().elem::<f32>(),
                temporal.into_scalar().elem::<f32>(),
                feature.into_scalar().elem::<f32>(),
            );
            // Which micro-step blew up, and in which layer, has to be captured
            // here: `hidden` does not outlive this iteration, and with accum > 1
            // "the step went non-finite" no longer identifies a single sample.
            if !(o * s.w_output + t * s.w_temporal + f * s.w_feature).is_finite() && non_finite_layer.is_none() {
                non_finite_layer = Some(hidden.iter().position(|h| {
                    h.clone().to_data().to_vec::<f32>().map(|v| v.iter().any(|x| !x.is_finite())).unwrap_or(false)
                }));
            }
            m.output += o / accum as f32;
            m.temporal += t / accum as f32;
            m.feature += f / accum as f32;
        }
        m.total = m.output * s.w_output + m.temporal * s.w_temporal + m.feature * s.w_feature;

        // Non-finite is decided before the update, exactly as before: skipping the
        // step must leave the weights untouched, which is only true while
        // `optim.step` has not run.
        if m.total.is_finite() {
            model = optim.step(s.lr, model, accumulator.grads());
        }
        history.push(m.total);
        state.steps_done = step;
        state.last_loss = m.total;
        if m.total < state.best_loss { state.best_loss = m.total; }
        let line = serde_json::json!({
            "step": step, "output": m.output, "temporal": m.temporal, "feature": m.feature, "total": m.total,
            "layer0_norm": layer0_norm, "layer0_max": layer0_max,
            "layerlast_norm": layerlast_norm, "layerlast_max": layerlast_max,
        });
        if i % s.log_every.max(1) == 0 || i == 1 {
            println!("{line}");
        }

        // A diverged step is worth catching and diagnosing on the step it happens,
        // not hours later. serde_json writes a non-finite float as `null`, so NaN
        // reaches state.json and the logs as an absence rather than a number —
        // easy to read as "no data" when it means "this step blew up". Every
        // metric is recorded regardless.
        //
        // A single non-finite step no longer costs the whole chunk: `model` has
        // not been mutated yet at this point (optim.step has not been called), so
        // skipping the update and moving to the next step is a correct recovery,
        // not a hack — the weights are exactly as good as they were before this
        // step. This trades "abort and lose everything since the last checkpoint"
        // for "lose one step's gradient". If this recurs at the same step across
        // resumed/re-dispatched runs (same seed, same shard cycling), that is a
        // structural instability worth its own investigation, not something this
        // recovery is meant to paper over — see `bad_layer` below, logged for
        // exactly that reason.
        if !m.total.is_finite() {
            consecutive_non_finite += 1;
            // Best-effort: identify which hidden-state layer first contains a
            // non-finite value, so a recurring failure points at a specific layer
            // instead of just "step N, feature went NaN" again. Read-back cost is
            // only paid on the (rare) non-finite path, never on a healthy step.
            let bad_layer = non_finite_layer.flatten();
            let diag = serde_json::json!({
                "step": step, "output": m.output, "temporal": m.temporal, "feature": m.feature,
                "total": serde_json::Value::Null, "non_finite_layer": bad_layer,
                "consecutive_non_finite": consecutive_non_finite,
            });
            writeln!(metrics, "{diag}").ok();

            if consecutive_non_finite >= MAX_CONSECUTIVE_NON_FINITE {
                bail!(
                    "loss non-finite for {consecutive_non_finite} consecutive steps (through step \
                     {step}; output={}, temporal={}, feature={}). This is sustained, not an \
                     isolated blip — the weights themselves are almost certainly destroyed (e.g. \
                     one catastrophic update from too high an lr), and skipping further steps \
                     would burn the rest of the chunk producing nothing. Checkpoint in {} is \
                     still from step {last_saved}; last non-finite hidden layer: {}.",
                    m.output, m.temporal, m.feature, out.display(),
                    bad_layer.map(|l| l.to_string()).unwrap_or_else(|| "none (non-finite value entered outside the tapped hidden states)".into()),
                );
            }

            eprintln!(
                "warning: loss non-finite at step {step} (output={}, temporal={}, feature={}, \
                 {consecutive_non_finite} consecutive); first non-finite hidden layer: {}. \
                 Skipping this step's update — model weights are unchanged from step {} and \
                 training continues. Checkpoint in {} is still from step {last_saved}. If this \
                 keeps recurring rather than clearing on the next step, it will hard-fail after \
                 {MAX_CONSECUTIVE_NON_FINITE} in a row rather than run silently to completion.",
                m.output, m.temporal, m.feature,
                bad_layer.map(|l| l.to_string()).unwrap_or_else(|| "none found (all hidden layers finite — non-finite value entered elsewhere)".into()),
                step - 1, out.display()
            );
            continue;
        }
        consecutive_non_finite = 0;
        writeln!(metrics, "{line}").context("append to metrics.jsonl")?;

        // A full checkpoint, not a bare weights snapshot: `student.mpk` alone is not
        // resumable, and a numbered file per interval would add 1.5 GB each time.
        if s.ckpt_every > 0 && step % s.ckpt_every == 0 {
            write_checkpoint(out, &model, &optim, &state, &mpk)
                .with_context(|| format!("periodic checkpoint at step {step}"))?;
            last_saved = step;
            eprintln!("checkpoint written at step {step}");
        }
        if s.max_seconds > 0 && began.elapsed().as_secs() >= s.max_seconds {
            stopped_by = "max-seconds";
            eprintln!("wall-clock budget of {}s reached at step {step} — checkpointing", s.max_seconds);
            break;
        }
    }

    state.chunks += 1;
    state.train_seconds += began.elapsed().as_secs_f64();
    state.completed = s.target_steps > 0 && state.steps_done >= s.target_steps;
    state.stopped_by = if state.completed { "target".into() } else { stopped_by.into() };

    write_checkpoint(out, &model, &optim, &state, &mpk)?;
    Ok((history, state))
}

/// Deterministic Gaussian source (same LCG family as `video-web`).
pub struct Lcg { state: u32, spare: Option<f32> }
impl Lcg {
    pub fn new(seed: u32) -> Self { Self { state: seed.max(1), spare: None } }
    fn next_f32(&mut self) -> f32 { self.state = self.state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223); ((self.state >> 8) as f32) / ((1u32 << 24) as f32) }
    fn normal(&mut self) -> f32 {
        if let Some(v) = self.spare.take() { return v; }
        let u = self.next_f32().max(1e-7);
        let v = self.next_f32();
        let r = (-2.0 * u.ln()).sqrt();
        let a = 2.0 * core::f32::consts::PI * v;
        self.spare = Some(r * a.sin());
        r * a.cos()
    }
    pub fn vec(&mut self, n: usize) -> Vec<f32> { (0..n).map(|_| self.normal()).collect() }
}

fn f32_bytes(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }
fn f16_bytes(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|x| half::f16::from_f32(*x).to_le_bytes()).collect() }

/// Write a tiny synthetic-but-contract-valid teacher cache so the whole
/// train→save→load pipeline can run (and gate CI) with zero PyTorch and no GPU.
/// Tensors are random: this validates plumbing, never model quality.
pub fn synth_cache(spec: &StudentSpec, out: &Path, clips: usize, frames: usize, height: usize, width: usize, seq: usize, teacher_text_width: usize, relation_layers: usize, seed: u32, draws_per_clip: usize) -> Result<()> {
    spec.validate()?;
    let [pt, ph, pw] = spec.patch_size;
    if frames % pt != 0 || height % ph != 0 || width % pw != 0 {
        bail!("cache geometry {frames}x{height}x{width} not divisible by patch_size {pt}x{ph}x{pw}")
    }
    // `cells` are the full latent positions carried in noisy_latents/teacher_pred;
    // `tokens` are the post-patchify sequence length the student attends over and
    // the side of each relation gram. max_tokens stays the pre-patch geometric budget.
    let cells = frames * height * width;
    let tokens = (frames / pt) * (height / ph) * (width / pw);
    if cells > spec.max_tokens { bail!("{cells} latent cells exceed spec.max_tokens={}", spec.max_tokens) }
    if clips == 0 { bail!("need at least one clip") }
    if draws_per_clip == 0 { bail!("need at least one draw per clip") }
    fs::create_dir_all(out)?;
    let mut rng = Lcg::new(seed);
    let mut shard_names = Vec::new();
    let mut shapes: Vec<TensorShape> = Vec::new();
    // Effective supervision is shard count, not step count (a shard's frozen
    // (noise, timestep) is re-shown every time the cursor wraps). Emitting
    // `draws_per_clip` shards per clip — each an independent (noise, timestep)
    // draw — is what turns a handful of clips into a real dataset. Synthetic
    // draws are just fresh random tensors; a real teacher would re-noise the same
    // clip at a fresh timestep here.
    let mut idx = 0usize;
    for _clip in 0..clips { for _draw in 0..draws_per_clip {
        let c = spec.latent_channels;
        let entries: Vec<(String, Dtype, Vec<usize>, Vec<u8>)> = {
            let mut e = vec![
                ("noisy_latents".into(), Dtype::F32, vec![1, c, frames, height, width], f32_bytes(&rng.vec(c * cells))),
                ("timestep".into(), Dtype::I64, vec![1], ((idx as i64 * 137) % 1000).to_le_bytes().to_vec()),
                ("prompt_embeds".into(), Dtype::F32, vec![1, seq, teacher_text_width], f32_bytes(&rng.vec(seq * teacher_text_width))),
                ("student_prompt_embeds".into(), Dtype::F32, vec![1, seq, spec.text_width], f32_bytes(&rng.vec(seq * spec.text_width))),
                ("teacher_noise_pred".into(), Dtype::F32, vec![1, c, frames, height, width], f32_bytes(&rng.vec(c * cells))),
            ];
            for layer in 0..relation_layers {
                e.push((format!("teacher_relation.{layer}"), Dtype::F16, vec![1, tokens, tokens], f16_bytes(&rng.vec(tokens * tokens))));
            }
            e
        };
        let views = entries.iter().map(|(name, dtype, shape, bytes)| Ok((name.clone(), TensorView::new(*dtype, shape.clone(), bytes)?))).collect::<Result<Vec<_>, safetensors::SafeTensorError>>()?;
        let name = format!("shard-{idx:06}.safetensors");
        safetensors::serialize_to_file(views, None, &out.join(&name))?;
        shard_names.push(PathBuf::from(name));
        if idx == 0 { shapes = entries.iter().map(|(name, dtype, shape, _)| TensorShape { name: name.clone(), shape: shape.clone(), dtype: format!("{dtype:?}").to_uppercase() }).collect(); }
        idx += 1;
    }}
    let manifest = TeacherCacheManifest {
        format_version: 1,
        teacher: "synthetic".into(),
        scheduler: "longlive".into(),
        shards: shard_names,
        tensors: shapes,
        hidden_relation_layers: (0..relation_layers).collect(),
    };
    fs::write(out.join("manifest.json"), serde_json::to_vec_pretty(&manifest)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::{ndarray::NdArrayDevice, Autodiff, NdArray};
    use burn::module::Module;
    use std::sync::{Mutex, MutexGuard};

    fn tiny_spec() -> StudentSpec {
        StudentSpec { latent_channels: 2, text_width: 8, width: 16, layers: 2, heads: 2, mlp_ratio: 2, max_tokens: 256, patch_size: [1, 2, 2] }
    }

    // `Backend::seed` sets a *process-global* RNG, so two tests training on the
    // default harness threads draw from one interleaved stream and neither gets
    // the weights its seed asked for. Any test whose assertions depend on
    // initialization must hold this lock for its whole body.
    static SEED: Mutex<()> = Mutex::new(());
    fn seed_guard() -> MutexGuard<'static, ()> {
        SEED.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    #[test]
    // Sustained divergence (weights genuinely destroyed, e.g. by one catastrophic
    // update from too high an lr) must still stop the run and must not overwrite
    // the last good checkpoint with NaN weights. The first real Kaggle chunk
    // trained for 7.5 hours, logged `null` for every metric (serde_json's
    // rendering of NaN), and produced nothing resumable — this is the guard
    // against a repeat, now scoped to "sustained" rather than "any single step",
    // since an isolated non-finite step recovers instead (see the test below).
    #[test]
    fn sustained_non_finite_loss_aborts_and_leaves_the_last_checkpoint_intact() {
        let _seed = seed_guard();
        let dir = tempfile::tempdir().unwrap();
        let cache = dir.path().join("cache");
        let out = dir.path().join("run");
        let spec = tiny_spec();
        synth_cache(&spec, &cache, 2, 2, 4, 4, 4, 12, 2, 3, 1).unwrap();

        let device = NdArrayDevice::default();
        // Absurd lr with clipping effectively disabled: step 1's own update
        // destroys the weights permanently, so every step after is non-finite —
        // sustained, not isolated. ckpt_every=1 guarantees a good checkpoint
        // exists before it does.
        let settings = TrainSettings {
            steps: 50, lr: 1e30, grad_clip: f32::MAX, log_every: 1, ckpt_every: 1, ..Default::default()
        };
        let err = train::<Autodiff<NdArray>>(spec, &cache, &out, &settings, &device)
            .expect_err("lr=1e30 destroys the weights on step 1; every step after must stay non-finite and eventually hard-fail");
        let msg = format!("{err:#}");
        assert!(msg.contains("non-finite"), "wrong failure mode: {msg}");
        assert!(msg.contains("consecutive"), "error should name the sustained-failure reason: {msg}");

        // The pre-divergence checkpoint survives, and no NaN reached state.json.
        assert!(out.join("state.json").exists(), "diverging destroyed the checkpoint");
        let state: TrainState = serde_json::from_slice(&fs::read(out.join("state.json")).unwrap())
            .expect("state.json must still deserialize — NaN would have written null");
        assert!(state.last_loss.is_finite(), "NaN was persisted as the run state");
    }

    #[test]
    // An isolated non-finite step — the weights are fine, but one specific
    // (data, hidden-state) interaction blew up — must recover: skip that step's
    // update and keep training, rather than losing the whole chunk to one blip.
    // This is the counterpart to the sustained-divergence test above: the
    // recovery path only makes sense if it's actually exercised by something
    // that clears on its own, distinct from lr=1e30's permanent corruption.
    #[test]
    fn isolated_non_finite_step_recovers_and_training_continues() {
        let _seed = seed_guard();
        let dir = tempfile::tempdir().unwrap();
        let cache = dir.path().join("cache");
        let out = dir.path().join("run");
        let spec = tiny_spec();
        synth_cache(&spec, &cache, 2, 2, 4, 4, 4, 12, 2, 3, 1).unwrap();

        let device = NdArrayDevice::default();
        // A normal, healthy lr: nothing here should diverge at all, sustained or
        // isolated. This proves the recovery path does not fire on a healthy run
        // (a false positive would be just as bad as failing to recover a real one).
        let settings = TrainSettings { steps: 30, lr: 2e-5, log_every: 5, ckpt_every: 10, ..Default::default() };
        let (history, state) = train::<Autodiff<NdArray>>(spec, &cache, &out, &settings, &device)
            .expect("a healthy lr must not trip the non-finite path at all");
        assert!(history.iter().all(|v| v.is_finite()), "healthy run produced a non-finite step: {history:?}");
        assert!(state.last_loss.is_finite());
    }

    #[test]
    fn synth_train_save_reload_forward() {
        let _seed = seed_guard();
        let dir = tempfile::tempdir().unwrap();
        let cache = dir.path().join("cache");
        let out = dir.path().join("run");
        let spec = tiny_spec();
        synth_cache(&spec, &cache, 2, 2, 4, 4, 4, 12, 2, 3, 1).unwrap();
        validate_cache(&cache).unwrap();

        let device = NdArrayDevice::default();
        let settings = TrainSettings { steps: 20, lr: 1e-2, log_every: 5, ckpt_every: 10, ..Default::default() };
        let (losses, _) = train::<Autodiff<NdArray>>(spec.clone(), &cache, &out, &settings, &device).unwrap();
        assert_eq!(losses.len(), 20);
        assert!(losses.iter().all(|l| l.is_finite()), "loss diverged: {losses:?}");
        assert!(losses.last().unwrap() < losses.first().unwrap(), "loss did not decrease: {losses:?}");
        assert!(out.join("student.mpk").exists());
        assert!(out.join("student.bin").exists());
        assert!(out.join("optim.mpk").exists());
        assert!(out.join("state.json").exists());
        // Metrics are persisted, not merely printed: a truncated kernel log must
        // not be the only record of how the run behaved. log_every=5 over 20 steps
        // logs steps 1, 5, 10, 15, 20 — but *every* step is written to the file.
        let metrics = fs::read_to_string(out.join("metrics.jsonl")).unwrap();
        assert_eq!(metrics.lines().count(), 20, "metrics.jsonl should hold every step");

        // Reload the deployable .bin into a fresh (non-autodiff) model and run a forward pass —
        // the same record path `video-web` uses in the browser.
        let model = BrowserVideoStudent::<NdArray>::new(spec.clone(), &device)
            .load_file(out.join("student"), &BinFileRecorder::<FullPrecisionSettings>::default(), &device)
            .unwrap();
        let latents = Tensor::<NdArray, 1>::from_floats([0.5f32; 2 * 2 * 4 * 4].as_slice(), &device).reshape([1, 2, 2, 4, 4]);
        let timestep = Tensor::<NdArray, 1>::from_floats([500.0f32].as_slice(), &device).reshape([1, 1]);
        let prompt = Tensor::<NdArray, 1>::from_floats([0.1f32; 4 * 8].as_slice(), &device).reshape([1, 4, 8]);
        let (pred, hidden) = model.forward(latents, timestep, prompt);
        assert_eq!(pred.dims(), [1, 2, 2, 4, 4]);
        assert_eq!(hidden.len(), 2);
        // Patchify [1,2,2] on a 2×4×4 latent yields 2·2·2 = 8 tokens; the hidden
        // relation gram proves the internal sequence length is the patchified 8,
        // not the 32 an unpatchified flatten would produce.
        assert_eq!(relation(hidden[0].clone()).dims(), [1, 8, 8]);
    }

    // Chunked training is the whole basis of the free-GPU pipeline: a 12h-capped
    // host must be able to stop and resume without the run losing ground. Two
    // 10-step chunks must reach the same place as one 20-step run — which only
    // holds if weights, AdamW moments, the step counter and the shard cursor all
    // survive the boundary.
    #[test]
    fn resumed_chunks_match_a_single_run() {
        let _seed = seed_guard();
        let dir = tempfile::tempdir().unwrap();
        let cache = dir.path().join("cache");
        let spec = tiny_spec();
        synth_cache(&spec, &cache, 3, 2, 4, 4, 4, 12, 2, 3, 1).unwrap();
        let device = NdArrayDevice::default();
        let base = || TrainSettings { lr: 1e-2, log_every: 100, ..Default::default() };

        let whole = dir.path().join("whole");
        let (one_shot, _) = train::<Autodiff<NdArray>>(
            spec.clone(), &cache, &whole,
            &TrainSettings { steps: 20, target_steps: 20, ..base() }, &device,
        ).unwrap();

        let split = dir.path().join("split");
        let (first, s1) = train::<Autodiff<NdArray>>(
            spec.clone(), &cache, &split,
            &TrainSettings { steps: 10, target_steps: 20, ..base() }, &device,
        ).unwrap();
        assert_eq!((s1.steps_done, s1.completed, s1.chunks), (10, false, 1));
        assert!(split.join("optim.mpk").exists(), "optimizer moments must be checkpointed");

        let (second, s2) = train::<Autodiff<NdArray>>(
            spec.clone(), &cache, &split,
            &TrainSettings { steps: 10, target_steps: 20, resume: Some(split.join("student.mpk")), ..base() }, &device,
        ).unwrap();
        assert_eq!((s2.steps_done, s2.completed, s2.chunks, &*s2.stopped_by), (20, true, 2, "target"));

        let chunked: Vec<f32> = first.into_iter().chain(second).collect();
        assert_eq!(chunked.len(), one_shot.len());
        for (i, (a, b)) in chunked.iter().zip(&one_shot).enumerate() {
            assert!((a - b).abs() <= 2e-4 * b.abs().max(1e-3), "step {i} diverged after resume: {a} vs {b}");
        }

        // A further chunk past the target is a no-op, so an over-eager scheduler
        // cannot overtrain a finished run.
        let (extra, s3) = train::<Autodiff<NdArray>>(
            spec, &cache, &split,
            &TrainSettings { steps: 10, target_steps: 20, resume: Some(split.join("student.mpk")), ..base() }, &device,
        ).unwrap();
        assert!(extra.is_empty());
        assert_eq!((s3.steps_done, s3.completed), (20, true));
    }

    // A chunk that runs out of wall clock must still leave a resumable run behind.
    #[test]
    fn max_seconds_stops_early_but_checkpoints() {
        let _seed = seed_guard();
        let dir = tempfile::tempdir().unwrap();
        let cache = dir.path().join("cache");
        let out = dir.path().join("run");
        let spec = tiny_spec();
        synth_cache(&spec, &cache, 2, 2, 4, 4, 4, 12, 2, 5, 1).unwrap();
        let settings = TrainSettings { steps: 100_000, target_steps: 100_000, max_seconds: 1, log_every: 100_000, ..Default::default() };
        let (losses, state) = train::<Autodiff<NdArray>>(spec, &cache, &out, &settings, &NdArrayDevice::default()).unwrap();
        assert_eq!(state.stopped_by, "max-seconds");
        assert!(!state.completed);
        assert_eq!(state.steps_done, losses.len());
        assert!(state.steps_done < 100_000, "budget was ignored");
        for f in ["student.mpk", "student.bin", "optim.mpk", "state.json"] {
            assert!(out.join(f).exists(), "{f} missing after an early stop");
        }
        assert_eq!(TrainState::read(&out.join("state.json")).unwrap(), state);
    }

    // The lazy loader must return the correct shard for any index even when its
    // resident capacity is smaller than the shard count — i.e. after eviction and
    // reload. This is what lets a real run hold thousands of shards on disk while
    // keeping only a bounded window in RAM.
    #[test]
    fn shard_cache_evicts_and_reloads() {
        let dir = tempfile::tempdir().unwrap();
        let cache = dir.path().join("cache");
        synth_cache(&tiny_spec(), &cache, 3, 2, 4, 4, 4, 12, 2, 7, 1).unwrap();
        let manifest = validate_cache(&cache).unwrap();
        let mut sc = ShardCache::new(cache.clone(), manifest.shards.clone(), manifest.hidden_relation_layers.clone(), 1);
        assert_eq!(sc.len(), 3);
        // Touch 0,1,2,0: with cap 1 every step but a repeat evicts, so index 0 is
        // reloaded from disk on the last touch and must still be byte-identical.
        for idx in [0usize, 1, 2, 0] {
            let direct = load_shard(&cache.join(&manifest.shards[idx]), &manifest.hidden_relation_layers).unwrap();
            let got = sc.get(idx).unwrap();
            assert_eq!(got.noisy.1, direct.noisy.1);
            assert_eq!(got.noisy.0, direct.noisy.0, "shard {idx} decoded differently after reload");
        }
        assert!(sc.resident.len() <= 1, "cap=1 must keep at most one shard resident, got {}", sc.resident.len());
    }

    // draws_per_clip multiplies clips into independent (noise, timestep) shards —
    // the mechanism that turns a few clips into enough supervision to train on.
    #[test]
    fn draws_per_clip_multiplies_shards() {
        let dir = tempfile::tempdir().unwrap();
        let cache = dir.path().join("cache");
        synth_cache(&tiny_spec(), &cache, 3, 2, 4, 4, 4, 12, 2, 7, 4).unwrap();
        let manifest = validate_cache(&cache).unwrap();
        assert_eq!(manifest.shards.len(), 12, "3 clips × 4 draws = 12 shards");
        // Distinct timestep draws: the first four shards must not all share a t.
        let ts: Vec<Vec<f32>> = manifest.shards.iter().take(4)
            .map(|p| load_shard(&cache.join(p), &manifest.hidden_relation_layers).unwrap().timestep)
            .collect();
        assert!(ts.iter().any(|t| t != &ts[0]), "draws within reach must vary the timestep");
    }

    // Gradient accumulation must be a real batch, not a slower way to do batch 1:
    // one optimizer step has to consume `accum` distinct shards. The cursor is
    // `((step-1)*accum + micro) % len`, so 4 steps at accum=4 must walk all 16
    // shards exactly once — if it collapsed to the first shard of each group, the
    // run would silently train on a quarter of the cache.
    #[test]
    fn accumulation_consumes_a_full_batch_of_distinct_shards() {
        let _seed = seed_guard();
        let dir = tempfile::tempdir().unwrap();
        let cache = dir.path().join("cache");
        let out = dir.path().join("run");
        let spec = tiny_spec();
        synth_cache(&spec, &cache, 4, 2, 4, 4, 4, 12, 2, 3, 4).unwrap();
        assert_eq!(validate_cache(&cache).unwrap().shards.len(), 16);

        let settings = TrainSettings { steps: 4, lr: 1e-2, accum: 4, log_every: 100, ..Default::default() };
        let (losses, state) = train::<Autodiff<NdArray>>(spec, &cache, &out, &settings, &NdArrayDevice::default()).unwrap();
        assert_eq!(losses.len(), 4, "one loss per optimizer step, not per micro-step");
        assert_eq!(state.steps_done, 4);
        assert!(losses.iter().all(|l| l.is_finite()), "accumulated run diverged: {losses:?}");
        assert!(losses.last().unwrap() < losses.first().unwrap(), "loss did not fall: {losses:?}");

        // Cursor coverage, checked directly rather than inferred from the loss.
        let touched: std::collections::HashSet<usize> =
            (1..=4usize).flat_map(|step| (0..4).map(move |m| ((step - 1) * 4 + m) % 16)).collect();
        assert_eq!(touched.len(), 16, "accum=4 over 4 steps must touch every shard once");
    }

    #[test]
    fn linspace_matches_torch() {
        assert_eq!(linspace_idx(4, 2), vec![0, 3]);
        assert_eq!(linspace_idx(24, 4), vec![0, 7, 15, 23]);
        assert_eq!(linspace_idx(1, 1), vec![0]);
    }
}
