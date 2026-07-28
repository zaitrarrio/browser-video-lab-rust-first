# Shipping a Video Diffusion Model to the Browser Tab

### Framework-Neutral, Free-Tier Distillation of Heavyweight Generative Teachers into Quantized WebGPU Students

**A white paper from the Browser Video Lab project**

---

## Abstract

State-of-the-art open text-to-video diffusion models — Wan2.1, LongLive, and their
successors — are tens of gigabytes of weights bound to a CUDA/PyTorch stack. None can
run in a web browser, where there is no PyTorch, GPU memory is a fraction of installed
VRAM, and even the text encoder is too large to download. We present a distillation
architecture that takes such a teacher to a **single ~383M-parameter student that runs
entirely inside a browser tab, driven by a typed prompt, with no server inference and no
ONNX Runtime**, quantized to ~191 MB (int4). The method's organizing principle is that
the teacher is a *one-time, framework-neutral target factory*: it runs in inference only
to manufacture a cache of supervision shards, after which the student trains, quantizes,
and deploys with no PyTorch in the loop. Three ingredients make the cross-stack gap
crossable: a **width-independent hidden-relation (Gram) loss** that lets a 512-wide
student match a 4096-wide teacher's internal structure; **patchify alignment** onto the
teacher's token geometry so that loss can be computed at all; and **cross-encoder
conditioning**, in which the student learns to reproduce a T5-XXL-conditioned teacher
from a small in-browser umt5-small encoder. The entire pipeline runs on free-tier
hardware — teacher caching on a Kaggle CPU kernel, resumable student training on a Kaggle
GPU — and is CI-driven end to end. We situate the work against knowledge distillation and
diffusion-distillation literature, describe the specific and complementary role of Pruna
Smash as a teacher-side accelerator, and argue the same spine transfers to image and text
models with the video-specific pieces removed. We are explicit about status: every
mechanism is implemented and unit-tested, but a quality-validated student from a real
teacher cache has not yet been produced, and we make no quality claim.

---

## 1. Introduction

The generative-model deployment story has bifurcated. Frontier models grow, and their
serving cost with them; meanwhile the browser has quietly become a capable compute
target through WebGPU and WebAssembly. The gap between the two is large: a model that
needs 40 GB and a datacenter GPU is not obviously related to a model that needs to fit
in a tab and touch only the visitor's own hardware. Bridging that gap for *generative
video* — the heaviest of the common modalities — is the subject of this paper.

Our target is deliberately concrete. Not "a smaller model," but *this* deliverable: a
text-to-video model you can type at, that runs in Chrome with WebGPU, needs no backend,
downloads on the order of a large web asset rather than a model checkpoint, and is
produced reproducibly from an openly-licensed teacher on hardware anyone can rent for
free. The engineering of the browser runtime is the easy half. The hard half is that the
teacher and any browser-viable student disagree on **framework** (PyTorch/CUDA vs.
Rust/WebGPU), **width** (4096 vs. 512 text conditioning), **architecture** (a causal DiT
vs. a compact chunk-bidirectional mixer), **text encoder** (umT5-XXL vs. umt5-small),
and **inference regime**. A naive online distillation loop would have to hold both models
in one process and reconcile all five at once.

Our contribution is an architecture that refuses to do that. We decouple teacher and
student through a **framework-neutral supervision cache**, reducing one intractable
cross-stack problem to two tractable ones, and we introduce the loss and geometry
machinery that lets a student this different from its teacher still learn from it.

## 2. Background and prior work

**Knowledge distillation.** Training a small "student" to match a large "teacher"
originates with Hinton et al. (2015), which matched softened output distributions.
*Feature-based* distillation extended the idea to internal representations: FitNets
(Romero et al., 2015) matched intermediate activations; Attention Transfer (Zagoruyko &
Komodakis, 2017) matched attention maps. Most relevant to us is *relational* distillation
— matching the geometry of representations rather than their values: "A Gift from
Knowledge Distillation" (Yim et al., 2017) used flow-between-layers Gram matrices, and
Relational KD (Park et al., 2019) matched pairwise structure. The Gram matrix itself as a
carrier of style/structure comes from neural style transfer (Gatys et al., 2016). Our
relation loss is a Gram-matrix relational distillation, chosen specifically because a
Gram matrix is indexed by tokens and is therefore **invariant to the width mismatch**
between a 4096-wide teacher and a 512-wide student.

**Diffusion distillation.** A parallel line compresses the *sampling* of diffusion
models: Progressive Distillation (Salimans & Ho, 2022), Consistency Models (Song et al.,
2023) and Latent Consistency Models (Luo et al., 2023), Rectified Flow / InstaFlow, and
Distribution Matching Distillation (DMD/DMD2, Yin et al., 2024), along with adversarial
approaches such as ADD / SD-Turbo (Sauer et al., 2023). These mostly reduce *step count*
while keeping the architecture. LongLive's upstream recipe uses AR teacher-forcing
followed by DMD. Our work is complementary and orthogonal: we change the *architecture,
framework, width, and encoder* for deployment, and could stack a few-step scheme on top.

**Video diffusion teachers.** Wan2.1/2.2 (Alibaba) are open DiT text-to-video bases;
LongLive (Efficient-Large-Model / NVIDIA) adds causal, streaming, long-horizon
generation with KV-recache on top of a Wan base; Kling's MemFlow adds a streaming memory
bank. Licensing differs sharply and drives our teacher choice (§4).

**Browser and edge ML.** ONNX Runtime Web, Transformers.js, and WebLLM / MLC brought
transformer inference to the browser via WebGPU and WASM. We build the recurring path on
**Burn**, a Rust deep-learning framework whose WGPU backend (via CubeCL) compiles to
WebAssembly, giving a single codebase for native and browser execution with no ONNX
export step. Weight-only int4/int8 quantization for deployment follows the now-standard
practice popularized by GPTQ, AWQ, and llama.cpp/GGUF.

**Model compression toolkits.** Pruna AI's *Smash* packages compilers, kernels, and
quantizers behind one API. We use it in its native-CUDA role, on the teacher — see §6.

## 3. Method

### 3.1 The teacher as a one-time target factory

The pivotal decision is to connect teacher and student through data, not gradients. The
frozen teacher runs **once**, in inference only, over prompts × sampled `(noise,
timestep)` draws. Each draw yields a `safetensors` *supervision shard* holding the noisy
latent, the timestep, the teacher's denoising prediction, both text-encoder embeddings of
the caption, and a set of hidden-state **relation matrices**. The student then trains
against these shards in Rust/Burn with **no PyTorch in the loop, ever**.

This buys two independent, cheap problems in place of one hard one. The teacher pass is
inference-only — patience rather than an H100, and viable on CPU. The student loop is a
compiled binary that runs happily on a free-tier GPU whose ageing CUDA is fatal to modern
PyTorch but irrelevant to Burn. And it makes the student *permanently* independent of the
teacher: at deploy time the browser needs only a spec JSON and a weight bundle.

### 3.2 The distillation objective

Per training step, three losses are combined (weights `1.0 / 0.25 / 0.05`; AdamW,
weight-decay `0.01`, gradient-clip norm `1.0`, learning rate `2e-5`):

1. **Output matching** — MSE between the student's and the frozen teacher's denoising
   prediction for the cached `(latent, timestep, prompt)`. The primary signal.
2. **Temporal-difference matching** — MSE between frame-to-frame differences of the two
   predictions, supervising motion directly (skipped for single-frame latents).
3. **Width-independent relation matching** — for selected layers, the student's hidden
   states `H ∈ ℝ^{tokens×width}` are reduced to a Gram matrix `R = H Hᵀ ∈
   ℝ^{tokens×tokens}` and matched against the teacher's cached Gram with MSE. Because `R`
   is indexed by tokens, the width cancels: a 512-wide student can match a 4096-wide
   teacher's internal structure. This is the mechanism that makes cross-width,
   cross-architecture distillation coherent.

### 3.3 Patchify alignment — the geometry that makes (3) possible

The Gram loss only broadcasts if both sides emit the **same token count**. Every
Wan-family DiT patchifies `[1,2,2]`; an un-patchified student at latent shape
`[1,16,4,32,48]` would produce 6144 tokens against the teacher's 1536, and the loss would
fail on the first real shard. Giving the student the same `[1,2,2]` patchify lands both at
`1536×1536` — and, as a bonus, cuts student self-attention cost 16×. The reason this was
subtle: a synthetic-cache generator that produces the teacher side *at the student's own
token count* hides the mismatch entirely, so a green test suite proved plumbing, not
correctness. "Synthetic ≠ real geometry" is now the project's first principle.

### 3.4 Cross-encoder conditioning

The teacher conditions on umT5-XXL (4096-dim, ~11 GB) — unshippable to a tab. The student
conditions on umt5-**small** (512-dim), the same encoder the browser already tokenizes
with, and **its text input `Linear` (512 → width) is the learned projection** — no extra
module. Each shard carries both embeddings of the *same* caption; the trainer routes each
to its model. The student thus learns to reproduce a T5-XXL-conditioned response from a
umt5-small-conditioned input. The cost, stated plainly, is weaker prompt adherence than
the teacher — the accepted price of full in-browser operation.

### 3.5 Supervision economics

Because the cache freezes `(latent, timestep)` per shard and training cycles shards
modulo their count, **effective supervision equals shard count, not step count**. The
unit of supervision is therefore a *timestep draw*, not a *clip*: emitting 8–32 draws per
prompt converts a few hundred prompts into a real dataset **at the cost of teacher
forward passes rather than a video corpus**. Caching 4–6 relation layers instead of all
30, and a bounded lazy shard loader in place of an eager full-RAM load, keep each shard
near ~1 MB and let the dataset scale.

### 3.6 Deployment

The student is exposed to JavaScript through `wasm-bindgen`
(`new BrowserModel(spec)` → `prepare()` → `generate(seed, steps, side)`), compiled with
`wasm-pack --target web`, running its denoiser directly on WebGPU via Burn — **no ONNX
Runtime**. Weights ship as an int4/int8 bundle (`weights.q{bits}` + an ordered
`index.json`) that the runtime dequantizes onto a fresh model **in module order, with no
tensor names**, so producer and consumer cannot silently drift. int4 is 8× smaller than
F32: ~1.53 GB → ~191 MB.

## 4. Choosing the teacher

Teacher choice is dominated by licensing, not capability. LongLive-1.3B (the model the
project was designed around) is **CC-BY-NC-SA-4.0** — non-commercial and share-alike,
constraints that plausibly follow a student distilled from it. **Wan2.1-T2V-1.3B** is
**Apache-2.0**, imposes no downstream restriction, is smaller (17.5 GB vs. 26 GB), is
`diffusers`-native (hidden states reachable through standard hooks), and was already
partially wired into the repo. We therefore distil from **Wan2.1** ("Plan B"). The cost
is that Wan2.1 is a bidirectional short-clip base, so long-horizon behavior must come from
architecture rather than from the teacher; the trade is deliberate and documented.

## 5. Free-tier training system

Distillation runs on free hardware as two GitHub Actions pipelines over Kaggle:

- **Teacher cache — Kaggle CPU** (`cache.yml`). A CPU script kernel encodes captions with
  both encoders, runs the frozen Wan2.1 teacher over each `(clip, draw)`, and versions the
  shards back as a Kaggle dataset. CPU is correct here precisely because the teacher pass
  is inference-only; it never competes for the GPU quota.
- **Student training — Kaggle GPU** (`train.yml`). Training proceeds in **resumable
  chunks**: `--resume` restores weights, AdamW moments, the step counter, and the shard
  cursor, so N short sessions equal one long run (asserted in a regression test). A
  content-addressed toolchain cache keyed on the Rust sources avoids recompiling
  Burn/CUDA; multi-GB checkpoints are versioned back *from inside* the kernel, so CI only
  ever downloads a few-KB `state.json`. On completion, weights publish to a rolling
  release.

Cache on CPU, train on GPU, quantize, bundle — the whole path is CI-driven and needs no
browser and no local GPU.

## 6. The role of Pruna Smash

Pruna **Smash** is a **teacher-side, native-CUDA accelerator**, and nothing else in this
system. `smash_wan21.py` compresses Wan2.1 with a compiler stack (`torch_compile` by
default, optional kernels such as `flash_attn3` where supported) and persists it;
`benchmark_wan21.py` times it. Its job is to make the *expensive* part — the thousands of
frozen-teacher forward passes that manufacture the cache — faster and cheaper on CUDA.

Two boundaries are intentional and load-bearing:

- **Pruna optimizes the teacher, never the student.** Smash compiler outputs are native
  CUDA artifacts, *not* WebGPU models, and are kept server-side. The shippable browser
  artifact is produced by Burn int4/int8 quantization, a completely separate mechanism.
- **It is CUDA-only, hence optional on the free tier.** The Kaggle CPU cache kernel
  deliberately omits Pruna. Smash is the switch you throw when a GPU teacher pass is
  available and throughput is the constraint.

The system thus runs **two complementary compression regimes on two different models**:
Pruna Smash (native CUDA) to accelerate the *teacher* that builds the cache, and Burn
int4/int8 (WebGPU) to shrink the *student* that ships. They never touch the same tensor,
and conflating them — trying to run a Smashed teacher in a browser — is exactly the
mistake the design is structured to prevent.

## 7. Applications

**Video (this work).** In-browser, prompt-driven text-to-video with no backend: a
generative-video effect or preview that runs on the visitor's own GPU, ships as a web
asset, and costs the operator nothing per generation. Privacy (nothing leaves the tab),
cost (no GPU serving), and offline capability follow directly.

**Image.** The nearest transfer. Drop the temporal loss; keep output + relation matching.
Distil an SD / SDXL / Flux teacher into a tiny browser image student through the same
cache → train → quantize → WASM spine. The repository already carries an SD-Turbo WebGPU
runtime and an ONNX student path, so the deployment target exists.

**Text / LLMs.** Cache a teacher LLM's logits and hidden relations over a prompt set;
distil a small student with the same **width-independent relation matching**; quantize to
int4; run in-browser in the WebLLM style. The diffusion-specific losses (output-as-noise,
temporal) fall away, but the transferable spine — *cache once, match relations, ship
quantized WASM* — carries over intact, and the cross-encoder trick generalizes to
shrinking the tokenizer/embedding side.

**The general pattern.** Across modalities the reusable idea is: *manufacture a
framework-neutral supervision cache from a frozen teacher once; distil a small,
framework-native student with a width-independent relation loss; quantize to int4; deploy
as WASM/WebGPU on free-tier hardware.* The teacher's framework, size, and serving cost
become a one-time offline expense; the recurring artifact is small, portable, and
private.

## 8. Novelty and positioning

The individual pieces have prior art — relational KD, diffusion distillation,
quantization, WebGPU inference. The contribution is their **composition into a single
reproducible spine** that clears a five-way teacher/student mismatch (framework, width,
architecture, encoder, regime) at once, plus two specific enablers:

1. **A framework-neutral supervision cache** as the sole teacher↔student interface,
   turning cross-stack distillation into an inference-only pass plus a PyTorch-free loop,
   and making free-tier hardware sufficient.
2. **Width-independent relation matching under patchify-aligned token geometry**, which is
   what lets a 512-wide, differently-architected browser student learn a 4096-wide DiT
   teacher's internal structure — with the honest observation that synthetic supervision
   masks exactly the geometry bugs that matter.

## 9. Limitations and honest status

We claim an architecture and a working pipeline, **not** a quality result. The system is
architecturally complete and operationally early: real-geometry alignment, the scalable
cache, end-to-end int4/int8 quantization, and genuine prompt conditioning are all
implemented and unit-tested (different prompts provably yield different output). The real
Wan2.1 CPU-cache pipeline exists. What does *not* yet exist is a validated, quality-graded
student trained from a real teacher cache: the most recent GPU run diverged (relation loss
→ NaN) from a learning rate left at `1e-4` against a reference `2e-5`, now corrected, and
the first end-to-end real run is only now unblocked. Known intrinsic trade-offs stand
regardless of tuning: umt5-small conditioning will trail T5-XXL on prompt adherence, and a
short-clip Wan teacher supervises no long-horizon behavior. No benchmark numbers are
reported because none have been earned.

## 10. Conclusion

Getting a heavyweight generative-video teacher into a browser tab is less a runtime
problem than a *training-architecture* problem: the two models share nothing that a
normal distillation loop assumes. By making the teacher a one-time, framework-neutral
target factory, matching internal structure through a width-independent relation loss over
aligned token geometry, conditioning the student on a browser-sized encoder, and shipping
an int4 WASM/WebGPU artifact — all on free-tier hardware, with Pruna Smash accelerating
the teacher and Burn quantization shrinking the student — the cross-stack gap becomes a
sequence of cheap, reproducible steps. The same spine, with its two video-specific losses
removed, points directly at image and text. What remains is to run it at scale and measure
what comes out.

---

## References

1. Hinton, Vinyals, Dean. *Distilling the Knowledge in a Neural Network.* 2015.
2. Romero et al. *FitNets: Hints for Thin Deep Nets.* ICLR 2015.
3. Zagoruyko, Komodakis. *Paying More Attention to Attention (Attention Transfer).* ICLR 2017.
4. Yim et al. *A Gift from Knowledge Distillation (FSP / Gram-flow).* CVPR 2017.
5. Park et al. *Relational Knowledge Distillation.* CVPR 2019.
6. Gatys, Ecker, Bethge. *Image Style Transfer Using CNNs (Gram matrices).* CVPR 2016.
7. Salimans, Ho. *Progressive Distillation for Fast Sampling of Diffusion Models.* ICLR 2022.
8. Song et al. *Consistency Models.* ICML 2023.
9. Luo et al. *Latent Consistency Models.* 2023.
10. Yin et al. *One-step Diffusion with Distribution Matching Distillation (DMD/DMD2).* 2024.
11. Sauer et al. *Adversarial Diffusion Distillation (SD-Turbo).* 2023.
12. Wan-AI. *Wan2.1 / Wan2.2 Text-to-Video (Diffusers).* Apache-2.0 model cards.
13. Efficient-Large-Model / NVIDIA. *LongLive: causal streaming long-video generation.*
14. Frantar et al. *GPTQ: Accurate Post-Training Quantization.* 2023. · Lin et al. *AWQ.* 2024.
15. *Burn* deep-learning framework (Rust, WGPU/CubeCL backend); *WebLLM/MLC*; ONNX Runtime Web; Transformers.js.
16. Pruna AI. *Smash* compression toolkit — documentation and Wan tutorial.

*Bibliographic details identify standard, publicly available works for orientation; consult the originals for exact venues and dates.*
