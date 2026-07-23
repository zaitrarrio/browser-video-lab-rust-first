import "./style.css";

type Mode = "sd" | "longlive" | "memflow" | "onnx" | "rust";
// Modes whose second control is a diffusion step count (rather than a chunk count).
const STEP_MODES: Mode[] = ["sd", "onnx", "rust"];
let mode: Mode = "sd";
const root = document.querySelector<HTMLDivElement>("#app")!;

// Resolve resources against Vite's base URL so paths stay correct when the demo
// is served from a subpath (e.g. GitHub Pages under /<repo>/). The ONNX runtimes
// point at the shipped `.example.json` manifest; the Rust student points at its
// compiled wasm bundle's student spec.
function manifestUrl(m: Mode): string {
  const base = import.meta.env.BASE_URL;
  return m === "rust"
    ? `${base}rust-video/student-spec.json`
    : `${base}models/${m}/manifest.example.json`;
}

const EXPERIMENTS: Record<Mode, {title: string; tag: string; blurb: string; contract: string}> = {
  sd: {
    title: "SD-Turbo",
    tag: "text → image",
    blurb: "Browser-side text encoding, denoising, and VAE decoding with ONNX Runtime Web on WebGPU.",
    contract: "text encoder: input_ids → last_hidden_state · UNet: sample, timestep, encoder_hidden_states → out_sample · VAE: latent_sample → sample",
  },
  longlive: {
    title: "LongLive",
    tag: "streaming · causal",
    blurb: "A causal streaming runtime with a persistent KV cache; present.* tensors are renamed to past.* between chunks.",
    contract: "generator: noise, prompt_embeds, chunk_index, past.* → sample | latents, present.*",
  },
  memflow: {
    title: "MemFlow",
    tag: "adaptive memory",
    blurb: "Prompt-conditioned retrieval over a bounded bank of historical video memories feeding the causal generator.",
    contract: "generator: noise, prompt_embeds, chunk_index, memory_values, memory_mask → latents, memory_key, memory_value",
  },
  onnx: {
    title: "ONNX student",
    tag: "distilled · ort-web",
    blurb: "The distilled latent-video student's denoiser.onnx run on ONNX Runtime Web over WebGPU — the path that carries real trained weights into the browser. First three latent channels are shown directly (no VAE exported yet).",
    contract: "denoiser: noisy_latents[1,C,1,H,W], timestep[1], prompt_embeds[1,S,text_width] → noise_pred[1,C,1,H,W]",
  },
  rust: {
    title: "Rust student",
    tag: "burn · wgpu · wasm",
    blurb: "The Burn latent-video student compiled to WebAssembly, running its denoiser directly on WebGPU — no ONNX Runtime.",
    contract: "student: latents[1,C,1,H,W], timestep[1,1], prompt_embeds[1,S,text_width] → sample[1,C,1,H,W]",
  },
};

const CARDS = [
  {t: "SD-Turbo WebGPU", d: "Few-step diffusion running entirely in the browser via ONNX Runtime Web."},
  {t: "LongLive WebGPU", d: "Streaming causal generation with a persistent, fixed-window KV cache."},
  {t: "MemFlow WebGPU", d: "Bounded adaptive memory bank with top-K cosine retrieval per chunk."},
  {t: "Wan 2.1 + Pruna Smash", d: "Native CUDA compression & benchmarking — server-side, kept out of the browser."},
  {t: "Rust-first student", d: "Burn causal latent-video student with Q8/Q4 bundles for native WGPU and WASM."},
  {t: "PyTorch-free path", d: "Framework-neutral Safetensors teacher cache enables Rust-only recurring training."},
];

function render() {
  const gpu = "gpu" in navigator;
  const ex = EXPERIMENTS[mode];
  root.innerHTML = `<main>
    <header>
      <h1>Browser Video Lab</h1>
      <p class="muted">WebGPU-first SD-Turbo, LongLive, and adaptive-memory MemFlow inference — with a Rust-first distillation pipeline.</p>
      <div class="badge ${gpu ? "ok" : "warn"}">WebGPU ${gpu ? "available" : "unavailable — use Chrome/Edge on a desktop GPU"}</div>
    </header>

    <section class="overview">
      ${CARDS.map((c) => `<article class="tile"><h3>${c.t}</h3><p>${c.d}</p></article>`).join("")}
    </section>

    <h2>Interactive runtime</h2>
    <nav>
      ${(Object.keys(EXPERIMENTS) as Mode[])
        .map((k) => `<button data-mode="${k}" class="${mode === k ? "active" : ""}">${EXPERIMENTS[k].title}</button>`)
        .join("")}
    </nav>

    <section class="card">
      <div class="mode-head"><strong>${ex.title}</strong><span class="pill">${ex.tag}</span></div>
      <p class="muted">${ex.blurb}</p>
      <label>Model manifest URL<input id="manifest" value="${manifestUrl(mode)}"></label>
      <label><div class="lbl">Prompt</div><textarea id="prompt">A cinematic tracking shot of a silver robot walking through Austin at sunset</textarea></label>
      <div class="grid">
        <label>Seed<input id="seed" type="number" value="42"></label>
        <label>${STEP_MODES.includes(mode) ? "Steps" : "Chunks"}<input id="steps" type="number" value="4" min="1" max="${STEP_MODES.includes(mode) ? 12 : 32}"></label>
      </div>
      <div class="actions">
        <button id="load">Load model</button>
        <button id="run" class="primary">Generate</button>
        <button id="stop">Stop</button>
      </div>
      <div class="status" id="status">Idle · point the manifest URL at an exported ${ex.title} graph.</div>
      <canvas id="output" width="512" height="512"></canvas>
      <details class="contract"><summary>Graph contract</summary><code>${ex.contract}</code></details>
    </section>

    <footer class="muted">Models are not bundled. Export ONNX graphs into <code>public/models</code> and copy each <code>manifest.example.json</code> to <code>manifest.json</code>. Large weights should be hosted with byte-range support.</footer>
  </main>`;

  document.querySelectorAll<HTMLButtonElement>("[data-mode]").forEach((b) => {
    b.onclick = () => {
      mode = b.dataset.mode as Mode;
      render();
    };
  });

  const worker = new Worker(new URL("./worker.ts", import.meta.url), {type: "module"});
  const status = document.querySelector<HTMLDivElement>("#status")!;
  const canvas = document.querySelector<HTMLCanvasElement>("#output")!;
  worker.onmessage = (e) => {
    const m = e.data;
    if (m.type === "status") status.textContent = m.message;
    if (m.type === "frame") {
      canvas.width = m.width;
      canvas.height = m.height;
      canvas.getContext("2d")!.putImageData(new ImageData(new Uint8ClampedArray(m.rgba), m.width, m.height), 0, 0);
    }
  };

  const payload = () => ({
    mode,
    manifestUrl: document.querySelector<HTMLInputElement>("#manifest")!.value,
    prompt: document.querySelector<HTMLTextAreaElement>("#prompt")!.value,
    seed: +document.querySelector<HTMLInputElement>("#seed")!.value,
    steps: +document.querySelector<HTMLInputElement>("#steps")!.value,
  });
  document.querySelector<HTMLButtonElement>("#load")!.onclick = () => worker.postMessage({type: "load", ...payload()});
  document.querySelector<HTMLButtonElement>("#run")!.onclick = () => worker.postMessage({type: "run", ...payload()});
  document.querySelector<HTMLButtonElement>("#stop")!.onclick = () => worker.postMessage({type: "stop"});
}

render();
