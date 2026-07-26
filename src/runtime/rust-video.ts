// Runtime that drives the Burn latent-video student compiled to WebAssembly.
// Unlike the SD-Turbo / LongLive / MemFlow runtimes it does not use ONNX Runtime
// for the denoiser — it dynamically imports the wasm-pack bundle produced by
// `task rust:wasm` (or CI) into `public/rust-video/` and calls its WebGPU kernels
// directly. Prompt conditioning reuses the shared umt5-small encoder path: an
// optional `rust-video/text-encoder.json` manifest supplies the encoder, and the
// resulting embedding is handed to the WASM `generate`. Without one the prompt
// still seeds a deterministic embedding, so typing a different prompt changes the
// output even before a real encoder is shipped.
import {encodePrompt, hashPrompt, loadTextEncoder, manifest, type TextEncoder} from "./common";

// Compact, browser-friendly student. The published 390M spec
// (rust/config/browser-390m.json) is far too heavy to random-init in a tab, so
// the demo defaults to this and only overrides it if a spec URL is reachable.
const DEMO_SPEC = {
  latent_channels: 16,
  text_width: 64,
  width: 192,
  layers: 4,
  heads: 6,
  mlp_ratio: 2,
  max_tokens: 8192,
  patch_size: [1, 2, 2],
};

// Latent side length; the decoded frame is SIDE×SIDE. Keep SIDE*SIDE within the
// spec's max_tokens (attention cost is ~O((SIDE^2)^2)).
const SIDE = 48;

type WasmModule = {
  default: (init?: unknown) => Promise<unknown>;
  BrowserModel: new (specJson: string) => {
    prepare(): Promise<void>;
    prepare_with_weights(weights: Uint8Array): Promise<void>;
    prepare_with_quantized(indexJson: string, weights: Uint8Array): Promise<void>;
    trained(): boolean;
    generate(seed: number, steps: number, side: number, promptEmbeds: Float32Array): Promise<Uint8Array>;
    backend(): string;
    parameters(): number;
  };
};

// Prompt tokens fed to the student; must match the seq the umt5 encoder pads to.
const SEQ = 8;

export class RustVideoRuntime {
  private model!: InstanceType<WasmModule["BrowserModel"]>;
  private enc?: TextEncoder;
  private textWidth = DEMO_SPEC.text_width;

  async load(url: string, progress: (s: string) => void) {
    if (!("gpu" in navigator)) throw new Error("WebGPU unavailable — the Rust student needs navigator.gpu");
    const base = import.meta.env.BASE_URL;
    progress("Loading Rust/WASM bundle…");
    const mod = (await import(/* @vite-ignore */ `${base}rust-video/video_web.js`)) as WasmModule;
    await mod.default();

    let spec = DEMO_SPEC;
    try {
      const r = await fetch(url);
      if (r.ok) spec = await r.json();
    } catch {
      /* no spec shipped — fall back to the compact demo spec */
    }
    this.textWidth = spec.text_width ?? DEMO_SPEC.text_width;

    // Optional umt5-small encoder, shipped beside the bundle. When present (and
    // its hidden width matches the student's text_width) prompts are semantically
    // encoded; otherwise a prompt-seeded fallback is used at generate time.
    try {
      const tm = await manifest(`${base}rust-video/text-encoder.json`);
      this.enc = await loadTextEncoder(tm);
    } catch {
      /* no encoder shipped — prompts fall back to a deterministic seed */
    }

    this.model = new mod.BrowserModel(JSON.stringify(spec));
    progress("Acquiring WebGPU adapter…");
    // Prefer a quantized bundle (index.json + weights.q{bits}), then the
    // full-precision student.bin, then random init so the demo still runs. The
    // status line always says which one the user is looking at.
    const isHtml = (r: Response) => (r.headers.get("content-type") ?? "").includes("text/html");
    let source = "random init";
    try {
      const ir = await fetch(`${base}rust-video/index.json`);
      if (ir.ok && !isHtml(ir)) {
        const indexText = await ir.text();
        const bits = (JSON.parse(indexText) as Array<{bits: number}>)[0]?.bits ?? 8;
        const wr = await fetch(`${base}rust-video/weights.q${bits}`);
        if (wr.ok && !isHtml(wr)) {
          await this.model.prepare_with_quantized(indexText, new Uint8Array(await wr.arrayBuffer()));
          source = `trained weights (int${bits})`;
        }
      }
    } catch {
      /* fall through to student.bin / random init */
    }
    if (source === "random init") {
      try {
        const wr = await fetch(`${base}rust-video/student.bin`);
        if (wr.ok && !isHtml(wr)) {
          await this.model.prepare_with_weights(new Uint8Array(await wr.arrayBuffer()));
          source = "trained weights (f32)";
        }
      } catch {
        /* fall through to random init */
      }
    }
    if (source === "random init") await this.model.prepare();
    const params = Math.round(this.model.parameters() / 1e6);
    const prompting = this.enc ? "prompt-conditioned" : "seeded prompt";
    progress(`Rust student ready · ${this.model.backend()} · ~${params}M params · ${source} · ${prompting}`);
  }

  async run(
    prompt: string,
    steps: number,
    seed: number,
    onFrame: (x: {rgba: Uint8ClampedArray; width: number; height: number}) => void,
    signal: AbortSignal,
  ) {
    if (!this.model) throw new Error("Load the model first");
    if (signal.aborted) throw new DOMException("Stopped", "AbortError");
    // Encode the prompt (real umt5 when available, else a prompt-seeded fallback)
    // and hand the [SEQ, text_width] embedding to the WASM student — the prompt is
    // no longer discarded.
    const {data} = await encodePrompt(this.enc, prompt, SEQ, this.textWidth, (seed ^ hashPrompt(prompt)) >>> 0);
    const bytes = await this.model.generate(seed >>> 0, Math.max(1, steps), SIDE, data);
    if (signal.aborted) throw new DOMException("Stopped", "AbortError");
    onFrame({rgba: new Uint8ClampedArray(bytes), width: SIDE, height: SIDE});
  }
}
