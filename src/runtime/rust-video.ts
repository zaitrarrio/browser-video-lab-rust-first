// Runtime that drives the Burn latent-video student compiled to WebAssembly.
// Unlike the SD-Turbo / LongLive / MemFlow runtimes it does not use ONNX Runtime
// at all — it dynamically imports the wasm-pack bundle produced by
// `task rust:wasm` (or CI) into `public/rust-video/` and calls its WebGPU kernels
// directly.

// Compact, browser-friendly student. The published 390M spec
// (rust/config/browser-390m.json) is far too heavy to random-init in a tab, so
// the demo defaults to this and only overrides it if a spec URL is reachable.
const DEMO_SPEC = {
  latent_channels: 4,
  text_width: 64,
  width: 192,
  layers: 4,
  heads: 6,
  mlp_ratio: 2,
  max_tokens: 8192,
};

// Latent side length; the decoded frame is SIDE×SIDE. Keep SIDE*SIDE within the
// spec's max_tokens (attention cost is ~O((SIDE^2)^2)).
const SIDE = 48;

type WasmModule = {
  default: (init?: unknown) => Promise<unknown>;
  BrowserModel: new (specJson: string) => {
    prepare(): Promise<void>;
    prepare_with_weights(weights: Uint8Array): Promise<void>;
    trained(): boolean;
    generate(seed: number, steps: number, side: number): Promise<Uint8Array>;
    backend(): string;
    parameters(): number;
  };
};

export class RustVideoRuntime {
  private model!: InstanceType<WasmModule["BrowserModel"]>;

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

    this.model = new mod.BrowserModel(JSON.stringify(spec));
    progress("Acquiring WebGPU adapter…");
    // Load trained weights (a video-train BinFileRecorder record) when the
    // bundle ships them; otherwise fall back to random init so the demo still
    // runs. The status line always says which one the user is looking at.
    let loaded = false;
    try {
      const wr = await fetch(`${base}rust-video/student.bin`);
      if (wr.ok && !(wr.headers.get("content-type") ?? "").includes("text/html")) {
        await this.model.prepare_with_weights(new Uint8Array(await wr.arrayBuffer()));
        loaded = true;
      }
    } catch {
      /* fall through to random init */
    }
    if (!loaded) await this.model.prepare();
    const params = Math.round(this.model.parameters() / 1e6);
    progress(`Rust student ready · ${this.model.backend()} · ~${params}M params · ${loaded ? "trained weights" : "random init"}`);
  }

  async run(
    _prompt: string,
    steps: number,
    seed: number,
    onFrame: (x: {rgba: Uint8ClampedArray; width: number; height: number}) => void,
    signal: AbortSignal,
  ) {
    if (!this.model) throw new Error("Load the model first");
    if (signal.aborted) throw new DOMException("Stopped", "AbortError");
    const bytes = await this.model.generate(seed >>> 0, Math.max(1, steps), SIDE);
    if (signal.aborted) throw new DOMException("Stopped", "AbortError");
    onFrame({rgba: new Uint8ClampedArray(bytes), width: SIDE, height: SIDE});
  }
}
