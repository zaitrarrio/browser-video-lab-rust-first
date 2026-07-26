#!/usr/bin/env node
// Builds the teacher-cache dataset on Kaggle's free CPU tier, headlessly.
//
// The CPU sibling of scripts/kaggle-orchestrate.mjs: it renders kaggle/cache_chunk.py
// with a CONFIG literal, pushes it as a CPU script kernel, polls to completion, and
// reads back the small state.json the kernel leaves behind. The kernel produces the
// `<user>/<slug>-teacher-cache` dataset that the training pipeline mounts — the one
// input that used to be built by hand (see kaggle/README.md).
//
// Usage: node scripts/kaggle-cache.mjs [--dry-run]
// Requires: kaggle CLI (2.x) on PATH, KAGGLE_API_TOKEN (KGAT_ token), KAGGLE_USERNAME.

import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, writeFileSync, readdirSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const ROOT = new URL("..", import.meta.url).pathname.replace(/\/$/, "");
const DRY_RUN = process.argv.includes("--dry-run");

// An unset GitHub secret arrives as "", not undefined — treat it as missing so a
// blank owner can't silently render a kernel id of "/slug".
const env = (name, fallback) => {
  const value = process.env[name] || fallback;
  if (value === undefined || value === "") {
    throw new Error(`missing or empty required environment variable ${name}`);
  }
  return value;
};

const OWNER = env("KAGGLE_USERNAME");
// Same base slug as the trainer, so `<slug>-teacher-cache` is exactly the dataset
// scripts/kaggle-orchestrate.mjs mounts. Override both together if you override it.
const SLUG = env("KAGGLE_KERNEL_SLUG", "browser-video-student-chunk");
const CACHE_SLUG = env("KAGGLE_CACHE_KERNEL_SLUG", `${SLUG}-teacher-cache-build`);

// A dry run renders without credentials (and keeps a real token out of any file it
// writes); a real run injects the KGAT_ token into the kernel source, as the
// trainer does — Kaggle has no API for attaching a notebook secret. See README.
const API_TOKEN = DRY_RUN ? "KGAT_placeholder_for_dry_run_only_00" : env("KAGGLE_API_TOKEN");
if (!API_TOKEN.startsWith("KGAT_")) {
  throw new Error(
    `KAGGLE_API_TOKEN should be a KGAT_… access token from kaggle.com → Settings → API; ` +
    `got ${API_TOKEN.length} chars starting "${API_TOKEN.slice(0, 5)}".`
  );
}

const titleFor = (slug) => slug.replace(/-/g, " ").replace(/^./, (c) => c.toUpperCase());
const latentShape = env("LATENT_SHAPE", "16,4,32,48").split(",").map((n) => Number(n.trim()));
if (latentShape.length !== 4 || latentShape.some(Number.isNaN)) {
  throw new Error(`LATENT_SHAPE must be four numbers "C,T,H,W"; got "${env("LATENT_SHAPE", "16,4,32,48")}"`);
}

const CONFIG = {
  repo_url: env("REPO_URL"),
  commit: env("GITHUB_SHA", execFileSync("git", ["rev-parse", "HEAD"], { cwd: ROOT }).toString().trim()),
  captions_file: env("CAPTIONS_FILE", "data/prompts.example.txt"),
  latent_shape: latentShape,
  seq: Number(env("SEQ", "128")),
  teacher_width: Number(env("TEACHER_WIDTH", "4096")),
  student_width: Number(env("STUDENT_WIDTH", "512")),
  teacher_encoder: env("TEACHER_ENCODER", "google/umt5-xxl"),
  student_encoder: env("STUDENT_ENCODER", "google/umt5-small"),
  model_id: env("TEACHER_MODEL_ID", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"),
  teacher_name: env("TEACHER_NAME", "Wan2.1-T2V-1.3B"),
  draws_per_clip: Number(env("DRAWS_PER_CLIP", "8")),
  relation_layers: Number(env("RELATION_LAYERS", "6")),
  limit: Number(env("CAPTION_LIMIT", "0")),
  // Poll timeout only — the kernel isn't wall-clock bounded like the trainer, but a
  // CPU teacher pass over a 1.3B DiT is long, so give the poller room.
  session_seconds: Number(env("SESSION_SECONDS", String(11 * 3600))),
  api_token: API_TOKEN,
  kaggle_cli_version: env("KAGGLE_CLI_VERSION", "2.2.4"),
  teacher_dataset: `${OWNER}/${SLUG}-teacher-cache`,
  teacher_title: titleFor(`${SLUG}-teacher-cache`),
};

function kaggle(args, { capture = true, check = true } = {}) {
  try {
    const out = execFileSync("kaggle", args, { encoding: "utf8", stdio: capture ? "pipe" : "inherit" });
    return { code: 0, out: out ?? "" };
  } catch (error) {
    if (check) {
      const detail = [error.stdout, error.stderr].filter(Boolean).join("\n").trim();
      throw new Error(`kaggle ${args.join(" ")} failed:\n${detail || error.message}`);
    }
    return { code: error.status ?? 1, out: [error.stdout, error.stderr].filter(Boolean).join("\n") };
  }
}

function renderKernel(config) {
  const dir = mkdtempSync(join(tmpdir(), "kaggle-cache-kernel-"));
  const template = readFileSync(join(ROOT, "kaggle/cache_chunk.py"), "utf8");
  if (!template.includes("{{CONFIG}}")) throw new Error("kaggle/cache_chunk.py lost its {{CONFIG}} marker");
  writeFileSync(join(dir, "cache_chunk.py"), template.replace("{{CONFIG}}", JSON.stringify(config, null, 2)));
  writeFileSync(join(dir, "kernel-metadata.json"), JSON.stringify({
    id: `${OWNER}/${CACHE_SLUG}`,
    title: titleFor(CACHE_SLUG),
    code_file: "cache_chunk.py",
    language: "python",
    kernel_type: "script",
    is_private: true,
    // CPU tier: the teacher pass is inference-only and the GPU quota is for training.
    enable_gpu: false,
    enable_internet: true,
    dataset_sources: [],
    competition_sources: [],
    kernel_sources: [],
  }, null, 2) + "\n");
  return dir;
}

const TERMINAL = { complete: "complete", error: "error", cancelAcknowledged: "cancelAcknowledged" };

function waitForKernel(ref, { pollSeconds = 60, timeoutSeconds }) {
  const deadline = Date.now() + timeoutSeconds * 1000;
  for (;;) {
    const { out } = kaggle(["kernels", "status", ref], { check: false });
    const status = /"?status"?\s*[:=]\s*"?(\w+)/i.exec(out)?.[1] ?? out.trim();
    process.stdout.write(`[${new Date().toISOString()}] ${status}\n`);
    if (status in TERMINAL || /complete|error|cancel/i.test(status)) return status;
    if (Date.now() > deadline) throw new Error(`kernel ${ref} still ${status} after ${timeoutSeconds}s`);
    Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, pollSeconds * 1000);
  }
}

console.log(`teacher cache → ${CONFIG.teacher_dataset} · captions ${CONFIG.captions_file} · commit ${CONFIG.commit.slice(0, 8)}`);
const dir = renderKernel(CONFIG);
console.log(`rendered kernel in ${dir}`);
if (DRY_RUN) {
  console.log(readFileSync(join(dir, "kernel-metadata.json"), "utf8"));
  process.exit(0);
}

const ref = `${OWNER}/${CACHE_SLUG}`;
kaggle(["kernels", "push", "-p", dir], { capture: false });
const status = waitForKernel(ref, { timeoutSeconds: CONFIG.session_seconds + 1800 });

const outDir = mkdtempSync(join(tmpdir(), "kaggle-cache-output-"));
kaggle(["kernels", "output", ref, "-p", outDir], { check: false });
kaggle(["kernels", "output", ref, "-p", outDir, "-w"], { check: false });

const statePath = join(outDir, "state.json");
if (!existsSync(statePath)) {
  const log = readdirSync(outDir).find((f) => f.endsWith(".log"));
  if (log) console.error(readFileSync(join(outDir, log), "utf8").slice(-8000));
  throw new Error(`kernel finished ${status} without writing state.json — see log above`);
}

const state = JSON.parse(readFileSync(statePath, "utf8"));
console.log(JSON.stringify(state, null, 2));
if (process.env.GITHUB_OUTPUT) {
  writeFileSync(process.env.GITHUB_OUTPUT,
    [`shards=${state.shards}`, `relation_layers=${state.relation_layers}`, `teacher_dataset=${state.teacher_dataset}`].join("\n") + "\n",
    { flag: "a" });
}
if (process.env.GITHUB_STEP_SUMMARY) {
  writeFileSync(process.env.GITHUB_STEP_SUMMARY,
    `### Teacher cache built\n\n` +
    `- **${state.shards} shards**, ${state.relation_layers} relation layers, teacher \`${state.teacher}\`\n` +
    `- versioned to dataset \`${state.teacher_dataset}\` — the trainer mounts it on the next run\n`,
    { flag: "a" });
}
if (status !== "complete") throw new Error(`kernel ended with status ${status}`);
