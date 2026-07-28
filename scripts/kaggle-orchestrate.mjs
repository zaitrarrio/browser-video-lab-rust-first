#!/usr/bin/env node
// Drives one training chunk on Kaggle's free GPU, headlessly.
//
// Renders `kaggle/run_chunk.py` with a content-addressed source key, pushes it
// as a script kernel, polls to completion, then reads back the tiny state.json
// the kernel leaves behind. Exits 0 when the chunk succeeded; the caller decides
// from `completed` whether another chunk is due.
//
// The source key is what makes the pipeline cache-friendly: it hashes exactly
// the inputs that can change the compiled trainer, so the ~20 minute Burn/CUDA
// build happens on the first run after a Rust change and never again.
//
// Usage: node scripts/kaggle-orchestrate.mjs [--dry-run]
// Requires: kaggle CLI (2.x) on PATH, plus KAGGLE_API_TOKEN for auth and
// KAGGLE_USERNAME to namespace the kernel and dataset ids.

import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, writeFileSync, readdirSync, statSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, relative } from "node:path";

const ROOT = new URL("..", import.meta.url).pathname.replace(/\/$/, "");
const DRY_RUN = process.argv.includes("--dry-run");

// An unset GitHub secret arrives as the empty string, not as undefined. Treat
// that as missing: an empty owner silently renders a kernel id of "/slug", and
// the push then fails several steps later with an unrelated-looking auth error.
const env = (name, fallback) => {
  const value = process.env[name] || fallback;
  if (value === undefined || value === "") {
    throw new Error(`missing or empty required environment variable ${name}`);
  }
  return value;
};

const OWNER = env("KAGGLE_USERNAME");
const SLUG = env("KAGGLE_KERNEL_SLUG", "browser-video-student-chunk");

// Authenticates both the CLI calls below and the kernel itself, which gets it
// injected into its rendered source. Kaggle exposes no API for attaching a
// notebook secret — no SDK service, no CLI subcommand, no field on the kernel
// push — so a secret-based kernel needs a manual UI step per slug, forever.
// Injection is what keeps this pipeline headless; the cost is a cleartext token
// in the kernel's source and version history. See kaggle/README.md.
//
// A dry run renders without credentials, so it gets a placeholder — that also
// keeps a real token out of any file written by `--dry-run`.
const API_TOKEN = DRY_RUN ? "KGAT_placeholder_for_dry_run_only_00" : env("KAGGLE_API_TOKEN");
if (!API_TOKEN.startsWith("KGAT_")) {
  throw new Error(
    `KAGGLE_API_TOKEN should be a KGAT_… access token from kaggle.com → Settings → API; ` +
    `got ${API_TOKEN.length} chars starting "${API_TOKEN.slice(0, 5)}". The legacy key from ` +
    `kaggle.json is not a substitute — the 2.x CLI cannot authenticate with it.`
  );
}

// Kaggle slugifies a kernel's *title* and warns when the result doesn't match the
// id we asked for — then may well create the thing under the title's slug, leaving
// every later `status`/`output`/`download` pointed at something that isn't there.
// Deriving titles from the slug keeps the two in lockstep under any SLUG override.
const titleFor = (slug) => slug.replace(/-/g, " ").replace(/^./, (c) => c.toUpperCase());
const CONFIG = {
  repo_url: env("REPO_URL"),
  commit: env("GITHUB_SHA", execFileSync("git", ["rev-parse", "HEAD"], { cwd: ROOT }).toString().trim()),
  spec: env("TRAIN_SPEC", "rust/config/browser-390m-umt5.json"),
  backend: env("TRAIN_BACKEND", "cuda"),
  // Empty is a legitimate, distinct choice here (an ndarray/CPU build needs no
  // extra cargo feature), so this bypasses the strict env() helper — which
  // treats an empty override as a mistake — rather than reusing it.
  features: process.env.TRAIN_FEATURES !== undefined ? process.env.TRAIN_FEATURES : "cuda",
  chunk_steps: Number(env("CHUNK_STEPS", "20000")),
  target_steps: Number(env("TARGET_STEPS", "200000")),
  // 2e-5 is what browser-384m-umt5.yaml specifies for this student, and what the
  // trainer's non-finite guard points you back to. 1e-4 diverges: the quadratic
  // feature/Gram loss goes NaN a few hundred steps in, even under grad-clip 1.0.
  lr: Number(env("TRAIN_LR", "2e-5")),
  log_every: Number(env("LOG_EVERY", "200")),
  // A full resumable checkpoint this often. The cost is one save per interval;
  // the alternative is what an end-of-chunk-only save cost on the first real run,
  // which was every step of a 7.5-hour chunk.
  ckpt_every: Number(env("CKPT_EVERY", "1000")),
  // Kaggle hard-stops a GPU session at 12h; stop the trainer before that so the
  // checkpoint is written by us rather than lost to the reaper.
  session_seconds: Number(env("SESSION_SECONDS", String(11 * 3600))),
  upload_reserve_seconds: Number(env("UPLOAD_RESERVE_SECONDS", "900")),
  allow_synthetic_teacher: env("ALLOW_SYNTHETIC_TEACHER", "false") === "true",
  // A CPU-backend repro kernel wants no GPU at all — requesting one it doesn't
  // need risks failing on a CPU-only host image or burning GPU quota for
  // nothing. Defaults true so every existing caller is unaffected.
  enable_gpu: env("ENABLE_GPU", "true") === "true",
  api_token: API_TOKEN,
  // The kernel installs this exact CLI before authenticating, so only one
  // credential shape is valid there — see authenticate() in run_chunk.py. CI
  // sets it once at the workflow level and pins itself to the same value; the
  // default here is for local runs.
  kaggle_cli_version: env("KAGGLE_CLI_VERSION", "2.2.4"),
  toolchain_dataset: `${OWNER}/${SLUG}-toolchain`,
  toolchain_title: titleFor(`${SLUG}-toolchain`),
  // Overridable so a kernel running under a different SLUG (e.g. a CPU-backend
  // repro, kept separate so it gets its own toolchain/checkpoint datasets
  // rather than colliding with a CUDA-built one) can still mount the one real
  // teacher cache instead of looking for `<slug>-teacher-cache` and finding
  // nothing.
  teacher_dataset: env("TEACHER_DATASET", `${OWNER}/${SLUG}-teacher-cache`),
  checkpoint_dataset: `${OWNER}/${SLUG}-checkpoint`,
  checkpoint_title: titleFor(`${SLUG}-checkpoint`),
};

// ---------------------------------------------------------------- source key

/** Every file whose content can change the compiled trainer, in stable order. */
function trainerInputs() {
  const files = [];
  const walk = (dir) => {
    for (const entry of readdirSync(dir).sort()) {
      const path = join(dir, entry);
      if (statSync(path).isDirectory()) {
        if (entry !== "target") walk(path);
      } else if (/\.(rs|toml|json)$/.test(entry)) {
        files.push(path);
      }
    }
  };
  walk(join(ROOT, "rust"));
  return files;
}

function sourceKey() {
  const hash = createHash("sha256");
  for (const file of trainerInputs()) {
    hash.update(relative(ROOT, file));
    hash.update("\0");
    hash.update(readFileSync(file));
    hash.update("\0");
  }
  return hash.digest("hex");
}

// ------------------------------------------------------------------- kaggle

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
  const dir = mkdtempSync(join(tmpdir(), "kaggle-kernel-"));
  const template = readFileSync(join(ROOT, "kaggle/run_chunk.py"), "utf8");
  if (!template.includes("{{CONFIG}}")) throw new Error("kaggle/run_chunk.py lost its {{CONFIG}} marker");
  // JSON inside a Python r"""...""" literal: the only sequence that could close
  // it early is a quote run, which JSON.stringify escapes as \" — safe.
  writeFileSync(join(dir, "run_chunk.py"), template.replace("{{CONFIG}}", JSON.stringify(config, null, 2)));
  writeFileSync(join(dir, "kernel-metadata.json"), JSON.stringify({
    id: `${OWNER}/${SLUG}`,
    title: titleFor(SLUG),
    code_file: "run_chunk.py",
    language: "python",
    kernel_type: "script",
    is_private: true,
    enable_gpu: config.enable_gpu,
    enable_internet: true,
    // Mounting a dataset that has no versions yet fails the push, so only ask
    // for caches that actually exist. A cold pipeline simply starts from zero.
    dataset_sources: [config.toolchain_dataset, config.teacher_dataset, config.checkpoint_dataset]
      .filter(datasetExists),
    competition_sources: [],
    kernel_sources: [],
  }, null, 2) + "\n");
  return dir;
}

function datasetExists(slug) {
  const { code, out } = kaggle(["datasets", "status", slug], { check: false });
  return code === 0 && !/not found|404/i.test(out);
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

// --------------------------------------------------------------------- main

const config = { ...CONFIG, source_key: sourceKey() };
console.log(`source key ${config.source_key.slice(0, 12)} · target ${config.target_steps} steps · commit ${config.commit.slice(0, 8)}`);

const dir = renderKernel(config);
console.log(`rendered kernel in ${dir}`);
if (DRY_RUN) {
  console.log(readFileSync(join(dir, "kernel-metadata.json"), "utf8"));
  process.exit(0);
}

const ref = `${OWNER}/${SLUG}`;
kaggle(["kernels", "push", "-p", dir], { capture: false });
const status = waitForKernel(ref, { timeoutSeconds: config.session_seconds + 1800 });

const outDir = mkdtempSync(join(tmpdir(), "kaggle-output-"));
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
  writeFileSync(process.env.GITHUB_OUTPUT, [
    `completed=${state.completed}`,
    `steps_done=${state.steps_done}`,
    `target_steps=${state.target_steps}`,
    `last_loss=${state.last_loss}`,
    `checkpoint_dataset=${config.checkpoint_dataset}`,
  ].join("\n") + "\n", { flag: "a" });
}
if (process.env.GITHUB_STEP_SUMMARY) {
  writeFileSync(process.env.GITHUB_STEP_SUMMARY,
    `### Training chunk\n\n` +
    `- progress **${state.steps_done} / ${state.target_steps}** steps (${state.chunks} chunks)\n` +
    `- last loss \`${state.last_loss}\` · best \`${state.best_loss}\`\n` +
    `- stopped by \`${state.stopped_by}\` · ${state.completed ? "**run complete**" : "more chunks due"}\n`,
    { flag: "a" });
}
if (status !== "complete") throw new Error(`kernel ended with status ${status}`);
