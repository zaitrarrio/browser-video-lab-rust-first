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
  features: env("TRAIN_FEATURES", "cuda"),
  chunk_steps: Number(env("CHUNK_STEPS", "20000")),
  target_steps: Number(env("TARGET_STEPS", "200000")),
  lr: Number(env("TRAIN_LR", "1e-4")),
  log_every: Number(env("LOG_EVERY", "200")),
  // Kaggle hard-stops a GPU session at 12h; stop the trainer before that so the
  // checkpoint is written by us rather than lost to the reaper.
  session_seconds: Number(env("SESSION_SECONDS", String(11 * 3600))),
  upload_reserve_seconds: Number(env("UPLOAD_RESERVE_SECONDS", "900")),
  allow_synthetic_teacher: env("ALLOW_SYNTHETIC_TEACHER", "false") === "true",
  toolchain_dataset: `${OWNER}/${SLUG}-toolchain`,
  toolchain_title: titleFor(`${SLUG}-toolchain`),
  teacher_dataset: `${OWNER}/${SLUG}-teacher-cache`,
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
    enable_gpu: true,
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

// KAGGLE_API_TOKEN is consumed by the CLI, not by us, so a missing one would
// surface only as the CLI's generic "Authentication required" text — and
// renderKernel's dataset probes below would emit it three times before we ever
// push. Name it up front. A dry run never authenticates, so it needs none.
//
// It must be the `KGAT_…` token from the settings UI, not the legacy `key` from
// kaggle.json: the 2.x CLI authenticates *only* via KAGGLE_API_TOKEN, and feeding
// it a token through KAGGLE_USERNAME/KAGGLE_KEY fails with that same generic text.
if (!DRY_RUN) {
  const token = env("KAGGLE_API_TOKEN");
  if (!token.startsWith("KGAT_")) {
    throw new Error(`KAGGLE_API_TOKEN should be a KGAT_… access token; got ${token.length} chars starting "${token.slice(0, 5)}"`);
  }
}

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
