// Build a content-addressed manifest of a model bundle directory.
//
//   node scripts/model-manifest.mjs <bundle-dir> [--release <version>] [--out <file>]
//
// Emits JSON describing every file (relative path, byte length, sha256) so the
// browser can verify and cache weights by content hash (service worker / OPFS).
// With no --out the manifest is written to <bundle-dir>/models-manifest.json.
import {createHash} from 'node:crypto';
import {readFile, readdir, stat, writeFile} from 'node:fs/promises';
import {join, relative, sep} from 'node:path';

const args = process.argv.slice(2);
const positional = [];
const opts = {};
for (let i = 0; i < args.length; i++) {
  const a = args[i];
  if (a === '--release') opts.release = args[++i];
  else if (a === '--out') opts.out = args[++i];
  else positional.push(a);
}
const dir = positional[0];
if (!dir) {
  console.error('usage: node scripts/model-manifest.mjs <bundle-dir> [--release <version>] [--out <file>]');
  process.exit(2);
}

async function walk(root, current = root, acc = []) {
  for (const entry of await readdir(current, {withFileTypes: true})) {
    const full = join(current, entry.name);
    if (entry.isDirectory()) await walk(root, full, acc);
    else if (entry.isFile()) acc.push(full);
  }
  return acc;
}

const paths = (await walk(dir)).sort();
const files = [];
for (const p of paths) {
  const name = relative(dir, p).split(sep).join('/');
  if (name === 'models-manifest.json') continue;
  const bytes = await readFile(p);
  files.push({
    name,
    bytes: (await stat(p)).size,
    sha256: createHash('sha256').update(bytes).digest('hex'),
  });
}

const manifest = {
  release: opts.release ?? process.env.RELEASE_VERSION ?? 'dev',
  generatedFrom: dir.split(sep).join('/'),
  files,
};
const out = opts.out ?? join(dir, 'models-manifest.json');
await writeFile(out, JSON.stringify(manifest, null, 2) + '\n');
console.log(`${out}: ${files.length} file(s), ${files.reduce((n, f) => n + f.bytes, 0)} bytes`);
for (const f of files) console.log(`  ${f.sha256.slice(0, 12)}  ${f.bytes}\t${f.name}`);
