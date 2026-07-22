// Zero-dependency static server for the built demonstration page.
//
// WebGPU + threaded WASM (ONNX Runtime Web) require a cross-origin isolated
// context, which needs the COOP/COEP headers below. The Vite dev server sets
// these via vite.config.ts, but a plain static host does not, so we replicate
// them here for the production build. HTTP Range support is included because
// large ONNX weights are expected to be served with byte-range requests.
import {createReadStream, promises as fs} from "node:fs";
import {createServer} from "node:http";
import {extname, join, normalize, resolve, sep} from "node:path";

const ROOT = resolve(process.env.STATIC_ROOT ?? "dist");
const PORT = Number(process.env.PORT ?? 8080);
const HOST = process.env.HOST ?? "0.0.0.0";

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".wasm": "application/wasm",
  ".onnx": "application/octet-stream",
  ".safetensors": "application/octet-stream",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".webp": "image/webp",
  ".ico": "image/x-icon",
  ".map": "application/json; charset=utf-8",
  ".txt": "text/plain; charset=utf-8",
  ".mp4": "video/mp4",
};

// Long-cache immutable, content-hashed Vite assets; always revalidate HTML.
function cacheControl(pathname) {
  if (pathname === "/" || pathname.endsWith(".html")) return "no-cache";
  if (pathname.startsWith("/assets/")) return "public, max-age=31536000, immutable";
  return "public, max-age=3600";
}

function securityHeaders(res) {
  // Cross-origin isolation for WebGPU + SharedArrayBuffer / threaded WASM.
  res.setHeader("Cross-Origin-Opener-Policy", "same-origin");
  res.setHeader("Cross-Origin-Embedder-Policy", "require-corp");
  res.setHeader("Cross-Origin-Resource-Policy", "same-origin");
  res.setHeader("X-Content-Type-Options", "nosniff");
  res.setHeader("Referrer-Policy", "strict-origin-when-cross-origin");
}

// Resolve a URL path to a file inside ROOT, refusing traversal outside it.
async function resolveFile(pathname) {
  const decoded = decodeURIComponent(pathname.split("?")[0]);
  const rel = normalize(decoded).replace(/^(\.\.[/\\])+/, "");
  let target = join(ROOT, rel);
  if (target !== ROOT && !target.startsWith(ROOT + sep)) return null;
  try {
    const stat = await fs.stat(target);
    if (stat.isDirectory()) {
      target = join(target, "index.html");
      return {path: target, stat: await fs.stat(target)};
    }
    return {path: target, stat};
  } catch {
    return null;
  }
}

const server = createServer(async (req, res) => {
  try {
    if (req.method !== "GET" && req.method !== "HEAD") {
      res.writeHead(405, {Allow: "GET, HEAD"}).end("Method Not Allowed");
      return;
    }

    const url = req.url ?? "/";
    if (url === "/healthz") {
      res.writeHead(200, {"Content-Type": "application/json"}).end('{"status":"ok"}');
      return;
    }

    let file = await resolveFile(url);
    // SPA fallback: unknown non-asset routes serve index.html.
    if (!file && !extname(url)) file = await resolveFile("/index.html");
    if (!file) {
      res.writeHead(404, {"Content-Type": "text/plain"}).end("Not Found");
      return;
    }

    const type = MIME[extname(file.path).toLowerCase()] ?? "application/octet-stream";
    securityHeaders(res);
    res.setHeader("Content-Type", type);
    res.setHeader("Cache-Control", cacheControl(url));
    res.setHeader("Accept-Ranges", "bytes");
    res.setHeader("Last-Modified", file.stat.mtime.toUTCString());

    const total = file.stat.size;
    const range = req.headers.range;
    if (range) {
      const match = /^bytes=(\d*)-(\d*)$/.exec(range);
      if (match) {
        let start = match[1] ? Number(match[1]) : 0;
        let end = match[2] ? Number(match[2]) : total - 1;
        if (Number.isNaN(start) || Number.isNaN(end) || start > end || end >= total) {
          res.writeHead(416, {"Content-Range": `bytes */${total}`}).end();
          return;
        }
        res.writeHead(206, {
          "Content-Range": `bytes ${start}-${end}/${total}`,
          "Content-Length": end - start + 1,
        });
        if (req.method === "HEAD") return res.end();
        createReadStream(file.path, {start, end}).pipe(res);
        return;
      }
    }

    res.setHeader("Content-Length", total);
    res.writeHead(200);
    if (req.method === "HEAD") return res.end();
    createReadStream(file.path).pipe(res);
  } catch (err) {
    res.writeHead(500, {"Content-Type": "text/plain"}).end("Internal Server Error");
    console.error(err);
  }
});

server.listen(PORT, HOST, () => {
  console.log(`serving ${ROOT} on http://${HOST}:${PORT}`);
});
