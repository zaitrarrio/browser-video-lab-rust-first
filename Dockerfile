# syntax=docker/dockerfile:1

# ---- Build stage: typecheck, test, and produce the static bundle ----
FROM node:20-bookworm-slim AS build
WORKDIR /app

# Install dependencies against the lockfile first for better layer caching.
COPY package.json package-lock.json ./
RUN npm ci --ignore-scripts

# Copy the sources needed to build the demonstration page.
COPY tsconfig.json vite.config.ts index.html ./
COPY src ./src
COPY public ./public
COPY scripts ./scripts

# Validate example manifests, then build. `npm run build` runs `tsc -b && vite build`.
RUN node scripts/check-models.mjs public/models/*/manifest.example.json \
 && npm run build

# ---- Runtime stage: serve dist/ with COOP/COEP + range support ----
FROM node:20-bookworm-slim AS runtime
WORKDIR /app
ENV NODE_ENV=production \
    PORT=8080 \
    STATIC_ROOT=/app/dist

COPY --from=build /app/dist ./dist
COPY server ./server

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD node -e "fetch('http://127.0.0.1:'+(process.env.PORT||8080)+'/healthz').then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))"

CMD ["node", "server/index.mjs"]
