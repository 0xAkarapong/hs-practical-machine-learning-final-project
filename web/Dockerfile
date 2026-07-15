# Standard Next.js standalone build. output: "standalone" in next.config.ts
# produces a self-contained server copied into a slim runtime image.
# .dockerignore excludes host node_modules/.next/.env* so deps stay Linux-native
# and API_URL is not baked from a local .env.local.
FROM node:22-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

FROM node:22-alpine AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
# Do not load local env during image build; runtime gets API_URL from compose.
ENV API_URL=
RUN npm run build

FROM node:22-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production
# API_URL is set per-environment by docker-compose (http://api:8000). Read at
# request time in lib/api-proxy.ts, so the standalone server uses the runtime env.
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
COPY --from=builder /app/public ./public
EXPOSE 3000
ENV PORT=3000 HOSTNAME=0.0.0.0
CMD ["node", "server.js"]