import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // standalone → single .next/standalone server for docker-compose.
  output: "standalone",
  // A stray ~/package-lock.json outside the repo confuses Turbopack's
  // workspace-root inference; pin the root to this app dir.
  turbopack: { root: process.cwd() },
};

export default nextConfig;