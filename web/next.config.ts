import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone", // ponytail: minimal Next image for docker-compose (single .next/standalone server)
  // ponytail: a stray ~/package-lock.json outside the repo confuses Turbopack's
  // workspace-root inference; pin the root to this app dir.
  turbopack: { root: process.cwd() },
};

export default nextConfig;