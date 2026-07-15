import { proxyPost } from "@/lib/api-proxy";

export async function POST(request: Request) {
  const body = await request.json();
  return proxyPost("/predict", body);
}