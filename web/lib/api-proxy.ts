// Tiny proxy so the browser only talks to Next (no CORS on the FastAPI backend).
// API_URL is read at request time via bracket access so Next does not inline a
// build-time value; Docker Compose can set API_URL=http://api:8000 at runtime.

function apiUrl(): string {
  return process.env["API_URL"] ?? "http://localhost:8000";
}

async function forward(res: Response): Promise<Response> {
  const body = await res.text();
  return new Response(body, {
    status: res.status,
    headers: {
      "content-type": res.headers.get("content-type") ?? "application/json",
    },
  });
}

export async function proxyGet(path: string): Promise<Response> {
  const res = await fetch(`${apiUrl()}${path}`, { cache: "no-store" });
  return forward(res);
}

export async function proxyPost(path: string, body: unknown): Promise<Response> {
  const res = await fetch(`${apiUrl()}${path}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  return forward(res);
}