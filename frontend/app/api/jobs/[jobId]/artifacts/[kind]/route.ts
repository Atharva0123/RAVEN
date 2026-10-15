import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(request: NextRequest, { params }: { params: Promise<{ jobId: string; kind: string }> }) {
  const { jobId, kind } = await params;
  const apiUrl = process.env.RAVEN_API_URL ?? "http://localhost:8000";
  const apiKey = process.env.RAVEN_API_KEY;
  if (!apiKey) return NextResponse.json({ detail: "RAVEN_API_KEY is not configured on the frontend server" }, { status: 503 });
  try {
    const headers = new Headers({ "x-api-key": apiKey });
    const range = request.headers.get("range");
    if (range) headers.set("range", range);
    const response = await fetch(`${apiUrl}/api/v1/jobs/${encodeURIComponent(jobId)}/artifacts/${encodeURIComponent(kind)}`, {
      headers, cache: "no-store", signal: AbortSignal.timeout(30 * 60 * 1000),
    });
    const forwarded = new Headers();
    for (const name of ["content-type", "content-length", "content-range", "accept-ranges", "content-disposition", "cache-control", "etag", "last-modified"]) {
      const value = response.headers.get(name);
      if (value) forwarded.set(name, value);
    }
    return new NextResponse(response.body, { status: response.status, statusText: response.statusText, headers: forwarded });
  } catch {
    return NextResponse.json({ detail: "Video result is unavailable" }, { status: 502 });
  }
}
