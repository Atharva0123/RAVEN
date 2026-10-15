import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(_request: NextRequest, { params }: { params: Promise<{ jobId: string }> }) {
  const { jobId } = await params;
  const apiUrl = process.env.RAVEN_API_URL ?? "http://localhost:8000";
  const apiKey = process.env.RAVEN_API_KEY;
  if (!apiKey) return NextResponse.json({ detail: "RAVEN_API_KEY is not configured on the frontend server" }, { status: 503 });
  try {
    const response = await fetch(`${apiUrl}/api/v1/jobs/${encodeURIComponent(jobId)}`, {
      headers: { "x-api-key": apiKey }, cache: "no-store", signal: AbortSignal.timeout(15_000),
    });
    return NextResponse.json(await response.json(), { status: response.status });
  } catch {
    return NextResponse.json({ detail: "Video job service is unavailable" }, { status: 502 });
  }
}
