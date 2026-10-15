import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(request: NextRequest) {
  const apiUrl = process.env.RAVEN_API_URL ?? "http://localhost:8000";
  const apiKey = process.env.RAVEN_API_KEY;
  if (!apiKey) return NextResponse.json({ detail: "RAVEN_API_KEY is not configured on the frontend server" }, { status: 503 });
  const incomingType = request.headers.get("content-type");
  if (!incomingType?.startsWith("multipart/form-data;")) {
    return NextResponse.json({ detail: "Expected a multipart video upload" }, { status: 415 });
  }
  try {
    const response = await fetch(`${apiUrl}/api/v1/videos`, {
      method: "POST",
      headers: {
        "content-type": incomingType,
        "x-api-key": apiKey,
        ...(request.headers.get("idempotency-key") ? { "idempotency-key": request.headers.get("idempotency-key")! } : {}),
      },
      body: request.body,
      // Node's fetch requires duplex for a streaming request body.
      duplex: "half",
      signal: AbortSignal.timeout(30 * 60 * 1000),
    } as RequestInit & { duplex: "half" });
    const result = await response.json();
    return NextResponse.json(result, { status: response.status });
  } catch {
    return NextResponse.json({ detail: "Video upload service is unavailable" }, { status: 502 });
  }
}
