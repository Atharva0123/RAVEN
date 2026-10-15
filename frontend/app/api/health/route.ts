import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";

export async function GET() {
  const apiUrl = process.env.RAVEN_API_URL ?? "http://localhost:8000";
  const browserApiUrl = process.env.RAVEN_BROWSER_API_URL ?? process.env.NEXT_PUBLIC_API_URL ?? apiUrl;
  const ws_url = browserApiUrl.replace(/^http/, "ws") + "/api/v1/ws/status";
  try {
    const response = await fetch(`${apiUrl}/ready`, { cache: "no-store", signal: AbortSignal.timeout(5000) });
    return NextResponse.json({ ...(await response.json()), ws_url }, { status: response.status });
  } catch {
    return NextResponse.json({ status: "unavailable", service: "raven", ws_url }, { status: 503 });
  }
}
