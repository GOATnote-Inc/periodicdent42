import { NextResponse } from "next/server";

const API_BASE = process.env.RAG_API_BASE_URL ?? "http://localhost:8000";

export async function POST(request: Request) {
  const payload = await request.json();
  try {
    const response = await fetch(`${API_BASE}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const errorBody = await response.text();
      return NextResponse.json(
        {
          error: "Upstream chat service returned an error",
          detail: errorBody,
          status: response.status,
        },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    return NextResponse.json(
      {
        error: "Unable to reach chat backend",
        detail: error instanceof Error ? error.message : "Unknown network error",
        status: 502,
      },
      { status: 502 }
    );
  }
}
