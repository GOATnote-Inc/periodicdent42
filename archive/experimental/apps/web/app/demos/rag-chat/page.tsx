"use client";

import { useCallback, useState } from "react";
import { ErrorBanner } from "@/components/ErrorBanner";
import { HowItWorks } from "@/components/HowItWorks";
import { ResultPane } from "@/components/ResultPane";
import { RunPanel } from "@/components/RunPanel";

interface Citation {
  doc_id: string;
  section: string;
  text: string;
}

interface GuardrailOutcome {
  rule: string;
  status: string;
  details?: string;
}

interface VectorStats {
  avg_similarity: number;
  retrieved: number;
  ann_probe_ms: number;
}

interface ChatResponse {
  answer: string;
  citations: Citation[];
  guardrails: GuardrailOutcome[];
  router: { arm: string; policy: string };
  vector_stats: VectorStats;
}

const FALLBACK_RESPONSE: ChatResponse = {
  answer:
    "This is a simulated answer. Start the FastAPI backend (make run.api) to stream live responses.",
  citations: [
    {
      doc_id: "demo_memo.pdf",
      section: "2",
      text: "Synthetic memo describing catalyst screening throughput gains in Q3.",
    },
  ],
  guardrails: [
    { rule: "pii", status: "pass", details: "No personal data detected." },
    { rule: "grounding", status: "pass", details: "Citations supplied." },
  ],
  router: { arm: "balanced", policy: "fallback" },
  vector_stats: { avg_similarity: 0.72, retrieved: 5, ann_probe_ms: 1.4 },
};

const API_BASE = process.env.NEXT_PUBLIC_RAG_API_BASE_URL ?? "http://localhost:8000";

export default function RagChatDemoPage() {
  const [response, setResponse] = useState<ChatResponse>(FALLBACK_RESPONSE);
  const [error, setError] = useState<string | null>(null);

  const handleRun = useCallback(
    async (input: string) => {
      if (!input.trim()) {
        setError("Please provide a question about your lab notes or datasets.");
        return;
      }
      setError(null);
      try {
        const result = await fetch("/api/rag-chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query: input }),
        });
        if (!result.ok) {
          const message = await result.text();
          throw new Error(message || `Request failed with status ${result.status}`);
        }
        const payload = (await result.json()) as ChatResponse;
        setResponse(payload);
      } catch (caught) {
        console.error(caught);
        setError(
          caught instanceof Error
            ? caught.message
            : "We could not reach the chat service. Check that make run.api is active."
        );
      }
    },
    []
  );

  return (
    <main className="mx-auto flex w-full max-w-6xl flex-col gap-6 p-6">
      <header className="space-y-2">
        <h1 className="text-3xl font-semibold">Hybrid RAG Chat Demo</h1>
        <p className="text-sm text-muted-foreground">
          Connects the FastAPI pipeline to a friendly UI so research partners can ask questions and inspect guardrails.
        </p>
      </header>

      <ErrorBanner visible={Boolean(error)}>{error}</ErrorBanner>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-5">
        <div className="lg:col-span-2 space-y-4">
          <RunPanel
            title="Ask a question"
            description="Queries the FastAPI /api/chat endpoint via a Next.js proxy."
            placeholder="e.g. Summarise the latest battery cycling experiments"
            onRun={handleRun}
            ctaLabel="Send"
          />
          <HowItWorks>
            <p>
              1. The form submits to <code>/api/rag-chat</code> which proxies to the FastAPI service running at
              <code> {API_BASE}/api/chat</code>.
            </p>
            <p>2. FastAPI ranks vector hits, composes a prompt, runs guardrails, and returns citations.</p>
            <p>3. The UI renders guardrail outcomes, router arm, and vector statistics for quick QA.</p>
          </HowItWorks>
        </div>
        <div className="lg:col-span-3 space-y-4">
          <ResultPane
            title={`Response (arm: ${response.router.arm}, policy: ${response.router.policy})`}
            footer={`Retrieved ${response.vector_stats.retrieved} chunks • Avg similarity ${response.vector_stats.avg_similarity.toFixed(
              2
            )} • ANN latency ${response.vector_stats.ann_probe_ms.toFixed(2)} ms`}
          >
            <p className="leading-relaxed">{response.answer}</p>
            <div className="space-y-2">
              <h3 className="text-sm font-semibold">Citations</h3>
              <ul className="list-disc space-y-1 pl-5 text-xs">
                {response.citations.map((citation) => (
                  <li key={`${citation.doc_id}-${citation.section}`}>
                    <span className="font-medium">{citation.doc_id}</span> · Section {citation.section} — {citation.text}
                  </li>
                ))}
              </ul>
            </div>
            <div className="space-y-2">
              <h3 className="text-sm font-semibold">Guardrails</h3>
              <ul className="space-y-1 text-xs">
                {response.guardrails.map((guardrail) => (
                  <li key={guardrail.rule} className="flex items-center justify-between gap-2">
                    <span className="font-medium">{guardrail.rule}</span>
                    <span className={guardrail.status === "pass" ? "text-emerald-600" : "text-amber-600"}>
                      {guardrail.status}
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          </ResultPane>
        </div>
      </div>
    </main>
  );
}
