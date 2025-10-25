"use client";

import React, { useState } from "react";
import GuardrailChips from "@/components/GuardrailChips";
import RagSources from "@/components/RagSources";
import RouterBadge from "@/components/RouterBadge";
import VectorStats from "@/components/VectorStats";

type DemoAnswer = {
  answer: string;
  sources: Array<{ docId: string; section: string; snippet: string }>;
  guardrails: Array<{ name: string; passed: boolean }>;
  vectorStats: { averageSimilarity: number; retrieved: number; annProbeMs: number };
  router: { arm: string; policy: string };
};

const DEFAULT_ANSWER: DemoAnswer = {
  answer: "This is a placeholder response referencing synthetic R&D memos [1].",
  sources: [
    {
      docId: "doc_001",
      section: "Section 1",
      snippet:
        "This synthetic report focuses on solid-state battery research experiment 1-1 with stable metrics.",
    },
  ],
  guardrails: [
    { name: "pii", passed: true },
    { name: "grounding", passed: true },
  ],
  vectorStats: { averageSimilarity: 0.78, retrieved: 5, annProbeMs: 1.2 },
  router: { arm: "balanced", policy: "bandit-placeholder" },
};

export default function DemoPage() {
  const [query, setQuery] = useState("");
  const [answer] = useState<DemoAnswer>(DEFAULT_ANSWER);

  return (
    <main className="mx-auto flex max-w-6xl flex-col gap-6 p-6">
      <header className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold">Mastery Demo</h1>
          <p className="text-sm text-muted-foreground">Hybrid RAG + routing showcase with synthetic data.</p>
        </div>
        <RouterBadge arm={answer.router.arm} policy={answer.router.policy} />
      </header>

      <section className="grid grid-cols-3 gap-6">
        <div className="col-span-2 space-y-4">
          <textarea
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Ask about the synthetic R&D memos..."
            className="h-32 w-full rounded-lg border p-3 text-sm"
          />
          <div className="space-y-3 rounded-lg border p-4">
            <h2 className="text-sm font-semibold">Answer</h2>
            <p className="text-sm leading-relaxed">{answer.answer}</p>
            <GuardrailChips guardrails={answer.guardrails} />
          </div>
          <div className="rounded-lg border p-4">
            <h2 className="text-sm font-semibold">RAG Sources</h2>
            <div className="mt-3">
              <RagSources sources={answer.sources} />
            </div>
          </div>
        </div>
        <aside className="space-y-4">
          <div className="rounded-lg border p-4">
            <h2 className="text-sm font-semibold">Vector Analysis</h2>
            <div className="mt-3">
              <VectorStats
                averageSimilarity={answer.vectorStats.averageSimilarity}
                retrieved={answer.vectorStats.retrieved}
                annProbeMs={answer.vectorStats.annProbeMs}
              />
            </div>
          </div>
          <div className="rounded-lg border p-4">
            <h2 className="text-sm font-semibold">Last Eval</h2>
            <p className="text-xs text-muted-foreground">Offline eval score: 0.72 (stub).</p>
          </div>
        </aside>
      </section>
    </main>
  );
}
