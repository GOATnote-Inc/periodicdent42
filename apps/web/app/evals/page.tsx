"use client";

import React from "react";
import EvalRunCard from "@/components/EvalRunCard";

const MOCK_RUNS = [
  {
    runId: "offline-stub",
    createdAt: new Date().toISOString(),
    metrics: [
      { name: "exact_match", value: 0.2 },
      { name: "rouge_l", value: 0.5 },
      { name: "faithfulness", value: 1.0 },
    ],
  },
];

export default function EvalsPage() {
  return (
    <main className="mx-auto flex max-w-5xl flex-col gap-6 p-6">
      <header>
        <h1 className="text-2xl font-semibold">Evaluation Runs</h1>
        <p className="text-sm text-muted-foreground">Synthetic offline evaluations for the mastery demo.</p>
      </header>
      <section className="grid gap-4">
        {MOCK_RUNS.map((run) => (
          <EvalRunCard key={run.runId} runId={run.runId} createdAt={run.createdAt} metrics={run.metrics} />
        ))}
      </section>
    </main>
  );
}
