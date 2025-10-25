"use client";

import { useEffect, useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const AUTH_HEADER = `Basic ${btoa(`${process.env.NEXT_PUBLIC_AUTH_USER || "pilot"}:${process.env.NEXT_PUBLIC_AUTH_PASS || "changeme"}`)}`;

interface ReportResponse {
  markdown_path: string;
  chart_path: string;
}

interface IterationPlanFile {
  markdown_path: string;
  json_path: string;
}

export default function ReportPage() {
  const [report, setReport] = useState<ReportResponse | null>(null);
  const [plan, setPlan] = useState<IterationPlanFile | null>(null);
  const [status, setStatus] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      const payload = {
        baseline_start: new Date(Date.now() - 14 * 24 * 3600 * 1000).toISOString(),
        baseline_end: new Date(Date.now() - 7 * 24 * 3600 * 1000).toISOString(),
        pilot_start: new Date(Date.now() - 7 * 24 * 3600 * 1000).toISOString(),
        pilot_end: new Date().toISOString(),
        mde_pct: 15,
        sample_size: 80,
        title: "Pilot Impact Report"
      };
      try {
        const response = await fetch(`${API_URL}/report/generate`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: AUTH_HEADER
          },
          body: JSON.stringify(payload)
        });
        if (!response.ok) throw new Error("Failed to generate report");
        const data = (await response.json()) as ReportResponse;
        setReport(data);
      } catch (err: any) {
        setStatus(err.message);
      }
    }
    load();
  }, []);

  async function createIterationPlan() {
    setStatus("Generating plan...");
    try {
      const payload = {
        metrics_delta: { cycle_time_pct: -20, yield_delta: 5 },
        top_feedback_themes: ["Queue delays", "Review friction", "Operator onboarding"],
        guardrails: { yield_rate: "Maintain > 0.95" }
      };
      const response = await fetch(`${API_URL}/iteration/plan`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: AUTH_HEADER
        },
        body: JSON.stringify(payload)
      });
      if (!response.ok) throw new Error("Failed to generate iteration plan");
      const data = (await response.json()) as IterationPlanFile;
      setPlan(data);
      setStatus("Iteration plan created.");
    } catch (err: any) {
      setStatus(err.message);
    }
  }

  return (
    <div className="card">
      <h2 className="text-xl font-semibold">Impact Report</h2>
      {status && <p className="mt-2 text-sm text-slate-400">{status}</p>}
      {report ? (
        <div className="mt-4 space-y-3 text-sm text-slate-200">
          <p>
            Markdown: <code>{report.markdown_path}</code>
          </p>
          <p>
            Chart: <code>{report.chart_path}</code>
          </p>
          <button
            onClick={createIterationPlan}
            className="rounded bg-emerald-500 px-4 py-2 text-sm font-semibold text-slate-900 hover:bg-emerald-400"
          >
            Create Next Iteration Plan
          </button>
          {plan && (
            <div className="rounded border border-slate-800 bg-slate-900 p-3 text-xs text-slate-300">
              <p>
                Markdown: <code>{plan.markdown_path}</code>
              </p>
              <p>
                JSON: <code>{plan.json_path}</code>
              </p>
            </div>
          )}
        </div>
      ) : (
        <p className="mt-4 text-sm text-slate-400">Generating pilot impact report...</p>
      )}
    </div>
  );
}
