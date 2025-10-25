"use client";

import { FormEvent, useState } from "react";

interface Theme {
  theme: string;
  triage: string;
  count: number;
  summary: string;
  proposed_fix: string;
}

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const AUTH_HEADER = `Basic ${btoa(`${process.env.NEXT_PUBLIC_AUTH_USER || "pilot"}:${process.env.NEXT_PUBLIC_AUTH_PASS || "changeme"}`)}`;

export default function FeedbackPage() {
  const [text, setText] = useState("");
  const [severity, setSeverity] = useState("P2");
  const [step, setStep] = useState("queue");
  const [themes, setThemes] = useState<Theme[]>([]);
  const [status, setStatus] = useState<string | null>(null);
  const [guide, setGuide] = useState<string | null>(null);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setStatus("Submitting...");
    const payload = {
      items: [
        {
          ts: new Date().toISOString(),
          severity,
          step,
          text,
          frustration: 4,
          task_success: false,
          time_on_task_s: 900,
          tags: []
        }
      ]
    };
    try {
      const response = await fetch(`${API_URL}/feedback`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: AUTH_HEADER
        },
        body: JSON.stringify(payload)
      });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const data = await response.json();
      setThemes(data.themes);
      setGuide(data.interview_guide);
      setStatus(`Captured ${data.count} item(s).`);
      setText("");
    } catch (err: any) {
      setStatus(err.message);
    }
  }

  return (
    <div className="card">
      <h2 className="text-xl font-semibold">Feedback Inbox</h2>
      <form onSubmit={submit} className="mt-4 grid gap-3 text-sm text-slate-200">
        <label className="grid gap-2">
          <span>Workflow Step</span>
          <select value={step} onChange={(e) => setStep(e.target.value)} className="rounded bg-slate-800 p-2">
            <option value="queue">Queue</option>
            <option value="work">Work</option>
            <option value="review">Review</option>
          </select>
        </label>
        <label className="grid gap-2">
          <span>Severity</span>
          <select value={severity} onChange={(e) => setSeverity(e.target.value)} className="rounded bg-slate-800 p-2">
            <option value="P0">P0 Critical</option>
            <option value="P1">P1 High</option>
            <option value="P2">P2 Medium</option>
            <option value="P3">P3 Low</option>
          </select>
        </label>
        <label className="grid gap-2">
          <span>Description</span>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            className="rounded bg-slate-800 p-3"
            rows={4}
            placeholder="Describe what happened"
          />
        </label>
        <button
          type="submit"
          className="rounded bg-emerald-500 px-4 py-2 text-sm font-semibold text-slate-900 hover:bg-emerald-400"
        >
          Submit Feedback
        </button>
      </form>
      {status && <p className="mt-3 text-xs text-slate-400">{status}</p>}
      <div className="mt-4 grid gap-3">
        {themes.map((theme) => (
          <div key={theme.theme} className="rounded border border-slate-800 bg-slate-900 p-3 text-sm">
            <div className="flex items-center justify-between">
              <h3 className="font-semibold text-slate-100">{theme.theme}</h3>
              <span className="text-xs text-amber-300">Severity {theme.triage}</span>
            </div>
            <p className="mt-1 text-slate-300">{theme.summary}</p>
            <p className="mt-2 text-xs text-emerald-300">Proposed fix: {theme.proposed_fix}</p>
          </div>
        ))}
      </div>
      {guide && (
        <div className="card mt-4">
          <h3 className="text-lg font-semibold">Interview Kit</h3>
          <pre className="mt-2 whitespace-pre-wrap text-xs text-slate-300">{guide}</pre>
        </div>
      )}
    </div>
  );
}
