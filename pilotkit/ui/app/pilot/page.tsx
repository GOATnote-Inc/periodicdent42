"use client";

import { useEffect, useState } from "react";

interface CandidateScore {
  candidate: {
    name: string;
    notes?: string;
  };
  score: number;
  rationale: string;
}

const SAMPLE_PAYLOAD = {
  weights: {
    exec_sponsor: 0.2,
    data_access: 0.2,
    workflow_speed: 0.2,
    potential_value: 0.2,
    risk: 0.1,
    champion: 0.1
  },
  candidates: [
    {
      name: "Lab A",
      exec_sponsor: 5,
      data_access: 4,
      workflow_speed: 5,
      potential_value: 5,
      risk: 2,
      champion: 5,
      notes: "Fast moving assay team"
    },
    {
      name: "Lab B",
      exec_sponsor: 3,
      data_access: 5,
      workflow_speed: 3,
      potential_value: 4,
      risk: 1,
      champion: 2,
      notes: "Data rich"
    }
  ]
};

function useCandidateScores() {
  const [scores, setScores] = useState<CandidateScore[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    async function load() {
      try {
        setLoading(true);
        const url = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
        const response = await fetch(`${url}/pilot/candidates/score`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Basic ${btoa(`${process.env.NEXT_PUBLIC_AUTH_USER || "pilot"}:${process.env.NEXT_PUBLIC_AUTH_PASS || "changeme"}`)}`
          },
          body: JSON.stringify(SAMPLE_PAYLOAD),
          signal: controller.signal
        });
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        const data = await response.json();
        setScores(data.ranked);
      } catch (err: any) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    }
    load();
    return () => controller.abort();
  }, []);

  return { scores, loading, error };
}

export default function PilotPage() {
  const { scores, loading, error } = useCandidateScores();

  return (
    <div className="card">
      <h2 className="text-xl font-semibold">Partner Selection</h2>
      <p className="mt-2 text-sm text-slate-300">
        Weighted scorecard ranking candidate partners based on sponsor fit, data readiness, workflow velocity, and risk posture.
      </p>
      {loading && <p className="mt-4 text-sm text-slate-400">Ranking candidates...</p>}
      {error && <p className="mt-4 text-sm text-red-400">{error}</p>}
      <div className="mt-4 grid gap-4 md:grid-cols-2">
        {scores.map((entry, idx) => (
          <div key={entry.candidate.name} className="card border border-slate-700">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold">{entry.candidate.name}</h3>
              <span className="text-sm text-emerald-400">Score {entry.score.toFixed(2)}</span>
            </div>
            <p className="mt-2 text-sm text-slate-400">{entry.candidate.notes}</p>
            <p className="mt-2 text-xs uppercase tracking-wide text-slate-500">
              Rationale: {entry.rationale}
            </p>
            <span className="mt-3 inline-block rounded-full bg-slate-800 px-2 py-1 text-xs text-slate-300">
              Rank #{idx + 1}
            </span>
          </div>
        ))}
      </div>
      <section className="mt-6">
        <h3 className="text-lg font-semibold">Pilot Playbook</h3>
        <p className="mt-2 text-sm text-slate-300">
          Once you lock a partner, generate the playbook and legal templates from <code>configs/pilot.sample.yaml</code> using <code>make demo</code>.
        </p>
      </section>
    </div>
  );
}
