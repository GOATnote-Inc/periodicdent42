"use client";

import { useEffect, useMemo, useState } from "react";

interface MetricsSummary {
  window: string;
  cycle_time_p50: number;
  cycle_time_p90: number;
  throughput_per_day: number;
  wip: number;
  yield_rate: number;
  defects: Record<string, number>;
  units: number;
}

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const AUTH_HEADER = `Basic ${btoa(`${process.env.NEXT_PUBLIC_AUTH_USER || "pilot"}:${process.env.NEXT_PUBLIC_AUTH_PASS || "changeme"}`)}`;

async function fetchSummary(window: string): Promise<MetricsSummary> {
  const response = await fetch(`${API_URL}/metrics/${window}-summary`, {
    headers: {
      Authorization: AUTH_HEADER
    }
  });
  if (!response.ok) {
    throw new Error("Failed to load metrics");
  }
  return response.json();
}

export default function DashboardPage() {
  const [baseline, setBaseline] = useState<MetricsSummary | null>(null);
  const [pilot, setPilot] = useState<MetricsSummary | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([fetchSummary("baseline"), fetchSummary("pilot")])
      .then(([baselineSummary, pilotSummary]) => {
        setBaseline(baselineSummary);
        setPilot(pilotSummary);
      })
      .catch((err) => setError(err.message));
  }, []);

  const yieldDelta = useMemo(() => {
    if (!baseline || !pilot) return 0;
    return (pilot.yield_rate - baseline.yield_rate) * 100;
  }, [baseline, pilot]);

  const cycleDelta = useMemo(() => {
    if (!baseline || !pilot) return 0;
    return ((pilot.cycle_time_p50 - baseline.cycle_time_p50) / Math.max(baseline.cycle_time_p50, 1)) * 100;
  }, [baseline, pilot]);

  return (
    <div className="grid gap-4">
      <section className="card">
        <h2 className="text-xl font-semibold">Pilot KPI Dashboard</h2>
        {error && <p className="mt-2 text-sm text-red-400">{error}</p>}
        <div className="mt-4 grid gap-4 md:grid-cols-3">
          <div className="rounded-lg bg-slate-800 p-4">
            <p className="text-xs uppercase text-slate-400">Median Cycle Time</p>
            <p className="text-2xl font-semibold">{pilot?.cycle_time_p50?.toFixed(0) ?? "--"}s</p>
            <p className={`text-xs ${cycleDelta < 0 ? "text-emerald-400" : "text-red-400"}`}>
              {cycleDelta.toFixed(1)}% vs baseline
            </p>
          </div>
          <div className="rounded-lg bg-slate-800 p-4">
            <p className="text-xs uppercase text-slate-400">Yield</p>
            <p className="text-2xl font-semibold">{(pilot?.yield_rate ?? 0).toFixed(2)}</p>
            <p className={`text-xs ${yieldDelta >= 0 ? "text-emerald-400" : "text-red-400"}`}>
              {yieldDelta.toFixed(1)} pts vs baseline
            </p>
          </div>
          <div className="rounded-lg bg-slate-800 p-4">
            <p className="text-xs uppercase text-slate-400">Throughput / day</p>
            <p className="text-2xl font-semibold">{pilot?.throughput_per_day?.toFixed(1) ?? "--"}</p>
            <p className="text-xs text-slate-400">WIP {pilot?.wip?.toFixed(1) ?? "--"}</p>
          </div>
        </div>
      </section>
      <section className="card">
        <h3 className="text-lg font-semibold">Defect Pareto</h3>
        <div className="mt-4 grid gap-2">
          {pilot &&
            Object.entries(pilot.defects)
              .sort((a, b) => b[1] - a[1])
              .map(([code, count]) => (
                <div key={code} className="flex items-center justify-between rounded bg-slate-800 px-3 py-2 text-sm">
                  <span>{code}</span>
                  <span className="text-slate-300">{count}</span>
                </div>
              ))}
        </div>
      </section>
      <section className="card">
        <h3 className="text-lg font-semibold">Live Events</h3>
        <p className="mt-2 text-sm text-slate-300">
          Subscribe to <code>/metrics/stream</code> for server-sent updates and wire into your preferred dashboard.
        </p>
      </section>
    </div>
  );
}
