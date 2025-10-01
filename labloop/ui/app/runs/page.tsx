import Link from "next/link";

interface RunRecord {
  run_id: string;
  created_at: string;
  updated_at: string;
  status: string;
  backend: string;
  plan: {
    name: string;
    operator: string;
    instrument: string;
  };
  eig_history: number[];
}

async function fetchRuns(): Promise<RunRecord[]> {
  const base = process.env.NEXT_PUBLIC_ORCHESTRATOR ?? "http://127.0.0.1:8000";
  const res = await fetch(`${base}/runs`, { cache: "no-store" });
  if (!res.ok) {
    throw new Error("Failed to load runs");
  }
  const data = await res.json();
  return data.runs ?? [];
}

export default async function RunsPage() {
  const runs = await fetchRuns();
  return (
    <section className="space-y-4">
      <h2 className="text-lg font-semibold">Runs</h2>
      <div className="overflow-hidden rounded border border-slate-800">
        <table className="min-w-full divide-y divide-slate-800 text-sm">
          <thead className="bg-slate-900">
            <tr>
              <th className="px-3 py-2 text-left">Run ID</th>
              <th className="px-3 py-2 text-left">Instrument</th>
              <th className="px-3 py-2 text-left">Backend</th>
              <th className="px-3 py-2 text-left">Operator</th>
              <th className="px-3 py-2 text-left">Status</th>
              <th className="px-3 py-2 text-left">EIG/hour (last)</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-900">
            {runs.map((run) => (
              <tr key={run.run_id} className="hover:bg-slate-900/60">
                <td className="px-3 py-2 font-mono text-xs">
                  <Link className="text-emerald-400" href={`/runs/${run.run_id}`}>
                    {run.run_id}
                  </Link>
                </td>
                <td className="px-3 py-2">{run.plan.instrument}</td>
                <td className="px-3 py-2">{run.backend}</td>
                <td className="px-3 py-2">{run.plan.operator}</td>
                <td className="px-3 py-2 capitalize">{run.status}</td>
                <td className="px-3 py-2">
                  {run.eig_history.length > 0
                    ? run.eig_history[run.eig_history.length - 1].toFixed(3)
                    : "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
