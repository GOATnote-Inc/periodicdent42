import LogsPanel from "../../../components/logs-panel";
import StatusBadge from "../../../components/status-badge";

interface MeasurementResult {
  task_id: string;
  summary: Record<string, unknown>;
  data_path: string;
  started_at: string;
  finished_at: string;
}

interface RunRecord {
  run_id: string;
  status: string;
  plan: {
    name: string;
    operator: string;
    instrument: string;
  };
  eig_history: number[];
  predictive_variance: number[];
}

async function fetchRun(runId: string) {
  const base = process.env.NEXT_PUBLIC_ORCHESTRATOR ?? "http://127.0.0.1:8000";
  const res = await fetch(`${base}/experiments/${runId}`, { cache: "no-store" });
  if (!res.ok) {
    throw new Error("Run not found");
  }
  return res.json();
}

export default async function RunDetailPage({ params }: { params: { id: string } }) {
  const data = await fetchRun(params.id);
  const record: RunRecord = data.record;
  const completed: MeasurementResult[] = data.completed;
  return (
    <section className="space-y-6">
      <header className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="text-lg font-semibold">Run {record.run_id}</h2>
          <p className="text-xs text-slate-400">
            {record.plan.name} — {record.plan.instrument} — Operator {record.plan.operator}
          </p>
        </div>
        <StatusBadge status={record.status} />
      </header>
      <div className="grid gap-6 md:grid-cols-2">
        <div className="rounded border border-slate-800 p-4">
          <h3 className="font-medium">Measurements</h3>
          <ul className="mt-3 space-y-2 text-sm">
            {completed.map((item) => (
              <li key={item.task_id} className="rounded border border-slate-800/60 p-2">
                <div className="font-mono text-xs text-emerald-300">{item.task_id}</div>
                <div className="text-xs text-slate-400">{item.data_path}</div>
                <pre className="mt-1 whitespace-pre-wrap text-xs text-slate-300">
                  {JSON.stringify(item.summary, null, 2)}
                </pre>
              </li>
            ))}
          </ul>
        </div>
        <div className="space-y-4">
          <div className="rounded border border-slate-800 p-4">
            <h3 className="font-medium">EIG/hour trend</h3>
            <div className="mt-2 text-xs text-slate-300">
              {record.eig_history.length > 0
                ? record.eig_history.map((value, idx) => (
                    <div key={idx}>
                      Step {idx + 1}: {value.toFixed(3)} bits/hour
                    </div>
                  ))
                : "No tasks executed yet."}
            </div>
          </div>
          <LogsPanel runId={record.run_id} />
        </div>
      </div>
    </section>
  );
}
