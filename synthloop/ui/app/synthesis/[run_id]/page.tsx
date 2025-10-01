import Link from "next/link";

const API = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8080";
const AUTH_HEADER =
  "Basic " + Buffer.from(`${process.env.BASIC_AUTH_USER || "admin"}:${process.env.BASIC_AUTH_PASS || "changeme"}`).toString("base64");

async function fetchRun(runId: string) {
  const res = await fetch(`${API}/synthesis/runs/${runId}`, {
    headers: { Authorization: AUTH_HEADER },
    cache: "no-store",
  });
  if (!res.ok) {
    throw new Error("Run not found");
  }
  return res.json();
}

async function fetchQC(runId: string) {
  const res = await fetch(`${API}/qc/${runId}`, {
    headers: { Authorization: AUTH_HEADER },
    cache: "no-store",
  });
  if (!res.ok) {
    return null;
  }
  return res.json();
}

export default async function RunPage({ params }: { params: { run_id: string } }) {
  const run = await fetchRun(params.run_id);
  const qc = await fetchQC(params.run_id);
  return (
    <div className="space-y-4">
      <Link href="/">← Back</Link>
      <div className="space-y-2">
        <h2 className="text-lg font-semibold">Run {params.run_id}</h2>
        <div>Status: {run.status}</div>
        {run.outcome && (
          <div>
            Outcome: {run.outcome.success ? "✅" : "❌"} {run.outcome.failure_mode || ""}
          </div>
        )}
      </div>
      {qc && (
        <div className="space-y-2">
          <h3 className="font-semibold">QC Report</h3>
          <div className={`badge badge-${qc.overall}`}>Overall: {qc.overall}</div>
          <ul className="space-y-1">
            {qc.rules.map((rule: any) => (
              <li key={rule.name} className="border border-slate-800 p-2 rounded">
                <div className="flex justify-between">
                  <span>{rule.name}</span>
                  <span className={`badge badge-${rule.status}`}>{rule.status}</span>
                </div>
                <pre className="text-xs whitespace-pre-wrap opacity-80">{JSON.stringify(rule.evidence, null, 2)}</pre>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
