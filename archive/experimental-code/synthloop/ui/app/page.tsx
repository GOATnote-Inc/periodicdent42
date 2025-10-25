import Link from "next/link";

const API = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8080";
const AUTH_HEADER =
  "Basic " + Buffer.from(`${process.env.BASIC_AUTH_USER || "admin"}:${process.env.BASIC_AUTH_PASS || "changeme"}`).toString("base64");

async function fetchRuns() {
  const res = await fetch(`${API}/synthesis/runs?outcome=all`, {
    headers: { Authorization: AUTH_HEADER },
    cache: "no-store",
  });
  if (!res.ok) {
    return [];
  }
  return res.json();
}

function badge(outcome: boolean | null, failure: string | null) {
  if (outcome === null) return "badge badge-warn";
  if (outcome) return "badge badge-pass";
  return "badge badge-fail";
}

export default async function Page() {
  const runs = await fetchRuns();
  const counts = runs.reduce(
    (acc: any, run: any) => {
      if (run.outcome === true) acc.positive += 1;
      else if (run.outcome === false) acc.negative += 1;
      return acc;
    },
    { positive: 0, negative: 0 }
  );
  return (
    <div className="space-y-4">
      <div className="flex gap-4">
        <div className="badge badge-pass">Positive: {counts.positive}</div>
        <div className="badge badge-fail">Negative: {counts.negative}</div>
      </div>
      <table className="w-full border border-slate-800 text-sm">
        <thead>
          <tr className="bg-slate-900">
            <th className="p-2 text-left">Run ID</th>
            <th className="p-2 text-left">Backend</th>
            <th className="p-2 text-left">Outcome</th>
            <th className="p-2 text-left">Failure Mode</th>
          </tr>
        </thead>
        <tbody>
          {runs.map((run: any) => (
            <tr key={run.run_id} className="border-t border-slate-800">
              <td className="p-2">
                <Link href={`/synthesis/${run.run_id}`} className="text-blue-400 underline">
                  {run.run_id}
                </Link>
              </td>
              <td className="p-2">{run.backend}</td>
              <td className="p-2">
                <span className={badge(run.outcome, run.failure_mode)}>
                  {run.outcome === null ? "pending" : run.outcome ? "positive" : "negative"}
                </span>
              </td>
              <td className="p-2">{run.failure_mode || ""}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
