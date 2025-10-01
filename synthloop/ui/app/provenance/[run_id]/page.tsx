import Link from "next/link";

const API = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8080";
const AUTH_HEADER =
  "Basic " + Buffer.from(`${process.env.BASIC_AUTH_USER || "admin"}:${process.env.BASIC_AUTH_PASS || "changeme"}`).toString("base64");

async function fetchProvenance(runId: string) {
  const res = await fetch(`${API}/provenance/${runId}`, {
    headers: { Authorization: AUTH_HEADER },
    cache: "no-store",
  });
  if (!res.ok) {
    throw new Error("Provenance unavailable");
  }
  return res.json();
}

export default async function ProvenancePage({ params }: { params: { run_id: string } }) {
  const prov = await fetchProvenance(params.run_id);
  return (
    <div className="space-y-4">
      <Link href="/">← Back</Link>
      <h2 className="text-lg font-semibold">Provenance {params.run_id}</h2>
      <section className="space-y-2">
        <h3 className="font-semibold">Events</h3>
        <pre className="text-xs whitespace-pre-wrap bg-slate-900 border border-slate-800 p-2 rounded">
          {JSON.stringify(prov.events, null, 2)}
        </pre>
      </section>
      <section className="space-y-2">
        <h3 className="font-semibold">Measurements</h3>
        <pre className="text-xs whitespace-pre-wrap bg-slate-900 border border-slate-800 p-2 rounded">
          {JSON.stringify(prov.measurements, null, 2)}
        </pre>
      </section>
    </div>
  );
}
