import Link from "next/link";

export default function HomePage() {
  return (
    <main className="space-y-6">
      <section className="bg-slate-900/70 border border-slate-800 rounded-xl p-6">
        <h2 className="text-xl font-semibold mb-2">Campaign Control Center</h2>
        <p className="text-slate-300">
          Use the navigation below to inspect in-flight campaigns, drill into a single campaign,
          or review the latest fine-tuned planner model with provenance.
        </p>
        <div className="mt-4 flex gap-4">
          <Link href="/campaigns" className="bg-emerald-500 text-black px-4 py-2 rounded-md font-medium">
            View Campaigns
          </Link>
          <Link href="/models" className="bg-sky-500 text-black px-4 py-2 rounded-md font-medium">
            View Models
          </Link>
        </div>
      </section>
    </main>
  );
}
