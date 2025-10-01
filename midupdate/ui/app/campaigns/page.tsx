import { CampaignTable } from "../../components/CampaignTable";

export default function CampaignsPage() {
  return (
    <main className="space-y-6">
      <section>
        <h2 className="text-2xl font-semibold mb-2">Campaign Overview</h2>
        <p className="text-slate-300 mb-4">
          Snapshots of each simulated campaign with the latest best-so-far metric, regret, and structured rationale summary.
          Run <code>make campaigns-demo</code> to populate live data.
        </p>
        <CampaignTable />
      </section>
    </main>
  );
}
