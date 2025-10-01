import { notFound } from "next/navigation";

async function loadCampaign(id: string) {
  const response = await fetch(`http://localhost:8081/campaigns/${id}`, { cache: "no-store" });
  if (!response.ok) {
    return null;
  }
  return response.json();
}

export default async function CampaignDetail({ params }: { params: { id: string } }) {
  const campaign = await loadCampaign(params.id);
  if (!campaign) {
    notFound();
  }

  return (
    <main className="space-y-6">
      <section className="bg-slate-900/60 border border-slate-800 rounded-xl p-6">
        <h2 className="text-2xl font-semibold mb-4">Campaign {campaign.campaign}</h2>
        <div className="grid grid-cols-2 gap-6 text-slate-200">
          <div>
            <h3 className="text-sm uppercase text-slate-400 mb-2">Best So Far</h3>
            <p className="text-3xl font-semibold">{campaign.best_value.toFixed(2)}</p>
          </div>
          <div>
            <h3 className="text-sm uppercase text-slate-400 mb-2">Steps</h3>
            <p className="text-3xl font-semibold">{campaign.steps}</p>
          </div>
        </div>
      </section>
      <section className="bg-slate-900/60 border border-slate-800 rounded-xl p-6">
        <h3 className="text-lg font-semibold mb-3">Glass-Box Snapshots</h3>
        <pre className="bg-slate-950/80 border border-slate-800 rounded-lg p-4 overflow-auto text-xs">
          {JSON.stringify(campaign.glass_box_snapshots, null, 2)}
        </pre>
      </section>
    </main>
  );
}
