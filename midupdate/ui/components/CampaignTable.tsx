"use client";

import { useEffect, useState } from "react";

type CampaignSummary = {
  id: string;
  bestValue: number;
  stepsCompleted: number;
  regret: number;
  eigPerHour: number;
  lastRationale: string;
};

export function CampaignTable() {
  const [campaigns, setCampaigns] = useState<CampaignSummary[]>([]);

  useEffect(() => {
    async function fetchCampaigns() {
      try {
        const response = await fetch("http://localhost:8081/campaigns");
        if (!response.ok) return;
        const payload = await response.json();
        const mapped = payload.campaigns.map((entry: any) => ({
          id: entry.campaign,
          bestValue: entry.best_value,
          stepsCompleted: entry.steps,
          regret: entry.regret_curve[entry.regret_curve.length - 1] ?? 0,
          eigPerHour: entry.eig_curve[entry.eig_curve.length - 1] ?? 0,
          lastRationale:
            entry.glass_box_snapshots?.[0]?.proposed_next?.justification ?? "Structured plan captured",
        }));
        setCampaigns(mapped);
      } catch (err) {
        console.error("Failed to load campaign summaries", err);
      }
    }
    fetchCampaigns();
    const evtSource = new EventSource("http://localhost:8081/events");
    evtSource.onmessage = () => {
      fetchCampaigns();
    };
    return () => evtSource.close();
  }, []);

  return (
    <div className="overflow-hidden rounded-lg border border-slate-800 bg-slate-900/60">
      <table className="min-w-full divide-y divide-slate-800">
        <thead className="bg-slate-900/80">
          <tr>
            <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-400">
              Campaign
            </th>
            <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-400">
              Best So Far
            </th>
            <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-400">
              Steps
            </th>
            <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-400">
              Regret
            </th>
            <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-400">
              EIG / hr
            </th>
            <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-400">
              Rationale
            </th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-800">
          {campaigns.map((campaign) => (
            <tr key={campaign.id}>
              <td className="px-4 py-3 font-medium text-slate-100">{campaign.id}</td>
              <td className="px-4 py-3 text-slate-200">{campaign.bestValue.toFixed(2)}</td>
              <td className="px-4 py-3 text-slate-200">{campaign.stepsCompleted}</td>
              <td className="px-4 py-3 text-slate-200">{campaign.regret.toFixed(2)}</td>
              <td className="px-4 py-3 text-slate-200">{campaign.eigPerHour.toFixed(3)}</td>
              <td className="px-4 py-3 text-slate-300">{campaign.lastRationale}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
