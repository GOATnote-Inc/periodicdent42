import React from "react";

type VectorStatsProps = {
  averageSimilarity: number;
  retrieved: number;
  annProbeMs: number;
};

export const VectorStats: React.FC<VectorStatsProps> = ({ averageSimilarity, retrieved, annProbeMs }) => {
  return (
    <dl className="grid grid-cols-3 gap-3 text-center">
      <div className="rounded-lg border p-3">
        <dt className="text-xs uppercase text-muted-foreground">Avg. Similarity</dt>
        <dd className="text-lg font-semibold">{averageSimilarity.toFixed(2)}</dd>
      </div>
      <div className="rounded-lg border p-3">
        <dt className="text-xs uppercase text-muted-foreground">Retrieved</dt>
        <dd className="text-lg font-semibold">{retrieved}</dd>
      </div>
      <div className="rounded-lg border p-3">
        <dt className="text-xs uppercase text-muted-foreground">ANN Probe (ms)</dt>
        <dd className="text-lg font-semibold">{annProbeMs.toFixed(1)}</dd>
      </div>
    </dl>
  );
};

export default VectorStats;
