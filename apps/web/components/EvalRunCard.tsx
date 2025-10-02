import React from "react";

type Metric = {
  name: string;
  value: number;
};

type EvalRunCardProps = {
  runId: string;
  createdAt: string;
  metrics: Metric[];
};

export const EvalRunCard: React.FC<EvalRunCardProps> = ({ runId, createdAt, metrics }) => {
  return (
    <article className="space-y-3 rounded-lg border p-4">
      <header className="flex items-center justify-between">
        <h3 className="text-sm font-semibold">Run {runId}</h3>
        <time className="text-xs text-muted-foreground">{createdAt}</time>
      </header>
      <dl className="grid grid-cols-3 gap-3 text-center text-sm">
        {metrics.map((metric) => (
          <div key={metric.name} className="rounded border p-2">
            <dt className="text-xs uppercase text-muted-foreground">{metric.name}</dt>
            <dd className="mt-1 font-semibold">{metric.value.toFixed(2)}</dd>
          </div>
        ))}
      </dl>
    </article>
  );
};

export default EvalRunCard;
