import React from "react";

type Source = {
  docId: string;
  section: string;
  snippet: string;
};

type RagSourcesProps = {
  sources: Source[];
};

export const RagSources: React.FC<RagSourcesProps> = ({ sources }) => {
  if (sources.length === 0) {
    return <p className="text-sm text-muted-foreground">No sources retrieved.</p>;
  }

  return (
    <div className="space-y-3">
      {sources.map((source, index) => (
        <article key={`${source.docId}-${index}`} className="rounded-lg border p-3">
          <header className="flex items-center justify-between text-xs text-muted-foreground">
            <span>Source {index + 1}</span>
            <span>{source.docId}</span>
          </header>
          <h3 className="mt-1 text-sm font-semibold">{source.section}</h3>
          <p className="mt-2 text-sm leading-relaxed">{source.snippet}</p>
        </article>
      ))}
    </div>
  );
};

export default RagSources;
