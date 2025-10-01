"use client";

import { useEffect, useState } from "react";

export default function LogsPanel({ runId }: { runId: string }) {
  const [logs, setLogs] = useState<string[]>([]);

  useEffect(() => {
    const base = process.env.NEXT_PUBLIC_ORCHESTRATOR ?? "http://127.0.0.1:8000";
    const source = new EventSource(`${base}/experiments/${runId}/logs`);
    source.onmessage = (event) => {
      setLogs((prev) => [...prev, event.data]);
    };
    return () => {
      source.close();
    };
  }, [runId]);

  return (
    <div className="rounded border border-slate-800 p-4">
      <h3 className="font-medium">Live logs</h3>
      <pre className="mt-2 max-h-56 overflow-y-auto whitespace-pre-wrap text-xs text-slate-300">
        {logs.length > 0 ? logs.join("\n") : "Awaiting log data..."}
      </pre>
    </div>
  );
}
