"use client";

import { useEffect, useState } from "react";
import ReactFlow, { Background, Controls, MiniMap, Node } from "react-flow-renderer";

interface ProvenanceNode {
  id: string;
  type: string;
  label: string;
  data: Record<string, unknown>;
  parents: string[];
}

async function fetchGraph(runId: string) {
  const base = process.env.NEXT_PUBLIC_ORCHESTRATOR ?? "http://127.0.0.1:8000";
  const res = await fetch(`${base}/provenance/${runId}`);
  if (!res.ok) {
    throw new Error("Failed to fetch provenance");
  }
  return res.json();
}

export default function ProvenancePage({ params }: { params: { id: string } }) {
  const [nodes, setNodes] = useState<Node[]>([]);
  const [selected, setSelected] = useState<ProvenanceNode | null>(null);

  useEffect(() => {
    fetchGraph(params.id).then((graph) => {
      const flowNodes: Node[] = graph.nodes.map((node: ProvenanceNode, idx: number) => ({
        id: node.id,
        data: { label: `${node.type}: ${node.label}` },
        position: { x: idx * 150, y: idx * 60 },
      }));
      setNodes(flowNodes);
    });
  }, [params.id]);

  return (
    <div className="grid gap-4 md:grid-cols-[2fr,1fr]">
      <div className="h-[480px] rounded border border-slate-800">
        <ReactFlow nodes={nodes} edges={[]} onNodeClick={(_, node) => setSelected(node as unknown as ProvenanceNode)}>
          <Background />
          <MiniMap />
          <Controls />
        </ReactFlow>
      </div>
      <div className="rounded border border-slate-800 p-4 text-xs">
        <h3 className="font-semibold">Node details</h3>
        {selected ? (
          <pre className="mt-2 whitespace-pre-wrap text-slate-300">
            {JSON.stringify(selected, null, 2)}
          </pre>
        ) : (
          <p className="mt-2 text-slate-400">Select a node to inspect payload.</p>
        )}
      </div>
    </div>
  );
}
