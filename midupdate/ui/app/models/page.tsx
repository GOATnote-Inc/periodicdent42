import { promises as fs } from "fs";
import path from "path";

type ModelInfo = {
  metrics: Record<string, number>;
  model_card: string;
  train_config: string;
};

async function loadModel(): Promise<ModelInfo | null> {
  const baseDir = path.join(process.cwd(), "training", "model_registry", "latest");
  const metricsPath = path.join(baseDir, "metrics.json");
  const cardPath = path.join(baseDir, "model_card.md");
  const configPath = path.join(baseDir, "train_config.yaml");
  try {
    const [metricsRaw, cardRaw, configRaw] = await Promise.all([
      fs.readFile(metricsPath, "utf-8"),
      fs.readFile(cardPath, "utf-8").catch(() => ""),
      fs.readFile(configPath, "utf-8").catch(() => ""),
    ]);
    return {
      metrics: JSON.parse(metricsRaw),
      model_card: cardRaw,
      train_config: configRaw,
    };
  } catch (err) {
    return null;
  }
}

export default async function ModelsPage() {
  const model = await loadModel();
  return (
    <main className="space-y-6">
      <section className="bg-slate-900/60 border border-slate-800 rounded-xl p-6">
        <h2 className="text-2xl font-semibold mb-4">Latest Planner Artifact</h2>
        {!model && <p className="text-slate-300">Run <code>make train</code> to produce a model artifact.</p>}
        {model && (
          <div className="space-y-4">
            <div>
              <h3 className="text-sm uppercase text-slate-400 mb-2">Metrics</h3>
              <pre className="bg-slate-950/70 border border-slate-800 rounded-lg p-4 text-xs">
                {JSON.stringify(model.metrics, null, 2)}
              </pre>
            </div>
            <div>
              <h3 className="text-sm uppercase text-slate-400 mb-2">Model Card</h3>
              <pre className="bg-slate-950/70 border border-slate-800 rounded-lg p-4 text-xs whitespace-pre-wrap">
                {model.model_card}
              </pre>
            </div>
            <div>
              <h3 className="text-sm uppercase text-slate-400 mb-2">Training Config</h3>
              <pre className="bg-slate-950/70 border border-slate-800 rounded-lg p-4 text-xs whitespace-pre-wrap">
                {model.train_config}
              </pre>
            </div>
          </div>
        )}
      </section>
    </main>
  );
}
