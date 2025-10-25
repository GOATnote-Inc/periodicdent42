"use client";

import { useState } from "react";
import dynamic from "next/dynamic";

// Dynamically import Plotly to avoid SSR issues
const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

interface BETEPrediction {
  formula: string;
  mp_id: string | null;
  tc_kelvin: number;
  tc_std: number;
  lambda_ep: number;
  lambda_std: number;
  omega_log_K: number;
  omega_log_std_K: number;
  mu_star: number;
  input_hash: string;
  evidence_url: string;
  timestamp: string;
  alpha2F?: {
    omega_eV: number[];
    mean: number[];
    std: number[];
  };
}

interface ScreeningResult {
  run_id: string;
  n_materials: number;
  status: string;
  results_url: string;
}

export default function BETEPage() {
  const [activeTab, setActiveTab] = useState<"single" | "batch">("single");
  const [inputType, setInputType] = useState<"mp_id" | "cif">("mp_id");
  const [mpId, setMpId] = useState("mp-48");
  const [cifContent, setCifContent] = useState("");
  const [muStar, setMuStar] = useState(0.1);
  
  const [prediction, setPrediction] = useState<BETEPrediction | null>(null);
  const [screeningResult, setScreeningResult] = useState<ScreeningResult | null>(null);
  const [batchMpIds, setBatchMpIds] = useState("mp-48\nmp-66\nmp-134");
  const [batchWorkers, setBatchWorkers] = useState(4);
  
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSinglePrediction = async () => {
    setLoading(true);
    setError(null);
    setPrediction(null);

    try {
      const payload: any = { mu_star: muStar };
      if (inputType === "mp_id") {
        payload.mp_id = mpId;
      } else {
        payload.cif_content = cifContent;
      }

      const response = await fetch("/api/bete/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Prediction failed");
      }

      const data = await response.json();
      setPrediction(data);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleBatchScreening = async () => {
    setLoading(true);
    setError(null);
    setScreeningResult(null);

    try {
      const mp_ids = batchMpIds.split("\n").filter(id => id.trim());
      const payload = {
        mp_ids,
        mu_star: muStar,
        n_workers: batchWorkers,
      };

      const response = await fetch("/api/bete/screen", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Screening failed");
      }

      const data = await response.json();
      setScreeningResult(data);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const downloadEvidence = (evidenceUrl: string) => {
    window.open(evidenceUrl, "_blank");
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-5xl font-bold text-gray-900 mb-4">
            BETE-NET Superconductor Screening
          </h1>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Predict electron-phonon coupling and Tc at <span className="font-semibold text-indigo-600">10⁵× speedup vs DFT</span> using bootstrapped ensemble GNNs
          </p>
          <div className="mt-6 flex justify-center gap-4 text-sm text-gray-500">
            <span>⚡ 5s per prediction</span>
            <span>•</span>
            <span>📊 Evidence packs</span>
            <span>•</span>
            <span>🔬 BCS/Allen-Dynes formula</span>
          </div>
        </div>

        {/* Tabs */}
        <div className="flex justify-center mb-8">
          <div className="inline-flex rounded-lg border border-gray-200 bg-white p-1">
            <button
              onClick={() => setActiveTab("single")}
              className={`px-6 py-2 rounded-md font-medium transition-colors ${
                activeTab === "single"
                  ? "bg-indigo-600 text-white"
                  : "text-gray-700 hover:bg-gray-100"
              }`}
            >
              Single Prediction
            </button>
            <button
              onClick={() => setActiveTab("batch")}
              className={`px-6 py-2 rounded-md font-medium transition-colors ${
                activeTab === "batch"
                  ? "bg-indigo-600 text-white"
                  : "text-gray-700 hover:bg-gray-100"
              }`}
            >
              Batch Screening
            </button>
          </div>
        </div>

        {/* Error Alert */}
        {error && (
          <div className="mb-8 bg-red-50 border border-red-200 text-red-800 px-6 py-4 rounded-lg">
            <p className="font-semibold">Error:</p>
            <p>{error}</p>
          </div>
        )}

        {/* Single Prediction Tab */}
        {activeTab === "single" && (
          <div className="bg-white rounded-2xl shadow-xl p-8 mb-8">
            <h2 className="text-2xl font-bold text-gray-900 mb-6">Single Structure Prediction</h2>

            {/* Input Type Selector */}
            <div className="mb-6">
              <label className="block text-sm font-medium text-gray-700 mb-2">Input Type</label>
              <div className="flex gap-4">
                <button
                  onClick={() => setInputType("mp_id")}
                  className={`flex-1 px-4 py-2 rounded-lg border-2 transition-colors ${
                    inputType === "mp_id"
                      ? "border-indigo-600 bg-indigo-50 text-indigo-700"
                      : "border-gray-200 hover:border-gray-300"
                  }`}
                >
                  Materials Project ID
                </button>
                <button
                  onClick={() => setInputType("cif")}
                  className={`flex-1 px-4 py-2 rounded-lg border-2 transition-colors ${
                    inputType === "cif"
                      ? "border-indigo-600 bg-indigo-50 text-indigo-700"
                      : "border-gray-200 hover:border-gray-300"
                  }`}
                >
                  Upload CIF
                </button>
              </div>
            </div>

            {/* MP-ID Input */}
            {inputType === "mp_id" && (
              <div className="mb-6">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Materials Project ID
                </label>
                <input
                  type="text"
                  value={mpId}
                  onChange={(e) => setMpId(e.target.value)}
                  placeholder="e.g., mp-48"
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-600 focus:border-transparent"
                />
                <p className="mt-2 text-sm text-gray-500">
                  Example: mp-48 (Nb), mp-763 (MgB₂), mp-134 (Al)
                </p>
              </div>
            )}

            {/* CIF Input */}
            {inputType === "cif" && (
              <div className="mb-6">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  CIF Content
                </label>
                <textarea
                  value={cifContent}
                  onChange={(e) => setCifContent(e.target.value)}
                  placeholder="Paste CIF file content here..."
                  rows={10}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-600 focus:border-transparent font-mono text-sm"
                />
              </div>
            )}

            {/* μ* Parameter */}
            <div className="mb-6">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Coulomb Pseudopotential (μ*)
              </label>
              <input
                type="number"
                value={muStar}
                onChange={(e) => setMuStar(parseFloat(e.target.value))}
                step={0.01}
                min={0.05}
                max={0.20}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-600 focus:border-transparent"
              />
              <p className="mt-2 text-sm text-gray-500">
                Typical values: 0.10 (default), 0.13 (higher screening)
              </p>
            </div>

            {/* Predict Button */}
            <button
              onClick={handleSinglePrediction}
              disabled={loading}
              className="w-full bg-indigo-600 text-white px-6 py-3 rounded-lg font-semibold hover:bg-indigo-700 transition-colors disabled:bg-gray-400 disabled:cursor-not-allowed"
            >
              {loading ? "Predicting..." : "Predict Tc"}
            </button>
          </div>
        )}

        {/* Batch Screening Tab */}
        {activeTab === "batch" && (
          <div className="bg-white rounded-2xl shadow-xl p-8 mb-8">
            <h2 className="text-2xl font-bold text-gray-900 mb-6">Batch Screening</h2>

            <div className="mb-6">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Materials Project IDs (one per line)
              </label>
              <textarea
                value={batchMpIds}
                onChange={(e) => setBatchMpIds(e.target.value)}
                rows={8}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-600 focus:border-transparent font-mono text-sm"
              />
            </div>

            <div className="grid grid-cols-2 gap-4 mb-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  μ*
                </label>
                <input
                  type="number"
                  value={muStar}
                  onChange={(e) => setMuStar(parseFloat(e.target.value))}
                  step={0.01}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-600 focus:border-transparent"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Workers
                </label>
                <input
                  type="number"
                  value={batchWorkers}
                  onChange={(e) => setBatchWorkers(parseInt(e.target.value))}
                  min={1}
                  max={16}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-600 focus:border-transparent"
                />
              </div>
            </div>

            <button
              onClick={handleBatchScreening}
              disabled={loading}
              className="w-full bg-indigo-600 text-white px-6 py-3 rounded-lg font-semibold hover:bg-indigo-700 transition-colors disabled:bg-gray-400 disabled:cursor-not-allowed"
            >
              {loading ? "Starting Screening..." : "Start Batch Screening"}
            </button>
          </div>
        )}

        {/* Single Prediction Results */}
        {prediction && (
          <div className="bg-white rounded-2xl shadow-xl p-8 mb-8">
            <h2 className="text-2xl font-bold text-gray-900 mb-6">Prediction Results</h2>

            {/* Summary Cards */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
              <div className="bg-gradient-to-br from-blue-50 to-blue-100 p-6 rounded-xl">
                <p className="text-sm text-blue-600 font-medium mb-1">Formula</p>
                <p className="text-3xl font-bold text-blue-900">{prediction.formula}</p>
              </div>
              <div className="bg-gradient-to-br from-purple-50 to-purple-100 p-6 rounded-xl">
                <p className="text-sm text-purple-600 font-medium mb-1">Tc (K)</p>
                <p className="text-3xl font-bold text-purple-900">
                  {prediction.tc_kelvin.toFixed(2)} <span className="text-lg">±{prediction.tc_std.toFixed(2)}</span>
                </p>
              </div>
              <div className="bg-gradient-to-br from-green-50 to-green-100 p-6 rounded-xl">
                <p className="text-sm text-green-600 font-medium mb-1">λ</p>
                <p className="text-3xl font-bold text-green-900">
                  {prediction.lambda_ep.toFixed(3)} <span className="text-lg">±{prediction.lambda_std.toFixed(3)}</span>
                </p>
              </div>
              <div className="bg-gradient-to-br from-orange-50 to-orange-100 p-6 rounded-xl">
                <p className="text-sm text-orange-600 font-medium mb-1">⟨ω_log⟩ (K)</p>
                <p className="text-3xl font-bold text-orange-900">
                  {prediction.omega_log_K.toFixed(1)}
                </p>
              </div>
            </div>

            {/* α²F(ω) Plot */}
            {prediction.alpha2F && (
              <div className="mb-8">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">
                  Electron-Phonon Spectral Function α²F(ω)
                </h3>
                <Plot
                  data={[
                    {
                      x: prediction.alpha2F.omega_eV.map(w => w * 1000), // eV → meV
                      y: prediction.alpha2F.mean,
                      type: "scatter",
                      mode: "lines",
                      name: "α²F(ω)",
                      line: { color: "#4f46e5", width: 3 },
                    },
                    {
                      x: prediction.alpha2F.omega_eV.map(w => w * 1000),
                      y: prediction.alpha2F.mean.map((m, i) => m + prediction.alpha2F!.std[i]),
                      type: "scatter",
                      mode: "lines",
                      name: "+1σ",
                      line: { color: "#4f46e5", width: 0 },
                      showlegend: false,
                    },
                    {
                      x: prediction.alpha2F.omega_eV.map(w => w * 1000),
                      y: prediction.alpha2F.mean.map((m, i) => m - prediction.alpha2F!.std[i]),
                      type: "scatter",
                      mode: "lines",
                      name: "±1σ",
                      fill: "tonexty",
                      fillcolor: "rgba(79, 70, 229, 0.2)",
                      line: { color: "#4f46e5", width: 0 },
                    },
                  ]}
                  layout={{
                    title: `${prediction.formula} | Tc = ${prediction.tc_kelvin.toFixed(2)}±${prediction.tc_std.toFixed(2)} K`,
                    xaxis: { title: "Phonon Frequency ω (meV)" },
                    yaxis: { title: "α²F(ω)" },
                    hovermode: "x unified",
                    height: 500,
                  }}
                  config={{ responsive: true }}
                  className="w-full"
                />
              </div>
            )}

            {/* Metadata */}
            <div className="bg-gray-50 p-6 rounded-xl">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Metadata</h3>
              <dl className="grid grid-cols-2 gap-4 text-sm">
                {prediction.mp_id && (
                  <>
                    <dt className="text-gray-600 font-medium">MP-ID:</dt>
                    <dd className="text-gray-900">{prediction.mp_id}</dd>
                  </>
                )}
                <dt className="text-gray-600 font-medium">Input Hash:</dt>
                <dd className="text-gray-900 font-mono text-xs">{prediction.input_hash.slice(0, 16)}...</dd>
                <dt className="text-gray-600 font-medium">μ*:</dt>
                <dd className="text-gray-900">{prediction.mu_star}</dd>
                <dt className="text-gray-600 font-medium">Timestamp:</dt>
                <dd className="text-gray-900">{new Date(prediction.timestamp).toLocaleString()}</dd>
              </dl>
            </div>

            {/* Download Evidence Pack */}
            <button
              onClick={() => downloadEvidence(prediction.evidence_url)}
              className="mt-6 w-full bg-green-600 text-white px-6 py-3 rounded-lg font-semibold hover:bg-green-700 transition-colors"
            >
              📦 Download Evidence Pack (ZIP)
            </button>
          </div>
        )}

        {/* Batch Screening Results */}
        {screeningResult && (
          <div className="bg-white rounded-2xl shadow-xl p-8">
            <h2 className="text-2xl font-bold text-gray-900 mb-6">Screening Queued</h2>
            <div className="bg-blue-50 border border-blue-200 p-6 rounded-lg">
              <p className="text-lg mb-2">
                <span className="font-semibold">Run ID:</span>{" "}
                <span className="font-mono text-sm">{screeningResult.run_id}</span>
              </p>
              <p className="text-lg mb-2">
                <span className="font-semibold">Materials:</span> {screeningResult.n_materials}
              </p>
              <p className="text-lg mb-4">
                <span className="font-semibold">Status:</span>{" "}
                <span className="capitalize text-blue-600">{screeningResult.status}</span>
              </p>
              <button
                onClick={() => downloadEvidence(screeningResult.results_url)}
                className="bg-blue-600 text-white px-6 py-2 rounded-lg font-semibold hover:bg-blue-700 transition-colors"
              >
                Download Results (when ready)
              </button>
            </div>
          </div>
        )}

        {/* Footer */}
        <div className="mt-12 text-center text-gray-600 text-sm">
          <p className="mb-2">
            Powered by <a href="https://github.com/henniggroup/BETE-NET" className="text-indigo-600 hover:underline" target="_blank" rel="noopener noreferrer">BETE-NET</a> (Apache 2.0)
          </p>
          <p>
            © 2025 GOATnote Autonomous Research Lab Initiative
          </p>
        </div>
      </div>
    </div>
  );
}

