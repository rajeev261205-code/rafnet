import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../services/api";
import type { AnalyzeResponse, DashboardResponse } from "../types";
import { Gauge, Panel } from "../components/ui";
import { VerdictBadge } from "../components/Verdict";
import type { VerdictLabel } from "../types";

function Stat({ label, value, color }: { label: string; value: number; color: string }): React.JSX.Element {
  return (
    <div className="soc-panel p-5">
      <div className="mono text-xs uppercase tracking-widest" style={{ color: "#7d93b2" }}>
        {label}
      </div>
      <div className="mono mt-1 text-3xl font-semibold" style={{ color }}>
        {value}
      </div>
    </div>
  );
}

export default function Dashboard(): React.JSX.Element {
  const [data, setData] = useState<DashboardResponse | null>(null);
  const [items, setItems] = useState<AnalyzeResponse[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [historyError, setHistoryError] = useState<string | null>(null);

  useEffect(() => {
    api.dashboard().then(setData).catch((e: Error) => setError(e.message));
    api.history(100, 0).then((h) => setItems(h.items)).catch((e: Error) => setHistoryError(e.message));
  }, []);

  if (error) return <div className="soc-panel p-6 text-red-300">Dashboard unavailable: {error}</div>;
  if (!data) return <div className="soc-panel p-6" style={{ color: "#7d93b2" }}>Loading live SOC metrics…</div>;

  const latencies = (items ?? []).map((r) => r.processing_time_ms).filter((v) => Number.isFinite(v));
  const avgLatency = latencies.length > 0 ? latencies.reduce((a, b) => a + b, 0) / latencies.length : null;
  const verdictMix = new Map<VerdictLabel | "legacy", number>();
  for (const r of items ?? []) {
    const label = (r.verdict?.label ?? "legacy") as VerdictLabel | "legacy";
    verdictMix.set(label, (verdictMix.get(label) ?? 0) + 1);
  }

  return (
    <div>
      <div className="mb-5 flex items-center gap-3">
        <h1 className="text-2xl font-bold">Overview</h1>
        {data.mock_mode && (
          <span className="soc-chip mono" style={{ color: "#fbbf24" }}>
            MOCK MODE — not a real prediction
          </span>
        )}
      </div>

      <div className="grid grid-cols-2 gap-4 xl:grid-cols-5">
        <Stat label="Total analyses" value={data.total_analyses} color="#22d3ee" />
        <Stat label="Synthetic" value={data.synthetic_detections} color="#f87171" />
        <Stat label="Authentic" value={data.authentic_detections} color="#34d399" />
        <Stat label="Uncertain" value={data.uncertain_detections} color="#fbbf24" />
        <Stat label="High-risk events" value={data.high_risk_events} color="#f87171" />
      </div>

      <div className="mt-4 grid grid-cols-1 gap-4 lg:grid-cols-3">
        <Panel title="Verdict mix (latest 100 analyses)">
          {historyError ? (
            <div className="text-sm text-red-300">History unavailable: {historyError}</div>
          ) : items === null ? (
            <div className="text-sm" style={{ color: "#7d93b2" }}>Loading…</div>
          ) : items.length === 0 ? (
            <div className="text-sm" style={{ color: "#7d93b2" }}>
              No analyses yet. Upload audio or start Live Detection.
            </div>
          ) : (
            <ul className="mono space-y-2 text-xs">
              {[...verdictMix.entries()].map(([label, count]) => (
                <li key={label} className="flex items-center gap-2">
                  {label === "legacy" ? (
                    <span style={{ color: "#7d93b2" }}>legacy (no verdict)</span>
                  ) : (
                    <VerdictBadge label={label} />
                  )}
                  <span className="flex-1" />
                  <span>{count}</span>
                </li>
              ))}
            </ul>
          )}
        </Panel>
        <Panel title="Average latency (measured)">
          <div className="mono text-3xl font-semibold" style={{ color: "#22d3ee" }}>
            {avgLatency === null ? "—" : `${avgLatency.toFixed(0)} ms`}
          </div>
          <div className="mono mt-1 text-xs" style={{ color: "#7d93b2" }}>
            {latencies.length === 0
              ? "Not available — measured per analysis from processing_time_ms."
              : `mean backend processing_time_ms over ${latencies.length} analyses (measured, not claimed).`}
          </div>
        </Panel>
        <Panel title="Model status">
          <ul className="mono space-y-2 text-xs">
            <li className="flex justify-between gap-4">
              <span style={{ color: "#7d93b2" }}>loaded</span>
              <span>{String(data.model_loaded)}</span>
            </li>
            <li className="flex justify-between gap-4">
              <span style={{ color: "#7d93b2" }}>version</span>
              <span>{data.model_version}</span>
            </li>
            <li className="flex justify-between gap-4">
              <span style={{ color: "#7d93b2" }}>mock mode</span>
              <span style={{ color: data.mock_mode ? "#fbbf24" : undefined }}>
                {String(data.mock_mode)}
              </span>
            </li>
          </ul>
          <Link to="/models" className="mono mt-3 inline-block text-xs" style={{ color: "#22d3ee" }}>
            Model Intelligence →
          </Link>
        </Panel>
      </div>

      <div className="mt-4 flex justify-center">
        <Gauge
          value={data.total_analyses === 0 ? 0 : (data.high_risk_events / data.total_analyses) * 100}
          label="high-risk share"
          size={150}
        />
      </div>

      <div className="soc-panel mt-5 overflow-x-auto">
        <div className="mono px-4 pt-4 text-xs uppercase tracking-widest" style={{ color: "#7d93b2" }}>
          Recent analyses
        </div>
        {data.recent.length === 0 ? (
          <div className="p-6" style={{ color: "#7d93b2" }}>
            No analyses yet. Upload audio or start Live Detection.
          </div>
        ) : (
          <table className="soc-table w-full">
            <thead>
              <tr>
                <th>File</th>
                <th>Prediction</th>
                <th>Confidence</th>
                <th>Risk</th>
                <th>Time</th>
              </tr>
            </thead>
            <tbody>
              {data.recent.map((r) => (
                <tr key={r.id}>
                  <td className="mono">
                    <Link to={`/history/${r.id}`} style={{ color: "#22d3ee" }}>
                      {r.audio_name}
                    </Link>
                  </td>
                  <td className={`pred-${r.prediction} font-semibold`}>{r.prediction}</td>
                  <td className="mono">{(r.confidence * 100).toFixed(1)}%</td>
                  <td className="mono">
                    {r.risk_score} / {r.risk_level}
                  </td>
                  <td className="mono text-xs" style={{ color: "#7d93b2" }}>
                    {new Date(r.created_at).toLocaleString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
