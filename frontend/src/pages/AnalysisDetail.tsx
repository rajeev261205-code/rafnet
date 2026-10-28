import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api } from "../services/api";
import type { AnalyzeResponse } from "../types";
import { Panel } from "../components/ui";
import { VerdictBadge } from "../components/Verdict";

function fmt(v: number | null | undefined, digits = 4): string {
  return typeof v === "number" && Number.isFinite(v) ? v.toFixed(digits) : "—";
}

export default function AnalysisDetail(): React.JSX.Element {
  const { id } = useParams<{ id: string }>();
  const [data, setData] = useState<AnalyzeResponse | null>(null);
  const [events, setEvents] = useState<unknown[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!id) {
      setError("Missing analysis id");
      return;
    }
    api.analysis(id).then(setData).catch((e: Error) => setError(e.message));
    api.analysisEvents(id).then((r) => setEvents(r.events)).catch(() => setEvents([]));
  }, [id]);

  if (error)
    return (
      <div className="soc-panel p-6">
        <div className="text-red-300">Analysis unavailable: {error}</div>
        <Link to="/history" className="mono mt-3 inline-block text-xs" style={{ color: "#22d3ee" }}>
          ← Back to history
        </Link>
      </div>
    );
  if (!data)
    return <div className="soc-panel p-6" style={{ color: "#7d93b2" }}>Loading analysis…</div>;

  return (
    <div>
      <Link to="/history" className="mono text-xs" style={{ color: "#22d3ee" }}>
        ← Back to history
      </Link>
      <div className="mb-5 mt-2 flex flex-wrap items-center gap-3">
        <h1 className="mono text-xl font-bold">{data.audio_name}</h1>
        <span className={`pred-${data.prediction} font-semibold`}>{data.prediction}</span>
        {data.verdict ? <VerdictBadge label={data.verdict.label} /> : null}
        {data.mock ? (
          <span className="soc-chip mono" style={{ color: "#fbbf24" }}>
            MOCK — not a real prediction
          </span>
        ) : null}
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <Panel title="Model result">
          <ul className="mono space-y-2 text-xs">
            <li className="flex justify-between gap-4"><span style={{ color: "#7d93b2" }}>confidence</span><span>{(data.confidence * 100).toFixed(1)}%</span></li>
            <li className="flex justify-between gap-4"><span style={{ color: "#7d93b2" }}>risk</span><span>{data.risk_score} / {data.risk_level}</span></li>
            <li className="flex justify-between gap-4"><span style={{ color: "#7d93b2" }}>P(synthetic)</span><span>{fmt(data.prob_synthetic)}</span></li>
            <li className="flex justify-between gap-4"><span style={{ color: "#7d93b2" }}>P(real)</span><span>{fmt(data.prob_real)}</span></li>
            <li className="flex justify-between gap-4"><span style={{ color: "#7d93b2" }}>classical</span><span>{fmt(data.classical_probability)}</span></li>
            <li className="flex justify-between gap-4"><span style={{ color: "#7d93b2" }}>quantum</span><span>{fmt(data.quantum_probability)}</span></li>
            <li className="flex justify-between gap-4"><span style={{ color: "#7d93b2" }}>fused</span><span>{fmt(data.fused_probability)}</span></li>
            <li className="flex justify-between gap-4"><span style={{ color: "#7d93b2" }}>calibrated</span><span>{fmt(data.calibrated_probability)}</span></li>
            <li className="flex justify-between gap-4"><span style={{ color: "#7d93b2" }}>gateway1 audio/network</span><span>{fmt(data.gateway1_audio_weight, 3)} / {fmt(data.gateway1_network_weight, 3)}</span></li>
            <li className="flex justify-between gap-4"><span style={{ color: "#7d93b2" }}>gateway2 classical/quantum</span><span>{fmt(data.gateway2_classical_weight, 3)} / {fmt(data.gateway2_quantum_weight, 3)}</span></li>
            <li className="flex justify-between gap-4"><span style={{ color: "#7d93b2" }}>model</span><span>{data.model_version}</span></li>
            <li className="flex justify-between gap-4"><span style={{ color: "#7d93b2" }}>latency</span><span>{data.processing_time_ms.toFixed(1)} ms</span></li>
            <li className="flex justify-between gap-4"><span style={{ color: "#7d93b2" }}>input</span><span>{data.input_lossy ? `lossy (${data.input_codec ?? "unknown codec"})` : `${data.input_codec ?? "unknown"} (lossless/PCM)`}</span></li>
            <li className="flex justify-between gap-4"><span style={{ color: "#7d93b2" }}>duration</span><span>{data.audio_duration_seconds.toFixed(2)} s</span></li>
            <li className="flex justify-between gap-4"><span style={{ color: "#7d93b2" }}>analyzed</span><span>{new Date(data.created_at).toLocaleString()}</span></li>
          </ul>
          {data.reasons.length > 0 && (
            <div className="mono mt-3 text-xs" style={{ color: "#7d93b2" }}>
              {data.reasons.map((r) => <div key={r.slice(0, 48)}>· {r}</div>)}
            </div>
          )}
        </Panel>

        <Panel title="Fused verdict">
          {data.verdict ? (
            <div>
              <div className="flex flex-wrap items-center gap-3">
                <VerdictBadge label={data.verdict.label} />
                <span className="mono text-xs">confidence {(data.verdict.confidence * 100).toFixed(1)}%</span>
              </div>
              <div className="mono mt-3 space-y-1 text-xs" style={{ color: "#7d93b2" }}>
                {data.verdict.reasons.map((r) => <div key={r.slice(0, 48)}>· {r}</div>)}
              </div>
              {data.verdict.limitations.length > 0 && (
                <div className="mono mt-3 space-y-1 text-xs" style={{ color: "#fbbf24" }}>
                  {data.verdict.limitations.map((r) => <div key={r.slice(0, 48)}>⚠ {r}</div>)}
                </div>
              )}
            </div>
          ) : (
            <div className="text-sm" style={{ color: "#7d93b2" }}>
              Not available — this record predates verdict fusion.
            </div>
          )}
        </Panel>
      </div>

      <Panel title="Detection layers" >
        <div className="mt-4 space-y-3">
          {!data.layers || data.layers.length === 0 ? (
            <div className="text-sm" style={{ color: "#7d93b2" }}>
              Not available — this record predates layered detection.
            </div>
          ) : (
            data.layers.map((l) => (
              <div key={l.id} className="rounded-lg border p-3" style={{ borderColor: "#1b2a44" }}>
                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-sm font-semibold">{l.name}</span>
                  <span className="soc-chip mono">{l.kind}</span>
                  {l.score !== null && <span className="mono text-xs">score {l.score.toFixed(1)}</span>}
                </div>
                <ul className="mono mt-2 space-y-1 text-xs" style={{ color: "#7d93b2" }}>
                  {l.evidence.map((e) => <li key={e.slice(0, 64)}>· {e}</li>)}
                </ul>
              </div>
            ))
          )}
        </div>
      </Panel>

      {events.length > 0 && (
        <Panel title="Live events">
          <ul className="mono mt-4 space-y-1 text-xs" style={{ color: "#7d93b2" }}>
            {events.slice(0, 20).map((e, i) => <li key={i}>{JSON.stringify(e).slice(0, 200)}</li>)}
          </ul>
        </Panel>
      )}
    </div>
  );
}
