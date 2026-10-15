// Fusion & Gateways: the real classical / quantum / fused / calibrated
// probabilities and gateway weights from the most recent stored analysis.
// Every number comes from GET /api/history; with no analyses yet the page
// says so instead of showing placeholder values.
import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../services/api";
import type { AnalyzeResponse } from "../types";
import { Badge, EmptyState, Panel, ProgressBar, Skeleton } from "../components/ui";
import { VerdictCard } from "../components/Verdict";

function WeightRow({
  label,
  value,
}: {
  label: string;
  value: number | null;
}): React.JSX.Element {
  return (
    <div className="py-2">
      <div className="mono mb-1 flex justify-between text-xs">
        <span style={{ color: "#7d93b2" }}>{label}</span>
        <span>{typeof value === "number" ? value.toFixed(4) : "Not available"}</span>
      </div>
      <ProgressBar value={typeof value === "number" ? value * 100 : 0} />
    </div>
  );
}

export default function Fusion(): React.JSX.Element {
  const [latest, setLatest] = useState<AnalyzeResponse | null | undefined>(undefined);
  const [error, setError] = useState<string | null>(null);

  const load = (): void => {
    api
      .history(1, 0)
      .then((h) => setLatest(h.items[0] ?? null))
      .catch((e: Error) => setError(e.message));
  };

  useEffect(load, []);

  if (error)
    return (
      <Panel title="Fusion & Gateways">
        <div className="text-sm text-red-300">Fusion data unavailable: {error}</div>
      </Panel>
    );
  if (latest === undefined)
    return (
      <Panel title="Fusion & Gateways">
        <Skeleton lines={5} />
      </Panel>
    );
  if (latest === null)
    return (
      <Panel title="Fusion & Gateways">
        <EmptyState
          title="Not available — no analyses yet"
          detail="Gateway weights and branch probabilities are produced per analysis. Run one on the Analyze page and they will appear here."
          action={
            <Link to="/analyze" className="raf-btn">
              Go to Analyze
            </Link>
          }
        />
      </Panel>
    );

  return (
    <div>
      <div className="mb-5 flex flex-wrap items-center gap-3">
        <h1 className="text-2xl font-bold">Fusion &amp; Gateways</h1>
        <span className="mono text-xs" style={{ color: "#7d93b2" }}>
          from analysis {latest.id.slice(0, 8)} · {latest.audio_name} · {latest.model_version}
        </span>
        <button className="raf-btn raf-btn-ghost" style={{ padding: "0.3rem 0.8rem" }} onClick={load}>
          Refresh
        </button>
      </div>
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <VerdictCard verdict={latest.verdict} title="Fused verdict (latest analysis)" />
        <Panel
          title="Detection branches (model)"
          right={<Badge tone="info">model</Badge>}
        >
          <WeightRow label="classical probability (p_c)" value={latest.classical_probability} />
          <WeightRow label="quantum probability (p_q)" value={latest.quantum_probability} />
          <WeightRow label="fused probability (p_fused)" value={latest.fused_probability} />
          <WeightRow label="calibrated probability" value={latest.calibrated_probability} />
          <div className="mono mt-3 text-xs" style={{ color: "#7d93b2" }}>
            decision {latest.prediction} · confidence {(latest.confidence * 100).toFixed(2)}% ·
            synthetic {latest.prob_synthetic.toFixed(4)} · real {latest.prob_real.toFixed(4)}
          </div>
        </Panel>
        <Panel
          title="Gateway weights (model)"
          right={<Badge tone="info">model</Badge>}
        >
          <WeightRow label="gateway1 audio weight (α)" value={latest.gateway1_audio_weight} />
          <WeightRow label="gateway1 network weight (1−α)" value={latest.gateway1_network_weight} />
          <WeightRow label="gateway2 classical weight (β)" value={latest.gateway2_classical_weight} />
          <WeightRow label="gateway2 quantum weight (1−β)" value={latest.gateway2_quantum_weight} />
          <div className="mono mt-3 text-xs" style={{ color: "#7d93b2" }}>
            audio-only inference pins Gateway1 to fixed-audio; network weights are reported
            as measured by the pipeline.
          </div>
        </Panel>
        <Panel
          title="Ensemble anti-spoofing (measured)"
          right={<Badge tone="info">ensemble</Badge>}
        >
          <WeightRow label="wav2vec2 deepfake P(AI) · mo-thecreator" value={latest.p_antispoof_mo ?? null} />
          <WeightRow label="wav2vec2 deepfake P(AI) · Bisher" value={latest.p_antispoof_bisher ?? null} />
          <WeightRow label="stacked calibrated P(AI)" value={latest.p_ensemble ?? null} />
          <WeightRow label="model agreement (fake votes / 3)" value={latest.ensemble_agreement ?? null} />
          <div className="mono mt-3 text-xs" style={{ color: "#7d93b2" }}>
            verdict {latest.ensemble_verdict ?? "Not available"} · alert {latest.ensemble_alert ?? "Not available"}
            {typeof latest.ensemble_votes_fake === "number" ? ` · votes ${latest.ensemble_votes_fake}/3` : ""}
          </div>
          {(latest.ensemble_reasons ?? []).length > 0 && (
            <ul className="mono mt-2 list-disc space-y-1 pl-5 text-xs" style={{ color: "#7d93b2" }}>
              {(latest.ensemble_reasons ?? []).map((r, i) => (
                <li key={i}>{r}</li>
              ))}
            </ul>
          )}
        </Panel>
      </div>
    </div>
  );
}
