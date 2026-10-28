import { useEffect, useState } from "react";
import { api } from "../services/api";
import type { AnalyzeResponse, ModelInfo } from "../types";
import { Panel } from "../components/ui";

interface EvalReport {
  generated_at?: string;
  files_total?: number;
  per_layer_accuracy?: Record<string, Record<string, unknown>>;
  confusion_matrix?: Record<string, Record<string, number>>;
}

function EvaluationPanel(): React.JSX.Element | null {
  const [report, setReport] = useState<EvalReport | null | undefined>(undefined);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .evaluation()
      .then((r) => setReport(r.evaluated ? (r.report as EvalReport) : null))
      .catch((e: Error) => setError(e.message));
  }, []);

  if (error)
    return (
      <Panel title="Measured layer evaluation">
        <div className="text-sm text-red-300">Evaluation unavailable: {error}</div>
      </Panel>
    );
  if (report === undefined)
    return (
      <Panel title="Measured layer evaluation">
        <div className="text-sm" style={{ color: "#7d93b2" }}>Loading…</div>
      </Panel>
    );
  if (report === null) return null;

  const filesTotal = typeof report.files_total === "number" ? report.files_total : 0;
  if (!Number.isFinite(filesTotal) || filesTotal <= 0) return null;

  const acc = report.per_layer_accuracy ?? {};
  const measured = Object.entries(acc).filter(
    ([, a]) => typeof (a as { accuracy?: unknown }).accuracy === "number",
  );
  const matrix = report.confusion_matrix ?? {};
  let matrixSum = 0;
  for (const row of Object.values(matrix)) {
    for (const v of Object.values(row)) {
      if (typeof v === "number" && Number.isFinite(v)) matrixSum += v;
    }
  }
  const firstRow = Object.values(matrix)[0] as Record<string, number> | undefined;
  const predLabels = firstRow ? Object.keys(firstRow) : [];
  const showMatrix = predLabels.length > 0 && matrixSum > 0;
  if (measured.length === 0 && !showMatrix) return null;

  return (
    <Panel
      title="Measured layer evaluation"
      right={
        <span className="mono text-xs" style={{ color: "#7d93b2" }}>
          {filesTotal} files · {String(report.generated_at ?? "")}
        </span>
      }
    >
      {measured.length > 0 && (
        <div>
          <div className="raf-panel-title mb-2">Per-layer accuracy (measured)</div>
          <ul className="mono space-y-1 text-xs">
            {measured.map(([layer, a]) => {
              const typed = a as { accuracy: number; correct?: unknown; evaluated?: unknown };
              return (
                <li key={layer} className="flex justify-between gap-4">
                  <span style={{ color: "#7d93b2" }}>{layer}</span>
                  <span>
                    {`${(typed.accuracy * 100).toFixed(1)}% (${String(typed.correct)}/${String(typed.evaluated)})`}
                  </span>
                </li>
              );
            })}
          </ul>
        </div>
      )}
      {showMatrix && (
        <div className="mt-4 overflow-x-auto">
          <div className="raf-panel-title mb-2">Verdict confusion matrix (rows = true class, measured)</div>
          <table className="raf-table w-full">
            <thead>
              <tr>
                <th>true \ predicted</th>
                {predLabels.map((l) => (
                  <th key={l}>{l}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {Object.entries(matrix).map(([trueCls, row]) => (
                <tr key={trueCls}>
                  <td className="mono">{trueCls}</td>
                  {predLabels.map((l) => (
                    <td key={l} className="mono">{row[l] ?? 0}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Panel>
  );
}

interface OriginReport {
  generated_at?: string;
  files_scored?: number;
  files_total?: number;
  metrics?: Record<string, Record<string, unknown>>;
  honest_notes?: string[];
  // Phase-5 ensemble schema (reports/model_eval.json): TEST + my_samples.
  test?: Record<string, unknown>;
  my_samples?: Record<string, unknown>;
  per_group?: Record<string, Record<string, unknown>>;
  missing_groups?: string[];
}

function MetricRow({ label, value }: { label: string; value: unknown }): React.JSX.Element {
  return (
    <li className="flex justify-between gap-4">
      <span style={{ color: "#7d93b2" }}>{label}</span>
      <span>{typeof value === "number" ? value : String(value ?? "—")}</span>
    </li>
  );
}

function EnsembleEvalBody({ report }: { report: OriginReport }): React.JSX.Element {
  const test = (report.test ?? {}) as Record<string, unknown>;
  const my = (report.my_samples ?? {}) as Record<string, unknown>;
  const groups = (report.per_group ?? {}) as Record<string, Record<string, unknown>>;
  const groupNames = Object.keys(groups);
  return (
    <div>
      <div className="raf-panel-title mb-2">Stacked ensemble (measured, frozen VAL-fit)</div>
      <ul className="mono space-y-1 text-xs">
        <MetricRow label="TEST accuracy (unseen speakers/setting)" value={test.accuracy as number} />
        <MetricRow label="TEST AUC" value={test.auc as number} />
        <MetricRow label="TEST false-alarm rate on human" value={test.false_alarm_rate_on_human as number} />
        <MetricRow label="my_samples accuracy (unseen generators)" value={my.accuracy as number} />
        <MetricRow label="my_samples false-alarm rate on human" value={my.false_alarm_rate_on_human as number} />
      </ul>
      {groupNames.length > 0 && (
        <div className="mt-3">
          <div className="raf-panel-title mb-2">Per-group accuracy (measured)</div>
          <ul className="mono space-y-1 text-xs">
            {groupNames.map((g) => (
              <li key={g} className="flex justify-between gap-4">
                <span style={{ color: "#7d93b2" }}>{g}</span>
                <span>{`n=${String(groups[g].n ?? "?")} acc=${String(groups[g].accuracy ?? "—")}`}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
      {Array.isArray(report.missing_groups) && report.missing_groups.length > 0 && (
        <div className="mono mt-3 text-xs" style={{ color: "#7d93b2" }}>
          unmeasured groups: {(report.missing_groups as string[]).join(", ")}
        </div>
      )}
    </div>
  );
}

function OriginEvalPanel(): React.JSX.Element | null {
  const [report, setReport] = useState<OriginReport | null | undefined>(undefined);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .originEvaluation()
      .then((r) => setReport(r.evaluated ? (r.report as OriginReport) : null))
      .catch((e: Error) => setError(e.message));
  }, []);

  if (error)
    return (
      <Panel title="Voice-origin evaluation">
        <div className="text-sm text-red-300">Evaluation unavailable: {error}</div>
      </Panel>
    );
  if (report === undefined)
    return (
      <Panel title="Voice-origin evaluation">
        <div className="text-sm" style={{ color: "#7d93b2" }}>Loading…</div>
      </Panel>
    );
  if (report === null) return null;

  const filesScored = Number(report.files_scored ?? report.files_total ?? 0);
  const hasEnsembleSchema =
    typeof (report.test as Record<string, unknown> | undefined)?.n === "number" ||
    typeof (report.my_samples as Record<string, unknown> | undefined)?.n === "number";
  if (!hasEnsembleSchema && (!Number.isFinite(filesScored) || filesScored <= 0)) return null;

  const metrics = (report.metrics ?? {}) as Record<string, Record<string, unknown>>;
  const clean = (metrics.clean ?? {}) as Record<string, unknown>;
  return (
    <Panel
      title="Voice-origin evaluation"
      right={
        <span className="mono text-xs" style={{ color: "#7d93b2" }}>
          {hasEnsembleSchema
            ? `TEST+my · ${String(report.generated_at ?? "")}`
            : `${filesScored} files · ${String(report.generated_at ?? "")}`}
        </span>
      }
    >
      {hasEnsembleSchema ? (
        <EnsembleEvalBody report={report} />
      ) : (
      <>
      <div className="raf-panel-title mb-2">Clean condition (measured)</div>
      <ul className="mono space-y-1 text-xs">
        <li className="flex justify-between gap-4">
          <span style={{ color: "#7d93b2" }}>strict accuracy</span>
          <span>{String(clean.strict_accuracy_uncertain_is_wrong ?? "—")}</span>
        </li>
        <li className="flex justify-between gap-4">
          <span style={{ color: "#7d93b2" }}>AUC</span>
          <span>{String(clean.auc ?? "—")}</span>
        </li>
        <li className="flex justify-between gap-4">
          <span style={{ color: "#7d93b2" }}>EER</span>
          <span>{JSON.stringify(clean.eer ?? "—")}</span>
        </li>
      </ul>
      {Array.isArray(report.honest_notes) && report.honest_notes.length > 0 && (
        <ul className="mt-3 list-disc space-y-1 pl-5 text-xs" style={{ color: "#7d93b2" }}>
          {(report.honest_notes as string[]).slice(0, 4).map((n, i) => (
            <li key={i}>{n}</li>
          ))}
        </ul>
      )}
      </>
      )}
    </Panel>
  );
}

const STAGE_LABELS: Record<string, string> = {
  decode_ms: "decode + validate",
  inference_ms: "model inference (total)",
  ensemble_hf_ms: "ensemble · HF anti-spoof (2 models)",
  forensics_spectral_ms: "forensics · spectral",
  forensics_liveness_ms: "forensics · liveness",
  risk_store_ms: "risk + store",
};

const MODEL_STAGE_LABELS: Record<string, string> = {
  audio_encoder_ms: "model · audio encoder (WavLM + handcrafted)",
  network_encoder_ms: "model · network encoder",
  gateway1_ms: "model · gateway 1",
  classical_branch_ms: "model · classical branch",
  quantum_branch_ms: "model · quantum branch",
  gateway2_ms: "model · gateway 2",
  calibration_ms: "model · calibration",
  decision_gate_ms: "model · decision gate",
};

function extractLatencyRows(analysis: AnalyzeResponse): Array<[string, number]> {
  const stages = (analysis.stage_timings_ms ?? {}) as Record<string, unknown>;
  const modelStages = (stages["model_stages_ms"] ?? {}) as Record<string, unknown>;
  const rows: Array<[string, number]> = [];
  for (const [key, label] of Object.entries(STAGE_LABELS)) {
    const v = stages[key];
    if (typeof v === "number" && Number.isFinite(v)) rows.push([label, v]);
  }
  for (const [key, label] of Object.entries(MODEL_STAGE_LABELS)) {
    const v = modelStages[key];
    if (typeof v === "number" && Number.isFinite(v)) rows.push([label, v]);
  }
  return rows;
}

function LatencyPanel(): React.JSX.Element | null {
  const [latest, setLatest] = useState<AnalyzeResponse | null | undefined>(undefined);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .history(1, 0)
      .then((h) => setLatest(h.items[0] ?? null))
      .catch((e: Error) => setError(e.message));
  }, []);

  if (error)
    return (
      <Panel title="Measured analysis latency (latest analysis)">
        <div className="text-sm text-red-300">Latency unavailable: {error}</div>
      </Panel>
    );
  if (latest === undefined)
    return (
      <Panel title="Measured analysis latency (latest analysis)">
        <div className="text-sm" style={{ color: "#7d93b2" }}>Loading…</div>
      </Panel>
    );
  if (latest === null) return null;

  const rows = extractLatencyRows(latest);
  if (rows.length === 0) return null;

  const max = Math.max(...rows.map(([, v]) => v), 1);
  return (
    <Panel title="Measured analysis latency (latest analysis)">
      <div>
        <div className="mono mb-2 text-xs" style={{ color: "#7d93b2" }}>
          {latest.audio_name} · {latest.audio_duration_seconds.toFixed(2)} s audio · pipeline total{" "}
          {latest.processing_time_ms.toFixed(0)} ms (measured, CPU)
        </div>
        <ul className="mono space-y-1 text-xs">
          {rows.map(([label, v]) => (
            <li key={label} className="flex items-center gap-3">
              <span className="w-64 shrink-0 truncate" style={{ color: "#7d93b2" }}>{label}</span>
              <span className="inline-block h-2 rounded" style={{ width: `${Math.max(2, (v / max) * 160)}px`, background: "#22d3ee" }} />
              <span>{v >= 100 ? v.toFixed(0) : v.toFixed(1)} ms</span>
            </li>
          ))}
        </ul>
      </div>
    </Panel>
  );
}

export default function Models(): React.JSX.Element {
  const [info, setInfo] = useState<ModelInfo | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.models().then(setInfo).catch((e: Error) => setError(e.message));
  }, []);

  if (error) return <div className="soc-panel p-6 text-red-300">Model info unavailable: {error}</div>;
  if (!info) return <div className="soc-panel p-6" style={{ color: "#7d93b2" }}>Loading model info…</div>;

  const checkpointEntries = Object.entries(info.checkpoints).filter(([k]) => k !== "note");
  const hasCheckpoints = checkpointEntries.length > 0;

  return (
    <div>
      <div className="mb-5 flex items-center gap-3">
        <h1 className="text-2xl font-bold">Model Intelligence</h1>
        {info.mock_mode && (
          <span className="soc-chip mono" style={{ color: "#fbbf24" }}>
            MOCK MODE — not a real prediction
          </span>
        )}
      </div>
      <div className="flex flex-col gap-4">
        <EvaluationPanel />
        <OriginEvalPanel />
        <LatencyPanel />
      </div>
      <div className="soc-panel mt-4 p-6">
        <div className="mono mb-3 text-xs uppercase tracking-widest" style={{ color: "#7d93b2" }}>
          {info.name} · {info.version}
        </div>
        <div className="text-sm">{info.architecture}</div>
        <div className="mono mt-2 text-xs" style={{ color: "#7d93b2" }}>
          framework {info.framework} · sample rate {info.sample_rate} Hz
        </div>
        <div className="mono mt-2 text-xs" style={{ color: "#7d93b2" }}>
          mock mode {String(info.mock_mode)}
        </div>
        {hasCheckpoints && (
          <div>
            <div className="mono mt-4 text-xs uppercase tracking-widest" style={{ color: "#7d93b2" }}>
              Checkpoints (bytes on disk)
            </div>
            <ul className="mono mt-1 text-xs">
              {checkpointEntries.map(([k, v]) => (
                <li key={k} className="flex justify-between gap-4">
                  <span>{k}</span>
                  <span>{String(v)}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
        <div className="mono mt-4 text-xs uppercase tracking-widest" style={{ color: "#7d93b2" }}>
          Decision thresholds
        </div>
        <ul className="mono mt-1 text-xs">
          {Object.entries(info.thresholds).map(([k, v]) => (
            <li key={k} className="flex justify-between gap-4">
              <span>{k}</span>
              <span>{String(v)}</span>
            </li>
          ))}
        </ul>
        <div className="mt-4 text-xs" style={{ color: "#7d93b2" }}>
          {info.limitations}
        </div>
      </div>
    </div>
  );
}
