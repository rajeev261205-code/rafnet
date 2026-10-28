// Verdict + Detection Stack + scam-prevention alert components.
// Everything rendered comes from backend layers/verdict payloads; heuristic
// layers are always tagged "heuristic" and confidence is shown separately
// from the security risk score.
import type { LayerResult, RiskLevel, Verdict, VerdictLabel } from "../types";
import { Badge, Panel, ProgressBar } from "./ui";

const LABEL_TONE: Record<VerdictLabel, "ok" | "warn" | "bad" | "info" | "muted"> = {
  HUMAN_LIVE: "ok",
  AI_SYNTHETIC: "bad",
  REPLAY_PRERECORDED: "warn",
  EXTERNAL_APP_SUSPECTED: "warn",
  UNCERTAIN: "warn",
};

const LABEL_TEXT: Record<VerdictLabel, string> = {
  HUMAN_LIVE: "Human · live",
  AI_SYNTHETIC: "AI synthetic",
  REPLAY_PRERECORDED: "Replay / pre-recorded",
  EXTERNAL_APP_SUSPECTED: "External app suspected",
  UNCERTAIN: "Uncertain",
};

export function VerdictBadge({ label }: { label: VerdictLabel }): React.JSX.Element {
  return <Badge tone={LABEL_TONE[label] ?? "muted"}>{LABEL_TEXT[label] ?? label}</Badge>;
}

export function riskTone(level: RiskLevel | string): "ok" | "warn" | "bad" | "info" {
  if (level === "CRITICAL" || level === "HIGH") return "bad";
  if (level === "MEDIUM") return "warn";
  return "ok";
}

export function VerdictCard({
  verdict,
  title = "Voice authenticity assessment",
}: {
  verdict: Verdict | null | undefined;
  title?: string;
}): React.JSX.Element {
  if (!verdict) {
    return (
      <Panel title={title}>
        <div className="text-sm" style={{ color: "#7d93b2" }}>
          Not available — no verdict produced yet.
        </div>
      </Panel>
    );
  }
  return (
    <Panel
      title={title}
      right={<VerdictBadge label={verdict.label} />}
    >
      <div className="mono flex flex-wrap gap-x-6 gap-y-1 text-sm">
        <span>
          model confidence: <strong>{(verdict.confidence * 100).toFixed(1)}%</strong>
        </span>
        <span>
          risk: <strong>{verdict.risk_score}</strong> / {verdict.risk_level}
        </span>
      </div>
      {verdict.reasons.length > 0 && (
        <div className="mt-3">
          <div className="raf-panel-title mb-1">Reasons</div>
          <ul className="list-disc space-y-1 pl-5 text-xs">
            {verdict.reasons.map((r, i) => (
              <li key={i}>{r}</li>
            ))}
          </ul>
        </div>
      )}
      {verdict.limitations.length > 0 && (
        <div className="mt-3">
          <div className="raf-panel-title mb-1">Limitations</div>
          <ul className="list-disc space-y-1 pl-5 text-xs" style={{ color: "#7d93b2" }}>
            {verdict.limitations.map((r, i) => (
              <li key={i}>{r}</li>
            ))}
          </ul>
        </div>
      )}
    </Panel>
  );
}

const KIND_TONE: Record<string, "info" | "warn" | "muted"> = {
  model: "info",
  heuristic: "warn",
  unavailable: "muted",
};

function layerBarColor(score: number | null): string {
  if (score === null) return "#7d93b2";
  if (score >= 65) return "#f87171";
  if (score >= 35) return "#fbbf24";
  return "#34d399";
}

export function DetectionStack({ layers }: { layers: LayerResult[] | undefined }): React.JSX.Element {
  if (!layers || layers.length === 0) {
    return (
      <Panel title="Detection Stack">
        <div className="text-sm" style={{ color: "#7d93b2" }}>
          Not available — no layer data produced yet.
        </div>
      </Panel>
    );
  }
  return (
    <Panel title="Detection Stack" right={<span className="mono text-xs" style={{ color: "#7d93b2" }}>{layers.length} layers</span>}>
      <div className="grid grid-cols-1 gap-3 xl:grid-cols-2">
        {layers.map((l) => (
          <div key={l.id} className="rounded-lg border p-4" style={{ borderColor: "#1b2a44" }}>
            <div className="mb-1 flex flex-wrap items-center gap-2">
              <span className="text-sm font-semibold">{l.name}</span>
              <Badge tone={KIND_TONE[l.kind] ?? "muted"}>{l.kind}</Badge>
              <span className="flex-1" />
              <span className="mono text-sm">
                {l.score === null ? "Not available" : `${l.score.toFixed(0)}`}
              </span>
            </div>
            <ProgressBar value={l.score ?? 0} color={layerBarColor(l.score)} />
            {l.evidence.length > 0 && (
              <ul className="mt-2 list-disc space-y-1 pl-5 text-xs" style={{ color: "#7d93b2" }}>
                {l.evidence.map((e, i) => (
                  <li key={i}>{e}</li>
                ))}
              </ul>
            )}
          </div>
        ))}
      </div>
    </Panel>
  );
}

/** Scam-prevention banner + recommended actions (advice, not fake controls). */
export function ScamAlert({ riskLevel, verdict }: { riskLevel: string; verdict?: Verdict | null }): React.JSX.Element {
  const critical = riskLevel === "CRITICAL";
  return (
    <div className={`raf-alert ${critical ? "raf-alert-threat" : "raf-alert-warn"}`} role="alert">
      <div className="flex-1">
        <div className="mono text-sm font-bold uppercase tracking-widest" style={{ color: critical ? "#f87171" : "#fbbf24" }}>
          ⚠ {critical ? "Critical voice risk — possible impersonation scam" : "Elevated voice risk — verify before acting"}
          {verdict ? ` · ${LABEL_TEXT[verdict.label] ?? verdict.label}` : ""}
        </div>
        <ul className="mt-2 list-disc space-y-1 pl-5 text-xs">
          <li>Hang up if the caller pressures you to act immediately.</li>
          <li>Verify identity out-of-band (call back on a known number).</li>
          <li>Do not transfer funds or share codes on this call.</li>
          <li>Use the liveness challenge below: ask the speaker to say the random phrase, then analyze the response.</li>
        </ul>
      </div>
    </div>
  );
}
