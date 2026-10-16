// Voice-origin result card (5-class origin + open-set generator match).
// Every state is explicit: collecting (not enough audio), unavailable
// (checkpoint not installed), ready (measured probabilities). No fake data.
import type { OriginResult } from "../types";
import { Panel } from "./ui";

const LABEL_TEXT: Record<string, string> = {
  HUMAN_LIVE: "Human live",
  HUMAN_RECORDED_REPLAY: "Human recorded / replay",
  AI_TTS: "AI text-to-speech",
  AI_VOICE_CLONE_OR_CONVERSION: "AI voice clone / conversion",
  UNCERTAIN: "Uncertain",
};

const LABEL_COLOR: Record<string, string> = {
  HUMAN_LIVE: "#34d399",
  HUMAN_RECORDED_REPLAY: "#fbbf24",
  AI_TTS: "#f87171",
  AI_VOICE_CLONE_OR_CONVERSION: "#f87171",
  UNCERTAIN: "#7d93b2",
};

export default function OriginCard({
  origin,
  status,
  windowSeconds,
  sessionOrigin,
  title = "Voice origin (latest window)",
}: {
  origin: OriginResult | null | undefined;
  status: string;
  windowSeconds: number;
  sessionOrigin: string | null | undefined;
  title?: string;
}): React.JSX.Element {
  if (status === "collecting") {
    return (
      <Panel title={title}>
        <div className="text-sm" style={{ color: "#7d93b2" }}>
          Collecting audio… origin analysis needs a rolling 4 s window
          ({windowSeconds.toFixed(1)} s so far). No guess is shown meanwhile.
        </div>
      </Panel>
    );
  }
  if (!origin) {
    return (
      <Panel title={title}>
        <div className="text-sm" style={{ color: "#7d93b2" }}>
          Origin model not installed
          {status === "disabled" ? " (disabled)" : ""} — train one per{" "}
          <span className="mono">reports/train_plan.md</span> and place{" "}
          <span className="mono">models/origin_classifier.pt</span>.
        </div>
      </Panel>
    );
  }
  const entries = Object.entries(origin.probabilities).sort((a, b) => b[1] - a[1]);
  const peak = Math.max(0.001, ...entries.map(([, p]) => p));
  return (
    <Panel
      title={title}
      right={
        <span className="mono text-xs" style={{ color: "#7d93b2" }}>
          {windowSeconds.toFixed(1)} s of audio · {origin.model_version}
        </span>
      }
    >
      <div className="mono text-sm">
        <strong style={{ color: LABEL_COLOR[origin.label] ?? "#dbe7f5" }}>
          {LABEL_TEXT[origin.label] ?? origin.label}
        </strong>
        <span style={{ color: "#7d93b2" }}> · calibrated {(origin.confidence * 100).toFixed(1)}%</span>
      </div>
      <div className="mt-3 space-y-1">
        {entries.map(([label, p]) => (
          <div key={label} className="flex items-center gap-2">
            <span className="mono w-56 truncate text-xs" style={{ color: "#7d93b2" }}>
              {LABEL_TEXT[label] ?? label}
            </span>
            <div className="h-2 flex-1 rounded bg-black/40">
              <div
                className="h-2 rounded"
                style={{ width: `${(p / peak) * 100}%`, background: LABEL_COLOR[label] ?? "#22d3ee" }}
              />
            </div>
            <span className="mono w-14 text-right text-xs">{(p * 100).toFixed(1)}%</span>
          </div>
        ))}
      </div>
      <div className="mono mt-3 text-xs" style={{ color: "#7d93b2" }}>
        {origin.ai_source ? (
          <span>
            AI source: closest match{" "}
            <strong style={{ color: "#dbe7f5" }}>{origin.ai_source.family}</strong> (
            {(origin.ai_source.confidence * 100).toFixed(1)}%) — match, never proof.
          </span>
        ) : (
          <span>AI source: unknown generator (no reliable source evidence).</span>
        )}
      </div>
      {sessionOrigin && (
        <div className="mono mt-1 text-xs" style={{ color: "#7d93b2" }}>
          Session origin (majority of recent windows):{" "}
          <strong style={{ color: LABEL_COLOR[sessionOrigin] ?? "#dbe7f5" }}>
            {LABEL_TEXT[sessionOrigin] ?? sessionOrigin}
          </strong>
        </div>
      )}
      {origin.limitations.length > 0 && (
        <div className="mono mt-2 text-xs" style={{ color: "#7d93b2" }}>
          Limits: {origin.limitations[0]}
        </div>
      )}
    </Panel>
  );
}
