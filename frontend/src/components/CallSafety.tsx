// Call safety check: transcript-based scam-risk analysis (separate from voice
// authenticity — a synthetic voice does not imply scam, nor human imply safe).
// The transcript is typed/pasted by the user; nothing is recorded here.
import { useState } from "react";
import { api } from "../services/api";
import type { CallRiskResult } from "../types";
import { Panel } from "./ui";

const RISK_COLOR: Record<string, string> = {
  low: "#34d399",
  suspicious: "#fbbf24",
  high: "#f87171",
  insufficient: "#7d93b2",
};

const RISK_LABEL: Record<string, string> = {
  low: "Low observed risk",
  suspicious: "Suspicious behavior detected",
  high: "High-risk scam indicators",
  insufficient: "Insufficient information",
};

export default function CallSafety(): React.JSX.Element {
  const [transcript, setTranscript] = useState("");
  const [carrier, setCarrier] = useState("unknown");
  const [result, setResult] = useState<CallRiskResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [checking, setChecking] = useState(false);

  const analyze = async (): Promise<void> => {
    setChecking(true);
    setError(null);
    setResult(null);
    try {
      setResult(await api.callRisk(transcript, carrier));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Check failed");
    } finally {
      setChecking(false);
    }
  };

  return (
    <Panel title="Call safety check (transcript-based, not voice-based)">
      <p className="mb-3 text-sm" style={{ color: "#7d93b2" }}>
        Paste what the caller said (English or Hindi-English). Scam behavior is
        judged from the words and pressure tactics — never from how the voice
        sounds, and never from the carrier.
      </p>
      <textarea
        className="soc-input w-full"
        rows={4}
        placeholder="e.g. This is your bank, your account will be blocked today, share your OTP now…"
        value={transcript}
        onChange={(e) => setTranscript(e.target.value)}
        aria-label="Call transcript"
      />
      <div className="mt-3 flex flex-wrap items-center gap-3">
        <label className="mono text-xs" style={{ color: "#7d93b2" }}>
          Receiving network (context only, never scored):{" "}
          <select
            className="soc-select"
            value={carrier}
            onChange={(e) => setCarrier(e.target.value)}
            aria-label="Receiving network"
          >
            <option value="unknown">Unknown / other</option>
            <option value="jio">Jio</option>
            <option value="airtel">Airtel</option>
            <option value="other">Other</option>
          </select>
        </label>
        <button className="soc-btn" disabled={checking || transcript.trim() === ""} onClick={() => void analyze()}>
          {checking ? "Checking…" : "Check call risk"}
        </button>
      </div>
      {error && <div className="mt-3 text-sm text-red-300">{error}</div>}
      {result && (
        <div className="mt-4">
          <div className="mono text-sm">
            <strong style={{ color: RISK_COLOR[result.risk] ?? "#dbe7f5" }}>
              {RISK_LABEL[result.risk] ?? result.risk}
            </strong>
            <span style={{ color: "#7d93b2" }}> · signals {result.score}/{result.scale}</span>
          </div>
          <p className="mt-2 text-sm" style={{ color: "#dbe7f5" }}>{result.explanation}</p>
          {result.evidence.length > 0 && (
            <ul className="mono mt-2 space-y-1 text-xs">
              {result.evidence.map((e, i) => (
                <li key={i} className="rounded border-l-2 border-amber-800 bg-black/30 p-2">
                  <span style={{ color: "#fbbf24" }}>{e.label}</span>{" "}
                  <span style={{ color: "#7d93b2" }}>“{e.excerpt}”</span>
                </li>
              ))}
            </ul>
          )}
          {result.actions.length > 0 && (
            <div className="mt-3">
              <div className="raf-panel-title mb-1">Suggested actions</div>
              <ul className="list-disc space-y-1 pl-5 text-xs">
                {result.actions.map((a, i) => (
                  <li key={i}>{a}</li>
                ))}
              </ul>
            </div>
          )}
          <div className="mono mt-2 text-xs" style={{ color: "#7d93b2" }}>
            Low risk never means guaranteed safe. Carrier “{result.carrier}” was recorded, not scored.
          </div>
        </div>
      )}
    </Panel>
  );
}
