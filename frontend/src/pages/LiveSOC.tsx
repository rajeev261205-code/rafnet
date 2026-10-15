// Live Detection: microphone -> 16 kHz analyzer (waveform / spectrogram /
// pitch track tabs, all computed while capturing) plus 1 s backend chunks and
// the /ws/live event stream. Verdict, layers, risk and patterns all come from
// the backend chunk responses; the timeline below is timestamped evidence.
import { useEffect, useRef, useState } from "react";
import { useLiveSocket } from "../lib/useLiveSocket";
import { useLiveCapture } from "../lib/useLiveCapture";
import { LiveAnalyzerPanel } from "../components/AnalyzerViews";
import { DetectionStack, ScamAlert, VerdictBadge, VerdictCard } from "../components/Verdict";
import OriginCard from "../components/OriginCard";
import CallSafety from "../components/CallSafety";
import Challenge from "../components/Challenge";
import { Badge, Chip, Gauge, Panel } from "../components/ui";
import { api } from "../services/api";
import type { Capabilities } from "../types";

function WsBadge({ status }: { status: string }): React.JSX.Element {
  const color = status === "LIVE" ? "#34d399" : status === "RECONNECTING" ? "#fbbf24" : "#f87171";
  return (
    <span className="soc-chip mono" style={{ color }}>
      {status}
    </span>
  );
}

interface PatternEntry {
  key: string;
  chunk: number;
  layer: string;
  text: string;
}

export default function LiveSOC(): React.JSX.Element {
  const { status, events, mockMode } = useLiveSocket(true);
  const mic = useLiveCapture();
  const [patterns, setPatterns] = useState<PatternEntry[]>([]);
  const [caps, setCaps] = useState<Capabilities | null>(null);
  const flatRef = useRef(false);
  const latest = mic.results.find((r) => r.status === "analyzed");
  const sessionVerdict = latest?.session_verdict;
  const highRisk = latest?.risk_level === "HIGH" || latest?.risk_level === "CRITICAL";

  useEffect(() => {
    api.capabilities().then(setCaps).catch(() => undefined);
  }, []);

  // Suspicious-pattern timeline rebuilt from backend layer evidence (chunk order).
  useEffect(() => {
    const entries: PatternEntry[] = [];
    const chronological = [...mic.results].reverse();
    for (const r of chronological) {
      if (r.status !== "analyzed" || !r.layers) continue;
      const n = r.analyses ?? 0;
      for (const l of r.layers) {
        for (const e of l.evidence) {
          if (/no .* trip-wire fired|^note:/.test(e)) continue;
          entries.push({ key: `${n}-${l.id}-${e}`, chunk: n, layer: l.name, text: e });
        }
      }
    }
    setPatterns(entries.slice(-30));
  }, [mic.results]);

  const flatPitch =
    mic.pitch.voiced > 20 &&
    mic.pitch.min !== null &&
    mic.pitch.max !== null &&
    mic.pitch.max - mic.pitch.min < 15;
  useEffect(() => {
    if (flatPitch && !flatRef.current) {
      flatRef.current = true;
      setPatterns((prev) =>
        [
          ...prev,
          {
            key: `flat-pitch-${Date.now()}`,
            chunk: latest?.analyses ?? 0,
            layer: "Pitch Track",
            text: `flat pitch ${(mic.pitch.min ?? 0).toFixed(0)}–${(mic.pitch.max ?? 0).toFixed(0)} Hz over ${mic.pitch.voiced} voiced frames (heuristic)`,
          },
        ].slice(-30),
      );
    } else if (!flatPitch) {
      flatRef.current = false;
    }
    // Transition detector: uses latest/pitch intentionally, runs on flatPitch flips.
  }, [flatPitch]);

  return (
    <div>
      <div className="mb-5 flex flex-wrap items-center gap-3">
        <h1 className="text-2xl font-bold">Live Detection</h1>
        <WsBadge status={status} />
        {mockMode === true && <Chip color="amber">Mock mode — not a real prediction</Chip>}
        <Chip color="cyan">Live microphone capture</Chip>
      </div>
      {caps && (
        <div className="mono mb-4 text-xs" style={{ color: "#7d93b2" }}>
          Models: neural {caps.neural.available ? "ready" : caps.neural.mock ? "mock (dev only)" : "unavailable"}
          {" · "}replay heuristics on
          {" · "}speaker comparison {caps.speaker.available ? "ready" : "unavailable — no reference backend"}
          {" · "}transcription {caps.transcription.available ? "ready" : "unavailable — paste a transcript below"}
          {" · "}provider attribution unavailable (returns unknown)
        </div>
      )}

      {highRisk && latest && <div className="mb-4"><ScamAlert riskLevel={latest.risk_level ?? ""} verdict={latest.verdict ?? undefined} /></div>}

      <Panel
        title="Mic live scan — real model, 1 s chunks"
        right={
          latest && latest.prediction ? (
            <span className="mono flex items-center gap-2 text-xs">
              <span className={`pred-${latest.prediction} font-semibold`}>
                {latest.prediction} {((latest.confidence ?? 0) * 100).toFixed(1)}%
              </span>
              <span style={{ color: "#7d93b2" }}>
                risk {latest.risk_score}/{latest.risk_level}
              </span>
            </span>
          ) : undefined
        }
      >
        <div className="mb-4 flex flex-wrap items-center gap-3">
          <select
            className="soc-select max-w-xs"
            aria-label="Microphone input device"
            value={mic.deviceId}
            disabled={mic.capturing}
            onChange={(e) => mic.setDeviceId(e.target.value)}
          >
            <option value="">Default microphone</option>
            {mic.devices
              .filter((d) => d.deviceId !== "" && d.deviceId !== "default")
              .map((d) => (
                <option key={d.deviceId} value={d.deviceId}>
                  {d.label}
                </option>
              ))}
          </select>
          {!mic.capturing ? (
            <button className="soc-btn" onClick={() => void mic.start()}>
              ▶ Start mic capture
            </button>
          ) : (
            <button className="soc-btn" onClick={mic.stop}>
              ■ Stop mic
            </button>
          )}
          {mic.info && (
            <span className="mono text-xs" style={{ color: "#7d93b2" }}>
              {mic.info.deviceLabel} · {mic.info.actualRate.toLocaleString()} Hz
              {mic.info.resampled ? " (resampled to 16 kHz)" : ""} · {mic.info.channels} ch
            </span>
          )}
          {flatPitch && <Badge tone="warn">flat pitch (heuristic)</Badge>}
        </div>
        {mic.error && <div className="mb-3 text-sm text-red-300">{mic.error}</div>}
        {mic.chunkError && (
          <div className="mb-3 text-sm" style={{ color: "#fbbf24" }}>
            Chunk upload: {mic.chunkError}
          </div>
        )}
        <LiveAnalyzerPanel engine={mic.engine} capturing={mic.capturing} pitch={mic.pitch} />
      </Panel>

      <div className="mt-4 grid grid-cols-1 gap-4 lg:grid-cols-2">
        <div>
          {sessionVerdict && (
            <div className="mono mb-2 flex flex-wrap items-center gap-2 text-xs">
              <span style={{ color: "#7d93b2" }}>
                session verdict ({sessionVerdict.votes}/{sessionVerdict.window} votes):
              </span>
              <VerdictBadge label={sessionVerdict.label} />
            </div>
          )}
          {latest && latest.status === "analyzed" && latest.aggregated_prediction && (
            <div className="mono mb-2 flex flex-wrap items-center gap-2 text-xs">
              <span style={{ color: "#7d93b2" }}>
                rolling neural median ({latest.aggregated_chunks ?? 0}/{latest.aggregated_window ?? 5} chunks):
              </span>
              <span className={`pred-${latest.aggregated_prediction} font-semibold`}>
                {latest.aggregated_prediction}{" "}
                {latest.aggregated_p_synthetic !== undefined && latest.aggregated_p_synthetic !== null
                  ? `${(latest.aggregated_p_synthetic * 100).toFixed(1)}%`
                  : ""}
              </span>
              <span style={{ color: "#7d93b2" }}>
                single chunk: {latest.prediction} {((latest.confidence ?? 0) * 100).toFixed(1)}%
              </span>
            </div>
          )}
          {latest && latest.status === "analyzed" && (
            <div className="mono mb-2 flex flex-wrap items-center gap-2 text-xs">
              <span style={{ color: "#7d93b2" }}>ensemble anti-spoof:</span>
              <span>
                mo {latest.ensemble?.p_mo !== undefined && latest.ensemble?.p_mo !== null ? latest.ensemble.p_mo.toFixed(3) : "n/a"}
                {" · "}bisher {latest.ensemble?.p_bisher !== undefined && latest.ensemble?.p_bisher !== null ? latest.ensemble.p_bisher.toFixed(3) : "n/a"}
                {" · "}stacked {latest.ensemble?.p_ensemble !== undefined && latest.ensemble?.p_ensemble !== null ? `${(latest.ensemble.p_ensemble * 100).toFixed(1)}%` : "Not available"}
              </span>
              <span style={{ color: "#7d93b2" }}>
                window {latest.ensemble?.window_n ?? 0}/{latest.ensemble?.window_need ?? 5} · alert {latest.ensemble?.alert ?? "n/a"}
              </span>
            </div>
          )}
          <VerdictCard verdict={latest?.verdict} title="Voice authenticity assessment (latest chunk)" />
          <div className="mono mt-2 space-y-1 text-xs" style={{ color: "#7d93b2" }}>
            <div>Voice cloning: reference comparison unavailable — no authorized reference sample.</div>
            <div>AI source attribution: unknown — no reliable source evidence (never inferred from scores).</div>
          </div>
        </div>
        <Panel title="Threat risk score" right={<span className="mono text-xs" style={{ color: "#7d93b2" }}>live model</span>}>
          <div className="flex flex-col items-center">
            <Gauge value={latest?.risk_score ?? 0} label="risk" sub={latest?.risk_level ?? "no data"} />
            <div className="mono mt-2 text-xs" style={{ color: "#7d93b2" }}>
              confidence (model){" "}
              <strong style={{ color: "#dbe7f5" }}>
                {latest?.confidence !== undefined ? `${(latest.confidence * 100).toFixed(1)}%` : "—"}
              </strong>
              {" · "}risk (engine){" "}
              <strong style={{ color: "#dbe7f5" }}>
                {latest ? `${latest.risk_score} / ${latest.risk_level}` : "—"}
              </strong>
            </div>
          </div>
        </Panel>
      </div>

      <div className="mt-4">
        <DetectionStack layers={latest?.layers} />
      </div>

      <div className="mt-4">
        <OriginCard
          origin={latest?.origin}
          status={latest?.origin_status ?? "unavailable"}
          windowSeconds={latest?.origin_window_seconds ?? latest?.buffered_seconds ?? 0}
          sessionOrigin={latest?.session_origin}
        />
      </div>

      <div className="mt-4">
        <Challenge />
      </div>

      <div className="mt-4">
        <CallSafety />
      </div>

      <div className="mt-4 grid grid-cols-1 gap-4 lg:grid-cols-2">
        <Panel title={`Suspicious pattern timeline (${patterns.length})`}>
          {patterns.length === 0 ? (
            <div className="text-sm" style={{ color: "#7d93b2" }}>
              No patterns flagged yet — start capture; layer evidence with chunk numbers appears here.
            </div>
          ) : (
            <ul className="mono max-h-72 space-y-1 overflow-y-auto text-xs">
              {[...patterns].reverse().map((p) => (
                <li key={p.key} className="rounded border-l-2 border-amber-800 bg-black/30 p-2">
                  <span style={{ color: "#fbbf24" }}>chunk {p.chunk}</span>{" "}
                  <span style={{ color: "#22d3ee" }}>{p.layer}</span>{" "}
                  <span style={{ color: "#7d93b2" }}>{p.text}</span>
                </li>
              ))}
            </ul>
          )}
        </Panel>

        <Panel title={`Live event stream (${status})`}>
          {events.length === 0 ? (
            <div className="text-sm" style={{ color: "#7d93b2" }}>
              Waiting for stream events… run an analysis or start mic capture.
            </div>
          ) : (
            <ul className="mono max-h-72 space-y-1 overflow-y-auto text-xs">
              {events.map((e, i) => (
                <li key={i} className="rounded border-l-2 border-cyan-800 bg-black/30 p-2">
                  <span style={{ color: "#22d3ee" }}>{e.event}</span>{" "}
                  <span style={{ color: "#7d93b2" }}>
                    {e.prediction ?? e.status ?? e.session_id ?? e.audio_name ?? ""}
                    {typeof e.confidence === "number" ? ` ${(e.confidence * 100).toFixed(1)}%` : ""}
                    {e.risk_level ? ` · ${e.risk_level}` : ""}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </Panel>
      </div>
    </div>
  );
}
