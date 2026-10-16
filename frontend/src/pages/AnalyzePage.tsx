import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../services/api";
import type { AnalyzeResponse } from "../types";
import RiskGauge from "../components/RiskGauge";
import { FileAnalyzerPanel, analyzeFile } from "../components/AnalyzerViews";
import { DetectionStack, ScamAlert, VerdictCard } from "../components/Verdict";
import { resampleLinear, ANALYSIS_RATE } from "../lib/dsp";
import { Panel } from "../components/ui";

interface DecodedFile {
  mono16k: Float32Array;
  sampleRate: number;
  duration: number;
}

async function decodeToMono(file: File): Promise<DecodedFile> {
  const Ctx: typeof AudioContext | undefined =
    window.AudioContext ??
    (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
  if (!Ctx) throw new Error("Web Audio is not supported in this browser");
  const ctx = new Ctx();
  try {
    const raw = await file.arrayBuffer();
    const buf = await ctx.decodeAudioData(raw);
    const n = buf.length;
    const mono = new Float32Array(n);
    for (let ch = 0; ch < buf.numberOfChannels; ch++) {
      const data = buf.getChannelData(ch);
      for (let i = 0; i < n; i++) mono[i] += data[i] / buf.numberOfChannels;
    }
    const mono16k =
      buf.sampleRate === ANALYSIS_RATE ? mono : resampleLinear(mono, buf.sampleRate, ANALYSIS_RATE);
    return { mono16k, sampleRate: buf.sampleRate, duration: buf.duration };
  } finally {
    void ctx.close().catch(() => undefined);
  }
}

export default function AnalyzePage(): React.JSX.Element {
  const [file, setFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [decoded, setDecoded] = useState<DecodedFile | null>(null);
  const [decodeError, setDecodeError] = useState<string | null>(null);
  const [decoding, setDecoding] = useState(false);

  const onFile = (f: File | null): void => {
    setFile(f);
    setDecoded(null);
    setDecodeError(null);
    setResult(null);
    if (!f) return;
    setDecoding(true);
    decodeToMono(f)
      .then(setDecoded)
      .catch((e: Error) => setDecodeError(e.message || "Could not decode this audio file"))
      .finally(() => setDecoding(false));
  };

  const fileAnalysis = useMemo(
    () => (decoded ? analyzeFile(decoded.mono16k) : null),
    [decoded],
  );

  const run = async (): Promise<void> => {
    if (!file) return;
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const r = await api.analyzeAudio(file);
      setResult(r);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Analysis failed");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div>
      <h1 className="mb-5 text-2xl font-bold">Analyze</h1>
      <div className="soc-panel p-6">
        <div className="mono mb-2 text-xs uppercase tracking-widest" style={{ color: "#7d93b2" }}>
          Upload audio (wav / flac / ogg / mp3 / m4a, ≤ 25 MB)
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <label htmlFor="analyze-file" className="mono text-xs uppercase tracking-widest" style={{ color: "#7d93b2" }}>
            Audio file
          </label>
          <input
            id="analyze-file"
            type="file"
            accept=".wav,.flac,.ogg,.oga,.mp3,.m4a,.mp4,.webm,audio/*"
            onChange={(e) => onFile(e.target.files?.[0] ?? null)}
            className="text-sm"
          />
          <button className="soc-btn" disabled={busy || !file} onClick={() => void run()}>
            {busy ? "Analyzing…" : "Start analysis"}
          </button>
        </div>
        {error && <div className="mt-3 text-sm text-red-300">{error}</div>}
        {decoding && (
          <div className="mono mt-3 text-xs" style={{ color: "#7d93b2" }}>
            Decoding audio for visualization…
          </div>
        )}
        {decodeError && <div className="mt-3 text-sm text-red-300">{decodeError}</div>}
      </div>

      {decoded && fileAnalysis && (
        <Panel
          title="Acoustic views (decoded in your browser)"
          className="mt-5"
        >
          <FileAnalyzerPanel
            analysis={fileAnalysis}
            sampleRate={decoded.sampleRate}
            duration={decoded.duration}
          />
        </Panel>
      )}

      {result && (
        <>
        {(result.risk_level === "HIGH" || result.risk_level === "CRITICAL") && (
          <div className="mb-4">
            <ScamAlert riskLevel={result.risk_level} verdict={result.verdict ?? undefined} />
          </div>
        )}
        <VerdictCard verdict={result.verdict} />
        <div className="mt-5 grid grid-cols-1 gap-4 lg:grid-cols-3">
          <div className="soc-panel flex flex-col items-center p-6">
            <div className="mono mb-2 text-xs uppercase tracking-widest" style={{ color: "#7d93b2" }}>
              Risk
            </div>
            <RiskGauge score={result.risk_score} />
            <div className="mono mt-1 text-sm">{result.risk_level}</div>
          </div>
          <div className="soc-panel p-6">
            <div className="mono mb-3 text-xs uppercase tracking-widest" style={{ color: "#7d93b2" }}>
              Verdict (real model output)
            </div>
            <div className={`pred-${result.prediction} text-3xl font-bold`}>{result.prediction}</div>
            <div className="mono mt-2 text-sm">confidence {(result.confidence * 100).toFixed(2)}%</div>
            <div className="mono mt-1 text-sm" style={{ color: "#7d93b2" }}>
              synthetic {result.prob_synthetic.toFixed(3)} · real {result.prob_real.toFixed(3)}
            </div>
            {result.mock && (
              <div className="mono mt-2 text-xs" style={{ color: "#fbbf24" }}>
                MOCK MODE — not a real prediction
              </div>
            )}
            {result.input_lossy && (
              <div className="mono mt-2 text-xs" style={{ color: "#fbbf24" }}>
                LOSSY INPUT ({result.input_codec ?? "unknown codec"}) — replay assessment limited,
                codec cutoff excluded from Layer 3
              </div>
            )}
            <div className="mono mt-3 text-xs" style={{ color: "#7d93b2" }}>
              model {result.model_version} · {result.processing_time_ms.toFixed(0)} ms ·{" "}
              {result.audio_duration_seconds.toFixed(2)} s audio
            </div>
            <Link to="/history" className="mono mt-3 inline-block text-sm" style={{ color: "#22d3ee" }}>
              View in History →
            </Link>
          </div>
          <div className="soc-panel p-6">
            <div className="mono mb-3 text-xs uppercase tracking-widest" style={{ color: "#7d93b2" }}>
              Forensic signals
            </div>
            <dl className="mono space-y-1 text-xs">
              {[
                ["classical", result.classical_probability],
                ["quantum", result.quantum_probability],
                ["fused", result.fused_probability],
                ["calibrated", result.calibrated_probability],
                ["gw1 audio w", result.gateway1_audio_weight],
                ["gw1 network w", result.gateway1_network_weight],
                ["gw2 classical w", result.gateway2_classical_weight],
                ["gw2 quantum w", result.gateway2_quantum_weight],
              ].map(([k, v]) => (
                <div key={k as string} className="flex justify-between gap-4">
                  <dt style={{ color: "#7d93b2" }}>{k}</dt>
                  <dd>{typeof v === "number" ? v.toFixed(4) : "—"}</dd>
                </div>
              ))}
            </dl>
            {result.reasons.length > 0 && (
              <ul className="mt-3 list-disc pl-5 text-xs" style={{ color: "#7d93b2" }}>
                {result.reasons.map((r) => (
                  <li key={r}>{r}</li>
                ))}
              </ul>
            )}
          </div>
        </div>
        <div className="mt-4">
          <DetectionStack layers={result.layers} />
        </div>
        </>
      )}
    </div>
  );
}
