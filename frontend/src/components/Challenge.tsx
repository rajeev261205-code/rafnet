// Liveness challenge flow (real): generate a random phrase, record the
// speaker saying it (3 s via the microphone), and re-run the backend
// analysis on that response. No mitigation buttons are faked — this records,
// uploads and analyzes, nothing else.
import { useEffect, useRef, useState } from "react";
import { api } from "../services/api";
import type { AnalyzeResponse } from "../types";
import { ANALYSIS_RATE, resampleLinear } from "../lib/dsp";
import { micErrorMessage } from "../lib/liveAnalyzer";
import { encodeWav } from "../lib/wav";
import { Panel } from "./ui";
import { VerdictBadge } from "./Verdict";

const WORDS = [
  "harbor", "candle", "rocket", "meadow", "silver", "thunder", "pocket", "garden",
  "mirror", "planet", "window", "forest", "bridge", "cloudy", "pepper", "sunset",
  "river", "stone", "tiger", "lantern", "ocean", "piano", "eagle", "canyon",
  "drum", "falcon", "glacier", "helmet", "island", "jungle", "kite", "lemon",
  "magnet", "needle", "orchard", "puzzle", "quartz", "raven", "saddle", "tunnel",
];

function randomPhrase(): string {
  const buf = new Uint32Array(4);
  crypto.getRandomValues(buf);
  return [0, 1, 2, 3].map((i) => WORDS[buf[i] % WORDS.length]).join(" ");
}

export default function Challenge(): React.JSX.Element {
  const [phrase, setPhrase] = useState<string>(() => randomPhrase());
  const [recording, setRecording] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const stopRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    return () => stopRef.current?.();
  }, []);

  const record = async (): Promise<void> => {
    setError(null);
    setResult(null);
    setRecording(true);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const Ctx: typeof AudioContext | undefined =
        window.AudioContext ??
        (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
      if (!Ctx) throw new Error("Web Audio is not supported in this browser");
      let ctx: AudioContext;
      try {
        ctx = new Ctx({ sampleRate: ANALYSIS_RATE });
      } catch {
        ctx = new Ctx();
      }
      const source = ctx.createMediaStreamSource(stream);
      const processor = ctx.createScriptProcessor(4096, 1, 1);
      const chunks: Float32Array[] = [];
      let need = ANALYSIS_RATE * 3;
      const stop = (): void => {
        try {
          processor.disconnect();
          source.disconnect();
        } catch {
          /* ignore */
        }
        stream.getTracks().forEach((t) => t.stop());
        void ctx.close().catch(() => undefined);
      };
      stopRef.current = stop;
      processor.onaudioprocess = (ev) => {
        if (need <= 0) return;
        const input = ev.inputBuffer.getChannelData(0);
        const mono = input instanceof Float32Array ? input : new Float32Array(input);
        const at16k =
          ctx.sampleRate === ANALYSIS_RATE ? mono.slice() : resampleLinear(mono, ctx.sampleRate, ANALYSIS_RATE);
        chunks.push(at16k);
        need -= at16k.length;
        if (need <= 0) void finish();
      };
      const finish = async (): Promise<void> => {
        stop();
        stopRef.current = null;
        setRecording(false);
        const total = chunks.reduce((a, c) => a + c.length, 0);
        const flat = new Float32Array(total);
        let off = 0;
        for (const c of chunks) {
          flat.set(c, off);
          off += c.length;
        }
        setBusy(true);
        try {
          const blob = encodeWav(flat, ANALYSIS_RATE);
          const r = await api.analyzeAudio(new File([blob], "challenge-response.wav"));
          setResult(r);
        } catch (e) {
          setError(e instanceof Error ? e.message : "Challenge analysis failed");
        } finally {
          setBusy(false);
        }
      };
      source.connect(processor);
      const sink = ctx.createGain();
      sink.gain.value = 0;
      processor.connect(sink);
      sink.connect(ctx.destination);
    } catch (e) {
      setRecording(false);
      setError(micErrorMessage(e));
    }
  };

  return (
    <Panel
      title="Liveness challenge (real)"
      right={
        <button className="raf-btn raf-btn-ghost" style={{ padding: "0.3rem 0.8rem" }} onClick={() => { setPhrase(randomPhrase()); setResult(null); }}>
          New phrase
        </button>
      }
    >
      <p className="text-sm" style={{ color: "#7d93b2" }}>
        Ask the speaker to say this phrase, record the response, and re-run the analysis on it.
      </p>
      <div className="mono mt-3 rounded-lg border p-4 text-center text-lg tracking-wide" style={{ borderColor: "#1b2a44", color: "#22d3ee" }}>
        “{phrase}”
      </div>
      <div className="mt-3 flex flex-wrap items-center gap-3">
        <button className="soc-btn" disabled={recording || busy} onClick={() => void record()}>
          {recording ? "● Recording response (3 s)…" : busy ? "Analyzing response…" : "Record response (3 s)"}
        </button>
      </div>
      {error && <div className="mt-3 text-sm text-red-300">{error}</div>}
      {result && !result.verdict && (
        <div className="mono mt-3 text-sm" style={{ color: "#7d93b2" }}>
          Model result {result.prediction} {(result.confidence * 100).toFixed(1)}% — no fused verdict on this record.
        </div>
      )}
      {result && result.verdict && (
        <div className="mono mt-3 flex flex-wrap items-center gap-3 text-sm">
          <VerdictBadge label={result.verdict.label} />
          <span>confidence {(result.verdict.confidence * 100).toFixed(1)}%</span>
          <span style={{ color: "#7d93b2" }}>
            risk {result.verdict.risk_score} / {result.verdict.risk_level} · model {result.prediction}{" "}
            {(result.confidence * 100).toFixed(1)}%
          </span>
        </div>
      )}
    </Panel>
  );
}
