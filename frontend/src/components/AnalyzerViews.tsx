// Analyzer canvases: scrolling waveform / spectrogram / pitch track driven
// by the live engine, plus static full-file views for uploads. All drawing is
// requestAnimationFrame + canvas; every pixel comes from measured audio.
import { useEffect, useRef, useState } from "react";
import {
  ANALYSIS_RATE,
  FFT_SIZE,
  SPEC_BINS,
  hannWindow,
  median,
  rms,
  stftColumn,
  themeColor,
  yinPitch,
} from "../lib/dsp";
import { SPEC_HOP, type LiveAudioEngine } from "../lib/liveAnalyzer";
import type { PitchStats } from "../lib/useLiveCapture";
import { Tabs } from "./ui";

const BG = "#04080f";
const GRID = "rgba(125,147,178,0.18)";
const MUTED = "#7d93b2";

function fitCanvas(canvas: HTMLCanvasElement): CanvasRenderingContext2D | null {
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const rect = canvas.getBoundingClientRect();
  const w = Math.max(1, Math.floor(rect.width * dpr));
  const h = Math.max(1, Math.floor(rect.height * dpr));
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  const ctx = canvas.getContext("2d");
  if (ctx) ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

function cssSize(canvas: HTMLCanvasElement): { w: number; h: number } {
  const rect = canvas.getBoundingClientRect();
  return { w: Math.max(1, rect.width), h: Math.max(1, rect.height) };
}

/* ---------------- live waveform (scrolling, from 16 kHz history) ---------------- */
function LiveWaveform({ engine }: { engine: LiveAudioEngine }): React.JSX.Element {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    let raf = 0;
    const draw = (): void => {
      raf = requestAnimationFrame(draw);
      const canvas = ref.current;
      if (!canvas) return;
      const ctx = fitCanvas(canvas);
      if (!ctx) return;
      const { w, h } = cssSize(canvas);
      ctx.fillStyle = BG;
      ctx.fillRect(0, 0, w, h);
      ctx.strokeStyle = GRID;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, h / 2);
      ctx.lineTo(w, h / 2);
      ctx.stroke();
      const wave = engine.wave;
      if (wave.length < 2) return;
      const seconds = 3;
      const count = Math.min(wave.length, ANALYSIS_RATE * seconds);
      const start = wave.length - count;
      const stride = Math.max(1, Math.floor(count / w));
      ctx.strokeStyle = "#22d3ee";
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      for (let x = 0; x < w; x++) {
        const idx = start + Math.min(count - 1, x * stride);
        const v = wave[idx] ?? 0;
        const y = h / 2 - v * (h / 2 - 4);
        if (x === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [engine]);
  return <canvas ref={ref} className="h-64 w-full" aria-label="Live waveform" />;
}

/* ---------------- live spectrogram (scrolling short-time FFT) ---------------- */
function LiveSpectrogram({ engine }: { engine: LiveAudioEngine }): React.JSX.Element {
  const ref = useRef<HTMLCanvasElement>(null);
  const offRef = useRef<HTMLCanvasElement | null>(null);
  const drawnRef = useRef(0);
  useEffect(() => {
    drawnRef.current = 0;
    offRef.current = null;
  }, [engine]);
  useEffect(() => {
    let raf = 0;
    const draw = (): void => {
      raf = requestAnimationFrame(draw);
      const canvas = ref.current;
      if (!canvas) return;
      const ctx = fitCanvas(canvas);
      if (!ctx) return;
      const { w, h } = cssSize(canvas);
      if (!offRef.current) {
        offRef.current = document.createElement("canvas");
        drawnRef.current = 0;
      }
      const off = offRef.current;
      if (off.width !== canvas.width || off.height !== canvas.height) {
        off.width = canvas.width;
        off.height = canvas.height;
        drawnRef.current = 0;
      }
      const offCtx = off.getContext("2d");
      if (!offCtx) return;
      const cols = engine.specColumns;
      const fresh = cols.length - drawnRef.current;
      const colW = 2;
      if (drawnRef.current === 0) {
        offCtx.fillStyle = BG;
        offCtx.fillRect(0, 0, off.width, off.height);
      } else if (fresh > 0) {
        offCtx.drawImage(off, -fresh * colW, 0);
      }
      const startCol = Math.max(drawnRef.current, cols.length - Math.ceil(w / colW) - 64);
      for (let i = Math.max(drawnRef.current, startCol); i < cols.length; i++) {
        const col = cols[i];
        const x = off.width - (cols.length - i) * colW;
        for (let b = 0; b < SPEC_BINS; b += 2) {
          const t = col[b] ?? 0;
          if (t <= 0.02) continue;
          offCtx.fillStyle = themeColor(t);
          const y = off.height - ((b + 2) / SPEC_BINS) * off.height;
          offCtx.fillRect(x, y, colW, Math.max(1, (2 / SPEC_BINS) * off.height));
        }
      }
      drawnRef.current = cols.length;
      ctx.fillStyle = BG;
      ctx.fillRect(0, 0, w, h);
      ctx.drawImage(off, 0, 0, w, h);
      // Frequency axis overlay: 0-8 kHz.
      ctx.fillStyle = MUTED;
      ctx.font = "10px 'JetBrains Mono', monospace";
      for (const f of [0, 2000, 4000, 6000, 8000]) {
        const y = h - (f / 8000) * h;
        ctx.fillText(`${f / 1000}k`, 4, y - 2);
        ctx.strokeStyle = GRID;
        ctx.beginPath();
        ctx.moveTo(34, y);
        ctx.lineTo(w, y);
        ctx.stroke();
      }
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [engine]);
  return <canvas ref={ref} className="h-64 w-full" aria-label="Live spectrogram 0 to 8 kilohertz" />;
}

/* ---------------- live pitch track ---------------- */
const PITCH_MIN = 50;
const PITCH_MAX = 800;

function LivePitch({ engine }: { engine: LiveAudioEngine }): React.JSX.Element {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    let raf = 0;
    const draw = (): void => {
      raf = requestAnimationFrame(draw);
      const canvas = ref.current;
      if (!canvas) return;
      const ctx = fitCanvas(canvas);
      if (!ctx) return;
      const { w, h } = cssSize(canvas);
      ctx.fillStyle = BG;
      ctx.fillRect(0, 0, w, h);
      const toY = (f: number): number => {
        const l = Math.log(f / PITCH_MIN) / Math.log(PITCH_MAX / PITCH_MIN);
        return h - 14 - l * (h - 28);
      };
      ctx.font = "10px 'JetBrains Mono', monospace";
      for (const f of [100, 200, 400]) {
        const y = toY(f);
        ctx.strokeStyle = GRID;
        ctx.beginPath();
        ctx.moveTo(34, y);
        ctx.lineTo(w, y);
        ctx.stroke();
        ctx.fillStyle = MUTED;
        ctx.fillText(`${f}`, 4, y + 3);
      }
      ctx.fillStyle = MUTED;
      ctx.fillText("Hz", 4, 12);
      const pts = engine.pitchPoints;
      if (pts.length < 2) return;
      const n = pts.length;
      const span = Math.min(n, w);
      ctx.strokeStyle = "#22d3ee";
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      let pen = false;
      for (let x = 0; x < span; x++) {
        const p = pts[n - span + x];
        if (p === null || p === undefined) {
          pen = false;
          continue;
        }
        const y = toY(Math.max(PITCH_MIN, Math.min(PITCH_MAX, p)));
        const px = w - span + x;
        if (!pen) {
          ctx.moveTo(px, y);
          pen = true;
        } else ctx.lineTo(px, y);
      }
      ctx.stroke();
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [engine]);
  return <canvas ref={ref} className="h-64 w-full" aria-label="Live pitch track in hertz" />;
}

export type AnalyzerTab = "waveform" | "spectrogram" | "pitch";

export function LiveAnalyzerPanel({
  engine,
  capturing,
  pitch,
}: {
  engine: LiveAudioEngine | null;
  capturing: boolean;
  pitch: PitchStats;
}): React.JSX.Element {
  const [tab, setTab] = useState<AnalyzerTab>("waveform");
  const fmt = (v: number | null): string => (v === null ? "—" : `${v.toFixed(0)} Hz`);
  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center gap-3">
        <Tabs<AnalyzerTab>
          tabs={[
            { id: "waveform", label: "Waveform" },
            { id: "spectrogram", label: "Spectrogram" },
            { id: "pitch", label: "Pitch Track" },
          ]}
          active={tab}
          onChange={setTab}
        />
        {capturing && <span className="raf-badge raf-badge-ok">Live</span>}
      </div>
      {!engine || !capturing ? (
        <div className="flex h-64 flex-col items-center justify-center gap-2 rounded-lg border border-dashed p-6 text-center" style={{ borderColor: "#1b2a44" }}>
          <div className="mono text-xs uppercase tracking-widest" style={{ color: "#7d93b2" }}>
            Analyzer idle — start microphone capture
          </div>
          <div className="text-xs" style={{ color: "#7d93b2" }}>
            Waveform, spectrogram and pitch track are computed live from the mic stream at 16 kHz.
          </div>
        </div>
      ) : (
        <>
          {tab === "waveform" && <LiveWaveform engine={engine} />}
          {tab === "spectrogram" && <LiveSpectrogram engine={engine} />}
          {tab === "pitch" && <LivePitch engine={engine} />}
        </>
      )}
      <div className="mono mt-3 flex flex-wrap gap-x-4 gap-y-1 text-xs" style={{ color: "#7d93b2" }}>
        <span>
          ● capturing <span style={{ color: "#34d399" }}>●</span> buffering{" "}
          <span style={{ color: "#fbbf24" }}>●</span>
        </span>
        <span>
          current view: <span style={{ color: "#22d3ee" }}>{tab.toUpperCase()}</span> · 16,000 Hz
        </span>
        <span>
          pitch min {fmt(pitch.min)} · median {fmt(pitch.median)} · max {fmt(pitch.max)} · voiced{" "}
          {pitch.voiced}
        </span>
      </div>
    </div>
  );
}

/* ---------------- static full-file views ---------------- */
export interface FileAnalysis {
  columns: Float32Array[];
  pitch: Array<number | null>;
  peaks: Float32Array; // per-column max abs, 0..1
  pitchMin: number | null;
  pitchMed: number | null;
  pitchMax: number | null;
  level: number;
}

export function analyzeFile(samples16k: Float32Array): FileAnalysis {
  const window = hannWindow(FFT_SIZE);
  const targetCols = 1200;
  const hop = Math.max(SPEC_HOP, Math.floor(samples16k.length / SPEC_HOP / targetCols) * SPEC_HOP || SPEC_HOP);
  const columns: Float32Array[] = [];
  for (let end = FFT_SIZE; end <= samples16k.length; end += hop) {
    const frame = samples16k.slice(end - FFT_SIZE, end);
    columns.push(stftColumn(frame, window));
  }
  const pitch: Array<number | null> = [];
  const pitchHop = hop * 2;
  for (let end = 2048; end <= samples16k.length; end += pitchHop) {
    const frame = samples16k.slice(end - 2048, end);
    const est = yinPitch(frame, ANALYSIS_RATE);
    pitch.push(est.voiced ? est.f0 : null);
  }
  const voiced = pitch.filter((p): p is number => p !== null);
  const peakCount = 900;
  const peaks = new Float32Array(peakCount);
  const stride = Math.max(1, Math.floor(samples16k.length / peakCount));
  for (let i = 0; i < peakCount; i++) {
    let m = 0;
    const s = i * stride;
    for (let j = s; j < Math.min(s + stride, samples16k.length); j++) {
      const a = Math.abs(samples16k[j]);
      if (a > m) m = a;
    }
    peaks[i] = m;
  }
  return {
    columns,
    pitch,
    peaks,
    pitchMin: voiced.length > 0 ? Math.min(...voiced) : null,
    pitchMed: median(voiced),
    pitchMax: voiced.length > 0 ? Math.max(...voiced) : null,
    level: rms(samples16k),
  };
}

function StaticCanvas({ draw, label }: { draw: (ctx: CanvasRenderingContext2D, w: number, h: number) => void; label: string }): React.JSX.Element {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = fitCanvas(canvas);
    if (!ctx) return;
    const { w, h } = cssSize(canvas);
    draw(ctx, w, h);
  });
  return <canvas ref={ref} className="h-64 w-full" aria-label={label} />;
}

export function FileAnalyzerPanel({
  analysis,
  sampleRate,
  duration,
}: {
  analysis: FileAnalysis;
  sampleRate: number;
  duration: number;
}): React.JSX.Element {
  const [tab, setTab] = useState<AnalyzerTab>("waveform");
  const fmt = (v: number | null): string => (v === null ? "—" : `${v.toFixed(0)} Hz`);

  const drawWave = (ctx: CanvasRenderingContext2D, w: number, h: number): void => {
    ctx.fillStyle = BG;
    ctx.fillRect(0, 0, w, h);
    ctx.strokeStyle = GRID;
    ctx.beginPath();
    ctx.moveTo(0, h / 2);
    ctx.lineTo(w, h / 2);
    ctx.stroke();
    ctx.strokeStyle = "#22d3ee";
    ctx.lineWidth = 1;
    ctx.beginPath();
    const n = analysis.peaks.length;
    for (let x = 0; x < w; x++) {
      const v = analysis.peaks[Math.floor((x / w) * n)] ?? 0;
      const y = v * (h / 2 - 4);
      ctx.moveTo(x + 0.5, h / 2 - y);
      ctx.lineTo(x + 0.5, h / 2 + y);
    }
    ctx.stroke();
  };

  const drawSpec = (ctx: CanvasRenderingContext2D, w: number, h: number): void => {
    ctx.fillStyle = BG;
    ctx.fillRect(0, 0, w, h);
    const cols = analysis.columns;
    if (cols.length === 0) return;
    const colW = w / cols.length;
    for (let i = 0; i < cols.length; i++) {
      const col = cols[i];
      for (let b = 0; b < SPEC_BINS; b += 2) {
        const t = col[b] ?? 0;
        if (t <= 0.02) continue;
        ctx.fillStyle = themeColor(t);
        const y = h - ((b + 2) / SPEC_BINS) * h;
        ctx.fillRect(i * colW, y, Math.max(1, colW), Math.max(1, (2 / SPEC_BINS) * h));
      }
    }
    ctx.fillStyle = MUTED;
    ctx.font = "10px 'JetBrains Mono', monospace";
    for (const f of [0, 2000, 4000, 6000, 8000]) {
      ctx.fillText(`${f / 1000}k`, 4, h - (f / 8000) * h - 2);
    }
  };

  const drawPitch = (ctx: CanvasRenderingContext2D, w: number, h: number): void => {
    ctx.fillStyle = BG;
    ctx.fillRect(0, 0, w, h);
    const toY = (f: number): number => {
      const l = Math.log(f / PITCH_MIN) / Math.log(PITCH_MAX / PITCH_MIN);
      return h - 14 - l * (h - 28);
    };
    ctx.font = "10px 'JetBrains Mono', monospace";
    for (const f of [100, 200, 400]) {
      const y = toY(f);
      ctx.strokeStyle = GRID;
      ctx.beginPath();
      ctx.moveTo(34, y);
      ctx.lineTo(w, y);
      ctx.stroke();
      ctx.fillStyle = MUTED;
      ctx.fillText(`${f}`, 4, y + 3);
    }
    const pts = analysis.pitch;
    if (pts.length < 2) return;
    ctx.strokeStyle = "#22d3ee";
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    let pen = false;
    for (let i = 0; i < pts.length; i++) {
      const p = pts[i];
      if (p === null) {
        pen = false;
        continue;
      }
      const x = (i / (pts.length - 1)) * w;
      const y = toY(Math.max(PITCH_MIN, Math.min(PITCH_MAX, p)));
      if (!pen) {
        ctx.moveTo(x, y);
        pen = true;
      } else ctx.lineTo(x, y);
    }
    ctx.stroke();
  };

  const draw = tab === "waveform" ? drawWave : tab === "spectrogram" ? drawSpec : drawPitch;

  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center gap-3">
        <Tabs<AnalyzerTab>
          tabs={[
            { id: "waveform", label: "Waveform" },
            { id: "spectrogram", label: "Spectrogram" },
            { id: "pitch", label: "Pitch Track" },
          ]}
          active={tab}
          onChange={setTab}
        />
      </div>
      <StaticCanvas draw={draw} label={`File ${tab}`} />
      <div className="mono mt-3 flex flex-wrap gap-x-4 gap-y-1 text-xs" style={{ color: "#7d93b2" }}>
        <span>duration {duration.toFixed(2)} s · sample rate {sampleRate.toLocaleString()} Hz (analysis at 16 kHz)</span>
        <span>
          pitch min {fmt(analysis.pitchMin)} · median {fmt(analysis.pitchMed)} · max {fmt(analysis.pitchMax)}
        </span>
      </div>
    </div>
  );
}
