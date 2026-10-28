// Live microphone analyzer engine (Web Audio API).
// Captures at (or resamples to) 16 kHz mono, then derives three views from the
// SAME stream: scrolling waveform, scrolling short-time FFT spectrogram
// (log-magnitude, 0-8 kHz) and a YIN pitch track with voiced/unvoiced gating.
// All DSP comes from ./dsp; this module only owns devices, capture and buffers.
// Cleanup: stop() halts tracks, disconnects nodes and closes the context.
import {
  ANALYSIS_RATE,
  FFT_SIZE,
  SPEC_BINS,
  hannWindow,
  stftColumn,
  yinPitch,
  rms,
  resampleLinear,
} from "./dsp";

export const SPEC_HOP = 256; // 16 ms at 16 kHz
export const SPEC_HISTORY = 240; // ~3.8 s of columns
export const PITCH_HISTORY = 600;
export const WAVE_HISTORY_SECONDS = 3;

export interface InputDevice {
  deviceId: string;
  label: string;
}

export async function listInputDevices(): Promise<InputDevice[]> {
  if (!navigator.mediaDevices?.enumerateDevices) return [];
  const all = await navigator.mediaDevices.enumerateDevices();
  return all
    .filter((d) => d.kind === "audioinput")
    .map((d, i) => ({
      deviceId: d.deviceId,
      label: d.label || `Microphone ${i + 1} (label hidden until permission)`,
    }));
}

/** Honest, user-readable microphone errors (no guessing at causes). */
export function micErrorMessage(e: unknown): string {
  if (e instanceof DOMException || e instanceof Error) {
    switch (e.name) {
      case "NotAllowedError":
        return "Microphone permission denied. Allow microphone access in the browser and retry.";
      case "NotFoundError":
        return "No microphone found. Connect an input device and retry.";
      case "NotReadableError":
        return "Microphone is in use by another application. Close it and retry.";
      case "OverconstrainedError":
        return "The selected microphone is unavailable. Choose another input device.";
      case "SecurityError":
        return "Microphone blocked: this page must be served over localhost or HTTPS.";
      default:
        return e.message || "Microphone unavailable.";
    }
  }
  return "Microphone unavailable.";
}

export interface EngineInfo {
  deviceLabel: string;
  actualRate: number;
  resampled: boolean;
  channels: number;
}

export class LiveAudioEngine {
  specColumns: Float32Array[] = [];
  pitchPoints: Array<number | null> = [];
  wave: number[] = [];
  level = 0;
  info: EngineInfo;
  onSamples: ((samples: Float32Array) => void) | null = null;

  private ctx: AudioContext;
  private stream: MediaStream;
  private source: MediaStreamAudioSourceNode;
  private analyser: AnalyserNode;
  private processor: ScriptProcessorNode;
  private pending: number[] = [];
  private recent: number[] = [];
  private startedAt = performance.now();
  private closed = false;
  private window = hannWindow(FFT_SIZE);

  private constructor(
    ctx: AudioContext,
    stream: MediaStream,
    source: MediaStreamAudioSourceNode,
    analyser: AnalyserNode,
    processor: ScriptProcessorNode,
    info: EngineInfo,
  ) {
    this.ctx = ctx;
    this.stream = stream;
    this.source = source;
    this.analyser = analyser;
    this.processor = processor;
    this.info = info;
  }

  static async create(deviceId?: string): Promise<LiveAudioEngine> {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: deviceId ? { deviceId: { exact: deviceId } } : true,
    });
    const Ctx: typeof AudioContext | undefined =
      window.AudioContext ??
      (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!Ctx) {
      stream.getTracks().forEach((t) => t.stop());
      throw new Error("Web Audio is not supported in this browser");
    }
    // Request 16 kHz so capture/analysis run at the model rate; the browser
    // resamples from the hardware rate. If it refuses, we resample manually.
    let ctx: AudioContext;
    try {
      ctx = new Ctx({ sampleRate: ANALYSIS_RATE });
    } catch {
      ctx = new Ctx();
    }
    const track = stream.getAudioTracks()[0];
    const label = track?.label || "Default microphone";
    const settings = track?.getSettings?.();
    const info: EngineInfo = {
      deviceLabel: label,
      actualRate: ctx.sampleRate,
      resampled: ctx.sampleRate !== ANALYSIS_RATE,
      channels: settings?.channelCount ?? 1,
    };
    const source = ctx.createMediaStreamSource(stream);
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 2048;
    const processor = ctx.createScriptProcessor(4096, 1, 1);
    const engine = new LiveAudioEngine(ctx, stream, source, analyser, processor, info);
    processor.onaudioprocess = (ev) => {
      if (engine.closed) return;
      const input = ev.inputBuffer.getChannelData(0);
      let mono16k: Float32Array;
      if (ctx.sampleRate === ANALYSIS_RATE) {
        mono16k = input instanceof Float32Array ? input.slice() : new Float32Array(input);
      } else {
        mono16k = resampleLinear(
          input instanceof Float32Array ? input : new Float32Array(input),
          ctx.sampleRate,
          ANALYSIS_RATE,
        );
      }
      for (let i = 0; i < mono16k.length; i++) engine.pending.push(mono16k[i]);
      engine.onSamples?.(mono16k);
    };
    source.connect(analyser);
    source.connect(processor);
    // ScriptProcessor only runs while connected to the destination; gain 0 keeps it silent.
    const sink = ctx.createGain();
    sink.gain.value = 0;
    processor.connect(sink);
    sink.connect(ctx.destination);
    await ctx.resume().catch(() => undefined);
    return engine;
  }

  elapsedSec(): number {
    return (performance.now() - this.startedAt) / 1000;
  }

  /** Move newly captured audio through the STFT / pitch front-end. Call each animation frame. */
  pump(): void {
    if (this.closed || this.pending.length === 0) return;
    for (const s of this.pending) {
      this.recent.push(s);
      this.wave.push(s);
    }
    this.pending = [];
    const waveCap = ANALYSIS_RATE * WAVE_HISTORY_SECONDS;
    if (this.wave.length > waveCap) this.wave.splice(0, this.wave.length - waveCap);
    if (this.recent.length > 8192) this.recent.splice(0, this.recent.length - 8192);

    // Spectrogram: one log-magnitude column per hop from the newest audio.
    const hops = Math.floor(this.recent.length / SPEC_HOP);
    const maxCols = Math.min(hops, 24);
    const base = this.recent.length - maxCols * SPEC_HOP;
    for (let h = 0; h < maxCols; h++) {
      const end = base + (h + 1) * SPEC_HOP;
      const start = end - FFT_SIZE;
      if (start < 0) continue;
      const frame = new Float32Array(FFT_SIZE);
      for (let i = 0; i < FFT_SIZE; i++) frame[i] = this.recent[start + i];
      const col = stftColumn(frame, this.window);
      this.specColumns.push(col.length === SPEC_BINS ? col : col.slice(0, SPEC_BINS));
    }
    while (this.specColumns.length > SPEC_HISTORY) this.specColumns.shift();

    // Pitch: one YIN estimate per pump on the latest ~128 ms.
    const plen = Math.min(2048, this.recent.length);
    if (plen >= FFT_SIZE) {
      const frame = new Float32Array(plen);
      for (let i = 0; i < plen; i++) frame[i] = this.recent[this.recent.length - plen + i];
      const est = yinPitch(frame, ANALYSIS_RATE);
      this.pitchPoints.push(est.voiced ? est.f0 : null);
      while (this.pitchPoints.length > PITCH_HISTORY) this.pitchPoints.shift();
      this.level = rms(frame.slice(Math.max(0, plen - FFT_SIZE)));
    }
  }

  /** Latest time-domain block for the waveform view. Returns false when unavailable. */
  readWaveform(out: Float32Array): boolean {
    try {
      this.analyser.getFloatTimeDomainData(out as Float32Array<ArrayBuffer>);
      return true;
    } catch {
      return false;
    }
  }

  stop(): void {
    this.closed = true;
    try {
      this.processor.disconnect();
    } catch {
      /* ignore */
    }
    try {
      this.source.disconnect();
    } catch {
      /* ignore */
    }
    try {
      this.analyser.disconnect();
    } catch {
      /* ignore */
    }
    this.stream.getTracks().forEach((t) => t.stop());
    void this.ctx.close().catch(() => undefined);
  }
}
