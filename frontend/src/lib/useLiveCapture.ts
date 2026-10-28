// React binding for the live analyzer engine: device selection, capture
// lifecycle, 1 s backend chunk posts (every verdict comes from the backend),
// pitch statistics, and full cleanup on unmount (tracks, context, rAF).
import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../services/api";
import type { LayerResult, OriginResult, SessionVerdict, Verdict } from "../types";
import { ANALYSIS_RATE, median } from "./dsp";
import {
  LiveAudioEngine,
  listInputDevices,
  micErrorMessage,
  type EngineInfo,
  type InputDevice,
} from "./liveAnalyzer";

export interface ChunkResult {
  status: "buffering" | "analyzed";
  prediction?: string;
  confidence?: number;
  risk_score?: number;
  risk_level?: string;
  mock?: boolean;
  analyses?: number;
  layers?: LayerResult[];
  verdict?: Verdict | null;
  session_verdict?: SessionVerdict | null;
  aggregated_prediction?: string | null;
  aggregated_p_synthetic?: number | null;
  aggregated_chunks?: number;
  aggregated_window?: number;
  thresholds?: { tau_low: number; tau_high: number };
  origin?: OriginResult | null;
  origin_status?: string;
  origin_window_seconds?: number;
  session_origin?: string | null;
  buffered_seconds?: number;
  // Additive Phase-5 live ensemble gate (absent → unavailable).
  ensemble?: {
    p_mo?: number | null;
    p_bisher?: number | null;
    p_ensemble?: number | null;
    verdict?: string | null;
    agreement?: number | null;
    alert?: string | null;
    reasons?: string[];
    window_n?: number;
    window_need?: number;
  } | null;
}

export interface PitchStats {
  min: number | null;
  median: number | null;
  max: number | null;
  voiced: number;
}

export function useLiveCapture(): {
  engine: LiveAudioEngine | null;
  capturing: boolean;
  error: string | null;
  chunkError: string | null;
  devices: InputDevice[];
  deviceId: string;
  setDeviceId: (id: string) => void;
  info: EngineInfo | null;
  pitch: PitchStats;
  results: ChunkResult[];
  start: () => Promise<void>;
  stop: () => void;
} {
  const [engine, setEngine] = useState<LiveAudioEngine | null>(null);
  const [capturing, setCapturing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [chunkError, setChunkError] = useState<string | null>(null);
  const [devices, setDevices] = useState<InputDevice[]>([]);
  const [deviceId, setDeviceId] = useState("");
  const [pitch, setPitch] = useState<PitchStats>({ min: null, median: null, max: null, voiced: 0 });
  const [results, setResults] = useState<ChunkResult[]>([]);
  const engineRef = useRef<LiveAudioEngine | null>(null);
  const rafRef = useRef(0);
  const statsRef = useRef(0);
  const postingRef = useRef(false);
  const bufferRef = useRef<number[]>([]);
  const sessionRef = useRef("");
  const deviceIdRef = useRef("");
  deviceIdRef.current = deviceId;

  const refreshDevices = useCallback(() => {
    listInputDevices()
      .then(setDevices)
      .catch(() => undefined);
  }, []);

  useEffect(() => {
    void refreshDevices();
    if (!navigator.mediaDevices?.addEventListener) return;
    const onChange = (): void => refreshDevices();
    navigator.mediaDevices.addEventListener("devicechange", onChange);
    return () => navigator.mediaDevices.removeEventListener("devicechange", onChange);
  }, [refreshDevices]);

  const stop = useCallback(() => {
    cancelAnimationFrame(rafRef.current);
    engineRef.current?.stop();
    engineRef.current = null;
    bufferRef.current = [];
    postingRef.current = false;
    setEngine(null);
    setCapturing(false);
  }, []);

  // Full cleanup on unmount: rAF, tracks, audio context.
  useEffect(() => {
    return () => {
      cancelAnimationFrame(rafRef.current);
      engineRef.current?.stop();
      engineRef.current = null;
    };
  }, []);

  const start = useCallback(async () => {
    setError(null);
    setChunkError(null);
    setResults([]);
    setPitch({ min: null, median: null, max: null, voiced: 0 });
    try {
      const eng = await LiveAudioEngine.create(deviceIdRef.current || undefined);
      engineRef.current = eng;
      bufferRef.current = [];
      const session = await api.liveSession();
      sessionRef.current = session.session_id;
      // Forward 16 kHz frames into 1.0 s backend windows.
      eng.onSamples = (samples) => {
        const buf = bufferRef.current;
        for (let i = 0; i < samples.length; i++) buf.push(samples[i]);
        while (buf.length >= ANALYSIS_RATE) {
          const window = buf.slice(0, ANALYSIS_RATE);
          buf.splice(0, ANALYSIS_RATE);
          if (postingRef.current) continue;
          postingRef.current = true;
          const info = engineRef.current?.info;
          api
            .liveChunk(
              sessionRef.current,
              ANALYSIS_RATE,
              window,
              info
                ? {
                    device_label: info.deviceLabel,
                    sample_rate: info.actualRate,
                    channels: info.channels,
                  }
                : undefined,
            )
            .then((r) => setResults((prev) => [r, ...prev].slice(0, 20)))
            .catch((e: Error) => setChunkError(e.message))
            .finally(() => {
              postingRef.current = false;
            });
        }
      };
      setEngine(eng);
      setCapturing(true);
      void refreshDevices();
      const tick = (): void => {
        eng.pump();
        // Throttle React state: pitch stats at ~2 Hz, canvases read the engine directly.
        const now = performance.now();
        if (now - statsRef.current > 500) {
          statsRef.current = now;
          const voiced = eng.pitchPoints.filter((p): p is number => p !== null);
          const sorted = [...voiced].sort((a, b) => a - b);
          setPitch({
            min: sorted.length > 0 ? sorted[0] : null,
            median: median(voiced),
            max: sorted.length > 0 ? sorted[sorted.length - 1] : null,
            voiced: voiced.length,
          });
        }
        rafRef.current = requestAnimationFrame(tick);
      };
      rafRef.current = requestAnimationFrame(tick);
    } catch (e) {
      engineRef.current?.stop();
      engineRef.current = null;
      setError(micErrorMessage(e));
      setCapturing(false);
    }
  }, [refreshDevices]);

  return {
    engine,
    capturing,
    error,
    chunkError,
    devices,
    deviceId,
    setDeviceId,
    info: engine?.info ?? null,
    pitch,
    results,
    start,
    stop,
  };
}
