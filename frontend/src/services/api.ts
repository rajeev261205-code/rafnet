// Typed backend client. Base URLs come from frontend/.env.local
// (VITE_API_URL, VITE_WS_URL). Supabase session token, when present,
// is attached as a Bearer token (backend validates it optionally).
import type {
  AnalyzeResponse,
  AttributionResult,
  CallRiskResult,
  CallSession,
  Capabilities,
  DashboardResponse,
  ExperimentRecord,
  HealthResponse,
  HistoryPage,
  InputPath,
  LiveChunkResponse,
  ModelInfo,
  ThreatRecord,
} from "../types";

export const API_BASE_URL =
  (import.meta.env.VITE_API_URL as string | undefined) || "http://localhost:8000";

export const WS_URL =
  (import.meta.env.VITE_WS_URL as string | undefined) || "ws://localhost:8000/ws/live";

let tokenGetter: () => string | null = () => null;

export function setAuthTokenGetter(fn: () => string | null): void {
  tokenGetter = fn;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const token = tokenGetter();
  const res = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: {
      ...(init?.headers ?? {}),
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
  });
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`Backend ${res.status} on ${path}: ${text.slice(0, 200)}`);
  }
  return (await res.json()) as T;
}

export const api = {
  health: () => request<HealthResponse>("/api/health"),
  dashboard: () => request<DashboardResponse>("/api/dashboard"),
  models: () => request<ModelInfo>("/api/models"),
  history: (limit = 50, offset = 0) =>
    request<HistoryPage>(`/api/history?limit=${limit}&offset=${offset}`),
  analysis: (id: string) => request<AnalyzeResponse>(`/api/analysis/${id}`),
  analysisEvents: (id: string) =>
    request<{ analysis_id: string; events: unknown[] }>(`/api/analysis/${id}/events`),
  threats: () => request<{ items: ThreatRecord[] }>("/api/threats"),
  setThreatStatus: (id: string, status: string) =>
    request<ThreatRecord>(`/api/threats/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ status }),
    }),
  clearAnalyses: () =>
    request<{ deleted: number }>("/api/analyses", { method: "DELETE" }),
  calls: () => request<{ items: CallSession[] }>("/api/calls"),
  experiments: () => request<{ items: ExperimentRecord[] }>("/api/experiments"),
  evaluation: () =>
    request<{ evaluated: boolean; report: Record<string, unknown> | null }>("/api/evaluation"),

  originEvaluation: () =>
    request<{ evaluated: boolean; report: Record<string, unknown> | null }>("/api/origin-evaluation"),

  analyzeAudio: async (file: File): Promise<AnalyzeResponse> => {
    const form = new FormData();
    form.append("file", file);
    const token = tokenGetter();
    const res = await fetch(`${API_BASE_URL}/api/analyze`, {
      method: "POST",
      headers: token ? { Authorization: `Bearer ${token}` } : {},
      body: form,
    });
    if (!res.ok) {
      const text = await res.text().catch(() => res.statusText);
      throw new Error(`Backend ${res.status} on /api/analyze: ${text.slice(0, 200)}`);
    }
    return (await res.json()) as AnalyzeResponse;
  },

  liveSession: () =>
    request<{ session_id: string; window_samples: number }>("/api/live/session", {
      method: "POST",
    }),

  liveChunk: (sessionId: string, sampleRate: number, samples: number[], inputPath?: InputPath) =>
    request<LiveChunkResponse>("/api/live/chunk", {      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: sessionId,
        sample_rate: sampleRate,
        samples,
        ...(inputPath ? { input_path: inputPath } : {}),
      }),
    }),

  callRisk: (transcript: string, carrier = "unknown", language = "auto") =>
    request<CallRiskResult>("/api/call-risk", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ transcript, carrier, language }),
    }),

  capabilities: () => request<Capabilities>("/api/capabilities"),

  attribution: (provenance?: Record<string, unknown>) =>
    request<AttributionResult>("/api/attribution", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ provenance: provenance ?? {} }),
    }),
};
