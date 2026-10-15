// Shared contract mirror of backend src/rafnet/soc/schemas.py + routes.py.
// Field names must stay identical to the backend JSON.

export type Prediction = "REAL" | "UNCERTAIN" | "SYNTHETIC";
export type RiskLevel = "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";

export interface AnalyzeResponse {
  id: string;
  prediction: Prediction;
  confidence: number;
  risk_score: number;
  risk_level: RiskLevel;
  model_version: string;
  processing_time_ms: number;
  audio_duration_seconds: number;
  audio_name: string;
  prob_synthetic: number;
  prob_real: number;
  classical_probability: number | null;
  quantum_probability: number | null;
  fused_probability: number | null;
  calibrated_probability: number | null;
  gateway1_audio_weight: number | null;
  gateway1_network_weight: number | null;
  gateway2_classical_weight: number | null;
  gateway2_quantum_weight: number | null;
  mock: boolean;
  created_at: string;
  reasons: string[];
  // Upload provenance (additive): lossy containers get Layer-3 codec handling.
  input_codec?: string | null;
  input_lossy?: boolean;
  // Measured per-stage timings in ms (additive diagnostics).
  stage_timings_ms?: Record<string, unknown>;
  // Additive Phase-4 fields (absent on old records → treat as unavailable).
  layers?: LayerResult[];
  verdict?: Verdict | null;
  origin?: OriginResult | null;
  // Additive Phase-5 ensemble (absent/null → "Not available").
  p_antispoof_mo?: number | null;
  p_antispoof_bisher?: number | null;
  p_ensemble?: number | null;
  ensemble_verdict?: string | null;
  ensemble_agreement?: number | null;
  ensemble_votes_fake?: number | null;
  ensemble_alert?: string | null;
  ensemble_reasons?: string[];
}

// Detection layers + verdict mirror backend src/rafnet/soc/schemas.py.
// kind "heuristic" outputs are unvalidated proxies, never certified detectors.
export type LayerKind = "heuristic" | "model" | "unavailable";

export interface LayerResult {
  id: string;
  name: string;
  kind: LayerKind;
  score: number | null;
  evidence: string[];
  metrics: Record<string, unknown>;
}

export type VerdictLabel =
  | "HUMAN_LIVE"
  | "AI_SYNTHETIC"
  | "REPLAY_PRERECORDED"
  | "EXTERNAL_APP_SUSPECTED"
  | "UNCERTAIN";

export interface Verdict {
  label: VerdictLabel;
  confidence: number;
  risk_score: number;
  risk_level: RiskLevel;
  reasons: string[];
  limitations: string[];
}

export interface SessionVerdict {
  label: VerdictLabel;
  reasons: string[];
  votes: number;
  window: number;
}

export interface LiveChunkResponse {
  session_id: string;
  status: "buffering" | "analyzed";
  prediction?: string;
  confidence?: number;
  risk_score?: number;
  risk_level?: string;
  model_version?: string;
  mock?: boolean;
  buffered_seconds?: number;
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

export interface OriginAISource {
  family: string;
  confidence: number;
  wording: string;
}

export interface OriginResult {
  label: string;
  probabilities: Record<string, number>;
  confidence: number;
  ai_source: OriginAISource | null;
  reasons: string[];
  limitations: string[];
  window_seconds_used: number;
  model_version: string;
}

export interface CallRiskEvidence {
  signal: string;
  label: string;
  match: string;
  excerpt: string;
}

export interface CallRiskResult {
  risk: "low" | "suspicious" | "high" | "insufficient";
  score: number;
  scale: number;
  evidence: CallRiskEvidence[];
  explanation: string;
  actions: string[];
  carrier: string;
  limitations: string[];
}

export interface CapabilityEntry {
  available: boolean;
  backend?: string | null;
  mock?: boolean;
  name?: string;
  kind?: string;
  task?: string;
  reason?: string;
}

export interface Capabilities {
  neural: CapabilityEntry;
  replay: CapabilityEntry;
  speaker: CapabilityEntry;
  transcription: CapabilityEntry;
  attribution: CapabilityEntry;
}

export interface AttributionResult {
  provider: string;
  grade: string;
  evidence: Record<string, unknown>[];
  explanation: string;
  limitations: string[];
}

export interface InputPath {
  device_label?: string | null;
  sample_rate?: number | null;
  channels?: number | null;
}

export interface HistoryPage {
  items: AnalyzeResponse[];
  total: number;
}

export interface DashboardResponse {
  total_analyses: number;
  synthetic_detections: number;
  authentic_detections: number;
  uncertain_detections: number;
  high_risk_events: number;
  recent: AnalyzeResponse[];
  model_loaded: boolean;
  model_version: string;
  mock_mode: boolean;
}

export interface ModelInfo {
  name: string;
  version: string;
  framework: string;
  architecture: string;
  sample_rate: number;
  checkpoints: Record<string, number | string>;
  thresholds: Record<string, number>;
  limitations: string;
  mock_mode: boolean;
  // Additive Phase-5 ensemble description (absent → unavailable).
  ensemble?: {
    models?: string[];
    stacking?: string;
    enabled?: boolean;
    taus?: number[] | null;
  } | null;
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  mock_mode: boolean;
  model: string;
  error?: string | null;
}

export interface ThreatRecord {
  id: string;
  analysis_id: string;
  threat_type: string;
  severity: string;
  confidence: number;
  status: string;
  verdict_label?: string | null;
  details: Record<string, unknown>;
  created_at: string;
}

export interface CallSession {
  id: string;
  analysis_id: string;
  caller: string;
  receiver: string;
  duration_seconds: number;
  status: string;
  risk_score: number;
  risk_level: string;
  prediction: string;
  verdict_label?: string | null;
  created_at: string;
}

export interface ExperimentRecord {
  id: string;
  name: string;
  model_version: string;
  metrics: Record<string, unknown>;
  configuration: Record<string, unknown>;
  notes: string;
  created_at: string;
}

// WebSocket frames from /ws/live (backend soc/websocket.py + events.py).
export interface LiveEvent {
  event: string;
  status?: string;
  id?: string;
  session_id?: string;
  source?: string;
  audio_name?: string;
  prediction?: string;
  confidence?: number;
  risk_level?: string;
  risk_score?: number;
  model_loaded?: boolean;
  mock_mode?: boolean;
  model?: string;
  auth_required?: boolean;
}
