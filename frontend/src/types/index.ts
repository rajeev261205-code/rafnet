export interface AnalysisResult {
  id: string;
  user_id: string;
  audio_name: string;
  prediction: string;
  confidence: number;
  risk_score: number;
  model_version: string;
  processing_time_ms: number;
  audio_duration_seconds: number;
  created_at: string;
}

export type RiskLevel = 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';

export interface ModelInfo {
  model_version: string;
  model_type: string;
  components: Record<string, string>;
  latency_targets: Record<string, number>;
}
