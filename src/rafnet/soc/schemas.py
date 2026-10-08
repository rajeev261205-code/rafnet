"""SOC schemas — stable detection-result contract for Frontend <-> Backend."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class AnalyzeResponse(BaseModel):
    id: str
    prediction: str = Field(description="REAL | UNCERTAIN | SYNTHETIC (model decision)")
    confidence: float = Field(ge=0.0, le=1.0, description="Model confidence, NOT risk")
    risk_score: int = Field(ge=0, le=100, description="SOC risk engine score")
    risk_level: str = Field(description="LOW | MEDIUM | HIGH | CRITICAL")
    model_version: str
    processing_time_ms: float
    audio_duration_seconds: float
    audio_name: str
    prob_synthetic: float
    prob_real: float
    classical_probability: float | None = None
    quantum_probability: float | None = None
    fused_probability: float | None = None
    calibrated_probability: float | None = None
    gateway1_audio_weight: float | None = None
    gateway1_network_weight: float | None = None
    gateway2_classical_weight: float | None = None
    gateway2_quantum_weight: float | None = None
    mock: bool = False
    created_at: datetime
    reasons: list[str] = Field(default_factory=list)


class DashboardResponse(BaseModel):
    total_analyses: int
    synthetic_detections: int
    authentic_detections: int
    uncertain_detections: int
    high_risk_events: int
    recent: list[dict[str, Any]] = Field(default_factory=list)
    model_loaded: bool
    model_version: str
    mock_mode: bool


class ModelInfo(BaseModel):
    name: str = "RAFNet"
    version: str
    framework: str = "PyTorch + scikit-learn"
    architecture: str = "WavLM + Handcrafted -> Gateway1 -> Classical/Quantum -> Gateway2 -> Calibration -> DecisionGate"
    sample_rate: int = 16000
    checkpoints: dict[str, Any] = Field(default_factory=dict)
    thresholds: dict[str, float] = Field(default_factory=dict)
    limitations: str = "See AI_MODEL_SPEC.md — synthetic-only training, untrained encoders, CPU latency over budget"
    mock_mode: bool = False


class CallSession(BaseModel):
    """Call/session tracking — derived 1:1 from an analysis unless the
    frontend supplies caller metadata (persisted in Supabase `calls`)."""

    id: str
    analysis_id: str
    caller: str = "unknown"
    receiver: str = "soc-console"
    duration_seconds: float = 0.0
    status: str = "completed"
    risk_score: int = 0
    risk_level: str = "LOW"
    prediction: str = "UNCERTAIN"
    created_at: datetime


class ThreatRecord(BaseModel):
    """Threat-management record — created for HIGH/CRITICAL analyses."""

    id: str
    analysis_id: str
    threat_type: str = "synthetic_voice"
    severity: str = "MEDIUM"
    confidence: float = 0.5
    status: str = "open"
    details: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class ExperimentRecord(BaseModel):
    """Experiment tracking — verbatim numbers from experiments/*.json only."""

    id: str
    name: str
    model_version: str = "rafnet-1.0.0-phase19"
    metrics: dict[str, Any] = Field(default_factory=dict)
    configuration: dict[str, Any] = Field(default_factory=dict)
    notes: str = ""
    created_at: datetime
