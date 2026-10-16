from typing import Optional, List
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime


class AudioInput(BaseModel):
    """Audio input for inference."""
    data: List[float] = Field(..., description="Audio samples as float array")
    sample_rate: int = Field(16000, description="Sample rate in Hz")
    frame_id: Optional[int] = Field(None, description="Frame identifier")


class NetworkTelemetryInput(BaseModel):
    """Network telemetry input for inference."""
    packet_loss: float = Field(..., ge=0.0, le=1.0)
    jitter: float = Field(..., ge=0.0)
    rtt: float = Field(..., ge=0.0)
    bitrate: float = Field(..., ge=0.0)
    codec: str = Field("opus", description="Audio codec")
    packet_timing: Optional[List[float]] = None
    packet_size: Optional[List[float]] = None
    burst_loss_count: Optional[int] = None
    burst_loss_duration: Optional[float] = None
    loss_burstiness: Optional[float] = None


class PredictRequest(BaseModel):
    """Request for single prediction."""
    audio: AudioInput
    network: NetworkTelemetryInput


class PredictResponse(BaseModel):
    """Response for single prediction."""
    decision: int = Field(..., description="0=REAL, 1=UNCERTAIN, 2=SYNTHETIC")
    decision_label: str = Field(..., description="REAL, UNCERTAIN, or SYNTHETIC")
    probability_real: float = Field(..., ge=0.0, le=1.0)
    probability_synthetic: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    
    # Intermediate values
    audio_reliability: Optional[float] = None
    network_reliability: Optional[float] = None
    gateway1_audio_weight: Optional[float] = None
    gateway1_network_weight: Optional[float] = None
    classical_probability: Optional[float] = None
    quantum_probability: Optional[float] = None
    gateway2_classical_weight: Optional[float] = None
    gateway2_quantum_weight: Optional[float] = None
    fused_probability: Optional[float] = None
    calibrated_probability: float
    
    # Latency info
    total_latency_ms: float
    stage_latencies: List[dict] = []
    within_total_target: bool
    
    # Metadata
    timestamp: datetime
    frame_id: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    engine_healthy: bool
    model_loaded: bool
    components: dict
    timestamp: datetime


class ModelInfoResponse(BaseModel):
    """Model information response."""
    model_version: str
    model_type: str
    components: dict
    latency_targets: dict
    calibration_method: str
    decision_thresholds: dict
    device: str


class VersionResponse(BaseModel):
    """API version response."""
    api_version: str
    model_version: str
    build_timestamp: Optional[str] = None


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: Optional[str] = None
    timestamp: datetime


class PredictionLog(BaseModel):
    """Prediction log entry."""
    frame_id: int
    timestamp: datetime
    decision: int
    decision_label: str
    probability_synthetic: float
    total_latency_ms: float
    within_target: bool