from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Optional

class AnalysisResult(BaseModel):
    id: str
    user_id: str
    audio_name: str
    prediction: str
    confidence: float
    risk_score: int
    model_version: str
    processing_time_ms: float
    audio_duration_seconds: float
    created_at: datetime

class RiskScoreRequest(BaseModel):
    prediction: str
    confidence: float
    telemetry_anomalies: bool = False
