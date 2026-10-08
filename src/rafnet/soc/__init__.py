"""Aegis Voice SOC adapter — thin layer over the real RAFNet pipeline.

Source of truth for model behavior remains `src/rafnet/*`.
This package adds: file-upload audio decoding, VoiceDetectionModel interface,
risk engine separation, and SOC-oriented service/routes. No model logic is
duplicated here; all inference delegates to InferencePipeline / RAFNet.
"""
from .auth import auth_required, get_current_user, get_optional_user
from .events import EventBus, get_event_bus
from .model_adapter import MockVoiceModel, RAFNetVoiceModel, VoiceDetectionModel, load_voice_model
from .risk_engine import RISK_THRESHOLDS, RiskResult, compute_risk
from .store import AnalysisStore, InMemoryAnalysisStore, SupabaseAnalysisStore, get_store

__all__ = [
    "VoiceDetectionModel",
    "RAFNetVoiceModel",
    "MockVoiceModel",
    "load_voice_model",
    "compute_risk",
    "RiskResult",
    "RISK_THRESHOLDS",
    "EventBus",
    "get_event_bus",
    "AnalysisStore",
    "InMemoryAnalysisStore",
    "SupabaseAnalysisStore",
    "get_store",
    "auth_required",
    "get_current_user",
    "get_optional_user",
]
