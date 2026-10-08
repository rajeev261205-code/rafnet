"""RAFNet API package — legacy names are optional (need torch).

`app` (the FastAPI application, SOC routes always mounted) is always available.
Legacy `/api/v1` names (`router`, `InferenceService`, ...) are only exported when
the torch stack imports cleanly; otherwise accessing them raises ImportError with
a clear message instead of breaking `import rafnet.api.main`.
"""
from __future__ import annotations

from typing import Any

from rafnet.api.main import app

try:
    from rafnet.api.routes import router
    from rafnet.api.service import InferenceService, get_inference_service, initialize_service
    from rafnet.api.schemas import (
        PredictRequest, PredictResponse, HealthResponse,
        ModelInfoResponse, VersionResponse, ErrorResponse,
        AudioInput, NetworkTelemetryInput,
    )

    __all__ = [
        "app",
        "router",
        "InferenceService",
        "get_inference_service",
        "initialize_service",
        "PredictRequest",
        "PredictResponse",
        "HealthResponse",
        "ModelInfoResponse",
        "VersionResponse",
        "ErrorResponse",
        "AudioInput",
        "NetworkTelemetryInput",
    ]
except ImportError as _exc:  # torch / heavy deps missing — SOC-only mode
    __all__ = ["app"]

    def __getattr__(name: str) -> Any:
        raise ImportError(
            f"rafnet.api.{name} requires the legacy torch stack ({_exc}); "
            "the SOC API (rafnet.soc) works without it."
        )
