"""SOC routes — file-upload analysis + dashboard + model info + health.

Mounted alongside the existing /api/v1 JSON-frame API (routes.py) without
changing it. All errors return JSON without stack traces or secrets.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from pydantic import BaseModel

from .auth import get_optional_user
from .events import get_event_bus
from .live import WINDOW_SAMPLES, get_live_registry, resample_to_16k, validate_chunk
from .model_adapter import load_voice_model
from .ratelimit import RateLimiter, get_analyze_limiter, get_chunk_limiter
from .risk_engine import compute_risk
from .schemas import AnalyzeResponse, DashboardResponse, ModelInfo
from .service import SOCService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["soc"])

_service: SOCService | None = None
_limiter: RateLimiter | None = None
_chunk_limiter: RateLimiter | None = None


def get_soc_service() -> SOCService:
    global _service
    if _service is None:
        _service = SOCService(load_voice_model())
    return _service


def override_soc_service(service: SOCService) -> None:
    global _service
    _service = service


def get_limiter() -> RateLimiter:
    global _limiter
    if _limiter is None:
        _limiter = get_analyze_limiter()
    return _limiter


def get_chunk_limiter_dep() -> RateLimiter:
    global _chunk_limiter
    if _chunk_limiter is None:
        _chunk_limiter = get_chunk_limiter()
    return _chunk_limiter


def _client_key(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


@router.post("/analyze", response_model=AnalyzeResponse, summary="Analyze uploaded audio file")
async def analyze_audio(
    request: Request,
    file: UploadFile = File(...),
    service: SOCService = Depends(get_soc_service),
    limiter: RateLimiter = Depends(get_limiter),
    user: dict | None = Depends(get_optional_user),
):
    allowed, retry_after = limiter.allow(_client_key(request))
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Retry in {retry_after:.0f}s.",
            headers={"Retry-After": f"{retry_after:.0f}"},
        )
    try:
        data = await file.read()
        return service.analyze_bytes(
            data, file.filename or "upload.wav", user_id=(user or {}).get("id")
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception:  # noqa: BLE001 — never leak internals
        logger.exception("analyze failed")
        raise HTTPException(status_code=500, detail="Analysis failed")


@router.get("/analysis/{analysis_id}", summary="Get analysis by id")
def get_analysis(
    analysis_id: str,
    service: SOCService = Depends(get_soc_service),
    user: dict | None = Depends(get_optional_user),
):
    record = service.get_analysis(analysis_id, user_id=(user or {}).get("id"))
    if record is None:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return record


@router.get("/history", summary="Analysis history (newest first)")
def history(
    limit: int = 50,
    offset: int = 0,
    service: SOCService = Depends(get_soc_service),
    user: dict | None = Depends(get_optional_user),
):
    limit = max(1, min(limit, 200))
    offset = max(0, offset)
    uid = (user or {}).get("id")
    return {
        "items": service.list_analyses(limit=limit, offset=offset, user_id=uid),
        "total": service.total_count(uid),
    }


@router.get("/analysis/{analysis_id}/events", summary="Recent live events for context")
def analysis_events(analysis_id: str):
    bus = get_event_bus()
    related = [e for e in bus.recent(limit=200) if e.get("id") == analysis_id]
    return {"analysis_id": analysis_id, "events": related}


@router.get("/threats", summary="Threat list (HIGH/CRITICAL analyses)")
def threats(
    limit: int = 50,
    service: SOCService = Depends(get_soc_service),
    user: dict | None = Depends(get_optional_user),
):
    return {"items": service.list_threats(limit=max(1, min(limit, 200)), user_id=(user or {}).get("id"))}


@router.get("/threats/{threat_id}", summary="Threat detail")
def threat_detail(
    threat_id: str,
    service: SOCService = Depends(get_soc_service),
    user: dict | None = Depends(get_optional_user),
):
    threat = service.get_threat(threat_id, user_id=(user or {}).get("id"))
    if threat is None:
        raise HTTPException(status_code=404, detail="Threat not found")
    return threat


class ThreatStatusUpdate(BaseModel):
    status: str


@router.patch("/threats/{threat_id}", summary="Triage a threat (open/acknowledged/closed)")
def update_threat(
    threat_id: str,
    body: ThreatStatusUpdate,
    service: SOCService = Depends(get_soc_service),
    user: dict | None = Depends(get_optional_user),
):
    try:
        threat = service.set_threat_status(threat_id, body.status)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if threat is None:
        raise HTTPException(status_code=404, detail="Threat not found")
    return threat


@router.get("/calls", summary="Call/session list")
def calls(
    limit: int = 50,
    service: SOCService = Depends(get_soc_service),
    user: dict | None = Depends(get_optional_user),
):
    return {"items": service.list_calls(limit=max(1, min(limit, 200)), user_id=(user or {}).get("id"))}


@router.get("/calls/{call_id}", summary="Call/session detail")
def call_detail(
    call_id: str,
    service: SOCService = Depends(get_soc_service),
    user: dict | None = Depends(get_optional_user),
):
    call = service.get_call(call_id, user_id=(user or {}).get("id"))
    if call is None:
        raise HTTPException(status_code=404, detail="Call not found")
    return call


@router.get("/experiments", summary="Experiment tracking (verbatim report data)")
def experiments(service: SOCService = Depends(get_soc_service)):
    return {"items": service.list_experiments()}


class LiveChunkRequest(BaseModel):
    session_id: str | None = None
    sample_rate: int = 16000
    samples: list[float]


@router.post("/live/session", summary="Create a live ingestion session")
def create_live_session(user: dict | None = Depends(get_optional_user)):
    from .live import get_live_registry as _registry

    session = _registry().create()
    get_event_bus().emit("analysis_started", {"session_id": session.id, "source": "live"})
    return {"session_id": session.id, "window_samples": WINDOW_SAMPLES}


@router.post("/live/chunk", summary="Ingest a mic audio chunk (runs inference per full window)")
async def ingest_live_chunk(
    request: Request,
    body: LiveChunkRequest,
    service: SOCService = Depends(get_soc_service),
    limiter: RateLimiter = Depends(get_chunk_limiter_dep),
    user: dict | None = Depends(get_optional_user),
):
    allowed, retry_after = limiter.allow("chunk:" + _client_key(request))
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Retry in {retry_after:.0f}s.",
            headers={"Retry-After": f"{retry_after:.0f}"},
        )
    registry = get_live_registry()
    session = registry.get(body.session_id or "")
    if session is None:
        session = registry.create()
    try:
        clean = validate_chunk(body.sample_rate, body.samples)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    mono16k = resample_to_16k(clean, body.sample_rate)
    session.append(mono16k)
    if len(session._buf) < WINDOW_SAMPLES:
        return {
            "session_id": session.id,
            "status": "buffering",
            "buffered_seconds": round(session.buffered_seconds, 2),
            "window_seconds": 1.0,
        }
    window = session.latest_window()
    assert window is not None
    try:
        result = service.model.predict(window, 16000)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    risk = compute_risk(result.prediction, result.confidence, 1.0)
    session.analyses += 1
    bus = get_event_bus()
    bus.emit("processing", {"session_id": session.id, "source": "live"})
    bus.emit(
        "detection",
        {
            "session_id": session.id,
            "prediction": result.prediction,
            "confidence": round(result.confidence, 4),
            "risk_level": risk.risk_level,
            "source": "live",
        },
    )
    if risk.risk_level in ("HIGH", "CRITICAL"):
        bus.emit("threat_detected", {"session_id": session.id, "risk_level": risk.risk_level})
    return {
        "session_id": session.id,
        "status": "analyzed",
        "prediction": result.prediction,
        "confidence": round(result.confidence, 4),
        "risk_score": risk.risk_score,
        "risk_level": risk.risk_level,
        "model_version": result.model_version,
        "mock": result.mock,
        "buffered_seconds": round(session.buffered_seconds, 2),
        "analyses": session.analyses,
    }


@router.get("/dashboard", response_model=DashboardResponse, summary="SOC dashboard metrics")
def dashboard(
    service: SOCService = Depends(get_soc_service),
    user: dict | None = Depends(get_optional_user),
):
    return service.dashboard(user_id=(user or {}).get("id"))


@router.get("/models", response_model=ModelInfo, summary="Actual model information")
def models(service: SOCService = Depends(get_soc_service)):
    return service.model_info()


@router.get("/health", summary="SOC health incl. real model status")
def soc_health(service: SOCService = Depends(get_soc_service)):
    loaded = service.model.is_loaded()
    return {
        "status": "ok" if loaded else "degraded",
        "model_loaded": loaded,
        "mock_mode": service.mock_mode,
        "model": getattr(service.model, "name", "unknown"),
    }
