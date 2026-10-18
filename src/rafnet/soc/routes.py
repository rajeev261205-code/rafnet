"""SOC routes — file-upload analysis + dashboard + model info + health.

Mounted alongside the existing /api/v1 JSON-frame API (routes.py) without
changing it. All errors return JSON without stack traces or secrets.
"""
from __future__ import annotations

import logging
import os
from typing import Literal

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from .auth import get_optional_user
from .events import get_event_bus
from .live import WINDOW_SAMPLES, get_live_registry, resample_to_16k, validate_chunk
from .model_adapter import load_voice_model
from .ratelimit import RateLimiter, get_analyze_limiter, get_chunk_limiter
from .risk_engine import compute_risk
from .schemas import AnalyzeResponse, DashboardResponse, InputPath, ModelInfo
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
    # Only honor X-Forwarded-For when explicitly deployed behind a trusted
    # proxy (TRUST_PROXY=true); otherwise a client could spoof any key and
    # evade or poison another client's rate-limit bucket.
    if os.getenv("TRUST_PROXY", "false").lower() == "true":
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


@router.delete("/analyses", summary="Delete the caller's analyses (history/threats/calls)")
def clear_analyses(
    service: SOCService = Depends(get_soc_service),
    user: dict | None = Depends(get_optional_user),
):
    deleted = service.clear_analyses(user_id=(user or {}).get("id"))
    return {"deleted": deleted}


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
    status: Literal["open", "acknowledged", "closed"]


@router.patch("/threats/{threat_id}", summary="Triage a threat (open/acknowledged/closed)")
def update_threat(
    threat_id: str,
    body: ThreatStatusUpdate,
    service: SOCService = Depends(get_soc_service),
    user: dict | None = Depends(get_optional_user),
):
    try:
        threat = service.set_threat_status(
            threat_id, body.status, user_id=(user or {}).get("id")
        )
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


@router.get("/evaluation", summary="Measured layer evaluation (if an eval report exists)")
def layer_evaluation(service: SOCService = Depends(get_soc_service)):
    return service.layer_evaluation()


@router.get("/origin-evaluation", summary="Measured origin-model evaluation (honest unevaluated state if no data)")
def origin_evaluation(service: SOCService = Depends(get_soc_service)):
    return service.origin_evaluation()


class CallRiskRequest(BaseModel):
    transcript: str = Field(default="", max_length=6000)
    carrier: str = "unknown"
    language: str = "auto"


@router.post("/call-risk", summary="Scam-risk analysis of a supplied call transcript")
def call_risk(
    body: CallRiskRequest,
    service: SOCService = Depends(get_soc_service),
):
    from .scam import CARRIERS, analyze_call_risk

    carrier = (body.carrier or "unknown").strip().lower()
    if carrier not in CARRIERS:
        raise HTTPException(
            status_code=422, detail=f"carrier must be one of {list(CARRIERS)}"
        )
    result = analyze_call_risk(body.transcript, carrier=carrier, language=body.language)
    return {
        "risk": result.risk,
        "score": result.score,
        "scale": result.max_score,
        "evidence": result.evidence,
        "explanation": result.explanation,
        "actions": result.actions,
        "carrier": result.carrier,
        "classifier_prob": result.classifier_prob,
        "limitations": result.limitations,
    }


@router.get("/capabilities", summary="Model/service readiness (API up vs models ready)")
def capabilities(service: SOCService = Depends(get_soc_service)):
    from .capabilities import get_capabilities

    return get_capabilities(service.model)


class AttributionRequest(BaseModel):
    provenance: dict = Field(default_factory=dict)


@router.post("/attribution", summary="AI-provider attribution (unknown unless proven)")
def attribution(body: AttributionRequest):
    from .attribution import attribute_provider

    if not isinstance(body.provenance, dict):
        raise HTTPException(status_code=422, detail="provenance must be an object")
    result = attribute_provider(body.provenance)
    return {
        "provider": result.provider,
        "grade": result.grade,
        "evidence": result.evidence,
        "explanation": result.explanation,
        "limitations": result.limitations,
    }


class LiveChunkRequest(BaseModel):
    session_id: str | None = None
    sample_rate: int = 16000
    samples: list[float]
    input_path: InputPath | None = None


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
    # Ensemble anti-spoof gate (additive Phase 5): per-model P(AI) on the
    # same 1 s window, stacked + calibrated, appended to the session's
    # rolling ensemble history. CRITICAL requires the full 5-window history
    # hot plus >=2 models agreeing — a lone hot chunk (e.g. live mic at
    # 90.6%) can never raise CRITICAL by itself.
    ens: dict = {}
    try:
        from . import antispoof
        from .ensemble import WINDOW_N as ENS_WINDOW
        from .ensemble import alert_for as ens_alert_for

        hs = antispoof.score_all(window, 16000)
        current = ens_alert_for(hs.get("p_mo"), hs.get("p_bisher"),
                                round(float(result.prob_synthetic), 4),
                                window=[], is_live=False)
        if current.p_cal is not None:
            session.ensemble_p_history.append(current.p_cal)
        gated = ens_alert_for(hs.get("p_mo"), hs.get("p_bisher"),
                              round(float(result.prob_synthetic), 4),
                              window=list(session.ensemble_p_history), is_live=True)
        ens = {"p_mo": hs.get("p_mo"), "p_bisher": hs.get("p_bisher"),
               "p_ensemble": gated.p_cal, "verdict": gated.verdict,
               "agreement": gated.agreement, "alert": gated.alert,
               "reasons": gated.reasons,
               "window_n": len(session.ensemble_p_history),
               "window_need": ENS_WINDOW}
        # Phase-6 additive v2: head-aware 4-feature stack (single-window; the
        # v1 gated alert above stays authoritative for risk so behavior is
        # unchanged when the head artifact is absent).
        try:
            from .ensemble import alert_for4

            p_head = None
            head = getattr(service, "head4", None)
            if head is not None:
                p_head = head.predict_pai(window, 16000)
            a4 = alert_for4({"head": p_head, "rafnet": round(float(result.prob_synthetic), 4),
                             "mo": hs.get("p_mo"), "bisher": hs.get("p_bisher")},
                            window=[], is_live=False)
            ens.update({"p_head": p_head, "p_ensemble_v2": a4.p_cal,
                        "verdict_v2": a4.verdict, "alert_v2": a4.alert,
                        "agreement_v2": a4.agreement, "reasons_v2": a4.reasons})
        except Exception:
            logger.warning("live ensemble v2 unavailable", exc_info=True)
        if risk.risk_level == "CRITICAL" and gated.alert != "CRITICAL":
            risk = type(risk)(risk_score=min(risk.risk_score, 84),
                              risk_level="HIGH",
                              reasons=[*risk.reasons,
                                       f"CRITICAL gated to HIGH by ensemble window rule "
                                       f"(alert={gated.alert}, p_cal={gated.p_cal}, "
                                       f"agreement={gated.agreement}, "
                                       f"window={len(session.ensemble_p_history)}/{ENS_WINDOW})"])
        elif gated.alert == "CRITICAL" and risk.risk_level != "CRITICAL":
            # Window speaks louder than the single chunk: sustained hot
            # window + agreement raises CRITICAL even when this one chunk
            # flickers (single-chunk flicker is exactly what the window
            # rule exists to smooth over).
            risk = type(risk)(risk_score=max(risk.risk_score, 90),
                              risk_level="CRITICAL",
                              reasons=[*risk.reasons,
                                       f"ensemble window CRITICAL overrules single chunk "
                                       f"(p_cal={gated.p_cal}, agreement={gated.agreement}, "
                                       f"window={len(session.ensemble_p_history)}/{ENS_WINDOW})"])
    except Exception:
        logger.warning("live ensemble unavailable", exc_info=True)
    # Rolling neural aggregation: median of the last AGG_WINDOW chunk scores
    # with hysteresis (single-chunk flicker must not flip the session view).
    # The per-chunk prediction above is preserved untouched for comparison.
    from .live import (
        AGG_WINDOW,
        aggregate_probs,
        update_aggregated_prediction,
    )
    from .thresholds import load_decision_thresholds

    tau_low, tau_high, _ = load_decision_thresholds()
    try:
        p_syn = float(result.prob_synthetic)
    except (TypeError, ValueError):
        p_syn = 0.5
    session.neural_probs.append(p_syn)
    agg = aggregate_probs(session.neural_probs, tau_low, tau_high, window=AGG_WINDOW)
    session.agg_prediction = update_aggregated_prediction(
        session.neural_labels, session.agg_prediction, agg["label"]
    )
    # Detection layers + verdict (additive; never blocks the chunk response).
    layers: list[dict] = []
    chunk_verdict: dict | None = None
    try:
        from . import forensics
        from .speaker import enrollment_layer

        # mono16k is already resampled above; reuse it for forensics.
        l1 = forensics.spectral_layer(mono16k, 16000)
        neural = forensics.neural_layer(result)
        window_start = session.total_samples - WINDOW_SAMPLES
        past = [
            spec
            for (offset, spec) in session.spec_history
            if window_start - offset >= WINDOW_SAMPLES
        ]
        current = forensics.window_log_spectra(mono16k)
        l3 = forensics.liveness_layer(
            mono16k,
            16000,
            window_spectra=(past + current) or None,
            input_path=body.input_path.model_dump() if body.input_path else None,
            # Live chunks are raw mic PCM floats — never lossy uploads, so
            # Layer 3 keeps full replay analysis here by design.
            source={"codec": "pcm", "is_lossy": False, "sample_rate_hz": body.sample_rate},
        )
        for spec in current:
            session.spec_history.append((window_start, spec))
        chunk_verdict = forensics.fuse_verdict(
            l1, neural, l3,
            {"risk_score": risk.risk_score, "risk_level": risk.risk_level},
            duration_seconds=1.0,
        )
        session.session_verdict = forensics.update_session_verdict(
            session.verdict_labels, session.session_verdict, chunk_verdict["label"]
        )
        if session.session_verdict == chunk_verdict["label"] or session.analyses <= 1:
            session.session_reasons = chunk_verdict["reasons"]
        layers = [l1, neural, l3, enrollment_layer()]
    except Exception:
        logger.warning("live forensic layers unavailable", exc_info=True)
    session_votes = sum(1 for v in session.verdict_labels if v == session.session_verdict)
    # Voice-origin (additive): needs a rolling 4 s window. Until enough audio
    # has arrived the UI shows "collecting audio..." instead of a guess.
    origin_obj: dict | None = None
    origin_status = "unavailable"
    if getattr(service, "origin_model", None) is not None:
        if session.buffered_seconds >= 4.0:
            try:
                origin_obj = service.analyze_origin(list(session._buf)[-64000:], 16000)
                origin_status = "ready" if origin_obj else "error"
                if origin_obj:
                    session.origin_labels.append(origin_obj["label"])
                    session.session_origin = max(
                        set(session.origin_labels),
                        key=lambda lab: (
                            sum(1 for v in session.origin_labels if v == lab),
                            lab,
                        ),
                    )
            except Exception:
                logger.warning("live origin inference failed", exc_info=True)
                origin_status = "error"
        else:
            origin_status = "collecting"
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
        "layers": layers,
        "verdict": chunk_verdict,
        "aggregated_prediction": session.agg_prediction,
        "aggregated_p_synthetic": agg["p_median"],
        "aggregated_chunks": agg["chunks_used"],
        "aggregated_window": agg["window"],
        "thresholds": {"tau_low": round(tau_low, 4), "tau_high": round(tau_high, 4)},
        "ensemble": ens or None,
        "origin": origin_obj,
        "origin_status": origin_status,
        "origin_window_seconds": round(session.buffered_seconds, 2),
        "session_origin": session.session_origin,
        "session_verdict": (
            {
                "label": session.session_verdict,
                "reasons": session.session_reasons,
                "votes": session_votes,
                "window": len(session.verdict_labels),
            }
            if session.session_verdict is not None
            else None
        ),
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
    # Surface the load failure explicitly: no silent degradation. Mock mode
    # is only ever on when MOCK_INFERENCE=true was set (see model_adapter).
    if loaded:
        error = None
    elif service.mock_mode and os.getenv("MOCK_INFERENCE", "false").lower() != "true":
        error = "MockVoiceModel active without explicit MOCK_INFERENCE=true"
    else:
        error = getattr(service.model, "_load_error", None) or "model failed to load (see server logs)"
    try:
        from .fingerprint import build_fingerprint

        fingerprint = build_fingerprint(service.model)
    except Exception as exc:
        fingerprint = {"error": f"fingerprint unavailable: {exc}"}
    return {
        "status": "ok" if loaded else "degraded",
        "model_loaded": loaded,
        "mock_mode": service.mock_mode,
        "model": getattr(service.model, "name", "unknown"),
        "error": error,
        "fingerprint": fingerprint,
    }


@router.get("/enrollment", summary="Voice enrollment status (real capability probe)")
def enrollment_status():
    from .speaker import enrollment_status as _status

    return _status()


@router.post("/enrollment", summary="Enroll a voice sample (unavailable without a speaker model)")
async def enroll_voice(file: UploadFile = File(...)):
    from .speaker import speaker_capability

    cap = speaker_capability()
    if not cap["available"]:
        raise HTTPException(status_code=503, detail=cap["reason"])
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty audio file")
    # A usable speaker backend is present (checked above); comparison wiring
    # lands with the Layer 4 model integration.
    raise HTTPException(status_code=501, detail="Enrollment comparison not wired yet")
