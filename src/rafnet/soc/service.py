"""SOC service — validation -> model -> risk -> store -> live events.

Persistence goes through `AnalysisStore` (in-memory by default, Supabase when
configured) so routes and the frontend contract never change. Every analysis
also emits real events on the `EventBus` (consumed by /ws/live) and derives:
- a call session (1:1 with the analysis),
- a threat record when risk is HIGH/CRITICAL.
Experiments are read verbatim from `experiments/*.json` (never fabricated).
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .audio_io import decode_audio
from .events import EventBus, get_event_bus
from .model_adapter import VoiceDetectionModel, VoiceResult
from .risk_engine import compute_risk
from .schemas import (
    AnalyzeResponse,
    CallSession,
    DashboardResponse,
    ExperimentRecord,
    ModelInfo,
    ThreatRecord,
)
from .store import AnalysisStore, get_store

logger = logging.getLogger(__name__)


def _experiments_dir() -> Path:
    override = os.getenv("EXPERIMENTS_DIR")
    if override:
        return Path(override)
    # src/rafnet/soc -> repo root -> experiments
    return Path(__file__).resolve().parents[3] / "experiments"


class SOCService:
    def __init__(
        self,
        model: VoiceDetectionModel,
        store: AnalysisStore | None = None,
        bus: EventBus | None = None,
    ) -> None:
        self.model = model
        self.store = store or get_store()
        self.bus = bus or get_event_bus()
        # Threat triage overrides (process-local; Supabase `threats.status`
        # is the persistent home once provisioned).
        self._threat_status: dict[str, str] = {}

    @property
    def mock_mode(self) -> bool:
        return getattr(self.model, "name", "") == "mock"

    # -- analysis -----------------------------------------------------
    def analyze_bytes(self, data: bytes, filename: str, user_id: str | None = None) -> AnalyzeResponse:
        t0 = time.perf_counter()
        safe_name = (filename or "upload.wav").rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        self.bus.emit("analysis_started", {"audio_name": safe_name})
        try:
            decoded = decode_audio(data, filename)
        except ValueError:
            self.bus.emit("error", {"stage": "validation", "audio_name": safe_name})
            raise
        self.bus.emit(
            "processing",
            {"audio_name": safe_name, "duration_seconds": round(decoded.duration_seconds, 2)},
        )
        try:
            result: VoiceResult = self.model.predict(decoded.samples, decoded.sample_rate)
        except RuntimeError:
            self.bus.emit("error", {"stage": "inference", "audio_name": safe_name})
            raise
        risk = compute_risk(result.prediction, result.confidence, result.audio_duration_seconds)
        record = AnalyzeResponse(
            id=str(uuid.uuid4()),
            prediction=result.prediction,
            confidence=round(result.confidence, 4),
            risk_score=risk.risk_score,
            risk_level=risk.risk_level,
            model_version=result.model_version,
            processing_time_ms=round(result.processing_time_ms or (time.perf_counter() - t0) * 1000, 1),
            audio_duration_seconds=round(result.audio_duration_seconds, 2),
            audio_name=safe_name,
            prob_synthetic=round(result.prob_synthetic, 4),
            prob_real=round(result.prob_real, 4),
            classical_probability=result.classical_probability,
            quantum_probability=result.quantum_probability,
            fused_probability=result.fused_probability,
            calibrated_probability=result.calibrated_probability,
            gateway1_audio_weight=result.gateway1_audio_weight,
            gateway1_network_weight=result.gateway1_network_weight,
            gateway2_classical_weight=result.gateway2_classical_weight,
            gateway2_quantum_weight=result.gateway2_quantum_weight,
            mock=result.mock,
            created_at=datetime.now(timezone.utc),
            reasons=risk.reasons,
        )
        dumped = record.model_dump(mode="json")
        if user_id:
            dumped["user_id"] = user_id
        self.store.save_analysis(dumped)
        self.bus.emit(
            "detection",
            {
                "id": record.id,
                "prediction": record.prediction,
                "confidence": record.confidence,
                "risk_level": record.risk_level,
            },
        )
        if record.risk_level in ("HIGH", "CRITICAL"):
            self.bus.emit(
                "threat_detected",
                {"id": record.id, "risk_score": record.risk_score, "risk_level": record.risk_level},
            )
        self.bus.emit("analysis_completed", {"id": record.id, "prediction": record.prediction})
        logger.info(
            "analyze id=%s pred=%s conf=%.3f risk=%s file=%s",
            record.id, record.prediction, record.confidence, record.risk_level, filename,
        )
        return record

    def get_analysis(self, analysis_id: str, user_id: str | None = None) -> dict | None:
        row = self.store.get_analysis(analysis_id)
        if row is None:
            return None
        owner = row.get("user_id")
        # Rows saved with an owner are visible only to that owner (when the
        # request is authenticated). Legacy anonymous rows stay visible to all.
        if owner is not None and user_id is not None and owner != user_id:
            return None
        return row

    def list_analyses(
        self, limit: int = 50, offset: int = 0, user_id: str | None = None
    ) -> list[dict]:
        rows = self.store.list_analyses(limit=500, offset=0)
        if user_id:
            # Authenticated view: own rows + legacy anonymous rows (local dev).
            # In production Supabase RLS enforces per-user isolation instead.
            rows = [r for r in rows if r.get("user_id") in (None, user_id)]
        return rows[offset : offset + limit]

    def total_count(self, user_id: str | None = None) -> int:
        if not user_id:
            return self.store.count()
        return len(self.list_analyses(limit=100000, user_id=user_id))

    # -- derived views -------------------------------------------------
    def _threat_for(self, row: dict) -> ThreatRecord | None:
        if row.get("risk_level") not in ("HIGH", "CRITICAL"):
            return None
        created = row.get("created_at")
        threat_id = f"thr-{row['id'][:8]}"
        return ThreatRecord(
            id=threat_id,
            analysis_id=row["id"],
            threat_type="synthetic_voice" if row.get("prediction") == "SYNTHETIC" else "uncertain_voice",
            severity="CRITICAL" if row.get("risk_level") == "CRITICAL" else "HIGH",
            confidence=float(row.get("confidence", 0.5)),
            status=self._threat_status.get(threat_id, "open"),
            details={
                "prob_synthetic": row.get("prob_synthetic"),
                "model_version": row.get("model_version"),
                "audio_name": row.get("audio_name"),
            },
            created_at=created if isinstance(created, datetime) else datetime.now(timezone.utc),
        )

    def list_threats(self, limit: int = 50, user_id: str | None = None) -> list[dict]:
        out: list[dict] = []
        for row in self.list_analyses(limit=500, user_id=user_id):
            threat = self._threat_for(row)
            if threat is not None:
                out.append(threat.model_dump(mode="json"))
            if len(out) >= limit:
                break
        return out

    def get_threat(self, threat_id: str, user_id: str | None = None) -> dict | None:
        return next((t for t in self.list_threats(limit=500, user_id=user_id) if t["id"] == threat_id), None)

    def set_threat_status(self, threat_id: str, status: str) -> dict | None:
        if status not in ("open", "acknowledged", "closed"):
            raise ValueError("status must be open, acknowledged, or closed")
        if self.get_threat(threat_id) is None:
            return None
        self._threat_status[threat_id] = status
        return self.get_threat(threat_id)

    def list_calls(self, limit: int = 50, user_id: str | None = None) -> list[dict]:
        out: list[dict] = []
        for row in self.list_analyses(limit=limit, user_id=user_id):
            created = row.get("created_at")
            out.append(
                CallSession(
                    id=f"call-{row['id'][:8]}",
                    analysis_id=row["id"],
                    duration_seconds=float(row.get("audio_duration_seconds", 0.0)),
                    status="completed",
                    risk_score=int(row.get("risk_score", 0)),
                    risk_level=str(row.get("risk_level", "LOW")),
                    prediction=str(row.get("prediction", "UNCERTAIN")),
                    created_at=created if isinstance(created, datetime) else datetime.now(timezone.utc),
                ).model_dump(mode="json")
            )
        return out

    def get_call(self, call_id: str, user_id: str | None = None) -> dict | None:
        return next((c for c in self.list_calls(limit=500, user_id=user_id) if c["id"] == call_id), None)

    # -- dashboard / models / experiments ------------------------------
    def dashboard(self, user_id: str | None = None) -> DashboardResponse:
        rows = self.list_analyses(limit=500, user_id=user_id)
        synth = sum(1 for r in rows if r["prediction"] == "SYNTHETIC")
        real = sum(1 for r in rows if r["prediction"] == "REAL")
        unc = sum(1 for r in rows if r["prediction"] == "UNCERTAIN")
        high = sum(1 for r in rows if r["risk_level"] in ("HIGH", "CRITICAL"))
        return DashboardResponse(
            total_analyses=self.total_count(user_id),
            synthetic_detections=synth,
            authentic_detections=real,
            uncertain_detections=unc,
            high_risk_events=high,
            recent=rows[:10],
            model_loaded=self.model.is_loaded(),
            model_version=getattr(getattr(self.model, "_pipeline", None), "model_version", "rafnet-1.0.0-phase19"),
            mock_mode=self.mock_mode,
        )

    def model_info(self) -> ModelInfo:
        ckpt = Path(os.getenv("MODEL_CHECKPOINT_DIR", "experiments/checkpoints"))
        files: dict = {}
        if ckpt.is_dir():
            for p in sorted(ckpt.glob("*")):
                if p.is_file():
                    files[p.name] = p.stat().st_size
        return ModelInfo(
            version=getattr(getattr(self.model, "_pipeline", None), "model_version", "rafnet-1.0.0-phase19"),
            checkpoints=files or {"note": "Not available from supplied model metadata"},
            thresholds={"tau_low": 0.361, "tau_high": 0.861},
            mock_mode=self.mock_mode,
        )

    def list_experiments(self) -> list[dict]:
        """Verbatim experiment records from real report files (never fabricated)."""
        experiments: list[dict] = []
        base = _experiments_dir()
        # Phase19 full report
        phase19 = base / "phase19_report.json"
        if phase19.is_file():
            try:
                data = json.loads(phase19.read_text())
                experiments.append(
                    ExperimentRecord(
                        id="phase19",
                        name="phase19 — full system report",
                        metrics={
                            "status": data.get("status"),
                            "test_decisions": ((data.get("evaluation") or {}).get("decisions")
                                               or (data.get("test") or {}).get("decisions")),
                            "latency_ms": (data.get("latency") or data.get("evaluation") or {}),
                        },
                        configuration={"dataset": data.get("dataset"), "components": list((data.get("components") or {}).keys())},
                        notes="Verbatim from phase19_report.json — see AI_MODEL_SPEC.md for caveats.",
                        created_at=datetime.now(timezone.utc),
                    ).model_dump(mode="json")
                )
            except Exception as exc:
                logger.warning("Could not parse %s: %s", phase19, exc)
        # Ablation smoke results (results.json per run dir)
        for results_path in sorted(base.glob("rafnet_ablation_smoke_*/results.json")):
            try:
                data = json.loads(results_path.read_text())
                experiments.append(
                    ExperimentRecord(
                        id=results_path.parent.name,
                        name=f"ablation smoke — {results_path.parent.name}",
                        metrics=data if isinstance(data, dict) else {"results": data},
                        configuration={},
                        notes="Verbatim from ablation results.json.",
                        created_at=datetime.now(timezone.utc),
                    ).model_dump(mode="json")
                )
                if len(experiments) >= 20:
                    break
            except Exception as exc:
                logger.warning("Could not parse %s: %s", results_path, exc)
        return experiments
