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
from typing import Any

from .audio_io import decode_audio
from .events import EventBus, get_event_bus
from .model_adapter import VoiceDetectionModel, VoiceResult
from .risk_engine import compute_risk
from .schemas import (
    AnalyzeResponse,
    CallSession,
    DashboardResponse,
    ExperimentRecord,
    LayerResult,
    ModelInfo,
    ThreatRecord,
    Verdict,
)
from .store import AnalysisStore, get_store

logger = logging.getLogger(__name__)


def _experiments_dir() -> Path:
    override = os.getenv("EXPERIMENTS_DIR")
    if override:
        path = Path(override)
        if not path.is_absolute():
            # Same cwd trap as MODEL_CHECKPOINT_DIR: .env paths are
            # repo-root-relative, the server runs with cwd=src/.
            path = Path(__file__).resolve().parents[3] / path
        return path
    # src/rafnet/soc -> repo root -> experiments
    return Path(__file__).resolve().parents[3] / "experiments"


def _reports_dir() -> Path:
    override = os.getenv("REPORTS_DIR")
    if override:
        path = Path(override)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[3] / path
        return path
    # src/rafnet/soc -> repo root -> reports
    return Path(__file__).resolve().parents[3] / "reports"


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
        # Voice-origin classifier (additive; None => origin:null downstream).
        self.origin_model = None
        try:
            from .origin_model import load_origin_model

            self.origin_model = load_origin_model()
        except Exception as exc:
            logger.warning("Origin model unavailable (non-fatal): %s", exc)
        # Phase-6 WavLM head (additive 4th stack feature; None => v2 fields omitted).
        self.head4 = None
        try:
            from .head4 import load_head

            self.head4 = load_head()
            if not getattr(self.head4, "is_loaded", lambda: False)():
                logger.warning("Head4 not loaded (non-fatal): %s",
                               getattr(self.head4, "error", "unknown"))
                self.head4 = None
        except Exception as exc:
            logger.warning("Head4 unavailable (non-fatal): %s", exc)

    def analyze_origin(self, samples: Any, sample_rate: int) -> dict | None:
        """Run the origin classifier when installed; else None (origin:null)."""
        if self.origin_model is None:
            return None
        try:
            return self.origin_model.predict(samples, sample_rate).to_dict()
        except Exception as exc:
            logger.warning("Origin inference failed (non-fatal): %s", exc)
            return None

    @property
    def mock_mode(self) -> bool:
        return getattr(self.model, "name", "") == "mock"

    # -- analysis -----------------------------------------------------
    def analyze_bytes(self, data: bytes, filename: str, user_id: str | None = None) -> AnalyzeResponse:
        t0 = time.perf_counter()
        safe_name = (filename or "upload.wav").rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        self.bus.emit("analysis_started", {"audio_name": safe_name})
        timings: dict[str, float] = {}
        try:
            t_decode = time.perf_counter()
            decoded = decode_audio(data, filename)
            timings["decode_ms"] = round((time.perf_counter() - t_decode) * 1000, 1)
        except ValueError:
            self.bus.emit("error", {"stage": "validation", "audio_name": safe_name})
            raise
        self.bus.emit(
            "processing",
            {"audio_name": safe_name, "duration_seconds": round(decoded.duration_seconds, 2)},
        )
        try:
            t_inf = time.perf_counter()
            result: VoiceResult = self.model.predict(decoded.samples, decoded.sample_rate)
            timings["inference_ms"] = round((time.perf_counter() - t_inf) * 1000, 1)
        except RuntimeError:
            self.bus.emit("error", {"stage": "inference", "audio_name": safe_name})
            raise
        risk = compute_risk(result.prediction, result.confidence, result.audio_duration_seconds)
        # Ensemble anti-spoofing (additive Phase 5; never blocks analysis).
        # Single-shot uploads carry no rolling window, so CRITICAL is
        # unreachable here by construction (file path caps at HIGH).
        ensemble: dict[str, Any] = {}
        try:
            from . import antispoof
            from .ensemble import alert_for

            t_ens = time.perf_counter()
            hs = antispoof.score_all(decoded.samples, decoded.sample_rate)
            timings["ensemble_hf_ms"] = round((time.perf_counter() - t_ens) * 1000, 1)
            alert = alert_for(hs.get("p_mo"), hs.get("p_bisher"),
                              round(result.prob_synthetic, 4),
                              window=[], is_live=False)
            ensemble = {"p_mo": hs.get("p_mo"), "p_bisher": hs.get("p_bisher"),
                        "p_ensemble": alert.p_cal, "verdict": alert.verdict,
                        "agreement": alert.agreement,
                        "votes": None, "alert": alert.alert, "reasons": alert.reasons,
                        "latencies": {k: v for k, v in hs.items() if k.startswith("latency_")}}
            try:
                from .ensemble import calibrated_pai, load_stacking

                s = calibrated_pai(hs.get("p_mo"), hs.get("p_bisher"),
                                   round(result.prob_synthetic, 4), load_stacking())
                ensemble["votes"] = s.get("votes_fake")
            except Exception:
                pass
            # Phase-6 additive v2: 4-feature stack incl. the WavLM head.
            try:
                from .ensemble import alert_for4

                p_head = None
                if self.head4 is not None:
                    t_head = time.perf_counter()
                    p_head = self.head4.predict_pai(decoded.samples, decoded.sample_rate)
                    timings["head_ms"] = round((time.perf_counter() - t_head) * 1000, 1)
                a4 = alert_for4({"head": p_head, "rafnet": round(result.prob_synthetic, 4),
                                 "mo": hs.get("p_mo"), "bisher": hs.get("p_bisher")},
                                window=[], is_live=False)
                ensemble["p_head"] = p_head
                ensemble["p_ensemble_v2"] = a4.p_cal
                ensemble["verdict_v2"] = a4.verdict
                ensemble["alert_v2"] = a4.alert
                ensemble["agreement_v2"] = a4.agreement
                ensemble["reasons_v2"] = a4.reasons
            except Exception:
                logger.warning("ensemble v2 unavailable for %s", safe_name, exc_info=True)
            if risk.risk_level == "CRITICAL":
                risk_reason = (f"CRITICAL gated to HIGH: single-shot upload cannot satisfy "
                               f"the live-window rule (ensemble alert={alert.alert}, "
                               f"p_cal={alert.p_cal}, agreement={alert.agreement})")
                risk = type(risk)(risk_score=min(risk.risk_score, 84),
                                  risk_level="HIGH",
                                  reasons=[*risk.reasons, risk_reason])
            elif alert.alert == "HIGH" and risk.risk_level == "MEDIUM":
                # Single-shot ensemble SYNTHETIC (no window, so never CRITICAL):
                # lift to HIGH so the SOC threat flow still fires for uploads.
                risk = type(risk)(risk_score=max(risk.risk_score, 65),
                                  risk_level="HIGH",
                                  reasons=[*risk.reasons,
                                           f"ensemble SYNTHETIC single-shot "
                                           f"(p_cal={alert.p_cal}, agreement={alert.agreement}) -> HIGH"]) 
        except Exception:
            logger.warning("ensemble scoring unavailable for %s", safe_name, exc_info=True)
        # Detection layers + verdict (additive; heuristic layers never block analysis).
        layers: list[LayerResult] = []
        verdict: Verdict | None = None
        try:
            from . import forensics
            from .speaker import enrollment_layer

            source = {
                "codec": decoded.original_suffix or None,
                "is_lossy": decoded.is_lossy,
                "sample_rate_hz": decoded.sample_rate,
            }
            t_l1 = time.perf_counter()
            l1 = forensics.spectral_layer(decoded.samples, decoded.sample_rate)
            timings["forensics_spectral_ms"] = round((time.perf_counter() - t_l1) * 1000, 1)
            neural = forensics.neural_layer(result)
            t_l3 = time.perf_counter()
            l3 = forensics.liveness_layer(decoded.samples, decoded.sample_rate, source=source)
            timings["forensics_liveness_ms"] = round((time.perf_counter() - t_l3) * 1000, 1)
            fused = forensics.fuse_verdict(
                l1,
                neural,
                l3,
                {"risk_score": risk.risk_score, "risk_level": risk.risk_level},
                duration_seconds=result.audio_duration_seconds,
            )
            layers = [
                LayerResult(**l1),
                LayerResult(**neural),
                LayerResult(**l3),
                LayerResult(**enrollment_layer()),
            ]
            verdict = Verdict(**fused)
        except Exception:
            logger.warning("forensic layers unavailable for %s", safe_name, exc_info=True)
        stage_timings: dict[str, Any] = dict(timings)
        model_stages = getattr(result, "stage_timings_ms", None)
        if isinstance(model_stages, dict) and model_stages:
            stage_timings["model_stages_ms"] = model_stages
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
            input_codec=decoded.original_suffix or None,
            input_lossy=bool(decoded.is_lossy),
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
            p_antispoof_mo=ensemble.get("p_mo"),
            p_antispoof_bisher=ensemble.get("p_bisher"),
            p_ensemble=ensemble.get("p_ensemble"),
            ensemble_verdict=ensemble.get("verdict"),
            ensemble_agreement=ensemble.get("agreement"),
            ensemble_votes_fake=ensemble.get("votes"),
            ensemble_alert=ensemble.get("alert"),
            ensemble_reasons=ensemble.get("reasons", []),
            mock=result.mock,
            created_at=datetime.now(timezone.utc),
            reasons=risk.reasons,
            layers=layers,
            verdict=verdict,
            stage_timings_ms=stage_timings,
            origin=self.analyze_origin(decoded.samples, decoded.sample_rate),
        )
        dumped = record.model_dump(mode="json")
        if user_id:
            dumped["user_id"] = user_id
        t_store = time.perf_counter()
        self.store.save_analysis(dumped)
        # Single save only (a second write would duplicate rows). The store
        # round-trip cost is added to the response timings; the in-memory
        # adapter holds the same dict object so its copy sees it too, while
        # the Supabase copy keeps timings without this sub-field.
        stage_timings["risk_store_ms"] = round((time.perf_counter() - t_store) * 1000, 1)
        dumped["stage_timings_ms"] = stage_timings
        record.stage_timings_ms = stage_timings
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
        # Rows saved with an owner are visible only to that owner.
        # Anonymous requests (user_id=None) may see legacy anonymous rows only.
        if owner is not None and owner != user_id:
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

    def clear_analyses(self, user_id: str | None = None) -> int:
        """Delete the caller's analyses (threats/calls derive from analyses,
        so they vanish too). Also drops process-local triage overrides for
        threats that no longer exist."""
        deleted = self.store.clear(user_id)
        surviving = {t["id"] for t in self.list_threats(limit=500, user_id=user_id)}
        for tid in [t for t in self._threat_status if t not in surviving]:
            del self._threat_status[tid]
        return deleted

    # -- derived views -------------------------------------------------
    def _verdict_label(self, row: dict) -> str | None:
        verdict = row.get("verdict")
        if isinstance(verdict, dict):
            label = verdict.get("label")
            return str(label) if label else None
        return None

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
            verdict_label=self._verdict_label(row),
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

    def set_threat_status(
        self, threat_id: str, status: str, user_id: str | None = None
    ) -> dict | None:
        if status not in ("open", "acknowledged", "closed"):
            raise ValueError("status must be open, acknowledged, or closed")
        if self.get_threat(threat_id, user_id=user_id) is None:
            return None
        self._threat_status[threat_id] = status
        return self.get_threat(threat_id, user_id=user_id)

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
                    verdict_label=self._verdict_label(row),
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
        from .model_adapter import resolve_checkpoint_dir

        ckpt = resolve_checkpoint_dir()
        files: dict = {}
        if ckpt.is_dir():
            for p in sorted(ckpt.glob("*")):
                if p.is_file():
                    files[p.name] = p.stat().st_size
        # Active thresholds: live gate when the real pipeline loaded, else the
        # measured config file, else the shipped defaults (never invented).
        tau_low, tau_high = 0.3611, 0.8611
        try:
            pipe = getattr(self.model, "_pipeline", None)
            gate = getattr(pipe, "decision_gate", None) if pipe is not None else None
            if gate is not None and hasattr(gate, "tau_low"):
                tl = gate.tau_low.item() if hasattr(gate.tau_low, "item") else float(gate.tau_low)
                th = gate.tau_high.item() if hasattr(gate.tau_high, "item") else float(gate.tau_high)
                tau_low, tau_high = round(float(tl), 4), round(float(th), 4)
            else:
                from .thresholds import load_decision_thresholds

                tl, th, _ = load_decision_thresholds()
                tau_low, tau_high = round(tl, 4), round(th, 4)
        except Exception:
            logger.warning("model_info thresholds unreadable; reporting defaults", exc_info=True)
        ensemble_info: dict[str, Any] | None = None
        try:
            from . import antispoof
            from .ensemble import load_stacking

            st = load_stacking()
            ensemble_info = {
                "models": ["mo-thecreator/Deepfake-audio-detection",
                           "Bisher/wav2vec2_ASV_deepfake_audio_detection",
                           "rafnet (existing branches)"],
                "stacking": "models/stacking.json (VAL-fit, frozen)",
                "enabled": antispoof.enabled(),
                "status": antispoof.status(),
                "taus": [st.tau_low, st.tau_high] if st else None,
            }
        except Exception:
            logger.warning("model_info ensemble unreadable", exc_info=True)
        return ModelInfo(
            version=getattr(getattr(self.model, "_pipeline", None), "model_version", "rafnet-1.0.0-phase19"),
            checkpoints=files or {"note": "Not available from supplied model metadata"},
            thresholds={"tau_low": tau_low, "tau_high": tau_high},
            mock_mode=self.mock_mode,
            ensemble=ensemble_info,
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

    def layer_evaluation(self) -> dict:
        """Measured layer evaluation (tools/evaluate_layers.py output, if present)."""
        path = _reports_dir() / "layer_eval.json"
        if not path.is_file():
            return {"evaluated": False, "report": None}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Could not parse %s: %s", path, exc)
            return {"evaluated": False, "report": None}
        return {"evaluated": True, "report": data if isinstance(data, dict) else {"report": data}}

    def origin_evaluation(self) -> dict:
        """Measured origin-model evaluation (tools/evaluate_model.py output).

        Returns the saved report verbatim — including its UNEVALUATED state
        when no labeled data exists. Never synthesized. Phase 5: also
        recognizes the stacked-ensemble schema (TEST/my_samples/per_group).
        """
        path = _reports_dir() / "model_eval.json"
        if not path.is_file():
            return {"evaluated": False, "report": None}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Could not parse %s: %s", path, exc)
            return {"evaluated": False, "report": None}
        if not isinstance(data, dict):
            return {"evaluated": False, "report": None}
        # Ensemble schema (Phase 4): test/my_samples blocks with n counts.
        test = data.get("test") or {}
        my = data.get("my_samples") or {}
        if isinstance(test, dict) and isinstance(my, dict) and (
                test.get("n", 0) or my.get("n", 0)):
            return {"evaluated": True, "report": data}
        evaluated = bool(data.get("files_scored")) and not data.get("unevaluated")
        return {"evaluated": evaluated, "report": data if isinstance(data, dict) else {"report": data}}
