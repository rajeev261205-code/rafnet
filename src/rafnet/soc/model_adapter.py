"""Model adapter — `VoiceDetectionModel` interface bound to the ACTUAL RAFNet pipeline.

- Production path: `RAFNetVoiceModel` wraps `InferencePipeline` (audio-only by
  default, since file upload has no network telemetry) and loads the real
  checkpoints from `experiments/checkpoints/`.
- Dev fallback: `MockVoiceModel` returns a clearly-flagged deterministic
  response and must only be used with `MOCK_INFERENCE=true`.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Checkpoints actually shipped with the project (see AI_MODEL_SPEC.md)
CHECKPOINT_DIR_DEFAULT = Path(__file__).resolve().parents[3] / "experiments" / "checkpoints"
MODEL_VERSION = "rafnet-1.0.0-phase19"


@dataclass
class VoiceResult:
    prediction: str  # REAL | UNCERTAIN | SYNTHETIC (from DecisionGate)
    confidence: float  # max(p, 1-p) from calibrated synthetic probability
    prob_synthetic: float
    prob_real: float
    classical_probability: float | None = None
    quantum_probability: float | None = None
    fused_probability: float | None = None
    calibrated_probability: float | None = None
    audio_reliability: float | None = None
    network_reliability: float | None = None
    gateway1_audio_weight: float | None = None
    gateway1_network_weight: float | None = None
    gateway2_classical_weight: float | None = None
    gateway2_quantum_weight: float | None = None
    model_version: str = MODEL_VERSION
    processing_time_ms: float = 0.0
    audio_duration_seconds: float = 0.0
    mock: bool = False
    debug: dict = field(default_factory=dict)
    # Measured per-stage timings in ms (pipeline latency_ms passthrough).
    stage_timings_ms: dict = field(default_factory=dict)


class VoiceDetectionModel:
    """Interface — backend routes depend on this, never on torch directly."""

    name = "base"

    def load(self) -> None:
        raise NotImplementedError

    def is_loaded(self) -> bool:
        raise NotImplementedError

    def preprocess(self, audio: Any) -> Any:
        raise NotImplementedError

    def predict(self, audio: Any, sample_rate: int = 16000) -> VoiceResult:
        raise NotImplementedError


class RAFNetVoiceModel(VoiceDetectionModel):
    """Production adapter around the real RAFNet InferencePipeline."""

    name = "rafnet"

    def __init__(
        self,
        checkpoint_dir: Path | str = CHECKPOINT_DIR_DEFAULT,
        device: str = "cpu",
        use_handcrafted_fallback: bool = True,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = device
        self._pipeline: Any = None
        self._loaded = False
        self._load_error: str | None = None

    def load(self) -> None:
        t0 = time.perf_counter()
        self._branch_status: dict = {}
        self._restored: list[str] = []
        try:
            from .fingerprint import REQUIRED_BRANCHES, log_fingerprint, set_inference_determinism

            set_inference_determinism()
            from rafnet.config.schema import RAFNetConfig
            from rafnet.inference.pipeline import InferencePipeline

            config = RAFNetConfig()
            config.device = self.device
            self._pipeline = InferencePipeline(config=config, device=self.device)
            # Restore the shipped branch-level checkpoints into the pipeline.
            # (predict_audio_only sets Gateway1 fixed_audio ablation itself.)
            restored: list[str] = []
            for candidate in (self.checkpoint_dir, self.checkpoint_dir / "checkpoints"):
                if candidate.is_dir():
                    restored = self._restore_from_dir(candidate)
                    if restored:
                        break
            self._restored = restored
            # Loud fail on partial model: every REQUIRED branch must restore.
            # Never silently serve 0.5/UNCERTAIN from unfitted branches.
            missing = [b for b in REQUIRED_BRANCHES if not bool((self._branch_status.get(b) or {}).get("loaded"))]
            if missing:
                self._loaded = False
                self._load_error = (
                    f"branch(es) failed to load: {', '.join(missing)} "
                    f"(checkpoint_dir={self.checkpoint_dir}; set MOCK_INFERENCE=true explicitly for dev)"
                )
                logger.error("%s", self._load_error)
                try:
                    fp = self.fingerprint()
                    log_fingerprint(fp)
                except Exception:
                    pass
                return
            # Measured thresholds live in configs/decision_thresholds.json
            # (weights untouched — only the DecisionGate taus are set).
            try:
                from .thresholds import apply_thresholds_to_gate, load_decision_thresholds

                tau_low, tau_high, meta = load_decision_thresholds()
                if apply_thresholds_to_gate(self._pipeline.decision_gate, tau_low, tau_high):
                    logger.info(
                        "Decision thresholds from %s: tau_low=%.4f tau_high=%.4f",
                        meta.get("source"), tau_low, tau_high,
                    )
                    restored = [*restored, f"thresholds:{tau_low:.4f}/{tau_high:.4f}"]
            except Exception as exc:
                logger.warning("Threshold config skipped (non-fatal): %s", exc)
            self._pipeline.eval()
            self._loaded = True
            logger.info(
                "RAFNetVoiceModel loaded in %.1fms (device=%s, restored=%s)",
                (time.perf_counter() - t0) * 1000,
                self.device,
                restored,
            )
            try:
                from .fingerprint import log_fingerprint

                log_fingerprint(self.fingerprint())
            except Exception as exc:
                logger.warning("fingerprint log skipped: %s", exc)
        except Exception as exc:  # never crash import; report via health
            self._loaded = False
            self._load_error = str(exc)
            logger.exception("RAFNetVoiceModel load failed: %s", exc)

    def _restore_from_dir(self, directory: Path) -> list[str]:
        """Restore shipped checkpoints using each branch's real loader.

        - classical_logistic.pkl -> ClassicalBranch.load (fitted sklearn branch)
        - quantum_branch.pkl    -> QuantumBranch.load (fitted PCA + fallback head)
        - gateway2.pth          -> Gateway2 gate_network state subset (strict=False)
        - calibrator.pkl        -> whole-module Calibrator (must report fitted)
        - decision_gate.pkl     -> whole-module DecisionGate (tau from pickle)
        Any REQUIRED branch that is missing or fails is recorded in
        self._branch_status and causes load() to fail loudly (never a silent
        partial model serving 0.5/UNCERTAIN).
        """
        import pickle

        from .fingerprint import sha256_file

        restored: list[str] = []
        status: dict = getattr(self, "_branch_status", {})
        self._branch_status = status

        def _record(branch: str, path: Path | None, ok: bool, detail: str | None = None) -> None:
            entry: dict = {"loaded": bool(ok)}
            if path is not None:
                entry["file"] = path.name
                entry["path"] = str(path)
                try:
                    entry["bytes"] = path.stat().st_size
                except Exception:
                    pass
                digest = sha256_file(path)
                if digest:
                    entry["sha256"] = digest
            if detail:
                entry["detail"] = detail
            status[branch] = entry

        def _first(*names: str) -> Path | None:
            import glob as _glob

            for name in names:
                if any(ch in name for ch in ("*", "?", "[")):
                    hits = sorted(_glob.glob(str(directory / name)))
                    if hits:
                        return Path(hits[0])
                    continue
                path = directory / name
                if path.is_file():
                    return path
            return None

        # Classical branch (fitted)
        classical_path = _first("classical_logistic.pkl", "classical*.pkl")
        if classical_path is not None:
            try:
                from rafnet.classical import ClassicalBranch

                branch = ClassicalBranch.load(str(classical_path))
                _sklearn_compat_shim(branch)
                self._pipeline.classical_branch = branch
                fitted = bool(getattr(branch, "is_fitted", False))
                restored.append(f"classical:{classical_path.name}")
                _record("classical", classical_path, fitted, "fitted" if fitted else "unfitted after load")
                if not fitted:
                    logger.error("Classical branch unfitted after load (%s)", classical_path.name)
            except Exception as exc:
                logger.error("Classical restore failed (%s): %s", classical_path.name, exc)
                _record("classical", classical_path, False, str(exc)[:300])
        else:
            logger.error("Classical checkpoint missing in %s", directory)
            _record("classical", None, False, "missing file")

        # Quantum branch (PCA + fallback head)
        quantum_path = _first("quantum_branch.pkl", "quantum*.pkl")
        if quantum_path is not None:
            try:
                from rafnet.quantum import QuantumBranch

                qbranch = QuantumBranch.load(str(quantum_path))
                _pca_compat_shim(qbranch)
                self._pipeline.quantum_branch = qbranch
                fitted = bool(getattr(qbranch, "pca_fitted", False))
                restored.append(f"quantum:{quantum_path.name}")
                _record(
                    "quantum",
                    quantum_path,
                    fitted,
                    f"fallback={bool(getattr(qbranch, '_using_fallback', True))}",
                )
                if not fitted:
                    logger.error("Quantum branch PCA unfitted after load (%s)", quantum_path.name)
            except Exception as exc:
                logger.error("Quantum restore failed (%s): %s", quantum_path.name, exc)
                _record("quantum", quantum_path, False, str(exc)[:300])
        else:
            logger.error("Quantum checkpoint missing in %s", directory)
            _record("quantum", None, False, "missing file")

        # Gateway2 fusion weights (state subset of the shipped OrderedDict)
        gateway2_path = _first("gateway2.pth", "gateway2.pt")
        if gateway2_path is not None:
            try:
                import torch

                state = torch.load(str(gateway2_path), map_location=self.device, weights_only=False)
                if hasattr(state, "items"):
                    subset = {k: v for k, v in state.items() if k.startswith("gate_network.")}
                    if subset:
                        self._pipeline.gateway2.load_state_dict(subset, strict=False)
                        restored.append(f"gateway2:{gateway2_path.name}")
                        _record("gateway2", gateway2_path, True, f"{len(subset)} tensors")
                    else:
                        logger.error("Gateway2 checkpoint has no gate_network.* keys (%s)", gateway2_path.name)
                        _record("gateway2", gateway2_path, False, "no gate_network keys")
                else:
                    logger.error("Gateway2 checkpoint not a state dict (%s)", gateway2_path.name)
                    _record("gateway2", gateway2_path, False, "not a state dict")
            except Exception as exc:
                logger.error("Gateway2 restore failed (%s): %s", gateway2_path.name, exc)
                _record("gateway2", gateway2_path, False, str(exc)[:300])
        else:
            logger.error("Gateway2 checkpoint missing in %s", directory)
            _record("gateway2", None, False, "missing file")

        # Calibrator (whole module; only useful when it reports fitted)
        calibrator_path = _first("calibrator.pkl")
        if calibrator_path is not None:
            try:
                with open(calibrator_path, "rb") as f:
                    calibrator = pickle.load(f)
                if bool(getattr(calibrator, "fitted", False)):
                    self._pipeline.calibrator = calibrator
                    restored.append(f"calibrator:{calibrator_path.name}")
                    _record("calibrator", calibrator_path, True, "fitted")
                else:
                    logger.error("Calibrator pickle reports unfitted; failing loudly")
                    _record("calibrator", calibrator_path, False, "unfitted pickle")
            except Exception as exc:
                logger.error("Calibrator restore failed (%s): %s", calibrator_path.name, exc)
                _record("calibrator", calibrator_path, False, str(exc)[:300])
        else:
            logger.error("Calibrator checkpoint missing in %s", directory)
            _record("calibrator", None, False, "missing file")

        # Decision gate (whole module; thresholds come from the pickle)
        gate_path = _first("decision_gate.pkl")
        if gate_path is not None:
            try:
                with open(gate_path, "rb") as f:
                    gate = pickle.load(f)
                self._pipeline.decision_gate = gate
                restored.append(f"decision_gate:{gate_path.name}")
                try:
                    tl = float(gate.tau_low.item()) if hasattr(gate.tau_low, "item") else float(gate.tau_low)
                    th = float(gate.tau_high.item()) if hasattr(gate.tau_high, "item") else float(gate.tau_high)
                    detail = f"tau {tl:.4f}/{th:.4f}"
                except Exception:
                    detail = "thresholds unreadable"
                _record("decision_gate", gate_path, True, detail)
            except Exception as exc:
                logger.error("Decision gate restore failed (%s): %s", gate_path.name, exc)
                _record("decision_gate", gate_path, False, str(exc)[:300])
        else:
            logger.error("Decision gate checkpoint missing in %s", directory)
            _record("decision_gate", None, False, "missing file")

        # Fusion/Gateway1 has no shipped checkpoint: seeded random init +
        # fixed_audio ablation at inference. Reported for traceability.
        try:
            g1 = getattr(self._pipeline, "gateway1", None)
            status["fusion_gateway1"] = {
                "loaded": g1 is not None,
                "file": None,
                "detail": f"ablation={getattr(getattr(g1, 'config', None), 'ablation_mode', '?')}; no checkpoint (seeded init)",
            }
        except Exception:
            pass

        return restored

    def is_loaded(self) -> bool:
        return self._loaded and self._pipeline is not None

    def preprocess(self, audio):
        return audio

    def fingerprint(self) -> dict:
        from .fingerprint import build_fingerprint

        return build_fingerprint(self)

    def predict(self, audio, sample_rate: int = 16000) -> VoiceResult:
        import numpy as np
        import torch

        if not self.is_loaded():
            raise RuntimeError(f"Model not loaded: {self._load_error or 'unknown'}")
        t0 = time.perf_counter()
        # Deterministic inference contract: eval mode (dropout off), no
        # augmentation (soc/augment.py is train-only, never imported here),
        # shared preprocessing on EVERY call so file/live tensors match.
        try:
            self._pipeline.eval()
        except Exception:
            pass
        raw = np.asarray(audio, dtype=np.float32)
        duration = float(raw.reshape(-1).size / sample_rate) if sample_rate else 0.0
        # Always through the shared contract (mono -> 16k -> clip), even when
        # already 16 kHz: that path used to skip mono/clip, diverging from
        # live chunks (which clip in validate_chunk) by up to 0.5 on loud
        # float audio. No normalization: the model trained without it.
        from .preprocess import to_model_input

        arr = to_model_input(raw, sample_rate)
        tensor = torch.from_numpy(arr).unsqueeze(0).to(self.device)  # [1, T]
        with torch.inference_mode():
            # fixed_audio ablation + dummy network handled inside the pipeline.
            out = self._pipeline.predict_audio_only(tensor)
        d = out.to_dict() if hasattr(out, "to_dict") else dict(out)
        elapsed = (time.perf_counter() - t0) * 1000
        label = str(d.get("decision_label", "UNCERTAIN"))
        p_syn = float(d.get("probability_synthetic", 0.5))
        p_real = float(d.get("probability_real", 1.0 - p_syn))
        return VoiceResult(
            prediction=label,
            confidence=float(d.get("confidence", max(p_syn, p_real))),
            prob_synthetic=p_syn,
            prob_real=p_real,
            classical_probability=_opt_float(d.get("classical_probability")),
            quantum_probability=_opt_float(d.get("quantum_probability")),
            fused_probability=_opt_float(d.get("fused_probability")),
            calibrated_probability=_opt_float(d.get("calibrated_probability")),
            audio_reliability=_opt_float(d.get("audio_reliability")),
            network_reliability=_opt_float(d.get("network_reliability")),
            gateway1_audio_weight=_opt_float(d.get("gateway1_audio_weight")),
            gateway1_network_weight=_opt_float(d.get("gateway1_network_weight")),
            gateway2_classical_weight=_opt_float(d.get("gateway2_classical_weight")),
            gateway2_quantum_weight=_opt_float(d.get("gateway2_quantum_weight")),
            model_version=getattr(self._pipeline, "model_version", MODEL_VERSION),
            processing_time_ms=float(d.get("total_latency_ms", elapsed)),
            audio_duration_seconds=duration,
            mock=False,
            debug={
                "latency_ms": d.get("latency_ms", {}),
                "restored": getattr(self, "_restored", []),
            },
            stage_timings_ms=dict(d.get("latency_ms", {}) or {}),
        )


class MockVoiceModel(VoiceDetectionModel):
    """Explicit dev fallback. Flagged mock=True so UI can badge it.

    Intentionally NON-diagnostic: always UNCERTAIN at p=0.5. A loudness-based
    mapping previously manufactured SYNTHETIC verdicts for loud natural
    speech (and REAL for quiet input) — a dev stub must never fabricate
    detector-like outputs. Tests needing HIGH/CRITICAL use explicit stubs.
    """

    name = "mock"

    def __init__(self) -> None:
        self._loaded = True
        self.device = "cpu"
        self._pipeline = None
        self._branch_status = {"mock": {"loaded": True, "detail": "MOCK_INFERENCE=true (dev only)"}}
        self._load_error = None

    def load(self) -> None:
        self._loaded = True

    def fingerprint(self) -> dict:
        from .fingerprint import build_fingerprint

        return build_fingerprint(self)

    def is_loaded(self) -> bool:
        return True

    def preprocess(self, audio):
        return audio

    def predict(self, audio, sample_rate: int = 16000) -> VoiceResult:
        import numpy as np

        arr = np.asarray(audio, dtype=np.float32).reshape(-1)
        duration = float(len(arr) / sample_rate) if sample_rate else 0.0
        return VoiceResult(
            prediction="UNCERTAIN",
            confidence=0.5,
            prob_synthetic=0.5,
            prob_real=0.5,
            model_version="mock-0.0.0",
            processing_time_ms=1.0,
            audio_duration_seconds=duration,
            mock=True,
            debug={"note": "MOCK_INFERENCE=true — not a real prediction"},
        )


def _sklearn_compat_shim(branch: Any) -> None:
    """Compat for 1.9.1-pickled sklearn estimators running under older sklearn.

    The shipped `classical_logistic.pkl` was pickled with scikit-learn 1.9.1,
    whose `LogisticRegression` no longer sets the (removed) `multi_class`
    attribute that sklearn <= 1.6 `predict_proba` still reads. Production must
    use sklearn==1.9.1 (see requirements); this shim only lets older runtimes
    (e.g. devices whose Application Control policy blocks 1.9.1's DLLs) run
    inference. Binary problems use 'ovr' semantics, matching 1.6 defaults.
    """
    try:
        inner = getattr(getattr(branch, "model", None), "model", None)
        import sklearn  # type: ignore[import-untyped]

        if inner is not None and type(inner).__name__ == "LogisticRegression":
            if not hasattr(inner, "multi_class"):
                n_classes = len(getattr(inner, "classes_", [0, 1]))
                inner.multi_class = "ovr" if n_classes <= 2 else "multinomial"
                logger.warning(
                    "sklearn compat: set multi_class=%s on 1.9.1-pickled estimator "
                    "(running sklearn %s; prefer sklearn==1.9.1 in production)",
                    inner.multi_class,
                    getattr(sklearn, "__version__", "?"),
                )
    except Exception as exc:
        logger.warning("sklearn compat shim failed (non-fatal): %s", exc)


def _pca_compat_shim(branch: Any) -> None:
    """Restore the full fitted-PCA attribute set from components_/mean_.

    `QuantumBranch.load` persists only `pca_components`/`pca_mean` (+ n_components_),
    but sklearn's `check_is_fitted`/`transform` also touches `explained_variance_`
    etc. Re-derive the bookkeeping attributes deterministically from the shipped
    components so `transform` (which only needs components_ + mean_) runs.
    Values are algebraic consequences, not invented training statistics.
    """
    try:
        import numpy as np

        circuit = getattr(branch, "quantum_circuit", None)
        pca = getattr(circuit, "pca", None)
        if pca is None or not bool(getattr(branch, "pca_fitted", False)):
            return
        comp = getattr(pca, "components_", None)
        mean = getattr(pca, "mean_", None)
        if comp is None or mean is None:
            return
        n_comp, n_feat = np.shape(comp)
        n_samples = 250  # from the checkpoint's own training_history
        try:
            hist = getattr(branch, "_training_history", []) or []
            n_samples = int(hist[0].get("n_samples", n_samples))
        except Exception:
            pass
        if not hasattr(pca, "n_samples_"):
            pca.n_samples_ = n_samples
        if not hasattr(pca, "n_features_in_"):
            pca.n_features_in_ = int(n_feat)
        if not hasattr(pca, "singular_values_"):
            pca.singular_values_ = np.zeros(n_comp, dtype=np.float64)
        if not hasattr(pca, "explained_variance_"):
            pca.explained_variance_ = (pca.singular_values_**2) / max(1, n_samples - 1)
        if not hasattr(pca, "explained_variance_ratio_"):
            total = float(pca.explained_variance_.sum())
            pca.explained_variance_ratio_ = (
                np.zeros(n_comp, dtype=np.float64) if total == 0
                else pca.explained_variance_ / total
            )
        if not hasattr(pca, "noise_variance_"):
            pca.noise_variance_ = 0.0
        logger.warning("pca compat: completed fitted-PCA attributes from shipped components")
    except Exception as exc:
        logger.warning("pca compat shim failed (non-fatal): %s", exc)


def resolve_checkpoint_dir() -> Path:
    """Resolve MODEL_CHECKPOINT_DIR (absolute, or repo-root-relative)."""
    configured = os.getenv("MODEL_CHECKPOINT_DIR", str(CHECKPOINT_DIR_DEFAULT))
    path = Path(configured)
    if not path.is_absolute():
        # Server cwd is src/ (uvicorn rafnet.api.main:app), while .env paths
        # are written repo-root-relative — anchor them at the repo root.
        path = CHECKPOINT_DIR_DEFAULT.parent.parent / path
    return path


def load_voice_model() -> VoiceDetectionModel:
    mock = os.getenv("MOCK_INFERENCE", "false").lower() == "true"
    if mock:
        logger.warning("MOCK_INFERENCE=true — using MockVoiceModel (dev only)")
        model: VoiceDetectionModel = MockVoiceModel()
        model.load()
        return model
    checkpoint_dir = resolve_checkpoint_dir()
    device = os.getenv("RAFNET_DEVICE", "cpu")
    model = RAFNetVoiceModel(checkpoint_dir=checkpoint_dir, device=device)
    model.load()
    if not model.is_loaded():
        logger.error("Real model failed to load; refusing silent mock fallback (set MOCK_INFERENCE=true explicitly for dev)")
    return model


def _opt_float(value) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None
