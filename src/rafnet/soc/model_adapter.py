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
        try:
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
            self._loaded = True
            logger.info(
                "RAFNetVoiceModel loaded in %.1fms (device=%s, restored=%s)",
                (time.perf_counter() - t0) * 1000,
                self.device,
                restored,
            )
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
        Missing files degrade gracefully (branches stay unfitted -> 0.5/UNCERTAIN).
        """
        import pickle

        restored: list[str] = []

        def _first(*names: str) -> Path | None:
            for name in names:
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
                restored.append(f"classical:{classical_path.name}")
            except Exception as exc:
                logger.warning("Classical restore skipped (%s): %s", classical_path.name, exc)

        # Quantum branch (PCA + fallback head)
        quantum_path = _first("quantum_branch.pkl", "quantum*.pkl")
        if quantum_path is not None:
            try:
                from rafnet.quantum import QuantumBranch

                qbranch = QuantumBranch.load(str(quantum_path))
                _pca_compat_shim(qbranch)
                self._pipeline.quantum_branch = qbranch
                restored.append(f"quantum:{quantum_path.name}")
            except Exception as exc:
                logger.warning("Quantum restore skipped (%s): %s", quantum_path.name, exc)

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
            except Exception as exc:
                logger.warning("Gateway2 restore skipped (%s): %s", gateway2_path.name, exc)

        # Calibrator (whole module; only useful when it reports fitted)
        calibrator_path = _first("calibrator.pkl")
        if calibrator_path is not None:
            try:
                with open(calibrator_path, "rb") as f:
                    calibrator = pickle.load(f)
                if bool(getattr(calibrator, "fitted", False)):
                    self._pipeline.calibrator = calibrator
                    restored.append(f"calibrator:{calibrator_path.name}")
                else:
                    logger.warning("Calibrator pickle reports unfitted; skipping")
            except Exception as exc:
                logger.warning("Calibrator restore skipped (%s): %s", calibrator_path.name, exc)

        # Decision gate (whole module; thresholds come from the pickle)
        gate_path = _first("decision_gate.pkl")
        if gate_path is not None:
            try:
                with open(gate_path, "rb") as f:
                    gate = pickle.load(f)
                self._pipeline.decision_gate = gate
                restored.append(f"decision_gate:{gate_path.name}")
            except Exception as exc:
                logger.warning("Decision gate restore skipped (%s): %s", gate_path.name, exc)

        return restored

    def is_loaded(self) -> bool:
        return self._loaded and self._pipeline is not None

    def preprocess(self, audio):
        return audio

    def predict(self, audio, sample_rate: int = 16000) -> VoiceResult:
        import numpy as np
        import torch

        if not self.is_loaded():
            raise RuntimeError(f"Model not loaded: {self._load_error or 'unknown'}")
        t0 = time.perf_counter()
        arr = np.asarray(audio, dtype=np.float32).reshape(-1)
        duration = float(len(arr) / sample_rate) if sample_rate else 0.0
        if sample_rate != 16000:
            # Linear resample to the model's native 16 kHz (chunk path shares this).
            old_idx = np.linspace(0, 1, num=len(arr), dtype=np.float32)
            new_len = max(1, int(round(duration * 16000)))
            arr = np.interp(
                np.linspace(0, 1, num=new_len, dtype=np.float32), old_idx, arr
            ).astype(np.float32)
        tensor = torch.from_numpy(arr).unsqueeze(0).to(self.device)  # [1, T]
        with torch.no_grad():
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
        )


class MockVoiceModel(VoiceDetectionModel):
    """Explicit dev fallback. Flagged mock=True so UI can badge it."""

    name = "mock"

    def __init__(self) -> None:
        self._loaded = True

    def load(self) -> None:
        self._loaded = True

    def is_loaded(self) -> bool:
        return True

    def preprocess(self, audio):
        return audio

    def predict(self, audio, sample_rate: int = 16000) -> VoiceResult:
        import numpy as np

        arr = np.asarray(audio, dtype=np.float32).reshape(-1)
        duration = float(len(arr) / sample_rate) if sample_rate else 0.0
        rms = float(np.sqrt(np.mean(arr**2))) if arr.size else 0.0
        # Deterministic placeholder derived from signal energy — NOT a model.
        p_syn = min(0.95, max(0.05, 0.5 + (rms - 0.05) * 2.0))
        return VoiceResult(
            prediction="UNCERTAIN" if 0.361 < p_syn < 0.861 else ("SYNTHETIC" if p_syn >= 0.861 else "REAL"),
            confidence=max(p_syn, 1 - p_syn),
            prob_synthetic=p_syn,
            prob_real=1 - p_syn,
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


def load_voice_model() -> VoiceDetectionModel:
    mock = os.getenv("MOCK_INFERENCE", "false").lower() == "true"
    if mock:
        logger.warning("MOCK_INFERENCE=true — using MockVoiceModel (dev only)")
        model: VoiceDetectionModel = MockVoiceModel()
        model.load()
        return model
    checkpoint_dir = Path(os.getenv("MODEL_CHECKPOINT_DIR", str(CHECKPOINT_DIR_DEFAULT)))
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
