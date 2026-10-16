"""Environment fingerprint for inference determinism.

Logged once at startup and exposed via GET /api/health so any score drift
can be traced to an environment/weight change instead of mistaken for a
model change. Never includes secrets.
"""
from __future__ import annotations

import hashlib
import logging
import platform
import random
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

INFERENCE_SEED = 42

# Branches that MUST restore from shipped checkpoints. Gateway1 / audio /
# network encoders have no shipped checkpoint files (seeded random init +
# frozen WavLM) and are reported, not required.
REQUIRED_BRANCHES = ("classical", "quantum", "gateway2", "calibrator", "decision_gate")


def sha256_file(path: Path, limit_bytes: int = 128 * 1024 * 1024) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            remaining = limit_bytes
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                h.update(chunk)
                remaining -= len(chunk)
                if remaining <= 0:
                    break
        return h.hexdigest()
    except Exception:
        return None


def set_inference_determinism(seed: int = INFERENCE_SEED) -> dict[str, Any]:
    """Fixed seeds + deterministic cuDNN flags for inference.

    Called once at model load (startup), NOT per request (reseeding per
    request would perturb concurrent requests sharing global RNG state).
    Inference itself uses no RNG: no augmentation, dropout disabled via
    model.eval(), wrapped in torch.inference_mode().
    """
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    info: dict[str, Any] = {"seed": seed}
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        info["torch_manual_seed_set"] = True
        info["cudnn_deterministic"] = bool(torch.backends.cudnn.deterministic)
        info["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
        try:
            info["deterministic_algorithms"] = bool(torch.are_deterministic_algorithms_enabled())
        except Exception:
            info["deterministic_algorithms"] = False
    except Exception as exc:
        info["torch_manual_seed_set"] = False
        info["error"] = str(exc)
    return info


def _versions() -> dict[str, str]:
    out: dict[str, str] = {"python": platform.python_version(), "executable": sys.executable}
    for mod, key in (("torch", "torch"), ("numpy", "numpy"), ("sklearn", "scikit-learn")):
        try:
            m = __import__(mod)
            out[key] = str(getattr(m, "__version__", "?"))
        except Exception:
            out[key] = "missing"
    try:
        import transformers

        out["transformers"] = str(getattr(transformers, "__version__", "?"))
    except Exception:
        out["transformers"] = "missing"
    return out


def _torch_state(model: Any) -> dict[str, Any]:
    state: dict[str, Any] = {"inference_mode": True, "augmentation_at_inference": False}
    try:
        import torch

        state["cuda_available"] = bool(torch.cuda.is_available())
        state["cudnn_deterministic"] = bool(torch.backends.cudnn.deterministic)
        state["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
        try:
            state["deterministic_algorithms"] = bool(torch.are_deterministic_algorithms_enabled())
        except Exception:
            state["deterministic_algorithms"] = False
    except Exception:
        state["cuda_available"] = False
    pipe = getattr(model, "_pipeline", None)
    if pipe is None:
        state["eval"] = None
        state["dropout_training"] = None
        return state
    try:
        state["eval"] = not bool(getattr(pipe, "training", True))
        bad: list[str] = []
        import torch.nn as nn

        for name, mod in pipe.named_modules():
            if isinstance(mod, nn.Dropout) and bool(getattr(mod, "training", False)):
                bad.append(name)
        # Restored heads (quantum classical_head, origin heads) live outside
        # pipe.named_modules() when replaced post-init; check them too.
        for attr in ("quantum_branch",):
            branch = getattr(pipe, attr, None)
            head = getattr(branch, "classical_head", None)
            if head is not None:
                for name, mod in head.named_modules():
                    if isinstance(mod, nn.Dropout) and bool(getattr(mod, "training", False)):
                        bad.append(f"{attr}.classical_head.{name}")
        state["dropout_training"] = bad
    except Exception as exc:
        state["eval_error"] = str(exc)
    return state


def build_fingerprint(model: Any) -> dict[str, Any]:
    """Full fingerprint dict for startup log + /api/health."""
    from .model_adapter import MockVoiceModel  # local import: no cycle at module load

    is_mock = isinstance(model, MockVoiceModel) or getattr(model, "name", "") == "mock"
    fp: dict[str, Any] = {
        "seed": INFERENCE_SEED,
        "mock_requested": is_mock,
        "versions": _versions(),
        "device": getattr(model, "device", "?"),
    }
    pipe = getattr(model, "_pipeline", None)
    if pipe is not None:
        fp["model_version"] = getattr(pipe, "model_version", "?")
    fp["torch"] = _torch_state(model)
    branches: dict[str, Any] = {}
    status = getattr(model, "_branch_status", None)
    if isinstance(status, dict):
        branches = status
    else:
        restored = getattr(model, "_restored", []) or []
        for token in restored:
            branches[token] = {"loaded": True}
    fp["branches"] = branches
    err = getattr(model, "_load_error", None)
    if err:
        fp["load_error"] = str(err)[:500]
    return fp


def log_fingerprint(fp: dict[str, Any]) -> None:
    try:
        import json

        logger.info("environment fingerprint: %s", json.dumps(fp, sort_keys=True, default=str)[:4000])
    except Exception:
        logger.info("environment fingerprint (unserializable): %r", fp)
