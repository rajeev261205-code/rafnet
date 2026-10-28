"""Frozen anti-spoof ensemble scorers (additive, Phase 5).

Loads each Hugging Face audio-classification checkpoint ONCE (eval mode,
torch.inference_mode at call time), exposes per-model P(AI) for a 16 kHz
mono clip. Never raises into the request path: unavailable models yield
None scores with a reason, and the UI renders "Not available".

Env: ANTISPOOF_ENABLE=false disables HF scoring (RAFNet-only, stated).
"""
from __future__ import annotations

import logging
import os
import threading
import time

import numpy as np

logger = logging.getLogger(__name__)

MODELS = (
    "mo-thecreator/Deepfake-audio-detection",
    "Bisher/wav2vec2_ASV_deepfake_audio_detection",
)
SHORT = {
    "mo-thecreator/Deepfake-audio-detection": "mo",
    "Bisher/wav2vec2_ASV_deepfake_audio_detection": "bisher",
}
MAX_SECONDS = 6

_lock = threading.Lock()
_loaded: dict = {}
_load_error: dict = {}


def enabled() -> bool:
    return os.getenv("ANTISPOOF_ENABLE", "true").lower() == "true"


def _cap_audio(samples, sample_rate: int) -> np.ndarray:
    arr = np.asarray(samples, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr.mean(axis=1).astype(np.float32)
    arr = arr.ravel().astype(np.float32)
    if int(sample_rate) != 16000 and arr.size:
        dur = arr.size / float(sample_rate)
        tgt = max(1, int(min(dur, float(MAX_SECONDS)) * 16000))
        arr = np.interp(np.linspace(0.0, 1.0, num=tgt),
                        np.linspace(0.0, 1.0, num=arr.size), arr).astype(np.float32)
    else:
        arr = arr[: 16000 * MAX_SECONDS]
    return arr


def _get(model_id: str):
    """Load-once (feature extractor + classification model, eval mode)."""
    if model_id in _loaded:
        return _loaded[model_id]
    with _lock:
        if model_id in _loaded:
            return _loaded[model_id]
        import torch
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

        t0 = time.perf_counter()
        feat = AutoFeatureExtractor.from_pretrained(model_id, trust_remote_code=False)
        mdl = AutoModelForAudioClassification.from_pretrained(model_id, trust_remote_code=False)
        mdl.eval()
        labels = [str(mdl.config.id2label.get(i, mdl.config.id2label.get(str(i), f"L{i}")))
                  for i in range(mdl.config.num_labels)]
        fake_idx = next((i for i, lab in enumerate(labels)
                         if any(k in lab.lower() for k in
                                ("fake", "spoof", "synthetic", "ai", "deepfake", "aivoice"))), 0)
        _loaded[model_id] = (feat, mdl, fake_idx, labels)
        logger.info("antispoof loaded %s (fake_idx=%d) in %.1fs",
                    model_id, fake_idx, time.perf_counter() - t0)
        return _loaded[model_id]


def score_clip(model_id: str, samples, sample_rate: int) -> dict:
    """P(AI) for one clip. Never raises: returns {p_ai, latency_ms, error}."""
    import torch

    t0 = time.perf_counter()
    try:
        feat, mdl, fake_idx, _ = _get(model_id)
        arr = _cap_audio(samples, sample_rate)
        import torch.nn.functional as F

        inp = feat(arr, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.inference_mode():
            mdl.eval()
            logits = mdl(**{k: v for k, v in inp.items()
                             if k in ("input_values", "attention_mask")}).logits
            vec = logits[0] if logits.ndim == 2 else logits.reshape(-1)
            p = float(F.softmax(vec, dim=-1)[fake_idx])
        return {"p_ai": round(p, 4), "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                "error": None}
    except Exception as exc:
        _load_error[model_id] = f"{type(exc).__name__}: {str(exc)[:200]}"
        logger.warning("antispoof %s unavailable: %s", model_id, exc)
        return {"p_ai": None, "latency_ms": None, "error": _load_error[model_id][:200]}


def score_all(samples, sample_rate: int) -> dict:
    """Score every ensemble model. Keys: p_mo, p_bisher (+latencies, errors)."""
    out: dict = {}
    if not enabled():
        return {"p_mo": None, "p_bisher": None, "disabled": True,
                "reason": "ANTISPOOF_ENABLE=false (RAFNet-only)"}
    for mid in MODELS:
        r = score_clip(mid, samples, sample_rate)
        tag = SHORT[mid]
        out[f"p_{tag}"] = r["p_ai"]
        out[f"latency_{tag}_ms"] = r["latency_ms"]
        if r["error"]:
            out[f"error_{tag}"] = r["error"]
    return out


def status() -> dict:
    return {"enabled": enabled(), "models": list(MODELS),
            "loaded": sorted(_loaded.keys()),
            "errors": dict(_load_error)}
