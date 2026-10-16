"""Voice-origin classifier service (5-class, open-set generator attribution).

Target output per window/session:
  origin: HUMAN_LIVE | HUMAN_RECORDED_REPLAY | AI_TTS
        | AI_VOICE_CLONE_OR_CONVERSION | UNCERTAIN
  ai_source (only for AI_*): closest known generator FAMILY + confidence,
    phrased as "closest match", never as proof; "unknown generator" on
    open-set rejection (energy threshold).

This module loads `models/origin_classifier.pt` ONCE (see notebooks/ +
reports/train_plan.md for how it is produced). If the checkpoint is missing,
torch is unavailable, or ORIGIN_MODEL_DISABLE=1, `load_origin_model()`
returns None and callers must emit origin:null — the UI then shows "Origin
model not installed". No fallback scores are ever fabricated here.

torch is imported lazily so the SOC-only runtime (no torch) keeps working.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

ORIGIN_LABELS = [
    "HUMAN_LIVE",
    "HUMAN_RECORDED_REPLAY",
    "AI_TTS",
    "AI_VOICE_CLONE_OR_CONVERSION",
    "UNCERTAIN",
]
AI_ORIGINS = {"AI_TTS", "AI_VOICE_CLONE_OR_CONVERSION"}
WINDOW_SECONDS = 4.0
ENCODER_DEFAULT = "microsoft/wavlm-base-plus"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def checkpoint_path() -> Path:
    override = os.getenv("ORIGIN_CHECKPOINT_PATH")
    if override:
        p = Path(override)
        if not p.is_absolute():
            p = _repo_root() / p
        return p
    return _repo_root() / "models" / "origin_classifier.pt"


def origin_config() -> dict[str, Any]:
    """Threshold/config file (defaults are PLACEHOLDERS until validation).

    `uncertain_below`: max-class probability below this renders UNCERTAIN.
    `energy_tau`: family-logit energy above this renders "unknown generator"
      (checkpoint value wins when present).
    """
    cfg = {"uncertain_below": 0.6, "energy_tau": None,
           "note": "PLACEHOLDER defaults — replace with validation-tuned values "
                   "from reports/train_*.md before trusting the UNCERTAIN band."}
    path = _repo_root() / "configs" / "origin_config.json"
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            cfg.update({k: data[k] for k in ("uncertain_below", "energy_tau") if k in data})
            cfg["note"] = data.get("note", cfg["note"])
        except Exception as exc:
            logger.warning("origin config unreadable (%s): %s", path, exc)
    return cfg


@dataclass
class AISource:
    family: str
    confidence: float
    wording: str = "closest match"  # NEVER proof of provenance


@dataclass
class OriginResult:
    label: str
    probabilities: dict[str, float]
    confidence: float  # calibrated max-class probability
    ai_source: AISource | None
    reasons: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    window_seconds_used: float = WINDOW_SECONDS
    model_version: str = "origin-0.0.0-missing"
    stage_timings_ms: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "probabilities": self.probabilities,
            "confidence": self.confidence,
            "ai_source": None if self.ai_source is None else {
                "family": self.ai_source.family,
                "confidence": self.ai_source.confidence,
                "wording": self.ai_source.wording,
            },
            "reasons": self.reasons,
            "limitations": self.limitations,
            "window_seconds_used": self.window_seconds_used,
            "model_version": self.model_version,
            "stage_timings_ms": self.stage_timings_ms,
        }


class OriginModel:
    """Loaded 5-class origin classifier (encoder + MLP heads + calibration)."""

    def __init__(self, checkpoint: dict, encoder_id: str = ENCODER_DEFAULT,
                 device: str = "cpu") -> None:
        self.device = device
        self.encoder_id = encoder_id
        self._encoder = None  # built lazily on first embed() (fast import/startup)
        self.classes: list[str] = list(checkpoint.get("classes_origin", ORIGIN_LABELS))
        self.families: list[str] = list(checkpoint.get("families", []))
        self.temperature = float(checkpoint.get("temperature", 1.0))
        self.energy_tau = checkpoint.get("energy_tau")
        self.model_version = str(checkpoint.get("model_version", "origin-1.0.0"))
        self.card = dict(checkpoint.get("model_card", {}))
        self._torch = None
        self.origin_head = self._build_head(768, len(self.classes), checkpoint["origin"])
        self.family_head = self._build_head(768, len(self.families), checkpoint["family"])

    def _build_head(self, d_in: int, n_cls: int, state: dict):
        import torch.nn as nn

        head = nn.Sequential(
            nn.Linear(d_in, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Dropout(0.2), nn.Linear(256, n_cls),
        )
        head.load_state_dict(state)
        return head.to(self.device).eval()

    @property
    def available(self) -> bool:
        return True

    def _get_torch(self):
        if self._torch is None:
            import torch

            self._torch = torch
        return self._torch

    def embed(self, window_16k: np.ndarray) -> np.ndarray:
        torch = self._get_torch()
        if self._encoder is None:
            from transformers import WavLMModel

            self._encoder = WavLMModel.from_pretrained(self.encoder_id).to(self.device).eval()
        t = torch.from_numpy(np.asarray(window_16k, dtype=np.float32)).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            h = self._encoder(t).last_hidden_state
            return h.mean(dim=1).squeeze(0).cpu().numpy().astype(np.float32)

    def predict(self, samples: Any, sample_rate: int = 16000) -> OriginResult:
        import torch.nn.functional as F

        from .preprocess import fixed_window, to_model_input

        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        x = to_model_input(samples, sample_rate)
        timings["preprocess_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        window = fixed_window(x, WINDOW_SECONDS)
        used = round(min(len(x) / 16000.0, WINDOW_SECONDS), 2)

        t0 = time.perf_counter()
        emb = self.embed(window)
        timings["encode_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        torch = self._get_torch()
        t = torch.from_numpy(emb).unsqueeze(0).to(self.device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            z = self.origin_head(t) / max(1e-3, self.temperature)
            probs = F.softmax(z, dim=1).squeeze(0).cpu().numpy()
            zf = self.family_head(t)
            fam_probs = F.softmax(zf, dim=1).squeeze(0).cpu().numpy()
            energy = float(-torch.logsumexp(zf, dim=1).item())
        timings["heads_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        order = np.argsort(-probs)
        prob_map = {c: round(float(probs[self.classes.index(c)]), 4)
                    for c in self.classes if c in ORIGIN_LABELS or c in self.classes}
        top = self.classes[int(order[0])]
        conf = round(float(probs[int(order[0])]), 4)
        cfg = origin_config()
        label = top
        reasons = [f"top class {top} at calibrated p={conf}"]
        if conf < float(cfg.get("uncertain_below", 0.6)):
            label = "UNCERTAIN"
            reasons.append(
                f"max probability {conf} below placeholder band "
                f"{cfg.get('uncertain_below')} (untuned — see limitations)")

        ai_source = None
        if label in AI_ORIGINS and self.families:
            tau = self.energy_tau if self.energy_tau is not None else cfg.get("energy_tau")
            fi = int(np.argmax(fam_probs))
            fam_conf = round(float(fam_probs[fi]), 4)
            fam = self.families[fi]
            if tau is not None and energy > float(tau):
                ai_source = AISource(family="unknown generator", confidence=round(1.0 - fam_conf, 4))
                reasons.append(f"family energy {energy:.2f} above open-set tau {tau} → unknown generator")
            else:
                ai_source = AISource(family=fam, confidence=fam_conf)
                reasons.append(f"closest generator family match: {fam} ({fam_conf}) — match, not proof")

        limitations = [
            "Origin head trained per reports/train_plan.md; see checkpoint model card for data/splits.",
            "UNCERTAIN band is a placeholder until validation tuning (configs/origin_config.json).",
            "Generator families are closest matches with open-set rejection, never provenance proof.",
        ]
        if self.card:
            limitations.append(f"model card: {json.dumps(self.card)[:300]}")
        return OriginResult(
            label=label, probabilities=prob_map, confidence=conf, ai_source=ai_source,
            reasons=reasons, limitations=limitations, window_seconds_used=used,
            model_version=self.model_version, stage_timings_ms=timings,
        )


def load_origin_model() -> OriginModel | None:
    """Load once at startup. Returns None (origin:null downstream) when the
    checkpoint is missing, torch is unavailable, or ORIGIN_MODEL_DISABLE=1."""
    if os.getenv("ORIGIN_MODEL_DISABLE", "false").lower() == "true":
        logger.info("ORIGIN_MODEL_DISABLE=1 — origin model disabled (origin:null)")
        return None
    path = checkpoint_path()
    if not path.is_file():
        logger.info("Origin checkpoint missing (%s) — origin:null", path)
        return None
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        logger.warning("torch unavailable — origin:null (%s)", exc)
        return None
    try:
        import torch as _torch

        checkpoint = _torch.load(str(path), map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict) or "origin" not in checkpoint or "family" not in checkpoint:
            logger.warning("Origin checkpoint %s lacks heads — origin:null", path)
            return None
        model = OriginModel(checkpoint, device=os.getenv("ORIGIN_DEVICE", "cpu"))
        logger.info("OriginModel loaded (%s, %s)", path, model.model_version)
        return model
    except Exception as exc:
        logger.warning("Origin checkpoint unloadable — origin:null (%s)", exc)
        return None
