"""Risk engine — separate from model confidence.

The model outputs P(synthetic) + confidence. The SOC risk score (0-100) and
level (LOW/MEDIUM/HIGH/CRITICAL) are a *security* interpretation owned by this
module, never claimed as model output. Thresholds are configurable via env.
See RISK_ENGINE.md.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


RISK_THRESHOLDS = {
    "high_confidence": _env_float("RISK_HIGH_CONFIDENCE", 0.85),
    "medium_confidence": _env_float("RISK_MEDIUM_CONFIDENCE", 0.60),
    "critical_score": _env_float("RISK_CRITICAL_SCORE", 85),
    "high_score": _env_float("RISK_HIGH_SCORE", 65),
    "medium_score": _env_float("RISK_MEDIUM_SCORE", 35),
}


@dataclass
class RiskResult:
    risk_score: int  # 0-100
    risk_level: str  # LOW | MEDIUM | HIGH | CRITICAL
    reasons: list


def compute_risk(
    prediction: str,
    confidence: float,
    duration_seconds: float = 0.0,
    repeat_detections: int = 0,
) -> RiskResult:
    """Transparent, documented heuristic. Tune thresholds via env, not code."""
    reasons: list[str] = []
    base = confidence * 100 if prediction == "SYNTHETIC" else (
        (1 - confidence) * 20 if prediction == "REAL" else 50.0
    )
    reasons.append(f"base={base:.1f} from {prediction}@{confidence:.2f}")
    if prediction == "UNCERTAIN":
        reasons.append("uncertain verdict adds +10 pending review")
        base += 10
    if duration_seconds > 30:
        base += 5
        reasons.append("long session (>30s) +5")
    if repeat_detections > 0:
        bump = min(15, repeat_detections * 5)
        base += bump
        reasons.append(f"repeat detections x{repeat_detections} +{bump}")
    score = int(min(100, max(0, round(base))))
    if score >= RISK_THRESHOLDS["critical_score"] or (prediction == "SYNTHETIC" and confidence >= 0.95):
        level = "CRITICAL"
    elif score >= RISK_THRESHOLDS["high_score"] or (prediction == "SYNTHETIC" and confidence >= RISK_THRESHOLDS["high_confidence"]):
        level = "HIGH"
    elif score >= RISK_THRESHOLDS["medium_score"] or prediction == "UNCERTAIN":
        level = "MEDIUM"
    else:
        level = "LOW"
    return RiskResult(risk_score=score, risk_level=level, reasons=reasons)
