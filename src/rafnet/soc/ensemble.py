"""Ensemble (stacked P(AI)) + alert gating — additive, Phase 3.

- calibrated_pai(): logistic stacking over [p_mo, p_bisher, p_rafnet] +
  temperature scaling, params from models/stacking.json (VAL-fit).
- alert_for(): CRITICAL only when ALL hold: live window full (5 chunks,
  several seconds) AND window median P(AI) >= 0.90 AND current calibrated
  P(AI) >= 0.90 AND >= 2 of 3 models vote fake at 0.5. Single-shot uploads
  can NEVER yield CRITICAL (no window). Otherwise the band verdict
  (REAL/UNCERTAIN/SYNTHETIC) from the VAL-chosen taus.
- If stacking.json is missing/unreadable: CRITICAL is disabled and the
  reason is stated (never silent).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

STACKING_DEFAULT = Path(__file__).resolve().parents[3] / "models" / "stacking.json"
STACKING4_DEFAULT = Path(__file__).resolve().parents[3] / "models" / "stack_phase4.json"
WINDOW_N = 5
P_CRITICAL = 0.90


@dataclass
class StackingParams:
    coef: list[float]
    intercept: float
    temperature: float
    tau_low: float
    tau_high: float
    source: str = ""


def load_stacking(path: str | Path = STACKING_DEFAULT) -> StackingParams | None:
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return StackingParams(coef=[float(c) for c in d["coef"]],
                              intercept=float(d["intercept"]),
                              temperature=float(d["temperature"]),
                              tau_low=float(d["tau_low"]),
                              tau_high=float(d["tau_high"]),
                              source=str(path))
    except Exception:
        return None


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-500.0, min(500.0, z))))


def calibrated_pai(p_mo: float | None, p_bisher: float | None, p_rafnet: float | None,
                   params: StackingParams | None) -> dict:
    """Stacked calibrated P(AI). Missing inputs abstain (p=None, reason)."""
    if params is None:
        return {"p_cal": None, "votes_fake": 0, "agreement": None,
                "reason": "stacking params missing (models/stacking.json unreadable)"}
    vals = [p_mo, p_bisher, p_rafnet]
    if any(v is None for v in vals):
        return {"p_cal": None, "votes_fake": sum(1 for v in vals if v is not None and v >= 0.5),
                "agreement": None, "reason": "a model score is missing — abstain"}
    z = sum(c * float(v) for c, v in zip(params.coef, vals)) + params.intercept
    p = _sigmoid(z / params.temperature)
    votes = sum(1 for v in vals if float(v) >= 0.5)
    return {"p_cal": round(p, 4), "votes_fake": votes,
            "agreement": round(votes / 3.0, 4), "reason": "stacked+temperature (VAL-fit)"}


@dataclass
class AlertResult:
    alert: str  # OK | REVIEW (uncertain) | HIGH | CRITICAL
    verdict: str  # REAL | UNCERTAIN | SYNTHETIC
    p_cal: float | None
    agreement: float | None
    reasons: list = field(default_factory=list)


def alert_for(p_mo: float | None, p_bisher: float | None, p_rafnet: float | None,
              window: list[float] | tuple = (), is_live: bool = False,
              params: StackingParams | None = None) -> AlertResult:
    """Decide alert. CRITICAL needs live full-window confirmation (never single-shot)."""
    if params is None:
        params = load_stacking()
    reasons: list[str] = []
    s = calibrated_pai(p_mo, p_bisher, p_rafnet, params)
    p = s["p_cal"]
    if p is None:
        reasons.append(s["reason"])
        return AlertResult(alert="REVIEW", verdict="UNCERTAIN", p_cal=None,
                           agreement=s["agreement"], reasons=reasons)
    assert params is not None
    if p < params.tau_low:
        verdict = "REAL"
    elif p > params.tau_high:
        verdict = "SYNTHETIC"
    else:
        verdict = "UNCERTAIN"
    reasons.append(f"p_cal={p} band=[{params.tau_low},{params.tau_high}] -> {verdict}; "
                   f"votes_fake={s['votes_fake']}/3 agreement={s['agreement']}")
    if verdict == "REAL":
        return AlertResult(alert="OK", verdict=verdict, p_cal=p, agreement=s["agreement"], reasons=reasons)
    if verdict == "UNCERTAIN":
        return AlertResult(alert="REVIEW", verdict=verdict, p_cal=p, agreement=s["agreement"], reasons=reasons)
    # SYNTHETIC branch: HIGH at most without a confirming live window.
    if not is_live or len(window) < WINDOW_N:
        reasons.append(f"single-shot or short window ({len(window)}/{WINDOW_N}) — CRITICAL needs a full live window")
        return AlertResult(alert="HIGH", verdict=verdict, p_cal=p, agreement=s["agreement"], reasons=reasons)
    med = sorted(window)[len(window) // 2]
    reasons.append(f"live window median={round(med, 4)} (n={len(window)})")
    if p >= P_CRITICAL and s["votes_fake"] >= 2 and med >= P_CRITICAL:
        reasons.append("CRITICAL: p_cal>=0.90 + >=2 models agree + window median>=0.90")
        return AlertResult(alert="CRITICAL", verdict=verdict, p_cal=p, agreement=s["agreement"], reasons=reasons)
    reasons.append("window or agreement below CRITICAL bar — HIGH, keep watching")
    return AlertResult(alert="HIGH", verdict=verdict, p_cal=p, agreement=s["agreement"], reasons=reasons)


# --- Phase 6 v2: head-aware 4-feature stack (purely additive) ---------------
# models/stack_phase4.json was fit on VAL with features [head, rafnet, mo,
# bisher]; the head feature comes from the frozen WavLM head (soc/head4.py).
# The legacy 3-feature path above is untouched, so a missing v2 artifact simply
# leaves the additive fields out (never silently degrades the old verdict).

@dataclass
class StackingParams4:
    features: list
    coef: list[float]
    intercept: float
    temperature: float
    tau_low: float
    tau_high: float
    source: str = ""


def load_stacking4(path: str | Path = STACKING4_DEFAULT) -> StackingParams4 | None:
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        feats = [str(f) for f in d["features"]]
        coef = [float(c) for c in d["coef"]]
        if len(feats) != len(coef):
            return None
        return StackingParams4(features=feats, coef=coef,
                               intercept=float(d["intercept"]),
                               temperature=float(d["temperature"]),
                               tau_low=float(d["tau_low"]),
                               tau_high=float(d["tau_high"]),
                               source=str(path))
    except Exception:
        return None


def calibrated_pai4(scores: dict, params: StackingParams4 | None) -> dict:
    """Stacked calibrated P(AI) over named features. Missing inputs abstain."""
    if params is None:
        return {"p_cal": None, "votes_fake": 0, "agreement": None,
                "reason": "stacking4 params missing (models/stack_phase4.json unreadable)"}
    vals = [scores.get(f) for f in params.features]
    n = len(params.features)
    if any(v is None for v in vals):
        return {"p_cal": None,
                "votes_fake": sum(1 for v in vals if v is not None and float(v) >= 0.5),
                "agreement": None, "reason": "a stacked feature is missing — abstain"}
    z = sum(c * float(v) for c, v in zip(params.coef, vals)) + params.intercept
    p = _sigmoid(z / params.temperature)
    votes = sum(1 for v in vals if float(v) >= 0.5)
    return {"p_cal": round(p, 4), "votes_fake": votes,
            "agreement": round(votes / float(n), 4),
            "reason": "stacked v2 (head+rafnet+mo+bisher) + temperature (VAL-fit)"}


def alert_for4(scores: dict, window: list[float] | tuple = (),
               is_live: bool = False, params: StackingParams4 | None = None) -> AlertResult:
    """Head-aware alert. Same CRITICAL rule as v1 (live window + >=2 votes)."""
    if params is None:
        params = load_stacking4()
    reasons: list[str] = []
    n = len(params.features) if params is not None else 4
    s = calibrated_pai4(scores, params)
    p = s["p_cal"]
    if p is None:
        reasons.append(s["reason"])
        return AlertResult(alert="REVIEW", verdict="UNCERTAIN", p_cal=None,
                           agreement=s["agreement"], reasons=reasons)
    assert params is not None
    if p < params.tau_low:
        verdict = "REAL"
    elif p > params.tau_high:
        verdict = "SYNTHETIC"
    else:
        verdict = "UNCERTAIN"
    reasons.append(f"p_cal_v2={p} band=[{params.tau_low},{params.tau_high}] -> {verdict}; "
                   f"votes_fake={s['votes_fake']}/{n} agreement={s['agreement']}")
    if verdict == "REAL":
        return AlertResult(alert="OK", verdict=verdict, p_cal=p, agreement=s["agreement"], reasons=reasons)
    if verdict == "UNCERTAIN":
        return AlertResult(alert="REVIEW", verdict=verdict, p_cal=p, agreement=s["agreement"], reasons=reasons)
    if not is_live or len(window) < WINDOW_N:
        reasons.append(f"single-shot or short window ({len(window)}/{WINDOW_N}) — CRITICAL needs a full live window")
        return AlertResult(alert="HIGH", verdict=verdict, p_cal=p, agreement=s["agreement"], reasons=reasons)
    med = sorted(window)[len(window) // 2]
    reasons.append(f"live window median={round(med, 4)} (n={len(window)})")
    if p >= P_CRITICAL and s["votes_fake"] >= 2 and med >= P_CRITICAL:
        reasons.append("CRITICAL: p_cal>=0.90 + >=2 features agree + window median>=0.90")
        return AlertResult(alert="CRITICAL", verdict=verdict, p_cal=p, agreement=s["agreement"], reasons=reasons)
    reasons.append("window or agreement below CRITICAL bar — HIGH, keep watching")
    return AlertResult(alert="HIGH", verdict=verdict, p_cal=p, agreement=s["agreement"], reasons=reasons)
