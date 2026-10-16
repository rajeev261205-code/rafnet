"""Decision-threshold config (no weight changes).

`configs/decision_thresholds.json` holds tau_low/tau_high. Defaults equal the
shipped `decision_gate.pkl` (0.3611/0.8611). `tools/evaluate_model.py
--calibrate` overwrites the taus with measured values plus provenance; the
backend applies them to the in-memory DecisionGate at startup (weights
untouched). Evaluation scripts read the same file via --thresholds.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_TAU_LOW = 0.3611111111111111
DEFAULT_TAU_HIGH = 0.861111111111111


def thresholds_path() -> Path:
    override = os.getenv("DECISION_THRESHOLDS_PATH")
    if override:
        p = Path(override)
        if not p.is_absolute():
            p = Path(__file__).resolve().parents[3] / p
        return p
    return Path(__file__).resolve().parents[3] / "configs" / "decision_thresholds.json"


def load_decision_thresholds() -> tuple[float, float, dict]:
    """Return (tau_low, tau_high, meta). Falls back to shipped defaults."""
    path = thresholds_path()
    meta: dict = {"source": "defaults (shipped decision_gate.pkl)", "path": str(path)}
    if not path.is_file():
        return DEFAULT_TAU_LOW, DEFAULT_TAU_HIGH, meta
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        low = float(data.get("tau_low", DEFAULT_TAU_LOW))
        high = float(data.get("tau_high", DEFAULT_TAU_HIGH))
        if not (0.0 <= low < high <= 1.0):
            raise ValueError(f"invalid thresholds tau_low={low} tau_high={high}")
        meta = {
            "source": "configs/decision_thresholds.json",
            "path": str(path),
            "calibrated": bool(data.get("calibrated", False)),
            "evaluation": data.get("evaluation", {}),
        }
        return low, high, meta
    except Exception as exc:
        logger.warning("Decision thresholds unreadable (%s); using defaults: %s", path, exc)
        return DEFAULT_TAU_LOW, DEFAULT_TAU_HIGH, meta


def apply_thresholds_to_gate(gate, tau_low: float, tau_high: float) -> bool:
    """Set gate thresholds without touching any learned weights. Returns True if changed."""
    try:
        cur_low = float(gate.tau_low.item()) if hasattr(gate.tau_low, "item") else float(gate.tau_low)
        cur_high = float(gate.tau_high.item()) if hasattr(gate.tau_high, "item") else float(gate.tau_high)
    except Exception:
        cur_low, cur_high = None, None
    if cur_low == tau_low and cur_high == tau_high:
        return False
    gate.set_thresholds(tau_low, tau_high)
    return True
