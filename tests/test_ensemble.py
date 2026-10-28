"""Unit tests for the Phase-3 ensemble gate (no model downloads)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rafnet.soc.ensemble import (  # noqa: E402
    StackingParams,
    alert_for,
    calibrated_pai,
    load_stacking,
)

PARAMS = StackingParams(coef=[3.4875, 1.3679, 0.0719], intercept=-1.4734,
                        temperature=0.5, tau_low=0.05, tau_high=0.55)


def test_stacking_file_loads():
    p = load_stacking()
    assert p is not None
    assert len(p.coef) == 3


def test_whatsapp_proxy_real_never_critical():
    # Opus-24k transcoded clean public-real scores (measured Phase 1):
    # mo≈0.0001, bisher≈0.0021, rafnet≈0.84 (its constant-high bias).
    s = calibrated_pai(0.0001, 0.0021, 0.84, PARAMS)
    assert s["p_cal"] is not None and s["p_cal"] < 0.55
    r = alert_for(0.0001, 0.0021, 0.84, window=[], is_live=False, params=PARAMS)
    assert r.alert != "CRITICAL"
    # even with a full live window of low scores: no CRITICAL
    r2 = alert_for(0.0001, 0.0021, 0.84, window=[0.05] * 5, is_live=True, params=PARAMS)
    assert r2.alert != "CRITICAL"


def test_single_fake_clip_never_critical():
    r = alert_for(0.99, 0.99, 0.85, window=[], is_live=False, params=PARAMS)
    assert r.verdict == "SYNTHETIC"
    assert r.alert == "HIGH"  # capped: no window, no CRITICAL


def test_full_window_fake_is_critical():
    r = alert_for(0.99, 0.99, 0.85, window=[0.97] * 5, is_live=True, params=PARAMS)
    assert r.alert == "CRITICAL"


def test_single_model_high_no_critical():
    # only 1 of 3 votes fake but p_cal high: agreement bar blocks CRITICAL
    r = alert_for(0.99, 0.01, 0.01, window=[0.99] * 5, is_live=True, params=PARAMS)
    assert r.alert != "CRITICAL"


def test_missing_stacking_disables_critical():
    from rafnet.soc.ensemble import load_stacking as _ls
    assert _ls("nonexistent_stacking.json") is None
    s = calibrated_pai(0.99, 0.99, 0.85, None)
    assert s["p_cal"] is None and s["agreement"] is None
