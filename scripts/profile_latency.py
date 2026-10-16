"""Latency profile — real RAFNet inference timing (needs torch runtime).

Runs the production adapter over synthetic tones of several durations and
reports per-stage ms (from the pipeline's own latency tracker) + wall total.
Verifies the response CONTRACT on each run. Numbers are printed for the record
(AI_MODEL_SPEC.md) — never asserted against targets (CPU over budget is known).

Usage (torch venv):
  PYTHONPATH=src MOCK_INFERENCE=false python scripts/profile_latency.py
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def tone(seconds: float, sr: int = 16000):
    import numpy as np

    t = np.arange(int(seconds * sr), dtype=np.float32) / sr
    return (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def main() -> int:
    try:
        from rafnet.soc.model_adapter import RAFNetVoiceModel
        from rafnet.soc.risk_engine import compute_risk
    except ImportError as exc:
        print(f"SKIP: {exc}")
        return 3

    model = RAFNetVoiceModel(checkpoint_dir=ROOT / "experiments" / "checkpoints")
    model.load()
    if not model.is_loaded():
        print("FAIL: model did not load")
        return 2

    print(f"{'dur(s)':>7} {'pred':>10} {'conf':>7} {'total_ms':>9}  per-stage ms")
    totals: list[float] = []
    for dur in (0.5, 1.0, 2.0, 5.0):
        for rep in range(3):
            audio = tone(dur)
            result = model.predict(audio, 16000)
            assert result.prediction in ("REAL", "UNCERTAIN", "SYNTHETIC")
            assert result.mock is False
            compute_risk(result.prediction, result.confidence, result.audio_duration_seconds)
            totals.append(result.processing_time_ms)
            stages = result.debug.get("latency_ms", {})
            top = {k: round(float(v), 1) for k, v in stages.items() if isinstance(v, (int, float))}
            audio_ms = top.get("audio_encoder_ms", top.get("audio_encoder", "?"))
            print(f"{dur:7.1f} {result.prediction:>10} {result.confidence:7.3f} {result.processing_time_ms:9.1f}  audio_enc={audio_ms} total_keys={len(top)}")
    print(f"\nmedian model ms: {statistics.median(totals):.1f}  min {min(totals):.1f}  max {max(totals):.1f}")
    print("PROFILE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
