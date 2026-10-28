"""Actual model smoke test — loads RAFNet, runs real preprocessing+inference.

Usage:
  set MOCK_INFERENCE=false
  python scripts/test_model.py [--wav path/to/sample.wav] [--no-wavlm]

- Loads the REAL pipeline (RAFNetVoiceModel). Fails clearly if unavailable.
- If --wav is given, decodes that file; otherwise synthesizes a 1s 440Hz tone
  at 16kHz (explicitly NOT a real voice sample — only proves the path runs).
- Prints prediction, confidence, risk, latency. Exit non-zero on failure.

Requires the Linux venv or `pip install -r requirements.txt` (torch,
torchaudio, transformers, librosa, soundfile, scikit-learn). On Windows
without torch this script reports exactly what is missing.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("MOCK_INFERENCE", "false")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", default=None, help="Path to a real .wav sample")
    args = ap.parse_args()

    try:
        import numpy as np
    except ImportError:
        print("FAIL: numpy not installed. Run: pip install -r requirements.txt")
        return 2

    print("== RAFNet model smoke test ==")
    print(f"ROOT={ROOT}")

    # Checkpoint presence (static, no torch needed)
    ckpt = ROOT / "experiments" / "checkpoints"
    expected = ["classical_logistic.pkl", "quantum_branch.pkl", "gateway2.pth", "calibrator.pkl", "decision_gate.pkl"]
    missing = [f for f in expected if not (ckpt / f).exists()]
    print(f"checkpoints: {ckpt} present={[p.name for p in ckpt.glob('*')] if ckpt.is_dir() else []}")
    if missing:
        print(f"FAIL: missing checkpoints: {missing}")
        return 2

    # Audio
    sr = 16000
    if args.wav:
        from rafnet.soc.audio_io import decode_audio

        data = Path(args.wav).read_bytes()
        decoded = decode_audio(data, Path(args.wav).name)
        audio, sr = decoded.samples, decoded.sample_rate
        print(f"audio: {args.wav} sr={sr} dur={decoded.duration_seconds:.2f}s")
    else:
        print("audio: synthetic 440Hz tone (NOT a voice sample — path check only)")
        t = np.arange(sr, dtype=np.float32) / sr
        audio = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    # Real model load
    try:
        from rafnet.soc.model_adapter import RAFNetVoiceModel
        from rafnet.soc.risk_engine import compute_risk
    except ImportError as exc:
        print(f"FAIL: cannot import rafnet.soc (need PYTHONPATH=src): {exc}")
        return 2

    try:
        model = RAFNetVoiceModel(checkpoint_dir=ckpt)
        model.load()
    except Exception as exc:
        print(f"FAIL: model import/load crashed: {exc}")
        print("HINT: install torch/torchaudio/transformers/librosa/soundfile/scikit-learn,")
        print("      and allow HF download of microsoft/wavlm-base-plus.")
        return 2

    if not model.is_loaded():
        print("FAIL: model did not load (see logs). Refusing mock fallback in smoke test.")
        return 2

    t0 = time.perf_counter()
    try:
        result = model.predict(audio, sr)
    except Exception as exc:
        print(f"FAIL: inference crashed: {exc}")
        return 2
    elapsed = (time.perf_counter() - t0) * 1000
    risk = compute_risk(result.prediction, result.confidence, result.audio_duration_seconds)
    print(f"prediction : {result.prediction}")
    print(f"confidence : {result.confidence:.4f} (model, NOT risk)")
    print(f"p_synth    : {result.prob_synthetic:.4f}  p_real={result.prob_real:.4f}")
    print(f"risk       : {risk.risk_score} / {risk.risk_level} reasons={risk.reasons}")
    print(f"model      : {result.model_version} mock={result.mock}")
    print(f"latency    : model={result.processing_time_ms:.1f}ms wall={elapsed:.1f}ms")
    if result.mock:
        print("FAIL: smoke test returned mock result in production mode")
        return 2
    print("PASS: real model inference completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
