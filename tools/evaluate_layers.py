"""Evaluate detection layers against labeled samples (no fake data).

Layout: samples/<class>/*.wav with classes:
  human_live, human_replayed (played through speakers and re-recorded),
  tts, voice_changed.

For every decodable file this runs the real heuristic layers (numpy) and, when
the RAFNet model loads, the neural layer + verdict fusion. It writes:
  reports/layer_eval.json  (machine-readable)
  reports/layer_eval.md    (human-readable confusion matrix + per-layer accuracy)

Evaluation-time thresholds (NOT detector claims, documented in the report):
  Layer 1 score >= 50  -> "synthetic-like" (expected for tts, voice_changed)
  Layer 3 score >= 60  -> "replay-like"   (expected for human_replayed)
  verdict label == expected class mapping (see EXPECTED_VERDICT)

Files that fail to decode are recorded as errors, never silently dropped.
If samples/ is missing or empty, an explicit unevaluated report is written.

Usage (repo root):
  .venv/Scripts/python tools/evaluate_layers.py [--samples samples] [--reports reports]
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

CLASSES = ("human_live", "human_replayed", "tts", "voice_changed")
EXPECTED_VERDICT = {
    "human_live": "HUMAN_LIVE",
    "human_replayed": "REPLAY_PRERECORDED",
    "tts": "AI_SYNTHETIC",
    "voice_changed": "AI_SYNTHETIC",
}
L1_SYNTHETIC_AT = 50.0
L3_REPLAY_AT = 60.0
SYNTHETIC_CLASSES = ("tts", "voice_changed")


def evaluate_file(path: Path, cls: str, model) -> dict:
    from rafnet.soc import forensics
    from rafnet.soc.audio_io import decode_audio

    entry: dict = {"file": str(path), "class": cls, "expected_verdict": EXPECTED_VERDICT[cls]}
    try:
        decoded = decode_audio(path.read_bytes(), path.name)
    except Exception as exc:
        entry["error"] = f"decode failed: {exc}"
        return entry
    entry["duration_seconds"] = round(decoded.duration_seconds, 2)
    entry["sample_rate_hz"] = decoded.sample_rate
    try:
        l1 = forensics.spectral_layer(decoded.samples, decoded.sample_rate)
        l3 = forensics.liveness_layer(decoded.samples, decoded.sample_rate)
    except Exception:
        entry["error"] = f"forensics failed: {traceback.format_exc(limit=3)}"
        return entry
    entry["layer1_score"] = l1.get("score")
    entry["layer3_score"] = l3.get("score")
    entry["layer1_evidence"] = l1.get("evidence", [])
    entry["layer3_evidence"] = l3.get("evidence", [])
    # Per-layer correctness under the documented evaluation thresholds.
    if l1.get("score") is not None:
        entry["layer1_correct"] = (l1["score"] >= L1_SYNTHETIC_AT) == (cls in SYNTHETIC_CLASSES)
    if l3.get("score") is not None:
        entry["layer3_correct"] = (l3["score"] >= L3_REPLAY_AT) == (cls == "human_replayed")
    if model is not None:
        try:
            result = model.predict(decoded.samples, decoded.sample_rate)
            neural = forensics.neural_layer(result)
            verdict = forensics.fuse_verdict(
                l1, neural, l3,
                {"risk_score": 0, "risk_level": "LOW"},
                duration_seconds=decoded.duration_seconds,
            )
            entry["neural_prediction"] = neural["metrics"].get("prediction")
            entry["neural_confidence"] = neural["metrics"].get("confidence")
            entry["verdict"] = verdict["label"]
            entry["verdict_correct"] = verdict["label"] == EXPECTED_VERDICT[cls]
        except Exception:
            entry["error"] = f"model inference failed: {traceback.format_exc(limit=3)}"
    return entry


def load_model():
    try:
        from rafnet.soc.model_adapter import load_voice_model

        model = load_voice_model()
        if model.is_loaded():
            return model, {"loaded": True, "name": getattr(model, "name", "unknown")}
        return None, {"loaded": False, "reason": "model did not load (checkpoints missing?)"}
    except Exception as exc:
        return None, {"loaded": False, "reason": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate RAFNet layers on labeled samples.")
    ap.add_argument("--samples", default="samples")
    ap.add_argument("--reports", default="reports")
    args = ap.parse_args()

    samples_dir = (ROOT / args.samples) if not Path(args.samples).is_absolute() else Path(args.samples)
    reports_dir = (ROOT / args.reports) if not Path(args.reports).is_absolute() else Path(args.reports)
    reports_dir.mkdir(parents=True, exist_ok=True)

    model, model_info = load_model()
    files: list[dict] = []
    for cls in CLASSES:
        cls_dir = samples_dir / cls
        if not cls_dir.is_dir():
            files.append({"class": cls, "missing_dir": str(cls_dir)})
            continue
        wavs = sorted([p for p in cls_dir.glob("*.wav") if p.is_file()])
        if not wavs:
            files.append({"class": cls, "note": "no .wav files"})
            continue
        for wav in wavs:
            files.append(evaluate_file(wav, cls, model))

    evaluated = [f for f in files if "error" not in f and "missing_dir" not in f and "note" not in f]
    l1_hits = [f for f in evaluated if "layer1_correct" in f]
    l3_hits = [f for f in evaluated if "layer3_correct" in f]
    verdict_hits = [f for f in evaluated if "verdict_correct" in f]

    def accuracy(rows: list[dict], key: str) -> dict:
        if not rows:
            return {"evaluated": 0, "accuracy": None, "note": "not evaluated"}
        correct = sum(1 for r in rows if r.get(key))
        return {"evaluated": len(rows), "correct": correct, "accuracy": round(correct / len(rows), 4)}

    labels = ["HUMAN_LIVE", "AI_SYNTHETIC", "REPLAY_PRERECORDED", "EXTERNAL_APP_SUSPECTED", "UNCERTAIN", "unevaluated"]
    confusion = {true: {lab: 0 for lab in labels} for true in CLASSES}
    for f in evaluated:
        true = f["class"]
        pred = f.get("verdict", "unevaluated")
        confusion[true][pred if pred in labels else "unevaluated"] += 1

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "classes": list(CLASSES),
        "expected_verdict": EXPECTED_VERDICT,
        "thresholds": {
            "layer1_synthetic_at": L1_SYNTHETIC_AT,
            "layer3_replay_at": L3_REPLAY_AT,
            "note": "Evaluation-time binarization choices, NOT validated detector thresholds.",
        },
        "model": model_info,
        "files_total": len(evaluated),
        "per_layer_accuracy": {
            "spectral_heuristic": accuracy(l1_hits, "layer1_correct"),
            "liveness_heuristic": accuracy(l3_hits, "layer3_correct"),
            "verdict": accuracy(verdict_hits, "verdict_correct"),
            "neural_model": {"note": "covered by verdict when the model loads; standalone accuracy needs labeled TTS corpora"},
            "enrollment_biometric": {"note": "not evaluated — no speaker-embedding model installed"},
        },
        "confusion_matrix": confusion,
        "files": files,
    }

    json_path = reports_dir / "layer_eval.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md = ["# Layer evaluation report", ""]
    md.append(f"Generated: {report['generated_at']}")
    md.append(f"Files evaluated: {report['files_total']}")
    md.append(f"Model: {json.dumps(model_info)}")
    md.append("")
    md.append("Thresholds (evaluation-time choices, not validated detector thresholds):")
    md.append(f"- Layer 1 synthetic-like at >= {L1_SYNTHETIC_AT}")
    md.append(f"- Layer 3 replay-like at >= {L3_REPLAY_AT}")
    md.append("")
    md.append("## Per-layer accuracy")
    for layer, acc in report["per_layer_accuracy"].items():
        md.append(f"- {layer}: {json.dumps(acc)}")
    md.append("")
    md.append("## Verdict confusion matrix (rows = true class)")
    md.append("| true \\ predicted | " + " | ".join(labels) + " |")
    md.append("| --- | " + " | ".join(["---"] * len(labels)) + " |")
    for true in CLASSES:
        md.append("| " + true + " | " + " | ".join(str(confusion[true][lab]) for lab in labels) + " |")
    md.append("")
    md.append("## Files")
    for f in files:
        md.append(f"- {json.dumps(f)}")
    (reports_dir / "layer_eval.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"files evaluated: {report['files_total']}")
    print(f"per-layer accuracy: {json.dumps(report['per_layer_accuracy'])}")
    print(f"wrote {json_path} and {reports_dir / 'layer_eval.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
