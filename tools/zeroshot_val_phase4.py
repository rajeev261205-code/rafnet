"""Phase 4.1: zero-shot candidate check on the NEW val split (speaker-disjoint).

Scores a stratified val subset (250 real + 250 fake, seeded) with cached HF
detectors, fully offline (HF_HUB_OFFLINE=1). Metrics: AUC, FAR-on-human,
binary accuracy. Keep rule: AUC > 0.6 AND runs end-to-end without custom code.
Writes reports/zeroshot_phase4.json/.md (committed, aggregates only).
Resume-safe per-model cache in data/ (local).
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT))

CANDIDATES = {
    "mo-thecreator": "mo-thecreator/Deepfake-audio-detection",
    "bisher": "Bisher/wav2vec2_ASV_deepfake_audio_detection",
}
N_REAL, N_FAKE = 250, 250

LICENSES = {"mo-thecreator": "apache-2.0 (Hub tag)",
            "bisher": "apache-2.0 (Hub tag)"}


def val_subset():
    import random

    d = json.loads((ROOT / "data" / "splits_phase4.json").read_text(encoding="utf-8"))
    real = [r for r in d["val"] if r["y"] == 0]
    fake = [r for r in d["val"] if r["y"] == 1]
    rng = random.Random(5)
    rng.shuffle(real)
    rng.shuffle(fake)
    return real[:N_REAL] + fake[:N_FAKE]


def load_audio(path: str, max_s: float = 6.0):
    import numpy as np
    import soundfile as sf

    arr, sr = sf.read(path, dtype="float32", always_2d=False)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr.mean(axis=1).astype(np.float32)
    arr = arr.ravel()
    if int(sr) != 16000 and arr.size:
        tgt = max(1, int(round(len(arr) / float(sr) * 16000)))
        arr = np.interp(np.linspace(0, 1, tgt),
                        np.linspace(0, 1, len(arr)), arr).astype(np.float32)
    cap = int(16000 * max_s)
    return (arr[:cap] if len(arr) > cap else arr), 16000


def score_model(mid: str, items: list[dict]) -> dict:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

    t0 = time.perf_counter()
    feat = AutoFeatureExtractor.from_pretrained(mid, trust_remote_code=False)
    mdl = AutoModelForAudioClassification.from_pretrained(mid, trust_remote_code=False)
    mdl.eval()
    n_params = sum(p.numel() for p in mdl.parameters())
    labels = [str(mdl.config.id2label.get(i, i)) for i in range(len(mdl.config.id2label))]
    low = " ".join(labels).lower()
    fi = next((i for i, l in enumerate(labels)
               if any(k in l.lower() for k in ("fake", "spoof", "synthetic", "ai", "deepfake"))), None)
    out = []
    lat = []
    with torch.inference_mode():
        for it in items:
            try:
                arr, sr = load_audio(it["path"])
                inp = feat(arr, sampling_rate=sr, return_tensors="pt", padding=True)
                s = time.perf_counter()
                logits = mdl(**{k: v for k, v in inp.items()
                                if k in ("input_values", "attention_mask")}).logits
                lat.append((time.perf_counter() - s) * 1000)
                vec = logits[0] if logits.ndim == 2 else logits.reshape(-1)
                probs = F.softmax(vec, dim=-1).tolist()
                if isinstance(probs, float):
                    probs = [1.0 - probs, probs]
                p = float(probs[fi]) if fi is not None else float(probs[1] if len(probs) == 2 else max(probs))
                out.append({"clip_id": it["clip_id"], "y": it["y"], "p": round(p, 4)})
            except Exception as exc:
                out.append({"clip_id": it["clip_id"], "y": it["y"],
                            "error": f"{type(exc).__name__}: {str(exc)[:120]}"})
    return {"model": mid, "labels": labels, "params_M": round(n_params / 1e6, 1),
            "load_s": round(time.perf_counter() - t0, 1),
            "mean_latency_ms": round(float(np.mean(lat)), 1) if lat else None,
            "scores": out}


def metrics(scores: list[dict]) -> dict:
    sys.path.insert(0, str(ROOT))
    from tools.evaluate_model import binary_metrics_at_05, compute_auc, compute_eer

    ok = [s for s in scores if "p" in s]
    y = [s["y"] for s in ok]
    p = [s["p"] for s in ok]
    far = sum(1 for s in ok if s["y"] == 0 and s["p"] >= 0.5) / max(1, sum(1 for s in ok if s["y"] == 0))
    return {"n": len(ok), "errors": len(scores) - len(ok), "auc": compute_auc(y, p),
            "eer": compute_eer(y, p), "far_human": round(far, 4),
            "binary_at_0.5": binary_metrics_at_05(y, p)}


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--keep", default="auc_gt_0.6")
    args = ap.parse_args()
    items = val_subset()
    print(f"val subset: {len(items)} (real={sum(1 for i in items if i['y']==0)})", flush=True)
    res: dict = {"generated_at": datetime.now(timezone.utc).isoformat(),
                 "val_n": len(items), "models": {}}
    jp = ROOT / "reports" / "zeroshot_phase4.json"
    if jp.is_file():
        try:
            res = json.loads(jp.read_text(encoding="utf-8"))
        except Exception:
            pass
    for name, mid in CANDIDATES.items():
        if args.only and name != args.only:
            continue
        cache = ROOT / "data" / f"zeroshot_cache_{name}.json"
        if cache.is_file():
            print(f"{name}: cached", flush=True)
            res["models"][name] = json.loads(cache.read_text(encoding="utf-8"))
            continue
        print(f"{name}: scoring {len(items)}...", flush=True)
        try:
            scored = score_model(mid, items)
        except Exception as exc:
            res["models"][name] = {"model": mid, "load_error": f"{type(exc).__name__}: {str(exc)[:200]}",
                                   "license": LICENSES[name], "verdict": "DROP (unloadable offline)"}
            continue
        m = metrics(scored["scores"])
        keep = (m["auc"] or 0) > 0.6 and scored.get("mean_latency_ms", 1e9) < 2000
        scored.update({"metrics": m, "license": LICENSES[name],
                       "verdict": "KEEP" if keep else "DROP (below bar)"})
        cache.write_text(json.dumps(scored, indent=1), encoding="utf-8")
        res["models"][name] = scored
        print(f"{name}: {json.dumps(m)} -> {scored['verdict']}", flush=True)
    res["generated_at"] = datetime.now(timezone.utc).isoformat()
    jp.write_text(json.dumps(res, indent=1), encoding="utf-8")
    L = ["# Phase 4.1 zero-shot candidates on NEW val split (speaker-disjoint)", "",
         f"val subset n={res['val_n']} (250 real + 250 fake); offline, cached weights.",
         ""]
    for name, r in res.get("models", {}).items():
        L.append(f"## {name} ({r.get('license')})")
        L.append(f"params={r.get('params_M')}M latency={r.get('mean_latency_ms')}ms "
                 f"metrics={json.dumps(r.get('metrics'))} verdict={r.get('verdict')}")
        L.append("")
    (ROOT / "reports" / "zeroshot_phase4.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote reports/zeroshot_phase4.*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
