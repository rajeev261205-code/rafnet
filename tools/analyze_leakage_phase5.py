"""Phase 5 supplementary: leakage / shortcut audit for the VAL-fit stack.

Why: VAL AUC is 1.0 with FAR 0.0018. That is implausible for a generalising
detector, so before trusting TEST numbers we check whether VAL is separable
for trivial reasons:

  1. split hygiene  - human-speaker and fake-generator overlap TRAIN vs VAL/TEST
  2. metadata leaks - dataset id (source) and file codec vs label
  3. trivial audio  - duration, loudness, bandwidth, noise floor, flatness, zcr
  4. shortcut model - logistic regression on TRAIN trivial audio features only,
                      then scored on VAL and on TEST (unseen generators)

Writes reports/leakage_phase5.json + reports/leakage_phase5.md (committed;
aggregates only, no paths, no private filenames). Feature cache is local under
data/leakage_features.json (git-ignored).

Run after TEST scoring:
  .venv/Scripts/python tools/analyze_leakage_phase5.py
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.evaluate_model import binary_metrics_at_05, compute_auc

SPLITS = ROOT / "data" / "splits_phase4.json"
FEATURE_CACHE = ROOT / "data" / "leakage_features.json"
AUDIO_FEATS = ["duration_s", "rms_dbfs", "peak", "rolloff_hz", "flatness",
               "zcr", "noise_floor_dbfs", "snr_db", "crest_db"]


def extract_features(path: str) -> dict:
    import numpy as np
    import soundfile as sf

    x, sr = sf.read(path, dtype="float32", always_2d=False)
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 2:
        x = x.mean(axis=1)
    x = x.ravel()
    if x.size == 0:
        raise ValueError("empty audio")
    n_full = x.size
    duration_s = n_full / float(sr)
    x = x[: int(sr * 15)]                       # cap spectral work at 15 s
    x = x - float(x.mean())                     # remove DC
    rms = math.sqrt(float(np.mean(x * x)) + 1e-20)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    rms_db = 20.0 * math.log10(rms + 1e-12)
    peak_db = 20.0 * math.log10(peak + 1e-12)

    fl = max(16, int(0.025 * sr))
    hop = max(8, int(0.010 * sr))
    if x.size < fl:
        frames = x[None, :]
    else:
        frames = np.lib.stride_tricks.sliding_window_view(x, fl)[::hop]
        if frames.shape[0] > 2000:              # cap frames for long clips
            frames = frames[:2000]
    win = np.hanning(frames.shape[1])
    spec = np.abs(np.fft.rfft(frames * win, axis=1)) ** 2
    freqs = np.fft.rfftfreq(frames.shape[1], 1.0 / sr)
    cum = np.cumsum(spec, axis=1)
    tot = np.maximum(cum[:, -1:], 1e-20)
    idx = np.argmax(cum >= 0.85 * tot, axis=1)
    rolloff_hz = float(np.mean(freqs[idx]))

    eps = 1e-12
    gm = np.exp(np.mean(np.log(spec + eps), axis=1))
    am = np.mean(spec, axis=1) + eps
    flatness = float(np.mean(gm / am))

    zcr = float(np.mean(np.signbit(x[:-1]) != np.signbit(x[1:]))) if x.size > 1 else 0.0

    fr_rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-20)
    fr_db = 20.0 * np.log10(fr_rms + 1e-12)
    noise_floor_dbfs = float(np.percentile(fr_db, 10))
    loud_frame_dbfs = float(np.percentile(fr_db, 90))
    snr_db = loud_frame_dbfs - noise_floor_dbfs
    crest_db = peak_db - rms_db
    return {
        "duration_s": round(float(duration_s), 4),
        "rms_dbfs": round(rms_db, 4),
        "peak": round(peak, 6),
        "rolloff_hz": round(rolloff_hz, 2),
        "flatness": round(flatness, 8),
        "zcr": round(zcr, 6),
        "noise_floor_dbfs": round(noise_floor_dbfs, 4),
        "snr_db": round(snr_db, 4),
        "crest_db": round(crest_db, 4),
    }


def codec_of(path: str) -> str:
    return Path(path).suffix.lower().lstrip(".") or "unknown"


def load_features(rows_by_split: dict[str, list]) -> dict[str, dict]:
    cache = {}
    if FEATURE_CACHE.is_file():
        try:
            cache = json.loads(FEATURE_CACHE.read_text(encoding="utf-8"))
        except Exception:
            cache = {}
    all_rows = [r for rows in rows_by_split.values() for r in rows]
    todo = [r for r in all_rows if r["clip_id"] not in cache]
    print(f"features: {len(all_rows)-len(todo)} cached, {len(todo)} to extract", flush=True)
    for n, r in enumerate(todo, 1):
        try:
            cache[r["clip_id"]] = extract_features(r["path"])
        except Exception as exc:
            cache[r["clip_id"]] = {"error": f"{type(exc).__name__}: {str(exc)[:80]}"}
        if n % 500 == 0:
            print(f"  extracted {n}/{len(todo)}", flush=True)
            FEATURE_CACHE.write_text(json.dumps(cache), encoding="utf-8")
    FEATURE_CACHE.write_text(json.dumps(cache), encoding="utf-8")
    return cache


def univariate(rows, feats):
    ys = [int(r["y"]) for r in rows]
    out = {}
    for f in AUDIO_FEATS:
        vals, yy = [], []
        for r in rows:
            v = feats.get(r["clip_id"], {}).get(f)
            if v is None:
                continue
            vals.append(float(v))
            yy.append(int(r["y"]))
        if len(set(yy)) < 2:
            continue
        auc = compute_auc(yy, vals)
        sep = max(auc, 1.0 - auc)
        real = [v for v, y in zip(vals, yy) if y == 0]
        fake = [v for v, y in zip(vals, yy) if y == 1]
        out[f] = {
            "auc": auc, "separability": round(sep, 4),
            "real_mean": round(sum(real) / len(real), 4),
            "fake_mean": round(sum(fake) / len(fake), 4),
        }
    return out


def shortcut_model(train_rows, eval_sets, feats):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def matrix(rows):
        X, y, keep = [], [], []
        for r in rows:
            fd = feats.get(r["clip_id"], {})
            vals = [fd.get(f) for f in AUDIO_FEATS]
            if any(v is None for v in vals):
                continue
            X.append([float(v) for v in vals])
            y.append(int(r["y"]))
            keep.append(r["clip_id"])
        return np.asarray(X, dtype=float), np.asarray(y, dtype=int), keep

    Xtr, ytr, _ = matrix(train_rows)
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"))
    clf.fit(Xtr, ytr)
    res = {}
    for name, rows in eval_sets.items():
        X, y, _ = matrix(rows)
        if len(set(y.tolist())) < 2:
            res[name] = {"n": len(y), "auc": None, "note": "needs both classes"}
            continue
        p = clf.predict_proba(X)[:, 1].tolist()
        m = binary_metrics_at_05(y.tolist(), p)
        fp = m["fp"]; tn = m["tn"]; fn = m["fn"]; tp = m["tp"]
        res[name] = {
            "n": int(len(y)),
            "auc": compute_auc(y.tolist(), p),
            "far_human_at_0.5": round(fp / (fp + tn), 4) if (fp + tn) else None,
            "detection_fake_at_0.5": round(tp / (tp + fn), 4) if (tp + fn) else None,
        }
    return res


def source_table(rows):
    c = defaultdict(lambda: [0, 0])
    for r in rows:
        c[r.get("source", "?")][int(r["y"])] += 1
    return {k: {"human": v[0], "fake": v[1]} for k, v in sorted(c.items())}


def codec_table(rows):
    c = defaultdict(lambda: [0, 0])
    for r in rows:
        c[codec_of(r["path"])][int(r["y"])] += 1
    return {k: {"human": v[0], "fake": v[1]} for k, v in sorted(c.items())}


def main() -> int:
    splits = json.loads(SPLITS.read_text(encoding="utf-8"))
    tr, va, te = splits["train"], splits["val"], splits["test"]

    ids = {s: [r["clip_id"] for r in rows] for s, rows in
           (("train", tr), ("val", va), ("test", te))}
    overlap = {
        "train_val": len(set(ids["train"]) & set(ids["val"])),
        "train_test": len(set(ids["train"]) & set(ids["test"])),
        "val_test": len(set(ids["val"]) & set(ids["test"])),
    }
    human = {s: set(r["speaker"] for r in rows if int(r["y"]) == 0)
             for s, rows in (("train", tr), ("val", va), ("test", te))}
    fake = {s: set(r["speaker"] for r in rows if int(r["y"]) == 1)
            for s, rows in (("train", tr), ("val", va), ("test", te))}
    gen = {
        "human_speakers": {
            "train": len(human["train"]), "val": len(human["val"]), "test": len(human["test"]),
            "train_val_overlap": len(human["train"] & human["val"]),
            "train_test_overlap": len(human["train"] & human["test"]),
            "val_test_overlap": len(human["val"] & human["test"]),
        },
        "fake_generators": {
            "train": len(fake["train"]), "val": len(fake["val"]), "test": len(fake["test"]),
            "train_val_overlap": len(fake["train"] & fake["val"]),
            "train_test_overlap": len(fake["train"] & fake["test"]),
            "val_test_overlap": len(fake["val"] & fake["test"]),
            "val_shared": sorted(fake["train"] & fake["val"]),
            "test_unseen": sorted(fake["test"] - fake["train"]),
        },
    }

    feats = load_features({"train": tr, "val": va, "test": te})
    n_err = sum(1 for v in feats.values() if "error" in v)
    tr = [r for r in tr if "error" not in feats.get(r["clip_id"], {})]
    va = [r for r in va if "error" not in feats.get(r["clip_id"], {})]
    te = [r for r in te if "error" not in feats.get(r["clip_id"], {})]

    uni_val = univariate(va, feats)
    uni_test = univariate(te, feats)
    shortcut = shortcut_model(tr, {"val": va, "test": te}, feats)

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "feature_errors": n_err,
        "split_id_overlap": overlap,
        "disjointness": gen,
        "source_vs_label": {
            "val": source_table(va), "test": source_table(te),
        },
        "codec_vs_label": {
            "val": codec_table(va), "test": codec_table(te),
        },
        "val_univariate": uni_val,
        "test_univariate": uni_test,
        "shortcut_model_train_audio_features": shortcut,
        "notes": [
            "dataset id (source) is label-defining by construction: human clips "
            "come from human datasets and synthetic clips from fake datasets, so "
            "it must never be used as an inference feature.",
            "VAL fake generators are the SAME voices/families as TRAIN (only "
            "human speakers are disjoint); VAL AUC 1.0 therefore measures "
            "in-distribution generator memorisation, not generalisation.",
            "TEST fake generators and private groups are disjoint from TRAIN; "
            "the stack's per-group TEST numbers are the honest generalisation "
            "measure.",
        ],
    }
    (ROOT / "reports" / "leakage_phase5.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")

    L = ["# Leakage / shortcut audit (VAL AUC 1.0 sanity check)", "",
         f"Generated: {out['generated_at']}", "",
         "## Split hygiene",
         f"- clip-id overlap: {json.dumps(overlap)}",
         f"- human speakers: train={len(human['train'])} val={len(human['val'])} "
         f"test={len(human['test'])}; train-val overlap="
         f"{gen['human_speakers']['train_val_overlap']}, train-test overlap="
         f"{gen['human_speakers']['train_test_overlap']}",
         f"- fake generators (voice ids): train={len(fake['train'])} "
         f"val={len(fake['val'])} test={len(fake['test'])}; train-val overlap="
         f"{gen['fake_generators']['train_val_overlap']}, train-test overlap="
         f"{gen['fake_generators']['train_test_overlap']}",
         f"- VAL fake generators shared with TRAIN: "
         f"{', '.join(gen['fake_generators']['val_shared'])}",
         f"- TEST unseen fake generators: "
         f"{', '.join(gen['fake_generators']['test_unseen'])}", "",
         "## Dataset id vs label (metadata shortcut, not usable at inference)"]
    for split in ("val", "test"):
        L.append(f"- {split}: " + ", ".join(
            f"{k}={v['human']}H/{v['fake']}F" for k, v in out["source_vs_label"][split].items()))
    L += ["", "## Codec vs label (file suffix) "]
    for split in ("val", "test"):
        L.append(f"- {split}: " + ", ".join(
            f"{k}={v['human']}H/{v['fake']}F" for k, v in out["codec_vs_label"][split].items()))
    L += ["", "## Trivial audio features on VAL (direction-normalised separability)",
          "| feature | AUC | separability | human mean | fake mean |",
          "| --- | --- | --- | --- | --- |"]
    for f, m in sorted(uni_val.items(), key=lambda kv: -kv[1]["separability"]):
        L.append(f"| {f} | {m['auc']} | {m['separability']} | "
                 f"{m['real_mean']} | {m['fake_mean']} |")
    L += ["", "## Shortcut model (logistic regression on TRAIN trivial audio "
          "features only; no dataset id)", "",
          "| eval set | n | AUC | FAR human @0.5 | detection fake @0.5 |",
          "| --- | --- | --- | --- | --- |"]
    for name, m in shortcut.items():
        L.append(f"| {name} | {m.get('n')} | {m.get('auc')} | "
                 f"{m.get('far_human_at_0.5')} | {m.get('detection_fake_at_0.5')} |")
    L += ["", "## Reading"] + ["- " + n for n in out["notes"]]
    (ROOT / "reports" / "leakage_phase5.md").write_text(
        "\n".join(L) + "\n", encoding="utf-8")

    print(f"wrote reports/leakage_phase5.json + reports/leakage_phase5.md "
          f"(feature errors={n_err})")
    print(f"val separability: " + ", ".join(
        f"{f}={m['separability']}" for f, m in
        sorted(uni_val.items(), key=lambda kv: -kv[1]["separability"])))
    print(f"shortcut model: {json.dumps(shortcut)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
