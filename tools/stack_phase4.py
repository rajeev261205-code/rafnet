"""Phase 4.2c: score val/test with the ensemble inputs + stack them.

Inputs (local, git-ignored):
  data/splits_phase4.json            clip_id -> path/y for val (+ test later)
  data/embed_chunks/*_clean_*.npz    head input (train tag = clean)
  models/head_phase4.pt              trained head (Phase 4.2b)
  data/scores_phase4_<split>.json    per-model cache (mo/bisher/rafnet)

Ensemble: [head, rafnet, mo, bisher] P(AI). Logistic stack fit on VAL only +
temperature scaling + UNCERTAIN band (val-only). TEST is scored but the stack
never reads test labels for fitting; Phase 5 does the honest measurement.

  --split val  : score missing models, fit stack, save models/stack_phase4.json
  --split test : score missing models only (no fit) -> data/scores_phase4_test.json
  --limit N    : deterministic stratified cap of the split rows (smoke tests)
  --out-suffix S: suffix every output path (data/scores_phase4_val_S.json,
                 models/stack_phase4_S.json, reports/stack_phase4_S.md) so
                 smoke artifacts never clobber the real ones

Offline (HF_HUB_OFFLINE=1). Aggregates only in committed reports.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.setdefault("MOCK_INFERENCE", "false")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

HF_MODELS = {
    "mo": "mo-thecreator/Deepfake-audio-detection",
    "bisher": "Bisher/wav2vec2_ASV_deepfake_audio_detection",
}
FEATS = ["head", "rafnet", "mo", "bisher"]


def split_rows(split: str):
    d = json.loads((ROOT / "data" / "splits_phase4.json").read_text(encoding="utf-8"))
    return d[split]


def load16(path: str, max_s: float = 6.0):
    import numpy as np
    import soundfile as sf

    arr, sr = sf.read(path, dtype="float32", always_2d=False)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr.mean(axis=1).astype(np.float32)
    arr = arr.ravel()
    if int(sr) != 16000 and arr.size:
        tgt = int(min(arr.size / float(sr), max_s) * 16000)
        arr = np.interp(np.linspace(0, 1, tgt), np.linspace(0, 1, arr.size), arr).astype(np.float32)
    else:
        arr = arr[:16000 * int(max_s)]
    return arr


def score_hf(rows, cache, only):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

    for name in only:
        mid = HF_MODELS[name]
        todo = [r for r in rows if cache.get(r["clip_id"], {}).get(name) is None]
        if not todo:
            print(f"{name}: cache complete", flush=True)
            continue
        feat = AutoFeatureExtractor.from_pretrained(mid, trust_remote_code=False)
        mdl = AutoModelForAudioClassification.from_pretrained(mid, trust_remote_code=False)
        mdl.eval()
        lab = [str(mdl.config.id2label.get(i, f"L{i}")) for i in range(mdl.config.num_labels)]
        fi = next((i for i, l in enumerate(lab)
                   if any(k in l.lower() for k in ("fake", "spoof", "synthetic", "ai", "deepfake", "aivoice"))), 0)
        t0 = time.perf_counter()
        with torch.inference_mode():
            for n, r in enumerate(todo, 1):
                try:
                    arr = load16(r["path"])
                    inp = feat(arr, sampling_rate=16000, return_tensors="pt", padding=True)
                    lg = mdl(**{k: v for k, v in inp.items()
                                if k in ("input_values", "attention_mask")}).logits
                    vec = lg[0] if lg.ndim == 2 else lg.reshape(-1)
                    cache.setdefault(r["clip_id"], {})[name] = round(float(F.softmax(vec, -1)[fi]), 4)
                except Exception as exc:
                    cache.setdefault(r["clip_id"], {})[name] = None
                    print(f"ERR {r['clip_id']} {name}: {type(exc).__name__}: {str(exc)[:100]}", flush=True)
                if n % 200 == 0:
                    print(f"  {name} {n}/{len(todo)} ({time.perf_counter()-t0:.0f}s)", flush=True)
                    _flush(cache)
        del mdl, feat
        import gc
        gc.collect()
        print(f"{name}: {len(todo)} scored in {time.perf_counter()-t0:.0f}s", flush=True)


def score_rafnet(rows, cache):
    todo = [r for r in rows if cache.get(r["clip_id"], {}).get("rafnet") is None]
    if not todo:
        print("rafnet: cache complete", flush=True)
        return
    from rafnet.soc.model_adapter import load_voice_model
    from rafnet.soc.audio_io import decode_audio

    rm = load_voice_model()
    assert rm.is_loaded() and getattr(rm, "name", "") != "mock", "real RAFNet must load"
    t0 = time.perf_counter()
    for n, r in enumerate(todo, 1):
        try:
            p = Path(r["path"])
            dec = decode_audio(p.read_bytes(), p.name)
            res = rm.predict(dec.samples, dec.sample_rate)
            cache.setdefault(r["clip_id"], {})["rafnet"] = round(float(res.prob_synthetic), 4)
        except Exception as exc:
            cache.setdefault(r["clip_id"], {})["rafnet"] = None
            print(f"RAFNet ERR {r['clip_id']}: {type(exc).__name__}: {str(exc)[:100]}", flush=True)
        if n % 200 == 0:
            print(f"  rafnet {n}/{len(todo)} ({time.perf_counter()-t0:.0f}s)", flush=True)
            _flush(cache)
    print(f"rafnet: {len(todo)} scored in {time.perf_counter()-t0:.0f}s", flush=True)


_CUR_CACHE_PATH = [None]


def _flush(cache):
    if _CUR_CACHE_PATH[0]:
        _CUR_CACHE_PATH[0].write_text(json.dumps(cache, indent=0), encoding="utf-8")


def head_scores(rows):
    """Head P(AI) for the given rows using clean embeddings (no fitting)."""
    import numpy as np
    import torch
    from tools.train_head_phase4 import (N_LAYERS, build_model, index_by_clip,
                                         load_tag, predict)

    ids, ys, mean, std = load_tag("clean")
    idx = index_by_clip(ids)
    ck = torch.load(str(ROOT / "models" / "head_phase4.pt"), map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = build_model(cfg["seed"], cfg["mid"], cfg["dropout"])
    model.load_state_dict(ck["state_dict"])
    model.eval()

    keep, mrows, srows = [], [], []
    for r in rows:
        j = idx.get(r["clip_id"])
        if j is None:
            continue
        keep.append(r["clip_id"])
        mrows.append(mean[j])
        srows.append(std[j])
    if not keep:
        return {}
    logit = predict(model, np.stack(mrows), np.stack(srows))
    import math
    out = {}
    for cid, z in zip(keep, logit):
        out[cid] = round(1.0 / (1.0 + math.exp(-max(-500.0, min(500.0, float(z))))), 4)
    return out


def fit_stack(rows, cache, head):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    sys.path.insert(0, str(ROOT))
    from tools.evaluate_model import binary_metrics_at_05, compute_auc, compute_eer

    X, y, used = [], [], 0
    for r in rows:
        cid = r["clip_id"]
        feats = [head.get(cid), cache.get(cid, {}).get("rafnet"),
                 cache.get(cid, {}).get("mo"), cache.get(cid, {}).get("bisher")]
        if any(v is None for v in feats):
            continue
        X.append([float(v) for v in feats])
        y.append(int(r["y"]))
        used += 1
    X = np.array(X)
    y = np.array(y)
    print(f"stack fit rows: {used}/{len(rows)} (real={int((y==0).sum())}, fake={int((y==1).sum())})", flush=True)
    assert len(set(y.tolist())) == 2, "need both classes on val"
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(X, y)
    z = clf.decision_function(X)

    def sig(T):
        return 1.0 / (1.0 + np.exp(-np.clip(z / T, -500, 500)))

    def nll(T):
        p = sig(T)
        return float(-(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)).mean())

    Ts = [round(t, 2) for t in np.linspace(0.2, 5.0, 49)]
    T = min(Ts, key=nll)
    pcal = [float(v) for v in sig(T)]

    # UNCERTAIN band: selective accuracy >= 0.95, max coverage, narrowest
    best = None
    for lo in [round(x, 4) for x in np.linspace(0.01, 0.45, 23)]:
        for hi in [round(x, 4) for x in np.linspace(0.55, 0.95, 17)]:
            if not (0.1 <= hi - lo <= 0.5):
                continue
            cert = [(pp, yy) for pp, yy in zip(pcal, y) if pp < lo or pp > hi]
            if not cert:
                continue
            cov = len(cert) / len(y)
            sel = sum(1 for pp, yy in cert if (pp > hi) == (yy == 1)) / len(cert)
            key = (sel >= 0.95, round(sel, 4), round(cov, 4), -round(hi - lo, 4))
            if best is None or key > best[0]:
                best = (key, lo, hi, round(sel, 4), round(cov, 4))
    if best is None:
        lo, hi, sel, cov = 0.3611, 0.8611, None, None
    else:
        _, lo, hi, sel, cov = best
    far = sum(1 for yy, pp in zip(y, pcal) if yy == 0 and pp >= 0.5) / max(1, int((y == 0).sum()))
    return {
        "features": FEATS,
        "coef": [round(float(c), 4) for c in clf.coef_[0]],
        "intercept": round(float(clf.intercept_[0]), 4),
        "temperature": T,
        "tau_low": lo, "tau_high": hi,
        "val_n": int(len(y)),
        "val_auc": compute_auc(y.tolist(), pcal),
        "val_eer": compute_eer(y.tolist(), pcal),
        "val_binary_at_0.5": binary_metrics_at_05(y.tolist(), pcal),
        "val_far_human_at_0.5": round(float(far), 4),
        "val_selective_accuracy": sel, "val_coverage": cov,
        "val_nll_T1": round(nll(1.0), 4), "val_nll_T": round(nll(T), 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--only", default="", help="subset of mo,bisher,rafnet")
    ap.add_argument("--skip-head", action="store_true")
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke test: cap rows, stratified, deterministic")
    ap.add_argument("--out-suffix", default="",
                    help="suffix output paths so smoke runs stay isolated")
    args = ap.parse_args()

    rows = split_rows(args.split)
    if args.limit and args.limit < len(rows):
        import numpy as np
        rng = np.random.default_rng(17)
        keep = []
        for cls in (0, 1):
            cls_rows = [r for r in rows if int(r["y"]) == cls]
            k = min(len(cls_rows), max(1, args.limit * len(cls_rows) // len(rows)))
            pick = rng.choice(len(cls_rows), size=k, replace=False)
            keep.extend(cls_rows[int(i)] for i in sorted(pick))
        rows = keep
        print(f"limit={args.limit}: capped split to {len(rows)} rows "
              f"(real={sum(1 for r in rows if int(r['y'])==0)}, "
              f"fake={sum(1 for r in rows if int(r['y'])==1)})", flush=True)
    print(f"split={args.split}: {len(rows)} rows", flush=True)
    suf = args.out_suffix
    cpath = ROOT / "data" / f"scores_phase4_{args.split}{suf}.json"
    _CUR_CACHE_PATH[0] = cpath
    cache = {}
    if cpath.is_file():
        try:
            cache = json.loads(cpath.read_text(encoding="utf-8"))
        except Exception:
            cache = {}
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    hf_only = [n for n in ("mo", "bisher") if (not only or n in only)]
    if not args.skip_head and (not only or "head" in only):
        head = head_scores(rows)
        for cid, p in head.items():
            cache.setdefault(cid, {})["head"] = p
        print(f"head: {len(head)} scored", flush=True)
        _flush(cache)
    if not only or "rafnet" in only:
        score_rafnet(rows, cache)
        _flush(cache)
    if hf_only:
        score_hf(rows, cache, hf_only)
        _flush(cache)
    cpath.write_text(json.dumps(cache, indent=0), encoding="utf-8")
    print(f"wrote {cpath.name} ({len(cache)} clips)", flush=True)

    if args.split == "val":
        head = {cid: cache[cid].get("head") for cid in cache}
        art = fit_stack(rows, cache, head)
        art["generated_at"] = datetime.now(timezone.utc).isoformat()
        art["note"] = ("Stack fit on VAL only (speaker-disjoint); TEST never read for "
                       "fitting. Band from val selective accuracy >= 0.95.")
        (ROOT / "models" / f"stack_phase4{suf}.json").write_text(json.dumps(art, indent=2), encoding="utf-8")
        md = ["# Phase 4.2c stacked ensemble (fit on VAL only)", "",
              f"features={art['features']}",
              f"coef={art['coef']} intercept={art['intercept']} T={art['temperature']}",
              f"band=[{art['tau_low']}, {art['tau_high']}] selective={art['val_selective_accuracy']} "
              f"coverage={art['val_coverage']}",
              f"VAL AUC={art['val_auc']} EER={json.dumps(art['val_eer'])} "
              f"FAR@0.5={art['val_far_human_at_0.5']} binary@0.5={json.dumps(art['val_binary_at_0.5'])}",
              f"NLL T=1: {art['val_nll_T1']} -> T={art['temperature']}: {art['val_nll_T']}", "",
              art["note"], "",
              "VAL fake shares train generators (in-distribution); unseen generators "
              "+ private groups live in TEST (Phase 5)."]
        (ROOT / "reports" / f"stack_phase4{suf}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
        print(f"stack: coef={art['coef']} T={art['temperature']} band=[{art['tau_low']},{art['tau_high']}] "
              f"val_auc={art['val_auc']} FAR={art['val_far_human_at_0.5']}")
        print(f"wrote models/stack_phase4{suf}.json + reports/stack_phase4{suf}.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
