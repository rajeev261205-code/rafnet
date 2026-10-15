"""Phase 3 step 2: logistic stacking + temperature scaling + abstain band on VAL.

Inputs: data/scores_cache.json (P(AI) per model) + data/phase2_manifest.csv.
Trained/selected on VAL ONLY (80 rows: 40 HUMAN_LIVE + 40 crude-VC fake).
TEST and my_samples are NEVER touched here (Phase 4).

Outputs: models/stacking.json (weights, T, taus — force-added, ~1 KB) +
reports/stacking.md (VAL metrics + band justification).
"""
import csv, json, math
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
FEATS = ["mo-thecreator/Deepfake-audio-detection",
         "Bisher/wav2vec2_ASV_deepfake_audio_detection", "rafnet"]

def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-max(-500, min(500, z))))

def main() -> int:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    man = [r for r in csv.DictReader(open(ROOT / "data" / "phase2_manifest.csv", encoding="utf-8")) if r["split"] == "val"]
    cache = json.loads((ROOT / "data" / "scores_cache.json").read_text(encoding="utf-8"))["scores"]
    X, y = [], []
    missing = 0
    for r in man:
        s = cache.get(r["path"], {})
        if any(s.get(f) is None for f in FEATS):
            missing += 1
            continue
        X.append([float(s[f]) for f in FEATS])
        y.append(0 if r["label_origin"] == "HUMAN_LIVE" else 1)
    X = np.array(X); y = np.array(y)
    print(f"val rows usable: {len(y)}/80 (missing={missing})")
    assert len(y) >= 40 and set(y.tolist()) == {0, 1}, "need both classes on val"
    clf = LogisticRegression(C=1.0, max_iter=1000).fit(X, y)
    z = clf.decision_function(X)
    # temperature scaling on VAL logits (grid search on NLL)
    def nll(T):
        return float(-(y * np.log([sigmoid(v / T) for v in z]) + (1 - y) * np.log([1 - sigmoid(v / T) for v in z])).mean())
    Ts = [round(t, 2) for t in np.linspace(0.2, 5.0, 49)]
    T = min(Ts, key=nll)
    pcal = np.array([sigmoid(v / T) for v in z])
    # abstain band: selective accuracy >= 0.95 with max coverage; ties -> narrower
    best = None
    for lo in np.linspace(0.01, 0.45, 23):
        for hi in np.linspace(0.55, 0.95, 17):
            if not (0.1 <= hi - lo <= 0.5):
                continue
            certain = [(pp, yy) for pp, yy in zip(pcal, y) if pp < lo or pp > hi]
            cov = len(certain) / len(y)
            sel = sum(1 for pp, yy in certain if (pp > hi) == (yy == 1)) / len(certain) if certain else 0.0
            key = (sel >= 0.95, round(sel, 4), round(cov, 4), -round(hi - lo, 4))
            if best is None or key > best[0]:
                best = (key, round(float(lo), 4), round(float(hi), 4), round(float(sel), 4), round(float(cov), 4))
    _, lo, hi, sel, cov = best
    # VAL metrics of calibrated stacker (binary@0.5 + 3-way with band)
    import sys
    sys.path.insert(0, str(ROOT))
    from tools.evaluate_model import binary_metrics_at_05, compute_auc, compute_eer, confusion_3way
    s = [float(v) for v in pcal]
    art = {"features": FEATS, "coef": [round(float(c), 4) for c in clf.coef_[0]],
           "intercept": round(float(clf.intercept_[0]), 4), "temperature": T,
           "tau_low": lo, "tau_high": hi,
           "val_n": len(y), "val_binary_at_0.5": binary_metrics_at_05(y.tolist(), s),
           "val_auc": compute_auc(y.tolist(), s), "val_eer": compute_eer(y.tolist(), s),
           "val_confusion_band": confusion_3way(y.tolist(), s, lo, hi),
           "val_selective_accuracy": sel, "val_coverage": cov,
           "val_nll_T1": round(nll(1.0), 4), "val_nll_T": round(nll(T), 4),
           "band_justification": (f"Grid on VAL: first band with selective accuracy>=0.95, "
                                  f"then max coverage, then narrowest. Chosen [{lo}, {hi}] "
                                  f"selective={sel} coverage={cov}. TEST/my_samples untouched."),
           "caveat": "VAL fake is crude-VC only; band + weights prove the pipeline, not generalization. Phase 4 re-measures on TEST + my_samples."}
    (ROOT / "models" / "stacking.json").write_text(json.dumps(art, indent=2), encoding="utf-8")
    md = ["# Phase 3 stacking (fit on VAL only)", "",
          f"features={FEATS}", f"coef={art['coef']} intercept={art['intercept']} T={T}",
          f"band=[{lo}, {hi}] selective={sel} coverage={cov}",
          f"VAL binary@0.5: {json.dumps(art['val_binary_at_0.5'])}",
          f"VAL AUC={art['val_auc']} EER={json.dumps(art['val_eer'])}",
          f"VAL confusion(band): {json.dumps(art['val_confusion_band'])}",
          f"NLL T=1: {art['val_nll_T1']} -> T={T}: {art['val_nll_T']}", "",
          art["band_justification"], art["caveat"]]
    (ROOT / "reports" / "stacking.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"coef={art['coef']} T={T} band=[{lo},{hi}] sel={sel} cov={cov} auc={art['val_auc']}")
    print("wrote models/stacking.json + reports/stacking.md (VAL only)")
    return 0
if __name__ == "__main__": raise SystemExit(main())
