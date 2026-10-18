"""Phase 5: honest evaluation of the Phase-4 stack on TEST (single touch).

Frozen inputs (fit elsewhere, VAL-only):
  models/stack_phase4.json         logreg coefs + temperature + band (VAL-fit)
  data/scores_phase4_test.json     per-clip [head, rafnet, mo, bisher] P(AI)
  data/splits_phase4.json          test rows (clip_id -> y, source, speaker)

TEST is READ HERE AND NEVER USED FOR TUNING anywhere in the pipeline: the
stack, its temperature, and the UNCERTAIN band were frozen from VAL before
this script ran. Nothing in this script selects thresholds or weights.

Writes reports/model_eval.json + reports/model_eval.md (committed; groups are
source/generator labels only, no paths, no private filenames). Existing
model_eval.* from the Phase-3-era ensemble are backed up to
model_eval.phase3-ensemble.* once.
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.evaluate_model import (binary_metrics_at_05, compute_auc, compute_eer,
                                  confusion_3way, coverage_and_selective)


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-500.0, min(500.0, z))))


def group_of(row: dict) -> str:
    src = row.get("source", "?")
    if src.startswith("private-"):
        return src.replace("private-", "private-")
    if row["y"] == 1:
        return row.get("speaker", "fake")
    return src


def metrics(rows, scores, tau_low, tau_high):
    ys = [int(r["y"]) for r in rows]
    ps = [scores[r["clip_id"]] for r in rows]
    binm = binary_metrics_at_05(ys, ps)
    tp, fn, tn, fp = binm["tp"], binm["fn"], binm["tn"], binm["fp"]
    far = round(fp / (fp + tn), 4) if (fp + tn) else None
    frr = round(fn / (fn + tp), 4) if (fn + tp) else None
    bal = None
    if (tp + fn) and (tn + fp):
        bal = round((tp / (tp + fn) + tn / (tn + fp)) / 2, 4)
    return {
        "n": len(ys), "n_human": sum(1 for v in ys if v == 0),
        "n_fake": sum(1 for v in ys if v == 1),
        "far_human_at_0.5": far, "frr_fake_at_0.5": frr,
        "accuracy": binm["accuracy"], "balanced_accuracy": bal,
        "auc": compute_auc(ys, ps), "eer": compute_eer(ys, ps),
        "binary_at_0.5": binm,
        "band": [tau_low, tau_high],
        "confusion_band": confusion_3way(ys, ps, tau_low, tau_high),
        **coverage_and_selective(ys, ps, tau_low, tau_high),
    }


def main() -> int:
    stack = json.loads((ROOT / "models" / "stack_phase4.json").read_text(encoding="utf-8"))
    splits = json.loads((ROOT / "data" / "splits_phase4.json").read_text(encoding="utf-8"))
    test_rows = splits["test"]
    cpath = ROOT / "data" / "scores_phase4_test.json"
    cache = json.loads(cpath.read_text(encoding="utf-8"))

    feats = stack["features"]
    coef, b, T = stack["coef"], stack["intercept"], stack["temperature"]
    tau_low, tau_high = stack["tau_low"], stack["tau_high"]
    print(f"stack: feats={feats} coef={coef} b={b} T={T} band=[{tau_low},{tau_high}]")

    # ---- apply frozen stack to TEST (no fitting anywhere in this script) ----
    scores: dict[str, float] = {}
    missing_per_model = Counter()
    incomplete = []
    for r in test_rows:
        cid = r["clip_id"]
        s = cache.get(cid, {})
        vals = [s.get(f) for f in feats]
        for f, v in zip(feats, vals):
            if v is None:
                missing_per_model[f] += 1
        if any(v is None for v in vals):
            incomplete.append(cid)
            continue
        z = sum(c * float(v) for c, v in zip(coef, vals)) + b
        scores[cid] = sigmoid(z / T)

    n_scored = len(scores)
    print(f"scored {n_scored}/{len(test_rows)} test rows; "
          f"missing-per-model={dict(missing_per_model)} incomplete={len(incomplete)}")

    # preserve the Phase-3-ensemble eval once (first Phase-5 run only)
    import shutil
    for pth in ("reports/model_eval.json", "reports/model_eval.md"):
        src, dst = ROOT / pth, ROOT / pth.replace("model_eval", "model_eval.phase3-ensemble")
        if src.is_file() and not dst.is_file():
            shutil.copy(src, dst)

    complete = [r for r in test_rows if r["clip_id"] in scores]
    overall = metrics(complete, scores, tau_low, tau_high)

    by_group: dict[str, list] = {}
    for r in complete:
        by_group.setdefault(group_of(r), []).append(r)
    group_m = {g: metrics(v, scores, tau_low, tau_high)
               for g, v in sorted(by_group.items(),
                                  key=lambda kv: (kv[1][0]["y"], kv[0]))}

    # reliability bins (calibrated p vs binary correctness)
    rel = []
    for lo, hi in [(0.0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]:
        sel = [(scores[r["clip_id"]], int(r["y"])) for r in complete
               if lo <= scores[r["clip_id"]] < hi]
        if not sel:
            rel.append({"bin": f"{lo}-{min(hi, 1.0)}", "n": 0})
            continue
        acc = sum(1 for p, y in sel if (1 if p >= 0.5 else 0) == y) / len(sel)
        rel.append({"bin": f"{lo}-{min(hi, 1.0)}", "n": len(sel),
                    "accuracy": round(acc, 4),
                    "avg_p": round(sum(p for p, _ in sel) / len(sel), 4)})

    humans = [r for r in complete if r["y"] == 0]
    fakes = [r for r in complete if r["y"] == 1]
    far_first = {
        "FAR_human_overall": overall["far_human_at_0.5"],
        "FAR_human_per_group": {g: m["far_human_at_0.5"] for g, m in group_m.items()
                                if m["n_human"]},
        "FRR_fake_overall": overall["frr_fake_at_0.5"],
        "coverage_band": overall.get("coverage"),
        "selective_accuracy_band": overall.get("selective_accuracy"),
    }

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": {"name": "wavlm-head + rafnet + mo + bisher (stacked)",
                  "stacking": "models/stack_phase4.json (VAL-fit, frozen before TEST)",
                  "features": feats, "coef": coef, "intercept": b,
                  "temperature": T, "band": [tau_low, tau_high]},
        "test": overall,
        "far_first": far_first,
        "per_group": group_m,
        "reliability": rel,
        "n_test_rows": len(test_rows), "n_scored": n_scored,
        "missing_per_model": dict(missing_per_model),
        "n_incomplete_rows": len(incomplete),
        "honesty": [
            "TEST was scored once with artifacts frozen from VAL; no threshold, "
            "weight, or temperature was chosen or revised after seeing TEST.",
            "Private clips never left this PC; groups are anonymous labels.",
            f"Private test n={sum(1 for r in complete if r['source'].startswith('private-'))} "
            "(small-n snapshot, not a benchmark).",
            "Unseen generators (elevenlabs, piper:hi-rohan, edge:ta-ValluvarNeural) "
            "were held out of train AND val by construction.",
        ],
    }
    (ROOT / "reports" / "model_eval.json").write_text(json.dumps(out, indent=2),
                                                      encoding="utf-8")

    L = ["# Phase 5 — honest TEST evaluation (frozen VAL-fit stack)", "",
         f"Generated: {out['generated_at']}",
         f"Model: {out['model']['name']}; stack frozen from VAL (models/stack_phase4.json).",
         f"Band: [{tau_low}, {tau_high}] (UNCERTAIN between). Temperature T={T}.",
         "",
         "> TEST was never used for training, tuning, calibration, or band "
         "selection. Numbers are as measured; no targets were pre-registered "
         "and nothing was tuned to look good.", "",
         "## FAR-first headline",
         f"- **FAR on human audio @0.5: {overall['far_human_at_0.5']}** "
         f"(n_human={overall['n_human']})",
         f"- FRR on synthetic audio @0.5: {overall['frr_fake_at_0.5']} "
         f"(n_fake={overall['n_fake']})",
         f"- Coverage inside band: {overall.get('coverage')} | selective accuracy: "
         f"{overall.get('selective_accuracy')}",
         f"- Overall: AUC={overall['auc']} EER={json.dumps(overall['eer'])} "
         f"acc={overall['accuracy']} balanced={overall['balanced_accuracy']}",
         f"- Band confusion: {json.dumps(overall['confusion_band'])}", "",
         "## Per group (FAR on human first, then miss rate on synthetic)"]
    for g, m in group_m.items():
        kind = "human" if m["n_human"] else "fake"
        if kind == "human":
            L.append(f"- **{g}** (human, n={m['n']}): FAR={m['far_human_at_0.5']} "
                     f"AUC={m['auc']} conf={json.dumps(m['confusion_band'])}")
        else:
            L.append(f"- {g} (synthetic, n={m['n']}): miss={m['frr_fake_at_0.5']} "
                     f"AUC={m['auc']} conf={json.dumps(m['confusion_band'])}")
    L += ["", "## Reliability (calibrated P(AI) bins vs binary accuracy on TEST)",
          "| bin | n | accuracy | avg_p |", "| --- | --- | --- | --- |"]
    for b_ in rel:
        L.append(f"| {b_['bin']} | {b_['n']} | {b_.get('accuracy')} | {b_.get('avg_p')} |")
    L += ["", "## Coverage of the score cache (honesty)",
          f"- test rows: {len(test_rows)}; fully scored: {n_scored}; "
          f"incomplete (excluded from metrics): {len(incomplete)}",
          f"- missing per model: {dict(missing_per_model) or 'none'}", "",
          "## Honest reading"]
    L += ["- " + h for h in out["honesty"]]
    (ROOT / "reports" / "model_eval.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    print(f"FAR_human={overall['far_human_at_0.5']} FRR_fake={overall['frr_fake_at_0.5']} "
          f"AUC={overall['auc']} coverage={overall.get('coverage')}")
    print("wrote reports/model_eval.json + reports/model_eval.md (Phase 5)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
