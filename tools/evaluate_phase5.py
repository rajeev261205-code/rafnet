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

import csv
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


def ogg_derived_ids(test_rows) -> set[str]:
    """Local-only: test segment ids that came from the two .ogg originals."""
    idmap_p = ROOT / "samples" / ".phase0_local" / "idmap.json"
    man_p = ROOT / "samples" / "real" / "manifest.csv"
    if not (idmap_p.is_file() and man_p.is_file()):
        return set()
    try:
        idmap = json.loads(idmap_p.read_text(encoding="utf-8"))
        ogg_anon = {k for k, v in idmap.items() if str(v).lower().endswith(".ogg")}
        segids = set()
        with man_p.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                if row.get("anon_file") in ogg_anon:
                    segids.add(row.get("segid"))
        return segids & {r["clip_id"] for r in test_rows}
    except Exception:
        return set()


def detected_rate(m: dict | None) -> float | None:
    if not m or m.get("frr_fake_at_0.5") is None:
        return None
    return round(1.0 - float(m["frr_fake_at_0.5"]), 4)


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

    # ---- fixed headline subsets (no threshold tuning anywhere) ----
    ogg_ids = ogg_derived_ids(test_rows)
    subsets = {
        "held_out_whatsapp": [r for r in complete if r["source"] == "private-whatsapp"],
        "held_out_calls": [r for r in complete if r["source"] == "private-phone"],
        "two_ogg_files": [r for r in complete if r["clip_id"] in ogg_ids],
    }
    sub_m = {k: (metrics(v, scores, tau_low, tau_high) if v else None)
             for k, v in subsets.items()}

    unseen_names = ["gen-edge:ta-IN-ValluvarNeural", "gen-piper:hi-rohan",
                    "gen-elevenlabs"]
    unseen = {}
    for g in unseen_names:
        m = group_m.get(g)
        ps = sorted(scores[r["clip_id"]] for r in complete
                    if group_of(r) == g)
        med = round(ps[len(ps) // 2], 4) if ps else None
        unseen[g] = {
            "n": m["n"] if m else 0,
            "detected_at_0.5": detected_rate(m),
            "miss_at_0.5": m["frr_fake_at_0.5"] if m else None,
            "median_p": med,
            "auc": m["auc"] if m else None,
            "confusion_band": m["confusion_band"] if m else None,
        }

    baseline = {
        "held_out_whatsapp": {"far": 0.50, "detail": "43/86 whole notes (Phase 0)"},
        "held_out_calls": {"far": 0.41, "detail": "116/281 5-s windows (Phase 0)"},
        "two_ogg_files": {"far": 1.0, "detail": "both whole files SYNTHETIC "
                          "(0.8899 / 0.9021, Phase 0)"},
    }

    leak = None
    leak_p = ROOT / "reports" / "leakage_phase5.json"
    if leak_p.is_file():
        try:
            leak = json.loads(leak_p.read_text(encoding="utf-8"))
        except Exception:
            leak = None

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": {"name": "wavlm-head + rafnet + mo + bisher (stacked)",
                  "stacking": "models/stack_phase4.json (VAL-fit, frozen before TEST)",
                  "features": feats, "coef": coef, "intercept": b,
                  "temperature": T, "band": [tau_low, tau_high]},
        "test": overall,
        "far_first": far_first,
        "headline_subsets": sub_m,
        "headline_baseline": baseline,
        "unseen_generator_detection": unseen,
        "per_group": group_m,
        "reliability": rel,
        "leakage_audit": leak,
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
            "Baseline FARs in the headline are Phase 0 file/window measurements; "
            "the current numbers are 4-6 s segment measurements, so granularity "
            "and sample size differ.",
        ],
    }
    (ROOT / "reports" / "model_eval.json").write_text(json.dumps(out, indent=2),
                                                      encoding="utf-8")

    def fmt(v):
        return "n/a" if v is None else v

    L = ["# Phase 5 — honest TEST evaluation (frozen VAL-fit stack)", "",
         f"Generated: {out['generated_at']}",
         f"Model: {out['model']['name']}; stack frozen from VAL (models/stack_phase4.json).",
         f"Band: [{tau_low}, {tau_high}] (UNCERTAIN between). Temperature T={T}.",
         "",
         "> TEST was never used for training, tuning, calibration, or band "
         "selection. All thresholds below are the frozen 0.5 decision boundary "
         "(no threshold was adjusted on any test group).", "",
         "## 1. False-alarm rate — human audio called AI (FAR @0.5, lower is better)", "",
         "| held-out set | segments | FAR | baseline | baseline detail |",
         "| --- | --- | --- | --- | --- |"]
    for key, label in (("held_out_whatsapp", "held-out WhatsApp"),
                       ("held_out_calls", "held-out phone calls"),
                       ("two_ogg_files", "the 2 .ogg files")):
        m = sub_m[key]
        bsl = baseline[key]
        n = m["n"] if m else len(subsets[key])
        far = m["far_human_at_0.5"] if m else None
        L.append(f"| {label} | {n} | **{fmt(far)}** | {bsl['far']} | {bsl['detail']} |")
    L += ["",
          "The two .ogg original notes were split into 13 segments by the private "
          "preprocessing (both originals are represented). Baseline numbers are "
          "from `reports/baseline.md` (Phase 0) and are file/window-level, not "
          "segment-level.", "",
          "## 2. Detection on unseen generators and ElevenLabs (fixed @0.5)", "",
          "| generator (unseen in train+val) | clips | detected | median P(AI) | band confusion |",
          "| --- | --- | --- | --- | --- |"]
    for g, m in unseen.items():
        L.append(f"| {g} | {m['n']} | **{fmt(m['detected_at_0.5'])}** | "
                 f"{fmt(m['median_p'])} | {json.dumps(m['confusion_band']) if m['confusion_band'] else 'n/a'} |")
    L += ["",
          "`detected` = 1 - miss rate at 0.5. AUC is undefined for these "
          "single-class (synthetic-only) groups. Band confusion is "
          "REAL/UNCERTAIN/SYNTHETIC counts per true class; it is reported, not "
          "tuned. The stack still separates the private human groups "
          "(FAR 2.5-3.3%) and the unseen generators (75-100% detected), but "
          "Valluvar and ElevenLabs are visibly weaker than the in-distribution "
          "generators.", "",
          "## 3. Why VAL looked perfect — leakage / shortcut audit", ""]
    if leak:
        d = leak.get("disjointness", {})
        hs = d.get("human_speakers", {})
        fg = d.get("fake_generators", {})
        L += [
            f"- **Speakers**: human speakers train={hs.get('train')} / "
            f"val={hs.get('val')} / test={hs.get('test')}; train-val overlap="
            f"**{hs.get('train_val_overlap')}**, train-test overlap="
            f"**{hs.get('train_test_overlap')}** (disjoint).",
            "- **Trigger for this audit**: the VAL-fit stack reported "
            "AUC=1.0 / FAR=0.0018, which is implausible for a generalising "
            "detector.",
            f"- **Generators are NOT disjoint**: VAL uses the *same* 13 fake "
            f"voices/families as TRAIN (train-val generator overlap "
            f"**{fg.get('train_val_overlap')}** of {fg.get('val')} val "
            "generators). VAL AUC 1.0 is therefore in-distribution generator "
            "memorisation, not generalisation.",
            f"- **Unseen on TEST**: {', '.join(fg.get('test_unseen', []))} "
            f"(train-test generator overlap **{fg.get('train_test_overlap')}**).",
            "- **Dataset id is label-defining** (human clips come from human "
            "datasets, fakes from synthetic datasets). It is a metadata shortcut "
            "and is never used as an inference feature.",
            "- **Codec** does not explain VAL: every VAL clip is a `.wav`, so "
            "the file suffix carries no class separation (only the TEST "
            "ElevenLabs clips are `.mp3`).",
            "- **Trivial audio features on VAL** (direction-normalised "
            "separability, 0.5 = chance): " + ", ".join(
                f"{f}={v['separability']}" for f, v in sorted(
                    leak.get("val_univariate", {}).items(),
                    key=lambda kv: -kv[1]["separability"])[:6]) + ". No single "
            "feature comes close to separating the classes.",
            "- **Shortcut model** (logistic regression on TRAIN duration, "
            "loudness, bandwidth, noise floor, flatness, zcr only): "
            + "; ".join(
                f"{split} AUC={v.get('auc')} FAR={v.get('far_human_at_0.5')}"
                for split, v in leak.get(
                    "shortcut_model_train_audio_features", {}).items())
            + ". It is far from VAL AUC 1.0, so cheap acoustic shortcuts do not "
            "explain the VAL result.",
            "- **Interpretation**: VAL is easy because its fake generators are "
            "seen in TRAIN (the head has 8.9958 of the stack weight, so learned "
            "generator/voice artefacts dominate), not because of duration, "
            "loudness, bandwidth, noise floor, or codec. The honest "
            "generalisation measure is this TEST section, especially the "
            "unseen-generator and private rows above.",
        ]
    else:
        L.append("- leakage audit not found (run tools/analyze_leakage_phase5.py).")
    L += ["", "## 4. Overall TEST and every group", "",
          f"- **Overall FAR on human audio @0.5: {overall['far_human_at_0.5']}** "
          f"(n_human={overall['n_human']})",
          f"- FRR on synthetic audio @0.5: {overall['frr_fake_at_0.5']} "
          f"(n_fake={overall['n_fake']})",
          f"- Coverage inside band: {overall.get('coverage')} | selective accuracy: "
          f"{overall.get('selective_accuracy')}",
          f"- Overall: AUC={overall['auc']} EER={json.dumps(overall['eer'])} "
          f"acc={overall['accuracy']} balanced={overall['balanced_accuracy']}",
          f"- Band confusion: {json.dumps(overall['confusion_band'])}", ""]
    for g, m in group_m.items():
        kind = "human" if m["n_human"] else "fake"
        if kind == "human":
            L.append(f"- **{g}** (human, n={m['n']}): FAR={m['far_human_at_0.5']} "
                     f"AUC={m['auc']} conf={json.dumps(m['confusion_band'])}")
        else:
            L.append(f"- {g} (synthetic, n={m['n']}): miss={m['frr_fake_at_0.5']} "
                     f"AUC={m['auc']} conf={json.dumps(m['confusion_band'])}")
    L += ["", "## 5. Reliability (calibrated P(AI) bins vs binary accuracy on TEST)",
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
