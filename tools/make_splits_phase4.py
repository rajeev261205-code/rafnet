"""Phase 4 splits: speaker- AND generator-disjoint train/val/test (local).

Human (public, data/manifest_phase1.csv): per-source speaker sort ->
70/15/15 speaker split (deterministic). Private adapt -> train (small share);
private TEST groups (whatsapp_late, held-out contact, 2 .ogg) -> test only.
Fake (data/manifest_phase2.csv): train-generator clips -> train/val;
whole unseen generators (elevenlabs, piper:hi-rohan, edge:ta-Valluvar) -> test.
Verifies zero speaker/generator/group overlap across splits; writes
data/splits_phase4.json (local, git-ignored) + prints the audit.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "data" / "splits_phase4.json"


def main() -> int:
    real_rows = list(csv.DictReader(open(ROOT / "data" / "manifest_phase1.csv",
                                         newline="", encoding="utf-8")))
    def corpus_of(src: str) -> str:
        if src.startswith("fleurs_hi"):
            return "fleurs-hi"
        if src.startswith("fleurs_ta"):
            return "fleurs-ta"
        if src.startswith("fleurs_en"):
            return "fleurs-en"
        if src.startswith("slr65"):
            return "slr65"
        if src.startswith("slr118"):
            return "slr118"
        if src.startswith("slr12"):
            return "slr12"
        return src

    # per-CORPUS speaker -> split (dev/test of one corpus share speakers)
    by_src: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in real_rows:
        by_src[corpus_of(r["source"])][r["speaker_id"]].append(r)
    train, val, test = [], [], []
    for src in sorted(by_src):
        spks = sorted(by_src[src])
        n = len(spks)
        n_test = max(1, int(round(n * 0.15)))
        n_val = max(1, int(round(n * 0.15)))
        test_spk = set(spks[:n_test])
        val_spk = set(spks[n_test:n_test + n_val])
        for s in spks:
            dest = test if s in test_spk else (val if s in val_spk else train)
            for r in by_src[src][s]:
                dest.append({"clip_id": r["clip_id"], "y": 0,
                             "path": str(ROOT / "data" / "phase1" / "human" / r["source"] /
                                         (r["clip_id"] + ".wav")),
                             "source": src, "speaker": s, "license": r["license"]})
    # private human
    n_ad = n_te = 0
    with open(ROOT / "samples" / "real" / "manifest.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            e = {"clip_id": r["segid"], "y": 0,
                 "path": str(ROOT / "samples" / "real" / "segments" / r["split"] /
                             (r["segid"] + ".wav")),
                 "source": "private-" + r["source"], "speaker": "private-" + r["group"],
                 "license": "private-local-only"}
            if r["split"] == "adapt":
                train.append(e)
                n_ad += 1
            elif r["split"] == "test":
                test.append(e)
                n_te += 1
    # fakes
    UNSEEN = {"elevenlabs", "piper:hi-rohan", "edge:ta-IN-ValluvarNeural"}
    n_ftr = n_fva = n_fte = 0
    with open(ROOT / "data" / "manifest_phase2.csv", newline="", encoding="utf-8") as f:
        fake_rows = list(csv.DictReader(f))
    with open(ROOT / "data" / "manifest_phase2_elevenlabs.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("status") == "ok":
                fake_rows.append({"clip_id": r["clip_id"], "engine": "elevenlabs",
                                  "generator": "elevenlabs", "lang": r["lang"],
                                  "split": "test_unseen",
                                  "file": r["file"], "duration_s": "6.0"})
    rng_spk = defaultdict(list)
    for r in fake_rows:
        if (r.get("split") == "test_unseen") or (r.get("generator") in UNSEEN):
            test.append({"clip_id": r["clip_id"], "y": 1,
                         "path": str(ROOT / r["file"]) if not str(r.get("file", "")).endswith(".mp3")
                         else str(ROOT / r["file"]),
                         "source": "fake-" + r.get("engine", "?"),
                         "speaker": "gen-" + r.get("generator", "?"),
                         "license": "synthetic-local"})
            n_fte += 1
        else:
            rng_spk[r.get("generator", "?")].append(r)
    import random

    rng = random.Random(11)
    for g in sorted(rng_spk):
        clips = rng_spk[g][:]
        rng.shuffle(clips)
        nv = max(2, int(round(len(clips) * 0.15)))
        for r in clips[:nv]:
            val.append({"clip_id": r["clip_id"], "y": 1,
                        "path": str(ROOT / r["file"]),
                        "source": "fake-" + r.get("engine", "?"),
                        "speaker": "gen-" + g, "license": "synthetic-local"})
            n_fva += 1
        for r in clips[nv:]:
            train.append({"clip_id": r["clip_id"], "y": 1,
                          "path": str(ROOT / r["file"]),
                          "source": "fake-" + r.get("engine", "?"),
                          "speaker": "gen-" + g, "license": "synthetic-local"})
            n_ftr += 1

    # ---- leakage audit ----
    def keys(rows, k):
        return {r[k] for r in rows}

    def human_spk(rows):
        return {r["speaker"] for r in rows
                if r["y"] == 0 and not r["speaker"].startswith("private-")}

    leak_spk = (human_spk(train) & human_spk(val)) | \
               (human_spk(train) & human_spk(test)) | \
               (human_spk(val) & human_spk(test))
    # private groups: strip prefix check
    priv_test_groups = {r["speaker"] for r in test if r["speaker"].startswith("private-")}
    priv_train_groups = {r["speaker"] for r in train if r["speaker"].startswith("private-")}
    leak_priv = priv_test_groups & priv_train_groups
    audit = {
        "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "train_y": dict(Counter(r["y"] for r in train)),
        "val_y": dict(Counter(r["y"] for r in val)),
        "test_y": dict(Counter(r["y"] for r in test)),
        "private_adapt_in_train": n_ad, "private_test_in_test": n_te,
        "fake_train": n_ftr, "fake_val": n_fva, "fake_test_unseen": n_fte,
        "speaker_overlap_across_splits": sorted(leak_spk),
        "private_group_overlap": sorted(leak_priv),
    }
    assert not leak_spk, f"HUMAN SPEAKER LEAK: {sorted(leak_spk)}"
    assert not leak_priv, f"PRIVATE GROUP LEAK: {leak_priv}"
    # fake-val may share train generators (threshold tuning); unseen must not
    for r in train + val:
        assert not r["speaker"].startswith("gen-elevenlabs"), r
        assert r["speaker"] not in {"gen-piper:hi-rohan", "gen-edge:ta-IN-ValluvarNeural"}, r
    OUT.write_text(json.dumps({"train": train, "val": val, "test": test,
                               "audit": audit}, indent=1), encoding="utf-8")
    print(json.dumps(audit, indent=1))
    print(f"wrote {OUT} (local)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
