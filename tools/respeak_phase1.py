"""Recompute speaker_id in data/manifest_phase1.csv from real provenance.

- fleurs: join audio basename -> speaker id from the per-split TSVs
  (columns: id, file, transcript, raw, phones, speaker, gender).
- slr65 (Google crowdsourced Tamil): taf_<SPK>_<...>.wav -> SPK.
- slr118 (GramVaani telephone): <dist>-<CALLER>-<n>.mp3 -> CALLER.
- slr12: unchanged (LibriSpeech speaker dirs, already correct).

Rewrites speaker_id in place and prints verified distinct-speaker counts.
"""
from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MANIFEST = ROOT / "data" / "manifest_phase1.csv"
TSV_DIR = ROOT / "data" / "_dl" / "tsv" / "data"


def load_fleurs() -> dict[str, str]:
    m: dict[str, str] = {}
    for lang in ["hi_in", "ta_in", "en_us"]:
        for split in ["train", "dev", "test"]:
            p = TSV_DIR / lang / f"{split}.tsv"
            if not p.is_file():
                continue
            for line in p.read_text(encoding="utf-8").splitlines():
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 7 and parts[1].endswith((".wav", ".mp3")):
                    m[parts[1]] = f"fleurs-{lang}-{parts[-2]}-{parts[-1]}"
    return m


def main() -> int:
    rows = list(csv.DictReader(open(MANIFEST, newline="", encoding="utf-8")))
    fl = load_fleurs()
    print(f"fleurs speaker map: {len(fl)} utterances")
    fixed = {"fleurs": 0, "slr65": 0, "slr118": 0}
    for r in rows:
        src, base = r["source"], r["orig_path"].rsplit("/", 1)[-1]
        if src.startswith("fleurs") and base in fl:
            r["speaker_id"] = fl[base]
            fixed["fleurs"] += 1
        elif src.startswith("slr65"):
            grp = base.split("_")
            if len(grp) >= 3:
                r["speaker_id"] = f"tamil-crowd-{grp[1]}"
                fixed["slr65"] += 1
        elif src.startswith("slr118"):
            grp = Path(base).stem.split("-")
            if len(grp) >= 3:
                r["speaker_id"] = f"gv-caller-{grp[0]}-{grp[1]}"
                fixed["slr118"] += 1
    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("remapped:", fixed)
    by_src = {}
    for r in rows:
        by_src.setdefault(r["source"], set()).add(r["speaker_id"])
    tot = 0
    for src in sorted(by_src):
        n = len(by_src[src])
        tot += n
        print(f"{src:18s} clips={sum(1 for r in rows if r['source']==src):5d} speakers={n}")
    print(f"sum (cross-source overlaps possible) = {tot}")
    # cross-check: same speaker string in two sources?
    inv: dict[str, set] = {}
    for r in rows:
        inv.setdefault(r["speaker_id"], set()).add(r["source"])
    print("speaker ids appearing in >1 source:", sum(1 for v in inv.values() if len(v) > 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
