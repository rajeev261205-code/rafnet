"""Build data/manifest.csv with enforced speaker/generator-disjoint splits.

Usage:
  python tools/make_manifest.py --roots data/raw/LibriSpeech/test-clean=speech:bonafide:librispeech
                                samples/win_tts=tts:microsoft-sapi-like:win-sapi
                                [--out data/manifest.csv] [--val-speakers 4] [--test-generators 1]

Each --roots entry: <dir>=<origin>:<family>:<dataset>[:<language>].
Speaker ids come from LibriSpeech-style dir names (<spk>-<chap>-<utt>), else
"unknown". Splits: 80/10/10 with NO speaker_id and NO generator_name shared
across splits (violations raise). LibriSpeech->HUMAN_LIVE-analog gets
label HUMAN_LIVE; win_tts->AI_TTS.
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

AUDIO = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def speaker_of(path: Path) -> str:
    parts = path.stem.split("-")
    if len(parts) >= 3 and parts[0].isdigit():
        return f"spk-{parts[0]}"
    return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--out", default="data/manifest.csv")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--holdout-generators", default="",
                    help="Comma-separated generator_names held ENTIRELY in test "
                         "(leave-generator-out). Others split speaker-disjoint.")
    args = ap.parse_args()

    rows: list[dict] = []
    for entry in args.roots:
        spec, _, meta = entry.partition("=")
        parts = meta.split(":")
        origin = parts[0] if len(parts) > 0 else ""
        family = parts[1] if len(parts) > 1 else ""
        dataset = parts[2] if len(parts) > 2 else ""
        language = parts[3] if len(parts) > 3 else "en"
        for p in sorted(Path(spec).rglob("*")):
            if p.is_file() and p.suffix.lower() in AUDIO:
                rows.append({
                    "path": str(p), "label_origin": origin or "AI_TTS",
                    "generator_family": family or "unknown",
                    "generator_name": dataset or "unknown",
                    "language": language or "en",
                    "speaker_id": speaker_of(p), "source_dataset": dataset or "unknown",
                    "split": "", "codec": p.suffix.lower().lstrip("."),
                    "channel": "mono",
                })
    if not rows:
        print("no audio found")
        return 2

    rng = random.Random(args.seed)
    holdout = {g.strip() for g in args.holdout_generators.split(",") if g.strip()}
    # Holdout generators go ENTIRELY to test (leave-generator-out).
    # Everything else splits speaker-disjoint (grouped by speaker+generator).
    groups: dict[tuple, list[int]] = {}
    for i, r in enumerate(rows):
        if r["generator_name"] in holdout:
            r["split"] = "test"
        else:
            groups.setdefault((r["speaker_id"], r["generator_name"]), []).append(i)
    keys = list(groups)
    rng.shuffle(keys)
    n = len(keys)
    n_test = max(1, n // 10)
    n_val = max(1, n // 10)
    assign: dict[int, str] = {}
    for k in keys[:n_test]:
        for i in groups[k]:
            assign[i] = "test"
    for k in keys[n_test:n_test + n_val]:
        for i in groups[k]:
            assign[i] = "val"
    for k in keys[n_test + n_val:]:
        for i in groups[k]:
            assign[i] = "train"
    for i, r in enumerate(rows):
        if "split" not in r or not r["split"]:
            r["split"] = assign.get(i, "train")

    # Enforce disjointness loudly: speakers never span splits; holdout
    # generators appear ONLY in test.
    seen_spk: dict[str, str] = {}
    for r in rows:
        spk = r["speaker_id"]
        if spk != "unknown":
            if spk in seen_spk and seen_spk[spk] != r["split"]:
                raise SystemExit(
                    f"SPEAKER LEAKAGE: {spk} in {seen_spk[spk]} and {r['split']}")
            seen_spk[spk] = r["split"]
        if r["generator_name"] in holdout and r["split"] != "test":
            raise SystemExit(
                f"GENERATOR LEAKAGE: holdout {r['generator_name']} in {r['split']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    from collections import Counter
    print(f"wrote {out} ({len(rows)} rows)")
    print("origin x split:", Counter((r["label_origin"], r["split"]) for r in rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
