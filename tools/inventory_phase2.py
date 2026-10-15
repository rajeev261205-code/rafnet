"""Phase 2 step 1: inventory existing data holdings (no downloads).

Reads data/manifest.csv + samples/labels.csv and writes
reports/data_inventory.json + reports/data_inventory.md with counts by
split/label/speaker/generator, plus a leakage audit of the CURRENT manifest
(speaker/generator overlap across splits) and a gap list vs the task brief.
"""
import csv, json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def main() -> int:
    man = ROOT / "data" / "manifest.csv"
    lab = ROOT / "samples" / "labels.csv"
    rows = list(csv.DictReader(open(man, encoding="utf-8"))) if man.is_file() else []
    slabs = list(csv.DictReader(open(lab, encoding="utf-8"))) if lab.is_file() else []
    # existing manifest stats
    splits = Counter(r.get("split", "?") for r in rows)
    labels = Counter(r.get("label_origin", "?") for r in rows)
    speakers = sorted({r.get("speaker_id", "?") for r in rows})
    spk_split = {}
    for r in rows:
        spk_split.setdefault(r.get("speaker_id", "?"), set()).add(r.get("split", "?"))
    overlap_spk = {s: sorted(v) for s, v in spk_split.items() if len(v) > 1}
    gens = Counter((r.get("generator_family", "?"), r.get("generator_name", "?")) for r in rows)
    gen_split = {}
    for r in rows:
        gen_split.setdefault((r.get("generator_family", "?"), r.get("generator_name", "?")), set()).add(r.get("split", "?"))
    overlap_gen = {str(k): sorted(v) for k, v in gen_split.items() if len(v) > 1}
    srcs = Counter(r.get("source_dataset", "?") for r in rows)
    # samples stats
    s_groups = Counter((r.get("label", "?"), r.get("group", "?")) for r in slabs)
    s_missing = [r["file"] for r in slabs if not (ROOT / r["file"].replace("\\", "/")).is_file()]
    # raw dir size hint
    raw_files = sum(1 for _ in (ROOT / "data" / "raw").rglob("*") if _.is_file()) if (ROOT / "data" / "raw").is_dir() else 0
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_manifest_rows": len(rows),
        "data_manifest_by_split": dict(splits),
        "data_manifest_by_label": dict(labels),
        "data_manifest_by_source": dict(srcs),
        "data_manifest_speakers_n": len(speakers),
        "data_manifest_speakers": speakers[:40],
        "data_manifest_generators": {str(k): v for k, v in gens.items()},
        "leakage_audit": {
            "speakers_in_multiple_splits": overlap_spk,
            "speakers_overlap_n": len(overlap_spk),
            "generators_in_multiple_splits": overlap_gen,
            "note": "Any non-empty overlap = leakage: same speaker/generator on both sides of a split boundary.",
        },
        "samples_labels_rows": len(slabs),
        "samples_by_label_group": {f"{k[0]}/{k[1]}": v for k, v in s_groups.items()},
        "samples_missing_files": s_missing,
        "data_raw_files_n": raw_files,
        "gaps_vs_brief": [
            "REAL: no Indian English/Hindi/Tamil, no WhatsApp/Opus voice notes, no phone-line (8 kHz), no laptop-mic/noisy-room sets — only LibriSpeech clean + samples/.",
            "FAKE: only SAPI David/Zira + 1 Windows-TTS + 1 ChatGPT loopback; no open-source TTS/VC diversity (Bark/Coqui/XTTS/YourTTS/RVC/Seed-VC), no commercial TTS (no user keys), no voice conversion at all.",
            "PUBLIC SETS: none downloaded yet (ASVspoof 2019/2021, In-the-Wild, WaveFake, MLAAD, Common Voice). See reports/data_plan.md for per-set license/size/approval.",
            "data/manifest.csv is REAL-ONLY (LibriSpeech) — unusable as a spoof train manifest until fake rows are added and splits rebuilt speaker-/generator-disjoint.",
        ],
    }
    (ROOT / "reports" / "data_inventory.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    L = ["# Data inventory (existing holdings, no downloads)", "",
         f"Generated: {report['generated_at']}", "",
         f"data/manifest.csv: {len(rows)} rows, splits={dict(splits)}, labels={dict(labels)}, sources={dict(srcs)}",
         f"speakers: {len(speakers)}; speakers spanning splits (LEAKAGE): {len(overlap_spk)}",
         (f"  leaked: {json.dumps(overlap_spk)[:400]}" if overlap_spk else "  none leaked across splits (but single-source REAL-only)."),
         f"generators: {json.dumps({str(k): v for k, v in gens.items()})}",
         f"samples/labels.csv: {len(slabs)} rows; " + "; ".join(f"{k}={v}" for k, v in s_groups.items()),
         f"samples missing files: {s_missing or 'none'}",
         f"data/raw files on disk: {raw_files}", "",
         "## Gaps vs task brief"]
    L += [f"- {g}" for g in report["gaps_vs_brief"]]
    (ROOT / "reports" / "data_inventory.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"manifest rows={len(rows)} speakers={len(speakers)} leaked_speakers={len(overlap_spk)} samples={len(slabs)} raw_files={raw_files}")
    print("wrote reports/data_inventory.json/.md")
    return 0
if __name__ == "__main__": raise SystemExit(main())
