"""Phase 2 step 2: build data/phase2_manifest.csv (speaker-/generator-disjoint).

- REAL train/val/test: LibriSpeech test-clean clips from data/manifest.csv,
  speakers disjoint across splits; speakers 260/4446/5142/6930 (used in
  samples/) are EXCLUDED everywhere (reserved for my_samples isolation).
- FAKE train/val/test: locally synthesized crude voice-conversion stopgap
  (librosa pitch-shift / time-stretch, family vc_librosa_crude, clearly
  labelled; NOT a real generator-diversity solution — see data_plan.md).
  Fake clips inherit disjoint speaker pools from the real split.
- MY_SAMPLES: all of samples/labels.csv (19 files), split="my_samples",
  never train/val. Untouched until Phase 4.

Writes data/synth_vc/*.wav + data/phase2_manifest.csv +
reports/data_splits.md. Verifies disjointness programmatically.
"""
import csv, random
from collections import Counter, defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SEED = 7
N_TRAIN_EACH, N_VAL_EACH, N_TEST_EACH = 120, 40, 40
RESERVED_SPEAKERS = {"spk-260", "spk-4446", "spk-5142", "spk-6930"}

def main() -> int:
    import numpy as np
    import soundfile as sf
    import librosa
    rng = random.Random(SEED)
    rows = [r for r in csv.DictReader(open(ROOT / "data" / "manifest.csv", encoding="utf-8"))
            if (ROOT / r["path"].replace("\\", "/")).is_file()]
    by_spk = defaultdict(list)
    for r in rows:
        if r.get("speaker_id") in RESERVED_SPEAKERS:
            continue
        by_spk[r["speaker_id"]].append(r)
    speakers = sorted(by_spk)
    print(f"usable speakers (excl. reserved {sorted(RESERVED_SPEAKERS)}): {len(speakers)}")
    rng.shuffle(speakers)
    n_tr = max(1, int(len(speakers) * 0.68)); n_va = max(1, int(len(speakers) * 0.16))
    tr_spk, va_spk, te_spk = set(speakers[:n_tr]), set(speakers[n_tr:n_tr + n_va]), set(speakers[n_tr + n_va:])
    print(f"train speakers={len(tr_spk)} val={len(va_spk)} test={len(te_spk)}")
    for s in tr_spk: rng.shuffle(by_spk[s])
    for s in va_spk: rng.shuffle(by_spk[s])
    for s in te_spk: rng.shuffle(by_spk[s])

    def take(spkset, n):
        pool = [r for s in spkset for r in by_spk[s]]
        rng.shuffle(pool)
        return pool[:n]

    synth_dir = ROOT / "data" / "synth_vc"
    synth_dir.mkdir(parents=True, exist_ok=True)

    def crude_vc(src_row, setting: str, idx: int) -> dict:
        src = ROOT / src_row["path"].replace("\\", "/")
        arr, sr = sf.read(str(src), dtype="float32", always_2d=False)
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 2: arr = arr.mean(axis=1)
        if sr != 16000:
            dur = len(arr) / float(sr); tgt = int(dur * 16000)
            arr = np.interp(np.linspace(0, 1, tgt), np.linspace(0, 1, len(arr)), arr).astype(np.float32)
            sr = 16000
        if setting == "ps_up":
            y = librosa.effects.pitch_shift(arr, sr=16000, n_steps=4)
        elif setting == "ps_down":
            y = librosa.effects.pitch_shift(arr, sr=16000, n_steps=-4)
        elif setting == "ts_slow":
            y = librosa.effects.time_stretch(arr, rate=0.85)
        else:
            y = librosa.effects.time_stretch(arr, rate=1.18)
        y = np.clip(np.asarray(y, dtype=np.float32), -1.0, 1.0)
        # cap 6 s
        y = y[:16000 * 6]
        name = f"vc_{setting}_{src_row['speaker_id']}_{idx:03d}.wav"
        sf.write(str(synth_dir / name), y, 16000)
        return {"path": f"data/synth_vc/{name}", "label_origin": "AI_VOICE_CLONE_OR_CONVERSION",
                "generator_family": "vc_librosa_crude", "generator_name": f"librosa_{setting}",
                "language": src_row.get("language", "en"), "speaker_id": f"src-{src_row['speaker_id']}-{setting}",
                "source_dataset": "local_synth_vc", "codec": "wav", "channel": "mono",
                "note": "STOPGAP crude VC (pitch/time warp of LibriSpeech); not real generator diversity."}

    out = []
    # REAL splits
    for r in take(tr_spk, N_TRAIN_EACH): out.append({**{k: r.get(k, "") for k in
        ["path", "label_origin", "generator_family", "generator_name", "language", "speaker_id", "source_dataset", "codec", "channel"]}, "split": "train"})
    for r in take(va_spk, N_VAL_EACH): out.append({**{k: r.get(k, "") for k in
        ["path", "label_origin", "generator_family", "generator_name", "language", "speaker_id", "source_dataset", "codec", "channel"]}, "split": "val"})
    for r in take(te_spk, N_TEST_EACH): out.append({**{k: r.get(k, "") for k in
        ["path", "label_origin", "generator_family", "generator_name", "language", "speaker_id", "source_dataset", "codec", "channel"]}, "split": "test"})
    # FAKE splits (settings disjoint: train ps_up/ps_down, val ts_slow, test ts_fast)
    fake_plan = [("train", tr_spk, N_TRAIN_EACH, ["ps_up", "ps_down"]), ("val", va_spk, N_VAL_EACH, ["ts_slow"]), ("test", te_spk, N_TEST_EACH, ["ts_fast"])]
    for split, spkset, n, settings in fake_plan:
        pool = [r for s in spkset for r in by_spk[s]]
        rng.shuffle(pool)
        for i in range(n):
            src = pool[i % len(pool)]
            row = crude_vc(src, settings[i % len(settings)], i)
            row["split"] = split
            out.append({k: row.get(k, "") for k in
                ["path", "label_origin", "generator_family", "generator_name", "language", "speaker_id", "source_dataset", "codec", "channel", "split"]})
    # MY_SAMPLES (untouched)
    for r in csv.DictReader(open(ROOT / "samples" / "labels.csv", encoding="utf-8")):
        fam = "human" if r["y"] == "0" else r.get("source", "samples")
        out.append({"path": r["file"], "label_origin": "HUMAN_LIVE" if r["y"] == "0" else "AI_TTS",
                    "generator_family": fam, "generator_name": r.get("group", ""), "language": "",
                    "speaker_id": "my_samples", "source_dataset": "my_samples",
                    "split": "my_samples", "codec": "", "channel": ""})
    # verify disjointness (train/val/test only).
    # Speakers: strict everywhere. Fake generators: strict (real human speech
    # is one class by design, so 'librispeech' recurring is expected, not leakage).
    for key in ("speaker_id",):
        seen = defaultdict(set)
        for r in out:
            if r["split"] in ("train", "val", "test"):
                seen[r[key]].add(r["split"])
        leaked = {k: sorted(v) for k, v in seen.items() if len(v) > 1}
        print(f"{key} overlap train/val/test: {len(leaked)}" + (f" {str(leaked)[:300]}" if leaked else " (clean)"))
    fseen = defaultdict(set)
    for r in out:
        if r["split"] in ("train", "val", "test") and r.get("generator_family", "").startswith("vc_"):
            fseen[r["generator_name"]].add(r["split"])
    fleaked = {k: sorted(v) for k, v in fseen.items() if len(v) > 1}
    print(f"fake-generator overlap train/val/test: {len(fleaked)}" + (f" {str(fleaked)[:300]}" if fleaked else " (clean)"))
    # my_samples isolation: generator_names in train?
    train_gens = {r["generator_name"] for r in out if r["split"] == "train"}
    my_gens = {r["generator_name"] for r in out if r["split"] == "my_samples"}
    inter = train_gens & my_gens
    print(f"train generators intersect my_samples generators: {sorted(inter) if inter else 'EMPTY (good)'}")
    cols = ["path", "label_origin", "generator_family", "generator_name", "language", "speaker_id", "source_dataset", "split", "codec", "channel"]
    with open(ROOT / "data" / "phase2_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in out: w.writerow({k: r.get(k, "") for k in cols})
    by = Counter((r["split"], r["label_origin"]) for r in out)
    md = ["# Phase 2 splits (speaker-/generator-disjoint + my_samples)", "",
          f"rows={len(out)} " + ", ".join(f"{k}={v}" for k, v in sorted(by.items())), "",
          f"train speakers: {len(tr_spk)}, val: {len(va_spk)}, test: {len(te_spk)} (reserved {sorted(RESERVED_SPEAKERS)} excluded).",
          "train fake settings: ps_up/ps_down; val: ts_slow; test: ts_fast (sub-setting split inside one crude family — stopgap, stated).",
          "my_samples (19 files, SAPI/ChatGPT/Windows-TTS generators) appears in NO train/val/test row.",
          "STOPGAP WARNING: train fake diversity is one crude librosa VC family. Real generator diversity (open TTS/VC, ASVspoof/WaveFake/MLAAD) is Phase-3-notebook + user-clip work (see data_plan.md)."]
    (ROOT / "reports" / "data_splits.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote data/phase2_manifest.csv ({len(out)} rows) + reports/data_splits.md")
    return 0
if __name__ == "__main__": raise SystemExit(main())
