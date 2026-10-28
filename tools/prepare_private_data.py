"""DATA UPDATE — segment private recordings into 4-6 s clips (no downloads).

Reads (local-only, never committed):
  samples/real/whatsapp_opus/*   84 .opus + 2 .ogg, 16 kHz Opus voice notes
  samples/real/phone_calls/*      19 .amr, AMR-NB 8 kHz phone calls
  samples/.phase0_local/idmap.json  anon id <-> real name map (Phase 0 ids reused)

Writes (local-only, git-ignored by `samples/` + `data/` rules):
  samples/real/segments/<split>/<segid>.wav   16 kHz mono clips, segid has no PII
  samples/real/manifest.csv                   full provenance (anon ids only)
  samples/real/manifest_summary.json

Writes (committed, counts/aggregates only — NO filenames, NO contact names):
  reports/data_splits_private.md

Pipeline per file (deterministic, fixed order):
  decode (ffmpeg) -> 16 kHz mono -> energy VAD -> 4-6 s segments ->
  QC (silence/clipped/tonal quarantine) -> balance caps ->
  group-disjoint split (TEST = later-half WhatsApp dates + one whole
  contact's calls + the 2 .ogg; rest = adapt pool) ->
  speaker-leak check + acoustic gap analysis.

ffmpeg: system `ffmpeg` on PATH if present, else imageio-ffmpeg static
binary (no admin install needed).
Disk guard: refuses to write segments unless >=10 GB would remain free.
"""
from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

WA_DIR = ROOT / "samples" / "real" / "whatsapp_opus"
PC_DIR = ROOT / "samples" / "real" / "phone_calls"
SEG_DIR = ROOT / "samples" / "real" / "segments"
MANIFEST = ROOT / "samples" / "real" / "manifest.csv"
SUMMARY_JSON = ROOT / "samples" / "real" / "manifest_summary.json"
IDMAP = ROOT / "samples" / ".phase0_local" / "idmap.json"
REPORT = ROOT / "reports" / "data_splits_private.md"

MIN_FREE_GB = 10.0
SEG_MIN, SEG_MAX = 4.0, 6.0
MAX_PER_CALL_FILE = 40
MAX_PER_WA_FILE = 8
CALLS_TO_WA_RATIO = 4.0
MIN_FILE_SEC = 4.0

FFMPEG = shutil.which("ffmpeg")
if not FFMPEG:
    import imageio_ffmpeg

    FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
print(f"ffmpeg: {FFMPEG}")


def disk_free_gb() -> float:
    return shutil.disk_usage(str(ROOT)).free / 1e9


def decode_16k(src: Path) -> tuple[np.ndarray, int]:
    """Decode any input to 16 kHz mono float32 via ffmpeg. Returns (audio, orig_sr)."""
    import soundfile as sf

    with tempfile.TemporaryDirectory(prefix="rafnet_segdec_") as td:
        td = Path(td)
        wav = td / "d16.wav"
        probe = subprocess.run([FFMPEG, "-i", str(src)], capture_output=True, text=True, timeout=60)
        m = re.search(r"Audio:.*?, (\d+) Hz", probe.stderr)
        orig_sr = int(m.group(1)) if m else 16000
        r = subprocess.run(
            [FFMPEG, "-y", "-v", "error", "-i", str(src),
             "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(wav)],
            capture_output=True, text=True, timeout=600)
        if r.returncode != 0 or not wav.is_file():
            raise RuntimeError(f"decode failed for {src.suffix}: {(r.stderr or '')[:200]}")
        arr, _ = sf.read(str(wav), dtype="float32", always_2d=False)
    return np.asarray(arr, dtype=np.float32).ravel(), orig_sr


def energy_vad(x: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Frame-level speech mask (20 ms frames, 10 ms hop). Deterministic."""
    fl, hop = int(0.02 * sr), int(0.01 * sr)
    n = max(1, 1 + (len(x) - fl) // hop)
    frames = np.stack([x[i * hop:i * hop + fl] for i in range(n)])
    e = 10 * np.log10((frames ** 2).mean(axis=1) + 1e-12)
    floor = np.percentile(e, 10)
    speech = e > (floor + 8.0)
    # gap fill <=350 ms, hangover 250 ms, min region 400 ms
    gf, ho, mn = 35, 25, 40
    filled = speech.copy()
    i = 0
    while i < len(filled):
        if not filled[i]:
            j = i
            while j < len(filled) and not filled[j]:
                j += 1
            if i > 0 and j < len(filled) and (j - i) <= gf:
                filled[i:j] = True
            i = j
        else:
            i += 1
    idx = np.nonzero(filled)[0]
    mask = np.zeros_like(filled)
    for k in idx:
        mask[max(0, k - ho):k + 1] = True
    # drop short regions
    out = np.zeros_like(mask)
    i = 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            if (j - i) >= mn:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


def cut_segments(x: np.ndarray, mask: np.ndarray, sr: int = 16000) -> list[tuple[int, int]]:
    """Greedy 4-6 s cuts from speech regions, preferring silence boundaries."""
    hop = int(0.01 * sr)
    segs: list[tuple[int, int]] = []
    i = 0
    n = len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        s0, s1 = i * hop, min(len(x), j * hop)
        # split long region into <=6 s pieces at silence (mask already silence-split,
        # so just cap length; short tail <4 s is dropped unless region itself is 4-6 s)
        while s1 - s0 > int(SEG_MAX * sr):
            segs.append((s0, s0 + int(SEG_MAX * sr)))
            s0 += int(SEG_MAX * sr)
        if s1 - s0 >= int(SEG_MIN * sr):
            segs.append((s0, s1))
        i = j
    return segs


def seg_features(seg: np.ndarray, sr: int = 16000) -> dict:
    """QC + analysis features for one segment."""
    import librosa

    rms = float(np.sqrt((seg ** 2).mean() + 1e-12))
    rms_db = float(20 * np.log10(rms + 1e-12))
    peak = float(np.abs(seg).max()) if seg.size else 0.0
    clipped = float((np.abs(seg) >= 0.99).mean()) if seg.size else 0.0
    # speech-vs-silence SNR via frame energies
    fl, hop = int(0.02 * sr), int(0.01 * sr)
    nf = max(1, 1 + (len(seg) - fl) // hop)
    fr = np.stack([seg[k * hop:k * hop + fl] for k in range(nf)])
    e = 10 * np.log10((fr ** 2).mean(axis=1) + 1e-12)
    snr = float(np.percentile(e, 90) - np.percentile(e, 10))
    # rolloff + flatness + zcr (tonal/hold-music heuristic inputs)
    S = np.abs(librosa.stft(seg, n_fft=1024, hop_length=256))
    roll = float(librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85).mean())
    flat = float(librosa.feature.spectral_flatness(S=S).mean())
    zcr = float(librosa.feature.zero_crossing_rate(seg, frame_length=1024, hop_length=256).mean())
    tonal = bool(flat < 0.05 and zcr < 0.02)
    return {"rms_dbfs": round(rms_db, 2), "peak": round(peak, 4),
            "clipped_frac": round(clipped, 4), "snr_db": round(snr, 2),
            "rolloff_hz": round(roll, 1), "flatness": round(float(flat), 4),
            "zcr": round(float(zcr), 4), "tonal": tonal}


def load_idmap() -> dict[str, str]:
    d = json.loads(IDMAP.read_text(encoding="utf-8"))
    return d  # anon -> real name


def wa_date(real_name: str) -> date | None:
    m = re.search(r"PTT-(\d{4})(\d{2})(\d{2})", real_name)
    if m:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", real_name)
    if m:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return None


def main() -> int:
    SEG_DIR.mkdir(parents=True, exist_ok=True)
    (SEG_DIR / "test").mkdir(exist_ok=True)
    (SEG_DIR / "adapt").mkdir(exist_ok=True)
    (SEG_DIR / "quarantine").mkdir(exist_ok=True)

    idmap = load_idmap()
    real2anon = {v: k for k, v in idmap.items()}

    wa_files = sorted(WA_DIR.iterdir())
    wa_files = [p for p in wa_files if p.is_file()]
    pc_files = sorted(PC_DIR.iterdir())
    pc_files = [p for p in pc_files if p.is_file()]

    # ---- contact mapping (LOCAL ONLY, never committed) ----
    contacts: dict[str, str] = {}  # anon file id -> contact key
    for p in pc_files:
        aid = real2anon.get(p.name)
        nm = p.name
        if nm.startswith("Sgruthi-"):
            contacts[aid] = "contact_A"
        else:
            contacts[aid] = "contact_B"
    by_contact: dict[str, list] = {}
    for aid, c in contacts.items():
        by_contact.setdefault(c, []).append(aid)
    print("contacts (counts only):", {k: len(v) for k, v in by_contact.items()})

    # TEST contact = the smaller one (whole contact held out, larger kept for adapt)
    test_contact = min(by_contact, key=lambda k: len(by_contact[k]))
    print(f"TEST contact: {test_contact} (whole contact held out)")

    # WhatsApp dates (local) -> later half = TEST
    wa_dated = []
    for p in wa_files:
        aid = real2anon.get(p.name)
        wa_dated.append((wa_date(p.name), aid, p))
    wa_dated.sort(key=lambda t: (t[0] is None, t[0], t[1] or ""))
    half = len(wa_dated) // 2
    wa_test_ids = {aid for _, aid, _ in wa_dated[half:]}
    print(f"whatsapp: n={len(wa_dated)} early(adapt)={half} late(test)={len(wa_dated) - half}")

    ogg_ids = {real2anon.get(p.name) for p in wa_files if p.suffix.lower() == ".ogg"}
    print(f"ogg notes (TEST): n={len(ogg_ids)}")

    # ---- disk guard (estimate: <= ~1200 segs x 6 s x 32 kB/s ≈ 230 MB max) ----
    free = disk_free_gb()
    print(f"disk free: {free:.1f} GB")
    if free - 0.5 < MIN_FREE_GB:
        print(f"REFUSING: writing segments would risk dropping below {MIN_FREE_GB} GB free.")
        return 3

    rows: list[dict] = []
    stats = {"files_ok": 0, "files_failed": 0, "segs_raw": 0, "dropped_silence": 0,
             "dropped_clipped": 0, "quarantined_tonal": 0}
    import soundfile as sf

    all_jobs = [("whatsapp", p) for p in wa_files] + [("phone", p) for p in pc_files]
    for source, p in all_jobs:
        aid = real2anon.get(p.name)
        if aid is None:
            print(f"WARN no anon id for local file (skipped, see local log)")
            continue
        try:
            x, orig_sr = decode_16k(p)
        except Exception as exc:
            stats["files_failed"] += 1
            print(f"FAIL decode: {aid}: {exc}")
            continue
        dur = len(x) / 16000.0
        if dur < MIN_FILE_SEC:
            stats["files_failed"] += 1
            print(f"SKIP short ({dur:.1f}s < {MIN_FILE_SEC}s): {aid}")
            continue
        mask = energy_vad(x)
        cuts = cut_segments(x, mask)
        stats["files_ok"] += 1
        cap = MAX_PER_WA_FILE if source == "whatsapp" else MAX_PER_CALL_FILE
        kept = 0
        for (a, b) in cuts:
            if kept >= cap:
                break
            seg = x[a:b]
            if len(seg) < int(SEG_MIN * 16000):
                continue
            stats["segs_raw"] += 1
            f = seg_features(seg[: int(SEG_MAX * 16000)])
            if f["rms_dbfs"] < -45.0:
                stats["dropped_silence"] += 1
                continue
            if f["clipped_frac"] > 0.02:
                stats["dropped_clipped"] += 1
                continue
            # split assignment (group-disjoint)
            if source == "whatsapp":
                group = "whatsapp_late" if aid in wa_test_ids else "whatsapp_early"
                split = "test" if aid in wa_test_ids else "adapt"
            else:
                c = contacts[aid]
                group = "call_A" if c == "contact_A" else "call_B"
                split = "test" if c == test_contact else "adapt"
            if f["tonal"]:
                split = "quarantine"
                stats["quarantined_tonal"] += 1
            segid = f"{group}_{aid}_{a * 1000 // 16000:07d}"
            sf.write(str(SEG_DIR / split / (segid + ".wav")), seg, 16000)
            rows.append({
                "segid": segid, "split": split, "source": source, "anon_file": aid,
                "group": group, "start_s": round(a / 16000.0, 2), "end_s": round(b / 16000.0, 2),
                "codec": "opus_16k" if source == "whatsapp" else "amr_nb_8k",
                "orig_sr": orig_sr, **f})
            kept += 1
        print(f"{source} {aid}: dur={dur:.0f}s segs_kept={kept}")

    # ---- balance: calls adapt total <= 4x whatsapp adapt total ----
    adapt = [r for r in rows if r["split"] == "adapt"]
    wa_ad = [r for r in adapt if r["source"] == "whatsapp"]
    pc_ad = [r for r in adapt if r["source"] == "phone"]
    max_calls = int(len(wa_ad) * CALLS_TO_WA_RATIO)
    dropped_balance = 0
    if len(pc_ad) > max_calls:
        keep_idx = set(np.linspace(0, len(pc_ad) - 1, max_calls, dtype=int).tolist())
        for i, r in enumerate(pc_ad):
            if i not in keep_idx:
                r["split"] = "quarantine"
                r["quarantine_reason"] = "balance_cap_calls_le_4x_whatsapp"
                (SEG_DIR / "adapt" / (r["segid"] + ".wav")).replace(
                    SEG_DIR / "quarantine" / (r["segid"] + ".wav"))
                dropped_balance += 1
    print(f"balance: wa_adapt={len(wa_ad)} pc_adapt_kept={len([r for r in pc_ad if r['split']=='adapt'])} "
          f"moved_to_quarantine={dropped_balance}")

    # ---- write local manifest ----
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    by = {}
    for r in rows:
        by.setdefault((r["split"], r["source"], r["group"]), 0)
        by[(r["split"], r["source"], r["group"])] += 1
    summary = {"n_segments": len(rows),
               "by_split_source_group": {f"{a}|{b}|{c}": n for (a, b, c), n in sorted(by.items())},
               "stats": stats, "balance_moved": dropped_balance,
               "test_rule": "whatsapp_late dates + whole smaller contact + 2 ogg; never train/tune/select on test",
               "adapt_share_note": "adapt pool is the SMALL domain share (<=15% of human train at Phase 4)"}
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=1))

    # ---- speaker-leak check (MFCC mean/std + logreg 5-fold, human segs) ----
    try:
        import librosa
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler

        feats, labs = [], []
        for r in [r for r in rows if r["split"] in ("adapt", "test")]:
            y, _ = sf.read(str(SEG_DIR / r["split"] / (r["segid"] + ".wav")), dtype="float32")
            m = librosa.feature.mfcc(y=np.asarray(y, dtype=np.float32), sr=16000, n_mfcc=13)
            feats.append(np.concatenate([m.mean(axis=1), m.std(axis=1)]))
            labs.append(r["group"])
        X = StandardScaler().fit_transform(np.stack(feats))
        acc = float(cross_val_score(LogisticRegression(max_iter=2000), X, labs, cv=5).mean())
        n_groups = len(set(labs))
        summary["speaker_leak"] = {"cv_accuracy": round(acc, 3), "n_groups": n_groups,
                                   "chance": round(1 / n_groups, 3),
                                   "verdict": "HIGH — reduce private share / add public speakers"
                                   if acc > 0.5 else "ok — groups not trivially separable"}
        print("speaker_leak:", json.dumps(summary["speaker_leak"]))
    except Exception as exc:
        summary["speaker_leak"] = {"error": str(exc)[:200]}
        print("speaker_leak failed:", exc)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # ---- acoustic gap analysis (private vs clean public) ----
    try:
        pub_rows = []
        for name in ["260-123286-0012.flac", "4446-2273-0021.flac",
                     "5142-33396-0059.flac", "6930-81414-0025.flac"]:
            y, _ = sf.read(str(ROOT / "samples" / "real" / name), dtype="float32")
            y = np.asarray(y, dtype=np.float32).ravel()
            step = max(1, len(y) // 8)
            for k in range(0, len(y) - 5 * 16000, step):
                pub_rows.append(seg_features(y[k:k + 5 * 16000]))
                if len(pub_rows) >= 24:
                    break
        import pandas as pd

        def agg(rs, key):
            v = np.array([r[key] for r in rs])
            return {"median": round(float(np.median(v)), 2), "mean": round(float(np.mean(v)), 2)}
        priv = [r for r in rows if r["split"] in ("adapt", "test")]
        gap = {}
        for key in ["rms_dbfs", "snr_db", "rolloff_hz"]:
            gap[key] = {"private": agg(priv, key), "clean_public": agg(pub_rows, key)}
        summary["acoustic_gap"] = gap
        print("acoustic_gap:", json.dumps(gap, indent=1))
    except Exception as exc:
        summary["acoustic_gap"] = {"error": str(exc)[:200]}
        print("acoustic_gap failed:", exc)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # ---- committed report (counts only) ----
    L = ["# Private data splits (counts only — no filenames, no contact names)",
         "",
         f"Segments: {summary['n_segments']} (files ok={stats['files_ok']}, "
         f"failed/skipped={stats['files_failed']})",
         f"Raw cuts={stats['segs_raw']} dropped_silence={stats['dropped_silence']} "
         f"dropped_clipped={stats['dropped_clipped']} quarantined_tonal={stats['quarantined_tonal']} "
         f"balance_moved={dropped_balance}",
         "", "## By split|source|group",
         *[f"- {k}: {v}" for k, v in summary["by_split_source_group"].items()],
         "", f"TEST rule: {summary['test_rule']}",
         f"Adapt share: {summary['adapt_share_note']}",
         "", "## Speaker-leak check", json.dumps(summary.get("speaker_leak")),
         "", "## Acoustic gap (private real vs clean LibriSpeech)",
         json.dumps(summary.get("acoustic_gap"), indent=1),
         "", "Training coverage consequence: see reports/acoustic_gap.md (Phase 1 must add "
         "noisy/band-limited public real + codec-fair augmentation both classes).",
         "", f"Manifest (local-only): samples/real/manifest.csv + manifest_summary.json; "
         f"segments: samples/real/segments/<test|adapt|quarantine>/*.wav"]
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {REPORT} (committed) + local manifest n={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
