"""Phase 0 baseline: run CURRENT RAFNet model on all samples/labels.csv.

Writes reports/baseline.json + reports/baseline.md with:
- per-file scores (clean, same path as POST /api/analyze)
- per-group accuracy (3-way + binary@0.5)
- codec test: each real file transcoded to Opus 24k (WhatsApp-like),
  MP3 128k, AMR-NB 12.2k/8kHz, G.711 A-law 8kHz via ffmpeg; score shift shown.
  If no ffmpeg binary is found, falls back to a clearly-labelled Python
  band-limit simulation for Opus/G711 only (never claimed as a real codec).
- AUC/EER (threshold-independent) + calibration table (stated confidence vs
  actual binary accuracy per confidence bin).

Honesty: mock inference is refused; every number comes from this run and is
saved to reports/baseline.json. No thresholds are changed here.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MOCK_INFERENCE", "false")
if os.getenv("MOCK_INFERENCE", "false").lower() == "true":
    print("REFUSING: MOCK_INFERENCE=true — baseline must use the real model.")
    raise SystemExit(2)

from tools.evaluate_model import (  # noqa: E402
    binary_metrics_at_05,
    compute_auc,
    compute_eer,
    confusion_3way,
    coverage_and_selective,
    label_for_p,
    load_active_thresholds,
    strict_accuracy,
)


def find_ffmpeg() -> str | None:
    for cand in ("ffmpeg", "ffmpeg.exe"):
        p = shutil.which(cand)
        if p:
            return p
    # imageio-ffmpeg bundled binary (pip package, no system install needed)
    try:
        import imageio_ffmpeg  # type: ignore

        p = imageio_ffmpeg.get_ffmpeg_exe()
        if Path(p).is_file():
            return p
    except Exception:
        pass
    # WinGet Gyan install location
    for base in (
        Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages",
        Path("C:/Program Files/ffmpeg/bin"),
        Path("C:/ffmpeg/bin"),
    ):
        for exe in (base / "ffmpeg.exe", base / "ffmpeg"):
            if exe.is_file():
                return str(exe)
        try:
            if base.is_dir():
                for sub in base.iterdir():
                    for exe in (sub / "bin" / "ffmpeg.exe", sub / "ffmpeg.exe"):
                        if exe.is_file():
                            return str(exe)
        except Exception:
            continue
    return None


def score_one(model, data_bytes: bytes, filename: str) -> dict:
    from rafnet.soc.audio_io import decode_audio

    decoded = decode_audio(data_bytes, filename)
    res = model.predict(decoded.samples, decoded.sample_rate)
    return {
        "p_synthetic": round(float(res.prob_synthetic), 4),
        "prediction": res.prediction,
        "confidence": round(float(res.confidence), 4),
        "classical_p": res.classical_probability,
        "quantum_p": res.quantum_probability,
        "fused_p": res.fused_probability,
        "duration_seconds": round(float(decoded.duration_seconds), 2),
        "sample_rate_hz": int(decoded.sample_rate),
        "input_lossy": bool(decoded.is_lossy),
    }


def transcode(ffmpeg: str, src_wav: Path, codec: str) -> Path:
    """Transcode src_wav (16k mono) to codec and back to 16k wav. Returns wav path.

    NOTE: caller must read the returned file promptly; the parent temp dir is
    cleaned by transcode_bytes(). This helper keeps the path form for debugging.
    """
    tmp = Path(tempfile.mkdtemp(prefix="rafnet_codec_"))
    if codec == "opus_24k":
        enc = tmp / "out.opus"
        cmd = [ffmpeg, "-y", "-v", "error", "-i", str(src_wav),
               "-c:a", "libopus", "-b:a", "24k", "-ar", "16000", "-ac", "1", str(enc)]
    elif codec == "mp3_128k":
        enc = tmp / "out.mp3"
        cmd = [ffmpeg, "-y", "-v", "error", "-i", str(src_wav),
               "-c:a", "libmp3lame", "-b:a", "128k", "-ar", "16000", "-ac", "1", str(enc)]
    elif codec == "amr_nb":
        enc = tmp / "out.amr"
        # imageio-ffmpeg build exposes OpenCORE AMR-NB as libopencore_amrnb
        cmd = [ffmpeg, "-y", "-v", "error", "-i", str(src_wav),
               "-ar", "8000", "-ac", "1", "-c:a", "libopencore_amrnb", "-b:a", "12.2k", str(enc)]
    elif codec == "g711_alaw":
        enc = tmp / "out_alaw.wav"
        cmd = [ffmpeg, "-y", "-v", "error", "-i", str(src_wav),
               "-ar", "8000", "-ac", "1", "-c:a", "pcm_alaw", str(enc)]
    else:
        raise ValueError(codec)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0 or not enc.is_file():
        raise RuntimeError(f"ffmpeg encode failed ({codec}): {(r.stderr or '')[:400]}")
    back = tmp / "back_16k.wav"
    r2 = subprocess.run([ffmpeg, "-y", "-v", "error", "-i", str(enc),
                         "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(back)],
                        capture_output=True, text=True, timeout=120)
    if r2.returncode != 0 or not back.is_file():
        raise RuntimeError(f"ffmpeg decode failed ({codec}): {(r2.stderr or '')[:400]}")
    return back


def transcode_bytes(ffmpeg: str, src_wav: Path, codec: str) -> tuple[bytes, str]:
    """Transcode and return (wav_bytes, tmpdir_note). Cleans the temp dir."""
    import shutil as _shutil

    back = transcode(ffmpeg, src_wav, codec)
    tmp = getattr(back, "_tmpdir", back.parent)
    try:
        data = back.read_bytes()
        return data, str(tmp)
    finally:
        try:
            _shutil.rmtree(tmp, ignore_errors=True)
        except Exception:
            pass


def python_bandlimit_fallback(samples_16k):
    """Clearly-labelled fallback when no ffmpeg exists (NOT a real codec)."""
    import numpy as np

    x = np.asarray(samples_16k, dtype=np.float32).ravel()
    spec = np.fft.rfft(x.astype(np.float64))
    freqs = np.fft.rfftfreq(x.size, d=1.0 / 16000.0)
    spec[freqs > 4000.0] = 0.0  # rough Opus-low-bitrate-like band limit
    return np.fft.irfft(spec, n=x.size).astype(np.float32)


def calibration_table(y_true, scores, preds_3way=None) -> list[dict]:
    """Bin by stated confidence = max(p, 1-p); accuracy = binary@0.5 vs truth."""
    bins = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
    rows = []
    for lo, hi in bins:
        idx = [i for i, s in enumerate(scores) if lo <= max(float(s), 1 - float(s)) < hi]
        if not idx:
            rows.append({"confidence_bin": f"{lo:.1f}-{min(hi,1.0):.1f}",
                         "n": 0, "accuracy": None, "avg_confidence": None, "gap": None})
            continue
        ok = sum(1 for i in idx if (1 if float(scores[i]) >= 0.5 else 0) == y_true[i])
        acc = ok / len(idx)
        avgc = sum(max(float(scores[i]), 1 - float(scores[i])) for i in idx) / len(idx)
        rows.append({"confidence_bin": f"{lo:.1f}-{min(hi,1.0):.1f}", "n": len(idx),
                     "accuracy": round(acc, 4), "avg_confidence": round(avgc, 4),
                     "gap": round(avgc - acc, 4)})
    return rows


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="samples/labels.csv")
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--thresholds", default=None)
    ap.add_argument("--codecs", default="opus_24k,mp3_128k,amr_nb,g711_alaw")
    args = ap.parse_args()

    labels_path = (ROOT / args.labels) if not Path(args.labels).is_absolute() else Path(args.labels)
    reports_dir = (ROOT / args.reports) if not Path(args.reports).is_absolute() else Path(args.reports)
    reports_dir.mkdir(parents=True, exist_ok=True)

    tau_low, tau_high, tmeta = load_active_thresholds(args.thresholds)

    from rafnet.soc.model_adapter import load_voice_model

    model = load_voice_model()
    if not model.is_loaded() or getattr(model, "name", "") == "mock":
        print(f"REFUSING baseline: model not loaded or mock (mock={getattr(model,'name','?')}).")
        return 2
    print(f"model loaded: name={model.name} version={getattr(model,'model_version','?') if hasattr(model,'model_version') else 'n/a'}")
    try:
        fp = model.fingerprint()
        print(f"fingerprint: {json.dumps(fp)[:300]}")
    except Exception as exc:
        fp = {"error": str(exc)[:200]}
        print(f"fingerprint unavailable: {exc}")

    rows: list[dict] = []
    with open(labels_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    print(f"labels: {len(rows)} rows from {labels_path}")

    ffmpeg = find_ffmpeg()
    if ffmpeg:
        print(f"ffmpeg binary: {ffmpeg}")
        try:
            v = subprocess.run([ffmpeg, "-version"], capture_output=True, text=True, timeout=30)
            print((v.stdout or "").splitlines()[0][:160] if v.stdout else "ffmpeg version unknown")
        except Exception as exc:
            print(f"ffmpeg -version failed: {exc}")
    else:
        print("WARNING: no ffmpeg binary found (tried PATH + imageio-ffmpeg). "
              "Codec columns fall back to a Python band-limit simulation and are "
              "labelled simulated=true. Install with: winget install --id Gyan.FFmpeg -e")

    files: list[dict] = []
    for r in rows:
        rel = r["file"].replace("\\", "/")
        p = ROOT / rel
        entry = dict(r)
        if not p.is_file():
            entry["error"] = "missing file"
            files.append(entry)
            print(f"MISSING {rel}")
            continue
        try:
            base = score_one(model, p.read_bytes(), p.name)
            entry.update(base)
        except Exception as exc:
            entry["error"] = f"inference failed: {type(exc).__name__}: {exc}"[:300]
            files.append(entry)
            print(f"ERROR {rel}: {entry['error']}")
            continue
        print(f"scored {r['label']:>4} {rel}: p={entry['p_synthetic']} -> {entry['prediction']} (conf {entry['confidence']})")
        files.append(entry)

    ok = [f for f in files if "p_synthetic" in f]
    y = [int(f["y"]) for f in ok]
    s = [float(f["p_synthetic"]) for f in ok]

    # Per-group accuracy
    groups: dict[str, dict] = {}
    for f in ok:
        g = f.get("group", "unknown")
        groups.setdefault(g, {"y": [], "s": [], "files": []})
        groups[g]["y"].append(int(f["y"]))
        groups[g]["s"].append(float(f["p_synthetic"]))
        groups[g]["files"].append(f["file"])
    group_metrics = {}
    for g, d in groups.items():
        group_metrics[g] = {
            "n": len(d["y"]),
            "n_real": sum(1 for v in d["y"] if v == 0),
            "n_fake": sum(1 for v in d["y"] if v == 1),
            "strict_accuracy": strict_accuracy(d["y"], d["s"], tau_low, tau_high),
            **coverage_and_selective(d["y"], d["s"], tau_low, tau_high),
            "binary_at_0.5": binary_metrics_at_05(d["y"], d["s"]),
            "auc": compute_auc(d["y"], d["s"]),
            "eer": compute_eer(d["y"], d["s"]),
        }

    overall = {
        "n": len(y),
        "n_real": sum(1 for v in y if v == 0),
        "n_fake": sum(1 for v in y if v == 1),
        "confusion_3way": confusion_3way(y, s, tau_low, tau_high),
        "strict_accuracy": strict_accuracy(y, s, tau_low, tau_high),
        **coverage_and_selective(y, s, tau_low, tau_high),
        "binary_at_0.5": binary_metrics_at_05(y, s),
        "auc": compute_auc(y, s),
        "eer": compute_eer(y, s),
    }
    cal_table = calibration_table(y, s)

    # Codec stress: every real file -> each codec
    codecs = [c.strip() for c in args.codecs.split(",") if c.strip()]
    codec_rows: list[dict] = []
    codec_summary: dict[str, dict] = {}
    real_files = [f for f in ok if int(f["y"]) == 0]
    import numpy as np
    import soundfile as sf

    for f in real_files:
        rel = f["file"]
        p = ROOT / rel.replace("\\", "/")
        raw = p.read_bytes()
        from rafnet.soc.audio_io import decode_audio

        dec = decode_audio(raw, p.name)
        arr = np.asarray(dec.samples, dtype=np.float32).ravel()
        # resample to 16k mono for ffmpeg input (simple linear interp if needed)
        if int(dec.sample_rate) != 16000 and arr.size:
            dur = arr.size / float(dec.sample_rate)
            tgt = max(1, int(round(dur * 16000)))
            old_idx = np.linspace(0.0, 1.0, num=arr.size)
            new_idx = np.linspace(0.0, 1.0, num=tgt)
            arr16 = np.interp(new_idx, old_idx, arr).astype(np.float32)
        else:
            arr16 = arr
        with tempfile.TemporaryDirectory(prefix="rafnet_src_") as td:
            src_wav = Path(td) / "src_16k.wav"
            sf.write(str(src_wav), arr16, 16000)
            for codec in codecs:
                row = {"file": rel, "codec": codec, "p_clean": float(f["p_synthetic"])}
                try:
                    if ffmpeg:
                        bdata, _ = transcode_bytes(ffmpeg, src_wav, codec)
                        scored = score_one(model, bdata, f"transcoded_{codec}.wav")
                        row.update({"p_codec": scored["p_synthetic"],
                                    "prediction_codec": scored["prediction"],
                                    "confidence_codec": scored["confidence"],
                                    "simulated": False,
                                    "method": codec})
                    else:
                        sim = python_bandlimit_fallback(arr16)
                        res = model.predict(sim.tolist(), 16000)
                        row.update({"p_codec": round(float(res.prob_synthetic), 4),
                                    "prediction_codec": res.prediction,
                                    "confidence_codec": round(float(res.confidence), 4),
                                    "simulated": True,
                                    "method": "python 4kHz band-limit simulation (NOT a real codec)"})
                    row["delta"] = round(float(row["p_codec"]) - float(row["p_clean"]), 4)
                except Exception as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"[:300]
                codec_rows.append(row)
                print(f"codec {codec:10s} {rel}: {row.get('p_clean')} -> {row.get('p_codec')} "
                      f"(delta {row.get('delta')}) {row.get('prediction_codec','')}"
                      f"{' [SIMULATED]' if row.get('simulated') else ''} {row.get('error','')}")

    for codec in codecs:
        deltas = [r["delta"] for r in codec_rows if r["codec"] == codec and "delta" in r]
        pcodec = [r["p_codec"] for r in codec_rows if r["codec"] == codec and "p_codec" in r]
        if deltas:
            codec_summary[codec] = {
                "n": len(deltas),
                "mean_delta": round(float(np.mean(deltas)), 4),
                "max_abs_delta": round(float(np.max(np.abs(deltas))), 4),
                "mean_p_codec": round(float(np.mean(pcodec)), 4),
                "simulated": None,
            }
    # fix simulated flag simply
    for codec in codecs:
        sim_flags = {r.get("simulated") for r in codec_rows if r["codec"] == codec}
        if codec in codec_summary:
            codec_summary[codec]["simulated"] = (sim_flags == {True})

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": {"loaded": True, "name": getattr(model, "name", "unknown"),
                  "mock": False, "fingerprint": fp},
        "thresholds": {"tau_low": tau_low, "tau_high": tau_high,
                       "source": tmeta.get("source", "?")},
        "labels_csv": str(labels_path),
        "files_total": len(rows),
        "files_scored": len(ok),
        "n_real": overall["n_real"],
        "n_fake": overall["n_fake"],
        "overall": overall,
        "per_group": group_metrics,
        "calibration_table": cal_table,
        "calibration_note": ("Bins use stated confidence=max(p,1-p); accuracy is "
                             "binary@0.5 vs truth. Gap>0 means overconfidence in that bin."),
        "ffmpeg": ffmpeg,
        "codecs_tested": codecs,
        "codec_rows": codec_rows,
        "codec_summary": codec_summary,
        "files": files,
        "honest_notes": [
            f"n={len(y)} (real={overall['n_real']}, fake={overall['n_fake']}): snapshot of THESE files only, NOT a general benchmark.",
            "No WhatsApp/Opus human voice notes, phone-line, Hindi/Tamil, noisy-room or voice-conversion clips are present in samples/ — codec bias for those channels is unmeasured until such clips are added.",
            "win_tts/ duplicates 4 of the samples/fake/*.wav files; per-group counts treat them as listed in labels.csv (duplicates noted).",
            ("ffmpeg missing: codec columns are Python band-limit simulations, NOT real Opus/AMR/G.711." if not ffmpeg
             else "ffmpeg present: codec columns are real transcodes (encode + decode back to 16k wav) through the stated codec settings."),
            "Thresholds unchanged in this phase; strict accuracy counts UNCERTAIN as wrong.",
        ],
    }
    jp = reports_dir / "baseline.json"
    mp = reports_dir / "baseline.md"
    jp.write_text(json.dumps(report, indent=2), encoding="utf-8")

    L: list[str] = []
    L.append("# Phase 0 baseline — CURRENT model on samples/labels.csv")
    L.append("")
    L.append(f"Generated: {report['generated_at']}")
    L.append(f"Model: loaded=true name={report['model']['name']} mock=false")
    L.append(f"Thresholds: tau_low={tau_low} tau_high={tau_high} ({report['thresholds']['source']})")
    L.append(f"Labels: {report['labels_csv']} — {report['files_scored']}/{report['files_total']} scored "
             f"(real={report['n_real']}, fake={report['n_fake']})")
    L.append(f"ffmpeg: {ffmpeg or 'MISSING (simulated codec columns)'}")
    L.append("")
    L.append("> Every number below is measured in this run and saved in reports/baseline.json. "
             "Small-n snapshot, NOT a general benchmark.")
    L.append("")
    L.append("## Sample inventory (what is actually in samples/)")
    for g, m in group_metrics.items():
        L.append(f"- {g}: n={m['n']} (real={m['n_real']}, fake={m['n_fake']}) "
                 f"strict_acc={m['strict_accuracy']} auc={m['auc']} eer={json.dumps(m['eer'])}")
    L.append("- MISSING vs task brief: real/whatsapp_opus/*, real/phone_line/*, "
             "real Hindi/Tamil/noisy-room sets, fake voice-conversion set — none present.")
    L.append("")
    L.append("## Overall (clean, full-clip POST /api/analyze path)")
    L.append(f"n={overall['n']} strict_accuracy={overall['strict_accuracy']} "
             f"coverage={overall['coverage']} selective={overall['selective_accuracy']} "
             f"AUC={overall['auc']} EER={json.dumps(overall['eer'])}")
    L.append(f"binary@0.5: {json.dumps(overall['binary_at_0.5'])}")
    L.append(f"confusion: {json.dumps(overall['confusion_3way'])}")
    L.append("")
    L.append("## Per-file scores (clean)")
    for fl in files:
        if "p_synthetic" in fl:
            L.append(f"- {fl['file']} [{fl['label']}/{fl.get('group')}] "
                     f"p={fl['p_synthetic']} -> {fl['prediction']} (conf {fl['confidence']}) "
                     f"dur={fl.get('duration_seconds')}s sr={fl.get('sample_rate_hz')} lossy={fl.get('input_lossy')}")
        else:
            L.append(f"- {fl.get('file')} ERROR: {fl.get('error')}")
    L.append("")
    L.append("## Codec stress (real files only: clean p -> codec p, delta)")
    for c in codecs:
        sm = codec_summary.get(c, {})
        L.append(f"### {c}: {json.dumps(sm)}")
        for r in [x for x in codec_rows if x["codec"] == c]:
            flag = " [SIMULATED]" if r.get("simulated") else ""
            if "p_codec" in r:
                L.append(f"- {r['file']}: {r['p_clean']} -> {r['p_codec']} "
                         f"(delta {r['delta']}) -> {r['prediction_codec']}{flag}")
            else:
                L.append(f"- {r['file']}: ERROR {r.get('error')}{flag}")
    L.append("")
    L.append("## Calibration table (stated confidence vs actual binary accuracy)")
    L.append("| confidence bin | n | accuracy | avg_confidence | gap (conf-acc) |")
    L.append("| --- | --- | --- | --- | --- |")
    for row in cal_table:
        L.append(f"| {row['confidence_bin']} | {row['n']} | {row['accuracy']} | "
                 f"{row['avg_confidence']} | {row['gap']} |")
    L.append("")
    L.append("_Gap > 0 = overconfident in that bin. Small n: treat as indicative only._")
    L.append("")
    L.append("## Honest notes")
    for n in report["honest_notes"]:
        L.append(f"- {n}")
    mp.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {jp} and {mp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
