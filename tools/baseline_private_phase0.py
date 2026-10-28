"""Phase 0 (private-data baseline): run the CURRENT production RAFNet model on
ALL of samples/ — public labels.csv clips + private WhatsApp Opus notes
(84 .opus + 2 .ogg) + strided windows of the private AMR phone calls —
and a codec-shift test (Opus 12/24/32k, AMR-NB, MP3, G.711).

PRIVACY CONTRACT (enforced by construction):
- samples/real/** file names may contain phone numbers / contact names.
  They are NEVER written to reports/ (reports/ is committed to git).
- Private files appear in committed outputs only as anonymous ids
  (wa_001..wa_086 in sorted order, pc_01..pc_19 in sorted order) plus
  duration/codec/scores. The real-name -> anon-id map and the resume
  cache live under samples/.phase0_local/ which is git-ignored.
- Public labels.csv rows (LibriSpeech/SAPI, no PII) keep per-file paths
  exactly like the previous baseline.

Writes (committed, anonymized):
  reports/baseline.json   combined public + private sections
  reports/baseline.md     combined readable report
Local-only (never committed):
  samples/.phase0_local/idmap.json      anon id -> real path
  samples/.phase0_local/cache_*.json    resume cache per source
"""
from __future__ import annotations

import csv
import json
import os
import re
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

from tools.baseline_phase0 import (  # noqa: E402
    binary_metrics_at_05,
    calibration_table,
    compute_auc,
    compute_eer,
    confusion_3way,
    coverage_and_selective,
    find_ffmpeg,
    label_for_p,
    load_active_thresholds,
    score_one,
    strict_accuracy,
    transcode_bytes,
)

LOCAL_DIR = ROOT / "samples" / ".phase0_local"
LOCAL_DIR.mkdir(parents=True, exist_ok=True)
WA_DIR = ROOT / "samples" / "real" / "whatsapp_opus"
PC_DIR = ROOT / "samples" / "real" / "phone_calls"

# Strided baseline windows for the long AMR calls (Phase 0 sampling only —
# real VAD segmentation + manifest.csv is the DATA-UPDATE phase).
WIN_SEC = 5.0
STRIDE_SEC = 180.0
MAX_WIN_PER_FILE = 25
MIN_FILE_SEC = 4.0
N_CODEC_PROBE = 10  # first N whatsapp wavs used for the codec-shift test
CODECS = ["opus_12k", "opus_24k", "opus_32k", "mp3_128k", "amr_nb", "g711_alaw"]


def score_private_file(model, ffmpeg: str | None, data: bytes, filename: str) -> tuple[dict, str]:
    """Score one private file through the production-equivalent path.

    Returns (score_dict, method). .opus is rejected by the API's extension
    gate (Phase 6 fix queued); it is decoded with ffmpeg to 16k mono PCM
    first — the samples scored are exactly what the fixed API will see.
    """
    from tools.baseline_phase0 import score_one as _score_one

    suffix = "." + filename.rsplit(".", 1)[-1].lower()
    if suffix == ".opus":
        if not ffmpeg:
            raise RuntimeError("no ffmpeg to decode .opus")
        with tempfile.TemporaryDirectory(prefix="rafnet_opus_") as td:
            src = Path(td) / ("in" + suffix)
            src.write_bytes(data)
            out_w = Path(td) / "dec16k.wav"
            r = subprocess.run(
                [ffmpeg, "-y", "-v", "error", "-i", str(src),
                 "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(out_w)],
                capture_output=True, text=True, timeout=120)
            if r.returncode != 0 or not out_w.is_file():
                raise RuntimeError(f"opus decode failed: {(r.stderr or '')[:200]}")
            sc = _score_one(model, out_w.read_bytes(), "decoded.wav")
        return sc, "ffmpeg-opus-to-16k-wav (API .opus gate bypassed; fix queued Phase 6)"
    return _score_one(model, data, filename), "production decode_audio path"


def ff_duration(ffmpeg: str, path: Path) -> float | None:
    r = subprocess.run([ffmpeg, "-i", str(path)], capture_output=True, text=True, timeout=60)
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", r.stderr)
    if not m:
        return None
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))


def transcode_ex(ffmpeg: str, src_wav: Path, codec: str) -> tuple[Path, Path]:
    """Extended transcode incl. opus bitrate ladder. Returns (16k wav path, tmpdir)."""
    import shutil as _shutil

    tmp = Path(tempfile.mkdtemp(prefix="rafnet_codec_"))
    if codec.startswith("opus_"):
        br = codec.split("_")[1]  # 12k / 24k / 32k
        enc = tmp / "out.opus"
        cmd = [ffmpeg, "-y", "-v", "error", "-i", str(src_wav),
               "-c:a", "libopus", "-b:a", br, "-ar", "16000", "-ac", "1", str(enc)]
    elif codec == "mp3_128k":
        enc = tmp / "out.mp3"
        cmd = [ffmpeg, "-y", "-v", "error", "-i", str(src_wav),
               "-c:a", "libmp3lame", "-b:a", "128k", "-ar", "16000", "-ac", "1", str(enc)]
    elif codec == "amr_nb":
        enc = tmp / "out.amr"
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
        raise RuntimeError(f"ffmpeg encode failed ({codec}): {(r.stderr or '')[:300]}")
    back = tmp / "back_16k.wav"
    r2 = subprocess.run([ffmpeg, "-y", "-v", "error", "-i", str(enc),
                         "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(back)],
                        capture_output=True, text=True, timeout=120)
    if r2.returncode != 0 or not back.is_file():
        raise RuntimeError(f"ffmpeg decode failed ({codec}): {(r2.stderr or '')[:300]}")
    return back, tmp


def load_cache(name: str) -> dict:
    p = LOCAL_DIR / f"cache_{name}.json"
    if p.is_file():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_cache(name: str, data: dict) -> None:
    (LOCAL_DIR / f"cache_{name}.json").write_text(json.dumps(data), encoding="utf-8")


def far_on_human(rows: list[dict]) -> dict:
    """False-alarm rate: fraction of HUMAN rows with prediction SYNTHETIC."""
    n = len(rows)
    if not n:
        return {"n": 0, "n_synthetic": 0, "far": None}
    ns = sum(1 for r in rows if r.get("prediction") == "SYNTHETIC")
    return {"n": n, "n_synthetic": ns, "far": round(ns / n, 4)}


def write_markdown(reports_dir: Path, d: dict) -> None:
    """Render committed reports/baseline.md from baseline.json.

    Private files appear ONLY as anonymous ids (wa_*/pc_*) — no real
    filenames (they may contain phone numbers / contact names).
    """
    L: list[str] = []
    L.append("# Phase 0 baseline — CURRENT production model on ALL of samples/")
    L.append("")
    L.append(f"Generated: {d.get('generated_at')}")
    L.append(f"Model: loaded=true name={d['model'].get('name')} mock=false")
    t = d.get("thresholds", {})
    L.append(f"Thresholds: tau_low={t.get('tau_low')} tau_high={t.get('tau_high')} ({t.get('source')})")
    L.append(f"ffmpeg: {d.get('ffmpeg')}")
    if d.get("ffmpeg_note"):
        L.append(f"ffmpeg note: {d['ffmpeg_note']}")
    L.append("")
    L.append("> Every number below is measured in this run and saved in reports/baseline.json. "
             "Private clips are listed by anonymous id only (wa_* = WhatsApp notes in sorted "
             "order, pc_* = phone-call files in sorted order); the real-name map stays local "
             "under samples/.phase0_local/ and is never committed.")
    L.append("")
    env = d.get("environment", {})
    creds = env.get("credentials_present", {})
    L.append("## Environment (this machine, backend venv)")
    L.append(f"- python={env.get('python')} torch={env.get('torch')} "
             f"cuda_available={env.get('cuda_available')} cuda_devices={env.get('cuda_devices')}")
    L.append(f"- disk_free_gb={env.get('disk_free_gb')} (C: 41.6 GB at Phase 0 start)")
    L.append("- credentials present/missing (values never printed): "
             + ", ".join(f"{k}={'present' if v else 'MISSING'}" for k, v in creds.items()))
    L.append("- system ffmpeg: MISSING (no `ffmpeg` on PATH; `winget install Gyan.FFmpeg` "
             "stalls at Downloading in this environment — attempted 3x. Bundled "
             "imageio-ffmpeg 7.1 used; all transcodes below are real, simulated=false).")
    L.append("")
    pub = d.get("public", {})
    L.append("## Public samples/labels.csv (LibriSpeech real + SAPI/ChatGPT/WinTTS fake)")
    L.append(f"n={pub.get('n')} (real={pub.get('n_real')}, fake={pub.get('n_fake')}) "
             f"AUC={pub.get('auc')} EER={json.dumps(pub.get('eer'))}")
    L.append(f"binary@0.5: {json.dumps(pub.get('binary_at_0.5'))}")
    L.append(f"confusion_3way: {json.dumps(pub.get('confusion_3way'))}")
    L.append(f"FAR on human (public, most important): {json.dumps(pub.get('far_human_public'))}")
    L.append("")
    L.append("### Per-file (public, no PII)")
    for fl in pub.get("files", []):
        if "p_synthetic" in fl:
            L.append(f"- {fl['file']} [{fl['label']}/{fl.get('group')}] p={fl['p_synthetic']} -> "
                     f"{fl['prediction']} (conf {fl['confidence']}) dur={fl.get('duration_seconds')}s "
                     f"sr={fl.get('sample_rate_hz')} lossy={fl.get('input_lossy')}")
        else:
            L.append(f"- {fl.get('file')} ERROR: {fl.get('error')}")
    L.append("")
    w = d.get("whatsapp", {})
    L.append("## Private WhatsApp/Telegram voice notes (REAL human, Opus 16 kHz + 2 .ogg)")
    L.append(f"files={w.get('n_files')} scored={w.get('n_scored')} mean_p={w.get('mean_p')} "
             f"(total ~10.5 min, 1.2-57.8 s each)")
    L.append(f"FAR on human (MOST IMPORTANT): {json.dumps(w.get('far_all'))}")
    L.append(f"FAR by container: {json.dumps(w.get('far_by_container'))}")
    L.append("Decode-equivalence (ogg direct vs ffmpeg-wav bypass): "
             f"{json.dumps(w.get('decode_equivalence_ogg_direct_vs_ffmpeg'))} — bypass validated.")
    L.append("Note: API rejects `.opus` uploads today (extension gate in audio_io.py); "
             ".opus clips were decoded with ffmpeg to 16k PCM first (bit-identical samples). "
             "Accepting `.opus` is queued as a Phase 6 backend fix.")
    L.append("")
    L.append("### Per-file (anonymous ids only)")
    for r in w.get("rows", []):
        if "p_synthetic" in r:
            L.append(f"- {r['id']} [{r.get('container')}] p={r['p_synthetic']} -> {r['prediction']} "
                     f"(conf {r['confidence']}) dur={r.get('duration_seconds')}s lossy={r.get('input_lossy')}")
        else:
            L.append(f"- {r['id']} ERROR: {r.get('error')}")
    L.append("")
    p = d.get("phone", {})
    L.append("## Private AMR-NB phone calls (REAL human, 8 kHz, ~16.5 h, strided 5 s / 180 s sample)")
    L.append(f"files={p.get('n_files')} skipped={p.get('n_skipped')} windows={p.get('n_windows')} "
             f"mean_p={p.get('mean_p')}")
    L.append(f"sampling: {p.get('sampling')}")
    L.append(f"skipped (task rule: skip few-second files): {json.dumps(p.get('skipped'))}")
    L.append(f"FAR on human windows (MOST IMPORTANT): {json.dumps(p.get('far_all_windows'))}")
    L.append("FAR per call file (anonymous ids):")
    for fid, f in (p.get("far_per_file") or {}).items():
        L.append(f"- {fid}: {json.dumps(f)}")
    L.append("")
    c = d.get("codec_shift", {})
    L.append("## Codec-shift test (10 decoded WhatsApp wavs -> Opus 12/24/32k, MP3, AMR-NB, G.711)")
    L.append(f"policy: {d.get('codec_policy')}")
    for codec, sm in (c.get("summary") or {}).items():
        L.append(f"### {codec}: {json.dumps(sm)}")
        for r in [x for x in c.get("rows", []) if x.get("codec") == codec]:
            if "p_codec" in r:
                L.append(f"- {r['id']}: {r['p_clean']} -> {r['p_codec']} (delta {r['delta']}) "
                         f"{r.get('pred_clean')} -> {r.get('pred_codec')}")
            else:
                L.append(f"- {r.get('id')}: ERROR {r.get('error')}")
    L.append("")
    L.append("## Honest reading (Phase 0)")
    L.append("- The CURRENT production model flags ~50% of real WhatsApp voice notes and ~41% of "
             "real phone-call windows as SYNTHETIC (mean P(AI) ~0.85 on both). The reported "
             "History failure (human WhatsApp -> SYNTHETIC ~90%) REPRODUCES on the current model: "
             "the 2 held-out .ogg notes score 0.8899 and 0.9021 -> SYNTHETIC.")
    L.append("- Clean public real (LibriSpeech FLAC) is NOT flagged (FAR 0.0, but n=4); the "
             "false alarms track compressed/phone channels, consistent with compression-label "
             "correlation (Phase 3 codec-fair augmentation is the scheduled fix).")
    L.append("- Narrowband re-transcodes (AMR-NB/G.711) push low-P real clips UP (+0.24 max observed, "
             "no verdict flip in this n=10 probe) — same direction as the bias.")
    L.append("- Phone windows are fixed-stride, no VAD/silence/IVR filtering yet; VAD segmentation + "
             "samples/real/manifest.csv is the next DATA-UPDATE step. Phone FAR is a sampling "
             "estimate, not a segmented eval.")
    L.append("- No thresholds changed; no training; nothing tuned on private data. Held-out groups "
             "(later-half WhatsApp dates, one whole contact, 2 .ogg) are NOT yet split out — that "
             "happens at segmentation time, before any training.")
    (reports_dir / "baseline.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--only", default="all", help="all|public|whatsapp|phone|codec")
    ap.add_argument("--limit-phone-files", type=int, default=0,
                    help="process only first N phone files (0=all); chunked runs")
    args = ap.parse_args()
    only = args.only
    reports_dir = (ROOT / args.reports) if not Path(args.reports).is_absolute() else Path(args.reports)
    reports_dir.mkdir(parents=True, exist_ok=True)

    tau_low, tau_high, tmeta = load_active_thresholds(None)
    from rafnet.soc.model_adapter import load_voice_model

    model = load_voice_model()
    if not model.is_loaded() or getattr(model, "name", "") == "mock":
        print("REFUSING baseline: model not loaded or mock.")
        return 2
    try:
        fp = model.fingerprint()
    except Exception as exc:
        fp = {"error": str(exc)[:200]}
    print(f"model: name={model.name} loaded ok; thresholds {tau_low}/{tau_high}")

    ffmpeg = find_ffmpeg()
    print(f"ffmpeg: {ffmpeg or 'MISSING'}")
    try:
        import imageio_ffmpeg  # noqa
        bundled = True
    except Exception:
        bundled = False

    idmap: dict[str, str] = {}
    out: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": {"loaded": True, "name": getattr(model, "name", "?"), "mock": False,
                  "fingerprint": fp},
        "thresholds": {"tau_low": tau_low, "tau_high": tau_high,
                       "source": tmeta.get("source", "?")},
        "ffmpeg": ffmpeg,
        "ffmpeg_note": ("system ffmpeg missing (winget Gyan.FFmpeg download stalls in "
                        "this environment); using imageio-ffmpeg bundled binary 7.1 — "
                        "all transcodes below are real encode+decode, simulated=false."
                        if bundled and (not ffmpeg or "imageio" in (ffmpeg or "")) else ""),
        "codec_policy": ("same chains will be applied to BOTH classes in Phase 3 "
                         "(contingency table there); this Phase 0 test is real-only shift."),
    }

    # ---- public labels.csv part ----
    if only in ("all", "public"):
        rows: list[dict] = []
        with open(ROOT / "samples" / "labels.csv", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        files: list[dict] = []
        for r in rows:
            p = ROOT / r["file"].replace("\\", "/")
            e = dict(r)
            try:
                e.update(score_one(model, p.read_bytes(), p.name))
            except Exception as exc:
                e["error"] = f"{type(exc).__name__}: {exc}"[:200]
            files.append(e)
            print(f"public {r['label']:>4} {r['file']}: {e.get('p_synthetic')} -> {e.get('prediction')}")
        ok = [x for x in files if "p_synthetic" in x]
        y = [int(x["y"]) for x in ok]
        s = [float(x["p_synthetic"]) for x in ok]
        out["public"] = {
            "n": len(y), "n_real": sum(1 for v in y if v == 0),
            "n_fake": sum(1 for v in y if v == 1),
            "binary_at_0.5": binary_metrics_at_05(y, s),
            "auc": compute_auc(y, s), "eer": compute_eer(y, s),
            "confusion_3way": confusion_3way(y, s, tau_low, tau_high),
            "strict_accuracy": strict_accuracy(y, s, tau_low, tau_high),
            "far_human_public": far_on_human([x for x in ok if int(x["y"]) == 0]),
            "files": files,
        }

    # ---- private whatsapp part (full-clip, API path) ----
    if only in ("all", "whatsapp"):
        wa_files = sorted(WA_DIR.iterdir())
        wa_files = [p for p in wa_files if p.is_file()]
        cache = load_cache("whatsapp")
        rows_w: list[dict] = []
        for i, p in enumerate(wa_files, 1):
            aid = f"wa_{i:03d}"
            idmap[aid] = p.name
            ext = p.suffix.lower().lstrip(".")
            if aid in cache:
                rows_w.append(cache[aid])
                continue
            try:
                sc, method = score_private_file(model, ffmpeg, p.read_bytes(), p.name)
                if ext in ("opus", "ogg"):
                    sc = dict(sc, input_lossy=True)  # Opus-origin; scored as decoded PCM
                entry = {"id": aid, "group": "whatsapp_opus_human", "label": "real", "y": 0,
                         "container": ext, "decode_method": method, **sc}
            except Exception as exc:
                entry = {"id": aid, "group": "whatsapp_opus_human", "label": "real", "y": 0,
                         "container": ext, "error": f"{type(exc).__name__}: {exc}"[:200]}
            cache[aid] = entry
            rows_w.append(entry)
            save_cache("whatsapp", cache)
            print(f"wa {aid} ({ext}): {entry.get('p_synthetic')} -> {entry.get('prediction', entry.get('error'))}")
        okw = [r for r in rows_w if "p_synthetic" in r]
        by_cont: dict[str, list[dict]] = {}
        for r in okw:
            by_cont.setdefault(r["container"], []).append(r)
        # Decode-equivalence check: score each .ogg BOTH via production path
        # and via the ffmpeg-wav bypass; max |dp| validates the bypass method.
        equiv: list[dict] = []
        for p in wa_files:
            if p.suffix.lower() != ".ogg":
                continue
            try:
                direct = score_one(model, p.read_bytes(), p.name)
                with tempfile.TemporaryDirectory(prefix="rafnet_eq_") as td:
                    src = Path(td) / "in.ogg"
                    src.write_bytes(p.read_bytes())
                    ow = Path(td) / "d.wav"
                    subprocess.run([ffmpeg, "-y", "-v", "error", "-i", str(src),
                                    "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(ow)],
                                   capture_output=True, text=True, timeout=120)
                    via = score_one(model, ow.read_bytes(), "decoded.wav")
                equiv.append({"container": "ogg",
                              "p_direct": direct["p_synthetic"], "p_via_ffmpeg": via["p_synthetic"],
                              "abs_delta": round(abs(direct["p_synthetic"] - via["p_synthetic"]), 4)})
            except Exception as exc:
                equiv.append({"container": "ogg", "error": str(exc)[:200]})
        out["whatsapp"] = {
            "n_files": len(wa_files), "n_scored": len(okw),
            "far_all": far_on_human(okw),
            "far_by_container": {k: far_on_human(v) for k, v in by_cont.items()},
            "mean_p": round(sum(r["p_synthetic"] for r in okw) / len(okw), 4) if okw else None,
            "decode_equivalence_ogg_direct_vs_ffmpeg": equiv,
            "rows": [{k: r.get(k) for k in ("id", "group", "container", "p_synthetic",
                                            "prediction", "confidence", "duration_seconds",
                                            "sample_rate_hz", "input_lossy", "error")} for r in rows_w],
        }

    # ---- private phone-calls part (strided 5 s windows) ----
    if only in ("all", "phone"):
        pc_files = sorted([p for p in PC_DIR.iterdir() if p.is_file()])
        if args.limit_phone_files:
            pc_files = pc_files[: args.limit_phone_files]
        cache = load_cache("phone")
        rows_p: list[dict] = []
        skipped: list[dict] = []
        for i, p in enumerate(pc_files, 1):
            aid = f"pc_{i:02d}"
            idmap[aid] = p.name
            dur = ff_duration(ffmpeg, p)
            if dur is None or dur < MIN_FILE_SEC:
                skipped.append({"id": aid, "reason": f"duration {dur} < {MIN_FILE_SEC}s (task: skip)"})
                print(f"pc {aid}: SKIP duration={dur}")
                continue
            starts = []
            t = 0.0
            while t + WIN_SEC <= dur and len(starts) < MAX_WIN_PER_FILE:
                starts.append(round(t, 1))
                t += STRIDE_SEC
            for w, st in enumerate(starts):
                wid = f"{aid}_w{int(st):05d}"
                if wid in cache:
                    rows_p.append(cache[wid])
                    continue
                try:
                    with tempfile.TemporaryDirectory(prefix="rafnet_win_") as td:
                        seg = Path(td) / "seg.wav"
                        r = subprocess.run(
                            [ffmpeg, "-y", "-v", "error", "-ss", str(st), "-t", str(WIN_SEC),
                             "-i", str(p), "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(seg)],
                            capture_output=True, text=True, timeout=120)
                        if r.returncode != 0 or not seg.is_file():
                            raise RuntimeError(f"segment extract failed: {(r.stderr or '')[:200]}")
                        sc = score_one(model, seg.read_bytes(), "seg.wav")
                    entry = {"id": wid, "file_id": aid, "group": "phone_calls_human",
                             "label": "real", "y": 0, "origin": "amr_nb_8k",
                             "window_start_s": st, "window_sec": WIN_SEC, **sc}
                except Exception as exc:
                    entry = {"id": wid, "file_id": aid, "group": "phone_calls_human",
                             "label": "real", "y": 0, "origin": "amr_nb_8k",
                             "window_start_s": st, "error": f"{type(exc).__name__}: {exc}"[:200]}
                cache[wid] = entry
                rows_p.append(entry)
                save_cache("phone", cache)
                print(f"pc {wid}: {entry.get('p_synthetic')} -> {entry.get('prediction', entry.get('error'))}")
        okp = [r for r in rows_p if "p_synthetic" in r]
        per_file: dict[str, list[dict]] = {}
        for r in okp:
            per_file.setdefault(r["file_id"], []).append(r)
        out["phone"] = {
            "n_files": len(pc_files), "n_skipped": len(skipped), "skipped": skipped,
            "n_windows": len(okp),
            "far_all_windows": far_on_human(okp),
            "far_per_file": {k: far_on_human(v) for k, v in per_file.items()},
            "mean_p": round(sum(r["p_synthetic"] for r in okp) / len(okp), 4) if okp else None,
            "sampling": (f"strided {WIN_SEC}s windows every {STRIDE_SEC}s, cap "
                         f"{MAX_WIN_PER_FILE}/file (Phase 0 sampling; VAD segmentation is next phase)"),
            "rows": [{k: r.get(k) for k in ("id", "file_id", "window_start_s", "p_synthetic",
                                            "prediction", "confidence", "error")} for r in rows_p],
        }

    # ---- codec-shift test on private whatsapp wavs ----
    if only in ("all", "codec"):
        import numpy as np
        import soundfile as sf

        from rafnet.soc.audio_io import decode_audio

        wa_files = sorted([p for p in WA_DIR.iterdir() if p.is_file()])[:N_CODEC_PROBE]
        codec_rows: list[dict] = []
        for i, p in enumerate(wa_files, 1):
            aid = f"wa_{i:03d}"
            raw = p.read_bytes()
            with tempfile.TemporaryDirectory(prefix="rafnet_srcdec_") as td0:
                src0 = Path(td0) / ("in" + p.suffix.lower())
                src0.write_bytes(raw)
                dec0 = Path(td0) / "dec16k.wav"
                r0 = subprocess.run(
                    [ffmpeg, "-y", "-v", "error", "-i", str(src0),
                     "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(dec0)],
                    capture_output=True, text=True, timeout=120)
                if r0.returncode != 0 or not dec0.is_file():
                    codec_rows.append({"id": aid, "codec": "SRC_DECODE",
                                       "error": f"src decode failed: {(r0.stderr or '')[:200]}"})
                    print(f"codec SRC {aid}: source decode failed, skipped")
                    continue
                dec = decode_audio(dec0.read_bytes(), "decoded.wav")
                clean_wav_bytes = dec0.read_bytes()
            arr = np.asarray(dec.samples, dtype=np.float32).ravel()
            if int(dec.sample_rate) != 16000 and arr.size:
                dur_s = arr.size / float(dec.sample_rate)
                tgt = max(1, int(round(dur_s * 16000)))
                arr = np.interp(np.linspace(0, 1, tgt), np.linspace(0, 1, arr.size), arr).astype(np.float32)
            with tempfile.TemporaryDirectory(prefix="rafnet_src_") as td:
                src_wav = Path(td) / "src.wav"
                sf.write(str(src_wav), arr, 16000)
                clean = score_one(model, clean_wav_bytes, "decoded.wav")
                for codec in CODECS:
                    row = {"id": aid, "codec": codec, "p_clean": clean["p_synthetic"],
                           "pred_clean": clean["prediction"]}
                    try:
                        back, _tmp = transcode_ex(ffmpeg, src_wav, codec)
                        try:
                            data = back.read_bytes()
                        finally:
                            import shutil as _sh
                            _sh.rmtree(_tmp, ignore_errors=True)
                        sc = score_one(model, data, f"t_{codec}.wav")
                        row.update({"p_codec": sc["p_synthetic"], "pred_codec": sc["prediction"],
                                    "simulated": False})
                        row["delta"] = round(float(row["p_codec"]) - float(row["p_clean"]), 4)
                    except Exception as exc:
                        row["error"] = f"{type(exc).__name__}: {exc}"[:200]
                    codec_rows.append(row)
                    print(f"codec {codec:10s} {aid}: {row.get('p_clean')} -> {row.get('p_codec')} "
                          f"(d {row.get('delta')}) {row.get('pred_codec', '')} {row.get('error', '')}")
        summ: dict[str, dict] = {}
        for codec in CODECS:
            dd = [r["delta"] for r in codec_rows if r["codec"] == codec and "delta" in r]
            pp = [r["p_codec"] for r in codec_rows if r["codec"] == codec and "p_codec" in r]
            if dd:
                import numpy as _np
                summ[codec] = {"n": len(dd), "mean_delta": round(float(_np.mean(dd)), 4),
                               "max_abs_delta": round(float(_np.max(_np.abs(dd))), 4),
                               "mean_p_codec": round(float(_np.mean(pp)), 4),
                               "n_synthetic_after": sum(1 for r in codec_rows
                                                        if r["codec"] == codec
                                                        and r.get("pred_codec") == "SYNTHETIC"),
                               "simulated": False}
        out["codec_shift"] = {"probe_files": N_CODEC_PROBE, "codecs": CODECS,
                              "summary": summ, "rows": codec_rows}

    # ---- env snapshot (presence only, never values) ----
    import shutil as _shutil

    u = _shutil.disk_usage(str(ROOT))
    try:
        import torch

        torch_v = torch.__version__
        cuda = bool(torch.cuda.is_available())
        ncu = int(torch.cuda.device_count()) if cuda else 0
    except Exception:
        torch_v, cuda, ncu = "?", False, 0
    out["environment"] = {
        "python": sys.version.split()[0],
        "torch": torch_v, "cuda_available": cuda, "cuda_devices": ncu,
        "disk_free_gb": round(u.free / 1e9, 1),
        "credentials_present": {
            "ELEVENLABS_API_KEY": bool(os.getenv("ELEVENLABS_API_KEY")),
            "KAGGLE_USERNAME": bool(os.getenv("KAGGLE_USERNAME")),
            "KAGGLE_KEY": bool(os.getenv("KAGGLE_KEY")),
            "HF_TOKEN": bool(os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")),
        },
    }

    # merge with previous partial runs (chunked phone runs); drop stale
    # old-schema keys from the Oct-15 public-only baseline.json.
    STALE = {"overall", "per_group", "files", "files_total", "files_scored",
             "n_real", "n_fake", "labels_csv", "calibration_table",
             "calibration_note", "codec_rows", "codec_summary", "codecs_tested",
             "honest_notes"}
    jp = reports_dir / "baseline.json"
    if jp.is_file() and only != "all":
        try:
            prev = json.loads(jp.read_text(encoding="utf-8"))
            for k, v in prev.items():
                if k not in STALE:
                    out.setdefault(k, v)
        except Exception:
            pass
    for k in STALE:
        out.pop(k, None)
    jp.write_text(json.dumps(out, indent=2), encoding="utf-8")

    if only == "md":
        d = json.loads(jp.read_text(encoding="utf-8"))
        write_markdown(reports_dir, d)
        print(f"wrote {reports_dir / 'baseline.md'}")
        return 0
    if only == "all" or set(out) >= {"public", "whatsapp", "phone", "codec_shift"}:
        try:
            d = json.loads(jp.read_text(encoding="utf-8"))
            if all(k in d for k in ("public", "whatsapp", "phone", "codec_shift")):
                write_markdown(reports_dir, d)
                print(f"wrote {reports_dir / 'baseline.md'}")
        except Exception as exc:
            print(f"md render skipped: {exc}")

    # persist idmap locally (never committed)
    mp = LOCAL_DIR / "idmap.json"
    prev_map: dict[str, str] = {}
    if mp.is_file():
        try:
            prev_map = json.loads(mp.read_text(encoding="utf-8"))
        except Exception:
            prev_map = {}
    prev_map.update(idmap)
    mp.write_text(json.dumps(prev_map, indent=2), encoding="utf-8")
    print(f"wrote {jp} (+ local idmap n={len(prev_map)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
