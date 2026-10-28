"""Phase 6 verification: additive v2 stack, determinism, guards, .opus accept.

Runs the SAME production paths the API uses:
  RAFNetVoiceModel.predict + antispoof.score_all + soc.head4.predict_pai
  -> ensemble v1 (3-feature) and v2 (4-feature, head-aware)

Checks (all LOCAL; private clips never leave this PC; committed report uses
anonymous ids only):
  1. .opus accepted by audio_io.validate_filename + decode_audio.
  2. Degenerate-segment guard: <1600-sample input raises ValueError (no crash).
  3. Determinism: 10 in-process repeats of the full score for one clip are
     bit-stable; a fresh subprocess reproduces the same digest (restart test).
  4. Before/after: v1 vs v2 calibrated P(AI) + verdict on 2 private .ogg notes
     (reproducing the History failure) and on held-out TEST clips
     (unseen generators) + a codec probe (Opus/AMR) on one real clip.

Writes reports/phase6_verify.json + reports/phase6_verify.md.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("MOCK_INFERENCE", "false")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

PRIVATE_WA = ROOT / "samples" / "real" / "whatsapp_opus"


def _score_clip(model, head, samples, sample_rate):
    from rafnet.soc import antispoof
    from rafnet.soc.ensemble import alert_for, alert_for4

    res = model.predict(samples, sample_rate)
    hs = antispoof.score_all(samples, sample_rate)
    p_head = head.predict_pai(samples, sample_rate) if head is not None else None
    ra = round(float(res.prob_synthetic), 4)
    v1 = alert_for(hs.get("p_mo"), hs.get("p_bisher"), ra, window=[], is_live=False)
    v2 = alert_for4({"head": p_head, "rafnet": ra, "mo": hs.get("p_mo"),
                     "bisher": hs.get("p_bisher")}, window=[], is_live=False)
    return {"rafnet": ra, "mo": hs.get("p_mo"), "bisher": hs.get("p_bisher"),
            "head": p_head, "p_v1": v1.p_cal, "verdict_v1": v1.verdict,
            "p_v2": v2.p_cal, "verdict_v2": v2.verdict}


def _digest(scores: dict) -> str:
    return hashlib.sha256(json.dumps(scores, sort_keys=True).encode()).hexdigest()[:16]


def _decode_path(path: Path):
    from rafnet.soc.audio_io import decode_audio

    return decode_audio(path.read_bytes(), path.name)


def main() -> int:
    from rafnet.soc.audio_io import decode_audio  # noqa: F401  (exercised below)
    from rafnet.soc.head4 import load_head
    from rafnet.soc.model_adapter import load_voice_model
    from rafnet.soc import audio_io

    report: dict = {"generated_at": datetime.now(timezone.utc).isoformat()}

    # 1. .opus accept
    oggs = sorted(PRIVATE_WA.glob("*.ogg"))
    opus_test = {"suffix_allowed": ".opus" in audio_io.ALLOWED_SUFFIXES,
                 "lossy": audio_io.is_lossy_suffix(".opus"), "decode_ok": [], "decode_err": []}
    for f in oggs[:2]:
        try:
            d = _decode_path(f)
            opus_test["decode_ok"].append({"name": f.suffix, "sr": d.sample_rate,
                                           "dur": round(d.duration_seconds, 2),
                                           "is_lossy": d.is_lossy})
        except Exception as exc:
            opus_test["decode_err"].append(f"{type(exc).__name__}: {exc}")
    # also relabel a real .ogg as .opus to prove the gate, not just soundfile
    if oggs:
        try:
            renamed = decode_audio(oggs[0].read_bytes(), "clip.opus")
            opus_test["renamed_opus_decode"] = {"sr": renamed.sample_rate,
                                                "dur": round(renamed.duration_seconds, 2)}
        except Exception as exc:
            opus_test["renamed_opus_decode_err"] = f"{type(exc).__name__}: {exc}"
    report["opus_accept"] = opus_test

    model = load_voice_model()
    assert model.is_loaded() and getattr(model, "name", "") != "mock", "real RAFNet required"
    head = load_head()
    report["artifacts"] = {"rafnet_loaded": model.is_loaded(),
                           "head_loaded": head.is_loaded(),
                           "head_error": head.error}

    # 2. degenerate guard
    import numpy as np
    guard = {}
    for label, size in (("empty", 0), ("tiny", 800), ("just_under", 1599)):
        try:
            model.predict(np.zeros(size, dtype=np.float32), 16000)
            guard[label] = "NO_ERROR (unexpected)"
        except ValueError as exc:
            guard[label] = f"ValueError: {str(exc)[:80]}"
        except Exception as exc:
            guard[label] = f"{type(exc).__name__}: {str(exc)[:80]}"
    report["degenerate_guard"] = guard

    # 3. determinism (10x in-process) on the first private .ogg
    det = {}
    if oggs:
        d = _decode_path(oggs[0])
        runs = [_score_clip(model, head, d.samples, d.sample_rate) for _ in range(10)]
        digs = {_digest(r) for r in runs}
        det = {"repeats": 10, "unique_digests": len(digs), "digest": sorted(digs)[0],
               "scores": runs[0], "stable": len(digs) == 1}
    report["determinism_in_process"] = det

    # 3b. restart test: a fresh interpreter must reproduce the same digest
    restart = {}
    if oggs:
        code = (
            "import json, sys\n"
            "from pathlib import Path\n"
            "from tools.phase6_verify import _decode_path, _score_clip, _digest\n"
            "from rafnet.soc.head4 import load_head\n"
            "from rafnet.soc.model_adapter import load_voice_model\n"
            f"p = Path({str(oggs[0])!r})\n"
            "d = _decode_path(p)\n"
            "m = load_voice_model(); h = load_head()\n"
            "s = _score_clip(m, h, d.samples, d.sample_rate)\n"
            "print(json.dumps({'digest': _digest(s), 'scores': s}, sort_keys=True))\n"
        )
        try:
            env = dict(os.environ, HF_HUB_OFFLINE="1", MOCK_INFERENCE="false")
            proc = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT),
                                  env=env, capture_output=True, text=True,
                                  timeout=600)
            line = [ln for ln in proc.stdout.splitlines() if ln.strip().startswith("{")]
            got = json.loads(line[-1]) if line else {}
            restart = {"rc": proc.returncode, "digest": got.get("digest"),
                       "in_process_digest": det.get("digest"),
                       "match": got.get("digest") == det.get("digest")}
            if proc.returncode != 0:
                restart["stderr_tail"] = proc.stderr[-300:]
        except Exception as exc:
            restart = {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
    report["determinism_restart"] = restart

    # 4. before/after v1 vs v2
    rows = []
    if oggs:
        for i, f in enumerate(oggs[:2], 1):
            d = _decode_path(f)
            s = _score_clip(model, head, d.samples, d.sample_rate)
            rows.append({"id": f"wa_ogg_{i}", "group": "private_whatsapp_opus",
                         "dur_s": round(d.duration_seconds, 2), **s})

    splits = json.loads((ROOT / "data" / "splits_phase4.json").read_text(encoding="utf-8"))
    test = splits["test"]
    fakes = [r for r in test if int(r["y"]) == 1][:2]
    reals = [r for r in test if int(r["y"]) == 0][:2]
    for r in reals + fakes:
        try:
            d = _decode_path(Path(r["path"]))
            s = _score_clip(model, head, d.samples, d.sample_rate)
            rows.append({"id": r["clip_id"], "group": r["source"], "y": int(r["y"]), **s})
        except Exception as exc:
            rows.append({"id": r["clip_id"], "error": f"{type(exc).__name__}: {str(exc)[:100]}"})
    report["before_after"] = rows

    # codec probe on one real test clip (Opus 24k + AMR-NB)
    codec_rows = []
    if reals:
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        r = reals[0]
        d = _decode_path(Path(r["path"]))
        import soundfile as sf
        for cname, args, ext in (("opus_24k", ["-c:a", "libopus", "-b:a", "24k", "-ar", "16000", "-ac", "1"], ".opus"),
                                 ("amr_nb", ["-ar", "8000", "-ac", "1", "-c:a", "libopencore_amrnb", "-b:a", "12.2k"], ".amr")):
            tmp = Path(tempfile.mkdtemp(prefix="p6_"))
            try:
                src = tmp / "s.wav"
                sf.write(str(src), d.samples, d.sample_rate)
                enc = tmp / f"out{ext}"
                subprocess.run([ff, "-y", "-v", "error", "-i", str(src)] + args + [str(enc)],
                               check=True, timeout=120, capture_output=True)
                dd = decode_audio(enc.read_bytes(), enc.name)
                s = _score_clip(model, head, dd.samples, dd.sample_rate)
                codec_rows.append({"id": r["clip_id"], "codec": cname, **s})
            except Exception as exc:
                codec_rows.append({"id": r["clip_id"], "codec": cname,
                                   "error": f"{type(exc).__name__}: {str(exc)[:100]}"})
            finally:
                import shutil
                shutil.rmtree(tmp, ignore_errors=True)
    report["codec_probe"] = codec_rows

    (ROOT / "reports" / "phase6_verify.json").write_text(json.dumps(report, indent=2),
                                                         encoding="utf-8")
    md = ["# Phase 6 verification (additive v2 stack + guards)", "",
          f"Generated: {report['generated_at']}", "",
          "## Artifacts", json.dumps(report["artifacts"]), "",
          "## .opus accept", json.dumps(report["opus_accept"]), "",
          "## Degenerate-segment guard (<1600 samples)", json.dumps(report["degenerate_guard"]), "",
          "## Determinism (10 in-process repeats, full score path)",
          json.dumps(report["determinism_in_process"]), "",
          "## Determinism (fresh-interpreter restart)",
          json.dumps(report["determinism_restart"]), "",
          "## Before/after (v1 3-feature vs v2 4-feature with head)",
          "| id | group | y | rafnet | mo | bisher | head | p_v1 | v1 | p_v2 | v2 |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in report["before_after"]:
        if "error" in r:
            md.append(f"| {r['id']} | {r.get('group','')} | - | error: {r['error']} |")
            continue
        md.append(f"| {r['id']} | {r['group']} | {r.get('y','-')} | {r['rafnet']} | {r['mo']} | "
                  f"{r['bisher']} | {r['head']} | {r['p_v1']} | {r['verdict_v1']} | "
                  f"{r['p_v2']} | {r['verdict_v2']} |")
    md += ["", "## Codec probe (real clip, Opus/AMR)",
           "| id | codec | p_v1 | v1 | p_v2 | v2 |", "| --- | --- | --- | --- | --- | --- |"]
    for r in codec_rows:
        if "error" in r:
            md.append(f"| {r['id']} | {r['codec']} | error: {r['error']} |")
            continue
        md.append(f"| {r['id']} | {r['codec']} | {r['p_v1']} | {r['verdict_v1']} | "
                  f"{r['p_v2']} | {r['verdict_v2']} |")
    md += ["", "Private clips stay local; ids are anonymous. v2 is additive: v1 fields "
           "and risk gating are unchanged when models/stack_phase4.json is absent."]
    (ROOT / "reports" / "phase6_verify.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("wrote reports/phase6_verify.json + .md")
    print(json.dumps({k: report[k] for k in ("opus_accept", "degenerate_guard",
                                             "determinism_in_process")}, indent=1)[:1500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
