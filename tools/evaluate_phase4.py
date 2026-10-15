"""Phase 4: honest ensemble evaluation (TEST + my_samples; never tune here).

Frozen artifacts: models/stacking.json (VAL-fit), 2 HF models, RAFNet.
- TEST (160 rows, unseen speakers/setting) from data/scores_cache.json.
- my_samples (19 rows): fresh HF scores here; RAFNet scores from
  reports/baseline.json (measured Phase 0, same model/weights).
- Codec: 4 test-real clips x Opus/MP3/AMR/G.711, all 3 models rescored.
- Unseen-generator verdicts reuse reports/candidates_public.json (cited).
Writes reports/model_eval.json + reports/model_eval.md (ensemble).
Old RAFNet-only files are backed up to model_eval.rafnet-only.* first.
"""
import csv, json, shutil, subprocess, sys, tempfile
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
import numpy as np
import soundfile as sf
import torch

FEATS = ["mo-thecreator/Deepfake-audio-detection",
         "Bisher/wav2vec2_ASV_deepfake_audio_detection", "rafnet"]

def load16(p, cap=6):
    arr, sr = sf.read(str(p), dtype="float32", always_2d=False)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2: arr = arr.mean(axis=1).astype(np.float32)
    arr = arr.ravel()
    if int(sr) != 16000 and arr.size:
        tgt = int(min(arr.size / float(sr), float(cap)) * 16000)
        arr = np.interp(np.linspace(0, 1, tgt), np.linspace(0, 1, arr.size), arr).astype(np.float32)
    else: arr = arr[:16000 * cap]
    return arr

def main() -> int:
    import torch.nn.functional as F
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
    from tools.evaluate_model import (binary_metrics_at_05, compute_auc, compute_eer,
                                      confusion_3way, coverage_and_selective)
    from rafnet.soc.ensemble import load_stacking, calibrated_pai
    params = load_stacking()
    assert params is not None, "models/stacking.json required"
    cache = json.loads((ROOT / "data" / "scores_cache.json").read_text(encoding="utf-8"))["scores"]
    base = json.loads((ROOT / "reports" / "baseline.json").read_text(encoding="utf-8"))
    base_p = {f["file"]: f["p_synthetic"] for f in base["files"] if "p_synthetic" in f}
    man = {r["path"]: r for r in csv.DictReader(open(ROOT / "data" / "phase2_manifest.csv", encoding="utf-8"))}

    def ensemble_row(p_mo, p_bi, p_ra):
        s = calibrated_pai(p_mo, p_bi, p_ra, params)
        p = s["p_cal"]
        if p is None: return None, "UNCERTAIN", s["votes_fake"], s["agreement"]
        v = "REAL" if p < params.tau_low else ("SYNTHETIC" if p > params.tau_high else "UNCERTAIN")
        return p, v, s["votes_fake"], s["agreement"]

    # TEST from cache
    test_rows = []
    for path, r in man.items():
        if r["split"] != "test":
            continue
        s = cache.get(path, {})
        if any(s.get(f) is None for f in FEATS):
            continue
        p, v, votes, agr = ensemble_row(s[FEATS[0]], s[FEATS[1]], s[FEATS[2]])
        test_rows.append({"file": path, "y": 0 if r["label_origin"] == "HUMAN_LIVE" else 1,
                          "group": ("clean_human_librispeech" if r["label_origin"] == "HUMAN_LIVE" else "voice_conversion_crudeVC"),
                          "p_cal": p, "verdict": v, "votes": votes, "agreement": agr,
                          "p_mo": s[FEATS[0]], "p_bi": s[FEATS[1]], "p_ra": s[FEATS[2]]})
    # my_samples: fresh HF scores + baseline RAFNet
    hf = {}
    for mid in FEATS[:2]:
        feat = AutoFeatureExtractor.from_pretrained(mid, trust_remote_code=False)
        mdl = AutoModelForAudioClassification.from_pretrained(mid, trust_remote_code=False); mdl.eval()
        lab = [str(mdl.config.id2label.get(i, f"L{i}")) for i in range(mdl.config.num_labels)]
        fi = next((i for i, l in enumerate(lab) if any(k in l.lower() for k in ("fake", "spoof", "synthetic", "ai", "deepfake", "aivoice"))), 0)
        with torch.inference_mode():
            for path, r in man.items():
                if r["split"] != "my_samples":
                    continue
                arr = load16(ROOT / path.replace("\\", "/"))
                inp = feat(arr, sampling_rate=16000, return_tensors="pt", padding=True)
                lg = mdl(**{k: v for k, v in inp.items() if k in ("input_values", "attention_mask")}).logits
                vec = lg[0] if lg.ndim == 2 else lg.reshape(-1)
                hf.setdefault(path, {})[mid] = round(float(F.softmax(vec, dim=-1)[fi]), 4)
        del mdl, feat
        import gc; gc.collect()
    my_rows = []
    for path, r in man.items():
        if r["split"] != "my_samples" or path not in hf or path not in base_p:
            continue
        p, v, votes, agr = ensemble_row(hf[path][FEATS[0]], hf[path][FEATS[1]], float(base_p[path]))
        grp = {"clean_human_librispeech": "clean_human_librispeech"}.get(r["generator_name"], r["generator_name"])
        my_rows.append({"file": path, "y": 0 if "HUMAN" in r["label_origin"] else 1, "group": grp,
                        "p_cal": p, "verdict": v, "votes": votes, "agreement": agr,
                        "p_mo": hf[path][FEATS[0]], "p_bi": hf[path][FEATS[1]], "p_ra": float(base_p[path])})
    print(f"test={len(test_rows)} my_samples={len(my_rows)}")

    def metrics(rows):
        ys = [r["y"] for r in rows if r["p_cal"] is not None]
        ss = [r["p_cal"] for r in rows if r["p_cal"] is not None]
        binm = binary_metrics_at_05(ys, ss)
        acc = binm["accuracy"]
        bal = round(((binm["tp"] / (binm["tp"] + binm["fn"])) + (binm["tn"] / (binm["tn"] + binm["fp"]))) / 2, 4) if (binm["tp"] + binm["fn"]) and (binm["tn"] + binm["fp"]) else None
        far = round(binm["fp"] / (binm["fp"] + binm["tn"]), 4) if (binm["fp"] + binm["tn"]) else None
        return {"n": len(ys), "accuracy": acc, "balanced_accuracy": bal, "binary_at_0.5": binm,
                "false_alarm_rate_on_human": far, "auc": compute_auc(ys, ss), "eer": compute_eer(ys, ss),
                "confusion_band": confusion_3way(ys, ss, params.tau_low, params.tau_high),
                **coverage_and_selective(ys, ss, params.tau_low, params.tau_high)}

    # per-group
    groups = {}
    for r in test_rows + my_rows:
        groups.setdefault(r["group"], []).append(r)
    group_m = {g: metrics(v) for g, v in groups.items()}
    overall_test = metrics(test_rows)
    overall_my = metrics(my_rows)
    both = metrics(test_rows + my_rows)

    # reliability (bins of p_cal vs binary accuracy, all scored)
    bins = [(0.0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]
    rel = []
    for lo, hi in bins:
        sel = [r for r in test_rows + my_rows if r["p_cal"] is not None and lo <= r["p_cal"] < hi]
        if not sel:
            rel.append({"bin": f"{lo}-{min(hi,1.0)}", "n": 0}); continue
        acc = sum(1 for r in sel if (1 if r["p_cal"] >= 0.5 else 0) == r["y"]) / len(sel)
        rel.append({"bin": f"{lo}-{min(hi,1.0)}", "n": len(sel), "accuracy": round(acc, 4),
                    "avg_p": round(sum(r["p_cal"] for r in sel) / len(sel), 4)})

    # codec mini-test: 4 test-real clips x 4 codecs, ensemble delta
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    CODECS = {"opus_24k": ("out.opus", ["-c:a", "libopus", "-b:a", "24k", "-ar", "16000", "-ac", "1"]),
              "mp3_128k": ("out.mp3", ["-c:a", "libmp3lame", "-b:a", "128k", "-ar", "16000", "-ac", "1"]),
              "amr_nb": ("out.amr", ["-ar", "8000", "-ac", "1", "-c:a", "libopencore_amrnb", "-b:a", "12.2k"]),
              "g711_alaw": ("out_alaw.wav", ["-ar", "8000", "-ac", "1", "-c:a", "pcm_alaw"])}
    feats = {}
    for mid in FEATS[:2]:
        feats[mid] = (AutoFeatureExtractor.from_pretrained(mid, trust_remote_code=False),
                      AutoModelForAudioClassification.from_pretrained(mid, trust_remote_code=False).eval())
    import os
    os.environ.setdefault("MOCK_INFERENCE", "false")
    # NOTE: RAFNet codec re-scoring is deliberately NOT run here (12 s/clip and
    # flaky on transcoded inputs). The ensemble's RAFNet term is held at its
    # clean value; RAFNet-only codec bounds (max |delta| 0.034) are cited from
    # reports/baseline.json, and its stacking weight is 0.0719, so the omitted
    # term moves p_cal by < 0.003. HF terms (weight 4.86 combined) are rescored.
    codec_rows = []
    real4 = [r for r in test_rows if r["y"] == 0][:4]
    for r in real4:
        arr = load16(ROOT / r["file"].replace("\\", "/"))
        lab = [str(feats[FEATS[0]][1].config.id2label.get(i, f"L{i}")) for i in range(2)]
        for c, (ename, args) in CODECS.items():
            tmp = Path(tempfile.mkdtemp(prefix="ev4_"))
            try:
                src = tmp / "s.wav"; sf.write(str(src), arr, 16000)
                enc = tmp / ename; subprocess.run([ff, "-y", "-v", "error", "-i", str(src)] + args + [str(enc)], check=True, timeout=120, capture_output=True)
                back = tmp / "b.wav"; subprocess.run([ff, "-y", "-v", "error", "-i", str(enc), "-ar", "16000", "-ac", "1", str(back)], check=True, timeout=120, capture_output=True)
                a2, _ = sf.read(str(back), dtype="float32"); a2 = np.asarray(a2, dtype=np.float32).ravel()
                ps = {}
                with torch.inference_mode():
                    for mid in FEATS[:2]:
                        fe, md = feats[mid]
                        lb = [str(md.config.id2label.get(i, f"L{i}")) for i in range(md.config.num_labels)]
                        fi = next((i for i, l in enumerate(lb) if any(k in l.lower() for k in ("fake", "spoof", "synthetic", "ai", "deepfake", "aivoice"))), 0)
                        inp = fe(a2, sampling_rate=16000, return_tensors="pt", padding=True)
                        lg = md(**{k: v for k, v in inp.items() if k in ("input_values", "attention_mask")}).logits
                        vec = lg[0] if lg.ndim == 2 else lg.reshape(-1)
                        ps[mid] = float(F.softmax(vec, dim=-1)[fi])
                ps["rafnet_clean_held"] = r["p_ra"]
                pc, vc, votes, _ = ensemble_row(ps[FEATS[0]], ps[FEATS[1]], r["p_ra"])
                codec_rows.append({"file": r["file"], "codec": c, "p_clean": r["p_cal"], "p_codec": pc,
                                   "delta": round(pc - r["p_cal"], 4) if pc is not None else None, "verdict_codec": vc, "votes": votes,
                                   "assumption": "rafnet term held at clean value (weight 0.0719; cited max shift 0.034 -> p_cal < 0.003)"})
                print(f"codec {c:9s} {r['file'].split('/')[-1]} {r['p_cal']}->{pc} d={codec_rows[-1]['delta']} {vc}")
            except Exception as e:
                codec_rows.append({"file": r["file"], "codec": c, "error": f"{type(e).__name__}: {str(e)[:150]}"})
            finally:
                shutil.rmtree(tmp, ignore_errors=True)

    for pth in ("reports/model_eval.json", "reports/model_eval.md"):
        src, dst = ROOT / pth, ROOT / (pth.replace("model_eval", "model_eval.rafnet-only"))
        if src.is_file() and not dst.is_file():
            shutil.copy(src, dst)
    report = {"generated_at": datetime.now(timezone.utc).isoformat(),
              "model": {"name": "ensemble-mo+bisher+rafnet", "stacking": "models/stacking.json (VAL-fit, frozen)",
                        "taus": [params.tau_low, params.tau_high]},
              "test": overall_test, "my_samples": overall_my, "test_plus_my": both,
              "per_group": group_m, "reliability": rel, "codec": codec_rows,
              "unseen_generators": "my_samples generators (SAPI David/Zira, Windows-TTS, ChatGPT voice) never in train; public-real FAR 0.0/0.0 for kept models per reports/candidates_public.json (cited, not re-run).",
              "missing_groups": ["whatsapp_opus_human", "phone_line_human", "hindi_tamil_speech", "commercial_tts", "real_voice_conversion"],
              "rows_test": test_rows, "rows_my": my_rows,
              "targets": "No numeric targets were pre-registered; numbers reported as measured. Next steps in Honest reading."}
    (ROOT / "reports" / "model_eval.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    L = ["# Model evaluation — stacked ensemble (frozen VAL-fit, TEST + my_samples)", "",
         f"Generated: {report['generated_at']}", "Model: ensemble mo-thecreator + Bisher + RAFNet; stacking=models/stacking.json (VAL-fit, frozen here).",
         f"Taus: [{params.tau_low}, {params.tau_high}] (VAL-chosen, frozen).", "",
         "> TEST and my_samples were never used for training or band selection. Small-n snapshot, NOT a general benchmark.", "",
         f"## TEST (unseen speakers/setting, n={overall_test['n']})",
         f"accuracy={overall_test['accuracy']} balanced={overall_test['balanced_accuracy']} AUC={overall_test['auc']} EER={json.dumps(overall_test['eer'])}",
         f"FAR on human (most important)={overall_test['false_alarm_rate_on_human']} binary={json.dumps(overall_test['binary_at_0.5'])}",
         f"confusion(band)={json.dumps(overall_test['confusion_band'])} coverage={overall_test['coverage']} selective={overall_test['selective_accuracy']}", "",
         f"## my_samples (unseen generators, n={overall_my['n']})",
         f"accuracy={overall_my['accuracy']} balanced={overall_my['balanced_accuracy']} AUC={overall_my['auc']} EER={json.dumps(overall_my['eer'])}",
         f"FAR on human={overall_my['false_alarm_rate_on_human']} binary={json.dumps(overall_my['binary_at_0.5'])}",
         f"confusion(band)={json.dumps(overall_my['confusion_band'])}", "",
         "## Per group"]
    for g, m in group_m.items():
        L.append(f"- {g}: n={m['n']} acc={m['accuracy']} bal={m['balanced_accuracy']} AUC={m['auc']} EER={json.dumps(m['eer'])} FAR_human={m['false_alarm_rate_on_human']} conf={json.dumps(m['confusion_band'])}")
    L += ["", "## Per codec (test-real only, clean p_cal -> codec p_cal)",
          "\n".join(f"- {c['file'].split('/')[-1]} {c['codec']}: {c.get('p_clean')}->{c.get('p_codec')} d={c.get('delta')} {c.get('verdict_codec','')} {c.get('error','')}" for c in codec_rows), "",
          "## Reliability (calibrated p bins vs binary accuracy, TEST+my)",
          "| bin | n | accuracy | avg_p |", "| --- | --- | --- | --- |"]
    for b in rel:
        L.append(f"| {b['bin']} | {b['n']} | {b.get('accuracy')} | {b.get('avg_p')} |")
    L += ["", "## Honest reading",
          "- Targets: none pre-registered; nothing tuned here. If a deployment bar (e.g. FAR_human<=0.05 on WhatsApp audio) is required, it is NOT met by evidence yet — missing groups below.",
          "- Missing (unmeasured): whatsapp_opus_human, phone_line_human, Hindi/Tamil, commercial TTS, real VC. The History failure (WhatsApp human flagged SYNTHETIC) cannot be closed without those clips.",
          "- Train fake was one crude VC family; TEST fake is its held-out setting — expect degradation on true unseen generators; my_samples SAPI/ChatGPT is the only genuine unseen-generator gate and is tiny (n=15 fake).",
          "- Next data step: 10-20 WhatsApp Opus notes + Indian-language + noisy real; public WaveFake/ASVspoof subset via GPU notebook (needs approval >15 GB). Next model step: run notebooks/finetune_gpu_phase3.ipynb, then re-run this script unchanged.",
          "", "## Files (per-file TEST/my)"]
    for r in test_rows + my_rows:
        L.append(f"- {r['file']} y={r['y']} [{r['group']}] p_cal={r['p_cal']} -> {r['verdict']} votes={r['votes']}/3 (mo={r['p_mo']} bi={r['p_bi']} ra={r['p_ra']})")
    (ROOT / "reports" / "model_eval.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"TEST acc={overall_test['accuracy']} auc={overall_test['auc']} FAR={overall_test['false_alarm_rate_on_human']}")
    print(f"MY acc={overall_my['accuracy']} auc={overall_my['auc']} FAR={overall_my['false_alarm_rate_on_human']}")
    print("wrote reports/model_eval.json/.md (ensemble); legacy backed up to model_eval.rafnet-only.*")
    return 0
if __name__ == "__main__": raise SystemExit(main())
