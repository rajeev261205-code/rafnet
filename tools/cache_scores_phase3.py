"""Phase 3 step 1: cache P(AI) scores for every phase2 row (except my_samples).

Models: mo-thecreator/Deepfake-audio-detection, Bisher/... (frozen,
eval mode, torch.inference_mode) + current RAFNet (as third ensemble input).
Audio capped at 6 s. Writes data/scores_cache.json (gitignored, regenerable
via this script). my_samples rows are SKIPPED (untouched until Phase 4).
"""
import csv, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import soundfile as sf
import torch

HF_MODELS = ["mo-thecreator/Deepfake-audio-detection",
             "Bisher/wav2vec2_ASV_deepfake_audio_detection"]
MAX_S = 6

def load16(p):
    arr, sr = sf.read(str(p), dtype="float32", always_2d=False)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2: arr = arr.mean(axis=1).astype(np.float32)
    arr = arr.ravel()
    if int(sr) != 16000 and arr.size:
        tgt = int(min(arr.size / float(sr), float(MAX_S)) * 16000)
        arr = np.interp(np.linspace(0, 1, tgt), np.linspace(0, 1, arr.size), arr).astype(np.float32)
    else: arr = arr[:16000 * MAX_S]
    return arr

def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma list subset of: mo,bisher,rafnet (default all)")
    ap.add_argument("--splits", default="", help="comma list subset of train,val,test (default all non-my_samples)")
    args = ap.parse_args()
    only = {s.strip() for s in args.only.split(",") if s.strip()} or {"mo", "bisher", "rafnet"}
    splits = {s.strip() for s in args.splits.split(",") if s.strip()} or {"train", "val", "test"}
    import torch.nn.functional as F
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
    rows = [r for r in csv.DictReader(open(ROOT / "data" / "phase2_manifest.csv", encoding="utf-8")) if r["split"] in splits]
    print(f"rows to score (splits={sorted(splits)}, no my_samples): {len(rows)}")
    cpath = ROOT / "data" / "scores_cache.json"
    cache = {}
    if cpath.is_file():
        try:
            cache = json.loads(cpath.read_text(encoding="utf-8")).get("scores", {})
            print(f"resumed {len(cache)} cached files")
        except Exception:
            cache = {}
    jobs = []
    if "mo" in only: jobs.append(HF_MODELS[0])
    if "bisher" in only: jobs.append(HF_MODELS[1])
    for mid in jobs:
        feat = AutoFeatureExtractor.from_pretrained(mid, trust_remote_code=False)
        mdl = AutoModelForAudioClassification.from_pretrained(mid, trust_remote_code=False); mdl.eval()
        lab = [str(mdl.config.id2label.get(i, f"L{i}")) for i in range(mdl.config.num_labels)]
        fi = next((i for i, l in enumerate(lab) if any(k in l.lower() for k in ("fake", "spoof", "synthetic", "ai", "deepfake", "aivoice"))), 0)
        t0 = time.perf_counter(); n = 0
        with torch.inference_mode():
            for r in rows:
                if r["path"] in cache and cache[r["path"]].get(mid) is not None:
                    continue
                p = ROOT / r["path"].replace("\\", "/")
                try:
                    arr = load16(p)
                    inp = feat(arr, sampling_rate=16000, return_tensors="pt", padding=True)
                    lg = mdl(**{k: v for k, v in inp.items() if k in ("input_values", "attention_mask")}).logits
                    vec = lg[0] if lg.ndim == 2 else lg.reshape(-1)
                    pr = float(F.softmax(vec, dim=-1)[fi])
                    cache.setdefault(r["path"], {})[mid] = round(pr, 4)
                    n += 1
                    if n % 100 == 0:
                        print(f"{mid} {n}/{len(rows)} ...", flush=True)
                        (ROOT / "data" / "scores_cache.json").write_text(json.dumps({"models": HF_MODELS + ["rafnet"], "scores": cache}, indent=0), encoding="utf-8")
                except Exception as e:
                    cache.setdefault(r["path"], {})[mid] = None
                    print(f"ERR {r['path']}: {e}")
        print(f"{mid}: {n}/{len(rows)} in {time.perf_counter()-t0:.0f}s")
        del mdl, feat
        import gc; gc.collect()
    # RAFNet current (third input)
    if "rafnet" in only:
        import os
        os.environ.setdefault("MOCK_INFERENCE", "false")
        from rafnet.soc.model_adapter import load_voice_model
        rm = load_voice_model()
        assert rm.is_loaded() and getattr(rm, "name", "") != "mock", "real RAFNet must load"
        t0 = time.perf_counter(); n = 0
        for r in rows:
            if r["path"] in cache and cache[r["path"]].get("rafnet") is not None:
                continue
            p = ROOT / r["path"].replace("\\", "/")
            try:
                from rafnet.soc.audio_io import decode_audio
                dec = decode_audio(p.read_bytes(), p.name)
                res = rm.predict(dec.samples, dec.sample_rate)
                cache.setdefault(r["path"], {})["rafnet"] = round(float(res.prob_synthetic), 4)
                n += 1
                if n % 50 == 0:
                    print(f"rafnet {n} ...", flush=True)
                    (ROOT / "data" / "scores_cache.json").write_text(json.dumps({"models": HF_MODELS + ["rafnet"], "scores": cache}, indent=0), encoding="utf-8")
            except Exception as e:
                cache.setdefault(r["path"], {})["rafnet"] = None
                print(f"RAFNet ERR {r['path']}: {e}")
        print(f"rafnet: {n} new in {time.perf_counter()-t0:.0f}s")
    (ROOT / "data" / "scores_cache.json").write_text(json.dumps({"models": HF_MODELS + ["rafnet"], "scores": cache}, indent=0), encoding="utf-8")
    print(f"wrote data/scores_cache.json ({len(cache)} files)")
    return 0
if __name__ == "__main__": raise SystemExit(main())
