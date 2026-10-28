"""Test ONE HF audio-classification candidate quickly (6 files, 4 s cap)."""
import json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch
FILES = [
 ("samples/real/260-123286-0012.flac", 0),
 ("samples/real/6930-81414-0025.flac", 0),
 ("samples/fake/sapi_david_2.wav", 1),
 ("samples/fake/sapi_zira_2.wav", 1),
 ("samples/fake/chatgpt_voice/rafnet_ai_voice_test.wav", 1),
 ("samples/fake/windows_tts/tts_test.wav", 1),
]
def load_audio(p, max_s=4):
    import soundfile as sf
    arr, sr = sf.read(str(p), dtype="float32", always_2d=False)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr.mean(axis=1).astype(np.float32)
    arr = arr.ravel().astype(np.float32)
    if int(sr) != 16000 and arr.size:
        dur = arr.size / float(sr)
        tgt = int(min(dur, float(max_s)) * 16000)
        old_idx = np.linspace(0.0, 1.0, num=arr.size)
        new_idx = np.linspace(0.0, 1.0, num=max(1, tgt))
        arr = np.interp(new_idx, old_idx, arr).astype(np.float32)
        sr = 16000
    else:
        arr = arr[: int(sr) * int(max_s)]
    return arr, 16000
def main():
    mid = sys.argv[1]
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
    import torch.nn.functional as F
    t0=time.perf_counter()
    try:
        feat = AutoFeatureExtractor.from_pretrained(mid, trust_remote_code=False)
        mdl = AutoModelForAudioClassification.from_pretrained(mid, trust_remote_code=False); mdl.eval()
        print(f"LOADED {mid} labels={dict(mdl.config.id2label)} params={sum(p.numel() for p in mdl.parameters())/1e6:.1f}M in {time.perf_counter()-t0:.1f}s", flush=True)
    except Exception as e:
        print(f"LOAD_FAIL {mid}: {type(e).__name__}: {str(e)[:300]}", flush=True); return 1
    sys.path.insert(0, str(ROOT))
    from tools.evaluate_model import compute_auc, compute_eer
    out=[]
    with torch.inference_mode():
        for rel, y in FILES:
            p = ROOT/rel
            try:
                arr, sr = load_audio(p)
                inp = feat(arr, sampling_rate=sr, return_tensors="pt", padding=True)
                s=time.perf_counter()
                out_logits = mdl(**{k:v for k,v in inp.items() if k in ("input_values","attention_mask")}).logits
                import torch.nn.functional as F
                vec = out_logits[0] if out_logits.ndim == 2 else out_logits.reshape(-1)
                dt=(time.perf_counter()-s)*1000
                probs=F.softmax(vec,dim=-1).tolist()
                if isinstance(probs, float):
                    probs=[1.0-probs, probs]
                labels=[str(mdl.config.id2label.get(i, mdl.config.id2label.get(str(i), f"L{i}"))) for i in range(len(probs))]
                low=" ".join(labels).lower()
                fi=next((i for i,l in enumerate(labels) if any(k in l.lower() for k in ("fake","spoof","synthetic","ai","deepfake"))), None)
                pai=float(probs[fi]) if fi is not None else (float(probs[1]) if len(probs)==2 else float(max(probs)))
                out.append({"file":rel,"y":y,"p":round(pai,4),"ms":round(dt,1),"labels":labels})
                print(f"  {rel.split('/')[-1][:30]:30s} y={y} p={pai:.4f} {dt:.0f}ms {labels}", flush=True)
            except Exception as e:
                import traceback
                print(f"  ERR {rel}: {type(e).__name__}: {e}\n{traceback.format_exc(limit=5)}", flush=True)
    ys=[o["y"] for o in out]; ss=[o["p"] for o in out]
    print(f"AUC={compute_auc(ys,ss)} EER={compute_eer(ys,ss)}", flush=True)
    safe=mid.replace("/","__")
    (ROOT/"reports"/f"zeroshot_{safe}.json").write_text(json.dumps({"id":mid,"files":out,"auc":compute_auc(ys,ss),"eer":compute_eer(ys,ss)},indent=2))
    print("wrote reports/zeroshot_"+safe+".json", flush=True)
if __name__=="__main__": main()
