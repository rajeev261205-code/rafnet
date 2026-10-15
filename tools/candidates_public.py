"""Phase 1 step 4: public-set + codec-robustness probe of top shortlisted models.

Public real: 6 LibriSpeech test-clean clips from data/raw, speakers
1320/1580 (disjoint from samples/ speakers 260/4446/5142/6930) — verifies the
candidates do not flag clean public speech as fake.
Codec: each of 2 public-real files transcoded to Opus 24k / MP3 128k /
AMR-NB / G.711 via the imageio-ffmpeg binary; score shift recorded.

Models: the already-cached top-3 (mo-thecreator, Hemgg, Bisher) only —
no new downloads. Writes reports/candidates_public.json.
"""
import json, subprocess, sys, tempfile, shutil
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np, torch
import soundfile as sf

PUB_REAL = [
 "data/raw/LibriSpeech/test-clean/1320/122612/1320-122612-0000.flac",
 "data/raw/LibriSpeech/test-clean/1320/122612/1320-122612-0001.flac",
 "data/raw/LibriSpeech/test-clean/1320/122612/1320-122612-0002.flac",
 "data/raw/LibriSpeech/test-clean/1580/141083/1580-141083-0000.flac",
 "data/raw/LibriSpeech/test-clean/1580/141083/1580-141083-0001.flac",
 "data/raw/LibriSpeech/test-clean/1580/141083/1580-141083-0002.flac",
]
MODELS = ["mo-thecreator/Deepfake-audio-detection",
          "Hemgg/Deepfake-audio-detection",
          "Bisher/wav2vec2_ASV_deepfake_audio_detection"]
CODECS = ["opus_24k", "mp3_128k", "amr_nb", "g711_alaw"]

def load16(p, max_s=6):
    arr, sr = sf.read(str(p), dtype="float32", always_2d=False)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2: arr = arr.mean(axis=1).astype(np.float32)
    arr = arr.ravel()
    if int(sr) != 16000 and arr.size:
        tgt = int(min(arr.size / float(sr), float(max_s)) * 16000)
        arr = np.interp(np.linspace(0, 1, tgt), np.linspace(0, 1, arr.size), arr).astype(np.float32)
    else: arr = arr[:16000 * max_s]
    return arr, 16000

def ffmpeg_bin():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()

def transcode_back(ff, src_wav: Path, codec: str) -> bytes:
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="pub_codec_"))
    try:
        if codec == "opus_24k":
            enc = tmp / "o.opus"; cmd = [ff, "-y", "-v", "error", "-i", str(src_wav), "-c:a", "libopus", "-b:a", "24k", "-ar", "16000", "-ac", "1", str(enc)]
        elif codec == "mp3_128k":
            enc = tmp / "o.mp3"; cmd = [ff, "-y", "-v", "error", "-i", str(src_wav), "-c:a", "libmp3lame", "-b:a", "128k", "-ar", "16000", "-ac", "1", str(enc)]
        elif codec == "amr_nb":
            enc = tmp / "o.amr"; cmd = [ff, "-y", "-v", "error", "-i", str(src_wav), "-ar", "8000", "-ac", "1", "-c:a", "libopencore_amrnb", "-b:a", "12.2k", str(enc)]
        else:
            enc = tmp / "o.wav"; cmd = [ff, "-y", "-v", "error", "-i", str(src_wav), "-ar", "8000", "-ac", "1", "-c:a", "pcm_alaw", str(enc)]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        assert r.returncode == 0 and enc.is_file(), (r.stderr or "")[:200]
        back = tmp / "b.wav"
        r2 = subprocess.run([ff, "-y", "-v", "error", "-i", str(enc), "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(back)], capture_output=True, text=True, timeout=120)
        assert r2.returncode == 0 and back.is_file(), (r2.stderr or "")[:200]
        return back.read_bytes()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

def main() -> int:
    import torch.nn.functional as F
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
    sys.path.insert(0, str(ROOT))
    from tools.evaluate_model import compute_auc, compute_eer
    ff = ffmpeg_bin()
    out = {"models": {}}
    for mid in MODELS:
        print(f"=== {mid} ===")
        feat = AutoFeatureExtractor.from_pretrained(mid, trust_remote_code=False)
        mdl = AutoModelForAudioClassification.from_pretrained(mid, trust_remote_code=False); mdl.eval()
        lab = [str(mdl.config.id2label.get(i, f"L{i}")) for i in range(mdl.config.num_labels)]
        fi = next((i for i, l in enumerate(lab) if any(k in l.lower() for k in ("fake", "spoof", "synthetic", "ai", "deepfake", "aivoice"))), 0)
        def score(arr):
            inp = feat(arr, sampling_rate=16000, return_tensors="pt", padding=True)
            with torch.inference_mode():
                lg = mdl(**{k: v for k, v in inp.items() if k in ("input_values", "attention_mask")}).logits
                vec = lg[0] if lg.ndim == 2 else lg.reshape(-1)
                return float(F.softmax(vec, dim=-1)[fi])
        pub = []
        for rel in PUB_REAL:
            arr, _ = load16(ROOT / rel)
            p = score(arr)
            pub.append({"file": rel, "p_ai": round(p, 4)})
            print(f"  pub-real {rel.split('/')[-1]} p={p:.4f}")
        # codec on first 2 public files
        codec_rows = []
        for rel in PUB_REAL[:2]:
            arr, _ = load16(ROOT / rel)
            p_clean = score(arr)
            with tempfile.TemporaryDirectory() as td:
                src = Path(td) / "s.wav"; sf.write(str(src), arr, 16000)
                for c in CODECS:
                    try:
                        b = transcode_back(ff, src, c)
                        import io
                        a2, s2 = sf.read(io.BytesIO(b), dtype="float32", always_2d=False)
                        a2 = np.asarray(a2, dtype=np.float32).ravel()
                        p_c = score(a2)
                        codec_rows.append({"file": rel, "codec": c, "p_clean": round(p_clean, 4), "p_codec": round(p_c, 4), "delta": round(p_c - p_clean, 4)})
                        print(f"  codec {c:9s} {rel.split('/')[-1]} {p_clean:.4f}->{p_c:.4f} d={p_c-p_clean:+.4f}")
                    except Exception as e:
                        codec_rows.append({"file": rel, "codec": c, "error": f"{type(e).__name__}: {str(e)[:150]}"})
        far_pub = sum(1 for r in pub if r["p_ai"] >= 0.5) / len(pub)
        out["models"][mid] = {"labels": lab, "fake_index": fi, "public_real": pub,
                              "public_real_far_at_0.5": round(far_pub, 4),
                              "codec": codec_rows}
        del mdl, feat
        import gc; gc.collect()
    (ROOT / "reports" / "candidates_public.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote reports/candidates_public.json")
    return 0
if __name__ == "__main__": raise SystemExit(main())
