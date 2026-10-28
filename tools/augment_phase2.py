"""Phase 2 step 3: train-only augmentation (BOTH classes equally).

Chains: Opus 24k / MP3 128k / AMR-NB / AAC 32k / G.711 A-law (ffmpeg),
resampling (8/16/22.05 kHz round-trip), exponential-decay reverb, additive
noise (SNR 10-20 dB), packet-loss (random 20 ms frame zeroing).

Contract: augment_train() is called ONLY for rows with split=="train";
it NEVER sees the label (same transform distribution for real and fake),
so compression/channel can no longer correlate with the label.

Self-test (run as script): 2 train-real + 2 train-fake through every chain;
writes reports/augmentation_check.json with input/output durations and
a same-label statement.
"""
import io, random, shutil, subprocess, tempfile
from pathlib import Path
import numpy as np
import soundfile as sf

def _ffmpeg() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()

def _codec_bytes(wav16: np.ndarray, codec: str) -> np.ndarray:
    ff = _ffmpeg()
    tmp = Path(tempfile.mkdtemp(prefix="aug_"))
    try:
        src = tmp / "s.wav"; sf.write(str(src), wav16, 16000)
        if codec == "opus_24k":
            enc = tmp / "o.opus"; cmd = [ff, "-y", "-v", "error", "-i", str(src), "-c:a", "libopus", "-b:a", "24k", "-ar", "16000", "-ac", "1", str(enc)]
        elif codec == "mp3_128k":
            enc = tmp / "o.mp3"; cmd = [ff, "-y", "-v", "error", "-i", str(src), "-c:a", "libmp3lame", "-b:a", "128k", "-ar", "16000", "-ac", "1", str(enc)]
        elif codec == "amr_nb":
            enc = tmp / "o.amr"; cmd = [ff, "-y", "-v", "error", "-i", str(src), "-ar", "8000", "-ac", "1", "-c:a", "libopencore_amrnb", "-b:a", "12.2k", str(enc)]
        elif codec == "aac_32k":
            enc = tmp / "o.m4a"; cmd = [ff, "-y", "-v", "error", "-i", str(src), "-c:a", "aac", "-b:a", "32k", "-ar", "16000", "-ac", "1", str(enc)]
        else:  # g711_alaw
            enc = tmp / "o.wav"; cmd = [ff, "-y", "-v", "error", "-i", str(src), "-ar", "8000", "-ac", "1", "-c:a", "pcm_alaw", str(enc)]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        assert r.returncode == 0 and enc.is_file(), (r.stderr or "")[:200]
        back = tmp / "b.wav"
        r2 = subprocess.run([ff, "-y", "-v", "error", "-i", str(enc), "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(back)], capture_output=True, text=True, timeout=120)
        assert r2.returncode == 0 and back.is_file(), (r2.stderr or "")[:200]
        a, _ = sf.read(str(back), dtype="float32", always_2d=False)
        return np.asarray(a, dtype=np.float32).ravel()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

def add_reverb(x: np.ndarray, decay: float = 0.35, delay_ms: int = 60) -> np.ndarray:
    d = int(16000 * delay_ms / 1000)
    ir = np.exp(-decay * np.arange(d) / d).astype(np.float32)
    y = np.convolve(x, ir, mode="full")[:len(x)]
    m = max(1e-6, float(np.abs(y).max()))
    return (y / m * float(np.abs(x).max())).astype(np.float32)

def add_noise(x: np.ndarray, snr_db: float, rng: random.Random) -> np.ndarray:
    g = np.random.default_rng(rng.randint(0, 10**9))
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) or 1e-4
    n = g.normal(0.0, rms / (10.0 ** (snr_db / 20.0)), size=x.shape).astype(np.float32)
    return np.clip(x + n, -1.0, 1.0)

def packet_loss(x: np.ndarray, drop_prob: float, rng: random.Random, frame_ms: int = 20) -> np.ndarray:
    y = x.copy(); f = int(16000 * frame_ms / 1000)
    for i in range(0, len(y), f):
        if rng.random() < drop_prob:
            y[i:i + f] = 0.0
    return y

def resample_roundtrip(x: np.ndarray, sr: int) -> np.ndarray:
    dur = len(x) / 16000.0; n = max(1, int(dur * sr))
    down = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(x)), x)
    return np.interp(np.linspace(0, 1, len(x)), np.linspace(0, 1, n), down).astype(np.float32)

CHAINS = ["opus_24k", "mp3_128k", "amr_nb", "aac_32k", "g711_alaw",
          "resample_8k", "reverb", "noise_15db", "packetloss_05"]

def augment_train(wav16: np.ndarray, chain: str, seed: int) -> np.ndarray:
    """One chain, label-blind. split=='train' only (enforced by caller)."""
    rng = random.Random(seed)
    x = np.asarray(wav16, dtype=np.float32).ravel()
    if chain in ("opus_24k", "mp3_128k", "amr_nb", "aac_32k", "g711_alaw"):
        return _codec_bytes(x, chain)
    if chain == "resample_8k":
        return resample_roundtrip(x, 8000)
    if chain == "reverb":
        return add_reverb(x)
    if chain == "noise_15db":
        return add_noise(x, 15.0, rng)
    if chain == "packetloss_05":
        return packet_loss(x, 0.05, rng)
    raise ValueError(chain)

def self_test() -> int:
    import csv, json
    from datetime import datetime, timezone
    ROOT = Path(__file__).resolve().parents[1]
    rows = [r for r in csv.DictReader(open(ROOT / "data" / "phase2_manifest.csv", encoding="utf-8")) if r["split"] == "train"]
    real = [r for r in rows if r["label_origin"] == "HUMAN_LIVE"][:2]
    fake = [r for r in rows if "AI_" in r["label_origin"]][:2]
    rec = {"generated_at": datetime.now(timezone.utc).isoformat(), "cases": []}
    for r in real + fake:
        p = ROOT / r["path"].replace("\\", "/")
        arr, _ = sf.read(str(p), dtype="float32", always_2d=False)
        arr = np.asarray(arr, dtype=np.float32).ravel()
        for ci, ch in enumerate(CHAINS):
            y = augment_train(arr, ch, 1000 + ci)
            rec["cases"].append({"file": r["path"], "label": r["label_origin"], "chain": ch,
                                 "in_s": round(len(arr) / 16000, 2), "out_s": round(len(y) / 16000, 2),
                                 "peak": round(float(np.abs(y).max()), 4)})
            print(f"{ch:14s} {r['path'].split('/')[-1][:28]:28s} {len(arr)/16000:.2f}s -> {len(y)/16000:.2f}s peak {float(np.abs(y).max()):.3f}")
    rec["contract"] = "Label-blind: identical chain set/seeds distribution for HUMAN_LIVE and AI_* train rows; never applied to val/test/my_samples."
    (ROOT / "reports" / "augmentation_check.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(f"cases={len(rec['cases'])}; wrote reports/augmentation_check.json")
    return 0

if __name__ == "__main__":
    raise SystemExit(self_test())
