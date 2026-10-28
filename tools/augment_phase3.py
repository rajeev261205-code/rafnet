"""Phase 3 codec-fair augmentation — identical random chains, both classes.

The Phase-0 failure (human WhatsApp -> SYNTHETIC) correlates with channel.
Fix: every training clip, real or fake, passes through the SAME seeded
random chain distribution, so compression/channel carries no label signal.
Batches are produced ON THE FLY (no full copies stored); intermediates are
tempfiles deleted after each clip.

Chain (seeded per (clip_id, epoch)):
  codec:  none | opus_12k | opus_24k | opus_32k | amr_nb | g711 | mp3_64k |
          mp3_128k | aac_32k   (each ~1/9; real ffmpeg encode+decode)
  rate:   none | 8k->16k | 44.1k->16k  (band-limiting resample round-trip)
  noise:  none | white/pink to SNR {30, 25, 20} dB  (private SNR was ~28)
  reverb: none | synthetic exponential-decay IR, RT60 {0.2, 0.4} s
  gain:   -6..+6 dB (gain probe showed legacy path gain-invariant; kept small)
  loss:   none | 1-3x20ms dropouts (packet-loss concealment = zeros)

SLR28 real RIRs skipped (1.3 GB, mirror ~300 kB/s, no Range support);
synthetic IRs documented as such. Noisy/reverberant REAL speech already
covered by GramVaani telephone + dev-other in the training pool.
"""
from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import imageio_ffmpeg

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

CODECS = ["none", "opus_12k", "opus_24k", "opus_32k", "amr_nb", "g711",
          "mp3_64k", "mp3_128k", "aac_32k"]
RATES = ["none", "8k", "44.1k"]
NOISES = ["none", "white30", "white25", "white20", "pink25"]
REVERBS = ["none", "rt60_02", "rt60_04"]
LOSSES = ["none", "drop1", "drop3"]


@dataclass
class Chain:
    codec: str = "none"
    rate: str = "none"
    noise: str = "none"
    reverb: str = "none"
    gain_db: float = 0.0
    loss: str = "none"

    def describe(self) -> str:
        return (f"codec={self.codec} rate={self.rate} noise={self.noise} "
                f"reverb={self.reverb} gain={self.gain_db:+.0f}dB loss={self.loss}")


def sample_chain(rng: np.random.Generator) -> Chain:
    return Chain(
        codec=str(rng.choice(CODECS)),
        rate=str(rng.choice(RATES, p=[0.6, 0.2, 0.2])),
        noise=str(rng.choice(NOISES, p=[0.4, 0.15, 0.15, 0.15, 0.15])),
        reverb=str(rng.choice(REVERBS, p=[0.6, 0.2, 0.2])),
        gain_db=float(rng.uniform(-6, 6)),
        loss=str(rng.choice(LOSSES, p=[0.7, 0.2, 0.1])))


def ffmpeg_codec(x: np.ndarray, codec: str) -> np.ndarray:
    import soundfile as sf

    with tempfile.TemporaryDirectory(prefix="rafnet_aug_") as td:
        td = Path(td)
        src = td / "s.wav"
        sf.write(str(src), x, 16000)
        if codec.startswith("opus_"):
            enc, cmd = td / "e.opus", None
            br = codec.split("_")[1]
            cmd = [FFMPEG, "-y", "-v", "error", "-i", str(src), "-c:a", "libopus",
                   "-b:a", br, "-ar", "16000", "-ac", "1", str(enc)]
        elif codec == "amr_nb":
            enc = td / "e.amr"
            cmd = [FFMPEG, "-y", "-v", "error", "-i", str(src), "-ar", "8000",
                   "-ac", "1", "-c:a", "libopencore_amrnb", "-b:a", "12.2k", str(enc)]
        elif codec == "g711":
            enc = td / "e.wav"
            cmd = [FFMPEG, "-y", "-v", "error", "-i", str(src), "-ar", "8000",
                   "-ac", "1", "-c:a", "pcm_alaw", str(enc)]
        elif codec.startswith("mp3_"):
            enc = td / "e.mp3"
            br = codec.split("_")[1]
            cmd = [FFMPEG, "-y", "-v", "error", "-i", str(src), "-c:a", "libmp3lame",
                   "-b:a", br, "-ar", "16000", "-ac", "1", str(enc)]
        elif codec == "aac_32k":
            enc = td / "e.m4a"
            cmd = [FFMPEG, "-y", "-v", "error", "-i", str(src), "-c:a", "aac",
                   "-b:a", "32k", "-ar", "16000", "-ac", "1", str(enc)]
        else:
            raise ValueError(codec)
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if r.returncode != 0 or not enc.is_file():
            raise RuntimeError(f"aug encode {codec}: {(r.stderr or '')[:150]}")
        back = td / "b.wav"
        r2 = subprocess.run([FFMPEG, "-y", "-v", "error", "-i", str(enc), "-ar",
                             "16000", "-ac", "1", "-c:a", "pcm_s16le", str(back)],
                            capture_output=True, text=True, timeout=120)
        if r2.returncode != 0 or not back.is_file():
            raise RuntimeError(f"aug decode {codec}: {(r2.stderr or '')[:150]}")
        y, _ = sf.read(str(back), dtype="float32")
    return np.asarray(y, dtype=np.float32).ravel()


def resample_rt(x: np.ndarray, rate: str) -> np.ndarray:
    import soundfile as sf

    if rate == "none":
        return x
    sr = 8000 if rate == "8k" else 44100
    with tempfile.TemporaryDirectory(prefix="rafnet_rate_") as td:
        td = Path(td)
        a, b = td / "a.wav", td / "b.wav"
        sf.write(str(a), x, 16000)
        subprocess.run([FFMPEG, "-y", "-v", "error", "-i", str(a), "-ar", str(sr),
                        "-ac", "1", str(b)], check=True, capture_output=True, timeout=120)
        subprocess.run([FFMPEG, "-y", "-v", "error", "-i", str(b), "-ar", "16000",
                        "-ac", "1", str(a)], check=True, capture_output=True, timeout=120)
        y, _ = sf.read(str(a), dtype="float32")
    return np.asarray(y, dtype=np.float32).ravel()


def add_noise(x: np.ndarray, spec: str, rng: np.random.Generator) -> np.ndarray:
    if spec == "none":
        return x
    kind, snr = ("pink", 25) if spec == "pink25" else ("white", int(spec[-2:]))
    n = rng.standard_normal(len(x)).astype(np.float64)
    if kind == "pink":
        f = np.fft.rfft(n)
        freq = np.fft.rfftfreq(len(n), 1 / 16000)
        f[1:] = f[1:] / np.sqrt(np.maximum(freq[1:], 1.0) / 1000.0)
        n = np.fft.irfft(f, len(n))
    sig_e = float((x.astype(np.float64) ** 2).mean() + 1e-12)
    n *= np.sqrt(sig_e / (float((n ** 2).mean() + 1e-12) * 10 ** (snr / 10)))
    return (x + n[:len(x)]).astype(np.float32)


def add_reverb(x: np.ndarray, spec: str, rng: np.random.Generator) -> np.ndarray:
    if spec == "none":
        return x
    rt60 = 0.2 if spec == "rt60_02" else 0.4
    L = int(rt60 * 16000)
    t = np.arange(L) / 16000.0
    ir = rng.standard_normal(L) * np.exp(-6.9 * t / rt60)
    ir[0] = 1.0
    y = np.convolve(x, ir, mode="full")[:len(x)]
    peak = float(np.abs(y).max())
    if peak > 0.99:
        y = y * (0.99 / peak)
    return y.astype(np.float32)


def apply_loss(x: np.ndarray, spec: str, rng: np.random.Generator) -> np.ndarray:
    if spec == "none":
        return x
    y = x.copy()
    for _ in range(1 if spec == "drop1" else 3):
        s = int(rng.integers(0, max(1, len(y) - 320)))
        y[s:s + 320] = 0.0
    return y


def apply_chain(x: np.ndarray, chain: Chain, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = x.astype(np.float32)
    if chain.codec != "none":
        y = ffmpeg_codec(y, chain.codec)
    y = resample_rt(y, chain.rate)
    y = add_noise(y, chain.noise, rng)
    y = add_reverb(y, chain.reverb, rng)
    y = np.clip(y * 10 ** (chain.gain_db / 20), -1.0, 1.0).astype(np.float32)
    y = apply_loss(y, chain.loss, rng)
    return y


@dataclass
class BatchItem:
    audio: np.ndarray
    label: int  # 0 real, 1 fake
    clip_id: str
    chain: Chain


def batch_stream(items: list[tuple[str, Path, int]], epoch: int,
                 sr_cache: dict | None = None) -> list[BatchItem]:
    """On-the-fly augmented batch. items = (clip_id, wav_path, label).
    Same sampler both classes (label-blind); seed = hash(clip_id, epoch)."""
    import soundfile as sf
    import zlib

    out: list[BatchItem] = []
    for clip_id, path, label in items:
        seed = zlib.crc32(f"{clip_id}@{epoch}".encode()) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)
        chain = sample_chain(rng)
        y, _ = sf.read(str(path), dtype="float32")
        y = np.asarray(y, dtype=np.float32).ravel()
        out.append(BatchItem(audio=apply_chain(y, chain, seed), label=label,
                             clip_id=clip_id, chain=chain))
    return out
