"""Train-only augmentation (origin labels preserved, never used at inference).

Every function maps 16 kHz mono float32 -> 16 kHz mono float32 and returns
the applied-op names for logging. Codecs that need unavailable encoders
(AMR, AAC, Opus) are SKIPPED with a note — never silently approximated.
MP3 round-trips only when soundfile/libsndfile supports it.

Intended use: training pipeline reads data/manifest.csv, applies a random
subset per clip (seeded), and keeps the origin label. See notebooks/ for the
full pipeline and DIAGNOSIS.md §6 for why this exists.
"""
from __future__ import annotations

import io
from typing import Any

import numpy as np

SR = 16000


def _rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def add_noise(x: np.ndarray, snr_db: float = 15.0, seed: int | None = None) -> np.ndarray:
    rng = _rng(seed)
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) or 1e-4
    noise = rng.normal(0.0, rms / (10.0 ** (snr_db / 20.0)), size=x.shape)
    return np.clip(x + noise, -1.0, 1.0).astype(np.float32)


def apply_gain(x: np.ndarray, db: float, seed: int | None = None) -> np.ndarray:
    del seed
    return np.clip(x * (10.0 ** (db / 20.0)), -1.0, 1.0).astype(np.float32)


def apply_clipping(x: np.ndarray, threshold: float = 0.7, seed: int | None = None) -> np.ndarray:
    del seed
    return np.clip(x, -abs(threshold), abs(threshold)).astype(np.float32)


def apply_speed(x: np.ndarray, factor: float = 1.0, seed: int | None = None) -> np.ndarray:
    """Tempo change in [0.9, 1.1] via resampling (keeps 16 kHz rate)."""
    del seed
    if abs(factor - 1.0) < 1e-9 or x.size == 0:
        return x
    n = max(1, int(round(len(x) / factor)))
    old = np.linspace(0.0, 1.0, num=len(x))
    new = np.linspace(0.0, 1.0, num=n)
    return np.interp(new, old, x).astype(np.float32)


def apply_eq(x: np.ndarray, low_db: float = 0.0, high_db: float = 0.0,
             seed: int | None = None) -> np.ndarray:
    """Mic/phone EQ: shelving below 300 Hz and above 3400 Hz (FFT)."""
    del seed
    if x.size == 0 or (low_db == 0.0 and high_db == 0.0):
        return x
    spec = np.fft.rfft(x.astype(np.float64))
    freqs = np.fft.rfftfreq(x.size, 1.0 / SR)
    gain = np.ones_like(freqs)
    gain[freqs < 300] *= 10.0 ** (low_db / 20.0)
    gain[freqs > 3400] *= 10.0 ** (high_db / 20.0)
    return np.fft.irfft(spec * gain, n=x.size).astype(np.float32)


def apply_rir(x: np.ndarray, rt60: float = 0.4, seed: int | None = None) -> np.ndarray:
    """SIMULATED room reverb: exponential-decay noise IR (labelled simulated;
    real RIRs replace this when available)."""
    rng = _rng(seed)
    n_ir = min(int(rt60 * SR), x.size)
    t = np.arange(n_ir) / SR
    ir = rng.normal(0.0, np.exp(-6.9 * t / max(rt60, 1e-3))).astype(np.float64)
    ir /= max(1e-9, np.abs(ir).max())
    wet = np.convolve(x.astype(np.float64), ir)[: len(x)]
    dry = x.astype(np.float64)
    out = 0.75 * dry + 0.25 * wet / max(1e-9, np.abs(wet).max())
    return (out / max(1e-9, np.abs(out).max()) * 0.9).astype(np.float32)


def apply_packet_loss(x: np.ndarray, loss_rate: float = 0.05,
                      seed: int | None = None) -> np.ndarray:
    """Zero random 20 ms frames (packet-loss concealment stress)."""
    rng = _rng(seed)
    y = x.copy()
    frame = SR // 50
    for s in range(0, len(y), frame):
        if rng.random() < loss_rate:
            y[s: s + frame] = 0.0
    return y


def apply_telephony_g711(x: np.ndarray, seed: int | None = None) -> np.ndarray:
    """8 kHz telephony + mu-law companding simulation (G.711-style).

    Anti-alias lowpass at 3.4 kHz BEFORE downsampling (real telephony
    filters first; naive decimation would alias high frequencies down and
    misrepresent the channel).
    """
    del seed
    if x.size == 0:
        return x
    spec = np.fft.rfft(x.astype(np.float64))
    freqs = np.fft.rfftfreq(x.size, 1.0 / SR)
    spec[freqs > 3400.0] = 0.0
    x = np.fft.irfft(spec, n=x.size)
    # downsample to 8 kHz, mu-law round-trip, back to 16 kHz
    idx8 = np.linspace(0.0, 1.0, num=max(1, len(x) // 2))
    idx16 = np.linspace(0.0, 1.0, num=len(x))
    x8 = np.interp(idx8, idx16, x)
    mu = 255.0
    comp = np.sign(x8) * np.log1p(mu * np.abs(x8)) / np.log1p(mu)
    quant = np.round(comp * 127.0) / 127.0
    exp = np.sign(quant) * (1.0 / mu) * ((1.0 + mu) ** np.abs(quant) - 1.0)
    return np.interp(idx16, idx8, exp).astype(np.float32)


def apply_mp3(x: np.ndarray, bitrate: str = "64k", seed: int | None = None) -> tuple[np.ndarray, bool]:
    """MP3 round-trip when libsndfile supports it. Returns (audio, applied)."""
    del seed, bitrate
    try:
        import soundfile as sf

        buf = io.BytesIO()
        sf.write(buf, x, SR, format="MP3")
        buf.seek(0)
        back, sr = sf.read(buf, dtype="float32", always_2d=False)
        back = np.asarray(back, dtype=np.float32).ravel()
        if back.size and sr == SR:
            return np.clip(back, -1.0, 1.0).astype(np.float32), True
    except Exception:
        pass
    return x, False


OPS = ("noise", "gain", "clip", "speed", "eq", "rir", "loss", "g711", "mp3")


def augment(x: np.ndarray, ops: tuple[str, ...] | list[str] | None = None,
            seed: int = 0) -> tuple[np.ndarray, list[str]]:
    """Apply a random subset of ops (seeded). Returns (audio, applied_names).
    Unavailable codecs (aac/opus/amr) are never approximated — absent by design."""
    import random

    rng = random.Random(seed)
    chosen = list(ops) if ops else [o for o in OPS if rng.random() < 0.35] or ["gain"]
    y = np.asarray(x, dtype=np.float32)
    applied: list[str] = []
    g = _rng(seed)
    for op in chosen:
        if op == "noise":
            y = add_noise(y, snr_db=float(g.uniform(8, 25)), seed=seed)
        elif op == "gain":
            y = apply_gain(y, db=float(g.uniform(-12, 6)))
        elif op == "clip":
            y = apply_clipping(y, threshold=float(g.uniform(0.4, 0.9)))
        elif op == "speed":
            y = apply_speed(y, factor=float(g.uniform(0.9, 1.1)))
        elif op == "eq":
            y = apply_eq(y, low_db=float(g.uniform(-6, 6)), high_db=float(g.uniform(-8, 2)))
        elif op == "rir":
            y = apply_rir(y, rt60=float(g.uniform(0.2, 0.7)), seed=seed)
        elif op == "loss":
            y = apply_packet_loss(y, loss_rate=float(g.uniform(0.02, 0.1)), seed=seed)
        elif op == "g711":
            y = apply_telephony_g711(y)
        elif op == "mp3":
            y, ok = apply_mp3(y)
            if not ok:
                continue  # encoder unavailable: skip, do not fake
        else:
            continue
        applied.append(op)
    return y, applied


def describe() -> dict[str, Any]:
    return {
        "ops": list(OPS),
        "skipped_codecs": ["aac", "opus", "amr-nb", "amr-wb"],
        "skip_reason": "no encoder available in this runtime; never approximated",
        "rir": "simulated exponential-decay IR (labelled simulated)",
        "note": "train-only; inference uses soc/preprocess.py with no augmentation",
    }
