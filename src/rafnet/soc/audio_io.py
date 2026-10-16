"""Audio file decoding for the SOC upload path.

Supports what the real model can safely process: WAV/FLAC/OGG via soundfile,
MP3/M4A via torchaudio fallback. Enforces size/type limits, mono conversion,
and resampling info passthrough (actual resample happens in RAFNet's
AudioPreprocessor to 16 kHz — we preserve the original rate for metadata).
"""
from __future__ import annotations

import io
from dataclasses import dataclass

MAX_AUDIO_BYTES = 25 * 1024 * 1024  # 25 MB
ALLOWED_SUFFIXES = {".wav", ".flac", ".ogg", ".oga", ".mp3", ".m4a", ".mp4", ".webm"}
# Lossy containers/codecs: band-limiting and codec noise floor are intrinsic to
# the encoding, NOT evidence of replay. Layer 3 must exclude band-limit/codec
# trip-wires for these (see forensics.liveness_layer `source` handling).
LOSSY_SUFFIXES = {".mp3", ".m4a", ".mp4", ".ogg", ".oga", ".webm"}


@dataclass
class DecodedAudio:
    samples: object  # np.ndarray float32 mono
    sample_rate: int
    duration_seconds: float
    num_channels: int
    original_suffix: str
    is_lossy: bool = False  # True when the container/codec is lossy


def is_lossy_suffix(suffix: str) -> bool:
    """True for lossy containers/codecs whose spectra are band-limited by design."""
    s = suffix.lower()
    return (s if s.startswith(".") else "." + s) in LOSSY_SUFFIXES


def validate_filename(filename: str) -> str:
    suffix = "." + (filename.rsplit(".", 1)[-1].lower() if "." in filename else "")
    if suffix not in ALLOWED_SUFFIXES:
        raise ValueError(f"Unsupported audio format '{suffix}'. Allowed: {sorted(ALLOWED_SUFFIXES)}")
    # Strip path components for safe temp handling
    return filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]


def decode_audio(data: bytes, filename: str) -> DecodedAudio:
    import numpy as np

    if not data:
        raise ValueError("Empty audio file")
    if len(data) > MAX_AUDIO_BYTES:
        raise ValueError(f"Audio file too large ({len(data)} bytes > {MAX_AUDIO_BYTES})")
    safe_name = validate_filename(filename)

    # Primary: soundfile (wav/flac/ogg). Fallback: torchaudio for mp3/m4a.
    try:
        import soundfile as sf

        arr, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        arr = np.asarray(arr, dtype=np.float32)
        channels = 1 if arr.ndim == 1 else arr.shape[1]
        if arr.ndim == 2:
            arr = arr.mean(axis=1).astype(np.float32)
        suffix = safe_name.rsplit(".", 1)[-1].lower()
        return DecodedAudio(
            samples=arr,
            sample_rate=int(sr),
            duration_seconds=float(len(arr) / sr) if sr else 0.0,
            num_channels=int(channels),
            original_suffix=suffix,
            is_lossy=is_lossy_suffix(suffix),
        )
    except Exception as sf_err:
        # Minimal WAV parse fallback so .wav works even without soundfile.
        # (No temp files: decode fully in memory from the uploaded bytes.)
        if safe_name.lower().endswith(".wav"):
            import wave

            with wave.open(io.BytesIO(data), "rb") as w:
                n, sw, fr, nframes = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
                raw = w.readframes(nframes)
                dtype = {1: np.int8, 2: np.int16, 4: np.int32}[sw]
                arr = np.frombuffer(raw, dtype=dtype).astype(np.float32)
                scale = float(2 ** (8 * sw - 1))
                arr = (arr / scale).reshape(-1, n).mean(axis=1)
                return DecodedAudio(arr, int(fr), float(len(arr) / fr), int(n), "wav", False)
        raise ValueError(f"Could not decode audio '{safe_name}': {sf_err}")


def to_mono_16k(samples, sample_rate: int) -> tuple[object, int]:
    """Return (mono float32, sample_rate). Resampling to 16 kHz is performed
    downstream by RAFNet AudioPreprocessor; this helper only ensures mono."""
    import numpy as np

    arr = np.asarray(samples, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr.mean(axis=1).astype(np.float32)
    return arr, int(sample_rate)
