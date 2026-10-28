"""Shared audio preprocessing — ONE implementation for every path.

`to_model_input` converts arbitrary input (any rate, mono/stereo, any
amplitude) to the model contract: 16 kHz mono float32 in [-1, 1]. File
uploads, live chunks, training windows, and augmentation all go through
this module, so train/inference skew cannot hide in duplicated resamplers.

numpy-only (safe for the SOC-only runtime without torch).
"""
from __future__ import annotations

from typing import Any

import numpy as np

TARGET_RATE = 16000


def to_mono(samples: Any) -> np.ndarray:
    """Average channels to mono float32 (no resampling)."""
    arr = np.asarray(samples, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr.mean(axis=1).astype(np.float32)
    return arr.ravel().astype(np.float32)


def resample(samples: np.ndarray, sample_rate: int, target_rate: int = TARGET_RATE) -> np.ndarray:
    """Linear-interpolation resample (genuine, not a label change)."""
    arr = np.asarray(samples, dtype=np.float32).ravel()
    if int(sample_rate) == int(target_rate) or arr.size == 0:
        return arr
    duration = arr.size / float(sample_rate)
    target_len = max(1, int(round(duration * target_rate)))
    old_idx = np.linspace(0.0, 1.0, num=arr.size, dtype=np.float64)
    new_idx = np.linspace(0.0, 1.0, num=target_len, dtype=np.float64)
    return np.interp(new_idx, old_idx, arr).astype(np.float32)


def to_model_input(samples: Any, sample_rate: int) -> np.ndarray:
    """Full contract: mono -> 16 kHz -> clipped float32. No normalization:
    the model was trained without amplitude normalization, so none is
    applied (normalizing here would be a train/inference mismatch)."""
    mono = to_mono(samples)
    out = resample(mono, sample_rate, TARGET_RATE)
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def fixed_window(audio_16k: np.ndarray, seconds: float = 4.0, pad: bool = True) -> np.ndarray:
    """First `seconds` of 16 kHz mono (truncate or zero-pad). Training and
    windowed evaluation cut 4 s windows with this exact function."""
    want = int(seconds * TARGET_RATE)
    arr = np.asarray(audio_16k, dtype=np.float32).ravel()
    if len(arr) >= want:
        return arr[:want]
    if not pad:
        return arr
    return np.concatenate([arr, np.zeros(want - len(arr), dtype=np.float32)])
