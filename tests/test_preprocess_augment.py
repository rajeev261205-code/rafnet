"""Tests for shared preprocessing + train-only augmentation (synthetic signals)."""
import numpy as np

from rafnet.soc import augment as A
from rafnet.soc import preprocess as P


def _sine(sr: int, seconds: float, freq: float = 440.0) -> np.ndarray:
    n = int(sr * seconds)
    t = np.arange(n) / sr
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_to_model_input_contract():
    stereo44 = np.stack([_sine(44100, 1.0), _sine(44100, 1.0)]).T
    out = P.to_model_input(stereo44, 44100)
    assert out.ndim == 1 and len(out) == 16000
    assert out.dtype == np.float32
    assert float(np.abs(out).max()) <= 1.0


def test_paths_agree_after_refactor():
    from rafnet.soc import forensics, live

    mono8 = _sine(8000, 1.0).tolist()
    a = np.asarray(live.resample_to_16k(mono8, 8000), dtype=np.float32)
    b = forensics.to_mono_16k(np.asarray(mono8, dtype=np.float32), 8000)
    assert len(a) == 16000 == len(b)
    assert np.array_equal(a, b), "live and file paths must share one resampler"


def test_fixed_window_truncate_pad():
    x = np.ones(16000 * 6, dtype=np.float32)
    assert len(P.fixed_window(x, 4.0)) == 64000
    short = np.ones(8000, dtype=np.float32)
    w = P.fixed_window(short, 4.0)
    assert len(w) == 64000 and float(w[8000:].sum()) == 0.0


def test_augment_preserves_rate_and_label_shape():
    x = _sine(16000, 2.0)
    for op in A.OPS:
        y, applied = A.augment(x, ops=[op], seed=3)
        assert y.dtype == np.float32
        assert applied == [op] or (op == "mp3" and applied == [])
        if op != "speed":
            assert len(y) == len(x), op


def test_augment_deterministic():
    x = _sine(16000, 1.0)
    y1, a1 = A.augment(x, seed=11)
    y2, a2 = A.augment(x, seed=11)
    assert a1 == a2 and np.array_equal(y1, y2)


def test_augment_speed_bounds():
    x = _sine(16000, 2.0)
    fast, _ = A.augment(x, ops=["speed"], seed=1)
    # speed is random in [0.9, 1.1]: length within 1/1.1..1/0.9 of input
    assert 28000 < len(fast) < 36000


def test_g711_downsamples_content():
    x = _sine(16000, 1.0, freq=6000.0)
    y = A.apply_telephony_g711(x)
    assert len(y) == len(x)
    # 6 kHz tone cannot survive 8 kHz telephony: energy must collapse
    assert float(np.sqrt(np.mean(y.astype(np.float64) ** 2))) < 0.05


def test_describe_documents_skips():
    d = A.describe()
    assert "amr-nb" in d["skipped_codecs"] and "train-only" in d["note"]
