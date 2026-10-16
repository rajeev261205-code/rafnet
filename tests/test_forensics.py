"""Unit tests for heuristic forensic layers + verdict fusion (synthetic signals).

These tests use generated tones/noise/silence to exercise the DSP paths —
they do NOT validate detection accuracy (that needs labeled data; see
tools/evaluate_layers.py in Phase 6).
"""
from collections import deque

import numpy as np
import pytest

from rafnet.soc import forensics as F
from rafnet.soc.speaker import enrollment_layer, speaker_capability

SR = 16000


def sine(freq: float, seconds: float, amp: float = 0.5) -> np.ndarray:
    n = int(SR * seconds)
    t = np.arange(n, dtype=np.float64) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def noise(seconds: float, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (0.4 * rng.standard_normal(int(SR * seconds))).astype(np.float32)


def neural(prediction: str, confidence: float, prob_synthetic: float) -> dict:
    return F.neural_layer(
        {
            "prediction": prediction,
            "confidence": confidence,
            "prob_synthetic": prob_synthetic,
            "prob_real": 1.0 - prob_synthetic,
            "classical_probability": prob_synthetic,
            "quantum_probability": prob_synthetic,
            "fused_probability": prob_synthetic,
            "calibrated_probability": prob_synthetic,
            "gateway1_audio_weight": 1.0,
            "gateway1_network_weight": 0.0,
            "gateway2_classical_weight": 0.5,
            "gateway2_quantum_weight": 0.5,
            "model_version": "test",
            "mock": False,
        }
    )


def test_spectral_layer_runs_on_tone():
    layer = F.spectral_layer(sine(440, 2.0), SR)
    assert layer["id"] == "spectral"
    assert layer["kind"] == "heuristic"
    assert layer["score"] is not None and 0.0 <= layer["score"] <= 100.0
    assert len(layer["evidence"]) > 0
    for key in ("flatness", "highband_ratio", "hnr_db", "flux", "rms"):
        assert key in layer["metrics"]


def test_silence_scores_none():
    for fn in (F.spectral_layer,):
        layer = fn(np.zeros(SR * 2, dtype=np.float32), SR)
        assert layer["score"] is None
        assert any("insufficient signal energy" in e for e in layer["evidence"])
    live = F.liveness_layer(np.zeros(SR * 2, dtype=np.float32), SR)
    assert live["score"] is None


def test_noise_flatter_than_tone():
    tone = F.spectral_layer(sine(440, 2.0), SR)
    hiss = F.spectral_layer(noise(2.0), SR)
    assert hiss["metrics"]["flatness"] > tone["metrics"]["flatness"]


def test_tone_abstains_from_bandlimit_and_loop():
    # A pure 440 Hz tone has no energy above 7 kHz and identical window
    # spectra BY CONSTRUCTION — calling that a codec cutoff or a replay loop
    # is a content artifact (regression: a sine wave scored REPLAY_PRERECORDED).
    layer = F.liveness_layer(sine(440, 3.0), SR)
    assert layer["metrics"]["highfreq_ratio"] < 0.008
    assert layer["metrics"]["tonal_stationary"] is True
    assert layer["metrics"]["loop_score"] is None
    assert not any("band-limited audio" in e for e in layer["evidence"])
    assert any("narrowband stationary tone" in e for e in layer["evidence"])
    assert layer["score"] is not None and layer["score"] < 60.0


def looped_burst(seconds_per: float = 1.0, reps: int = 2, seed: int = 21) -> np.ndarray:
    """Exact repeats of a NON-tonal segment (bandpassed hot noise).

    Repeating a pure tone proves nothing about loop detection — a continuous
    tone is already self-identical, so the tonal gate abstains there. A
    repeated broadband segment is genuine loop ground truth.
    """
    n = int(SR * seconds_per)
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / SR)
    mask = (freqs >= 800) & (freqs <= 3200)
    spec[~mask] = 0.0
    x = np.fft.irfft(spec, n)
    seg = np.clip(3.0 * x / np.abs(x).max(), -1.0, 1.0).astype(np.float32)
    return np.concatenate([seg] * reps)


def test_loop_detection_on_exact_repeat():
    looped = looped_burst(reps=2)
    layer = F.liveness_layer(looped, SR)
    assert layer["metrics"]["tonal_stationary"] is False
    assert layer["metrics"]["loop_score"] is not None
    assert layer["metrics"]["loop_score"] >= 0.99
    assert any("looped" in e for e in layer["evidence"])


def test_loop_absent_for_single_window():
    layer = F.liveness_layer(sine(440, 1.0), SR)
    assert layer["metrics"]["loop_score"] is None


def test_input_path_virtual_flagged():
    hit = F.input_path_signal({"device_label": "CABLE Output (VB-Audio Virtual Cable)"})
    assert hit["flagged"] is True
    assert hit["keyword"] == "virtual cable"
    assert "another application" in hit["note"]


def test_input_path_hardware_not_flagged():
    assert F.input_path_signal({"device_label": "MacBook Pro Microphone"})["flagged"] is False
    assert F.input_path_signal(None)["flagged"] is False
    assert F.input_path_signal({})["flagged"] is False


def test_neural_layer_passes_values_through():
    layer = neural("SYNTHETIC", 0.92, 0.92)
    assert layer["kind"] == "model"
    assert layer["score"] == pytest.approx(92.0)
    assert layer["metrics"]["classical_probability"] == pytest.approx(0.92)


def test_fuse_virtual_input_suspected():
    l1 = F.spectral_layer(sine(440, 1.0), SR)
    l3 = F.liveness_layer(
        sine(440, 1.0), SR, input_path={"device_label": "Voicemeeter Output"}
    )
    verdict = F.fuse_verdict(
        l1, neural("REAL", 0.9, 0.1), l3,
        {"risk_score": 20, "risk_level": "LOW"}, duration_seconds=1.0,
    )
    assert verdict["label"] == "EXTERNAL_APP_SUSPECTED"
    assert any("another application" in r for r in verdict["reasons"])
    assert verdict["risk_score"] == 20 and verdict["risk_level"] == "LOW"


def test_fuse_strong_synthetic():
    l1 = F.spectral_layer(noise(2.0, seed=3), SR)
    l3 = F.liveness_layer(noise(2.0, seed=3), SR)
    verdict = F.fuse_verdict(
        l1, neural("SYNTHETIC", 0.9, 0.9), l3,
        {"risk_score": 90, "risk_level": "CRITICAL"}, duration_seconds=3.0,
    )
    assert verdict["label"] == "AI_SYNTHETIC"
    assert verdict["confidence"] > 0.5


def test_fuse_uncertain_on_silence():
    silence = np.zeros(SR * 2, dtype=np.float32)
    l1 = F.spectral_layer(silence, SR)
    l3 = F.liveness_layer(silence, SR)
    verdict = F.fuse_verdict(
        l1, neural("UNCERTAIN", 0.5, 0.5), l3,
        {"risk_score": 40, "risk_level": "MEDIUM"}, duration_seconds=2.0,
    )
    assert verdict["label"] == "UNCERTAIN"
    assert len(verdict["limitations"]) >= 3


def test_fuse_short_audio_limitation():
    l1 = F.spectral_layer(sine(440, 1.0), SR)
    l3 = F.liveness_layer(sine(440, 1.0), SR)
    verdict = F.fuse_verdict(
        l1, neural("UNCERTAIN", 0.5, 0.5), l3,
        {"risk_score": 40, "risk_level": "MEDIUM"}, duration_seconds=1.0,
    )
    assert any("short audio" in lim for lim in verdict["limitations"])


def test_hysteresis_resists_flicker():
    labels: deque = deque(maxlen=5)
    current: str | None = None
    for vote in ["HUMAN_LIVE", "HUMAN_LIVE", "HUMAN_LIVE"]:
        current = F.update_session_verdict(labels, current, vote)
    assert current == "HUMAN_LIVE"
    # A single dissenting chunk must not flip the session verdict.
    assert F.update_session_verdict(labels, current, "AI_SYNTHETIC") == "HUMAN_LIVE"
    # Sustained dissent flips it.
    for _ in range(3):
        current = F.update_session_verdict(labels, current, "AI_SYNTHETIC")
    assert current == "AI_SYNTHETIC"


def test_enrollment_unavailable():
    layer = enrollment_layer()
    assert layer["id"] == "enrollment"
    assert layer["kind"] == "unavailable"
    assert layer["score"] is None
    assert layer["metrics"]["enrolled"] is False
    assert speaker_capability()["available"] is False


def lowpassed_noise_1s(cutoff_hz: float = 5000.0) -> np.ndarray:
    """1 s of noise with nothing above `cutoff_hz`.

    Short clip => pause analysis skipped and loop_score None, so the ONLY
    replay trip-wire that can fire is the band-limit/codec cutoff. A pure
    lowpassed signal must never become REPLAY just from that on lossy input.
    """
    n = int(SR * 1.0)
    rng = np.random.default_rng(11)
    x = rng.standard_normal(n)
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / SR)
    spec[freqs > cutoff_hz] = 0.0
    x = np.fft.irfft(spec, n=n)
    return (0.4 * x / max(1e-9, np.abs(x).max())).astype(np.float32)


def test_bandlimit_evidence_kept_for_lossless():
    layer = F.liveness_layer(lowpassed_noise_1s(), SR)
    assert layer["metrics"]["highfreq_ratio"] < 0.008
    assert any("band-limited audio" in e for e in layer["evidence"])
    assert layer["metrics"]["bandlimit_excluded"] is False
    assert layer["metrics"]["codec_only"] is False


def test_bandlimit_excluded_for_lossy_upload():
    sig = lowpassed_noise_1s()
    layer = F.liveness_layer(
        sig, SR,
        source={"codec": "mp3", "is_lossy": True, "sample_rate_hz": 44100},
    )
    assert layer["metrics"]["highfreq_ratio"] < 0.008
    assert layer["metrics"]["bandlimit_excluded"] is True
    assert layer["metrics"]["source_lossy"] is True
    assert layer["metrics"]["source_codec"] == "mp3"
    assert layer["metrics"]["source_sample_rate_hz"] == 44100
    assert not any("band-limited audio" in e for e in layer["evidence"])
    assert any("lossy input (mp3)" in e for e in layer["evidence"])
    # Only codec evidence exists -> replay cannot be assessed from this file.
    assert layer["metrics"]["codec_only"] is True


def test_lossy_codec_only_fuses_uncertain_not_replay():
    sig = lowpassed_noise_1s()
    l1 = F.spectral_layer(sig, SR)
    l3 = F.liveness_layer(
        sig, SR,
        source={"codec": "m4a", "is_lossy": True, "sample_rate_hz": 44100},
    )
    verdict = F.fuse_verdict(
        l1, neural("REAL", 0.9, 0.1), l3,
        {"risk_score": 10, "risk_level": "LOW"}, duration_seconds=1.0,
    )
    assert verdict["label"] == "UNCERTAIN"
    assert verdict["label"] != "REPLAY_PRERECORDED"
    assert any("lossy file: replay cannot be assessed" in r for r in verdict["reasons"])


def test_lossy_genuine_loop_still_replays():
    # Real repetition + saturation evidence must still drive REPLAY on lossy.
    # (Non-tonal repeats: a clipped sine repeat would trip the stationary
    # gate, which is correct — it is content-identical with or without loss.)
    looped = looped_burst(reps=4)
    l1 = F.spectral_layer(looped, SR)
    l3 = F.liveness_layer(
        looped, SR,
        source={"codec": "ogg", "is_lossy": True, "sample_rate_hz": 48000},
    )
    assert l3["metrics"]["codec_only"] is False
    verdict = F.fuse_verdict(
        l1, neural("REAL", 0.7, 0.3), l3,
        {"risk_score": 50, "risk_level": "MEDIUM"}, duration_seconds=4.0,
    )
    assert verdict["label"] == "REPLAY_PRERECORDED"


def test_is_lossy_suffix_mapping():
    from rafnet.soc.audio_io import is_lossy_suffix

    for suffix in ("mp3", ".mp3", "m4a", ".m4a", "mp4", "ogg", "oga", "webm"):
        assert is_lossy_suffix(suffix) is True
    for suffix in ("wav", ".wav", "flac", ".flac", "", "txt"):
        assert is_lossy_suffix(suffix) is False


def harmonic_stack(freq: float, seconds: float, seed: int = 0) -> np.ndarray:
    """Voiced-like harmonic complex (steady pitch, no FM/AM)."""
    n = int(SR * seconds)
    t = np.arange(n, dtype=np.float64) / SR
    rng = np.random.default_rng(seed)
    x = sum(
        (1.0 / h) * np.sin(2 * np.pi * freq * h * t + rng.uniform(0, 2 * np.pi))
        for h in range(1, 9)
    )
    return (0.3 * x / np.abs(x).max()).astype(np.float32)


def test_flux_sane_on_phrase_with_gaps_and_pitch_change():
    # Two steady phrases at different pitches separated by digital silence.
    # Regression: pause boundaries divided by ~1e-9, so flux read 715392 on
    # such input and every natural utterance with pauses looked "spliced".
    sig = np.concatenate([
        harmonic_stack(440, 0.8, seed=1),
        np.zeros(int(SR * 0.2), dtype=np.float32),
        harmonic_stack(554, 0.8, seed=2),
    ])
    layer = F.spectral_layer(sig, SR)
    assert layer["metrics"]["flux"] < 0.05
    assert not any("spectral flux" in e for e in layer["evidence"])


def test_jitter_local_ignores_intonation():
    # Same phrase signal: global pitch spread is wide (440 -> 554 Hz) but
    # cycle-to-cycle variation is ~0. The old global std/mean read 0.11 here.
    sig = np.concatenate([
        harmonic_stack(440, 0.8, seed=1),
        np.zeros(int(SR * 0.2), dtype=np.float32),
        harmonic_stack(554, 0.8, seed=2),
    ])
    layer = F.spectral_layer(sig, SR)
    assert layer["metrics"]["jitter"] is not None
    assert layer["metrics"]["jitter"] < 0.06
    assert not any("pitch jitter" in e for e in layer["evidence"])


def test_frame_hnr_high_on_steady_voicing_low_on_noise():
    voiced = F.spectral_layer(harmonic_stack(120, 2.0, seed=5), SR)
    assert voiced["metrics"]["hnr_db"] >= 6.0
    hiss = F.spectral_layer(noise(2.0), SR)
    assert hiss["metrics"]["hnr_db"] < 6.0


def test_noise_floor_ignores_digital_silence_gaps():
    # Same vowel with and without inserted digital-black gaps must report
    # the same background floor (regression: gaps forced -180 dB).
    vowel = harmonic_stack(120, 2.0, seed=5)
    gaps = np.concatenate([
        vowel[: SR], np.zeros(SR // 2, dtype=np.float32), vowel[SR:],
    ])
    a = F.liveness_layer(vowel, SR)["metrics"]["noise_floor_db"]
    b = F.liveness_layer(gaps, SR)["metrics"]["noise_floor_db"]
    assert abs(a - b) < 6.0, f"{a} vs {b}"
