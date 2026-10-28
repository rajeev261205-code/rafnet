"""Heuristic forensic layers (numpy only — no torch/librosa so SOC-only mode works).

Layer 1 "Spectral & Vocoder Artifacts" (kind: heuristic) and Layer 3
"Liveness & Replay" (kind: heuristic) are computed here. Layer 2 (neural) is
assembled from the real RAFNet outputs without modification, and Layer 4
(enrollment) is reported by speaker.py as unavailable.

All scores are 0-100 heuristic likelihoods (or None when the signal is
insufficient); every trip-wire is reported as a measured value plus the
heuristic threshold that fired. Nothing here is a validated detector.
"""
from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from . import layer_config as cfg

TARGET_RATE = 16000
N_FFT = 1024
HOP = 256
PITCH_MIN_HZ = 50.0
PITCH_MAX_HZ = 500.0


def _layer(
    id: str,
    name: str,
    kind: str,
    score: float | None,
    evidence: list[str],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": id,
        "name": name,
        "kind": kind,
        "score": None if score is None else round(float(score), 1),
        "evidence": evidence,
        "metrics": metrics,
    }


def to_mono_16k(samples: Any, sample_rate: int) -> np.ndarray:
    """Mono float32 at 16 kHz — delegates to the shared preprocess module."""
    from .preprocess import to_model_input

    return to_model_input(samples, sample_rate)


def _stft_mag(x: np.ndarray, n_fft: int = N_FFT, hop: int = HOP) -> np.ndarray:
    """Magnitude spectrogram [frames x bins] of a 16 kHz mono clip."""
    if x.size < n_fft:
        pad = np.zeros(n_fft - x.size, dtype=np.float32)
        x = np.concatenate([x, pad])
    window = np.hanning(n_fft).astype(np.float32)
    frames = 1 + (x.size - n_fft) // hop
    mag = np.empty((frames, n_fft // 2 + 1), dtype=np.float32)
    for f in range(frames):
        seg = x[f * hop : f * hop + n_fft] * window
        mag[f] = np.abs(np.fft.rfft(seg)).astype(np.float32)
    return mag


def _ramp(value: float | None, lo: float, hi: float, invert: bool) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    t = (value - lo) / (hi - lo) if hi != lo else 0.0
    t = min(1.0, max(0.0, t))
    return (1.0 - t) * 100.0 if invert else t * 100.0


def _spectral_flux(mag: np.ndarray) -> float:
    """Mean frame-to-frame spectral change, normalized per frame.

    The per-frame denominator is floored at 1% of the clip's mean frame
    energy: without the floor, near-silent frames (pauses) divide by ~1e-9
    and a single pause boundary dominates the mean, so ANY speech with
    pauses reads as extreme "splicing". The floor keeps the quantity a
    discontinuity measure instead of a silence detector.
    """
    m = mag.astype(np.float64)
    frame_energy = m.sum(axis=1, keepdims=True)
    floor = 0.01 * float(frame_energy.mean())
    denom = np.maximum(frame_energy[:-1], floor) + 1e-12
    return float(np.mean(np.abs(np.diff(m, axis=0)) / denom))


def _local_jitter(f0: np.ndarray) -> float | None:
    """Cycle-to-cycle pitch variation (median relative successive difference).

    Computed over consecutive voiced frames in temporal order, so slow
    intonation drift across an utterance does NOT inflate it. The previous
    global std/mean formulation measured intonation range, not jitter, and
    flagged expressive natural speech. Returns None with <2 valid steps.
    """
    vals = np.asarray(f0, dtype=np.float64)
    steps: list[float] = []
    for a, b in zip(vals[:-1], vals[1:]):
        if np.isfinite(a) and np.isfinite(b) and a > 0 and b > 0:
            steps.append(abs(b - a) / b)
    if len(steps) < 2:
        return None
    return float(np.median(steps))


def _frame_hnr_db(
    x: np.ndarray,
    sr: int = TARGET_RATE,
    frame: int = 2048,
    hop: int = 512,
    max_frames: int = 240,
) -> float:
    """Median per-frame harmonic-to-noise ratio (dB, clipped to [-10, 40]).

    The previous whole-clip autocorrelation smeared vibrato and intonation
    across the clip, so clean sustained voicing scored ~1 dB ("noisy").
    Per-frame estimation over ~128 ms windows matches the F0 framing and
    measures harmonicity instead of stationarity. At most `max_frames`
    evenly-spaced frames are evaluated (deterministic latency bound on
    long clips).
    """
    n = int(x.size)
    if n < frame:
        return 0.0
    xc = x.astype(np.float64)
    lo_lag = max(2, int(sr / 500))
    hi_lag = min(frame - 1, int(sr / 50))
    starts = list(range(0, n - frame + 1, hop))
    if len(starts) > max_frames:
        idx = np.linspace(0, len(starts) - 1, num=max_frames).astype(int)
        starts = [starts[i] for i in idx]
    vals: list[float] = []
    for start in starts:
        seg = xc[start : start + frame]
        seg = seg - seg.mean()
        denom0 = float(seg @ seg)
        if denom0 < 1e-10:
            continue
        peak = 0.0
        for lag in range(lo_lag, hi_lag):
            cand = float(seg[: frame - lag] @ seg[lag:]) / denom0
            if cand > peak:
                peak = cand
        vals.append(40.0 if peak >= 1.0 else 10.0 * np.log10(peak / max(1e-6, 1.0 - peak)))
    if not vals:
        return 0.0
    return round(float(np.clip(np.median(vals), -10.0, 40.0)), 2)


def _f0_contour(x: np.ndarray, sr: int = TARGET_RATE) -> np.ndarray:
    """Per-frame F0 via autocorrelation peak picking (NaN when unvoiced/weak)."""
    n = 2048
    hop = 512
    min_lag = max(2, int(sr / PITCH_MAX_HZ))
    max_lag = min(n - 4, int(sr / PITCH_MIN_HZ))
    out: list[float] = []
    if x.size < n:
        return np.array([], dtype=np.float32)
    for start in range(0, x.size - n + 1, hop):
        frame = x[start : start + n].astype(np.float64)
        frame = frame - frame.mean()
        if np.dot(frame, frame) < 1e-10:
            out.append(np.nan)
            continue
        corr = np.correlate(frame, frame, mode="full")[n - 1 :]
        corr = corr / max(corr[0], 1e-12)
        band = corr[min_lag : max_lag + 1]
        peak = int(np.argmax(band))
        if band[peak] < 0.3:
            out.append(np.nan)
            continue
        lag = min_lag + peak
        out.append(sr / lag)
    return np.array(out, dtype=np.float32)


# -- Layer 1 ---------------------------------------------------------------

SPECTRAL_ID = "spectral"
SPECTRAL_NAME = "Spectral & Vocoder Artifacts"


def spectral_layer(samples: Any, sample_rate: int) -> dict[str, Any]:
    """Heuristic synthetic-likelihood from spectral/vocoder proxies."""
    x = to_mono_16k(samples, sample_rate)
    metrics: dict[str, Any] = {
        "samples": int(x.size),
        "sample_rate_hz": TARGET_RATE,
        "rms": float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) if x.size else 0.0,
    }
    if x.size == 0 or metrics["rms"] < cfg.MIN_SIGNAL_RMS:
        return _layer(
            SPECTRAL_ID, SPECTRAL_NAME, "heuristic", None,
            ["insufficient signal energy (silence) — spectral analysis unavailable (heuristic)"],
            metrics,
        )
    mag = _stft_mag(x)
    power = mag.astype(np.float64) ** 2 + 1e-12
    with np.errstate(divide="ignore", invalid="ignore"):
        flat = float(np.mean(np.exp(np.mean(np.log(power[:, 1:]), axis=1) - np.log(np.mean(power[:, 1:], axis=1)))))
    freqs = np.fft.rfftfreq(N_FFT, d=1.0 / TARGET_RATE)
    cumsum = np.cumsum(power, axis=1)
    total = cumsum[:, -1:]
    rolloff = freqs[np.argmax(cumsum >= 0.85 * total, axis=1)]
    metrics["flatness"] = round(float(np.nan_to_num(flat, nan=0.0, posinf=1.0, neginf=0.0)), 4)
    metrics["rolloff_hz"] = round(float(np.mean(rolloff)), 1)
    hb = power[:, freqs >= 4000].sum() / power.sum()
    metrics["highband_ratio"] = round(float(hb), 4)
    # Frame-median HNR: harmonicity, not whole-clip stationarity (see helper).
    hnr = _frame_hnr_db(x)
    metrics["hnr_db"] = hnr
    # Local jitter: cycle-to-cycle variation, robust to intonation (see helper).
    f0 = _f0_contour(x)
    jitter = _local_jitter(f0)
    voiced = f0[np.isfinite(f0)]
    metrics["jitter"] = round(jitter, 4) if jitter is not None else None
    metrics["voiced_frames"] = int(voiced.size)
    # Shimmer proxy: relative spread of frame energy.
    frame_rms = np.sqrt((mag.astype(np.float64) ** 2).mean(axis=1))
    shimmer = float(np.std(frame_rms) / max(1e-9, np.mean(frame_rms)))
    metrics["shimmer"] = round(shimmer, 4)
    # Spectral flux: frame-to-frame discontinuity (vocoder splicing proxy).
    # Floored per-frame normalization so pause boundaries in natural speech
    # don't dominate (see _spectral_flux).
    flux = _spectral_flux(mag)
    metrics["flux"] = round(flux, 4)

    values = {
        "flatness": metrics["flatness"],
        "highband_ratio": metrics["highband_ratio"],
        "hnr_db": metrics["hnr_db"],
        "jitter": metrics["jitter"],
        "flux": metrics["flux"],
    }
    total_w = 0.0
    acc = 0.0
    for key, spec in cfg.SPECTRAL_METRICS.items():
        sub = _ramp(values.get(key), spec["lo"], spec["hi"], spec["invert"])
        if sub is None:
            continue
        acc += sub * spec["weight"]
        total_w += spec["weight"]
    score = acc / total_w if total_w > 0 else None

    evidence: list[str] = []
    eth = cfg.SPECTRAL_EVIDENCE
    if metrics["flatness"] > eth["flatness"]:
        evidence.append(
            f"spectral flatness {metrics['flatness']} exceeds heuristic threshold {eth['flatness']} (heuristic)"
        )
    if metrics["highband_ratio"] > eth["highband_ratio"]:
        evidence.append(
            f"high-band energy ratio {metrics['highband_ratio']} exceeds heuristic threshold {eth['highband_ratio']} (heuristic)"
        )
    if metrics["hnr_db"] < eth["hnr_db"]:
        evidence.append(
            f"harmonic-to-noise ratio {metrics['hnr_db']} dB below heuristic threshold {eth['hnr_db']} dB (heuristic)"
        )
    if metrics["jitter"] is not None and metrics["jitter"] > eth["jitter"]:
        evidence.append(
            f"pitch jitter proxy {metrics['jitter']} exceeds heuristic threshold {eth['jitter']} (heuristic)"
        )
    if metrics["flux"] > eth["flux"]:
        evidence.append(
            f"spectral flux {metrics['flux']} exceeds heuristic threshold {eth['flux']} (heuristic)"
        )
    if not evidence:
        evidence.append("no spectral trip-wire fired (heuristic)")
    return _layer(SPECTRAL_ID, SPECTRAL_NAME, "heuristic", score, evidence, metrics)


# -- Input path -------------------------------------------------------------

def input_path_signal(input_path: dict[str, Any] | None) -> dict[str, Any]:
    """Flag virtual/loopback capture devices by label (real, labeled signal)."""
    label = (input_path or {}).get("device_label") if isinstance(input_path, dict) else None
    label = str(label).strip() if label else None
    if not label:
        return {"flagged": False, "keyword": None, "label": None, "note": "no capture device label provided"}
    lowered = label.lower()
    for keyword in cfg.VIRTUAL_INPUT_KEYWORDS:
        if keyword in lowered:
            return {
                "flagged": True,
                "keyword": keyword,
                "label": label,
                "note": "audio is routed from another application (heuristic input-path signal)",
            }
    return {"flagged": False, "keyword": None, "label": label, "note": "hardware microphone label"}


# -- Layer 3 ----------------------------------------------------------------

LIVENESS_ID = "liveness"
LIVENESS_NAME = "Liveness & Replay"


def _window_spectra(x: np.ndarray, window_samples: int = TARGET_RATE) -> list[np.ndarray]:
    """Mean log-magnitude spectrum per non-overlapping 1 s window."""
    out: list[np.ndarray] = []
    for start in range(0, x.size, window_samples):
        seg = x[start : start + window_samples]
        if seg.size < window_samples // 2:
            continue
        mag = _stft_mag(seg)
        out.append(np.log10(mag.mean(axis=0).astype(np.float64) + 1e-9))
    return out


def window_log_spectra(samples_16k: Any) -> list[np.ndarray]:
    """Public helper: mean log spectra of non-overlapping 1 s windows (16 kHz in)."""
    x = np.asarray(samples_16k, dtype=np.float32).ravel()
    return _window_spectra(x)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def liveness_layer(
    samples: Any,
    sample_rate: int,
    window_spectra: list[np.ndarray] | None = None,
    input_path: dict[str, Any] | None = None,
    source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Heuristic replay/pre-recorded likelihood + input-path signal.

    `source` describes the container/codec provenance for uploads, e.g.
    `{"codec": "mp3", "is_lossy": True, "sample_rate_hz": 44100}`. Lossy
    encodings band-limit spectra and set a codec noise floor BY DESIGN, so
    band-limit/codec evidence is excluded from the score for lossy input
    (it must never drive a REPLAY verdict on its own). Live-mic chunks pass
    `{"codec": "pcm", "is_lossy": False, ...}` and keep full analysis.
    """
    x = to_mono_16k(samples, sample_rate)
    duration = x.size / TARGET_RATE
    src = source if isinstance(source, dict) else {}
    codec = str(src.get("codec") or "").lower() or None
    is_lossy = bool(src.get("is_lossy", False))
    src_rate = src.get("sample_rate_hz", sample_rate)
    metrics: dict[str, Any] = {
        "samples": int(x.size),
        "duration_seconds": round(duration, 2),
        "rms": float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) if x.size else 0.0,
        "source_codec": codec,
        "codec": codec,
        "source_lossy": is_lossy,
        "lossy_input": is_lossy,
        "source_sample_rate_hz": int(src_rate) if src_rate else int(sample_rate),
    }
    if x.size == 0 or metrics["rms"] < cfg.MIN_SIGNAL_RMS:
        return _layer(
            LIVENESS_ID, LIVENESS_NAME, "heuristic", None,
            ["insufficient signal energy (silence) — liveness analysis unavailable (heuristic)"],
            metrics,
        )
    # Block-level energy (200 ms blocks).
    block = max(1, TARGET_RATE // 5)
    nblocks = max(1, x.size // block)
    rms_blocks = np.array(
        [np.sqrt(np.mean(x[i * block : (i + 1) * block].astype(np.float64) ** 2)) for i in range(nblocks)]
    )
    pause_fraction = float(np.mean(rms_blocks < 0.02))
    metrics["pause_fraction"] = round(pause_fraction, 4)
    # Noise floor is the BACKGROUND level: digital-black gaps (inserted
    # silence, TTS padding) carry no room information, so blocks below the
    # silence floor are excluded. Otherwise any clip with pauses reports a
    # -180 dB "floor" and trips the noise-floor wire on content, not noise.
    audible = rms_blocks[rms_blocks >= 0.005]
    floor_blocks = audible if audible.size else rms_blocks
    metrics["noise_floor_db"] = round(float(20.0 * np.log10(float(np.min(floor_blocks)) + 1e-9)), 1)
    metrics["clip_fraction"] = round(float(np.mean(np.abs(x) > 0.98)), 5)
    metrics["tail_ratio"] = round(float(np.min(rms_blocks) / (np.max(rms_blocks) + 1e-9)), 4)
    # Band-limiting: energy above 7 kHz.
    mag = _stft_mag(x)
    freqs = np.fft.rfftfreq(N_FFT, d=1.0 / TARGET_RATE)
    power = mag.astype(np.float64) ** 2
    hf = float(power[:, freqs > 7000].sum() / (power.sum() + 1e-12))
    metrics["highfreq_ratio"] = round(hf, 5)
    # Narrowband-stationary content gate: a sustained tone has ~zero energy
    # above 7 kHz and identical window spectra BY CONSTRUCTION, not because
    # of a codec cutoff or a replay loop. Without this gate a pure sine wave
    # scores REPLAY_PRERECORDED. When it fires, band-limit and loop abstain
    # (recorded below); pause/noise/clipping/tail evidence is unaffected.
    # The gate characterizes content (flat + static spectrum), it is not a
    # validated detector threshold — see LAYERS.md.
    mean_power = power.mean(axis=0)[1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        flat_tone = float(
            np.exp(np.mean(np.log(mean_power + 1e-12))) / (np.mean(mean_power) + 1e-12)
        )
    flux_tone = _spectral_flux(mag)
    metrics["tone_flatness"] = round(float(np.nan_to_num(flat_tone, nan=1.0)), 4)
    metrics["tone_flux"] = round(flux_tone, 4)
    # Peakiness separates tones from filtered noise: both are stationary with
    # steep spectral drop, but a tone concentrates energy in sparse peaks
    # (crest >> 10) while band-limited noise stays flat in-band (crest ~ 2-5).
    crest = float(mean_power.max() / (np.mean(mean_power) + 1e-12))
    metrics["tone_crest"] = round(crest, 1)
    tonal_stationary = bool(flat_tone < 0.02 and flux_tone < 0.05 and crest > 10.0)
    metrics["tonal_stationary"] = tonal_stationary
    metrics["tonal_excluded"] = tonal_stationary
    # Loop/repeat: max cosine similarity across non-overlapping 1 s windows.
    # Skipped for narrowband-stationary content (see gate above).
    spectra = [] if tonal_stationary else (
        window_spectra if window_spectra is not None else _window_spectra(x)
    )
    metrics["windows_compared"] = len(spectra)
    loop: float | None = None
    if len(spectra) >= 2:
        best = 0.0
        for i in range(len(spectra)):
            for j in range(i + 1, len(spectra)):
                best = max(best, _cosine(spectra[i], spectra[j]))
        loop = best
    metrics["loop_score"] = round(loop, 4) if loop is not None else None

    path = input_path_signal(input_path)
    metrics["input_path"] = path

    values = {
        "pause_fraction": metrics["pause_fraction"],
        "bandlimit": metrics["highfreq_ratio"],
        "loop": metrics["loop_score"],
        "noise_floor_db": metrics["noise_floor_db"],
        "clipping": metrics["clip_fraction"],
        "dry_tail": metrics["tail_ratio"],
    }
    # Pause analysis needs context; skip it on short clips.
    skipped: list[str] = []
    if duration < cfg.MIN_PAUSE_ANALYSIS_SECONDS:
        values["pause_fraction"] = None
        skipped.append(f"pause/breath analysis needs >= {cfg.MIN_PAUSE_ANALYSIS_SECONDS} s of audio")
    # Lossy encodings (mp3/aac/m4a/ogg) band-limit by design: a codec cutoff
    # is not replay evidence, so it never contributes to the score.
    # Narrowband-stationary tones abstain the same way (see gate above).
    bandlimit_excluded = False
    if is_lossy:
        values["bandlimit"] = None
        bandlimit_excluded = True
    if tonal_stationary:
        values["bandlimit"] = None
        values["loop"] = None
        bandlimit_excluded = True
    metrics["bandlimit_excluded"] = bandlimit_excluded
    total_w = 0.0
    acc = 0.0
    for key, spec in cfg.LIVENESS_METRICS.items():
        sub = _ramp(values.get(key), spec["lo"], spec["hi"], spec["invert"])
        if sub is None:
            continue
        acc += sub * spec["weight"]
        total_w += spec["weight"]
    score = acc / total_w if total_w > 0 else None

    evidence: list[str] = []
    eth = cfg.LIVENESS_EVIDENCE
    pause_fired = duration >= cfg.MIN_PAUSE_ANALYSIS_SECONDS and metrics["pause_fraction"] < eth["pause_fraction"]
    loop_fired = metrics["loop_score"] is not None and metrics["loop_score"] > eth["loop_score"]
    clip_fired = metrics["clip_fraction"] > eth["clip_fraction"]
    if pause_fired:
        evidence.append(
            f"absence of natural pauses/breaths (pause fraction {metrics['pause_fraction']} "
            f"below heuristic threshold {eth['pause_fraction']}) (heuristic)"
        )
    if metrics["highfreq_ratio"] < eth["highfreq_ratio"]:
        if tonal_stationary:
            evidence.append(
                "narrowband stationary tone: band-limit/loop metrics are "
                "content-determined, excluded from replay evidence (heuristic)"
            )
        elif bandlimit_excluded:
            evidence.append(
                f"lossy input ({codec or 'unknown codec'}): band-limit/codec cutoff excluded — "
                f"energy above 7 kHz {metrics['highfreq_ratio']} is expected from lossy encoding, "
                f"not replay evidence (heuristic)"
            )
        else:
            evidence.append(
                f"band-limited audio (energy above 7 kHz {metrics['highfreq_ratio']} "
                f"below heuristic threshold {eth['highfreq_ratio']}, possible codec cutoff) (heuristic)"
            )
    if loop_fired:
        evidence.append(
            f"repeated or looped segments across windows (similarity {metrics['loop_score']} "
            f"above heuristic threshold {eth['loop_score']}) (heuristic)"
        )
    if clip_fired:
        evidence.append(
            f"loudspeaker-style saturation (clipping fraction {metrics['clip_fraction']} "
            f"above heuristic threshold {eth['clip_fraction']}) (heuristic)"
        )
    if path["flagged"]:
        evidence.append(
            f"audio is routed from another application (input '{path['label']}' "
            f"matched '{path['keyword']}') (heuristic input-path signal)"
        )
    # True when a lossy upload shows ONLY codec/band-limit evidence: replay
    # cannot be assessed from it (fusion maps this to UNCERTAIN, never REPLAY).
    metrics["codec_only"] = bool(
        is_lossy and not pause_fired and not loop_fired and not clip_fired and not path["flagged"]
    )
    if not evidence:
        evidence.append("no liveness trip-wire fired (heuristic)")
    evidence.extend(f"note: {note}" for note in skipped)
    return _layer(LIVENESS_ID, LIVENESS_NAME, "heuristic", score, evidence, metrics)


# -- Layer 2 (neural, unchanged values) --------------------------------------

NEURAL_ID = "neural"
NEURAL_NAME = "Neural Detection (RAFNet)"


def neural_layer(result: Any) -> dict[str, Any]:
    """Pass through the real RAFNet outputs (kind: model). Values unchanged."""
    get = (lambda k: getattr(result, k, None)) if not isinstance(result, dict) else result.get
    pred = get("prediction")
    conf = get("confidence")
    prob_synth = get("prob_synthetic")
    metrics = {
        "prediction": pred,
        "confidence": conf,
        "prob_synthetic": prob_synth,
        "prob_real": get("prob_real"),
        "classical_probability": get("classical_probability"),
        "quantum_probability": get("quantum_probability"),
        "fused_probability": get("fused_probability"),
        "calibrated_probability": get("calibrated_probability"),
        "gateway1_audio_weight": get("gateway1_audio_weight"),
        "gateway1_network_weight": get("gateway1_network_weight"),
        "gateway2_classical_weight": get("gateway2_classical_weight"),
        "gateway2_quantum_weight": get("gateway2_quantum_weight"),
        "model_version": get("model_version"),
        "mock": bool(get("mock")),
    }
    score = round(float(prob_synth) * 100.0, 1) if prob_synth is not None else None
    evidence = [f"RAFNet prediction {pred} at model confidence {conf} (model)"]
    if metrics["mock"]:
        evidence.append("mock inference active — development only, not a real prediction (model)")
    return _layer(NEURAL_ID, NEURAL_NAME, "model", score, evidence, metrics)


# -- Verdict fusion ------------------------------------------------------------

def fuse_verdict(
    l1: dict[str, Any] | None,
    neural: dict[str, Any],
    l3: dict[str, Any] | None,
    risk: dict[str, Any],
    duration_seconds: float = 0.0,
) -> dict[str, Any]:
    """Transparent rule-based fusion (rules + defaults in layer_config.py)."""
    nm = neural.get("metrics", {})
    pred = nm.get("prediction")
    conf = float(nm.get("confidence") or 0.0)
    p_synth = nm.get("prob_synthetic")
    if p_synth is None:
        p_synth = conf if pred == "SYNTHETIC" else 1.0 - conf
    p_synth = float(p_synth)
    s1 = float(l1["score"]) if l1 and l1.get("score") is not None else None
    s3 = float(l3["score"]) if l3 and l3.get("score") is not None else None
    path = ((l3 or {}).get("metrics", {}) or {}).get("input_path", {}) or {}
    l3m = ((l3 or {}).get("metrics", {}) or {}) if isinstance((l3 or {}).get("metrics"), dict) else {}
    # Lossy upload (mp3/aac/m4a/ogg) whose ONLY replay evidence is the codec
    # cutoff: replay cannot be assessed from it — never REPLAY on that basis.
    # (Neural synthetic evidence and input-path routing are independent and
    # keep their own branches above/below.)
    lossy_codec_only = bool(l3m.get("lossy_input")) and bool(l3m.get("codec_only"))
    limitations = list(cfg.VERDICT_LIMITATIONS)
    if duration_seconds and duration_seconds < cfg.MIN_PAUSE_ANALYSIS_SECONDS:
        limitations.append(
            f"short audio ({duration_seconds:.2f} s): pause/breath analysis weak (heuristic)"
        )
    if l1 is not None and l1.get("score") is None:
        limitations.append("Layer 1 unavailable: insufficient signal energy (heuristic)")
    if l3 is not None and (l3.get("metrics", {}) or {}).get("loop_score") is None:
        limitations.append("loop analysis unavailable: needs >= 2 s of audio (heuristic)")

    l1_ev = (l1 or {}).get("evidence", [])
    l3_ev = (l3 or {}).get("evidence", [])

    if path.get("flagged"):
        strength = max(p_synth, (s1 or 0.0) / 100.0, (s3 or 0.0) / 100.0)
        confidence = round(min(0.85, 0.55 + 0.30 * strength), 4)
        reasons = [
            f"input path: audio is routed from another application ('{path.get('label')}') (heuristic)",
            f"RAFNet prediction {pred} at model confidence {conf:.4f} (model)",
        ]
        reasons += [f"layer 1: {e}" for e in l1_ev[:2]]
        reasons += [f"layer 3: {e}" for e in l3_ev[:2]]
        label = "EXTERNAL_APP_SUSPECTED"
    elif p_synth >= cfg.NEURAL_STRONG_PROB:
        label = "AI_SYNTHETIC"
        confidence = round(0.5 * p_synth + 0.5 * conf, 4)
        reasons = [
            f"RAFNet synthetic probability {p_synth:.4f} at/above {cfg.NEURAL_STRONG_PROB} (model)"
        ] + [f"layer 1: {e}" for e in l1_ev[:2]]
    elif pred == "SYNTHETIC" and conf >= cfg.NEURAL_SUPPORT_PROB and (s1 or 0.0) >= cfg.LAYER1_SUPPORT_SCORE:
        label = "AI_SYNTHETIC"
        confidence = round(0.5 * conf + 0.5 * (s1 or 0.0) / 100.0, 4)
        reasons = [
            f"RAFNet prediction SYNTHETIC at model confidence {conf:.4f} (model)",
            f"spectral heuristic agrees at {s1} (heuristic)",
        ] + [f"layer 1: {e}" for e in l1_ev[:2]]
    elif lossy_codec_only:
        label = "UNCERTAIN"
        confidence = round(conf, 4)
        codec = l3m.get("codec") or "lossy"
        reasons = [
            f"lossy file: replay cannot be assessed (band-limit/codec evidence excluded for {codec} input) (heuristic)",
            f"RAFNet prediction {pred} at model confidence {conf:.4f} (model)",
        ] + [f"layer 3: {e}" for e in l3_ev[:1]]
    elif (s3 or 0.0) >= cfg.LIVENESS_REPLAY_SCORE:
        label = "REPLAY_PRERECORDED"
        confidence = round((s3 or 0.0) / 100.0, 4)
        reasons = [f"liveness heuristic at {s3} (heuristic)"] + [f"layer 3: {e}" for e in l3_ev[:3]]
    elif pred == "REAL" and conf >= cfg.HUMAN_REAL_CONF and (s1 or 100.0) < cfg.HUMAN_MAX_LAYER_SCORE and (s3 or 100.0) < cfg.HUMAN_MAX_LAYER_SCORE:
        label = "HUMAN_LIVE"
        confidence = round(conf, 4)
        reasons = [
            f"RAFNet prediction REAL at model confidence {conf:.4f} (model)",
            "heuristic layers below suspicion thresholds (heuristic)",
        ]
    else:
        label = "UNCERTAIN"
        confidence = round(conf, 4)
        reasons = [
            f"layers disagree or evidence insufficient (RAFNet: {pred} at {conf:.4f}) (model+heuristic)"
        ]
        reasons += [f"layer 1: {e}" for e in l1_ev[:1]]
        reasons += [f"layer 3: {e}" for e in l3_ev[:1]]

    return {
        "label": label,
        "confidence": confidence,
        "risk_score": int(risk.get("risk_score", 0)),
        "risk_level": str(risk.get("risk_level", "LOW")),
        "reasons": reasons,
        "limitations": limitations,
    }


def update_session_verdict(
    labels: deque,
    current: str | None,
    new_label: str,
    window: int = cfg.HYSTERESIS_WINDOW,
    min_votes: int = cfg.HYSTERESIS_MIN_VOTES,
) -> str:
    """Rolling-window hysteresis: switch the session verdict only when the new
    label holds a majority of recent chunk votes (prevents flicker)."""
    labels.append(new_label)
    while len(labels) > window:
        labels.popleft()
    if current is None:
        return new_label
    if new_label == current:
        return current
    votes = sum(1 for v in labels if v == new_label)
    return new_label if votes >= min_votes else current
