"""Evidence-fusion configuration for detection layers + verdict.

IMPORTANT: every weight and threshold below is an engineering default chosen
without labeled replay/voice-conversion data. They are NOT validated detector
thresholds — see LAYERS.md (Phase 6) and the `limitations` field on every
verdict. Change them only with a measured evaluation (tools/evaluate_layers.py).

Score semantics:
- Layer 1 `spectral` (heuristic): 0-100 synthetic-likelihood from spectral/vocoder proxies.
- Layer 3 `liveness` (heuristic): 0-100 replay/pre-recorded-likelihood from liveness proxies.
- Layer 2 `neural` (model): the real RAFNet outputs, never re-scaled here.
- Layer 4 `enrollment` (unavailable): no speaker model installed; excluded from fusion.
"""
from __future__ import annotations

# -- Layer 1 sub-metric mapping -------------------------------------------
# Each entry maps a raw metric to 0-100 via a piecewise-linear ramp
# (lo -> 0, hi -> 100), clipped. `invert` flips the direction for metrics
# where LOW values are suspicious (e.g. HNR in dB).
SPECTRAL_METRICS: dict = {
    "flatness": {"lo": 0.05, "hi": 0.50, "weight": 0.25, "invert": False},
    "highband_ratio": {"lo": 0.05, "hi": 0.40, "weight": 0.20, "invert": False},
    "hnr_db": {"lo": 0.0, "hi": 20.0, "weight": 0.20, "invert": True},
    "jitter": {"lo": 0.01, "hi": 0.10, "weight": 0.15, "invert": False},
    "flux": {"lo": 0.10, "hi": 0.80, "weight": 0.20, "invert": False},
}

# Raw-metric evidence thresholds (heuristic trip-wires; reported with measured values).
SPECTRAL_EVIDENCE: dict = {
    "flatness": 0.30,
    "highband_ratio": 0.25,
    "hnr_db": 6.0,  # below = suspicious (inverted)
    "jitter": 0.06,
    "flux": 0.50,
}

# -- Layer 3 sub-metric mapping -------------------------------------------
LIVENESS_METRICS: dict = {
    "pause_fraction": {"lo": 0.0, "hi": 0.20, "weight": 0.20, "invert": True},
    "bandlimit": {"lo": 0.005, "hi": 0.030, "weight": 0.20, "invert": True},
    "loop": {"lo": 0.70, "hi": 0.95, "weight": 0.25, "invert": False},
    "noise_floor_db": {"lo": -90.0, "hi": -50.0, "weight": 0.15, "invert": True},
    "clipping": {"lo": 0.0, "hi": 0.05, "weight": 0.10, "invert": False},
    "dry_tail": {"lo": 0.02, "hi": 0.30, "weight": 0.10, "invert": True},
}

LIVENESS_EVIDENCE: dict = {
    "pause_fraction": 0.03,  # below = suspicious (over clips >= 2 s)
    "highfreq_ratio": 0.008,  # below = suspicious (band-limited/codec cutoff)
    "loop_score": 0.90,  # above = suspicious (repeated/looped segments)
    "clip_fraction": 0.01,  # above = suspicious (loudspeaker-style saturation)
}

MIN_PAUSE_ANALYSIS_SECONDS = 2.0  # pause/breath analysis needs context
MIN_SIGNAL_RMS = 1e-4  # below this the clip is treated as silence (scores None)

# -- Input-path (virtual/loopback device) keywords -------------------------
# Matched case-insensitively against the capture device label sent by the
# frontend. A match means audio is routed from another application, NOT a live
# microphone — a real, labeled signal, not a voice-authenticity judgment.
VIRTUAL_INPUT_KEYWORDS: tuple = (
    "virtual cable",
    "vb-audio",
    "voicemeeter",
    "stereo mix",
    "loopback",
    "blackhole",
    "soundflower",
    "obs",
    "cable output",
    "voicemod",
    "virtual",
)

# -- Verdict fusion rules (evaluated in order) ------------------------------
NEURAL_STRONG_PROB = 0.86  # prob_synthetic at/above this alone => AI_SYNTHETIC
NEURAL_SUPPORT_PROB = 0.60  # supporting threshold when Layer 1 agrees
LAYER1_SUPPORT_SCORE = 50.0
LIVENESS_REPLAY_SCORE = 60.0
HUMAN_REAL_CONF = 0.60
HUMAN_MAX_LAYER_SCORE = 50.0

# -- Live hysteresis (rolling window over per-chunk verdicts) ----------------
HYSTERESIS_WINDOW = 5  # votes considered
HYSTERESIS_MIN_VOTES = 3  # votes needed to switch the session verdict

# -- Static limitations attached to every verdict ---------------------------
VERDICT_LIMITATIONS: tuple = (
    "Rule-based fusion with unvalidated thresholds — not a certified detector.",
    "Replay and voice-conversion performance need labeled data to validate.",
    "Voice enrollment biometric unavailable: no speaker-embedding model installed.",
)
