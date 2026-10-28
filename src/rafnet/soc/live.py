"""Chunked live ingestion — honest mic-to-model path (no fake streaming).

Architecture (documented, not pretending continuous inference):
- Frontend captures mic audio, resamples to 16 kHz mono, POSTs small float
  chunks (`POST /api/live/chunk`, typically 0.25–1.0 s each).
- Server appends each chunk to a per-session rolling buffer (TTL + caps).
- When the buffer holds >= 1 full analysis window (16000 samples = 1.0 s),
  the server runs the SAME `VoiceDetectionModel.predict` used by /api/analyze
  on the latest window and emits real WS events (detection chunk). Shorter
  buffers return `buffering` without inference — never a fabricated verdict.

Sessions are process-local (like the analysis store); Supabase persistence for
live chunks is future work. Caps: 50 sessions, 30 s audio each, 10 min TTL.
"""
from __future__ import annotations

import logging
import math
import time
import uuid
from collections import deque
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

WINDOW_SAMPLES = 16000  # 1.0 s @ 16 kHz — same window as the file pipeline
MAX_SESSIONS = 50
MAX_BUFFER_SAMPLES = 16000 * 30  # 30 s cap per session
SESSION_TTL_SECONDS = 600.0
MAX_CHUNK_SAMPLES = 48000 * 2  # 2 s @ 48 kHz upper bound per chunk


@dataclass
class LiveSession:
    id: str
    # Owning principal: Supabase user id, or None for an explicitly isolated
    # anonymous dev session. Never inferred from a client-supplied id.
    owner_id: str | None = None
    sample_rate: int = 16000
    created_at: float = field(default_factory=time.monotonic)
    last_seen: float = field(default_factory=time.monotonic)
    chunks: int = 0
    analyses: int = 0
    total_samples: int = 0
    _buf: deque = field(default_factory=deque)
    # Rolling forensic state (process-local, additive to the chunk contract):
    # per-window mean log spectra with sample offsets (non-overlap loop check),
    # recent chunk verdict labels (hysteresis votes), and the session verdict.
    spec_history: deque = field(default_factory=lambda: deque(maxlen=16))
    verdict_labels: deque = field(default_factory=lambda: deque(maxlen=5))
    session_verdict: str | None = None
    session_reasons: list = field(default_factory=list)
    # Rolling neural aggregation (JOB2.5): per-chunk P(synthetic) history for a
    # median-of-last-N decision with hysteresis. Window 8 keeps the last ~8 s;
    # the evaluated window is 5 (see tools/evaluate_model.py).
    neural_probs: deque = field(default_factory=lambda: deque(maxlen=8))
    neural_labels: deque = field(default_factory=lambda: deque(maxlen=8))
    agg_prediction: str | None = None
    # Rolling ensemble P(AI) for the CRITICAL gate (Phase 5): calibrated
    # stacked scores, last 5 analyzed windows (~5 s). CRITICAL needs this
    # window full and hot; a single hot chunk can never do it alone.
    ensemble_p_history: deque = field(default_factory=lambda: deque(maxlen=5))
    # Rolling origin votes (5-class labels from the 4 s origin window).
    origin_labels: deque = field(default_factory=lambda: deque(maxlen=5))
    session_origin: str | None = None

    def append(self, samples: list[float]) -> int:
        self._buf.extend(samples)
        self.total_samples += len(samples)
        while len(self._buf) > MAX_BUFFER_SAMPLES:
            remove = len(self._buf) - MAX_BUFFER_SAMPLES
            for _ in range(remove):
                self._buf.popleft()
        self.last_seen = time.monotonic()
        self.chunks += 1
        return len(self._buf)

    def latest_window(self) -> list[float] | None:
        if len(self._buf) < WINDOW_SAMPLES:
            return None
        return list(self._buf)[-WINDOW_SAMPLES:]

    @property
    def buffered_seconds(self) -> float:
        return len(self._buf) / 16000.0


class LiveSessionRegistry:
    def __init__(self) -> None:
        self._sessions: dict[str, LiveSession] = {}

    def create(self, owner_id: str | None = None) -> LiveSession:
        self._evict()
        if len(self._sessions) >= MAX_SESSIONS:
            oldest = min(self._sessions.values(), key=lambda s: s.last_seen)
            del self._sessions[oldest.id]
        session = LiveSession(
            id=f"live-{uuid.uuid4().hex[:12]}", owner_id=owner_id
        )
        self._sessions[session.id] = session
        return session

    def get(self, session_id: str) -> LiveSession | None:
        session = self._sessions.get(session_id)
        if session is None:
            return None
        if time.monotonic() - session.last_seen > SESSION_TTL_SECONDS:
            del self._sessions[session_id]
            return None
        return session

    def get_for(self, session_id: str, user_id: str | None) -> LiveSession | None:
        """Return a live session only when it belongs to `user_id`.

        `user_id=None` is the anonymous principal and matches only anonymous
        sessions. Callers must use this (not `get`) for any client-supplied
        session id.
        """
        session = self.get(session_id)
        if session is None or session.owner_id != user_id:
            return None
        return session

    def _evict(self) -> None:
        now = time.monotonic()
        expired = [sid for sid, s in self._sessions.items() if now - s.last_seen > SESSION_TTL_SECONDS]
        for sid in expired:
            del self._sessions[sid]


def validate_chunk(sample_rate: int, samples: list) -> list[float]:
    if sample_rate not in range(8000, 48001):
        raise ValueError(f"sample_rate {sample_rate} out of supported range 8000–48000")
    if not isinstance(samples, list) or not samples:
        raise ValueError("samples must be a non-empty float array")
    if len(samples) > MAX_CHUNK_SAMPLES:
        raise ValueError(f"chunk too large ({len(samples)} > {MAX_CHUNK_SAMPLES} samples)")
    clean: list[float] = []
    for value in samples:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("samples must be finite numbers")
        # No pre-clip here: the single clip happens AFTER resampling inside
        # the shared to_model_input() contract (mono -> 16 kHz -> clip).
        # Pre-clipping would reshape loud waveforms before interpolation and
        # diverge from the file path by up to 0.02 (measured). Range is
        # bounded downstream by to_model_input().
        clean.append(number)
    return clean


def resample_to_16k(samples: list[float], sample_rate: int) -> list[float]:
    """Chunk-path resample — identical math to the file-upload path.

    Delegates to the shared `to_model_input` contract (mono -> 16 kHz ->
    clip), list in/out for API compat. Previously this called only
    `resample()` (no mono mix, clip-before-resample), diverging from files
    by up to 0.02 on loud audio and 0.35 on stereo. No normalization and
    no trimming here: the 1 s window is selected by the session buffer.
    """
    import numpy as np

    from .preprocess import to_model_input

    return to_model_input(np.asarray(samples, dtype=np.float32), sample_rate).tolist()


# -- Rolling neural aggregation (median of last N + hysteresis) --------------
AGG_WINDOW = 5  # evaluated window: median of the last 5 analyzed chunks
AGG_MIN_VOTES = 3  # hysteresis: need 3/5 votes to switch the aggregated label


def label_for_p(p: float, tau_low: float, tau_high: float) -> str:
    if p < tau_low:
        return "REAL"
    if p > tau_high:
        return "SYNTHETIC"
    return "UNCERTAIN"


def aggregate_probs(
    probs: deque | list[float],
    tau_low: float,
    tau_high: float,
    window: int = AGG_WINDOW,
) -> dict:
    """Median-of-last-`window` P(synthetic) with hysteresis bookkeeping.

    Returns {p_median, label, chunks_used, window}. Hysteresis itself lives in
    `update_aggregated_prediction` (majority vote to switch), mirroring the
    forensic session verdict so single-chunk flicker never flips the display.
    """
    import numpy as np

    recent = list(probs)[-window:]
    if not recent:
        return {"p_median": None, "label": "UNCERTAIN", "chunks_used": 0, "window": window}
    median = float(np.median(np.asarray(recent, dtype=np.float64)))
    return {
        "p_median": round(median, 4),
        "label": label_for_p(median, tau_low, tau_high),
        "chunks_used": len(recent),
        "window": window,
    }


def update_aggregated_prediction(
    labels: deque,
    current: str | None,
    new_label: str,
    window: int = AGG_WINDOW,
    min_votes: int = AGG_MIN_VOTES,
) -> str:
    """Hysteresis for the aggregated neural label: switch only on majority."""
    labels.append(new_label)
    while len(labels) > window:
        labels.popleft()
    if current is None:
        return new_label
    if new_label == current:
        return current
    votes = sum(1 for v in labels if v == new_label)
    return new_label if votes >= min_votes else current


_registry: LiveSessionRegistry | None = None


def get_live_registry() -> LiveSessionRegistry:
    global _registry
    if _registry is None:
        _registry = LiveSessionRegistry()
    return _registry
