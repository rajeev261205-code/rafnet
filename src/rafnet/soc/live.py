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
    sample_rate: int = 16000
    created_at: float = field(default_factory=time.monotonic)
    last_seen: float = field(default_factory=time.monotonic)
    chunks: int = 0
    analyses: int = 0
    _buf: deque = field(default_factory=deque)

    def append(self, samples: list[float]) -> int:
        self._buf.extend(samples)
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

    def create(self) -> LiveSession:
        self._evict()
        if len(self._sessions) >= MAX_SESSIONS:
            oldest = min(self._sessions.values(), key=lambda s: s.last_seen)
            del self._sessions[oldest.id]
        session = LiveSession(id=f"live-{uuid.uuid4().hex[:12]}")
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
        clean.append(max(-1.0, min(1.0, number)))
    return clean


def resample_to_16k(samples: list[float], sample_rate: int) -> list[float]:
    """Naive linear resample to 16 kHz (chunk path only; file path uses torchaudio)."""
    if sample_rate == 16000:
        return samples
    import numpy as np

    arr = np.asarray(samples, dtype=np.float32)
    duration = len(arr) / sample_rate
    target_len = max(1, int(round(duration * 16000)))
    old_idx = np.linspace(0, 1, num=len(arr), dtype=np.float32)
    new_idx = np.linspace(0, 1, num=target_len, dtype=np.float32)
    resampled: list[float] = list(np.interp(new_idx, old_idx, arr).astype(np.float32).tolist())
    return resampled


_registry: LiveSessionRegistry | None = None


def get_live_registry() -> LiveSessionRegistry:
    global _registry
    if _registry is None:
        _registry = LiveSessionRegistry()
    return _registry
