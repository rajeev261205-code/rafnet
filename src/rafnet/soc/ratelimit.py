"""Simple in-memory rate limiter for expensive endpoints (e.g. /api/analyze).

Per-key token bucket: `max_requests` per `window_seconds`. Returns 429 with
Retry-After when exhausted. Process-local (sufficient for single-replica dev;
use Redis/an API gateway for multi-replica production).
"""
from __future__ import annotations

import os
import time
from collections import defaultdict, deque


class RateLimiter:
    def __init__(self, max_requests: int = 20, window_seconds: float = 60.0) -> None:
        self.max_requests = max_requests
        self.window = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str) -> tuple[bool, float]:
        now = time.monotonic()
        hits = self._hits[key]
        while hits and now - hits[0] > self.window:
            hits.popleft()
        if len(hits) >= self.max_requests:
            return False, max(0.0, self.window - (now - hits[0]))
        hits.append(now)
        return True, 0.0


def get_analyze_limiter() -> RateLimiter:
    try:
        max_req = int(os.getenv("RATE_LIMIT_ANALYZE_PER_MIN", "30"))
    except ValueError:
        max_req = 30
    return RateLimiter(max_requests=max_req, window_seconds=60.0)


def get_chunk_limiter() -> RateLimiter:
    """Higher budget for 1 s mic chunks (~1 req/s sustained).

    Chunk inference is the same cost as /api/analyze per window, but throttling
    mid-scan would break the feature; 120/min sustains continuous capture while
    still bounding abuse. Tune via RATE_LIMIT_CHUNK_PER_MIN.
    """
    try:
        max_req = int(os.getenv("RATE_LIMIT_CHUNK_PER_MIN", "120"))
    except ValueError:
        max_req = 120
    return RateLimiter(max_requests=max_req, window_seconds=60.0)
