"""Event bus for live SOC analysis events (backend -> WebSocket -> frontend).

Real events only — emitted by SOCService during analyze_bytes:
analysis_started -> processing -> detection -> threat_detected (if HIGH/CRITICAL)
-> analysis_completed | error. No fake timers.
"""
from __future__ import annotations

import asyncio
import logging
from collections import deque
from typing import Any

logger = logging.getLogger(__name__)


class EventBus:
    def __init__(self, max_history: int = 200) -> None:
        self._subscribers: set[asyncio.Queue] = set()
        self._history: deque[dict[str, Any]] = deque(maxlen=max_history)

    def emit(self, event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        event = {"event": event_type, **payload}
        self._history.append(event)
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning("Live subscriber queue full; dropping event %s", event_type)
        return event

    def subscribe(self, maxsize: int = 100) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self._subscribers.discard(queue)

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        return list(self._history)[-limit:]


_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    global _bus
    if _bus is None:
        _bus = EventBus()
    return _bus
