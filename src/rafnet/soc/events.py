"""Event bus for live SOC analysis events (backend -> WebSocket -> frontend).

Real events only �?" emitted by SOCService during analyze_bytes:
analysis_started -> processing -> detection -> threat_detected (if HIGH/CRITICAL)
-> analysis_completed | error. No fake timers.

Every event is tagged with the owning principal (authenticated user id or
anonymous `None`) and subscribers only receive events for their own principal.
This prevents `/ws/live` replay/live streams from leaking another user's
analysis metadata.
"""
from __future__ import annotations

import asyncio
import logging
from collections import deque
from typing import Any

logger = logging.getLogger(__name__)


class EventBus:
    def __init__(self, max_history: int = 200) -> None:
        # queue -> owning principal (None = anonymous)
        self._subscribers: dict[asyncio.Queue, str | None] = {}
        self._history: deque[tuple[str | None, dict[str, Any]]] = deque(
            maxlen=max_history
        )

    def emit(
        self,
        event_type: str,
        payload: dict[str, Any],
        *,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """Publish an event scoped to `user_id` (None = anonymous principal)."""
        event = {"event": event_type, **payload}
        self._history.append((user_id, event))
        for queue, owner in list(self._subscribers.items()):
            if owner != user_id:
                continue
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning("Live subscriber queue full; dropping event %s", event_type)
        return event

    def subscribe(
        self, maxsize: int = 100, *, user_id: str | None = None
    ) -> asyncio.Queue:
        """Subscribe to events for exactly one principal (None = anonymous)."""
        queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self._subscribers[queue] = user_id
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self._subscribers.pop(queue, None)

    def recent(
        self, limit: int = 50, *, user_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Recent events visible to `user_id` (defaults to anonymous only)."""
        visible = [event for owner, event in self._history if owner == user_id]
        return visible[-limit:]


_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    global _bus
    if _bus is None:
        _bus = EventBus()
    return _bus
