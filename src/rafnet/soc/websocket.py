"""Live WebSocket — streams REAL analysis events (no fake timers).

Endpoint: /ws/live
Outbound frames: system_status (on connect), analysis_started, processing,
detection, threat_detected, analysis_completed, error + heartbeat ping every
25s. On connect, the server sends system_status, replays up to 20 recent
events for the caller's principal, then sends connected so the console has
immediate context. Handles disconnects and slow consumers gracefully
(per-client queue).

Auth: browsers cannot attach Authorization headers to a WebSocket upgrade, so
the Supabase access token is accepted as `?token=`. When auth is required,
missing/invalid tokens close with 4401. In dev mode (auth not required) a
supplied token is still validated rather than silently downgraded to anonymous.
Events are always filtered by the caller's principal so one user's analysis
metadata can never be replayed or streamed to another user.
"""
from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from .auth import auth_required, validate_supabase_token
from .events import get_event_bus

logger = logging.getLogger(__name__)
ws_router = APIRouter()


def _system_status() -> dict:
    """Real model status for the on-connect system_status frame.

    Best effort: never raises, so a model hiccup can't break the handshake.
    """
    try:
        from .routes import get_soc_service

        svc = get_soc_service()
        return {
            "event": "system_status",
            "status": "connected",
            "model_loaded": bool(svc.model.is_loaded()),
            "mock_mode": bool(svc.mock_mode),
            "model": getattr(svc.model, "name", "unknown"),
            "auth_required": auth_required(),
        }
    except Exception as exc:  # noqa: BLE001 — status must never fail connect
        logger.warning("system_status degraded: %s", exc)
        return {
            "event": "system_status",
            "status": "connected",
            "auth_required": auth_required(),
        }


def _token(websocket: WebSocket) -> str | None:
    token = websocket.query_params.get("token")
    if token:
        return token.strip() or None
    header = websocket.headers.get("authorization", "")
    if header.lower().startswith("bearer "):
        return header[7:].strip() or None
    return None


@ws_router.websocket("/ws/live")
async def live_socket(websocket: WebSocket) -> None:
    token = _token(websocket)
    if token is None and auth_required():
        await websocket.close(code=4401)
        return
    user_id: str | None = None
    if token is not None:
        try:
            info = await asyncio.to_thread(validate_supabase_token, token)
            user_id = info.get("id") or info.get("sub")
        except Exception as exc:
            logger.info("Live socket rejected invalid token: %s", exc)
            await websocket.close(code=4401)
            return
        if not user_id:
            await websocket.close(code=4401)
            return
    await websocket.accept()
    bus = get_event_bus()
    queue = bus.subscribe(user_id=user_id)
    heartbeat: asyncio.Task | None = None

    async def ping() -> None:
        try:
            while True:
                await asyncio.sleep(25)
                await websocket.send_json({"event": "ping", "status": "alive"})
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

    try:
        await websocket.send_json(_system_status())
        for recent in bus.recent(limit=20, user_id=user_id):
            await websocket.send_json(recent)
        await websocket.send_json({"event": "connected", "status": "subscribed"})
        heartbeat = asyncio.create_task(ping())
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=60.0)
                await websocket.send_json(event)
            except asyncio.TimeoutError:
                continue
    except WebSocketDisconnect:
        logger.info("Live socket disconnected")
    except Exception as exc:
        logger.warning("Live socket error: %s", exc)
    finally:
        if heartbeat is not None:
            heartbeat.cancel()
        bus.unsubscribe(queue)
        try:
            await websocket.close()
        except Exception:
            pass
