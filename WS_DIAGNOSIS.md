# WS_DIAGNOSIS — /ws/live OFFLINE/RECONNECTING (diagnosis only, no code changed)

Date: 2026-10-11. Backend run as: `python -m uvicorn rafnet.api.main:app --port 8000`
(Python 3.12.0, `C:\Program Files\Python312` layout, site-packages under
`C:\Users\rajee\AppData\Roaming\Python\Python312\site-packages`).

## 1. Backend route search

- `grep "websocket|ws/live|WebSocket"` hits the REAL route:
  - `src/rafnet/soc/websocket.py:23` — `@ws_router.websocket("/ws/live")`
    (`live_socket`), mounted in `src/rafnet/api/main.py:191-194`
    (`app.include_router(ws_router)`).
  - Behavior today: optional auth gate (`SOC_REQUIRE_AUTH`, `?token=` or
    `Authorization: Bearer`), then `await websocket.accept()`, replay
    `bus.recent(limit=20)`, send `{"event": "connected", "status": "subscribed"}`,
    25 s `{"event": "ping"}` heartbeat task, then forwards every `EventBus`
    event (`analysis_started/processing/detection/threat_detected/
    analysis_completed/error`) via per-client `asyncio.Queue`. `finally`
    cancels heartbeat, unsubscribes, closes. Disconnects handled
    (`WebSocketDisconnect` + generic `except`).
  - Producer side is real: `src/rafnet/soc/service.py:67-125`
    (`analyze_bytes` emits started→processing→detection→threat?→completed)
    and `src/rafnet/soc/routes.py:209-262` (`POST /api/live/chunk` emits
    `processing/detection/threat_detected` on the same bus). No fake timers.
- Dead code (NOT served by the stated uvicorn command):
  `backend/app/main.py:77-89` defines a second `@app.websocket("/ws/live")`
  with a different manager/heartbeat contract. `rafnet.api.main:app` never
  imports it. Ignore for the fix.
- `src/rafnet/soc/limits.py:25` explicitly bypasses the body-size guard for
  `/ws/live`. Good.

Conclusion: **a /ws/live route exists and already broadcasts REAL pipeline
events.** "No route" is NOT the cause.

## 2. WebSocket transport libraries (the cause)

Run with the same Python that runs uvicorn:

- `python -m pip show websockets` → **not installed** (`Package(s) not found`).
- `python -m pip show wsproto httptools` → **both not installed**.
- `python -m pip show uvicorn` → 0.54.0, `Requires: click, h11` only —
  i.e. minimal install, no `uvicorn[standard]` extras.
- `pip list` confirms: `fastapi 0.142.4 / starlette 1.7.0 / anyio 4.15.1 /
  httpx 0.28.1 / uvicorn 0.54.0`, no `websockets`/`wsproto`/`httptools`.

Why this breaks exactly this page: REST (`/api/health`, mic `POST
/api/live/chunk`, `POST /api/analyze`) is plain HTTP via `h11` so it works
(`model_loaded:true, mock_mode:false`). WebSocket upgrade needs a WS
implementation inside uvicorn (`websockets` or `wsproto`). With neither,
uvicorn cannot complete the `ws://localhost:8000/ws/live` handshake — the
browser `new WebSocket(...)` immediately errors/closes. `useLiveEvents`
then does `onclose → disconnected → exponential backoff → connecting…`,
i.e. the observed `LIVE: RECONNECTING… / OFFLINE, 0 events`. The existing
pytest (`tests/test_soc.py::test_soc_live_websocket`) still passes because
`starlette.testclient.TestClient.websocket_connect` is an in-process
transport that does NOT need `websockets`/uvicorn — it masks the prod
failure.

## 3. Frontend contract (`rafnet-frontend`, read-only)

- `src/services/api.ts:289-298` `liveWebSocketUrl()`:
  `NEXT_PUBLIC_BACKEND_URL=http://localhost:8000` → `ws://localhost:8000/ws/live`
  (protocol swap http→ws, https→wss, strip trailing `/`, append `/ws/live`).
  Returns `null` when unconfigured.
- `src/hooks/useLiveEvents.ts`:
  - Connects to that URL, appends `?token=<supabase JWT>` when logged in
    (harmless when `SOC_REQUIRE_AUTH=false`).
  - `onopen → status 'connected'`; `onclose/onerror → 'disconnected'` +
    backoff reconnect (`min(15000, 1000*2**retry)`); no backend URL →
    `'unavailable'`.
  - `onmessage`: `JSON.parse`, **drop only `event === 'ping'`**, else prepend
    to `events` (cap 100). Message shape: `{event: string, ...unknown}`.
  - No auth header (browsers can't set WS headers) — query token only.
- `src/app/(dashboard)/live/page.tsx:45-48,107-111,267-271`:
  `connected → LIVE: CONNECTED / LIVE chip`; `connecting → CONNECTING…`;
  else `OFFLINE / RECONNECTING…`. Event log renders `ev.event` +
  `prediction/confidence/risk_level/id`; empty state says `start the backend`.
  Mic scan (`useMicChunks` → `POST /api/live/session|chunk`) is independent
  of the socket, which is why mic REST works while WS shows OFFLINE.

## 4. Contract gaps (secondary, after transport)

- Path matches exactly: frontend `/ws/live` == backend `/ws/live`. No fix needed.
- Envelope matches: both use `{"event": "<type>", ...}`. Frontend ignores
  `ping`, backend sends `ping` heartbeat — compatible.
- Event types overlap: backend emits `analysis_started/processing/detection/
  threat_detected/analysis_completed/error` + `connected`; `API.md:23-26`
  documents exactly these. Frontend renders any `event` generically, so all
  flow through.
- One TASK-spec gap: the task requires **a `system_status` message on
  connect**; the backend currently sends `{"event": "connected", ...}`
  instead. The frontend does not strictly require it (any non-ping frame
  counts as an event and `LIVE` comes from `onopen`), but to meet the stated
  contract the backend should emit a `system_status` frame on connect while
  keeping `connected` for backward compat.
- Connection-manager requirement is already met (per-client queue, heartbeat
  cancel, unsubscribe, close in `finally`); keep it, don't simulate.

## 5. CORS / auth

- `src/rafnet/api/main.py:153-166`: `CORS_ORIGINS` defaults to
  `http://localhost:3000,http://localhost:8000`; `.env` sets exactly that —
  **http://localhost:3000 already allowed** (REST). WS upgrade is not
  CORS-gated by Starlette anyway.
- `src/rafnet/soc/auth.py` + `.env:SOC_REQUIRE_AUTH=false`: anonymous WS
  allowed; `?token=`/Bearer validated only when `true` (close 4401). Matches
  the frontend's optional-token behavior. No auth change needed for local
  dev; keep the existing gate as-is.

## Root cause (one line)

**Missing WebSocket server transport: neither `websockets` nor `wsproto`
is installed for the uvicorn Python, so the existing, otherwise-correct
`/ws/live` route can never complete a real browser handshake** — REST works,
WS loops RECONNECTING. Fix = add `websockets` (or `uvicorn[standard]`) to
deps + install with that same Python, plus emit the required
`system_status` frame on connect.
