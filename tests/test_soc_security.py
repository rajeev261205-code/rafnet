"""Security regression tests: tenant isolation, auth, WS scoping, startup guards."""
import asyncio

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from tests.test_soc import _mock_client, _wav_bytes


def _drain(queue: asyncio.Queue) -> list[dict]:
    items: list[dict] = []
    while not queue.empty():
        items.append(queue.get_nowait())
    return items


def _auth_env(monkeypatch) -> None:
    monkeypatch.setenv("SOC_REQUIRE_AUTH", "true")
    monkeypatch.setenv("SUPABASE_URL", "http://supabase.test")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "test-anon-key")


def test_event_bus_scopes_history_and_live_queues():
    from rafnet.soc.events import EventBus

    bus = EventBus()
    alice = bus.subscribe(user_id="alice")
    bob = bus.subscribe(user_id="bob")
    anon = bus.subscribe()

    bus.emit("detection", {"id": "a-1"}, user_id="alice")
    bus.emit("detection", {"id": "b-1"}, user_id="bob")
    bus.emit("detection", {"id": "n-1"})

    assert [e["id"] for e in _drain(alice)] == ["a-1"]
    assert [e["id"] for e in _drain(bob)] == ["b-1"]
    assert [e["id"] for e in _drain(anon)] == ["n-1"]
    assert [e["id"] for e in bus.recent(user_id="alice")] == ["a-1"]
    assert [e["id"] for e in bus.recent(user_id="bob")] == ["b-1"]
    assert [e["id"] for e in bus.recent()] == ["n-1"]


def test_websocket_replays_and_streams_only_own_events(monkeypatch):
    from rafnet.soc import routes
    from rafnet.soc.events import get_event_bus
    from rafnet.soc.model_adapter import MockVoiceModel
    from rafnet.soc.service import SOCService
    from rafnet.soc.websocket import ws_router

    _auth_env(monkeypatch)
    monkeypatch.setattr(
        "rafnet.soc.websocket.validate_supabase_token",
        lambda token: {"id": token},
    )
    routes.override_soc_service(SOCService(MockVoiceModel()))
    bus = get_event_bus()
    bus.emit("detection", {"id": "alice-replay"}, user_id="alice")
    bus.emit("detection", {"id": "bob-replay"}, user_id="bob")

    app = FastAPI()
    app.include_router(routes.router)
    app.include_router(ws_router)
    client = TestClient(app)

    with client.websocket_connect("/ws/live?token=alice") as ws:
        first = ws.receive_json()
        assert first["event"] == "system_status"
        replayed: list[dict] = []
        while True:
            msg = ws.receive_json()
            if msg["event"] == "connected":
                break
            replayed.append(msg)
        ids = [m.get("id") for m in replayed]
        assert "alice-replay" in ids
        assert "bob-replay" not in ids

        bus.emit("detection", {"id": "bob-live"}, user_id="bob")
        bus.emit("detection", {"id": "alice-live"}, user_id="alice")
        assert ws.receive_json()["id"] == "alice-live"


def test_websocket_rejects_invalid_supplied_token(monkeypatch):
    from rafnet.soc import routes
    from rafnet.soc.websocket import ws_router

    _auth_env(monkeypatch)

    def _reject(_token: str) -> dict:
        raise ValueError("expired")

    monkeypatch.setattr("rafnet.soc.websocket.validate_supabase_token", _reject)
    monkeypatch.setattr(
        "rafnet.soc.auth.validate_supabase_token", lambda token: {"id": token}
    )
    app = FastAPI()
    app.include_router(routes.router)
    app.include_router(ws_router)
    client = TestClient(app)
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect("/ws/live?token=stale"):
            pass
    assert exc.value.code == 4401


def test_auth_required_protects_all_soc_data_routes(monkeypatch):
    _auth_env(monkeypatch)
    client, _ = _mock_client()
    for path in (
        "/api/health",
        "/api/models",
        "/api/capabilities",
        "/api/experiments",
        "/api/evaluation",
        "/api/enrollment",
    ):
        assert client.get(path).status_code == 401, path


def test_live_sessions_are_owner_scoped_and_never_hijacked(monkeypatch):
    from rafnet.soc import auth

    _auth_env(monkeypatch)
    monkeypatch.setattr(auth, "validate_supabase_token", lambda token: {"id": token})
    client, _ = _mock_client()

    created = client.post(
        "/api/live/session", headers={"Authorization": "Bearer alice"}
    )
    assert created.status_code == 200, created.text
    session_id = created.json()["session_id"]

    chunk = {"session_id": session_id, "sample_rate": 16000, "samples": [0.0] * 160}
    assert (
        client.post(
            "/api/live/chunk",
            json=chunk,
            headers={"Authorization": "Bearer bob"},
        ).status_code
        == 404
    )
    assert (
        client.post(
            "/api/live/chunk",
            json={"session_id": "live-does-not-exist", "sample_rate": 16000, "samples": [0.0] * 160},
            headers={"Authorization": "Bearer alice"},
        ).status_code
        == 404
    )
    own = client.post(
        "/api/live/chunk",
        json=chunk,
        headers={"Authorization": "Bearer alice"},
    )
    assert own.status_code == 200, own.text
    assert own.json()["status"] == "buffering"


def test_live_chunk_sample_cap_is_enforced_before_processing():
    from rafnet.soc.live import MAX_CHUNK_SAMPLES

    client, _ = _mock_client()
    session_id = client.post("/api/live/session").json()["session_id"]
    r = client.post(
        "/api/live/chunk",
        json={
            "session_id": session_id,
            "sample_rate": 16000,
            "samples": [0.0] * (MAX_CHUNK_SAMPLES + 1),
        },
    )
    assert r.status_code == 422, r.text


def test_analysis_events_require_analysis_access(monkeypatch):
    from rafnet.soc import auth, routes

    monkeypatch.setenv("SUPABASE_URL", "http://supabase.test")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "test-anon-key")
    monkeypatch.delenv("SOC_REQUIRE_AUTH", raising=False)
    monkeypatch.setattr(auth, "validate_supabase_token", lambda token: {"id": token})
    client, _ = _mock_client()
    service = routes.get_soc_service()
    mine = service.analyze_bytes(_wav_bytes(), "mine.wav", user_id="alice")

    assert (
        client.get(
            f"/api/analysis/{mine.id}/events",
            headers={"Authorization": "Bearer alice"},
        ).status_code
        == 200
    )
    assert (
        client.get(
            f"/api/analysis/{mine.id}/events",
            headers={"Authorization": "Bearer bob"},
        ).status_code
        == 404
    )
    assert client.get(f"/api/analysis/{mine.id}/events").status_code == 404


def test_startup_security_validation_fails_closed(monkeypatch):
    from rafnet.soc.auth import validate_security_config

    monkeypatch.setenv("RAFNET_ENV", "production")
    monkeypatch.setenv("SOC_REQUIRE_AUTH", "true")
    monkeypatch.setenv("SUPABASE_URL", "http://supabase.test")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "test-anon-key")
    monkeypatch.setenv("SOC_CORS_ORIGINS", "https://app.example.com")
    validate_security_config()

    monkeypatch.setenv("SOC_REQUIRE_AUTH", "false")
    with pytest.raises(RuntimeError, match="SOC_REQUIRE_AUTH"):
        validate_security_config()

    monkeypatch.setenv("SOC_REQUIRE_AUTH", "true")
    monkeypatch.setenv("SOC_CORS_ORIGINS", "*")
    with pytest.raises(RuntimeError, match="CORS"):
        validate_security_config()

    monkeypatch.setenv("SOC_CORS_ORIGINS", "https://app.example.com")
    monkeypatch.delenv("SUPABASE_ANON_KEY")
    with pytest.raises(RuntimeError, match="SUPABASE_URL"):
        validate_security_config()


def test_security_headers_are_set(monkeypatch):
    from rafnet.soc.limits import SecurityHeadersMiddleware

    monkeypatch.delenv("RAFNET_ENV", raising=False)
    app = FastAPI()
    app.add_middleware(SecurityHeadersMiddleware)

    @app.get("/x")
    def x():
        return {"ok": True}

    r = TestClient(app).get("/x")
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["x-frame-options"] == "DENY"
    assert r.headers["referrer-policy"] == "no-referrer"
    assert r.headers["cache-control"] == "no-store"
    assert "strict-transport-security" not in r.headers


def test_hsts_only_in_production(monkeypatch):
    from rafnet.soc.limits import SecurityHeadersMiddleware

    monkeypatch.setenv("RAFNET_ENV", "production")
    app = FastAPI()
    app.add_middleware(SecurityHeadersMiddleware)

    @app.get("/x")
    def x():
        return {"ok": True}

    r = TestClient(app).get("/x")
    assert "strict-transport-security" in r.headers
