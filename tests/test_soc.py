"""SOC tests — no torch required (MockVoiceModel + validation paths)."""
import io
import struct
import wave

from fastapi.testclient import TestClient


def _wav_bytes(seconds: float = 0.5, sr: int = 16000) -> bytes:
    import math

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        n = int(seconds * sr)
        for i in range(n):
            v = int(10000 * math.sin(2 * math.pi * 440 * i / sr))
            w.writeframes(struct.pack("<h", v))
    return buf.getvalue()


def _mock_client():
    # Standalone app (avoids importing torch-dependent rafnet.api.main on
    # machines without the heavy ML runtime; production mounts the same
    # soc router inside api/main.py).
    from fastapi import FastAPI

    from rafnet.soc import routes
    from rafnet.soc.model_adapter import MockVoiceModel
    from rafnet.soc.service import SOCService

    service = SOCService(MockVoiceModel())
    routes.override_soc_service(service)
    app = FastAPI()
    app.include_router(routes.router)

    return TestClient(app), routes


def test_soc_health_reports_mock():
    client, _ = _mock_client()
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["mock_mode"] is True
    assert body["model_loaded"] is True


def test_soc_analyze_rejects_unsupported():
    client, _ = _mock_client()
    r = client.post("/api/analyze", files={"file": ("evil.txt", b"hello", "text/plain")})
    assert r.status_code == 400


def test_soc_analyze_rejects_empty():
    client, _ = _mock_client()
    r = client.post("/api/analyze", files={"file": ("empty.wav", b"", "audio/wav")})
    assert r.status_code in (400, 500)


def test_soc_analyze_wav_mock_contract():
    client, routes = _mock_client()
    r = client.post("/api/analyze", files={"file": ("tone.wav", _wav_bytes(), "audio/wav")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["prediction"] in ("REAL", "UNCERTAIN", "SYNTHETIC")
    assert 0.0 <= body["confidence"] <= 1.0
    assert 0 <= body["risk_score"] <= 100
    assert body["risk_level"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
    assert body["mock"] is True
    assert body["model_version"] == "mock-0.0.0"
    # retrieval + dashboard
    g = client.get(f"/api/analysis/{body['id']}")
    assert g.status_code == 200
    d = client.get("/api/dashboard")
    assert d.status_code == 200
    assert d.json()["total_analyses"] >= 1
    routes.override_soc_service.__self__ if False else None


def test_soc_models_info():
    client, _ = _mock_client()
    r = client.get("/api/models")
    assert r.status_code == 200
    assert r.json()["name"] == "RAFNet"


def test_soc_history_threats_calls():
    client, _ = _mock_client()
    r = client.post("/api/analyze", files={"file": ("tone.wav", _wav_bytes(), "audio/wav")})
    assert r.status_code == 200, r.text
    body = r.json()

    h = client.get("/api/history?limit=10")
    assert h.status_code == 200
    assert h.json()["total"] >= 1
    assert any(item["id"] == body["id"] for item in h.json()["items"])

    t = client.get("/api/threats")
    assert t.status_code == 200
    assert isinstance(t.json()["items"], list)

    c = client.get("/api/calls")
    assert c.status_code == 200
    items = c.json()["items"]
    assert len(items) >= 1
    call_id = items[0]["id"]
    d = client.get(f"/api/calls/{call_id}")
    assert d.status_code == 200

    e = client.get(f"/api/analysis/{body['id']}/events")
    assert e.status_code == 200
    assert any(ev.get("id") == body["id"] for ev in e.json()["events"])

    assert client.get("/api/analysis/does-not-exist").status_code == 404
    assert client.get("/api/calls/does-not-exist").status_code == 404
    assert client.get("/api/threats/does-not-exist").status_code == 404


def test_soc_experiments_verbatim(tmp_path, monkeypatch):
    import json

    report = {
        "status": "COMPLETED_WITH_CAVEATS",
        "dataset": {"size": 5000},
        "components": {"classical_branch": {}, "gateway2": {}},
    }
    (tmp_path / "phase19_report.json").write_text(json.dumps(report))
    monkeypatch.setenv("EXPERIMENTS_DIR", str(tmp_path))
    client, _ = _mock_client()
    r = client.get("/api/experiments")
    assert r.status_code == 200
    items = r.json()["items"]
    assert any(item["id"] == "phase19" for item in items)
    phase19 = next(item for item in items if item["id"] == "phase19")
    assert phase19["metrics"]["status"] == "COMPLETED_WITH_CAVEATS"


def test_soc_analyze_rate_limited():
    from fastapi import FastAPI

    from rafnet.soc import routes
    from rafnet.soc.model_adapter import MockVoiceModel
    from rafnet.soc.ratelimit import RateLimiter
    from rafnet.soc.service import SOCService

    service = SOCService(MockVoiceModel())
    routes.override_soc_service(service)
    routes._limiter = RateLimiter(max_requests=1, window_seconds=60.0)
    try:
        app = FastAPI()
        app.include_router(routes.router)
        from fastapi.testclient import TestClient

        client = TestClient(app)
        first = client.post("/api/analyze", files={"file": ("a.wav", _wav_bytes(), "audio/wav")})
        assert first.status_code == 200
        second = client.post("/api/analyze", files={"file": ("b.wav", _wav_bytes(), "audio/wav")})
        assert second.status_code == 429
        assert "Retry-After" in second.headers
    finally:
        routes._limiter = None


def test_soc_auth_open_by_default():
    """Without SOC_REQUIRE_AUTH, anonymous requests work and carry no user_id."""
    import os

    os.environ.pop("SOC_REQUIRE_AUTH", None)
    client, _ = _mock_client()
    r = client.post("/api/analyze", files={"file": ("tone.wav", _wav_bytes(), "audio/wav")})
    assert r.status_code == 200, r.text
    assert "user_id" not in r.json()


def test_soc_auth_enforced_when_required(monkeypatch):
    """SOC_REQUIRE_AUTH=true rejects anonymous writes and bad tokens."""

    monkeypatch.setenv("SOC_REQUIRE_AUTH", "true")
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_ANON_KEY", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    client, _ = _mock_client()
    r = client.post("/api/analyze", files={"file": ("tone.wav", _wav_bytes(), "audio/wav")})
    assert r.status_code in (401, 503)
    bad = client.post(
        "/api/analyze",
        files={"file": ("tone.wav", _wav_bytes(), "audio/wav")},
        headers={"Authorization": "Bearer invalid-token"},
    )
    assert bad.status_code in (401, 503)


def test_live_chunk_buffering_then_analyzed():
    import numpy as np

    client, _ = _mock_client()
    s = client.post("/api/live/session")
    assert s.status_code == 200, s.text
    session_id = s.json()["session_id"]

    small = (0.05 * np.sin(2 * np.pi * 440 * np.arange(4000) / 16000)).tolist()
    r1 = client.post("/api/live/chunk", json={"session_id": session_id, "sample_rate": 16000, "samples": small})
    assert r1.status_code == 200, r1.text
    assert r1.json()["status"] == "buffering"

    big = (0.05 * np.sin(2 * np.pi * 440 * np.arange(16000) / 16000)).tolist()
    r2 = client.post("/api/live/chunk", json={"session_id": session_id, "sample_rate": 16000, "samples": big})
    assert r2.status_code == 200, r2.text
    body = r2.json()
    assert body["status"] == "analyzed"
    assert body["prediction"] in ("REAL", "UNCERTAIN", "SYNTHETIC")
    assert body["session_id"] == session_id

    bad = client.post("/api/live/chunk", json={"session_id": session_id, "sample_rate": 16000, "samples": []})
    assert bad.status_code in (400, 422)
    bad_sr = client.post("/api/live/chunk", json={"session_id": session_id, "sample_rate": 3000, "samples": [0.0] * 100})
    assert bad_sr.status_code in (400, 422)


def _loud_wav_bytes(seconds: float = 0.5, sr: int = 16000) -> bytes:
    """Loud tone -> mock model returns SYNTHETIC/CRITICAL (exercises threats)."""
    import math

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        n = int(seconds * sr)
        for i in range(n):
            v = int(20000 * math.sin(2 * math.pi * 440 * i / sr))
            w.writeframes(struct.pack("<h", v))
    return buf.getvalue()


def test_threat_triage_workflow():
    client, _ = _mock_client()
    missing = client.patch("/api/threats/thr-nope", json={"status": "acknowledged"})
    assert missing.status_code == 404
    r = client.post("/api/analyze", files={"file": ("loud.wav", _loud_wav_bytes(), "audio/wav")})
    assert r.status_code == 200, r.text
    assert r.json()["risk_level"] in ("HIGH", "CRITICAL")
    t = client.get("/api/threats")
    assert t.status_code == 200
    items = t.json()["items"]
    assert len(items) >= 1, "loud mock analysis must create a threat"
    threat_id = items[0]["id"]
    updated = client.patch(f"/api/threats/{threat_id}", json={"status": "acknowledged"})
    assert updated.status_code == 200
    assert updated.json()["status"] == "acknowledged"
    detail = client.get(f"/api/threats/{threat_id}")
    assert detail.json()["status"] == "acknowledged"
    bad = client.patch(f"/api/threats/{threat_id}", json={"status": "bogus"})
    assert bad.status_code == 400


def test_chunk_limiter_independent_of_analyze():
    """Mic chunks must not share the tight /api/analyze budget."""
    from rafnet.soc import routes
    from rafnet.soc.ratelimit import RateLimiter

    routes._limiter = RateLimiter(max_requests=1, window_seconds=60.0)
    routes._chunk_limiter = RateLimiter(max_requests=1000, window_seconds=60.0)
    try:
        import numpy as np
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from rafnet.soc.model_adapter import MockVoiceModel
        from rafnet.soc.service import SOCService

        routes.override_soc_service(SOCService(MockVoiceModel()))
        app = FastAPI()
        app.include_router(routes.router)
        client = TestClient(app)
        small = (0.05 * np.sin(2 * np.pi * 440 * np.arange(16000) / 16000)).tolist()
        for _ in range(3):
            r = client.post("/api/live/chunk", json={"sample_rate": 16000, "samples": small})
            assert r.status_code == 200, r.text
    finally:
        routes._limiter = None
        routes._chunk_limiter = None


def test_body_size_guard():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from rafnet.soc import routes
    from rafnet.soc.limits import MaxBodySizeMiddleware
    from rafnet.soc.model_adapter import MockVoiceModel
    from rafnet.soc.service import SOCService

    routes.override_soc_service(SOCService(MockVoiceModel()))
    app = FastAPI()
    app.add_middleware(MaxBodySizeMiddleware)
    app.include_router(routes.router)
    client = TestClient(app)
    big = b"x" * (33 * 1024 * 1024)
    r = client.post(
        "/api/analyze",
        content=big,
        headers={"Content-Type": "audio/wav", "Content-Length": str(len(big))},
    )
    assert r.status_code == 413


def test_analysis_detail_scoped_by_owner():
    from rafnet.soc.model_adapter import MockVoiceModel
    from rafnet.soc.service import SOCService

    service = SOCService(MockVoiceModel())
    mine = service.analyze_bytes(_wav_bytes(), "mine.wav", user_id="user-1")
    assert service.get_analysis(mine.id, user_id="user-1") is not None
    assert service.get_analysis(mine.id, user_id="user-2") is None
    assert service.get_analysis(mine.id) is not None  # anonymous still sees legacy rows
    anon = service.analyze_bytes(_wav_bytes(), "anon.wav")
    assert service.get_analysis(anon.id, user_id="user-2") is not None


def test_websocket_rejects_anonymous_when_auth_required(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from starlette.websockets import WebSocketDisconnect

    from rafnet.soc import routes
    from rafnet.soc.model_adapter import MockVoiceModel
    from rafnet.soc.service import SOCService
    from rafnet.soc.websocket import ws_router

    monkeypatch.setenv("SOC_REQUIRE_AUTH", "true")
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_ANON_KEY", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    routes.override_soc_service(SOCService(MockVoiceModel()))
    app = FastAPI()
    app.include_router(routes.router)
    app.include_router(ws_router)
    client = TestClient(app)
    try:
        with client.websocket_connect("/ws/live"):
            raise AssertionError("anonymous WS should have been rejected")
    except WebSocketDisconnect as exc:
        assert exc.code == 4401


class _FakePostgrest:
    """Minimal in-memory PostgREST stand-in (stdlib only) for adapter tests.

    Multi-table (`tables` dict) with RLS emulation: requests carrying the
    service key see everything; any other key sees zero `analysis_results`
    rows (mirroring the owner-only RLS policies) but can still read `models`.
    """

    def __init__(self, service_key: str = "test-service-key") -> None:
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from urllib.parse import parse_qs, urlparse

        outer = self
        outer.tables: dict[str, list[dict]] = {
            "profiles": [],
            "analysis_results": [],
            "calls": [],
            "threats": [],
            "models": [],
            "experiments": [],
            "analysis_events": [],
        }
        outer.service_key = service_key
        outer.saw_apikey = False

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # silence test output
                pass

            def _send(self, obj, status=200):
                import json as _json

                data = _json.dumps(obj).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _table(self):
                parsed = urlparse(self.path)
                parts = parsed.path.rsplit("/rest/v1/", 1)
                return parts[1].split("?")[0] if len(parts) == 2 else "", parsed

            def do_GET(self):
                if self.headers.get("apikey"):
                    outer.saw_apikey = True
                table, parsed = self._table()
                if table not in outer.tables:
                    return self._send({"error": "no table"}, 404)
                # RLS emulation: non-service keys see no analysis rows.
                if table == "analysis_results" and self.headers.get("apikey") != outer.service_key:
                    return self._send([])
                query = parse_qs(parsed.query)
                rows = list(outer.tables[table])
                for field, values in query.items():
                    wanted = (values[0] or "")
                    if wanted.startswith("eq."):
                        rows = [r for r in rows if str(r.get(field, "")) == wanted[3:]]
                if query.get("order", [""])[0] == "created_at.desc":
                    rows.sort(key=lambda r: r.get("created_at", ""), reverse=True)
                offset = int(query.get("offset", ["0"])[0])
                limit = query.get("limit", [None])[0]
                rows = rows[offset:]
                if limit is not None:
                    rows = rows[: int(limit)]
                if query.get("select", ["*"])[0] == "id":
                    rows = [{"id": r.get("id")} for r in rows]
                return self._send(rows)

            def do_POST(self):
                if self.headers.get("apikey"):
                    outer.saw_apikey = True
                import json as _json

                table, _ = self._table()
                if table not in outer.tables:
                    return self._send({"error": "no table"}, 404)
                length = int(self.headers.get("Content-Length", "0"))
                body = _json.loads(self.rfile.read(length).decode() or "null")
                outer.tables[table].append(body)
                return self._send([body], 201)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def rows(self) -> list[dict]:
        """Backward-compat alias: the analysis_results table."""
        return self.tables["analysis_results"]

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()


def test_supabase_store_against_fake_postgrest():
    from rafnet.soc.store import SupabaseAnalysisStore

    fake = _FakePostgrest()
    try:
        store = SupabaseAnalysisStore(fake.url, "test-service-key")
        rec_a = {"id": "a1", "created_at": "2026-01-01T00:00:00+00:00", "prediction": "REAL"}
        rec_b = {"id": "b2", "created_at": "2026-01-02T00:00:00+00:00", "prediction": "SYNTHETIC"}
        store.save_analysis(rec_b)
        store.save_analysis(rec_a)
        assert fake.saw_apikey, "adapter must send the apikey header"
        assert store.get_analysis("a1")["prediction"] == "REAL"
        assert store.get_analysis("missing") is None
        items = store.list_analyses(limit=10)
        assert [r["id"] for r in items] == ["b2", "a1"], "created_at.desc order"
        assert [r["id"] for r in store.list_analyses(limit=1, offset=1)] == ["a1"]
        assert store.count() == 2
    finally:
        fake.stop()


def test_get_store_fallbacks():
    import os

    from rafnet.soc.store import InMemoryAnalysisStore, SupabaseAnalysisStore, get_store

    old_url, old_key = os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    try:
        os.environ.pop("SUPABASE_URL", None)
        os.environ.pop("SUPABASE_SERVICE_ROLE_KEY", None)
        assert isinstance(get_store(), InMemoryAnalysisStore)
        # Unreachable URL must degrade to memory, never raise.
        os.environ["SUPABASE_URL"] = "http://127.0.0.1:1"
        os.environ["SUPABASE_SERVICE_ROLE_KEY"] = "x"
        assert isinstance(get_store(), InMemoryAnalysisStore)
        # Reachable fake works end to end through get_store.
        fake = _FakePostgrest()
        try:
            os.environ["SUPABASE_URL"] = fake.url
            assert isinstance(get_store(), SupabaseAnalysisStore)
        finally:
            fake.stop()
    finally:
        if old_url is None:
            os.environ.pop("SUPABASE_URL", None)
        else:
            os.environ["SUPABASE_URL"] = old_url
        if old_key is None:
            os.environ.pop("SUPABASE_SERVICE_ROLE_KEY", None)
        else:
            os.environ["SUPABASE_SERVICE_ROLE_KEY"] = old_key


def _load_script(name: str):
    import importlib.util

    path = f"supabase/{name}.py"
    spec = importlib.util.spec_from_file_location(f"soc_{name}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_seed_collect_uses_real_files():
    seed = _load_script("seed")
    model, experiments = seed.collect()
    assert model["id"] == "rafnet-1.0.0-phase19"
    sizes = model["metadata"]["checkpoints"]
    assert sizes["classical_logistic.pkl"] == 2465
    assert sizes["gateway2.pth"] == 11019
    assert sizes["quantum_branch.pkl"] == 16634
    assert any(e["name"].startswith("phase19") for e in experiments)
    assert all(e["model_id"] == model["id"] for e in experiments)


def test_seed_rest_writes_to_fake():
    seed = _load_script("seed")
    fake = _FakePostgrest()
    try:
        model, _ = seed.collect()
        seed.rest(fake.url, "test-service-key", "models", model, "id")
        assert any(r.get("id") == "rafnet-1.0.0-phase19" for r in fake.tables["models"])
    finally:
        fake.stop()


def test_verify_pass_and_fail(monkeypatch):
    verify = _load_script("verify")
    fake = _FakePostgrest()
    try:
        # Seed the models row the migration would create.
        fake.tables["models"].append({"id": "rafnet-1.0.0-phase19"})
        monkeypatch.setenv("SUPABASE_URL", fake.url)
        monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "test-service-key")
        monkeypatch.setenv("SUPABASE_ANON_KEY", "anon-key")
        verify.FAILURES.clear()
        assert verify.main() == 0, verify.FAILURES
        # Drop the seed row -> must fail with the migration hint path.
        fake.tables["models"].clear()
        verify.FAILURES.clear()
        assert verify.main() == 1
        assert verify.FAILURES, "missing seed row must be reported"
    finally:
        fake.stop()
        verify.FAILURES.clear()


def test_full_app_boots_without_torch():
    """Full FastAPI app (api.main) must import and serve SOC even without torch."""
    import rafnet.api.main as app_module

    assert hasattr(app_module, "app")
    assert hasattr(app_module, "_LEGACY_AVAILABLE")
    from fastapi.testclient import TestClient

    from rafnet.soc import routes
    from rafnet.soc.model_adapter import MockVoiceModel
    from rafnet.soc.service import SOCService

    routes.override_soc_service(SOCService(MockVoiceModel()))
    try:
        client = TestClient(app_module.app, raise_server_exceptions=False)
        r = client.get("/api/health")
        assert r.status_code == 200, r.text
        assert r.json()["mock_mode"] is True
        z = client.get("/healthz")
        assert z.status_code == 200
    finally:
        routes.override_soc_service.__self__ if False else None


def test_soc_live_websocket():
    from fastapi import FastAPI

    from rafnet.soc import routes
    from rafnet.soc.model_adapter import MockVoiceModel
    from rafnet.soc.service import SOCService
    from rafnet.soc.websocket import ws_router

    service = SOCService(MockVoiceModel())
    routes.override_soc_service(service)
    app = FastAPI()
    app.include_router(routes.router)
    app.include_router(ws_router)
    from fastapi.testclient import TestClient

    client = TestClient(app)
    with client.websocket_connect("/ws/live") as ws:
        # Drain replay + connected frame (bus is process-global across tests).
        seen_connect = False
        for _ in range(25):
            msg = ws.receive_json()
            if msg.get("event") == "connected":
                seen_connect = True
                break
        assert seen_connect
        r = client.post("/api/analyze", files={"file": ("tone.wav", _wav_bytes(), "audio/wav")})
        assert r.status_code == 200
        seen = set()
        for _ in range(10):
            msg = ws.receive_json()
            seen.add(msg.get("event"))
            if "analysis_completed" in seen:
                break
        assert "analysis_started" in seen
        assert "detection" in seen
        assert "analysis_completed" in seen
