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


def _silence_wav_bytes(seconds: float = 0.5, sr: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"\x00\x00" * int(seconds * sr))
    return buf.getvalue()


def _stereo_8k_wav_bytes(seconds: float = 1.0) -> bytes:
    """Stereo 8 kHz fixture: exercises mono-mix + resample paths."""
    import math

    sr = 8000
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        n = int(seconds * sr)
        for i in range(n):
            left = int(10000 * math.sin(2 * math.pi * 440 * i / sr))
            right = int(10000 * math.cos(2 * math.pi * 440 * i / sr))
            w.writeframes(struct.pack("<hh", left, right))
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
    """Loud tone fixture (used to prove the mock stays neutral on hot input)."""
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


class _FixedThreatModel:
    """Explicit test stub with fixed SYNTHETIC output (drives HIGH/CRITICAL).

    Replaces the old trick of playing a loud tone at the mock: the mock is
    intentionally non-diagnostic (always UNCERTAIN), so threat plumbing tests
    must state their model outputs instead of tickling an energy heuristic.
    """

    name = "fixed-threat-stub"

    def is_loaded(self) -> bool:
        return True

    def load(self) -> None:
        pass

    def preprocess(self, audio):
        return audio

    def predict(self, audio, sample_rate: int = 16000):
        import numpy as np

        from rafnet.soc.model_adapter import VoiceResult

        arr = np.asarray(audio, dtype=np.float32).reshape(-1)
        return VoiceResult(
            prediction="SYNTHETIC",
            confidence=0.95,
            prob_synthetic=0.95,
            prob_real=0.05,
            model_version="stub-0.0.0",
            processing_time_ms=1.0,
            audio_duration_seconds=float(len(arr) / sample_rate) if sample_rate else 0.0,
            mock=True,
        )


def _threat_client():
    from fastapi import FastAPI

    from rafnet.soc import routes
    from rafnet.soc.service import SOCService

    routes.override_soc_service(SOCService(_FixedThreatModel()))
    app = FastAPI()
    app.include_router(routes.router)
    from fastapi.testclient import TestClient

    return TestClient(app)


def test_mock_neutral_on_loud_input():
    """Regression: the dev mock must never fabricate SYNTHETIC for loud speech.

    (Previously loud natural speech read as SYNTHETIC/CRITICAL — the reported
    live-voice false positive for anyone running mock mode.)
    """
    client, _ = _mock_client()
    r = client.post("/api/analyze", files={"file": ("loud.wav", _loud_wav_bytes(), "audio/wav")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["prediction"] == "UNCERTAIN"
    assert body["prob_synthetic"] == 0.5
    assert body["mock"] is True


def test_mock_only_when_explicitly_requested(monkeypatch):
    """Mock must never be the default: without MOCK_INFERENCE=true the loader
    returns the real (possibly unloaded) adapter, never MockVoiceModel."""
    from rafnet.soc import model_adapter as ma

    monkeypatch.delenv("MOCK_INFERENCE", raising=False)
    model = ma.load_voice_model()
    assert model.name != "mock", "mock must be explicit opt-in, never default"
    monkeypatch.setenv("MOCK_INFERENCE", "true")
    assert ma.load_voice_model().name == "mock"


def test_health_reports_error_field():
    """Health must expose the load error (or null) — no silent degradation."""
    client, _ = _mock_client()
    r = client.get("/api/health")
    assert r.status_code == 200, r.text
    body = r.json()
    assert "error" in body
    assert body["mock_mode"] is True


def test_analyze_silence_ok():
    """Silence must analyze without crashing; heuristic layers report None."""
    client, _ = _mock_client()
    r = client.post("/api/analyze", files={"file": ("quiet.wav", _silence_wav_bytes(), "audio/wav")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["prediction"] == "UNCERTAIN"
    by_id = {layer["id"]: layer for layer in body["layers"]}
    assert by_id["spectral"]["score"] is None
    assert by_id["liveness"]["score"] is None
    assert body["verdict"]["label"] == "UNCERTAIN"


def test_analyze_stereo_8k_ok():
    """Non-native rate + stereo must decode (mono-mix) with correct duration."""
    client, _ = _mock_client()
    r = client.post("/api/analyze", files={"file": ("stereo.wav", _stereo_8k_wav_bytes(), "audio/wav")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["audio_duration_seconds"] == 1.0
    assert body["prediction"] == "UNCERTAIN"


def test_analyze_short_clip_ok():
    """Very short audio (0.1 s) must not crash the pipeline."""
    client, _ = _mock_client()
    r = client.post("/api/analyze", files={"file": ("blip.wav", _wav_bytes(seconds=0.1), "audio/wav")})
    assert r.status_code == 200, r.text


def test_analyze_model_unavailable_503():
    """A dead model must surface 503 (never a fabricated prediction)."""

    class _Down:
        name = "down"

        def is_loaded(self) -> bool:
            return False

        def load(self) -> None:
            pass

        def preprocess(self, audio):
            return audio

        def predict(self, audio, sample_rate: int = 16000):
            raise RuntimeError("Model not loaded: test outage")

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from rafnet.soc import routes
    from rafnet.soc.service import SOCService

    routes.override_soc_service(SOCService(_Down()))
    app = FastAPI()
    app.include_router(routes.router)
    client = TestClient(app)
    r = client.post("/api/analyze", files={"file": ("tone.wav", _wav_bytes(), "audio/wav")})
    assert r.status_code == 503
    s = client.post("/api/live/session")
    assert s.status_code == 200
    import numpy as np

    tone = (0.05 * np.sin(2 * np.pi * 440 * np.arange(16000) / 16000)).tolist()
    c = client.post(
        "/api/live/chunk",
        json={"session_id": s.json()["session_id"], "sample_rate": 16000, "samples": tone},
    )
    assert c.status_code == 503


def test_live_upload_resample_compatible():
    """Live (8 kHz chunk) and upload (8 kHz stereo file) paths must agree.

    Both funnel to 16 kHz mono float32 via the same linear resample before
    inference: chunk audio resampled in the route, file audio carrying its
    native rate to the model adapter. A shared sine must arrive identically.
    """
    import math

    import numpy as np

    from rafnet.soc import forensics, live
    from rafnet.soc.audio_io import decode_audio

    sr = 8000
    n = sr  # 1 s
    mono = [0.3 * math.sin(2 * math.pi * 440 * i / sr) for i in range(n)]
    # Live path: mono chunk resampled in the route.
    via_live = np.asarray(live.resample_to_16k(mono, sr), dtype=np.float32)
    # Upload path: stereo file decoded (mono-mixed, native rate kept).
    import io as _io
    import struct as _struct
    import wave as _wave

    buf = _io.BytesIO()
    with _wave.open(buf, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        for v in mono:
            q = int(max(-1.0, min(1.0, v)) * 32767)
            w.writeframes(_struct.pack("<hh", q, q))
    decoded = decode_audio(buf.getvalue(), "stereo.wav")
    assert decoded.sample_rate == sr
    via_upload = forensics.to_mono_16k(decoded.samples, decoded.sample_rate)
    assert len(via_live) == 16000 == len(via_upload)
    assert np.allclose(via_live, via_upload, atol=5e-3)


def test_threat_triage_workflow():
    client = _threat_client()
    missing = client.patch("/api/threats/thr-nope", json={"status": "acknowledged"})
    assert missing.status_code == 404
    r = client.post("/api/analyze", files={"file": ("loud.wav", _loud_wav_bytes(), "audio/wav")})
    assert r.status_code == 200, r.text
    assert r.json()["risk_level"] in ("HIGH", "CRITICAL")
    t = client.get("/api/threats")
    assert t.status_code == 200
    items = t.json()["items"]
    assert len(items) >= 1, "stub SYNTHETIC analysis must create a threat"
    threat_id = items[0]["id"]
    updated = client.patch(f"/api/threats/{threat_id}", json={"status": "acknowledged"})
    assert updated.status_code == 200
    assert updated.json()["status"] == "acknowledged"
    detail = client.get(f"/api/threats/{threat_id}")
    assert detail.json()["status"] == "acknowledged"
    bad = client.patch(f"/api/threats/{threat_id}", json={"status": "bogus"})
    # 422 from schema validation (Literal) or 400 from the service guard.
    assert bad.status_code in (400, 422)


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
    # Owned rows are hidden from anonymous callers (fail-closed).
    assert service.get_analysis(mine.id) is None
    anon = service.analyze_bytes(_wav_bytes(), "anon.wav")
    assert service.get_analysis(anon.id, user_id="user-2") is not None
    assert service.get_analysis(anon.id) is not None


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


def test_supabase_store_maps_schema_columns():
    """App-level records must round-trip through real schema columns.

    Guards the production failure where POSTing AnalyzeResponse-shaped keys
    (model_version, audio_duration_seconds, flat probabilities) 400s on
    PostgREST unknown columns or silently drops verdict/layers/reasons.
    """
    from rafnet.soc.store import SupabaseAnalysisStore

    fake = _FakePostgrest()
    try:
        store = SupabaseAnalysisStore(fake.url, "test-service-key")
        record = {
            "id": "full-1",
            "audio_name": "tone.wav",
            "prediction": "REAL",
            "confidence": 0.87,
            "risk_score": 5,
            "risk_level": "LOW",
            "model_version": "1.0.0",
            "processing_time_ms": 900.0,
            "audio_duration_seconds": 2.0,
            "prob_synthetic": 0.13,
            "prob_real": 0.87,
            "classical_probability": 0.05,
            "quantum_probability": 0.04,
            "fused_probability": 0.05,
            "calibrated_probability": 0.13,
            "gateway1_audio_weight": 1.0,
            "gateway1_network_weight": 0.0,
            "gateway2_classical_weight": 0.88,
            "gateway2_quantum_weight": 0.12,
            "mock": False,
            "created_at": "2026-01-03T00:00:00+00:00",
            "reasons": ["base=5 from REAL@0.87"],
            "layers": [{"id": "neural", "kind": "model"}],
            "verdict": {"label": "HUMAN_LIVE"},
            "user_id": "user-9",
        }
        store.save_analysis(record)
        posted = fake.rows[-1]
        # Only schema columns go over the wire (migration 001 + 002).
        assert posted["model_id"] == "1.0.0"
        assert posted["duration_seconds"] == 2.0
        assert posted["probs"]["prob_synthetic"] == 0.13
        assert posted["gateway"]["gateway2_classical_weight"] == 0.88
        assert posted["user_id"] == "user-9"
        assert posted["verdict"] == {"label": "HUMAN_LIVE"}
        assert "model_version" not in posted
        assert "audio_duration_seconds" not in posted
        assert "prob_synthetic" not in posted
        # Read path rebuilds the app-level shape.
        back = store.get_analysis("full-1")
        assert back is not None
        assert back["model_version"] == "1.0.0"
        assert back["audio_duration_seconds"] == 2.0
        assert back["prob_synthetic"] == 0.13
        assert back["gateway2_quantum_weight"] == 0.12
        assert back["reasons"] == ["base=5 from REAL@0.87"]
        assert back["verdict"] == {"label": "HUMAN_LIVE"}
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
    import pytest

    try:
        import rafnet.api.main as app_module
    except OSError as exc:
        # Torch is installed but its native libs are blocked by OS policy
        # (e.g. WinError 4551 Application Control) — the no-torch boot path
        # cannot be exercised in this environment; the SOC-only runtime
        # (system python, no torch) covers it instead.
        pytest.skip(f"torch runtime blocked by OS policy: {exc}")

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


def test_ws_live_sends_system_status_on_connect():
    """Regression: /ws/live must open with a system_status frame (frontend LIVE)."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from rafnet.soc import routes
    from rafnet.soc.model_adapter import MockVoiceModel
    from rafnet.soc.service import SOCService
    from rafnet.soc.websocket import ws_router

    routes.override_soc_service(SOCService(MockVoiceModel()))
    app = FastAPI()
    app.include_router(routes.router)
    app.include_router(ws_router)
    client = TestClient(app)
    with client.websocket_connect("/ws/live") as ws:
        first = ws.receive_json()
        assert first.get("event") == "system_status"
        assert first.get("status") == "connected"
        # connected frame must still follow (backward compat with useLiveEvents).
        seen_connected = False
        for _ in range(25):
            msg = ws.receive_json()
            if msg.get("event") == "connected":
                seen_connected = True
                break
        assert seen_connected


def test_ws_live_streams_mic_chunk_detection():
    """Mic path (POST /api/live/chunk, same as useMicChunks) emits WS detection."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from rafnet.soc import routes
    from rafnet.soc.model_adapter import MockVoiceModel
    from rafnet.soc.service import SOCService
    from rafnet.soc.websocket import ws_router

    routes.override_soc_service(SOCService(MockVoiceModel()))
    app = FastAPI()
    app.include_router(routes.router)
    app.include_router(ws_router)
    client = TestClient(app)
    with client.websocket_connect("/ws/live") as ws:
        # Drain system_status + replay + connected.
        for _ in range(30):
            msg = ws.receive_json()
            if msg.get("event") == "connected":
                break
        sess = client.post("/api/live/session").json()["session_id"]
        # One full 1.0 s window of silence triggers analyzed + WS detection.
        r = client.post(
            "/api/live/chunk",
            json={"session_id": sess, "sample_rate": 16000, "samples": [0.0] * 16000},
        )
        assert r.status_code == 200, r.text
        assert r.json()["status"] == "analyzed"
        seen = set()
        for _ in range(10):
            msg = ws.receive_json()
            seen.add(msg.get("event"))
            if "detection" in seen:
                break
        assert "detection" in seen


def test_clear_analyses_scoped_and_reported():
    client, routes = _mock_client()
    from rafnet.soc.service import SOCService

    svc = routes.get_soc_service()
    assert isinstance(svc, SOCService)
    svc.analyze_bytes(_wav_bytes(), "anon.wav")
    svc.analyze_bytes(_wav_bytes(), "mine.wav", user_id="alice")
    assert client.get("/api/history").json()["total"] == 2
    # Anonymous clear removes legacy anonymous rows only.
    r = client.delete("/api/analyses")
    assert r.status_code == 200
    assert r.json() == {"deleted": 1}
    remaining = client.get("/api/history").json()
    assert remaining["total"] == 1
    assert remaining["items"][0]["audio_name"] == "mine.wav"
    # Threat triage overrides for deleted rows are dropped (no crash).
    assert client.get("/api/threats").status_code == 200


def test_threat_triage_scoped_to_owner():
    """Triage of another user's threat must behave as not-found."""
    from rafnet.soc.service import SOCService

    svc = SOCService(_FixedThreatModel())
    rec = svc.analyze_bytes(_loud_wav_bytes(), "loud.wav", user_id="alice")
    assert rec.risk_level in ("HIGH", "CRITICAL")
    threats = svc.list_threats(user_id="alice")
    assert len(threats) >= 1
    threat_id = threats[0]["id"]
    assert svc.set_threat_status(threat_id, "acknowledged", user_id="bob") is None
    updated = svc.set_threat_status(threat_id, "acknowledged", user_id="alice")
    assert updated is not None and updated["status"] == "acknowledged"


def _load_eval_model_script():
    import importlib.util

    path = "tools/evaluate_model.py"
    spec = importlib.util.spec_from_file_location("rafnet_eval_model", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_eval_metrics_pure_helpers():
    ev = _load_eval_model_script()
    y = [0, 0, 1, 1]
    s = [0.1, 0.2, 0.87, 0.9]
    assert ev.strict_accuracy(y, s, 0.3611, 0.8611) == 1.0
    assert ev.compute_auc(y, s) == 1.0
    eer = ev.compute_eer(y, s)
    assert eer["eer"] == 0.0
    b = ev.binary_metrics_at_05(y, s)
    assert (b["accuracy"], b["precision"], b["recall"]) == (1.0, 1.0, 1.0)
    # UNCERTAIN counts as wrong under the stated tradeoff.
    assert ev.strict_accuracy([0, 1], [0.5, 0.5], 0.3611, 0.8611) == 0.0
    # Single-class EER/AUC are null, never invented.
    assert ev.compute_eer([0, 0], [0.1, 0.2])["eer"] is None
    assert ev.compute_auc([0, 0], [0.1, 0.2]) is None


def test_eval_calibration_grid_picks_separating_band():
    ev = _load_eval_model_script()
    y = [0, 0, 0, 1, 1, 1]
    s = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    best = ev.grid_calibrate(y, s)
    assert best["tau_low"] is not None and best["tau_high"] is not None
    assert best["tau_low"] < best["tau_high"]
    assert best["accuracy"] == 1.0


def test_live_aggregation_median_and_hysteresis():
    from collections import deque

    from rafnet.soc.live import (
        AGG_WINDOW,
        aggregate_probs,
        update_aggregated_prediction,
    )

    # Median of last 5 suppresses a single flicker chunk.
    probs = deque([0.1, 0.1, 0.95, 0.1, 0.1], maxlen=8)
    agg = aggregate_probs(probs, 0.3611, 0.8611, window=5)
    assert agg["p_median"] == 0.1
    assert agg["label"] == "REAL"
    assert agg["chunks_used"] == 5
    # Hysteresis: a lone SYNTHETIC vote must not flip an established REAL.
    labels: deque = deque(["REAL", "REAL", "REAL", "REAL"], maxlen=8)
    assert update_aggregated_prediction(labels, "REAL", "SYNTHETIC") == "REAL"
    labels2: deque = deque(["SYNTHETIC", "SYNTHETIC"], maxlen=8)
    assert update_aggregated_prediction(labels2, "REAL", "SYNTHETIC") == "SYNTHETIC"
    assert AGG_WINDOW == 5


def test_decision_thresholds_roundtrip(tmp_path, monkeypatch):
    import json

    from rafnet.soc import thresholds as thr

    cfg = tmp_path / "thresholds.json"
    cfg.write_text(json.dumps({"tau_low": 0.2, "tau_high": 0.8, "calibrated": True}))
    monkeypatch.setenv("DECISION_THRESHOLDS_PATH", str(cfg))
    low, high, meta = thr.load_decision_thresholds()
    assert (low, high) == (0.2, 0.8)
    assert meta["calibrated"] is True


def test_live_chunk_reports_aggregated_fields():
    import numpy as np

    client, _ = _mock_client()
    sess = client.post("/api/live/session").json()["session_id"]
    tone = (0.05 * np.sin(2 * np.pi * 440 * np.arange(16000) / 16000)).tolist()
    r = client.post("/api/live/chunk", json={"session_id": sess, "sample_rate": 16000, "samples": tone})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "analyzed"
    assert body["aggregated_prediction"] in ("REAL", "UNCERTAIN", "SYNTHETIC")
    assert body["aggregated_chunks"] >= 1
    assert body["aggregated_window"] == 5
    assert "tau_low" in body["thresholds"]
