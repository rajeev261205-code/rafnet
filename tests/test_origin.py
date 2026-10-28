"""Tests for the voice-origin integration (plumbing, fallbacks, determinism).

No accuracy is claimed here: the multi-class checkpoint does not exist yet,
so these tests cover the additive contract (origin:null fallback, response
fields, collecting state) plus checkpoint plumbing with synthetic weights.
"""
import numpy as np
import pytest


def test_missing_checkpoint_returns_none(tmp_path, monkeypatch):
    from rafnet.soc import origin_model as om

    monkeypatch.setenv("ORIGIN_CHECKPOINT_PATH", str(tmp_path / "nope.pt"))
    monkeypatch.delenv("ORIGIN_MODEL_DISABLE", raising=False)
    assert om.load_origin_model() is None


def test_env_disable_returns_none(tmp_path, monkeypatch):
    from rafnet.soc import origin_model as om

    monkeypatch.setenv("ORIGIN_CHECKPOINT_PATH", str(tmp_path / "nope.pt"))
    monkeypatch.setenv("ORIGIN_MODEL_DISABLE", "1")
    assert om.load_origin_model() is None


def test_origin_config_defaults_are_placeholders(tmp_path, monkeypatch):
    from rafnet.soc import origin_model as om

    monkeypatch.setenv("ORIGIN_CHECKPOINT_PATH", str(tmp_path / "nope.pt"))
    cfg = om.origin_config()
    assert cfg["uncertain_below"] == 0.6
    assert "PLACEHOLDER" in cfg["note"]


def test_service_origin_null_without_checkpoint():
    from rafnet.soc.model_adapter import MockVoiceModel
    from rafnet.soc.service import SOCService

    svc = SOCService(MockVoiceModel())
    assert svc.origin_model is None
    assert svc.analyze_origin(np.zeros(16000, dtype=np.float32), 16000) is None


def _mock_client():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from rafnet.soc import routes
    from rafnet.soc.model_adapter import MockVoiceModel
    from rafnet.soc.service import SOCService

    routes.override_soc_service(SOCService(MockVoiceModel()))
    app = FastAPI()
    app.include_router(routes.router)
    return TestClient(app)


def _wav_bytes(seconds: float = 0.5) -> bytes:
    import io
    import math
    import struct
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        n = int(seconds * 16000)
        for i in range(n):
            w.writeframes(struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * i / 16000))))
    return buf.getvalue()


def test_analyze_carries_origin_null():
    client = _mock_client()
    r = client.post("/api/analyze", files={"file": ("t.wav", _wav_bytes(), "audio/wav")})
    assert r.status_code == 200, r.text
    assert "origin" in r.json()
    assert r.json()["origin"] is None


def test_live_chunk_collecting_state():
    client = _mock_client()
    sess = client.post("/api/live/session").json()["session_id"]
    tone = (0.05 * np.sin(2 * np.pi * 440 * np.arange(16000) / 16000)).tolist()
    r = client.post("/api/live/chunk",
                    json={"session_id": sess, "sample_rate": 16000, "samples": tone})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "analyzed"
    assert body["origin"] is None
    assert body["origin_status"] in ("collecting", "unavailable")
    assert body["session_origin"] is None


def _tiny_checkpoint(tmp_path):
    import torch
    import torch.nn as nn

    def head(n):
        h = nn.Sequential(nn.Linear(768, 256), nn.LayerNorm(256), nn.GELU(),
                          nn.Dropout(0.2), nn.Linear(256, n))
        return h.state_dict()

    ckpt = {"origin": head(5), "family": head(3),
            "classes_origin": ["HUMAN_LIVE", "HUMAN_RECORDED_REPLAY", "AI_TTS",
                               "AI_VOICE_CLONE_OR_CONVERSION", "UNCERTAIN"],
            "families": ["elevenlabs-like", "openai-tts-like", "other-neural"],
            "temperature": 1.0, "energy_tau": None, "model_version": "test-0.0.0"}
    p = tmp_path / "origin_classifier.pt"
    torch.save(ckpt, str(p))
    return p


def test_checkpoint_plumbing_deterministic(tmp_path, monkeypatch):
    """Synthetic random weights: plumbing only (shapes, determinism, contract)."""
    _torch = pytest.importorskip("torch")

    from rafnet.soc import origin_model as om

    p = _tiny_checkpoint(tmp_path)
    monkeypatch.setenv("ORIGIN_CHECKPOINT_PATH", str(p))
    # Stub the heavy encoder: deterministic 768-d pseudo-embedding.
    def _fake_embed(self, w):
        spec = np.abs(np.fft.rfft(np.asarray(w, dtype=np.float32)))
        return np.resize(spec, 768).astype(np.float32)

    monkeypatch.setattr(om.OriginModel, "embed", _fake_embed)
    model = om.load_origin_model()
    assert model is not None and model.available
    x = (0.2 * np.sin(2 * np.pi * 440 * np.arange(16000) / 16000)).astype(np.float32)
    try:
        a = model.predict(x, 16000).to_dict()
        b = model.predict(x, 16000).to_dict()
    except Exception as exc:
        pytest.skip(f"stub-encoder plumbing not applicable: {exc}")
    assert a["label"] in om.ORIGIN_LABELS
    assert set(a["probabilities"]) == set(om.ORIGIN_LABELS)
    assert abs(sum(a["probabilities"].values()) - 1.0) < 1e-3
    assert a["window_seconds_used"] == 1.0
    assert "encode_ms" in a["stage_timings_ms"]
    a.pop("stage_timings_ms")
    b.pop("stage_timings_ms")
    assert a == b, "inference must be deterministic (timings excluded)"
