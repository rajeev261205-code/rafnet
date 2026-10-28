"""Live-server E2E probe — exercises the real HTTP surface, asserts 200s.

Runs against a booted backend (default http://localhost:8000, override with
BASE_URL env). Uses the mock or real model transparently: asserts the response
CONTRACT (prediction/confidence/risk/mock fields, status codes), never specific
model values. Fails loudly on connection errors.

Usage:
  MOCK_INFERENCE=true uvicorn rafnet.api.main:app --port 8000 &  # from src/
  BASE_URL=http://localhost:8000 python scripts/e2e_probe.py
"""
from __future__ import annotations

import io
import json
import math
import os
import struct
import urllib.request
import wave

BASE = os.getenv("BASE_URL", "http://localhost:8000")
FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(("PASS " if condition else "FAIL ") + name + (f" — {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


def get(path: str):
    with urllib.request.urlopen(BASE + path, timeout=20) as r:
        return r.status, json.loads(r.read().decode())


def post_json(path: str, obj: dict):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(obj).encode(), method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.status, json.loads(r.read().decode())


def patch_json(path: str, obj: dict):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(obj).encode(), method="PATCH",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.status, json.loads(r.read().decode())


def _tone_wav(amplitude: int, seconds: float = 0.5, sr: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        n = int(seconds * sr)
        for i in range(n):
            w.writeframes(struct.pack("<h", int(amplitude * math.sin(2 * math.pi * 440 * i / sr))))
    return buf.getvalue()


def post_wav(amplitude: int = 8000):
    data = _tone_wav(amplitude)
    boundary = "BOUNDARY123"
    head = (
        "--" + boundary + "\r\n"
        'Content-Disposition: form-data; name="file"; filename="tone.wav"\r\n'
        "Content-Type: audio/wav\r\n\r\n"
    ).encode()
    tail = ("\r\n--" + boundary + "--\r\n").encode()
    req = urllib.request.Request(
        BASE + "/api/analyze", data=head + data + tail, method="POST",
        headers={"Content-Type": "multipart/form-data; boundary=" + boundary},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.status, json.loads(r.read().decode())


def main() -> int:
    print(f"E2E probe against {BASE}")
    try:
        s, health = get("/api/health")
        check("health 200", s == 200, f"status={s}")
        check("health has model_loaded", isinstance(health.get("model_loaded"), bool))
        print(f"  model={health.get('model')} mock_mode={health.get('mock_mode')}")
        mock_mode = bool(health.get("mock_mode"))
    except Exception as exc:
        check("backend reachable", False, str(exc))
        return 1

    s, _ = get("/healthz")
    check("healthz 200", s == 200, f"status={s}")

    s, created = post_wav()
    check("analyze 200", s == 200, f"status={s}")
    check("prediction valid", created.get("prediction") in ("REAL", "UNCERTAIN", "SYNTHETIC"))
    check("confidence in [0,1]", 0.0 <= float(created.get("confidence", -1)) <= 1.0)
    check("risk contracted", created.get("risk_level") in ("LOW", "MEDIUM", "HIGH", "CRITICAL"))
    aid = created["id"]

    s, _ = get("/api/analysis/" + aid)
    check("detail 200", s == 200, f"status={s}")
    s, hist = get("/api/history?limit=5")
    check("history 200 + total>=1", s == 200 and hist.get("total", 0) >= 1)
    s, threats = get("/api/threats")
    check("threats 200", s == 200)
    s, calls = get("/api/calls")
    check("calls 200 + >=1 session", s == 200 and len(calls.get("items", [])) >= 1)
    s, models = get("/api/models")
    check("models RAFNet", s == 200 and models.get("name") == "RAFNet")
    s, exps = get("/api/experiments")
    check("experiments non-empty", s == 200 and len(exps.get("items", [])) >= 1)

    s, ls = post_json("/api/live/session", {})
    check("live session 200", s == 200)
    chunk = [0.05 * math.sin(2 * math.pi * 440 * i / 16000) for i in range(16000)]
    s, cr = post_json("/api/live/chunk", {"session_id": ls["session_id"], "sample_rate": 16000, "samples": chunk})
    check("live chunk analyzed", s == 200 and cr.get("status") == "analyzed", str(cr)[:120])

    # Triage path: HIGH/CRITICAL threat -> acknowledge -> verify.
    # The mock is intentionally neutral (always UNCERTAIN, never a threat),
    # so on mock backends assert that honest neutral result and skip triage
    # (threat plumbing is covered by pytest with explicit stub models).
    s, loud = post_wav(amplitude=20000)
    check("loud analyze 200", s == 200, f"status={s}")
    if mock_mode:
        check("mock stays neutral on loud input",
              loud.get("prediction") == "UNCERTAIN" and loud.get("mock") is True,
              str({k: loud.get(k) for k in ("prediction", "mock")}))
        print("SKIP triage on mock backend (neutral mock makes no threats; see pytest)")
        loud = None
    else:
        check("loud is HIGH/CRITICAL", loud.get("risk_level") in ("HIGH", "CRITICAL"), str(loud.get("risk_level")))
    if loud is not None:
        s, threats = get("/api/threats")
        check("threats include loud analysis", s == 200 and any(
            t.get("analysis_id") == loud["id"] for t in threats.get("items", [])), f"status={s}")
        tid = next(t["id"] for t in threats["items"] if t["analysis_id"] == loud["id"])
        s, triaged = patch_json(f"/api/threats/{tid}", {"status": "acknowledged"})
        check("triage PATCH 200", s == 200 and triaged.get("status") == "acknowledged", f"status={s}")
        s, got = get(f"/api/threats/{tid}")
        check("triage persisted", s == 200 and got.get("status") == "acknowledged")

    if FAILURES:
        print(f"\n{len(FAILURES)} FAILURES: {FAILURES}")
        return 1
    print("\nE2E OK — all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
