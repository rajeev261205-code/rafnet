"""Phase 5 live-window verification (running backend).

- 6 x 1 s chunks of SAPI fake -> ensemble window fills; CRITICAL allowed
  only from chunk 5 with hot window + agreement.
- 6 x 1 s chunks of clean real -> alert NEVER CRITICAL.
Saves reports/verify_live.json.
"""
import json, sys, urllib.request
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, soundfile as sf
ROOT = Path(__file__).resolve().parents[1]
BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"

def rpc(path, payload=None, method="GET"):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(f"{BASE}{path}", data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode())

def load16(p, start_s=0, dur_s=1):
    a, sr = sf.read(str(p), dtype="float32"); a = np.asarray(a, dtype=np.float32).ravel()
    if sr != 16000:
        a = np.interp(np.linspace(0, 1, int(len(a) / sr * 16000)), np.linspace(0, 1, len(a)), a).astype(np.float32)
    seg = a[start_s * 16000:(start_s + dur_s) * 16000]
    if len(seg) < 16000:
        seg = np.pad(seg, (0, 16000 - len(seg)))
    return seg.tolist()

def run(path, tag, n_chunks=6):
    sid = rpc("/api/live/session", {}, "POST")["session_id"]
    rows = []
    for i in range(n_chunks):
        seg = load16(path, start_s=i, dur_s=1)
        r = rpc("/api/live/chunk", {"session_id": sid, "sample_rate": 16000, "samples": seg}, "POST")
        e = r.get("ensemble") or {}
        rows.append({"chunk": i + 1, "risk": r.get("risk_level"), "pred": r.get("prediction"),
                     "p_ens": e.get("p_ensemble"), "alert": e.get("alert"),
                     "window": f"{e.get('window_n')}/{e.get('window_need')}"})
        print(f"{tag} chunk{i+1}: risk={r.get('risk_level')} pred={r.get('prediction')} "
              f"ens={e.get('verdict')}/{e.get('alert')} p={e.get('p_ensemble')} win={e.get('window_n')}/{e.get('window_need')}")
    return rows

def main() -> int:
    fake = run(ROOT / "samples/fake/sapi_zira_2.wav", "FAKE", 6)
    real = run(ROOT / "samples/real/4446-2273-0021.flac", "REAL", 6)
    fake_crit = [r for r in fake if r["risk"] == "CRITICAL"]
    real_crit = [r for r in real if r["risk"] == "CRITICAL"]
    early_crit = [r for r in fake if r["risk"] == "CRITICAL" and r["chunk"] < 5]
    ok = (not real_crit) and (not early_crit)
    rep = {"generated_at": datetime.now(timezone.utc).isoformat(), "fake_chunks": fake, "real_chunks": real,
           "real_never_critical": not real_crit, "no_early_critical_before_window": not early_crit,
           "fake_critical_chunks": [r["chunk"] for r in fake_crit], "pass": ok}
    (ROOT / "reports" / "verify_live.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"\nreal CRITICAL: {len(real_crit)} (must be 0); early CRITICAL (chunk<5): {len(early_crit)} (must be 0); "
          f"fake CRITICAL chunks: {[r['chunk'] for r in fake_crit]}; PASS={ok}")
    print("wrote reports/verify_live.json")
    return 0 if ok else 1
if __name__ == "__main__": raise SystemExit(main())
