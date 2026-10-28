"""Phase 5 verification: POST samples/labels.csv to a running backend.

Checks (all measured, saved to reports/verify_before_after.json):
- ensemble fields present on every response (no silent None except stated)
- NO single-shot upload yields CRITICAL (false-critical regression gate)
- before (reports/baseline.json RAFNet p/prediction) vs after (API now)
Usage: backend running (start-local.ps1 or uvicorn), then run this.
"""
import csv, json, sys, time, urllib.request, uuid
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"

def post(path: Path):
    boundary = f"----v5{uuid.uuid4().hex}"
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{path.name}\"\r\n"
            "Content-Type: application/octet-stream\r\n\r\n").encode() + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    for _ in range(6):
        try:
            req = urllib.request.Request(f"{BASE}/api/analyze", data=body, method="POST",
                                         headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            if "429" in str(e):
                time.sleep(20); continue
            raise
    raise RuntimeError("unreachable")

def main() -> int:
    rows = list(csv.DictReader(open(ROOT / "samples" / "labels.csv", encoding="utf-8")))
    base = {f["file"]: f for f in json.loads((ROOT / "reports" / "baseline.json").read_text(encoding="utf-8"))["files"] if "p_synthetic" in f}
    out, crits, missing_ens = [], [], []
    for r in rows:
        p = ROOT / r["file"].replace("\\", "/")
        b = post(p)
        before = base.get(r["file"], {})
        rec = {"file": r["file"], "label": r["label"],
               "before_p": before.get("p_synthetic"), "before_pred": before.get("prediction"),
               "after_pred": b["prediction"], "after_conf": b["confidence"],
               "after_risk": b["risk_level"], "after_score": b["risk_score"],
               "rafnet_p": b["prob_synthetic"],
               "p_mo": b.get("p_antispoof_mo"), "p_bi": b.get("p_antispoof_bisher"),
               "p_ens": b.get("p_ensemble"), "ens_verdict": b.get("ensemble_verdict"),
               "agreement": b.get("ensemble_agreement"), "ens_alert": b.get("ensemble_alert")}
        out.append(rec)
        print(f"{r['label']:>4} {p.name[:32]:32s} {str(before.get('prediction')):>9s} -> {b['prediction']:>9s} "
              f"risk {b['risk_level']:>8s} ens {b.get('ensemble_verdict')}/{b.get('ensemble_alert')} "
              f"(mo={b.get('p_antispoof_mo')} bi={b.get('p_antispoof_bisher')} p={b.get('p_ensemble')} agr={b.get('ensemble_agreement')})")
        if b["risk_level"] == "CRITICAL":
            crits.append(r["file"])
        if b.get("p_ensemble") is None:
            missing_ens.append(r["file"])
        time.sleep(1)
    rep = {"generated_at": datetime.now(timezone.utc).isoformat(), "base_url": BASE,
           "n": len(out), "critical_singleshot": crits, "missing_ensemble": missing_ens,
           "gate1_no_critical_singleshot": len(crits) == 0,
           "gate2_ensemble_present": len(missing_ens) == 0, "rows": out}
    (ROOT / "reports" / "verify_before_after.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"\nCRITICAL on single-shot: {crits or 'NONE (gate passes)'}")
    print(f"missing ensemble: {missing_ens or 'NONE (gate passes)'}")
    print("wrote reports/verify_before_after.json")
    return 0 if (not crits and not missing_ens) else 1
if __name__ == "__main__": raise SystemExit(main())
