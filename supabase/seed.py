"""Seed Supabase models/experiments from REAL report files (dry-run by default).

Reads (verbatim, never fabricated):
- experiments/phase19_report.json
- experiments/rafnet_ablation_smoke_*/results.json (up to 20)
- experiments/checkpoints/* (file sizes for model metadata)

Upserts `models` row `rafnet-1.0.0-phase19` + one `experiments` row per report
via Supabase REST (needs SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY).
Without env, prints the payload (dry-run) and exits 0.

Usage:
  python supabase/seed.py [--dry-run] [--apply]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = Path(os.getenv("EXPERIMENTS_DIR", str(ROOT / "experiments")))


def collect() -> tuple[dict, list[dict]]:
    ckpt = ROOT / os.getenv("MODEL_CHECKPOINT_DIR", "experiments/checkpoints")
    sizes: dict[str, int] = {}
    if ckpt.is_dir():
        for p in sorted(ckpt.glob("*")):
            if p.is_file():
                sizes[p.name] = p.stat().st_size
    model = {
        "id": "rafnet-1.0.0-phase19",
        "name": "RAFNet",
        "version": "1.0.0-phase19",
        "framework": "PyTorch + scikit-learn",
        "description": "WavLM + Handcrafted -> Gateway1 -> Classical/Quantum -> Gateway2 -> Calibration -> DecisionGate. Synthetic-only training; see AI_MODEL_SPEC.md.",
        "status": "experimental",
        "metadata": {"sample_rate": 16000, "tau_low": 0.361, "tau_high": 0.861, "checkpoints": sizes},
    }
    experiments: list[dict] = []
    phase19 = EXP / "phase19_report.json"
    if phase19.is_file():
        data = json.loads(phase19.read_text())
        experiments.append({
            "model_id": model["id"],
            "name": "phase19 — full system report",
            "metrics": {"status": data.get("status")},
            "configuration": {"dataset": data.get("dataset")},
            "notes": "Verbatim from phase19_report.json; see AI_MODEL_SPEC.md for caveats.",
        })
    for results in sorted(EXP.glob("rafnet_ablation_smoke_*/results.json"))[:20]:
        try:
            data = json.loads(results.read_text())
            experiments.append({
                "model_id": model["id"],
                "name": f"ablation smoke — {results.parent.name}",
                "metrics": data if isinstance(data, dict) else {"results": data},
                "configuration": {},
                "notes": "Verbatim from ablation results.json.",
            })
        except Exception as exc:
            print(f"skip {results}: {exc}", file=sys.stderr)
    return model, experiments


def rest(url: str, key: str, table: str, body: dict | list, upsert_key: str | None = None) -> str:
    query = f"?on_conflict={upsert_key}" if upsert_key else ""
    req = urllib.request.Request(
        f"{url}/rest/v1/{table}{query}",
        data=json.dumps(body).encode(),
        method="POST",
        headers={
            "apikey": key, "Authorization": f"Bearer {key}",
            "Content-Type": "application/json", "Prefer": "resolution=merge-duplicates,return=representation",
        },
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        return resp.read().decode()[:500]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="Write to Supabase (needs env)")
    args = ap.parse_args()
    model, experiments = collect()
    print(f"model: {model['id']} checkpoints={model['metadata']['checkpoints']}")
    print(f"experiments: {len(experiments)}")
    if not args.apply:
        print("dry-run: pass --apply with SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY to write.")
        return 0
    url = os.getenv("SUPABASE_URL", "").rstrip("/")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
    if not url or not key:
        print("ERROR: SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY required for --apply", file=sys.stderr)
        return 2
    print("models:", rest(url, key, "models", model, "id"))
    for exp in experiments:
        print("experiments:", rest(url, key, "experiments", exp)[:200])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
