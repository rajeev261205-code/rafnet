"""Consistency check: same audio must give the same result every call + restart.

Procedure (per task):
  - takes 3 audio files (human, TTS, any),
  - calls POST /api/analyze 10 times each,
  - restarts the backend twice (run start-local.ps1 again each time) and repeats,
  - prints a table of scores per run and the max difference.
  - any difference above 1e-4 in prob_synthetic is a bug.

Thresholds are never touched here. Timing/id/created_at fields are expected
to vary and are excluded from the comparison (only prediction scores count).

Usage (repo root, backend running via start-local.ps1):
  .venv/Scripts/python tools/consistency_check.py \
      samples/real/260-123286-0012.flac \
      samples/fake/sapi_david_2.wav \
      samples/win_tts/sapi_zira_0.wav \
      [--base-url http://127.0.0.1:8000] [--repeats 10] [--restarts 2]
      [--tolerance 1e-4] [--restart-cmd "..."] [--yes]

Restart handling: after each round except the last, the tool either runs
--restart-cmd (e.g. 'powershell -ExecutionPolicy Bypass -File start-local.ps1')
or, by default, prints the manual step and waits for Enter (use --yes to
continue without restarting, e.g. for a quick single-round check).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

DEFAULT_FILES = [
    "samples/real/260-123286-0012.flac",
    "samples/fake/sapi_david_2.wav",
    "samples/win_tts/sapi_zira_0.wav",
]


def get_health(base_url: str, timeout: float = 10.0) -> dict:
    req = urllib.request.Request(f"{base_url}/api/health", method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def wait_for_health(base_url: str, deadline_s: float = 300.0) -> dict:
    start = time.monotonic()
    last: Exception | None = None
    while time.monotonic() - start < deadline_s:
        try:
            return get_health(base_url)
        except Exception as exc:
            last = exc
            time.sleep(5)
    raise RuntimeError(f"backend not healthy at {base_url}: {last}")


def post_analyze(base_url: str, path: Path, timeout: float = 120.0) -> dict:
    import uuid

    boundary = f"----consistency{uuid.uuid4().hex}"
    data = path.read_bytes()
    filename = path.name
    # Minimal multipart/form-data (stdlib only so the tool runs anywhere).
    head = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n"
    ).encode()
    tail = f"\r\n--{boundary}--\r\n".encode()
    body = head + data + tail
    for attempt in range(6):
        req = urllib.request.Request(
            f"{base_url}/api/analyze",
            data=body,
            method="POST",
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as exc:
            msg = str(exc)
            if "429" in msg and attempt < 5:
                time.sleep(15)
                continue
            raise
    raise RuntimeError("unreachable")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="*", default=DEFAULT_FILES)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--restarts", type=int, default=2)
    ap.add_argument("--tolerance", type=float, default=1e-4)
    ap.add_argument("--restart-cmd", default=None,
                    help="shell command to restart the backend between rounds")
    ap.add_argument("--yes", action="store_true",
                    help="do not prompt between rounds (for CI)")
    ap.add_argument("--wait-health-s", type=float, default=300.0)
    args = ap.parse_args()

    paths = [Path(f) for f in args.files[:3]]
    if len(paths) != 3:
        print("need exactly 3 audio files", file=sys.stderr)
        return 2
    for p in paths:
        if not p.is_file():
            print(f"missing file: {p}", file=sys.stderr)
            return 2

    rounds = 1 + max(0, args.restarts)
    # scores[file][round] = list of prob_synthetic per repeat
    scores: dict[str, list[list[float]]] = {p.name: [] for p in paths}
    preds: dict[str, list[list[str]]] = {p.name: [] for p in paths}
    # Phase 5: same determinism contract for the stacked ensemble score.
    ens_scores: dict[str, list[list[float | None]]] = {p.name: [] for p in paths}
    overall_ok = True

    for rnd in range(rounds):
        if rnd > 0:
            if args.restart_cmd:
                print(f"\n=== restarting backend (round {rnd + 1}/{rounds}): {args.restart_cmd} ===")
                subprocess.run(args.restart_cmd, shell=True, check=False)
                print("waiting for /api/health ...")
                wait_for_health(args.base_url, args.wait_health_s)
            elif not args.yes:
                print(f"\n=== round {rnd + 1}/{rounds}: restart the backend now ===")
                print("Run start-local.ps1 again, wait for health, then press Enter ...")
                try:
                    input()
                except EOFError:
                    pass
                wait_for_health(args.base_url, args.wait_health_s)
            else:
                print(f"\n=== round {rnd + 1}/{rounds} (--yes: no restart performed) ===")
        else:
            print(f"=== round 1/{rounds} ===")

        try:
            health = get_health(args.base_url)
        except Exception as exc:
            print(f"health check failed: {exc}", file=sys.stderr)
            return 1
        fp = health.get("fingerprint", {})
        print(f"health: model_loaded={health.get('model_loaded')} mock={health.get('mock_mode')} "
              f"model={health.get('model')} error={health.get('error')}")
        if isinstance(fp, dict) and fp:
            vers = fp.get("versions", {})
            print(f"fingerprint: py={vers.get('python')} torch={vers.get('torch')} "
                  f"numpy={vers.get('numpy')} sklearn={vers.get('scikit-learn')} "
                  f"device={fp.get('device')} seed={fp.get('seed')}")

        for p in paths:
            run_scores: list[float] = []
            run_preds: list[str] = []
            run_ens: list[float | None] = []
            for i in range(args.repeats):
                body = post_analyze(args.base_url, p)
                run_scores.append(float(body["prob_synthetic"]))
                run_preds.append(str(body["prediction"]))
                run_ens.append(None if body.get("p_ensemble") is None else float(body["p_ensemble"]))
                time.sleep(0.2)
            scores[p.name].append(run_scores)
            preds[p.name].append(run_preds)
            ens_scores[p.name].append(run_ens)

    print("\n================ consistency table (prob_synthetic per repeat) ================")
    global_max_diff = 0.0
    for p in paths:
        print(f"\n-- {p} --")
        all_vals: list[float] = []
        for rnd, run in enumerate(scores[p.name]):
            all_vals.extend(run)
            vals = " ".join(f"{v:.6f}" for v in run)
            print(f"  round{rnd + 1}: {vals}")
        mx, mn = max(all_vals), min(all_vals)
        diff = mx - mn
        global_max_diff = max(global_max_diff, diff)
        pred_set = sorted({pr for run in preds[p.name] for pr in run})
        status = "OK" if diff <= args.tolerance else "BUG"
        if diff > args.tolerance:
            overall_ok = False
        print(f"  min={mn:.6f} max={mx:.6f} maxdiff={diff:.6e} predictions={pred_set} [{status}]")
        ev: list[float] = [v for run in ens_scores[p.name] for v in run if v is not None]
        if len(ev) < sum(len(run) for run in ens_scores[p.name]):
            print(f"  ensemble: MISSING on some repeats [BUG]")
            overall_ok = False
        elif ev:
            ediff = max(ev) - min(ev)
            global_max_diff = max(global_max_diff, ediff)
            estatus = "OK" if ediff <= args.tolerance else "BUG"
            if ediff > args.tolerance:
                overall_ok = False
            print(f"  ensemble p_ensemble maxdiff={ediff:.6e} [{estatus}]")

    print(f"\nGLOBAL maxdiff={global_max_diff:.6e} tolerance={args.tolerance:.0e}")
    if overall_ok:
        print("PASS: same audio gives the same result on every call and restart.")
        return 0
    print("FAIL: difference above tolerance — find and fix the nondeterminism.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
