"""Verify a provisioned Supabase project for RAFNet (stdlib only).

Checks (no writes except an optional seed dry-run read):
1. REST reachable with the service_role key.
2. All 7 SOC tables exist (from supabase/migrations/001_soc_schema.sql).
3. `models` seed row `rafnet-1.0.0-phase19` present.
4. RLS enforced: the ANON key must see ZERO analysis_results rows
   (owner-only policies, no anon access) — an empty list (not an error)
   proves RLS is on; rows visible to anon means RLS is OFF (fail).

Usage:
  SUPABASE_URL=https://xyz.supabase.co SUPABASE_SERVICE_ROLE_KEY=eyJ... \\
  SUPABASE_ANON_KEY=eyJ... python supabase/verify.py
Exit 0 = ready for backend wiring; non-zero = exact failure printed.
"""
from __future__ import annotations

import json
import os
import urllib.request

TABLES = ["profiles", "analysis_results", "calls", "threats", "models",
          "experiments", "analysis_events"]
FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f" — {detail}"))
    if not ok:
        FAILURES.append(name)


def rest(url: str, key: str, path: str, timeout: float = 15.0):
    req = urllib.request.Request(
        f"{url}/rest/v1/{path}",
        method="GET",
        headers={"apikey": key, "Authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        text = resp.read().decode()
        return resp.status, (json.loads(text) if text else None)


def main() -> int:
    url = os.getenv("SUPABASE_URL", "").rstrip("/")
    svc = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
    anon = os.getenv("SUPABASE_ANON_KEY", "")
    if not url or not svc:
        print("ERROR: set SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY (+ ANON for the RLS check)")
        return 2
    print(f"Verifying {url}")

    try:
        s, _ = rest(url, svc, "analysis_results?select=id&limit=1")
        check("REST reachable (service key)", s == 200, f"status={s}")
    except Exception as exc:
        check("REST reachable (service key)", False, str(exc)[:150])
        print(f"\n{len(FAILURES)} FAILURES")
        return 1

    for table in TABLES:
        try:
            s, _ = rest(url, svc, f"{table}?select=id&limit=1")
            check(f"table {table} exists", s == 200, f"status={s}")
        except Exception as exc:
            check(f"table {table} exists", False, str(exc)[:150])

    try:
        s, rows = rest(url, svc, "models?id=eq.rafnet-1.0.0-phase19&select=id")
        check("models seed row present", s == 200 and bool(rows), f"status={s} rows={rows}")
    except Exception as exc:
        check("models seed row present", False, str(exc)[:150])

    if anon:
        try:
            s, rows = rest(url, anon, "analysis_results?select=id&limit=1")
            check("RLS enforced (anon sees zero rows)", s == 200 and rows == [], f"status={s} rows={rows}")
        except Exception as exc:
            check("RLS enforced (anon sees zero rows)", False, str(exc)[:150])
    else:
        print("SKIP RLS check (SUPABASE_ANON_KEY not set)")

    if FAILURES:
        print(f"\n{len(FAILURES)} FAILURES: {FAILURES}")
        print("Fix: run supabase/migrations/001_soc_schema.sql in the SQL Editor, then re-run.")
        return 1
    print("\nVERIFY OK — backend can be wired (SUPABASE_URL + SERVICE_ROLE_KEY + SOC_REQUIRE_AUTH=true)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
