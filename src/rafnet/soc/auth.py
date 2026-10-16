"""Optional Supabase JWT authentication for SOC routes.

Modes (env `SOC_REQUIRE_AUTH`, default false):
- false: all routes open; when a valid Bearer token IS supplied and Supabase is
  configured, the user_id is attached to saved records (best effort).
- true:  Bearer token required on `/api/analyze` and `/api/history` (writes +
  personal reads). Tokens are validated against the Supabase Auth API
  (`GET {SUPABASE_URL}/auth/v1/user`, stdlib only — no new dependencies).
  Fails closed: 401 on invalid token, 503 when auth is required but Supabase
  env (`SUPABASE_URL` + `SUPABASE_ANON_KEY` or service key) is missing.

Frontend sends the Supabase session access_token as the Bearer token.
Dev mode (no Supabase): leave SOC_REQUIRE_AUTH=false and use the labeled
dev bypass in the UI.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.request

from fastapi import Depends, HTTPException, Request

logger = logging.getLogger(__name__)


def auth_required() -> bool:
    return os.getenv("SOC_REQUIRE_AUTH", "false").lower() == "true"


def _supabase_cfg() -> tuple[str, str] | None:
    url = os.getenv("SUPABASE_URL", "").strip()
    key = os.getenv("SUPABASE_ANON_KEY", "").strip() or os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    if url and key:
        return url.rstrip("/"), key
    return None


def validate_supabase_token(token: str, timeout: float = 8.0) -> dict:
    """Validate a Supabase access token via the Auth API. Raises on failure."""
    cfg = _supabase_cfg()
    if cfg is None:
        raise RuntimeError("Supabase env not configured (SUPABASE_URL + ANON/SERVICE key)")
    url, key = cfg
    req = urllib.request.Request(
        f"{url}/auth/v1/user",
        method="GET",
        headers={"apikey": key, "Authorization": f"Bearer {token}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload: dict = json.loads(resp.read().decode())
            return payload
    except Exception as exc:
        raise ValueError(f"Token validation failed: {exc}") from exc


def _bearer(request: Request) -> str | None:
    header = request.headers.get("authorization", "")
    if header.lower().startswith("bearer "):
        token = header[7:].strip()
        return token or None
    return None


async def get_current_user(request: Request) -> dict | None:
    """Optional auth dependency: returns user dict, None when anonymous allowed."""
    token = _bearer(request)
    if token is None:
        if auth_required():
            raise HTTPException(status_code=401, detail="Missing Bearer token")
        return None
    try:
        info = validate_supabase_token(token)
    except RuntimeError as exc:
        # Auth required but server misconfigured — fail closed, loudly.
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc))
    user_id = info.get("id") or info.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Token has no user id")
    return {"id": user_id, "email": info.get("email", "")}


async def get_optional_user(request: Request) -> dict | None:
    """Best-effort variant: never rejects; returns None when token absent/invalid
    and auth is not required (used to attach user_id opportunistically)."""
    if not auth_required():
        token = _bearer(request)
        if token is None:
            return None
        try:
            info = validate_supabase_token(token)
            user_id = info.get("id") or info.get("sub")
            return {"id": user_id, "email": info.get("email", "")} if user_id else None
        except Exception as exc:
            logger.info("Optional auth ignored invalid token: %s", exc)
            return None
    return await get_current_user(request)


# Re-export as FastAPI dependency marker
CurrentUser = Depends(get_current_user)
OptionalUser = Depends(get_optional_user)
