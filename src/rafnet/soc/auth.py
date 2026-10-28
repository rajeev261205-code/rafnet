"""Supabase JWT authentication for SOC routes.

Modes (env `SOC_REQUIRE_AUTH`):
- unset: defaults to true when `RAFNET_ENV=production`, false otherwise.
- false: dev/demo mode — routes work anonymously; when a valid Bearer token IS
  supplied and Supabase is configured, the user_id scopes saved records.
- true:  authentication is required on the whole SOC API. Tokens are validated
  against the Supabase Auth API (`GET {SUPABASE_URL}/auth/v1/user`, stdlib only
  — no new dependencies). Fails closed: 401 on missing/invalid token, 503 when
  auth is required but Supabase env (`SUPABASE_URL` + `SUPABASE_ANON_KEY` or
  service key) is missing.

Frontend sends the Supabase session access_token as the Bearer token (and as a
`?token=` query parameter for the WebSocket handshake, which cannot set headers
in browsers). Startup validation (`validate_security_config`) refuses to boot a
production deployment with auth disabled, Supabase missing, or wildcard CORS.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import urllib.request

from fastapi import Depends, HTTPException, Request

logger = logging.getLogger(__name__)


def is_production() -> bool:
    return os.getenv("RAFNET_ENV", "development").strip().lower() == "production"


def auth_required() -> bool:
    explicit = os.getenv("SOC_REQUIRE_AUTH")
    if explicit is not None and explicit.strip():
        return explicit.strip().lower() == "true"
    return is_production()


def cors_origins() -> list[str]:
    raw = os.getenv("SOC_CORS_ORIGINS") or os.getenv("CORS_ORIGINS", "")
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def validate_security_config() -> None:
    """Fail closed at startup for unsafe production configuration.

    Development/demo mode is intentionally permissive; production must have
    authentication, a Supabase project to validate tokens, and an explicit
    non-wildcard CORS allow-list.
    """
    if not is_production():
        return
    problems: list[str] = []
    if not auth_required():
        problems.append("SOC_REQUIRE_AUTH must be true in production")
    if _supabase_cfg() is None:
        problems.append(
            "SUPABASE_URL and SUPABASE_ANON_KEY (or SUPABASE_SERVICE_ROLE_KEY) "
            "are required in production"
        )
    origins = cors_origins()
    if not origins:
        problems.append(
            "SOC_CORS_ORIGINS must list the frontend origin(s) in production"
        )
    elif "*" in origins:
        problems.append("SOC_CORS_ORIGINS must not contain '*' in production")
    if problems:
        raise RuntimeError("Unsafe production configuration: " + "; ".join(problems))


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


async def _user_from_token(token: str) -> dict:
    """Validate without blocking the event loop on the Supabase HTTP call."""
    try:
        info = await asyncio.to_thread(validate_supabase_token, token)
    except RuntimeError:
        # Auth infrastructure unavailable/misconfigured: fail closed, but do
        # not expose internal configuration to the caller.
        raise HTTPException(status_code=503, detail="Authentication unavailable")
    except ValueError as exc:
        logger.info("Rejected Supabase token: %s", exc)
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    user_id = info.get("id") or info.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Token has no user id")
    return {"id": user_id, "email": info.get("email", "")}


async def get_current_user(request: Request) -> dict | None:
    """Optional auth dependency: returns user dict, None when anonymous allowed."""
    token = _bearer(request)
    if token is None:
        if auth_required():
            raise HTTPException(status_code=401, detail="Missing Bearer token")
        return None
    return await _user_from_token(token)


async def get_optional_user(request: Request) -> dict | None:
    """Optional variant for dev mode: anonymous is allowed, but a supplied
    token is never silently ignored �?" invalid tokens fail closed with 401 so a
    client with an expired session cannot accidentally write as anonymous."""
    if not auth_required():
        token = _bearer(request)
        if token is None:
            return None
        return await _user_from_token(token)
    return await get_current_user(request)


# Re-export as FastAPI dependency marker
CurrentUser = Depends(get_current_user)
OptionalUser = Depends(get_optional_user)
