"""Persistence abstraction — Supabase-ready, in-memory by default.

`AnalysisStore` is the port; `InMemoryAnalysisStore` is the default adapter
(local/demo, same record shape as Supabase rows). `SupabaseAnalysisStore`
( брутто opt-in) uses the Supabase REST API via `SUPABASE_URL` +
`SUPABASE_SERVICE_ROLE_KEY` (server-only env, never exposed to the frontend).
When env is missing, `get_store()` returns the in-memory adapter and logs why —
no silent behavior change.

Supabase SQL + RLS lives in `supabase/migrations/001_soc_schema.sql`.
"""
from __future__ import annotations

import logging
import os
from collections import deque
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class AnalysisStore(Protocol):
    def save_analysis(self, record: dict[str, Any]) -> dict[str, Any]: ...
    def get_analysis(self, analysis_id: str) -> dict[str, Any] | None: ...
    def list_analyses(self, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]: ...
    def count(self) -> int: ...


class InMemoryAnalysisStore:
    """Default local/demo adapter — ring buffer, newest first."""

    def __init__(self, maxlen: int = 500) -> None:
        self._rows: deque[dict[str, Any]] = deque(maxlen=maxlen)

    def save_analysis(self, record: dict[str, Any]) -> dict[str, Any]:
        self._rows.appendleft(record)
        return record

    def get_analysis(self, analysis_id: str) -> dict[str, Any] | None:
        return next((r for r in self._rows if r.get("id") == analysis_id), None)

    def list_analyses(self, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        rows = list(self._rows)
        return rows[offset : offset + limit]

    def count(self) -> int:
        return len(self._rows)


class SupabaseAnalysisStore:
    """Production adapter over Supabase Postgres REST (table: analysis_results).

    Activated only when SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY are set.
    Uses only stdlib (urllib) so `supabase-py` is optional.
    """

    def __init__(self, url: str, service_key: str, timeout: float = 10.0) -> None:
        import json
        import urllib.request

        self._url = url.rstrip("/")
        self._key = service_key
        self._timeout = timeout
        self._json = json
        self._request = urllib.request
        # Fail fast with a cheap read so misconfiguration surfaces at startup.
        self._rest("analysis_results", {"select": "id", "limit": "1"})

    def _rest(
        self,
        table: str,
        params: dict[str, str] | None = None,
        method: str = "GET",
        body: dict[str, Any] | None = None,
    ) -> Any:
        import urllib.parse

        query = ("?" + urllib.parse.urlencode(params)) if params else ""
        req = self._request.Request(
            f"{self._url}/rest/v1/{table}{query}",
            data=self._json.dumps(body).encode() if body is not None else None,
            method=method,
            headers={
                "apikey": self._key,
                "Authorization": f"Bearer {self._key}",
                "Content-Type": "application/json",
                "Prefer": "return=representation",
            },
        )
        with self._request.urlopen(req, timeout=self._timeout) as resp:
            text = resp.read().decode()
            return self._json.loads(text) if text else []

    def save_analysis(self, record: dict[str, Any]) -> dict[str, Any]:
        rows = self._rest("analysis_results", method="POST", body=record)
        return rows[0] if isinstance(rows, list) and rows else record

    def get_analysis(self, analysis_id: str) -> dict[str, Any] | None:
        rows = self._rest("analysis_results", {"id": f"eq.{analysis_id}", "limit": "1"})
        return rows[0] if rows else None

    def list_analyses(self, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = self._rest(
            "analysis_results",
            {"select": "*", "order": "created_at.desc", "limit": str(limit), "offset": str(offset)},
        )
        return rows

    def count(self) -> int:
        rows = self._rest("analysis_results", {"select": "id"})
        return len(rows) if isinstance(rows, list) else 0


def get_store() -> AnalysisStore:
    url = os.getenv("SUPABASE_URL", "").strip()
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    if url and key:
        try:
            store = SupabaseAnalysisStore(url, key)
            logger.info("Using SupabaseAnalysisStore (%s)", url)
            return store
        except Exception as exc:
            logger.error("Supabase store init failed, falling back to memory: %s", exc)
    else:
        logger.info("SUPABASE_URL/SERVICE_ROLE_KEY not set — using InMemoryAnalysisStore")
    return InMemoryAnalysisStore()
