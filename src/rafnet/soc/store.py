"""Persistence abstraction — Supabase-ready, in-memory by default.

`AnalysisStore` is the port; `InMemoryAnalysisStore` is the default adapter
(local/demo, same record shape as Supabase rows). `SupabaseAnalysisStore`
( брутто opt-in) uses the Supabase REST API via `SUPABASE_URL` +
`SUPABASE_SERVICE_ROLE_KEY` (server-only env, never exposed to the frontend).
When env is missing, `get_store()` returns the in-memory adapter and logs why —
no silent behavior change.

Supabase SQL + RLS lives in `supabase/migrations/001_soc_schema.sql`; `004`
bootstraps `profiles` rows for Auth users (required by the `user_id` FKs).
"""
from __future__ import annotations

import logging
import os
from collections import deque
from typing import Any, Protocol

logger = logging.getLogger(__name__)


def _owned_by(row: dict[str, Any], user_id: str | None) -> bool:
    """True when `row` belongs to the caller's principal.

    `user_id=None` is an explicit anonymous principal: it matches only rows
    with no owner. Unowned/legacy rows are never visible to authenticated
    users, and owned rows are never visible to anonymous callers.
    """
    owner = row.get("user_id")
    if user_id is None:
        return owner is None
    return owner == user_id


class AnalysisStore(Protocol):
    def save_analysis(self, record: dict[str, Any]) -> dict[str, Any]: ...
    def get_analysis(
        self, analysis_id: str, user_id: str | None = None
    ) -> dict[str, Any] | None: ...
    def list_analyses(
        self, limit: int = 50, offset: int = 0, user_id: str | None = None
    ) -> list[dict[str, Any]]: ...
    def count(self, user_id: str | None = None) -> int: ...
    def clear(self, user_id: str | None = None) -> int: ...


class InMemoryAnalysisStore:
    """Default local/demo adapter �?" ring buffer, newest first."""

    def __init__(self, maxlen: int = 500) -> None:
        self._rows: deque[dict[str, Any]] = deque(maxlen=maxlen)

    def save_analysis(self, record: dict[str, Any]) -> dict[str, Any]:
        self._rows.appendleft(record)
        return record

    def get_analysis(
        self, analysis_id: str, user_id: str | None = None
    ) -> dict[str, Any] | None:
        return next(
            (
                r
                for r in self._rows
                if r.get("id") == analysis_id and _owned_by(r, user_id)
            ),
            None,
        )

    def list_analyses(
        self, limit: int = 50, offset: int = 0, user_id: str | None = None
    ) -> list[dict[str, Any]]:
        rows = [r for r in self._rows if _owned_by(r, user_id)]
        return rows[offset : offset + limit]

    def count(self, user_id: str | None = None) -> int:
        return sum(1 for r in self._rows if _owned_by(r, user_id))

    def clear(self, user_id: str | None = None) -> int:
        """Delete exactly the caller's analyses (anonymous callers clear
        anonymous rows only; an authenticated caller never deletes another
        principal's rows, including unowned/legacy rows)."""
        before = len(self._rows)
        self._rows = deque(
            (r for r in self._rows if not _owned_by(r, user_id)),
            maxlen=self._rows.maxlen,
        )
        return before - len(self._rows)


class SupabaseAnalysisStore:
    """Production adapter over Supabase Postgres REST (table: analysis_results).

    Activated only when SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY are set.
    Uses only stdlib (urllib) so `supabase-py` is optional.

    Column mapping: the app-level record (AnalyzeResponse-shaped) uses
    `model_version` / `audio_duration_seconds` / flat probability fields,
    while the SQL schema (`supabase/migrations/001_soc_schema.sql`, plus
    002 for verdict/layers/reasons and 004 for Auth-user profiles) stores
    `model_id`, `duration_seconds`,
    and nested `probs`/`gateway` jsonb. `_to_row`/`_from_row` translate both
    ways so production rows round-trip without 400s from PostgREST on
    unknown columns and without KeyErrors on read.
    """

    # Flat app-level keys packed into the schema's jsonb columns.
    _PROB_KEYS = (
        "prob_synthetic",
        "prob_real",
        "classical_probability",
        "quantum_probability",
        "fused_probability",
        "calibrated_probability",
    )
    _GATEWAY_KEYS = (
        "gateway1_audio_weight",
        "gateway1_network_weight",
        "gateway2_classical_weight",
        "gateway2_quantum_weight",
    )

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

    @staticmethod
    def _user_filter(user_id: str | None) -> str:
        """PostgREST filter for the caller's principal.

        `None` is the anonymous principal and maps to `user_id=is.null`.
        Authenticated callers map to an exact owner match. Backend queries
        are always scoped, even though the service-role key bypasses RLS.
        """
        return "is.null" if user_id is None else f"eq.{user_id}"

    def save_analysis(self, record: dict[str, Any]) -> dict[str, Any]:
        saved = self._rest("analysis_results", method="POST", body=self._to_row(record))
        row = saved[0] if isinstance(saved, list) and saved else record
        return self._from_row(row) if isinstance(row, dict) else record

    def get_analysis(
        self, analysis_id: str, user_id: str | None = None
    ) -> dict[str, Any] | None:
        rows = self._rest(
            "analysis_results",
            {
                "id": f"eq.{analysis_id}",
                "user_id": self._user_filter(user_id),
                "limit": "1",
            },
        )
        if not rows:
            return None
        row = rows[0]
        return self._from_row(row) if isinstance(row, dict) else None

    def list_analyses(
        self, limit: int = 50, offset: int = 0, user_id: str | None = None
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = self._rest(
            "analysis_results",
            {
                "select": "*",
                "user_id": self._user_filter(user_id),
                "order": "created_at.desc",
                "limit": str(limit),
                "offset": str(offset),
            },
        )
        return [self._from_row(r) if isinstance(r, dict) else r for r in rows]

    @classmethod
    def _to_row(cls, record: dict[str, Any]) -> dict[str, Any]:
        """Translate an app-level record to schema columns (drops Nones so
        partial dicts and legacy rows stay insertable)."""
        row: dict[str, Any] = {
            "id": record.get("id"),
            "audio_name": record.get("audio_name"),
            "prediction": record.get("prediction"),
            "confidence": record.get("confidence"),
            "risk_score": record.get("risk_score"),
            "risk_level": record.get("risk_level"),
            "model_id": record.get("model_version", record.get("model_id")),
            "processing_time_ms": record.get("processing_time_ms"),
            "duration_seconds": record.get(
                "audio_duration_seconds", record.get("duration_seconds")
            ),
            "probs": {
                k: record[k] for k in cls._PROB_KEYS if record.get(k) is not None
            },
            "gateway": {
                k: record[k] for k in cls._GATEWAY_KEYS if record.get(k) is not None
            },
            "mock": record.get("mock", False),
            "created_at": record.get("created_at"),
            "verdict": record.get("verdict"),
            "layers": record.get("layers"),
            "reasons": record.get("reasons"),
            "input_codec": record.get("input_codec"),
            "input_lossy": record.get("input_lossy", False),
        }
        if record.get("user_id") is not None:
            row["user_id"] = record["user_id"]
        return {k: v for k, v in row.items() if v is not None}

    @classmethod
    def _from_row(cls, row: dict[str, Any]) -> dict[str, Any]:
        """Rebuild an app-level record from a schema row (tolerates missing
        columns on old rows; preserves any extra keys)."""
        probs = row.get("probs") or {}
        gateway = row.get("gateway") or {}
        out = dict(row)
        out.setdefault("model_version", row.get("model_id"))
        out.setdefault(
            "audio_duration_seconds", row.get("duration_seconds", 0.0)
        )
        for key in cls._PROB_KEYS:
            out.setdefault(key, probs.get(key))
        for key in cls._GATEWAY_KEYS:
            out.setdefault(key, gateway.get(key))
        out.setdefault("reasons", row.get("reasons") or [])
        out.setdefault("layers", row.get("layers") or [])
        out.setdefault("verdict", row.get("verdict"))
        out.setdefault("input_codec", row.get("input_codec"))
        out.setdefault("input_lossy", row.get("input_lossy", False))
        # Per-analysis stage timings are transport-only diagnostics (kept in
        # the in-memory record); the SQL schema does not persist them.
        out.setdefault("stage_timings_ms", row.get("stage_timings_ms") or {})
        return out

    def count(self, user_id: str | None = None) -> int:
        rows = self._rest(
            "analysis_results",
            {"select": "id", "user_id": self._user_filter(user_id)},
        )
        return len(rows) if isinstance(rows, list) else 0

    def clear(self, user_id: str | None = None) -> int:
        """Delete exactly the caller's analyses via Supabase REST. Anonymous
        callers delete anonymous rows (`user_id=is.null`) only."""
        resp = self._rest(
            "analysis_results",
            {"user_id": self._user_filter(user_id)},
            method="DELETE",
        )
        if isinstance(resp, list):
            return len(resp)
        return 0


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
