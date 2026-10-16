"""Request size guard — reject oversized bodies before they are buffered.

Checks the `Content-Length` header (when present) against MAX_REQUEST_BYTES
(default 32 MB: 25 MB audio cap + multipart/JSON overhead) and returns 413
immediately. Bodies without a declared length still hit the per-parser checks
(audio_io 25 MB cap, chunk sample cap) after read.
"""
from __future__ import annotations

import os

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


def _max_bytes() -> int:
    try:
        return int(os.getenv("MAX_REQUEST_BYTES", str(32 * 1024 * 1024)))
    except ValueError:
        return 32 * 1024 * 1024


class MaxBodySizeMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if request.url.path == "/ws/live":
            return await call_next(request)
        declared = request.headers.get("content-length")
        if declared is not None:
            try:
                if int(declared) > _max_bytes():
                    return JSONResponse(
                        status_code=413,
                        content={"detail": f"Request body too large (limit {_max_bytes()} bytes)"},
                    )
            except ValueError:
                pass
        return await call_next(request)
