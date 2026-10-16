"""Voice enrollment biometric (Layer 4) — currently unavailable.

No open-source speaker-embedding model is installed in this environment
(checked: speechbrain and resemblyzer absent; only the RAFNet WavLM encoder,
torchaudio and librosa are present). Until such a model is approved and
installed, enrollment returns kind "unavailable" and no similarity is ever
invented. Enabling it needs e.g. `pip install resemblyzer` (plus its model
download) — ask before installing.
"""
from __future__ import annotations

from typing import Any

ENROLLMENT_ID = "enrollment"
ENROLLMENT_NAME = "Voice Enrollment Biometric"

REASON = (
    "Voice enrollment unavailable: no speaker-embedding model installed "
    "(checked speechbrain, resemblyzer — absent). "
    "Candidate: resemblyzer (pip install resemblyzer + model download) pending approval."
)


def speaker_capability() -> dict[str, Any]:
    """Probe for a usable speaker-embedding backend (real check, no guessing)."""
    try:
        import importlib.util as _util

        for module in ("speechbrain", "resemblyzer"):
            if _util.find_spec(module) is not None:
                return {
                    "available": True,
                    "backend": module,
                    "reason": f"speaker backend available via {module}",
                }
    except Exception:
        pass
    return {"available": False, "backend": None, "reason": REASON}


def enrollment_layer() -> dict[str, Any]:
    """Layer 4 payload — kind "unavailable" until a speaker model exists."""
    cap = speaker_capability()
    return {
        "id": ENROLLMENT_ID,
        "name": ENROLLMENT_NAME,
        "kind": "unavailable",
        "score": None,
        "evidence": [f"not enrolled / not available — {cap['reason']}"],
        "metrics": {"enrolled": False, "backend": cap["backend"]},
    }


def enrollment_status() -> dict[str, Any]:
    """Status for GET /api/enrollment (real capability probe)."""
    cap = speaker_capability()
    return {"enrolled": False, "kind": "unavailable", "backend": cap["backend"], "reason": cap["reason"]}
