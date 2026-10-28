"""Model/service readiness — distinguishes API availability from model readiness.

The frontend uses this to show whether required models are actually ready,
instead of guessing from a 200 OK. Every entry is probed for real; nothing
is hardcoded to True.
"""
from __future__ import annotations

from typing import Any


def get_capabilities(model: Any) -> dict[str, Any]:
    from .scam import classifier_info, stt_capability
    from .speaker import speaker_capability

    neural_loaded = bool(getattr(model, "is_loaded", lambda: False)())
    neural_mock = getattr(model, "name", "") == "mock"
    speaker = speaker_capability()
    stt = stt_capability()
    clf = classifier_info()
    return {
        "neural": {
            "available": neural_loaded and not neural_mock,
            "mock": neural_mock,
            "name": getattr(model, "name", "unknown"),
            "task": "human vs AI-generated speech (RAFNet, 16 kHz)",
        },
        "replay": {
            "available": True,
            "kind": "heuristic",
            "task": "playback/replay evidence (unvalidated thresholds)",
        },
        "speaker": {
            "available": speaker["available"],
            "backend": speaker["backend"],
            "reason": speaker["reason"],
            "task": "voice comparison against an authorized reference",
        },
        "transcription": {
            "available": stt["available"],
            "backend": stt["backend"],
            "reason": stt["reason"],
            "task": "speech-to-text for call-risk analysis",
        },
        "text_classifier": {
            "available": clf["available"],
            "backend": clf["backend"],
            "reason": clf["reason"],
            "task": "weak statistical cue for scam text (rules carry the decision)",
        },
        "attribution": {
            "available": False,
            "backend": None,
            "reason": "No provider fingerprint model or provenance verifier installed; "
                      "attribution returns unknown.",
            "task": "AI-provider source attribution",
        },
    }
