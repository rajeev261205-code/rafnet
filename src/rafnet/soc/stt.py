"""Optional speech-to-text layer (CPU-friendly backends only).

`transcribe()` returns {"available": True, "text", "language", "backend"} when
a supported STT backend is installed (faster-whisper preferred, whisper
fallback), else {"available": False, "reason"} with setup instructions.
Callers (call-risk UI/API) must show the unavailable state and accept a
pasted transcript instead — never a fabricated one.

torch is NOT required for the unavailable path.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def stt_info() -> dict[str, Any]:
    from .scam import stt_capability

    info = stt_capability()
    return {"backend": info["backend"], "available": info["available"], "reason": info["reason"]}


def transcribe(samples: Any, sample_rate: int = 16000,
               model_size: str = "small") -> dict[str, Any]:
    """Transcribe 16 kHz mono audio. CPU-friendly sizes only (tiny/base/small)."""
    info = stt_info()
    if not info["available"]:
        return {"available": False, "text": None, "language": None,
                "backend": None, "reason": info["reason"]}
    audio = np.asarray(samples, dtype=np.float32).ravel()
    if audio.size < 16000:
        return {"available": False, "text": None, "language": None,
                "backend": info["backend"],
                "reason": "Audio too short for transcription (< 1 s)."}
    try:
        if info["backend"] == "faster_whisper":
            from faster_whisper import WhisperModel

            model = WhisperModel(model_size, device="cpu", compute_type="int8")
            segments, detected = model.transcribe(audio, beam_size=1)
            text = " ".join(s.text for s in segments).strip()
            lang = getattr(detected, "language", "auto")
        else:
            import whisper

            model = whisper.load_model(model_size if model_size in
                                       ("tiny", "base", "small") else "small",
                                       device="cpu")
            out = model.transcribe(audio, fp16=False)
            text, lang = str(out.get("text", "")).strip(), str(out.get("language", "auto"))
    except Exception as exc:
        return {"available": False, "text": None, "language": None,
                "backend": info["backend"], "reason": f"Transcription failed: {exc}"}
    if not text:
        return {"available": False, "text": None, "language": lang,
                "backend": info["backend"], "reason": "No speech recognized."}
    return {"available": True, "text": text, "language": lang, "backend": info["backend"]}
