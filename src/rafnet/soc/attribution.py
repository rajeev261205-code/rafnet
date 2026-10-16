"""AI-provider attribution — honest unknown by default, evidence-graded.

The system is asked whether audio came from tools associated with Google
Gemini, Anthropic Claude, Meta AI, or other TTS/voice-cloning providers.
There is NO validated audio fingerprint for any of these providers in this
repository, and a generic synthetic-speech score must never be mapped to a
provider name. So this module:

- returns ``provider: "unknown"`` unless authenticated provenance is supplied
  AND verifiable (no verifier is installed here, so unknown stands);
- records whatever provenance/context the caller supplies;
- grades evidence explicitly (verified / compatible / unknown);
- documents the factual product landscape without overclaiming.

Factual notes (product capabilities, not detection claims):
- Anthropic Claude has no public text-to-speech/voice product; there is no
  plausible "Claude voice" to attribute.
- Google (Gemini TTS APIs) and Meta (speech models/APIs) do offer speech
  generation, but their outputs carry no fingerprint this system can verify.
- Accent, language, content, or style say nothing about the provider.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Attribution:
    provider: str  # "unknown" unless authenticated provenance verifies otherwise
    grade: str  # verified | compatible | unknown
    evidence: list[dict] = field(default_factory=list)
    explanation: str = ""
    limitations: list[str] = field(default_factory=list)


_LIMITATIONS = (
    "No provider fingerprint model exists in this system; generic deepfake "
    "scores are never mapped to provider names.",
    "Spoken content, accent, language, or style cannot identify a provider.",
    "Only authenticated provenance (e.g. verified C2PA manifest or a signed "
    "provider receipt checked by a verifier) could establish a source — no "
    "verifier is installed, so provenance cannot be confirmed here.",
)


def attribute_provider(provenance: dict[str, Any] | None = None) -> Attribution:
    """Assess AI-provider attribution for an analyzed clip.

    `provenance` is optional caller-supplied metadata (file container
    metadata, declared origin, watermark-detector output). It is RECORDED,
    not trusted: without a verifier it cannot establish a source.
    """
    provenance = dict(provenance or {})
    evidence: list[dict] = []
    if provenance:
        evidence.append({"kind": "supplied-provenance", "grade": "unknown",
                         "detail": "Caller-supplied metadata recorded but not "
                                   "verifiable here; it does not establish origin.",
                         "data": {k: v for k, v in provenance.items()}})
    return Attribution(
        provider="unknown",
        grade="unknown",
        evidence=evidence,
        explanation=(
            "Attribution unavailable: no reliable source evidence. The clip may "
            "still be AI-generated (see voice-authenticity result); who generated "
            "it cannot be determined from audio analysis alone."
        ),
        limitations=list(_LIMITATIONS),
    )
