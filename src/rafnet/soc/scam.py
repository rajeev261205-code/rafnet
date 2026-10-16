"""Call scam-risk analysis — transcript-based, rule-driven, context-aware.

Voice authenticity and scam behavior are DIFFERENT problems and must never
share one score. This module reads a call TRANSCRIPT (typed/pasted by the
user, or produced by an external STT step) plus optional user-supplied call
context (carrier, language) and returns a risk category with quoted evidence,
a plain-language explanation, and safety actions.

What it is NOT:
- Not a voice classifier: a synthetic voice does not imply scam, and a human
  voice does not imply safety. This module never sees audio.
- Not a carrier reputation lookup: Jio/Airtel identity is recorded context,
  never risk evidence. There is no telecom-network integration here.
- Not automatic call interception: transcripts are supplied explicitly by
  the user; nothing is recorded or uploaded by this module.

Anti-false-positive design: fraud-AWARENESS messages ("never share your OTP",
"beware of KYC fraud") match scam keywords but are explicitly suppressed via
advisory-context guards. Rules consider context; a legitimate OTP mention is
not automatically a scam.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

MAX_TRANSCRIPT_CHARS = 6000
MIN_WORDS_FOR_ASSESSMENT = 5

CARRIERS = ("unknown", "jio", "airtel", "other")


@dataclass
class _Signal:
    key: str
    label: str
    patterns: list[str]
    weight: int  # 1 = weak cue, 2 = strong cue
    advisory_immune: bool = True  # advisory context suppresses this signal


# Strong cues need a request directed AT the listener nearby to count
# (context rule). This separates "share your OTP now" (predatory) from "I
# entered my OTP in the app" (first-person account, benign).
_REQUEST_VERBS = (
    "share", "tell", "send", "give", "enter", "type", "confirm", "verify",
    "transfer", "pay", "install", "download", "batao", "bataiye", "bhejo",
    "karo", "karna", "daldo", "likho",
)
_REQUEST_TARGETS = (
    "your", "you", "yours", "me", "us", "tum", "tumhe", "tumhara",
    "tujhe", "tera", "aap", "aapko", "aapka", "mujhe", "hame",
)

# Explicit money destinations also count as request targets ("transfer
# Rs.50000 to account 98765" needs no second-person pronoun to be a demand).
_DESTINATIONS = (
    r"account.{0,20}\d{4,}",
    r"[\w.\-]{2,}@[a-zA-Z]{2,}",
    r"(rs\.?|₹|inr|rupees?)\s?\d[\d,]*",
)

_SIGNALS: list[_Signal] = [
    _Signal("credentials", "Requests OTP/PIN/password/banking credentials", [
        r"otp\b", r"\bpin\b", r"password", r"cvv\b", r"card number",
        r"account number", r"aadhaar", r"aadhar", r"pan number",
    ], 2),
    _Signal("money_transfer", "Requests money transfer to an unfamiliar account", [
        r"transfer (the )?money", r"transfer.{0,30}\d[\d,]*", r"send money",
        r"send.{0,30}(rs\.?|₹|inr|\d)", r"pay (to|into|on)",
        r"\bupi\b.{0,20}(id|pay)", r"account .{0,20}transfer",
        r"paisa (transfer|bhej|pay)", r"paise bhejo",
    ], 2),
    _Signal("threat_pressure", "Threats of blocking, legal action, or arrest", [
        r"account (will be |is )?blocked", r"sim (will be |is ).{0,10}deactivat",
        r"account band", r"sim band", r"arrest", r"police complaint",
        r"\bcbi\b", r"legal action", r"court case", r"fir against you",
    ], 2),
    _Signal("impersonation", "Impersonates bank/telecom/government/support", [
        r"bank (officer|manager|call|representative)", r"customer care",
        r"telecom (company|operator|department)", r"\brbi\b",
        r"income tax", r"customs? (officer|department)",
        r"technical support", r"microsoft support",
        r"kyc (department|officer|update|verification)",
        r"jio (care|office|representative|agent)",
        r"airtel (care|office|representative|agent)",
    ], 1),
    _Signal("remote_access", "Requests installing remote-access applications", [
        r"anydesk", r"teamviewer", r"quicksupport", r"quick support",
        r"screen shar", r"remote access", r"rustdesk",
    ], 2),
    _Signal("lure", "Prize/refund/KYC/SIM/verification lure", [
        r"lottery", r"you (have )?won", r"prize money", r"lucky draw",
        r"refund", r"cashback", r"kyc update", r"kyc expir",
        r"sim (verification|upgrade|deactivation)",
        r"congratulations.*(won|selected|prize)",
    ], 1),
    _Signal("urgency", "Artificial urgency to prevent verification", [
        r"immediately", r"right now", r"within (24|2) hours?", r"today itself",
        r"last warning", r"final (notice|warning)", r"do not (disconnect|hang up|cut)",
        r"stay on (the )?line", r"turant", r"abhi ke abhi", r"foran",
    ], 1),
]

# Advisory/awareness phrasing: when the transcript WARNS about fraud, scam
# keywords in it are educational, not predatory. Any window containing one of
# these suppresses signal matches in the same window.
_ADVISORY = (
    "never share", "do not share", "don't share", "dont share",
    "beware of", "be aware of", "fraud alert", "scam alert",
    "stay safe from", "protect yourself", "awareness",
    "kabhi (mat|nahi|na) (share|batao|batana)",
    "otp kisi (ko|se) (mat|nahi|na)",
    "dhokhadhadi", "savdhaan", "satark rahen",
)

_WINDOW = 120  # chars around a match inspected for advisory/request context


def _text_model_path():
    import os
    from pathlib import Path as _Path

    override = os.getenv("SCAM_TEXT_MODEL_PATH")
    if override:
        p = _Path(override)
        if not p.is_absolute():
            p = _Path(__file__).resolve().parents[3] / p
        return p
    return _Path(__file__).resolve().parents[3] / "models" / "scam_text.joblib"


_TEXT_MODEL: dict = {"loaded": False, "bundle": None, "tried": False}


def classifier_proba(text: str) -> float | None:
    """Tiny TF-IDF classifier signal (weak cue). Returns None when the
    weights file is absent or sklearn/joblib are unavailable — rules-only
    mode, never an error. See tools/train_scam_text.py + reports/scam_text_eval.md."""
    if not _TEXT_MODEL["tried"]:
        _TEXT_MODEL["tried"] = True
        try:
            import joblib

            path = _text_model_path()
            if path.is_file():
                _TEXT_MODEL["bundle"] = joblib.load(str(path))
                _TEXT_MODEL["loaded"] = True
        except Exception:
            _TEXT_MODEL["bundle"] = None
    bundle = _TEXT_MODEL["bundle"]
    if not bundle:
        return None
    try:
        from scipy.sparse import hstack

        xw = bundle["word_vec"].transform([text])
        xc = bundle["char_vec"].transform([text])
        return round(float(bundle["clf"].predict_proba(hstack([xw, xc]))[0, 1]), 4)
    except Exception:
        return None


def classifier_info() -> dict:
    """Availability info for /api/capabilities (real probe, no guessing)."""
    probed = classifier_proba("hello, this is a capability probe sentence for testing")
    if probed is None:
        return {"available": False, "backend": None,
                "reason": "No trained text weights (models/scam_text.joblib absent "
                          "or sklearn unavailable) — rules-only mode. Train with "
                          "tools/train_scam_text.py."}
    return {"available": True, "backend": "tfidf-logreg",
            "reason": "Weak cue only (see reports/scam_text_eval.md); rules carry the decision."}


def _windows(text: str, start: int, end: int) -> str:
    return text[max(0, start - _WINDOW): end + _WINDOW]


def _advisory_nearby(window: str) -> bool:
    return any(a in window for a in _ADVISORY)


def _request_nearby(window: str) -> bool:
    return any(v in window for v in _REQUEST_VERBS)


def _target_nearby(window: str) -> bool:
    # Whole-word match: short markers like "me"/"us" must not fire inside
    # "message" or "immediately".
    if any(re.search(r"\b" + re.escape(t) + r"\b", window) for t in _REQUEST_TARGETS):
        return True
    return any(re.search(p, window) for p in _DESTINATIONS)


@dataclass
class CallRisk:
    risk: str  # low | suspicious | high | insufficient
    score: int  # summed signal weights (transparent, not a probability)
    max_score: int  # reference scale (sum of all strong weights + weak)
    evidence: list[dict] = field(default_factory=list)
    explanation: str = ""
    actions: list[str] = field(default_factory=list)
    carrier: str = "unknown"
    classifier_prob: float | None = None  # tiny-text-model P(scam), if loaded
    limitations: list[str] = field(default_factory=list)


_LIMITATIONS = (
    "Transcript-only analysis: no audio, caller identity, or network data is used.",
    "Carrier context (Jio/Airtel/other) is recorded for reference and never scored.",
    "A low-risk result does not guarantee a call is safe; new scam scripts appear constantly.",
    "No automatic recording, interception, or telecom lookup is performed.",
)


def analyze_call_risk(
    transcript: str,
    carrier: str = "unknown",
    language: str = "auto",
) -> CallRisk:
    """Assess scam risk from an explicitly supplied call transcript."""
    carrier = (carrier or "unknown").strip().lower()
    if carrier not in CARRIERS:
        carrier = "other"
    text = (transcript or "").strip()
    if len(text) > MAX_TRANSCRIPT_CHARS:
        text = text[:MAX_TRANSCRIPT_CHARS]
    lowered = text.lower()
    words = re.findall(r"[a-zA-Z\u0900-\u097F]+", lowered)

    if len(words) < MIN_WORDS_FOR_ASSESSMENT:
        return CallRisk(
            risk="insufficient", score=0, max_score=_max_score(),
            explanation="Too little text to assess — paste more of what the caller said.",
            actions=["If the call felt wrong, hang up and verify independently."],
            carrier=carrier, limitations=list(_LIMITATIONS),
        )

    advisory_doc = _advisory_nearby(lowered)
    clf_prob = classifier_proba(text)

    evidence: list[dict] = []
    score = 0
    strong_hits = 0
    for sig in _SIGNALS:
        for pat in sig.patterns:
            for m in re.finditer(pat, lowered):
                window = _windows(lowered, m.start(), m.end())
                if sig.advisory_immune and _advisory_nearby(window):
                    continue  # fraud-awareness context, not predatory
                if sig.weight >= 2 and not (
                    _request_nearby(window) and _target_nearby(window)
                ):
                    continue  # strong cues need a request aimed at the listener
                excerpt = text[max(0, m.start() - 40): m.end() + 40].strip()
                evidence.append({"signal": sig.key, "label": sig.label,
                                 "match": m.group(0), "excerpt": excerpt})
                score += sig.weight
                if sig.weight >= 2:
                    strong_hits += 1
                break  # one excerpt per pattern is enough

    # Tiny classifier: ONE weak cue (+1 at p>=0.75), advisory-guarded, and
    # only reported — rules carry the decision (see reports/scam_text_eval.md
    # for why it stays weak: tiny constructed training set).
    if clf_prob is not None and clf_prob >= 0.75 and not advisory_doc:
        evidence.append({"signal": "text_classifier", "label": "Statistical text model",
                         "match": f"P(scam)={clf_prob}", "excerpt": text[:80]})
        score += 1

    if strong_hits >= 2 or score >= 5:
        risk = "high"
        explanation = (
            "Multiple high-risk scam indicators in one call (e.g. credential "
            "requests plus threats or impersonation). Treat as a likely scam attempt."
        )
        actions = [
            "Hang up immediately; do not share OTPs, PINs, or bank details.",
            "Do not transfer money or install any app the caller suggested.",
            "Call the organization back on its official number to verify.",
            "Report the number on your national cyber-fraud helpline.",
        ]
    elif strong_hits == 1 or score >= 2:
        risk = "suspicious"
        explanation = (
            "At least one scam-like behavior detected. It may still be legitimate, "
            "but verify through an independent channel before acting."
        )
        actions = [
            "Do not share credentials or transfer money during the call.",
            "Verify the claim independently (official app, website, or number).",
        ]
    else:
        risk = "low"
        explanation = (
            "No recognized scam patterns in the supplied text. This does not "
            "guarantee the call is safe."
        )
        actions = ["Stay cautious with unexpected requests even on ordinary calls."]

    return CallRisk(
        risk=risk, score=score, max_score=_max_score(), evidence=evidence,
        explanation=explanation, actions=actions, carrier=carrier,
        classifier_prob=clf_prob, limitations=list(_LIMITATIONS),
    )


def _max_score() -> int:
    return sum(s.weight for s in _SIGNALS)


def stt_capability() -> dict:
    """Probe for a usable speech-to-text backend (real check, no guessing)."""
    try:
        import importlib.util as _util

        for module in ("faster_whisper", "whisper"):
            if _util.find_spec(module) is not None:
                return {"available": True, "backend": module,
                        "reason": f"transcription available via {module}"}
    except Exception:
        pass
    return {
        "available": False, "backend": None,
        "reason": "No STT backend installed (checked faster_whisper, whisper). "
                  "Paste a transcript for call-risk analysis, or install an STT "
                  "backend pending approval (e.g. faster-whisper + its model download).",
    }
