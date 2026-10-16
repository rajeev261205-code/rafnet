"""Tests for transcript-based scam risk, attribution, and capabilities.

Rule-engine behavior is asserted on constructed transcripts with known
intent (ground truth by construction, not by label invention). These tests
validate the RULES, never detector accuracy on real calls.
"""
from fastapi.testclient import TestClient


def test_scam_english_bank_otp_threat_high():
    from rafnet.soc.scam import analyze_call_risk

    r = analyze_call_risk(
        "Hello, I am calling from State Bank head office. Your account will be "
        "blocked today itself. To verify, share your OTP with me right now, "
        "otherwise legal action will follow."
    )
    assert r.risk == "high"
    keys = {e["signal"] for e in r.evidence}
    assert "credentials" in keys and "threat_pressure" in keys
    assert r.actions, "high risk must carry safety actions"


def test_scam_hinglish_jio_sim_block():
    from rafnet.soc.scam import analyze_call_risk

    r = analyze_call_risk(
        "Main Jio customer care se bol raha hoon. Aapka SIM aaj band ho jayega. "
        "OTP batao turant, warna number deactivat ho jayega.",
        carrier="jio",
    )
    assert r.risk in ("suspicious", "high")
    assert r.carrier == "jio"


def test_benign_family_call_low():
    from rafnet.soc.scam import analyze_call_risk

    r = analyze_call_risk(
        "Hi mom, I will be home for dinner around seven. Can you keep some "
        "rice and dal ready? I will bring groceries on the way back."
    )
    assert r.risk == "low"
    assert r.evidence == []


def test_fraud_awareness_message_not_scam():
    """Advisory context must suppress keyword matches (false-positive guard)."""
    from rafnet.soc.scam import analyze_call_risk

    r = analyze_call_risk(
        "Never share your OTP with anyone who calls you. Banks never ask for "
        "your PIN on a call. Beware of KYC fraud and stay safe from scam alerts."
    )
    assert r.risk == "low", r.evidence


def test_first_person_otp_account_not_scam():
    """Describing your own legitimate OTP use is not a scam indicator."""
    from rafnet.soc.scam import analyze_call_risk

    r = analyze_call_risk(
        "I got an OTP for my own netbanking login and I entered it in the "
        "official app myself. Everything worked fine after that."
    )
    assert r.risk == "low", r.evidence


def test_money_demand_with_destination_high():
    from rafnet.soc.scam import analyze_call_risk

    r = analyze_call_risk(
        "Transfer Rs.50000 to account 9876543210 immediately or the police "
        "will arrest you today."
    )
    assert r.risk == "high"


def test_remote_access_request_suspicious():
    from rafnet.soc.scam import analyze_call_risk

    r = analyze_call_risk(
        "This is technical support. Please install AnyDesk so I can check "
        "your computer for viruses right now."
    )
    assert r.risk in ("suspicious", "high")
    assert any(e["signal"] == "remote_access" for e in r.evidence)


def test_too_short_insufficient():
    from rafnet.soc.scam import analyze_call_risk

    r = analyze_call_risk("Hello? Who is this?")
    assert r.risk == "insufficient"


def test_carrier_recorded_never_scored():
    """Same scam text on Jio vs unknown must score identically."""
    from rafnet.soc.scam import analyze_call_risk

    text = ("Your account will be blocked today. Share your OTP with me "
            "immediately.")
    a = analyze_call_risk(text, carrier="jio")
    b = analyze_call_risk(text, carrier="unknown")
    c = analyze_call_risk(text, carrier="airtel")
    assert (a.risk, a.score) == (b.risk, b.score) == (c.risk, c.score)
    assert a.carrier == "jio" and c.carrier == "airtel"


def test_attribution_always_unknown():
    from rafnet.soc.attribution import attribute_provider

    for prov in (None, {}, {"declared_by": "user",
                            "note": "sounds like a Gemini voice"}):
        r = attribute_provider(prov)
        assert r.provider == "unknown"
        assert r.grade == "unknown"
        assert r.limitations, "limitations must be stated"


def test_capabilities_shape_mock():
    from rafnet.soc.capabilities import get_capabilities
    from rafnet.soc.model_adapter import MockVoiceModel

    caps = get_capabilities(MockVoiceModel())
    assert set(caps) == {"neural", "replay", "speaker", "transcription",
                         "text_classifier", "attribution"}
    assert caps["neural"]["mock"] is True
    assert caps["neural"]["available"] is False
    assert caps["replay"]["available"] is True
    assert caps["speaker"]["available"] is False
    assert caps["transcription"]["available"] is False
    assert caps["attribution"]["available"] is False
    assert isinstance(caps["text_classifier"]["available"], bool)


def _mock_client():
    from fastapi import FastAPI

    from rafnet.soc import routes
    from rafnet.soc.model_adapter import MockVoiceModel
    from rafnet.soc.service import SOCService

    routes.override_soc_service(SOCService(MockVoiceModel()))
    app = FastAPI()
    app.include_router(routes.router)
    return TestClient(app)


def test_call_risk_endpoint():
    client = _mock_client()
    r = client.post("/api/call-risk", json={
        "transcript": "Your account will be blocked today. Share your OTP with me now.",
        "carrier": "airtel",
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["risk"] in ("suspicious", "high")
    assert body["carrier"] == "airtel"
    assert body["actions"] and body["limitations"]


def test_call_risk_endpoint_rejects_bad_carrier():
    client = _mock_client()
    r = client.post("/api/call-risk", json={"transcript": "hello " * 20, "carrier": "vodafone"})
    assert r.status_code == 422


def test_attribution_endpoint_unknown():
    client = _mock_client()
    r = client.post("/api/attribution", json={"provenance": {"note": "test"}})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["provider"] == "unknown"
    assert body["grade"] == "unknown"


def test_capabilities_endpoint():
    client = _mock_client()
    r = client.get("/api/capabilities")
    assert r.status_code == 200, r.text
    assert r.json()["transcription"]["available"] is False
