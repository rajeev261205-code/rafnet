"""Phase 2: ElevenLabs UNSEEN-GENERATOR TEST set (commercial, built-in voices only).

Hard caps: max 25 clips AND max 3000 characters total (free tier limit is
10000/mo; we stay far below). Subscription usage is read BEFORE and AFTER;
the run aborts if the projection exceeds the cap or the account limit.
Built-in (premade) voices only — NO voice cloning, nothing from the user's
recordings is ever sent. All clips cached under data/phase2/fake/elevenlabs
(git-ignored); nothing redistributed. TEST ONLY: never train/tune on these.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "data" / "phase2" / "fake" / "elevenlabs"
MANIFEST = ROOT / "data" / "manifest_phase2_elevenlabs.csv"
MAX_CLIPS = 25
MAX_CHARS = 3000
MODEL = "eleven_multilingual_v2"

# scam-style + neutral reads, en/hi/ta (each <= 130 chars)
TEXTS = [
    ("en", "Dear customer, your bank account will be blocked today. Share your OTP now to verify your KYC immediately."),
    ("en", "This is to inform you that your telecom SIM will be deactivated. Press one to talk to our KYC officer."),
    ("en", "Your parcel is held at customs. Pay a small clearance fee now or it will be returned to sender."),
    ("en", "Congratulations, you have won a lottery of fifty thousand rupees. Share your bank details to claim it."),
    ("en", "Your electricity connection will be cut tonight due to an unpaid bill. Pay now to avoid disconnection."),
    ("en", "We detected suspicious activity on your account. Confirm your full card number to secure it."),
    ("en", "Your KYC documents have expired. Update them today or your account services will be suspended."),
    ("en", "You have a pending refund of tax overpaid last year. Enter your account number to receive it."),
    ("en", "This is the income tax office calling about a legal case in your name. Stay on the line."),
    ("en", "Your mobile number has won a free recharge. Click the link and enter your OTP to activate."),
    ("hi", "Namaste, aapka bank khata aaj band ho jayega. Kripya apna OTP batakar KYC poora karein."),
    ("hi", "Aapke SIM card ka KYC adhura hai. Turant usse update karein warna seva band ho jayegi."),
    ("hi", "Aapki bijli ka bill bhugtan lambit hai. Aaj hi bhugtan karein warna connection kaat diya jayega."),
    ("hi", "Badhaai ho, aapne pachaas hazaar rupaye ka inaam jeeta hai. Daava karne ke liye jaankari dein."),
    ("hi", "Aapke khaate mein sandigdh gatividhi mili hai. Suraksha ke liye apna card number batayein."),
    ("ta", "Vanakkam, ungaludaiya vangi kanakku indru adaikkappadum. OTP-ai pagirndu KYC-ai mudiyungal."),
    ("ta", "Ungal SIM card-in KYC mudivillai. Udane puthuppiththal seyyungal, illaiyendraal sevai niruththappadum."),
    ("ta", "Ungal minsaara kattanam seluththappadavillai. Indru seluththungal, illaiyendraal inaippu thundikkappadum."),
    ("ta", "Vaazhththukkal, neengal aimpathaayiram rubai parisu vettirukkirirgal. Peruvatharku vivarangalai kodungal."),
    ("ta", "Ungal kanakkile santhegamana nadavadikkai kandupidikkappattathu. Padukappukku card ennaik kodungal."),
    ("en", "The morning bus was late again, so we walked the last mile to the office in the rain."),
    ("en", "She keeps a small garden behind the house with roses, mint and two mango trees."),
    ("hi", " Mausam aaj saaf hai aur hawa mein thandak hai. Bachche maidan mein khel rahe hain."),
    ("ta", " Kaalaiyil mazhai peythathu, aanaal ippozhuthu vaanilakam thelivaaga ullathu."),
    ("en", "Please call back on the landline after six in the evening to confirm the appointment."),
]


def api(path: str, key: str, data: bytes | None = None) -> dict | bytes:
    req = urllib.request.Request(
        f"https://api.elevenlabs.io{path}",
        data=data,
        headers={"xi-api-key": key, "Content-Type": "application/json",
                 "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        body = r.read()
        try:
            return json.loads(body)
        except Exception:
            return body


def usage(key: str) -> tuple[int, int]:
    d = api("/v1/user/subscription", key)
    assert isinstance(d, dict)
    return int(d.get("character_count", -1)), int(d.get("character_limit", -1))


def main() -> int:
    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        print("ELEVENLABS_API_KEY missing — skipping (TEST set stays empty).")
        return 0
    total_chars = sum(len(t) for _, t in TEXTS)
    print(f"planned: {len(TEXTS)} clips, {total_chars} chars "
          f"(caps: {MAX_CLIPS} clips, {MAX_CHARS} chars)")
    if len(TEXTS) > MAX_CLIPS or total_chars > MAX_CHARS:
        print("REFUSING: plan exceeds hard caps.")
        return 2
    used_before, limit = usage(key)
    print(f"account: used={used_before} limit={limit}")
    if used_before < 0:
        print("REFUSING: could not read quota.")
        return 2
    if used_before + total_chars > min(limit, MAX_CHARS + used_before):
        print("REFUSING: would exceed account limit or prudent share.")
        return 2

    voices = api("/v1/voices", key)
    assert isinstance(voices, dict)
    premade = sorted([v for v in voices.get("voices", []) if v.get("category") == "premade"],
                     key=lambda v: v.get("name", ""))
    print(f"premade voices available: {len(premade)}")
    picks = premade[:3]
    for v in picks:
        print(f"  using built-in voice: {v.get('name')} "
              f"(labels={v.get('labels', {})}) — id kept in local manifest only")

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    per_voice = [0] * len(picks)
    for i, (lang, text) in enumerate(TEXTS):
        v = picks[i % len(picks)]
        payload = json.dumps({
            "text": text, "model_id": MODEL,
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}}).encode()
        req = urllib.request.Request(
            f"https://api.elevenlabs.io/v1/text-to-speech/{v['voice_id']}?output_format=mp3_22050_32",
            data=payload,
            headers={"xi-api-key": key, "Content-Type": "application/json",
                     "Accept": "audio/mpeg"})
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                audio = r.read()
        except Exception as exc:
            print(f"clip {i} ({lang}) FAILED: {str(exc)[:150]} — continuing (documented)")
            rows.append({"clip_id": f"el_{i:02d}", "status": f"failed: {type(exc).__name__}",
                         "lang": lang, "voice": v.get("name"), "chars": len(text)})
            continue
        p = OUT / f"el_{i:02d}.mp3"
        p.write_bytes(audio)
        per_voice[i % len(picks)] += 1
        rows.append({"clip_id": f"el_{i:02d}", "status": "ok", "lang": lang,
                     "voice": v.get("name"), "voice_id": v["voice_id"],
                     "chars": len(text), "file": str(p.relative_to(ROOT))})
        print(f"clip {i} ({lang}, {v.get('name')}): {len(audio)} bytes")
        if sum(r.get("chars", 0) for r in rows if r["status"] == "ok") > MAX_CHARS:
            print("STOP: char cap reached mid-run.")
            break

    import csv

    new = [r for r in rows if "chars" in r]
    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["clip_id", "status", "lang", "voice",
                                          "voice_id", "chars", "file"])
        w.writeheader()
        w.writerows(rows)
    used_after, _ = usage(key)
    ok = sum(1 for r in rows if r.get("status") == "ok")
    spent = sum(r.get("chars", 0) for r in rows if r.get("status") == "ok")
    print(f"DONE: ok={ok}/{len(TEXTS)} chars_spent~{spent} "
          f"account used {used_before} -> {used_after} (limit {limit})")
    print("terms note: ElevenLabs API output; free-tier account; clips are cached "
          "locally for evaluation only, never redistributed or committed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
