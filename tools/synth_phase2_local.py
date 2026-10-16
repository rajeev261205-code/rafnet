"""Phase 2 local training fakes (no keys, mostly offline).

Engines (each voice = a generator id `engine:voice`):
  edge  : edge-tts service, 6 voices (en-IN x2, hi-IN x2, ta-IN x2), rate varied
  piper : local Piper onnx (en amy/ryan, hi pratham/priyamvada/rohan), 5 voices
  sapi  : Windows SAPI David + Zira via tools/make_win_tts.py pattern
  vc    : crude librosa VC (pitch-shift / time-stretch) of public Phase-1 real

UNSEEN holdout (whole generators, never train/tune/select):
  elevenlabs (all, separate TEST set), piper:hi-rohan, edge:ta-Valluvar.

Usage: python tools/synth_phase2_local.py --engine edge|piper|sapi|vc [--only VOICE]
Clips: 16 kHz mono wav 3-12 s to data/phase2/fake/<engine>/, manifest rows to
data/manifest_phase2.csv (local, git-ignored). Disk floor 10 GB enforced.
"""
from __future__ import annotations

import csv
import random
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "data" / "phase2" / "fake"
MANIFEST = ROOT / "data" / "manifest_phase2.csv"
MIN_FREE = 10 * 10**9

import imageio_ffmpeg

FFMPEG = shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()

EDGE_VOICES = ["en-IN-NeerjaNeural", "en-IN-PrabhatNeural",
               "hi-IN-SwaraNeural", "hi-IN-MadhurNeural",
               "ta-IN-PallaviNeural", "ta-IN-ValluvarNeural"]
PIPER_VOICES = {"piper:en-amy": "en/en_US/amy/medium/en_US-amy-medium.onnx",
                "piper:en-ryan": "en/en_US/ryan/medium/en_US-ryan-medium.onnx",
                "piper:hi-pratham": "hi/hi_IN/pratham/medium/hi_IN-pratham-medium.onnx",
                "piper:hi-priyamvada": "hi/hi_IN/priyamvada/medium/hi_IN-priyamvada-medium.onnx",
                "piper:hi-rohan": "hi/hi_IN/rohan/medium/hi_IN-rohan-medium.onnx"}
SAPI_VOICES = ["Microsoft David Desktop", "Microsoft Zira Desktop"]
UNSEEN = {"elevenlabs", "piper:hi-rohan", "edge:ta-IN-ValluvarNeural"}

# (lang, text) bank — scam-style + neutral, sized for 3-10 s synthesis
TEXTS_EN = [
    "Dear customer, your bank account will be blocked today. Share your OTP now to verify your KYC immediately.",
    "Your SIM card will be deactivated tonight. Press one to speak with our verification officer right away.",
    "A parcel in your name is held at customs. Pay the clearance fee now or it will be returned.",
    "Congratulations, you have won fifty thousand rupees. Share your account details to claim the prize.",
    "Your electricity bill is overdue. Pay tonight or the connection will be disconnected without warning.",
    "We found suspicious activity on your card. Read out the full number so we can block the fraud.",
    "Your KYC papers have expired. Update them today or all account services will stop working.",
    "A tax refund is waiting in your name. Enter your bank account number to receive the money.",
    "The morning bus was late again, so we walked the last mile to the office in the rain.",
    "She keeps a small garden with roses and mint behind the house near the old well.",
    "The new bridge across the river will open to traffic before the festival season begins.",
    "Please bring your identity card and a passport photo to the enrolment centre tomorrow.",
    "Children play cricket in the street every evening until their mothers call them home.",
    "The train from the city arrives at platform two fifteen minutes behind schedule today.",
    "Fishermen return with their boats before sunrise when the sea turns rough and grey.",
]
TEXTS_HI = [
    "Namaste, aapka bank khata aaj band ho jayega. Kripya OTP batakar apna KYC poora karein.",
    "Aapke SIM ka KYC adhura hai. Turant update karein warna seva band ho jayegi.",
    "Aapka bijli bill lambit hai. Aaj bhugtan karein warna connection kaat diya jayega.",
    "Badhaai ho, aapne pachaas hazaar rupaye jeete hain. Daave ke liye apni jaankari dein.",
    "Aapke card par sandigdh len-den mila hai. Suraksha ke liye poora number batayein.",
    "Aapke KYC kagaz ki avadhi samaapt ho gayi hai. Aaj hi naye kagaz jama karein.",
    " Mausam saaf hai aur thandi hawa chal rahi hai. Bachche maidan mein khel rahe hain.",
    "Gaon ke mele mein is baar bahut bheed thi. Log door door se aaye the.",
    "Nayi sadak banne se gaon aur sheher ke beech safar aasaan ho jayega.",
    "Kripya apna pehchaan patra aur photu kal kendra par lekar aayein.",
    "Machuare subah suraj nikalne se pehle apni naav lekar samundar jaate hain.",
    "Bazaar mein taazi sabziyaan aur phal aaye hain. Daam bhi theek hain.",
]
TEXTS_TA = [
    "Vanakkam, ungal vangi kanakku indru adaikkappadum. OTP-ai koduththu KYC-ai mudiyungal.",
    "Ungal SIM-in KYC mudiyavillai. Udane puthuppiyungal, illaiyendraal sevai niruththappadum.",
    "Minsaara kattanam seluththavillai. Indru seluththungal, illaiyendraal thundippu seiyappadum.",
    "Vaazhththukkal, aimpathaayiram rubai parisu. Peruvatharku vivaram kodungal.",
    "Ungal attaiyil santhegamaana selavu. Padukaappirkku ennaik kodungal.",
    "Ungal KYC aavanam kaalamudinthathu. Indru puthiya aavanam kodungal.",
    "Kaalaiyil mazhai peythathu, aanaal ippozhuthu vaanilakam thelivaaga ullathu.",
    "Sandhaiyil putthunarchi kaaigarigal vanthullana. Vilaigalum niyayamானவை.",
    "Meenavargal suriyan uthikkum munne padagudan kadalukku selgirargal.",
    "Pullaaigal maalaiyil theruvil cricket vilaiyaadugiraargal magizhchchiyudan.",
    "Palaththirku appaal ulla uurukku puthiya paalam kattappadugirathu.",
    "Thayavu seithu adaiyaala attaiyudan naalai maiyaththirku varavum.",
]


def free_ok() -> bool:
    return shutil.disk_usage(str(ROOT)).free > MIN_FREE


def manifest() -> list[dict]:
    if MANIFEST.is_file():
        import csv as _csv

        return list(_csv.DictReader(open(MANIFEST, newline="", encoding="utf-8")))
    return []


def save_manifest(rows: list[dict]) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def check_wav(p: Path) -> float | None:
    """Return duration if 2-12 s + audible, else None."""
    import soundfile as sf

    try:
        y, sr = sf.read(str(p), dtype="float32")
    except Exception:
        return None
    import numpy as np

    y = np.asarray(y, dtype=np.float32).ravel()
    dur = len(y) / 16000.0
    rms = float(20 * np.log10(np.sqrt((y ** 2).mean() + 1e-12)))
    if 2.0 <= dur <= 12.0 and rms > -45.0:
        return round(dur, 2)
    return None


def to16k(src: Path, dest: Path) -> bool:
    r = subprocess.run([FFMPEG, "-y", "-v", "error", "-i", str(src),
                        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(dest)],
                       capture_output=True, text=True, timeout=300)
    return r.returncode == 0 and dest.is_file()


def texts_for(lang: str, n: int, seed: int) -> list[str]:
    bank = {"en": TEXTS_EN, "hi": TEXTS_HI, "ta": TEXTS_TA}[lang]
    rng = random.Random(seed)
    out = []
    i = 0
    while len(out) < n:
        out.append(bank[i % len(bank)])
        i += 1
    rng.shuffle(out)
    return out


def run_edge(only: str, quota: int) -> None:
    import asyncio

    import edge_tts

    rows, have = manifest(), None
    have = {r["clip_id"] for r in rows}
    n = 0
    for vi, voice in enumerate(EDGE_VOICES):
        if only and voice != only:
            continue
        lang = voice.split("-")[0]
        gid = f"edge:{voice}"
        split = "test_unseen" if gid in UNSEEN else "train"
        for ti, text in enumerate(texts_for(lang, quota, 1000 + vi)):
            if not free_ok():
                print("STOP: disk floor"); return
            cid = f"edge-{voice.split('-')[2].replace('Neural', '')}-{ti:03d}"
            if cid in have:
                continue
            rate = "+0%" if ti % 3 == 0 else ("+10%" if ti % 3 == 1 else "-10%")
            tmp = OUT / "edge" / f"_{cid}.mp3"
            tmp.parent.mkdir(parents=True, exist_ok=True)
            try:
                asyncio.run(edge_tts.Communicate(
                    text, voice, rate=rate).save(str(tmp)))
            except Exception as exc:
                print(f"edge {voice} t{ti} FAIL: {str(exc)[:120]}")
                tmp.unlink(missing_ok=True)
                continue
            dest = OUT / "edge" / (cid + ".wav")
            if to16k(tmp, dest):
                tmp.unlink(missing_ok=True)
                d = check_wav(dest)
                if d:
                    rows.append({"clip_id": cid, "engine": "edge", "generator": gid,
                                 "lang": lang, "split": split, "duration_s": d,
                                 "file": str(dest.relative_to(ROOT))})
                    have.add(cid)
                    n += 1
                else:
                    dest.unlink(missing_ok=True)
            print(f"edge {voice}: {n} kept", flush=True)
    save_manifest(rows)
    print(f"edge done: +{n}")


def run_piper(only: str, quota: int) -> None:
    from huggingface_hub import hf_hub_download
    from piper import PiperVoice

    vdir = ROOT / "data" / "_dl" / "piper"
    vdir.mkdir(parents=True, exist_ok=True)
    rows = manifest()
    have = {r["clip_id"] for r in rows}
    n = 0
    for gid, rel in PIPER_VOICES.items():
        if only and gid != only:
            continue
        lang = "hi" if ":hi-" in gid else "en"
        split = "test_unseen" if gid in UNSEEN else "train"
        onnx = vdir / Path(rel).name
        cfg = vdir / (Path(rel).name + ".json")
        if not onnx.is_file():
            hf_hub_download("rhasspy/piper-voices", rel, repo_type="model",
                            local_dir=vdir, local_dir_use_symlinks=False)
            hf_hub_download("rhasspy/piper-voices", rel + ".json", repo_type="model",
                            local_dir=vdir, local_dir_use_symlinks=False)
            # flatten nested dirs
            for p in vdir.rglob(Path(rel).name):
                if p != onnx:
                    shutil.copy(p, onnx)
            for p in vdir.rglob(Path(rel).name + ".json"):
                if p != cfg:
                    shutil.copy(p, cfg)
        voice = PiperVoice.load(str(onnx), config_path=str(cfg))
        import soundfile as sf

        for ti, text in enumerate(texts_for(lang, quota, 2000 + len(gid))):
            if not free_ok():
                print("STOP: disk floor"); return
            cid = f"{gid.replace(':', '-')}-{ti:03d}"
            if cid in have:
                continue
            tmp = OUT / "piper" / f"_{cid}.wav"
            tmp.parent.mkdir(parents=True, exist_ok=True)
            try:
                chunks = list(voice.synthesize(text))
                import numpy as np

                y = np.concatenate([c.audio_float_array for c in chunks]).astype(np.float32)
                sr = voice.config.sample_rate
                sf.write(str(tmp), y, sr)
            except Exception as exc:
                print(f"piper {gid} t{ti} FAIL: {str(exc)[:120]}")
                tmp.unlink(missing_ok=True)
                continue
            dest = OUT / "piper" / (cid + ".wav")
            if to16k(tmp, dest):
                tmp.unlink(missing_ok=True)
                d = check_wav(dest)
                if d:
                    rows.append({"clip_id": cid, "engine": "piper", "generator": gid,
                                 "lang": lang, "split": split, "duration_s": d,
                                 "file": str(dest.relative_to(ROOT))})
                    have.add(cid)
                    n += 1
                else:
                    dest.unlink(missing_ok=True)
        print(f"piper {gid}: kept (running total +{n})", flush=True)
    save_manifest(rows)
    print(f"piper done: +{n}")


PS_TMPL = """
Add-Type -AssemblyName System.Speech;
$sp = New-Object System.Speech.Synthesis.SpeechSynthesizer;
$sp.SelectVoice('{voice}');
$sp.Rate = {rate};
$sp.SetOutputToWaveFile('{path}');
$sp.Speak(@'
{text}
'@);
$sp.Dispose();
"""


def run_sapi(only: str, quota: int) -> None:
    rows = manifest()
    have = {r["clip_id"] for r in rows}
    n = 0
    for voice in SAPI_VOICES:
        if only and voice != only:
            continue
        gid = f"sapi:{voice.split()[1].lower()}"
        texts = texts_for("en", quota, 3000 + len(voice))
        for ti, text in enumerate(texts):
            if not free_ok():
                print("STOP: disk floor"); return
            cid = f"{gid.replace(':', '-')}-{ti:03d}"
            if cid in have:
                continue
            tmp = OUT / "sapi" / f"_{cid}.wav"
            tmp.parent.mkdir(parents=True, exist_ok=True)
            ps = PS_TMPL.format(voice=voice, rate=-2 if ti % 2 else 0,
                                path=str(tmp).replace("'", "''"),
                                text=text.replace("'", "''"))
            r = subprocess.run(["powershell.exe", "-NoProfile", "-Command", ps],
                               capture_output=True, text=True, timeout=300)
            if r.returncode != 0 or not tmp.is_file():
                tmp.unlink(missing_ok=True)
                continue
            dest = OUT / "sapi" / (cid + ".wav")
            if to16k(tmp, dest):
                tmp.unlink(missing_ok=True)
                d = check_wav(dest)
                if d:
                    rows.append({"clip_id": cid, "engine": "sapi", "generator": gid,
                                 "lang": "en", "split": "train", "duration_s": d,
                                 "file": str(dest.relative_to(ROOT))})
                    have.add(cid)
                    n += 1
                else:
                    dest.unlink(missing_ok=True)
        print(f"sapi {voice}: kept (running total +{n})", flush=True)
    save_manifest(rows)
    print(f"sapi done: +{n}")


def run_vc(quota_each: int) -> None:
    import librosa
    import soundfile as sf

    publ = list((ROOT / "data" / "phase1" / "human").rglob("*.wav"))
    rng = random.Random(42)
    rng.shuffle(publ)
    rows = manifest()
    have = {r["clip_id"] for r in rows}
    n = 0
    for setting, fn in [("ps", lambda y, sr: librosa.effects.pitch_shift(y, sr=sr, n_steps=4.0)),
                        ("ts", lambda y, sr: librosa.effects.time_stretch(y, rate=1.12))]:
        gid = f"crudeVC:{setting}"
        for ti in range(quota_each):
            if not free_ok():
                print("STOP: disk floor"); return
            cid = f"vc-{setting}-{ti:03d}"
            if cid in have:
                continue
            src = publ[(ti * 7) % len(publ)]
            try:
                y, _ = sf.read(str(src), dtype="float32")
                import numpy as np

                y = np.asarray(y, dtype=np.float32).ravel()
                v = fn(y, 16000).astype(np.float32)
                if len(v) > 10 * 16000:
                    v = v[:10 * 16000]
                tmp = OUT / "vc" / f"_{cid}.wav"
                tmp.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(tmp), v, 16000)
            except Exception as exc:
                print(f"vc {setting} t{ti} FAIL: {str(exc)[:120]}")
                continue
            dest = OUT / "vc" / (cid + ".wav")
            if to16k(tmp, dest):
                tmp.unlink(missing_ok=True)
                d = check_wav(dest)
                if d:
                    rows.append({"clip_id": cid, "engine": "crudeVC", "generator": gid,
                                 "lang": "mixed", "split": "train", "duration_s": d,
                                 "file": str(dest.relative_to(ROOT)),
                                 "vc_source": src.name})
                    have.add(cid)
                    n += 1
                else:
                    dest.unlink(missing_ok=True)
        print(f"vc {setting}: kept (running total +{n})", flush=True)
    save_manifest(rows)
    print(f"vc done: +{n}")


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=["edge", "piper", "sapi", "vc"])
    ap.add_argument("--only", default="")
    ap.add_argument("--quota", type=int, default=60)
    args = ap.parse_args()
    if not free_ok():
        print("REFUSING: C: free below 10 GB floor.")
        return 3
    if args.engine == "edge":
        run_edge(args.only, args.quota)
    elif args.engine == "piper":
        run_piper(args.only, args.quota)
    elif args.engine == "sapi":
        run_sapi(args.only, args.quota)
    else:
        run_vc(args.quota)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
