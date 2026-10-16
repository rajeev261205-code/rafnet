"""Generate local TTS clips with Windows SAPI voices (offline, no keys).

Writes labelled microsoft-SAPI-like TTS wav files for DIAGNOSIS/eval use:
  samples/win_tts/<voice>_<idx>.wav  (16 kHz mono WAV)

Voices present on this box: Microsoft David Desktop (en-US), Microsoft Zira
Desktop (en-US). Requires Windows + System.Speech (runs under Windows
PowerShell with python launching powershell.exe).

Usage:
  python tools/make_win_tts.py [--out samples/win_tts] [--rate 0]
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

TEXTS = [
    "Hello, this is a test of the text to speech system. How are you doing today?",
    "Your account will be reviewed shortly. Please stay on the line for verification.",
    "The weather in Chennai tomorrow will be partly cloudy with a high of thirty four degrees.",
    "Vanakkam. Indha oru makkal sevai amaippu. Ungal karuthu mukkiyam.",
    "Please do not share your one time password with anyone who calls you.",
    "Namaste, main aapki sahayata ke liye yahaan hoon. Kripya apna vivaran dein.",
]

PS = r"""
Add-Type -AssemblyName System.Speech;
$sp = New-Object System.Speech.Synthesis.SpeechSynthesizer;
$sp.SelectVoice('{voice}');
$sp.Rate = {rate};
$sp.SetOutputToWaveFile('{path}');
$sp.Speak('{text}');
$sp.Dispose();
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="samples/win_tts")
    ap.add_argument("--rate", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    count = 0
    for voice in ("Microsoft David Desktop", "Microsoft Zira Desktop"):
        for i, text in enumerate(TEXTS):
            if voice.startswith("Microsoft David") and i % 2:
                continue  # David: even texts; Zira: all texts (variety, not bloat)
            path = (out / f"sapi_{voice.split()[1].lower()}_{i}.wav").resolve()
            safe = text.replace("'", "''")
            script = PS.format(voice=voice, rate=args.rate, path=str(path), text=safe)
            r = subprocess.run(
                ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
                capture_output=True, text=True, timeout=120,
            )
            if r.returncode != 0 or not path.is_file():
                print(f"FAIL {voice} #{i}: {r.stderr.strip()[:200]}")
                continue
            count += 1
            print(f"wrote {path.name} ({path.stat().st_size} bytes)")
    # manifest rows appended by tools/make_manifest.py conventions
    print(f"done: {count} clips in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
