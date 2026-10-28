"""Guide + helper for recording YOUR OWN assistant-voice sessions.

Why this exists: ChatGPT voice / Gemini Live / Meta AI voices are not in
public datasets, and scraping them is off-limits. Record your own sessions
with loopback capture, then drop the files into samples/assistant/<name>/.

Setup (Windows, one time):
  1. Install a virtual-audio device (e.g. VB-Audio Virtual Cable) OR enable
     "Stereo Mix" (Sound settings -> Recording -> Show Disabled Devices).
  2. Route the assistant app output to that device; set it as the recording
     source below.
  3. pip install sounddevice numpy soundfile

Usage:
  python tools/record_assistant_voice.py --list
  python tools/record_assistant_voice.py --device 3 --seconds 30 --name gemini --out samples/assistant/gemini

Rules: your own sessions only. Speak and let the assistant answer; keep 2-10 s
per saved clip (split afterwards). Note the assistant name/date in the
filename. Nothing is uploaded anywhere by this script.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--name", default="assistant")
    ap.add_argument("--out", default="samples/assistant")
    args = ap.parse_args()
    try:
        import sounddevice as sd  # type: ignore
    except ImportError:
        print("pip install sounddevice numpy soundfile first (see module docstring).", file=sys.stderr)
        return 2
    if args.list:
        print(sd.query_devices())
        return 0
    if args.device is None:
        print("--list devices, then pass --device N (your loopback/cable input).", file=sys.stderr)
        return 2
    import numpy as np
    import soundfile as sf

    out = Path(args.out) / args.name
    out.mkdir(parents=True, exist_ok=True)
    print(f"Recording {args.seconds}s from device {args.device} — play the assistant now...")
    rec = sd.rec(int(args.seconds * 16000), samplerate=16000, channels=1,
                 dtype="float32", device=args.device, blocking=True)
    dest = out / f"{args.name}_loopback_{int(args.seconds)}s.wav"
    sf.write(str(dest), np.asarray(rec, dtype=np.float32), 16000)
    print(f"saved {dest} — split into 2-10 s clips and label generator in data/manifest.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
