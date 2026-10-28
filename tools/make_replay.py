"""Self-made replay data: play a clip through speakers, re-record with the mic.

Labels output HUMAN_RECORDED_REPLAY with exact provenance (room, distance,
devices). Needs `pip install sounddevice` (numpy already required).

Usage:
  python tools/make_replay.py --in data/raw/LibriSpeech/test-clean --out data/replay --room bedroom --distance-m 2
  python tools/make_replay.py --in path/to/clip.wav --out data/replay --room office --distance-m 1

Without sounddevice (or without speakers/mic), the script prints setup
instructions and exits 2 — it never fabricates replay audio. Simulated RIR
replay augmentation lives in src/rafnet/soc/augment.py instead.
"""
from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path


def _need_sounddevice():
    try:
        import sounddevice as sd  # type: ignore
        return sd
    except ImportError:
        print("sounddevice is not installed. Run: pip install sounddevice", file=sys.stderr)
        print("You also need numpy (already a dependency) plus working speakers and a mic.", file=sys.stderr)
        raise SystemExit(2)


def play_and_record(path: Path, seconds: float, samplerate: int, out_path: Path) -> None:
    import numpy as np

    sd = _need_sounddevice()
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    print(f"PLAYING {path.name} ({len(audio)/sr:.1f}s) — stay quiet, recording...")
    recorded = sd.playrec(audio, samplerate=sr, channels=1, blocking=True)
    sf.write(str(out_path), recorded, sr)
    print(f"recorded {out_path.name}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", default="data/replay")
    ap.add_argument("--room", default="unknown-room")
    ap.add_argument("--distance-m", type=float, default=1.0)
    ap.add_argument("--max-clips", type=int, default=20)
    args = ap.parse_args()

    _need_sounddevice()  # fail fast with instructions
    src = Path(args.inp)
    files = sorted(src.rglob("*.wav")) + sorted(src.rglob("*.flac"))
    if src.is_file():
        files = [src]
    if not files:
        print(f"no audio found under {src}", file=sys.stderr)
        return 2
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for i, path in enumerate(files[: args.max_clips]):
        import soundfile as sf

        info = sf.info(str(path))
        dest = out / f"replay_{args.room}_{args.distance_m:g}m_{i:03d}_{path.stem}.wav"
        play_and_record(path, 0.0, info.samplerate, dest)
        manifest_rows.append(
            f"{dest},HUMAN_RECORDED_REPLAY,replay-sim,room-playback,en,unknown,"
            f"self-made-replay,train,pcm,mono"
        )
    man = out / "replay_manifest.csv"
    man.write_text(
        "path,label_origin,generator_family,generator_name,language,speaker_id,"
        "source_dataset,split,codec,channel\n" + "\n".join(manifest_rows) + "\n"
    )
    print(f"manifest: {man} ({len(manifest_rows)} clips, room={args.room})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
