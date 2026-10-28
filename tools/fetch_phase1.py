"""Phase 1 fetch: ungated public HUMAN speech (direct HTTP only).

No HF_TOKEN / Kaggle needed. Every download is size-checked FIRST (HEAD /
Hub metadata) against two rules:
  1. total downloaded bytes across all datasets <= 15 GB budget
  2. C: free space must never drop below 10 GB (checked before each file;
     archives + extracted originals are deleted after clip selection).

Usage:
  python tools/fetch_phase1.py --plan    # verify URLs + sizes only
  python tools/fetch_phase1.py --do      # download, select, convert, manifest

Outputs (all under data/, git-ignored):
  data/_dl/                        archives (deleted after use)
  data/phase1/human/<src>/*.wav   16 kHz mono clips, 2-10 s
  data/manifest_phase1.csv        clip_id, source, speaker_id, license, ...
  data/phase1/download_log.json   bytes per file vs 15 GB budget
"""
from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DL_DIR = ROOT / "data" / "_dl"
WAV_DIR = ROOT / "data" / "phase1" / "human"
MANIFEST = ROOT / "data" / "manifest_phase1.csv"
LOG = ROOT / "data" / "phase1" / "download_log.json"

BUDGET_BYTES = 15 * 10**9
MIN_FREE_BYTES = 10 * 10**9
WORK_MARGIN = 2 * 10**9  # extraction headroom

import imageio_ffmpeg

FFMPEG = shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()

# (key, kind, url, expected_bytes, license, clip_quota, max_per_speaker)
SOURCES = [
    ("fleurs_hi_dev", "hf", "google/fleurs:data/hi_in/audio/dev.tar.gz", 132_000_000,
     "CC-BY-4.0", 700, 60),
    ("fleurs_hi_test", "hf", "google/fleurs:data/hi_in/audio/test.tar.gz", 249_000_000,
     "CC-BY-4.0", 700, 60),
    ("fleurs_ta_dev", "hf", "google/fleurs:data/ta_in/audio/dev.tar.gz", 239_000_000,
     "CC-BY-4.0", 700, 60),
    ("fleurs_ta_test", "hf", "google/fleurs:data/ta_in/audio/test.tar.gz", 407_000_000,
     "CC-BY-4.0", 700, 60),
    ("fleurs_en_train", "hf", "google/fleurs:data/en_us/audio/train.tar.gz", 1_381_000_000,
     "CC-BY-4.0", 1500, 60),
    ("slr65_ta_f", "http", "https://www.openslr.org/resources/65/ta_in_female.zip", 769_000_000,
     "CC-BY-SA-4.0", 700, 8),
    ("slr65_ta_m", "http", "https://www.openslr.org/resources/65/ta_in_male.zip", 603_000_000,
     "CC-BY-SA-4.0", 700, 8),
    ("slr118_hi_dev", "http", "https://asr.iitm.ac.in/Gramvaani/NEW/GV_Dev_5h.tar.gz", 98_000_000,
     "GramVaani-academic-only", 1500, 6),
    ("slr118_hi_eval", "http", "https://asr.iitm.ac.in/Gramvaani/NEW/GV_Eval_3h.tar.gz", 62_000_000,
     "GramVaani-academic-only", 1000, 6),
    ("slr12_dev_clean", "http", "https://www.openslr.org/resources/12/dev-clean.tar.gz", 337_000_000,
     "CC-BY-4.0", 700, 30),
    ("slr12_dev_other", "http", "https://www.openslr.org/resources/12/dev-other.tar.gz", 314_000_000,
     "CC-BY-4.0", 700, 30),
]


def free_bytes() -> int:
    return shutil.disk_usage(str(ROOT)).free


def log() -> dict:
    if LOG.is_file():
        return json.loads(LOG.read_text(encoding="utf-8"))
    return {"files": {}, "total_bytes": 0}


def save_log(d: dict) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text(json.dumps(d, indent=1), encoding="utf-8")


def head_size(url: str) -> int | None:
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            n = r.headers.get("Content-Length")
            return int(n) if n else None
    except Exception as exc:
        print(f"  HEAD failed: {exc}")
        return None


def fetch_hf(repo: str, path: str, dest: Path) -> int:
    from huggingface_hub import hf_hub_download

    got = hf_hub_download(repo, path, repo_type="dataset", local_dir=DL_DIR,
                          local_dir_use_symlinks=False)
    shutil.copy(got, dest)
    try:  # drop the Hub cache copy; dest is the canonical archive (deleted after use)
        os.unlink(got)
    except Exception:
        pass
    return dest.stat().st_size


def fetch_http(url: str, dest: Path) -> int:
    have = dest.stat().st_size if dest.is_file() else 0
    headers = {"User-Agent": "RAFNet-phase1"}
    mode = "wb"
    if have > 0:
        headers["Range"] = f"bytes={have}-"
        mode = "ab"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=3600) as r, open(dest, mode) as f:
        if have > 0 and r.status == 200:
            print("  server ignored Range; restarting from 0")
            f.seek(0)
            f.truncate()
        while True:
            chunk = r.read(4 * 1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    return dest.stat().st_size


def audio_duration(path: Path) -> float | None:
    r = subprocess.run([FFMPEG, "-i", str(path)], capture_output=True, text=True, timeout=60)
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", r.stderr)
    if not m:
        return None
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))


def to_16k_wav(src: Path, dest: Path) -> bool:
    r = subprocess.run([FFMPEG, "-y", "-v", "error", "-i", str(src),
                        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(dest)],
                       capture_output=True, text=True, timeout=300)
    return r.returncode == 0 and dest.is_file()


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--do", action="store_true")
    ap.add_argument("--only", default="",
                    help="process a single source key (bounded unit per run)")
    args = ap.parse_args()

    DL_DIR.mkdir(parents=True, exist_ok=True)
    WAV_DIR.mkdir(parents=True, exist_ok=True)
    dlog = log()

    print(f"budget: {BUDGET_BYTES / 1e9:.0f} GB total; used so far {dlog['total_bytes'] / 1e9:.2f} GB; "
          f"C: free {free_bytes() / 1e9:.1f} GB (floor 10 GB)")
    plan = []
    for key, kind, url, expect, lic, quota, _ in SOURCES:
        if kind == "hf":
            repo, path = url.split(":", 1)
            try:
                from huggingface_hub import hf_hub_url

                size = head_size(hf_hub_url(repo, path, repo_type="dataset"))
            except Exception as exc:
                size = None
                print(f"{key}: size check failed: {exc}")
        else:
            size = head_size(url)
        size = size or expect
        done = key in dlog["files"]
        plan.append((key, size, lic, quota, done))
        print(f"{key}: {size / 1e9:.2f} GB license={lic} quota={quota} "
              f"{'DONE' if done else 'pending'}")
    tot = sum(s for _, s, _, _, done in plan if not done)
    print(f"remaining to fetch: {tot / 1e9:.2f} GB; "
          f"budget headroom {(BUDGET_BYTES - dlog['total_bytes']) / 1e9:.2f} GB")
    if BUDGET_BYTES - dlog["total_bytes"] - tot < 0:
        print("REFUSING: plan exceeds 15 GB budget.")
        return 4
    if args.plan or not args.do:
        return 0

    # ---- download + select ----
    manifest_rows: list[dict] = []
    if MANIFEST.is_file():
        with open(MANIFEST, newline="", encoding="utf-8") as f:
            manifest_rows = list(csv.DictReader(f))
    have = {r["clip_id"] for r in manifest_rows}

    for (key, kind, url, expect, lic, quota, max_spk) in SOURCES:
        if args.only and key != args.only:
            continue
        if key in dlog["files"] and any(r.get("source") == key for r in manifest_rows):
            print(f"{key}: already fetched, skipping download (clips kept).")
            continue
        if key in dlog["files"]:
            # downloaded earlier but selection never finished: resume selection
            dest = DL_DIR / Path(url.split(":")[-1]).name
            if dest.is_file():
                print(f"{key}: resuming selection from existing archive...")
                select_clips(key, dest, lic, quota, max_spk, manifest_rows, have)
                try:
                    dest.unlink(missing_ok=True)
                    shutil.rmtree(DL_DIR / (key + "_x"), ignore_errors=True)
                except Exception:
                    pass
                with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
                    w.writeheader()
                    w.writerows(manifest_rows)
                print(f"  manifest now {len(manifest_rows)} rows; C: free {free_bytes() / 1e9:.1f} GB")
                continue
            else:
                print(f"{key}: archive missing, re-downloading (bytes counted once).")
                del dlog["files"][key]
                dlog["total_bytes"] = sum(v["bytes"] for v in dlog["files"].values())
                save_log(dlog)
            print(f"{key}: already fetched, skipping download (clips kept).")
            continue
        size = dict((k, s) for k, s, _, _, _ in plan)[key]
        if dlog["total_bytes"] + size > BUDGET_BYTES:
            print(f"STOP: {key} would exceed 15 GB budget "
                  f"({(dlog['total_bytes'] + size) / 1e9:.2f} GB). Tell user.")
            return 5
        if free_bytes() - size - WORK_MARGIN < MIN_FREE_BYTES:
            print(f"STOP: {key} would push C: under 10 GB free. Tell user.")
            return 6
        dest = DL_DIR / Path(url.split(":")[-1]).name
        print(f"downloading {key} ({size / 1e9:.2f} GB)...")
        try:
            if kind == "hf":
                repo, path = url.split(":", 1)
                got = fetch_hf(repo, path, dest)
            else:
                got = fetch_http(url, dest)
        except Exception as exc:
            print(f"FAIL {key}: {type(exc).__name__}: {exc}")
            continue
        dlog["files"][key] = {"bytes": got, "license": lic}
        dlog["total_bytes"] += got
        save_log(dlog)
        print(f"  got {got / 1e9:.2f} GB; budget used {dlog['total_bytes'] / 1e9:.2f}/15 GB; "
              f"C: free {free_bytes() / 1e9:.1f} GB")
        select_clips(key, dest, lic, quota, max_spk, manifest_rows, have)
        # free archive + extracted tree
        try:
            dest.unlink(missing_ok=True)
            xt = DL_DIR / (key + "_x")
            shutil.rmtree(xt, ignore_errors=True)
        except Exception:
            pass
        with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
            w.writeheader()
            w.writerows(manifest_rows)
        print(f"  manifest now {len(manifest_rows)} rows; C: free {free_bytes() / 1e9:.1f} GB")

    print(f"DONE. budget used {dlog['total_bytes'] / 1e9:.2f}/15 GB.")
    return 0


def iter_audio_files(xt: Path) -> list[Path]:
    exts = {".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".webm"}
    return sorted(p for p in xt.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def speaker_of(key: str, path: Path, xt: Path) -> str:
    rel = path.relative_to(xt).as_posix()
    if key.startswith("slr12"):
        # LibriSpeech/<split>/<speaker>/<chapter>/<utt>.flac
        parts = rel.split("/")
        return f"libri-{parts[-3]}" if len(parts) >= 3 else "libri-?"
    if key.startswith("slr118"):
        # GramVaani: use parent dir / file stem prefix as caller proxy
        return f"gv-{path.parent.name}-{path.stem.split('_')[0]}"
    if key.startswith("slr65"):
        return f"tamil-crowd-{path.stem[:12]}"
    if key.startswith("fleurs"):
        return f"fleurs-{key.split('_')[1]}-{path.stem[:12]}"
    return f"{key}-{path.stem[:12]}"


def select_clips(key: str, archive: Path, lic: str, quota: int, max_spk: int,
                 manifest_rows: list[dict], have: set[str]) -> None:
    xt = DL_DIR / (key + "_x")
    xt.mkdir(exist_ok=True)
    print(f"  extracting {archive.name}...")
    try:
        if archive.suffix == ".zip":
            with zipfile.ZipFile(archive) as z:
                z.extractall(xt)
        else:
            with tarfile.open(archive) as t:
                t.extractall(xt, filter="data")
    except Exception as exc:
        print(f"  EXTRACT FAIL {key}: {exc}")
        return
    cands = iter_audio_files(xt)
    print(f"  candidate audio files: {len(cands)}")
    outdir = WAV_DIR / key
    outdir.mkdir(parents=True, exist_ok=True)
    kept = 0
    per_spk: dict[str, int] = {}
    for p in cands:
        if kept >= quota:
            break
        spk = speaker_of(key, p, xt)
        if per_spk.get(spk, 0) >= max_spk:
            continue
        try:
            dur = audio_duration(p)
        except Exception:
            continue
        if dur is None or not (2.0 <= dur <= 12.0):
            continue
        cid = f"{key}_{kept:05d}"
        if cid in have:
            kept += 1
            continue
        dest = outdir / (cid + ".wav")
        # crop/pad-free: take up to first 10 s via -t
        r = subprocess.run([FFMPEG, "-y", "-v", "error", "-i", str(p),
                            "-t", "10", "-ar", "16000", "-ac", "1",
                            "-c:a", "pcm_s16le", str(dest)],
                           capture_output=True, text=True, timeout=300)
        if r.returncode != 0 or not dest.is_file():
            continue
        manifest_rows.append({"clip_id": cid, "source": key, "speaker_id": spk,
                              "license": lic, "orig_path": p.relative_to(xt).as_posix(),
                              "duration_s": round(dur if dur <= 10 else 10.0, 2),
                              "label": "real", "y": 0})
        have.add(cid)
        per_spk[spk] = per_spk.get(spk, 0) + 1
        kept += 1
    print(f"  kept {kept} clips, {len(per_spk)} speakers (cap {max_spk}/spk)")


if __name__ == "__main__":
    raise SystemExit(main())
