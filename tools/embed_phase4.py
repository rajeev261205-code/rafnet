"""Phase 4.2a: frozen WavLM-base-plus embeddings (layer means+stds, float16).

CPU-sized per the brief (WavLM-large cached but too slow on CPU).
Covers train+val ONLY (test never touched here). Chunked + resumable:
  python tools/embed_phase4.py --chunk 0 --chunks 8   (repeat per chunk)
Cache: data/embeddings_phase4.npz (local, git-ignored) with keys
  ids[N], y[N], mean[L,H] fp16, std[L,H] fp16  (L=13 states, H=768).
Augmented copies: --epoch E applies the Phase-3 seeded chain before
embedding (same chains both classes); stored as separate key sets.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CACHE = ROOT / "data" / "embeddings_phase4.npz"
CHUNK_CACHE = ROOT / "data" / "embed_chunks"
ENCODER = "microsoft/wavlm-base-plus"


def load_split(split: str):
    d = json.loads((ROOT / "data" / "splits_phase4.json").read_text(encoding="utf-8"))
    return d[split]


def load_audio_16k(path: str, max_s: float = 12.0):
    import numpy as np
    import soundfile as sf

    arr, sr = sf.read(path, dtype="float32", always_2d=False)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr.mean(axis=1).astype(np.float32)
    arr = arr.ravel()
    if int(sr) != 16000 and arr.size:
        tgt = max(1, int(round(len(arr) / float(sr) * 16000)))
        arr = np.interp(np.linspace(0, 1, tgt),
                        np.linspace(0, 1, len(arr)), arr).astype(np.float32)
    cap = int(16000 * max_s)
    return (arr[:cap] if len(arr) > cap else arr)


def main() -> int:
    import argparse

    import numpy as np

    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--chunks", type=int, default=8)
    ap.add_argument("--epoch", type=int, default=-1,
                    help="-1 = clean; >=0 = Phase-3 augmented with this epoch seed")
    ap.add_argument("--split", default="train+val",
                    help="which split(s) to embed; 'test' is touched only in Phase 5")
    args = ap.parse_args()

    if args.split == "train+val":
        items = load_split("train") + load_split("val")
    else:
        items = load_split(args.split)
    part = [it for i, it in enumerate(items) if i % args.chunks == args.chunk]
    tag = "clean" if args.epoch < 0 else f"aug{args.epoch}"
    print(f"tag={tag} chunk {args.chunk + 1}/{args.chunks}: {len(part)} clips", flush=True)

    import torch
    from transformers import AutoModel

    # No preprocessor config in the offline cache; replicate the
    # Wav2Vec2FeatureExtractor default (zero-mean / unit-variance).
    mdl = AutoModel.from_pretrained(ENCODER, trust_remote_code=False)
    mdl.eval()
    for p in mdl.parameters():
        p.requires_grad_(False)

    if args.epoch >= 0:
        sys.path.insert(0, str(ROOT))
        from tools.augment_phase3 import apply_chain, sample_chain
        import zlib

    have: dict[str, int] = {}
    CHUNK_CACHE.mkdir(parents=True, exist_ok=True)
    prefix = f"chunk_{args.chunk:02d}_{tag}_"
    ids_done: set[str] = set()
    for pf in sorted(CHUNK_CACHE.glob(prefix + "*.npz")):
        try:
            z = np.load(str(pf), allow_pickle=True)
            ids_done.update(z["ids"].tolist())
        except Exception:
            pass
    print(f"chunk cache: {len(ids_done)} rows done in parts", flush=True)
    import re as _re

    _idx = [int(_re.search(r"part(\d+)\.npz$", p.name).group(1))
            for p in CHUNK_CACHE.glob(prefix + "*.npz")
            if _re.search(r"part(\d+)\.npz$", p.name)]
    part_idx = [max(_idx) + 1 if _idx else 0]  # never overwrite prior runs' parts

    def _save_part(means, stds, ids, ys) -> None:
        import numpy as _np

        pf = CHUNK_CACHE / f"{prefix}part{part_idx[0]}.npz"
        part_idx[0] += 1
        _np.savez_compressed(str(pf), ids=_np.array(ids),
                             y=_np.array(ys, dtype=_np.int64),
                             mean=_np.stack(means), std=_np.stack(stds))
        print(f"  saved part {pf.name}: "
              f"{len(_np.load(str(pf), allow_pickle=True)['ids'])} rows", flush=True)

    means, stds, ids, ys = [], [], [], []
    import time

    t0 = time.perf_counter()
    done = 0
    with torch.inference_mode():
        for it in part:
            key = f"{tag}:{it['clip_id']}"
            if key in ids_done:
                continue
            try:
                arr = load_audio_16k(it["path"])
                if args.epoch >= 0:
                    seed = zlib.crc32(f"{it['clip_id']}@{args.epoch}".encode()) & 0xFFFFFFFF
                    chain = sample_chain(np.random.default_rng(seed))
                    arr = apply_chain(arr, chain, seed)
                inp = torch.from_numpy(
                    ((arr - arr.mean()) / (arr.std() + 1e-7)).astype(np.float32)
                ).unsqueeze(0)
                out = mdl(input_values=inp, output_hidden_states=True)
                hs = torch.stack(out.hidden_states, dim=0).squeeze(1)  # L,T,H
                means.append(hs.float().mean(dim=1).half().numpy())
                stds.append(hs.float().std(dim=1).half().numpy())
                ids.append(key)
                ys.append(it["y"])
                done += 1
            except Exception as exc:
                print(f"ERR {it['clip_id']}: {type(exc).__name__}: {str(exc)[:120]}", flush=True)
            if done % 50 == 0 and done:
                print(f"  {done} new this run ({time.perf_counter()-t0:.0f}s)", flush=True)
            if done % 200 == 0 and done:
                _save_part(means, stds, ids, ys)
                means, stds, ids, ys = [], [], [], []
    print(f"extracted {done} new rows in {time.perf_counter()-t0:.0f}s", flush=True)
    if means:
        _save_part(means, stds, ids, ys)
    n_parts = len(list(CHUNK_CACHE.glob(prefix + "*.npz")))
    print(f"chunk parts on disk: {n_parts}", flush=True)
    return 0


def _save_chunk(cfile: Path, means, stds, ids, ys) -> None:
    import os as _os

    import numpy as _np

    old = {}
    if cfile.is_file():
        try:
            z = _np.load(str(cfile), allow_pickle=True)
            old = {k: _np.asarray(z[k]) for k in z.files}
        except Exception:
            old = {}

    def cat(k, arr):
        a = _np.asarray(arr)
        return _np.concatenate([old[k], a], axis=0) if k in old else a

    tmp = cfile.with_suffix(".tmp.npz")
    _np.savez_compressed(str(tmp), ids=cat("ids", _np.array(ids)),
                         y=cat("y", _np.array(ys, dtype=_np.int64)),
                         mean=cat("mean", _np.stack(means)),
                         std=cat("std", _np.stack(stds)))
    import time as _t

    for _ in range(5):  # Windows AV can briefly lock fresh files
        try:
            if cfile.is_file():
                cfile.unlink()
            _os.replace(tmp, cfile)  # atomic: timeout kill can never corrupt
            break
        except PermissionError:
            _t.sleep(2)
    else:
        _os.replace(tmp, cfile)
    print(f"  saved chunk rows: {len(_np.load(str(cfile), allow_pickle=True)['ids'])}",
          flush=True)


def _save(done: int, means, stds, ids, ys) -> None:
    import numpy as _np

    old = {}
    if CACHE.is_file():
        z = _np.load(str(CACHE), allow_pickle=True)
        old = {k: _np.asarray(z[k]) for k in z.files}

    def cat(k, arr):
        a = _np.asarray(arr)
        return _np.concatenate([old[k], a], axis=0) if k in old else a

    _np.savez_compressed(str(CACHE), ids=cat("ids", _np.array(ids)),
                         y=cat("y", _np.array(ys, dtype=_np.int64)),
                         mean=cat("mean", _np.stack(means)),
                         std=cat("std", _np.stack(stds)))
    print(f"  saved ({done} this run); cache now: "
          f"{len(_np.load(str(CACHE), allow_pickle=True)['ids'])} rows", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
