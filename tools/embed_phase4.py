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
    args = ap.parse_args()

    items = load_split("train") + load_split("val")
    part = [it for i, it in enumerate(items) if i % args.chunks == args.chunk]
    tag = "clean" if args.epoch < 0 else f"aug{args.epoch}"
    print(f"tag={tag} chunk {args.chunk + 1}/{args.chunks}: {len(part)} clips", flush=True)

    import torch
    from transformers import AutoFeatureExtractor, AutoModel

    feat = AutoFeatureExtractor.from_pretrained(ENCODER, trust_remote_code=False)
    mdl = AutoModel.from_pretrained(ENCODER, trust_remote_code=False)
    mdl.eval()
    for p in mdl.parameters():
        p.requires_grad_(False)

    if args.epoch >= 0:
        sys.path.insert(0, str(ROOT))
        from tools.augment_phase3 import apply_chain, sample_chain
        import zlib

    have: dict[str, int] = {}
    if CACHE.is_file():
        z = np.load(str(CACHE), allow_pickle=True)
        ids_done = set(z["ids"].tolist())
        print(f"cache exists: {len(ids_done)} {tag}-rows done", flush=True)
    else:
        ids_done = set()

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
                inp = feat(arr, sampling_rate=16000, return_tensors="pt", padding=True)
                out = mdl(**{k: v for k, v in inp.items()
                             if k in ("input_values", "attention_mask")},
                          output_hidden_states=True)
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
    print(f"extracted {done} new rows in {time.perf_counter()-t0:.0f}s", flush=True)
    if done:
        old = {}
        if CACHE.is_file():
            z = np.load(str(CACHE), allow_pickle=True)
            old = {k: np.asarray(z[k]) for k in z.files}
        def cat(k, arr):
            a = np.asarray(arr)
            return np.concatenate([old[k], a], axis=0) if k in old else a
        np.savez_compressed(str(CACHE), ids=cat("ids", np.array(ids)),
                            y=cat("y", np.array(ys, dtype=np.int64)),
                            mean=cat("mean", np.stack(means)),
                            std=cat("std", np.stack(stds)))
        print(f"cache now: {len(np.load(str(CACHE), allow_pickle=True)['ids'])} rows", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
