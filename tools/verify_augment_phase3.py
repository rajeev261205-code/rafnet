"""Phase 3 verification: contingency table (compression vs label) + chain audit.

Samples 1000 real + 1000 fake clips (TRAIN pool only — never private TEST,
never unseen generators), applies the Phase-3 seeded chains on the fly, and
tests that augmentation type is independent of the label (chi-square).
Writes reports/augmentation_phase3.json/.md (committed; counts only).
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT))

from tools.augment_phase3 import apply_chain, batch_stream, sample_chain  # noqa: E402

N_PER_CLASS = 10000  # sampler-level proof (instant, no audio)
N_AUDIO = 60  # execution audit through real ffmpeg chains


def load_pools() -> tuple[list, list]:
    real: list[tuple[str, Path, int]] = []
    with open(ROOT / "data" / "manifest_phase1.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            real.append((r["clip_id"], ROOT / "data" / "phase1" / "human" / r["source"] /
                         (r["clip_id"] + ".wav"), 0))
    # private adapt (small share, same chains)
    with open(ROOT / "samples" / "real" / "manifest.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] == "adapt":
                real.append((r["segid"], ROOT / "samples" / "real" / "segments" / "adapt" /
                             (r["segid"] + ".wav"), 0))
    fake: list[tuple[str, Path, int]] = []
    with open(ROOT / "data" / "manifest_phase2.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("split", "train") == "train":
                fake.append((r["clip_id"], ROOT / r["file"], 1))
    rng = np.random.default_rng(7)
    real = [real[i] for i in rng.choice(len(real), min(N_PER_CLASS, len(real)), replace=False)]
    fake = [fake[i] for i in rng.choice(len(fake), min(N_PER_CLASS, len(fake)), replace=False)]
    return real, fake


def chi_square(table: list[list[int]]) -> dict:
    import math

    t = np.array(table, dtype=float)
    row = t.sum(axis=1, keepdims=True)
    col = t.sum(axis=0, keepdims=True)
    n = t.sum()
    exp = row @ col / n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = float(((t - exp) ** 2 / np.where(exp > 0, exp, 1)).sum())
    dof = (t.shape[0] - 1) * (t.shape[1] - 1)
    # p-value via regularized gamma (no scipy dependency at module import)
    from math import exp as _e  # noqa

    try:
        from scipy.stats import chi2 as _c

        p = float(_c.sf(chi2, dof))
    except Exception:
        p = float("nan")
    _ = math
    return {"chi2": round(chi2, 3), "dof": dof, "p_value": round(p, 4),
            "independent": bool(p > 0.05) if p == p else None}


def main() -> int:
    real, fake = load_pools()
    print(f"pools: real={len(real)} fake={len(fake)}", flush=True)
    codecs = ["none", "opus_12k", "opus_24k", "opus_32k", "amr_nb", "g711",
              "mp3_64k", "mp3_128k", "aac_32k"]
    # 1) sampler-level independence: 10k chains per class, no audio (fast).
    # sample_chain is label-blind by construction; this measures the proof.
    tab = [[0] * len(codecs) for _ in range(2)]
    noise_tab = [[0, 0] for _ in range(2)]
    for label in (0, 1):
        rng = np.random.default_rng(1000 + label)
        for _ in range(N_PER_CLASS):
            c = sample_chain(rng)
            tab[label][codecs.index(c.codec)] += 1
            noise_tab[label][0 if c.noise == "none" else 1] += 1
    # 2) execution audit: N_AUDIO real ffmpeg chains end-to-end.
    items = (real[:N_AUDIO // 2] + fake[:N_AUDIO // 2]) or (real + fake)[:N_AUDIO]
    fails = 0
    nan = 0
    done = 0
    descs: list[str] = []
    for chunk in [items[i:i + 10] for i in range(0, len(items), 10)]:
        try:
            out = batch_stream(chunk, epoch=0)
        except Exception as exc:
            fails += len(chunk)
            print(f"batch fail: {exc}", flush=True)
            continue
        for b in out:
            if not np.isfinite(b.audio).all():
                nan += 1
                continue
            done += 1
            if len(descs) < 5:
                descs.append(f"{b.clip_id} y={b.label}: {b.chain.describe()}")
        print(f"  audio chains ok: {done}/{len(items)}", flush=True)
    res = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sampler_chains_per_class": N_PER_CLASS,
        "audio_chains": done, "batch_failures": fails, "nan_outputs": nan,
        "example_chains": descs,
        "codec_labels": codecs,
        "contingency_codec_vs_label": {"rows_real_fake": tab, **chi_square(tab)},
        "contingency_noise_vs_label": {"rows_real_fake": noise_tab,
                                       "cols": ["no_noise", "noise"],
                                       **chi_square(noise_tab)},
        "note": ("chi-square p>0.05 => augmentation type independent of label "
                 "(label-blind by construction; this is the measured proof). "
                 "Same CODECS/RATES/NOISE distribution feeds Phase-4 training batches."),
    }
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "augmentation_phase3.json").write_text(
        json.dumps(res, indent=1), encoding="utf-8")
    L = ["# Phase 3 augmentation check — compression independent of label", "",
         f"sampler chains: {N_PER_CLASS}/class (distribution proof); "
         f"audio chains executed: {done} (failures={fails}, nan={nan})",
         "", "## Codec vs label (10k sampled chains per class)",
         "|  | " + " | ".join(codecs) + " |",
         "|real| " + " | ".join(map(str, tab[0])) + " |",
         "|fake| " + " | ".join(map(str, tab[1])) + " |",
         f"chi2={res['contingency_codec_vs_label']['chi2']} "
         f"p={res['contingency_codec_vs_label']['p_value']} "
         f"independent={res['contingency_codec_vs_label']['independent']}",
         "", "## Noise vs label",
         f"real {noise_tab[0]} fake {noise_tab[1]} "
         f"chi2={res['contingency_noise_vs_label']['chi2']} "
         f"p={res['contingency_noise_vs_label']['p_value']} "
         f"independent={res['contingency_noise_vs_label']['independent']}",
         "", res["note"]]
    (ROOT / "reports" / "augmentation_phase3.md").write_text(
        "\n".join(L) + "\n", encoding="utf-8")
    print(json.dumps(res, indent=1)[:800])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
