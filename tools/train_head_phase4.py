"""Phase 4.2b: train a lightweight head on cached WavLM embeddings.

Inputs (all LOCAL, git-ignored):
  data/splits_phase4.json                 train/val/test clip_id -> y mapping
  data/embed_chunks/chunk_*_<tag>_*.npz   mean[L,H] + std[L,H] fp16 per clip

Model: per-layer attention pooling over (layer-mean || layer-std) -> small MLP
(class-weighted BCE). Trained on TRAIN clips only; VAL is used for early
stopping / threshold band and is NEVER used to touch the TEST split. TEST is
not read in this script at all.

Augmented copies (--train-tags clean,aug0) expand the TRAIN set only; VAL is
scored with CLEAN embeddings for honest selection (augmented val is reported
for robustness, never used for selection/thresholds).

Outputs:
  models/head_phase4.pt           (local, git-ignored — weights + config)
  reports/head_phase4.json/.md    (committed; aggregates only, no clip ids)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHUNK_CACHE = ROOT / "data" / "embed_chunks"
SPLITS = ROOT / "data" / "splits_phase4.json"
N_LAYERS, HID = 13, 768


def load_tag(tag: str):
    """Union every part for one tag -> (ids[list[str]], y[np.int64], mean, std).

    mean/std are float16 arrays shaped [N, L, H]. Rows are de-duplicated by id
    (first occurrence wins) so a re-run that left a stale part cannot double-count.
    """
    import numpy as np

    ids: list[str] = []
    seen: set[str] = set()
    ys: list[int] = []
    means: list = []
    stds: list = []
    for p in sorted(CHUNK_CACHE.glob(f"chunk_*_{tag}_part*.npz")):
        z = np.load(str(p), allow_pickle=True)
        pids = z["ids"].tolist()
        py = np.asarray(z["y"])
        pm = np.asarray(z["mean"])
        ps = np.asarray(z["std"])
        for i, cid in enumerate(pids):
            if cid in seen:
                continue
            seen.add(cid)
            ids.append(cid)
            ys.append(int(py[i]))
            means.append(pm[i])
            stds.append(ps[i])
    if not ids:
        raise SystemExit(f"no embedding parts for tag={tag} under {CHUNK_CACHE}")
    return ids, np.asarray(ys, dtype=np.int64), np.stack(means), np.stack(stds)


def clip_of(embed_id: str) -> str:
    return embed_id.split(":", 1)[1] if ":" in embed_id else embed_id


def index_by_clip(ids: list[str]):
    return {clip_of(v): i for i, v in enumerate(ids)}


def build_matrix(ids, ys, mean, std, split_rows, want_y: int | None = None):
    """Select rows whose clip is in split_rows, optionally filtered by y."""
    import numpy as np

    idx = index_by_clip(ids)
    Xm, Xs, Y, C = [], [], [], []
    for r in split_rows:
        if want_y is not None and int(r["y"]) != want_y:
            continue
        j = idx.get(r["clip_id"])
        if j is None:
            continue
        Xm.append(mean[j])
        Xs.append(std[j])
        Y.append(int(r["y"]))
        C.append(r["clip_id"])
    if not Xm:
        return (np.zeros((0, N_LAYERS, HID), np.float16),
                np.zeros((0, N_LAYERS, HID), np.float16),
                np.zeros((0,), np.int64), [])
    return (np.stack(Xm), np.stack(Xs), np.asarray(Y, np.int64), C)


def build_model(seed: int, mid: int, dropout: float):
    import torch
    import torch.nn as nn

    class LayerAttnHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_score = nn.Sequential(
                nn.Linear(2 * HID, mid), nn.GELU(), nn.Linear(mid, 1))
            self.mlp = nn.Sequential(
                nn.Linear(2 * HID, mid), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(mid, 1))

        def forward(self, mean, std):
            x = torch.cat([mean, std], dim=-1)          # [B, L, 2H]
            a = self.layer_score(x).squeeze(-1)          # [B, L]
            w = torch.softmax(a, dim=1)                  # [B, L]
            agg = (w.unsqueeze(-1) * x).sum(dim=1)       # [B, 2H]
            return self.mlp(agg).squeeze(-1), w

    torch.manual_seed(seed)
    return LayerAttnHead()


def predict(model, mean, std, batch=512):
    import numpy as np
    import torch

    model.eval()
    out = np.zeros(len(mean), dtype=np.float64)
    with torch.inference_mode():
        for s in range(0, len(mean), batch):
            m = torch.from_numpy(mean[s:s + batch].astype(np.float32))
            sd = torch.from_numpy(std[s:s + batch].astype(np.float32))
            logit, _ = model(m, sd)
            out[s:s + len(m)] = logit.double().numpy()
    return out


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-max(-500.0, min(500.0, z))))


def metrics_from_logits(y, logits, tau_low=None, tau_high=None):
    sys.path.insert(0, str(ROOT))
    from tools.evaluate_model import (binary_metrics_at_05, compute_auc,
                                      compute_eer, coverage_and_selective,
                                      strict_accuracy, confusion_3way)

    p = [sigmoid(float(z)) for z in logits]
    yl = [int(v) for v in y]
    m = {"n": len(yl), "auc": compute_auc(yl, p), "eer": compute_eer(yl, p),
         "binary_at_0.5": binary_metrics_at_05(yl, p)}
    far = sum(1 for yy, pp in zip(yl, p) if yy == 0 and pp >= 0.5) / max(1, sum(1 for yy in yl if yy == 0))
    m["far_human_at_0.5"] = round(far, 4)
    if tau_low is not None and tau_high is not None:
        m["band"] = [round(tau_low, 4), round(tau_high, 4)]
        m["strict_accuracy"] = strict_accuracy(yl, p, tau_low, tau_high)
        m["confusion"] = confusion_3way(yl, p, tau_low, tau_high)
        m.update({k: v for k, v in coverage_and_selective(yl, p, tau_low, tau_high).items()})
    return m, p


def pick_band(y, p):
    """VAL-only band: selective accuracy >= 0.95, then max coverage, then narrowest."""
    best = None
    for lo in [round(x, 4) for x in _lin(0.01, 0.45, 23)]:
        for hi in [round(x, 4) for x in _lin(0.55, 0.95, 17)]:
            if not (0.1 <= hi - lo <= 0.5):
                continue
            certain = [(pp, yy) for pp, yy in zip(p, y) if pp < lo or pp > hi]
            if not certain:
                continue
            cov = len(certain) / len(y)
            sel = sum(1 for pp, yy in certain if (pp > hi) == (yy == 1)) / len(certain)
            key = (sel >= 0.95, round(sel, 4), round(cov, 4), -round(hi - lo, 4))
            if best is None or key > best[0]:
                best = (key, float(lo), float(hi), round(sel, 4), round(cov, 4))
    if best is None:
        return 0.3611, 0.8611, None, None
    _, lo, hi, sel, cov = best
    return lo, hi, sel, cov


def _lin(a, b, n):
    return [a + (b - a) * i / (n - 1) for i in range(n)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-tags", default="clean,aug0")
    ap.add_argument("--val-tag", default="clean")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--mid", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: cap rows")
    args = ap.parse_args()

    import numpy as np
    import torch
    import torch.nn as nn

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    splits = json.loads(SPLITS.read_text(encoding="utf-8"))
    train_rows, val_rows = splits["train"], splits["val"]

    # ---- load embeddings ----
    cache = {}
    for tag in set(args.train_tags.split(",")) | {args.val_tag}:
        tag = tag.strip()
        cache[tag] = load_tag(tag)
        print(f"tag {tag}: {len(cache[tag][0])} rows", flush=True)

    # ---- assemble train (all train tags) ----
    tr_m, tr_s, tr_y = [], [], []
    for tag in args.train_tags.split(","):
        tag = tag.strip()
        ids, ys, mn, sd = cache[tag]
        m, s, y, _ = build_matrix(ids, ys, mn, sd, train_rows)
        tr_m.append(m); tr_s.append(s); tr_y.append(y)
        print(f"train tag {tag}: {len(y)} rows (real={int((y==0).sum())}, fake={int((y==1).sum())})", flush=True)
    tr_m = np.concatenate(tr_m); tr_s = np.concatenate(tr_s); tr_y = np.concatenate(tr_y)

    # ---- val (clean) ----
    vids, vys, vmn, vsd = cache[args.val_tag]
    va_m, va_s, va_y, _ = build_matrix(vids, vys, vmn, vsd, val_rows)
    print(f"val tag {args.val_tag}: {len(va_y)} rows (real={int((va_y==0).sum())}, fake={int((va_y==1).sum())})", flush=True)

    if args.limit:
        tr_m, tr_s, tr_y = tr_m[:args.limit], tr_s[:args.limit], tr_y[:args.limit]
        va_m, va_s, va_y = va_m[:args.limit], va_s[:args.limit], va_y[:args.limit]

    # ---- class weights ----
    n_real = int((tr_y == 0).sum())
    n_fake = int((tr_y == 1).sum())
    w_real = len(tr_y) / (2.0 * max(1, n_real))
    w_fake = len(tr_y) / (2.0 * max(1, n_fake))
    print(f"class weights: real={w_real:.3f} fake={w_fake:.3f}", flush=True)

    model = build_model(args.seed, args.mid, args.dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lossfn = nn.BCEWithLogitsLoss(reduction="none")
    cw = torch.tensor([w_real, w_fake], dtype=torch.float32)

    rng = np.random.default_rng(args.seed)
    n = len(tr_y)
    best = {"auc": -1.0, "epoch": -1, "state": None}
    hist = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = rng.permutation(n)
        tot = 0.0
        for s in range(0, n, args.batch):
            idx = perm[s:s + args.batch]
            m = torch.from_numpy(tr_m[idx].astype(np.float32))
            sd = torch.from_numpy(tr_s[idx].astype(np.float32))
            yb = torch.from_numpy(tr_y[idx].astype(np.float32))
            opt.zero_grad()
            logit, _ = model(m, sd)
            w = cw[tr_y[idx].astype(np.int64)]
            loss = (lossfn(logit, yb) * w).mean()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(idx)
        vlogit = predict(model, va_m, va_s)
        vm, _ = metrics_from_logits(va_y, vlogit)
        auc = vm["auc"] or 0.0
        hist.append({"epoch": epoch, "train_loss": round(tot / max(1, n), 4),
                     "val_auc": auc, "val_far": vm["far_human_at_0.5"]})
        if epoch % 5 == 0 or epoch == 1:
            print(f"epoch {epoch:3d} loss={tot/max(1,n):.4f} val_auc={auc} far={vm['far_human_at_0.5']}", flush=True)
        if auc > best["auc"] + 1e-4:
            best = {"auc": auc, "epoch": epoch,
                    "state": {k: v.clone() for k, v in model.state_dict().items()}}
        elif epoch - best["epoch"] >= args.patience:
            print(f"early stop at epoch {epoch} (best {best['epoch']} auc={best['auc']})", flush=True)
            break

    model.load_state_dict(best["state"])
    vlogit = predict(model, va_m, va_s)
    _, vp = metrics_from_logits(va_y, vlogit)
    lo, hi, sel, cov = pick_band([int(v) for v in va_y], vp)
    vm, vp = metrics_from_logits(va_y, vlogit, lo, hi)

    # robustness: augmented val (NOT used for selection)
    aug_val = None
    val_tag_aug = "aug0"
    if val_tag_aug in cache:
        a_ids, a_ys, a_mn, a_sd = cache[val_tag_aug]
        am, as_, ay, _ = build_matrix(a_ids, a_ys, a_mn, a_sd, val_rows)
        if len(ay):
            alogit = predict(model, am, as_)
            aug_val, _ = metrics_from_logits(ay, alogit, lo, hi)

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {k: v for k, v in vars(args).items()},
        "n_train_rows": int(len(tr_y)),
        "n_train_real": n_real, "n_train_fake": n_fake,
        "n_val_rows": int(len(va_y)),
        "class_weights": {"real": round(w_real, 4), "fake": round(w_fake, 4)},
        "best_epoch": best["epoch"], "best_val_auc": round(best["auc"], 4),
        "val_clean": vm,
        "val_aug0_untuned": aug_val,
        "history": hist,
        "note": ("VAL is clean-only and never TEST. aug0 val shown for robustness "
                 "only. Class weights from train. No test rows read by this script."),
    }
    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "config": {"n_layers": N_LAYERS, "hid": HID, "mid": args.mid,
                           "dropout": args.dropout, "seed": args.seed},
                "band": {"tau_low": lo, "tau_high": hi},
                "encoder": "microsoft/wavlm-base-plus"},
               str(models_dir / "head_phase4.pt"))
    (ROOT / "reports" / "head_phase4.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    md = ["# Phase 4.2b head (WavLM embeddings, trained on TRAIN only)", "",
          f"train rows={len(tr_y)} (real={n_real}, fake={n_fake}); "
          f"val rows={len(va_y)} (clean)",
          f"best epoch={best['epoch']} val AUC={round(best['auc'],4)}",
          f"band=[{lo}, {hi}] (val-only selective>=0.95, max coverage, narrowest)",
          f"VAL clean: AUC={vm['auc']} EER={json.dumps(vm['eer'])} "
          f"FAR@0.5={vm['far_human_at_0.5']} binary@0.5={json.dumps(vm['binary_at_0.5'])}",
          f"VAL clean confusion(band): {json.dumps(vm.get('confusion'))}",
          f"VAL aug0 (robustness, NOT tuned): "
          f"{json.dumps(aug_val)}" if aug_val else "VAL aug0: n/a", "",
          "Honest note: val fake shares train generators (in-distribution); the "
          "unseen generators live in TEST and are measured only in Phase 5.",
          "VAL is speaker-disjoint from train; TEST (incl. private groups + unseen "
          "generators) is never read here."]
    (ROOT / "reports" / "head_phase4.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"BEST epoch={best['epoch']} val_auc={best['auc']} band=[{lo},{hi}]")
    print(f"val clean: AUC={vm['auc']} FAR={vm['far_human_at_0.5']} EER={json.dumps(vm['eer'])}")
    print("wrote models/head_phase4.pt + reports/head_phase4.json/.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
