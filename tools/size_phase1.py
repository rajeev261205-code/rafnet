"""Phase 1 sizing: metadata-only availability check for public human-speech
sources (no audio downloaded). Prints: exists, gated?, license tags, and
per-config parquet byte sizes so the 15 GB budget can be planned.

Skips anything needing HF_TOKEN / Kaggle / manual agreements (listed).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CANDIDATES = [
    # (hub_id, configs of interest)
    ("google/fleurs", ["hi_in", "ta_in", "en_us"]),
    ("mozilla-foundation/common_voice_17_0", ["en", "hi", "ta"]),
    ("mozilla-foundation/common_voice_16_1", ["en", "hi", "ta"]),
    ("facebook/voxpopuli", ["en", "hi", "ta"]),
    ("LIUM/tedlium", ["release3"]),
    ("edinburghcstr/ami", ["ihm"]),
    ("vctk", []),
    ("cstr-vctk-corpus", []),
    ("librispeech_asr", ["clean"]),
    ("ai4bharat/indictts", []),
    ("openslr", []),
    ("ProgramChair/voxceleb", []),
    ("DTU54/vctk", []),
]


def main() -> int:
    from huggingface_hub import HfApi
    from huggingface_hub.utils import HfHubHTTPError

    api = HfApi()
    for hub_id, configs in CANDIDATES:
        try:
            info = api.dataset_info(hub_id)
        except HfHubHTTPError as exc:
            print(f"{hub_id}: MISSING/INACCESSIBLE ({str(exc)[:120]})")
            continue
        except Exception as exc:
            print(f"{hub_id}: ERROR {type(exc).__name__}: {str(exc)[:120]}")
            continue
        gated = bool(getattr(info, "gated", False))
        tags = getattr(info, "tags", []) or []
        lic = [t for t in tags if t.startswith("license:")]
        print(f"{hub_id}: EXISTS gated={gated} {lic} sha={str(info.sha)[:8]}")
        # list data files (parquet) with sizes
        try:
            files = api.list_repo_tree(hub_id, repo_type="dataset", recursive=True)
            pars = [(f.path, getattr(f, "size", None)) for f in files
                    if f.path.endswith(".parquet")]
        except Exception as exc:
            print(f"  tree list failed: {str(exc)[:100]}")
            continue
        if not pars:
            print("  (no .parquet files at root listing)")
            continue
        SEL = 6
        tot = 0
        per_cfg: dict[str, list] = {}
        for path, size in pars:
            cfg = path.split("/")[0] if "/" in path else "(root)"
            per_cfg.setdefault(cfg, []).append(size or 0)
        want = configs if configs else sorted(per_cfg)[:SEL]
        for cfg in want:
            sizes = per_cfg.get(cfg, [])
            if isinstance(sizes, list):
                s = sum(sizes)
                print(f"  cfg {cfg}: {len(sizes)} parquet files, {s / 1e9:.2f} GB")
                tot += s
        others = [c for c in per_cfg if c not in want]
        if others:
            print(f"  other cfgs: {sorted(others)[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
