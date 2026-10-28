"""Phase 1 step 1: list candidate models from Hugging Face Hub (no downloads).

Queries HfApi.list_models for deepfake-audio / anti-spoofing / ASVspoof /
spoof + audio-classification, plus explicit checks for the backbones and
checkpoints named in the task brief. Writes reports/candidates_search.json
with verified existence, license, size hints and card snippets.

Never invents: every entry records what the Hub actually returned.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

QUERIES = [
    "deepfake audio",
    "anti-spoofing",
    "ASVspoof",
    "spoof",
    "voice spoof detection",
    "synthetic speech detection",
]

EXPLICIT_CHECKS = [
    # Strong frozen backbones named in the brief (verify existence only here)
    "facebook/wav2vec2-xls-r-300m",
    "facebook/wav2vec2-xls-r-1b",
    "microsoft/wavlm-base-plus",
    "microsoft/wavlm-large",
    "openai/whisper-small",
    "openai/whisper-medium",
    # Speaker verification for the enrollment layer
    "speechbrain/spkrec-ecapa-voxceleb",
]


def main() -> int:
    try:
        from huggingface_hub import HfApi
    except Exception as exc:
        print(f"huggingface_hub missing: {exc}")
        return 2
    api = HfApi()
    found: dict[str, dict] = {}

    for q in QUERIES:
        try:
            models = list(api.list_models(search=q, limit=30))
        except Exception as exc:
            print(f"query {q!r} failed: {exc}")
            continue
        print(f"query {q!r}: {len(models)} hits")
        for m in models:
            mid = m.id
            if mid in found:
                continue
            tags = sorted(list(getattr(m, "tags", []) or []))
            likes = getattr(m, "likes", 0)
            # keep only audio-relevant hits
            blob = f"{mid} {' '.join(tags)}".lower()
            if not any(k in blob for k in ("audio", "speech", "voice", "spoof", "deepfake",
                                          "asvspoof", "wav", "antispoof", "anti-spoof",
                                          "synthetic", "vocoder", "tts")):
                continue
            found[mid] = {"id": mid, "likes": likes, "tags": tags,
                          "query": q, "card": None, "license": None, "pipeline": getattr(m, "pipeline_tag", None)}

    # Explicit existence checks (authoritative: model_info succeeds or not)
    explicit: dict[str, dict] = {}
    for mid in EXPLICIT_CHECKS:
        try:
            info = api.model_info(mid)
            card = (getattr(info, "cardData", None) or {})
            if not isinstance(card, dict):
                try:
                    card = dict(card)
                except Exception:
                    card = {"raw": str(card)[:500]}
            siblings = [getattr(s, "rfilename", "?") for s in (getattr(info, "siblings", []) or [])][:40]
            total = None
            try:
                total = sum(int(getattr(s, "size", 0) or 0) for s in (getattr(info, "siblings", []) or []))
            except Exception:
                pass
            explicit[mid] = {"exists": True, "license": card.get("license"),
                             "language": card.get("language"), "tags": sorted(list(getattr(info, "tags", []) or []))[:20],
                             "pipeline": getattr(info, "pipeline_tag", None),
                             "siblings_bytes": total,
                             "siblings_sample": siblings[:15]}
            print(f"EXISTS {mid} license={card.get('license')} bytes={total}")
        except Exception as exc:
            explicit[mid] = {"exists": False, "error": f"{type(exc).__name__}: {exc}"[:300]}
            print(f"MISSING {mid}: {exc}")

    # Fetch card/license for top audio hits (bounded: top 25 by likes)
    ranked = sorted(found.values(), key=lambda d: d.get("likes", 0), reverse=True)[:25]
    for d in ranked:
        try:
            info = api.model_info(d["id"])
            card = (getattr(info, "cardData", None) or {})
            if not isinstance(card, dict):
                try:
                    card = dict(card)
                except Exception:
                    card = {}
            d["license"] = card.get("license")
            d["pipeline"] = getattr(info, "pipeline_tag", None) or d.get("pipeline")
            d["tags"] = sorted(list(getattr(info, "tags", []) or []))[:20]
            sibs = list(getattr(info, "siblings", []) or [])
            tot = 0
            names = []
            for s in sibs:
                try:
                    tot += int(getattr(s, "size", 0) or 0)
                except Exception:
                    pass
                names.append(getattr(s, "rfilename", "?"))
            d["siblings_bytes"] = tot
            d["has_safetensors"] = any(n.endswith(".safetensors") for n in names)
            d["has_pytorch_bin"] = any(n.endswith(".bin") for n in names)
        except Exception as exc:
            d["info_error"] = f"{type(exc).__name__}: {exc}"[:200]

    out = {"generated_at": datetime.now(timezone.utc).isoformat(),
           "queries": QUERIES,
           "audio_hits": ranked,
           "audio_hits_total_kept": len(ranked),
           "explicit_checks": explicit}
    (REPORTS / "candidates_search.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"kept {len(ranked)} audio-relevant hits; wrote reports/candidates_search.json")
    for d in ranked:
        print(f"  {d['likes']:5d} {d['id']} license={d.get('license')} pipe={d.get('pipeline')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
