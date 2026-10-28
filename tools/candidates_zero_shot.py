"""Phase 1 step 2: zero-shot test of HF audio-classification candidates.

For each candidate in reports/candidates_search.json (audio-classification
pipeline only, top by likes + the RawNet2/AST/SafeEar entries), this script:
- reads the model card (license, training data snippet),
- tries to load AutoFeatureExtractor + AutoModelForAudioClassification,
- scores a representative subset of samples/labels.csv (4 real + all 6
  top-level fake + chatgpt + tts_test = 12 files, truncated to 8 s),
- records per-file P(AI), CPU latency, AUC/EER on the subset and notes
  the id2label -> P(AI) mapping assumption explicitly.

Writes reports/candidates_zeroshot.json. Never invents: load failures and
mapping ambiguities are recorded, not guessed over.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import torch

SUBSET = [
    "samples/real/260-123286-0012.flac",
    "samples/real/4446-2273-0021.flac",
    "samples/real/5142-33396-0059.flac",
    "samples/real/6930-81414-0025.flac",
    "samples/fake/sapi_david_2.wav",
    "samples/fake/sapi_david_4.wav",
    "samples/fake/sapi_zira_2.wav",
    "samples/fake/sapi_zira_4.wav",
    "samples/fake/chatgpt_voice/rafnet_ai_voice_test.wav",
    "samples/fake/windows_tts/tts_test.wav",
    "samples/win_tts/sapi_zira_3.wav",
    "samples/win_tts/sapi_david_0.wav",
]
MAX_SECONDS = 8
MAX_MODELS = 14


def map_logits_to_pai(logits, id2label: dict) -> tuple[float, str]:
    """Map logits vector (1-D, num_labels) to P(AI). Returns (p, mapping_note)."""
    import torch.nn.functional as F

    vec = logits[0] if logits.ndim == 2 else logits.reshape(-1)
    probs = F.softmax(vec, dim=-1).tolist()
    if isinstance(probs, float):
        probs = [probs]
    labels = [str(id2label.get(i, id2label.get(str(i), f"LABEL_{i}"))) for i in range(len(probs))]
    low = " ".join(labels).lower()
    # Find the 'fake-like' index
    fake_keys = ("fake", "spoof", "synthetic", "artificial", "ai", "deepfake", "tts", "vc", "generated")
    real_keys = ("bona", "real", "human", "genuine", "authentic", "natural", "live")
    fake_idx = next((i for i, l in enumerate(labels) if any(k in l.lower() for k in fake_keys)), None)
    real_idx = next((i for i, l in enumerate(labels) if any(k in l.lower() for k in real_keys)), None)
    note = f"id2label={labels}"
    if fake_idx is not None:
        return float(probs[fake_idx]), note + f" -> P(AI)=class {fake_idx}"
    if real_idx is not None and len(probs) == 2:
        other = 1 - real_idx
        return float(1.0 - probs[real_idx]) if len(probs) == 2 else float(probs[other]), note + f" -> P(AI)=1-P(class {real_idx})"
    # Ambiguous LABEL_0/LABEL_1: assume index 1 = fake, but FLAG it
    if len(probs) == 2:
        return float(probs[1]), note + " -> ASSUMED P(AI)=class 1 (labels ambiguous, verify card)"
    return float(max(probs)), note + " -> FALLBACK max prob (non-binary head, not a spoof detector?)"


def load_audio_16k(path: Path, max_s: int = MAX_SECONDS):
    import numpy as np
    import soundfile as sf

    arr, sr = sf.read(str(path), dtype="float32", always_2d=False)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr.mean(axis=1)
    if sr != 16000 and len(arr):
        dur = len(arr) / float(sr)
        tgt = int(min(dur, max_s) * 16000)
        old = np.linspace(0.0, 1.0, num=len(arr))
        new = np.linspace(0.0, 1.0, num=max(1, tgt))
        arr = np.interp(new, old, arr).astype(np.float32)
        sr = 16000
    else:
        arr = arr[: sr * max_s]
    return arr, 16000


def main() -> int:
    search = json.loads((ROOT / "reports" / "candidates_search.json").read_text(encoding="utf-8"))
    cands = [d for d in search["audio_hits"] if (d.get("pipeline") or "") == "audio-classification"]
    # ensure the interesting low-like but architecturally relevant ones are kept
    keep_ids = {"caa-speech-detection-asvspoof2019/rawnet2",
                "MattyB95/AST-ASVspoof5-Synthetic-Voice-Detection",
                "MattyB95/AST-ASVspoof2019-Synthetic-Voice-Detection",
                "TEC2004/SafeEar-ASV19-spoof-detection",
                "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification",
                "Bisher/wav2vec2_ASV_deepfake_audio_detection",
                "WWWxp/wav2vec2_spoof_dection1"}
    for d in search["audio_hits"]:
        if d["id"] in keep_ids and d not in cands:
            cands.append(d)
    cands = sorted(cands, key=lambda d: d.get("likes", 0), reverse=True)[:MAX_MODELS]
    print(f"testing {len(cands)} audio-classification candidates on {len(SUBSET)} files")

    from huggingface_hub import HfApi
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

    api = HfApi()
    sys.path.insert(0, str(ROOT))
    from tools.evaluate_model import compute_auc, compute_eer, binary_metrics_at_05

    results = []
    for d in cands:
        mid = d["id"]
        print(f"\n=== {mid} ===")
        entry: dict = {"id": mid, "likes": d.get("likes"), "license": d.get("license"),
                       "siblings_bytes": d.get("siblings_bytes"),
                       "has_safetensors": d.get("has_safetensors")}
        # card snippet (training data / dataset mentions)
        try:
            info = api.model_info(mid)
            card = getattr(info, "cardData", None) or {}
            try:
                card = dict(card)
            except Exception:
                card = {}
            entry["license"] = card.get("license", entry["license"])
            entry["card_datasets"] = card.get("datasets")
            entry["card_language"] = card.get("language")
            # README snippet
            try:
                from huggingface_hub import hf_hub_download

                rp = hf_hub_download(mid, "README.md")
                txt = Path(rp).read_text(encoding="utf-8", errors="ignore")[:1500]
                entry["readme_snippet"] = txt
            except Exception as exc:
                entry["readme_snippet"] = f"unavailable: {type(exc).__name__}"
        except Exception as exc:
            entry["card_error"] = f"{type(exc).__name__}: {exc}"[:200]

        t_load0 = time.perf_counter()
        try:
            feat = AutoFeatureExtractor.from_pretrained(mid, trust_remote_code=False)
            mdl = AutoModelForAudioClassification.from_pretrained(mid, trust_remote_code=False)
            mdl.eval()
            entry["load_ok"] = True
            entry["load_seconds"] = round(time.perf_counter() - t_load0, 1)
            entry["num_labels"] = int(getattr(mdl.config, "num_labels", -1))
            try:
                entry["id2label"] = {str(k): str(v) for k, v in dict(getattr(mdl.config, "id2label", {})).items()}
            except Exception:
                entry["id2label"] = {}
            n_params = sum(p.numel() for p in mdl.parameters())
            entry["params"] = int(n_params)
            entry["params_M"] = round(n_params / 1e6, 1)
        except Exception as exc:
            entry["load_ok"] = False
            entry["load_error"] = f"{type(exc).__name__}: {str(exc)[:400]}"
            entry["load_seconds"] = round(time.perf_counter() - t_load0, 1)
            print(f"  LOAD FAILED: {entry['load_error'][:200]}")
            results.append(entry)
            continue
        print(f"  loaded: labels={entry['id2label']} params={entry['params_M']}M in {entry['load_seconds']}s")

        # score subset
        per_file = []
        latencies = []
        with torch.inference_mode():
            for rel in SUBSET:
                p = ROOT / rel
                y = 0 if "/real/" in rel else 1
                try:
                    arr, sr = load_audio_16k(p)
                    inputs = feat(arr, sampling_rate=sr, return_tensors="pt", padding=True)
                    t0 = time.perf_counter()
                    logits = mdl(**{k: v for k, v in inputs.items() if k in ("input_values", "attention_mask")}).logits[0]
                    dt = (time.perf_counter() - t0) * 1000
                    pai, note = map_logits_to_pai(logits, mdl.config.id2label)
                    per_file.append({"file": rel, "y": y, "p_ai": round(float(pai), 4),
                                     "latency_ms": round(dt, 1), "map": note})
                    latencies.append(dt)
                    print(f"  {rel.split('/')[-1]:35s} y={y} p={float(pai):.4f} {dt:.0f}ms")
                except Exception as exc:
                    per_file.append({"file": rel, "y": y, "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
                    print(f"  {rel} ERROR {exc}")
        scored = [f for f in per_file if "p_ai" in f]
        entry["files"] = per_file
        if scored:
            ys = [f["y"] for f in scored]
            ss = [f["p_ai"] for f in scored]
            entry["n_scored"] = len(scored)
            entry["auc"] = compute_auc(ys, ss)
            entry["eer"] = compute_eer(ys, ss)
            entry["binary_at_0.5"] = binary_metrics_at_05(ys, ss)
            entry["mean_latency_ms"] = round(sum(latencies) / len(latencies), 1) if latencies else None
            entry["verdict"] = ("AT CHANCE — drop" if (entry["auc"] is not None and entry["auc"] < 0.6)
                                else "keep for Phase 3 shortlist" if (entry["auc"] is not None and entry["auc"] >= 0.7)
                                else "weak — likely drop")
        else:
            entry["verdict"] = "no scores — drop"
        # free memory before next large model
        try:
            del mdl, feat
            import gc
            gc.collect()
        except Exception:
            pass
        results.append(entry)

    out = {"generated_at": datetime.now(timezone.utc).isoformat(),
           "subset": SUBSET, "max_seconds": MAX_SECONDS, "device": "cpu",
           "note": "Zero-shot P(AI) mapping assumptions are recorded per file (see map). AUC<0.6 = at chance.",
           "results": results}
    (ROOT / "reports" / "candidates_zeroshot.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\nwrote reports/candidates_zeroshot.json")
    for r in results:
        print(f"- {r['id']}: load={r.get('load_ok')} auc={r.get('auc')} eer={r.get('eer')} "
              f"lat={r.get('mean_latency_ms')}ms params={r.get('params_M')}M verdict={r.get('verdict')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
