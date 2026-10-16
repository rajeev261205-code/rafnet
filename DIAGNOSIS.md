# DIAGNOSIS.md — why live results hover at ~50–80% UNCERTAIN (measured, 2026-10-12)

No behaviour was changed to produce this file. Every number below was measured
with the shipped checkpoints (`experiments/checkpoints/*`, weights untouched).

## 1. Score histogram: real humans vs real TTS on the CURRENT model

12 LibriSpeech test-clean clips (CC-BY 4.0, one per speaker dir → 12 disjoint
speakers, 2.4–12 s, FLAC 16 kHz) and 9 Windows-SAPI TTS clips (2 voices,
generated locally with `tools/make_win_tts.py`), scored full-clip with the
REAL pipeline (aegis-venv, all 5 checkpoints restored). `p` = P(synthetic),
decision band = [0.3611, 0.8611] (`|`: band edges, `!`: UNCERTAIN mass).

```
p:     0.0  0.1  0.2  0.3  0.4  0.5  0.6  0.7  0.8  0.9  1.0
band:                      |tau_lo                tau_hi|
HUMAN (n=12, LibriSpeech, disjoint speakers):
  counts per 0.1 bin:      0    0    0    0    0    0    0    0    9    3    0
  range 0.7917–0.8870 · 9 UNCERTAIN · !!! 3 SYNTHETIC false positives !!!
TTS (n=9, MS SAPI David/Zira):
  counts per 0.1 bin:      0    0    0    0    0    0    0    0    9    0    0
  range 0.8077–0.8495 · 9 UNCERTAIN · 0 SYNTHETIC
```

The two distributions OVERLAP COMPLETELY. The detector does not separate
human from TTS speech on this probe: humans score 0.79–0.89 (3/12 cross into
SYNTHETIC), TTS scores 0.81–0.85 (all UNCERTAIN). "Around 50–80% UNCERTAIN"
is therefore not a threshold artifact — it is the model seeing all speech as
mildly-to-strongly synthetic-like, with the 0.50-wide UNCERTAIN band catching
nearly everything and its upper edge leaking human FPs.

Raw scores: `C:\Users\rajee\AppData\Local\Temp\opencode\libri_scores.json`
(humans), `.../win_scores.json` (TTS) — local run logs, not committed.

## 2. Which branch drives it (same runs)

- Classical (logistic): **0.94–0.99 on REAL humans**, 0.98–0.99 on TTS.
  It saturates pro-synthetic on out-of-distribution human speech. Trained on
  n=200 embeddings from a synthetic-only corpus (per AI_MODEL_SPEC.md); which
  200 windows is NOT RECORDED — likely never saw bonafide speech, or saw it
  under a shifted window distribution.
- Quantum (PCA8→fallback MLP): 0.13–0.75 on humans, 0.09–0.33 on TTS.
  Varies, weakly separates, but Gateway2 trusts classical (β≈0.9, measured).
- Gateway2 MLP gate → p_fused ≈ p_classical; temperature calibrator then maps
  ~0.91 → ~0.82 (fitted=True, but on WHAT data is NOT RECORDED; ECE NOT_MEASURED).

Conclusion: the classical head is the false-positive engine. It must be
re-trained (or gated by OOD detection) on data containing real human speech —
a weight change requiring explicit approval. No threshold move fixes a head
that scores bonafide speech 0.97.

## 3. Train vs inference preprocessing (code audit)

| Step | Training | File inference (`/api/analyze`) | Live (`/api/live/chunk`) |
|---|---|---|---|
| Sample rate | 16000, torchaudio sinc resample | 16000, linear np.interp resample | 16000, linear np.interp resample |
| Channels | mono | mono (mean-mix) | mono (browser sends mono) |
| Amplitude norm | none | none | none |
| Clip length | **1.0 s windows, 0.5 s hop** | **full clip (any length)** | 1.0 s windows ✓ |
| Padding | zeros to ≥0.5 s | none | n/a (waits for full window) |
| VAD/trimming | none | none | silence analyzed, not skipped |
| Features | WavLM-base-plus frozen (768, mean-pool /320) + handcrafted (LFCC60/MFCC40/CQT84/…) → 256 | same encoder | same encoder |
| Decision | gate τ 0.3611/0.8611 + temperature | same | same + rolling median-5 display |

Live 1 s chunks match the training window contract; full-clip uploads do not
(WavLM mean-pools the whole clip → embedding distribution the heads never
saw). Live is the more trustworthy path, but §1 shows even windowed inputs
land UNCERTAIN/FP — the head, not the length, is the binding constraint.

## 4. What "50" means in each mode

- Mock (`MOCK_INFERENCE=true`): exactly 0.50, constant, by design
  (non-diagnostic stub). Displayed ×100 as "50.0%" with a MOCK badge.
- Real model: 0.50 is NOT special — scores are calibrated-probability outputs
  in [0,1]; the pile-up at 0.6–0.9 is the classical head saturating, and
  0.36–0.86 renders as UNCERTAIN. No logit/probability bug, no reversed
  labels, no stale state found in the trace.

## 5. Hardware (measured 2026-10-12)

- CPU: AMD Ryzen 7 7445HS, 6 cores. RAM: 15.2 GB. Disk C: 50.1 GB free.
- GPU: NONE (`torch.cuda.is_available()` → False on every runtime here).
- Verdict: CPU-friendly plan (frozen SSL encoder + cached embeddings + small
  heads, §Phase 3) plus `notebooks/` for any GPU step. 15 GB RAM fits
  WavLM-base inference + head training on cached embeddings; wav2vec2-xls-r-300m
  inference on CPU is slow per clip but feasible for embedding caching in
  batches; fine-tuning the encoder is NOT feasible here.

## 6. What must happen (no thresholds were touched)

1. Labeled data with REAL human speech (Phase 1: LibriSpeech test-clean
   downloaded, CC-BY 4.0, `data/raw/` — git-ignored; ASVspoof/WaveFake/MLAAD
   need license checks + owner approval before download).
2. Re-train/re-fit the classical head (+ PCA + gateway2) WITH bonafide speech
   in the mix, speaker- AND generator-disjoint splits (Phase 3, CPU plan +
   Colab/Kaggle notebooks for GPU steps).
3. Validate on held-out + unseen-generator + telephony-degraded sets, calibrate
   on validation only, and only then set the UNCERTAIN band (Phase 4).
4. Until then: every UI number comes from saved reports; unevaluated states
   stay unevaluated (Phases 5–7).
