# Final report — measured ensemble replaces the weak detector

All five phases committed. Every number below is measured and saved in the
report named next to it; nothing is invented.

## Models used (verified to exist and load)

| role | id / file | license | size | CPU latency |
|---|---|---|---|---|
| primary anti-spoof | mo-thecreator/Deepfake-audio-detection (wav2vec2-base finetune) | apache-2.0 (Hub tag) | 94.6 M params (~350 MB) | ~330 ms / 4 s clip |
| secondary anti-spoof | Bisher/wav2vec2_ASV_deepfake_audio_detection (wav2vec2-base finetune) | apache-2.0 (Hub tag) | 94.6 M params (~350 MB) | ~340 ms / 4 s clip |
| third ensemble input | existing RAFNet branches (experiments/checkpoints/*) | repo-owned | ~5 checkpoints, ~32 KB + WavLM | ~2–12 s / clip |
| stacking | models/stacking.json (logistic, VAL-fit) + temperature 0.5, band [0.05, 0.55] | derived | 1 KB | — |

Dropped at chance (AUC≈0.5): AST-ASVspoof5, WWWxp. Dropped for codec bias
(FAR 1.0 on public real): Hemgg. Unloadable (custom code): SafeEar, joka13,
RawNet2-HF. Not downloaded (bandwidth): Gustking-large, SSL_Anti-spoofing /
AASIST weights, XLS-R-300M / WavLM-large / Whisper weights (existence
verified; GPU notebook path in notebooks/finetune_gpu_phase3.ipynb).
Speechbrain ECAPA verified on Hub (apache-2.0) but not installed — enrollment
still reports "not installed" honestly.

## Data counts

- phase2_manifest: 419 rows — train 240 (120 LibriSpeech real + 120 crude-VC),
  val 80, test 80, my_samples 19. Speaker-disjoint (0 overlap, verified);
  my_samples generators in no train row (verified).
- data/synth_vc: 240 stopgap crude-VC wavs (regenerable, not in git).
- Public sets: none downloaded (Kaggle creds missing; >15 GB needs approval).

## Metrics (frozen VAL-fit; TEST + my_samples never tuned)

- TEST n=80: acc 0.9375, bal 0.9375, AUC 0.9953, EER 0.0375,
  FAR on human 0.10 (4/40; mo-thecreator speaker-dependent FPs, e.g. spk 6829).
- my_samples n=19 (unseen SAPI/ChatGPT/WinTTS): acc 1.0, AUC 1.0, FAR 0.0.
- Reliability: p 0–0.1 → 97.6% correct (n=41); p 0.9–1.0 → 93.1% (n=58);
  middle bins empty (extreme scores — honest, not forced).
- Codec (test-real): Opus/MP3 deltas ≈0.0; AMR/G.711 flip one mo-FP clip back
  to UNCERTAIN (codec sensitivity documented, not hidden).
- Live: sustained ChatGPT fake → CRITICAL at chunk 5–6 (window 5/5 hot +
  agreement); clean real → never CRITICAL; single hot chunk → max HIGH.
- Consistency: same file x10 + post-restart maxdiff 0.0 (RAFNet and ensemble).
- Full pytest: 404 passed, 3 skipped (test_quantum_branch hangs pre-existing,
  untouched by all phases). Frontend: tsc + vite build pass.
- start-local.ps1 verified end-to-end (sanitized venv print, real model,
  backend healthy, frontend serving).

## Before / after (your four History files)

Your originals (WhatsApp .ogg x2, ChatGPT WAV, formant-synth WAV) are not in
this repo, so they cannot be re-run here — re-upload them in Analyze and the
new columns (per-model P(AI), stacked P(AI), agreement, ensemble alert) will
show measured values. Closest measured equivalents (live API, Phase 5 code):

| case | before (your History) | after (measured) |
|---|---|---|
| human WhatsApp voice note | SYNTHETIC 90% / 89% | No WhatsApp clips on disk — UNMEASURED. Proxy: Opus-24k transcoded public-real → p_cal ≈0.000–0.001, UNCERTAIN/REVIEW, never CRITICAL (reports/candidates_public.json + verify logs) |
| ChatGPT-voice WAV | UNCERTAIN 86% | SYNTHETIC, p_ens 0.999, agreement 1.0, alert HIGH single-shot; live sustained → CRITICAL chunk 5+ |
| crude formant-synth WAV | SYNTHETIC 91% | Nearest: crude-VC TEST 39/40 SYNTHETIC (one UNCERTAIN); no threshold inflation — scores are calibrated stack outputs |
| live mic real person | CRITICAL 90.6% | Gated: single hot chunk → max HIGH; CRITICAL needs 5/5 hot window + ≥2-model agreement (unit + live verified) |

## Known weaknesses (plain)

1. FAR on human is 0.10 on TEST (mo-thecreator FPs on unseen speakers 6829/1580-segment) and UNMEASURED on WhatsApp/phone/Hindi/Tamil — the deployment bar is not met by evidence.
2. Train fake is one crude VC family; TEST fake its held-out setting — expect degradation on true unseen generators.
3. No REAL verdicts issued on VAL/TEST (all real land UNCERTAIN at p≈0.056 — RAFNet's constant ~0.8 bias + band edge); coverage 0.44–0.54.
4. Codec flips mo-thecreator both ways on some speakers (±0.93 observed once).
5. test_quantum_branch.py hangs (pre-existing, environmental).

## Commands to retrain and re-evaluate

```
# scores (skips my_samples; resume-safe)
.venv/Scripts/python tools/cache_scores_phase3.py --only mo
.venv/Scripts/python tools/cache_scores_phase3.py --only bisher
.venv/Scripts/python tools/cache_scores_phase3.py --only rafnet --splits val,test
# stacking (VAL only) + honest eval (TEST + my_samples, frozen)
.venv/Scripts/python tools/train_stacking_phase3.py
.venv/Scripts/python tools/evaluate_phase4.py
# live verification (backend running)
.venv/Scripts/python tools/verify_phase5.py http://127.0.0.1:8000
.venv/Scripts/python tools/verify_live_phase5.py http://127.0.0.1:8000
.venv/Scripts/python tools/consistency_check.py <3 files> --repeats 10 --restarts 0 --yes
```
GPU fine-tune: upload data/phase2_manifest.csv + clips to notebooks/finetune_gpu_phase3.ipynb (Colab/Kaggle), download head_finetuned.pt + finetune_report.json.
