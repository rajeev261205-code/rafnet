# ACCURACY_NOTES.md — what the RAFNet model was trained on, what it expects, and where accuracy can break

Measured from code + checkpoint pickles on 2026-10-11. No retraining done, no
weights changed. Anything not found is marked NOT RECORDED, not guessed.

## 1. What was trained, on what data

- Corpus (per `AI_MODEL_SPEC.md`, itself summarizing `experiments/*.json`):
  **synthetic-only, n=5000** (train 3478 / val 753 / test 769), 100 speakers
  (disjoint splits), **21 generators overlapped across splits (leakage)**.
  No real-human speech in training. Train/val/test accuracy beyond the notes
  below was NOT MEASURED.
- Shipped heads (read from the pickles, not docs):
  - `classical_logistic.pkl`: fitted `LogisticRegression` on **n=200
    embeddings (93/107 split), 256 features**; eval history EMPTY
    (val/test NOT_MEASURED). Train acc 1.0 on those 200 (trivially fittable).
  - `quantum_branch.pkl`: PCA fitted on **n=250 → 8 dims**, fallback MLP head
    (10 epochs, final loss 0.469); VQC NOT trained (`using_fallback: True`).
  - `gateway2.pth`: MLP gate (`gate_network` 3→32→32→1), trained 5 epochs.
  - `calibrator.pkl`: temperature calibrator, `fitted=True`, ECE NOT_MEASURED.
  - `decision_gate.pkl`: tau_low=0.3611, tau_high=0.8611.
- Provenance gap: the repo does not record which 200/250 embeddings (which
  corpus windows, which script run) produced the shipped heads. Treat the
  heads as fitted-but-unvalidated.

## 2. Expected input (training contract)

- 16 kHz mono float32 in [-1, 1] (`src/rafnet/data/dataset.py`: sample_rate
  16000, torchaudio sinc resample when needed, no RMS normalization — the
  audio `Normalizer` is fitted but never attached; only network features are
  normalized).
- **1.0 s windows, 0.5 s hop** (`AudioWindowConfig`: window 16000, hop 8000,
  min 8000 zero-padded). Every training embedding is one window.
- Features: frozen WavLM-base-plus (768, mean-pool stride 320) + handcrafted
  (LFCC60/MFCC40/CQT84/spectral/prosodic/voice-quality → 128) → fused 256.

## 3. How scores become REAL / SYNTHETIC / UNCERTAIN

`audio → z_a (WavLM+handcrafted, full clip, mean-pooled) → Gateway1
fixed_audio → z_f → p_c (logistic) + p_q (PCA8→fallback MLP) → Gateway2 MLP
→ p_fused → temperature calibrator → p_cal = P(synthetic) → DecisionGate:`

- `p_cal < 0.3611` → REAL, `p_cal > 0.8611` → SYNTHETIC, else UNCERTAIN.
- `confidence = max(p_cal, 1 - p_cal)`, `prob_real = 1 - p_cal`.
- Threshold origin: grid search on *some* validation set (gate pickle); the
  validation data and the accuracy it achieved are NOT RECORDED in the repo.

## 4. Inference vs training preprocessing audit

| Step | Training | SOC file inference | Match? |
|---|---|---|---|
| Sample rate | 16000 (sinc resample) | 16000 (linear np.interp) | close (impl differs, negligible) |
| Channels | mono | mono | yes |
| Amplitude norm | none | none | yes |
| Clip length | **1.0 s windows** | **full clip (any length)** | **NO — biggest caveat** |
| Padding | zeros to ≥0.5 s | none (any length flows) | minor |
| int16 scale | float in [-1,1] | float32 in [-1,1] | yes |

The length mismatch is the top accuracy risk: WavLM mean-pools over the whole
clip, so a 30 s upload produces an embedding distribution the 1 s-trained
heads never saw. **Live 1 s chunks match the training contract best** and
should be the most trustworthy path. Changing file inference to windowed
aggregation would change every output, so it is NOT done here — it needs
labeled validation first (see §6).

## 5. Current thresholds and calibration status

- Active: tau 0.3611 / 0.8611 (from `decision_gate.pkl`), temperature
  calibrator fitted (ECE unmeasured). The UNCERTAIN band is 0.50 wide and its
  optimality on real data is unvalidated.
- `configs/decision_thresholds.json` (defaults = pickle values) lets
  `tools/evaluate_model.py --calibrate` install measured thresholds without
  touching weights. Nothing calibrated yet — no labeled data in repo.

## 6. What would actually fix accuracy (not done — needs data/approval)

1. Labeled 16 kHz clips: real speech + TTS + voice-conversion + replayed,
   disjoint speakers *and* generators, ≥20 clips per class (`samples/`).
2. Measure with `tools/evaluate_model.py`, then calibrate thresholds and/or
   refit classical + PCA + gateway2 on windowed embeddings (weight change —
   requires explicit approval).
3. Validate the full-clip vs windowed-aggregation choice on that data before
   changing inference.

## 7. 2026-10-12 follow-up (no weights changed, no inference numbers altered)

- `tools/evaluate_model.py` (new): scores `samples/real/` + `samples/fake/`
  (wav/flac/mp3), writes `reports/model_eval.json/.md` with 3-way confusion,
  strict accuracy (UNCERTAIN = wrong, stated tradeoff), coverage/selective
  accuracy, binary@0.5 precision/recall, EER, AUC, per-class score stats +
  ASCII histograms with the UNCERTAIN band marked, a laptop-mic variant
  (room noise + 80 Hz–6.8 kHz brickwall + MP3 round-trip when supported),
  held-out threshold calibration (`--calibrate`), and single-vs-median
  live-aggregation comparison. Current status: UNEVALUATED — `samples/real/`
  and `samples/fake/` are empty and this box has no working torch runtime,
  so the report records that instead of numbers. Mock inference is refused
  as a substitute (mock is not the model).
- `configs/decision_thresholds.json` (new): tau 0.3611/0.8611, calibrated=false.
  `src/rafnet/soc/thresholds.py` applies it to the in-memory DecisionGate at
  startup (taus only — weights untouched); `--calibrate` overwrites the taus
  with measured values + provenance. `GET /api/models` now reports the live
  gate thresholds instead of hardcoded 0.361/0.861.
- Live rolling window: `LiveSession` keeps the last 8 per-chunk P(synthetic);
  `POST /api/live/chunk` returns `aggregated_prediction` /
  `aggregated_p_synthetic` / `aggregated_chunks` / `aggregated_window` alongside
  the unchanged single-chunk prediction — median of the last 5 with majority
  hysteresis (3/5 to switch), same constants the eval script uses. The Live
  page shows "rolling neural median (N/5 chunks)" next to the single chunk.
- Preprocessing: re-audited, no inference code changed. The §4 table still
  holds (full-clip vs 1 s-window caveat is a validation question, not a bug
  fixed blind). Genuine bugs fixed with tests: EER sweep initialization in the
  new script (never converged — caught by `test_eval_metrics_pure_helpers`),
  clobbered `test_threat_triage_scoped_to_owner` (restored as its own test),
  hardcoded model-info thresholds (now read from the live gate).

## 8. 2026-10-12 live-voice false-positive session (weights still untouched)

Symptom: natural speech on `/live` reads as synthetic. Reproduced with the
REAL model (aegis-venv: torch 2.14.1+cpu, sklearn 1.6.1 + shipped shims, all
5 checkpoints restored) on constructed speech-like signals — therefore
measurement, not guesswork. No labeled human/TTS data exists, so nothing
below is an accuracy claim.

Root causes found (each with before → after on the probes):
1. Dev mock mapped loudness → P(synthetic): loud natural speech read
   SYNTHETIC/CRITICAL. The mock is now constant UNCERTAIN @ 0.5
   (`test_mock_neutral_on_loud_input`); threat tests use an explicit stub.
2. L1 flux divided by ~1e-9 on pause boundaries: any speech with pauses
   read as extreme "splicing" (715392 on a phrase with gaps → 0.0005).
   Per-frame denominator floored at 1% of clip mean frame energy.
3. L1 jitter was global std/mean: intonation range misread as jitter
   (0.303 on a two-pitch phrase → 0.009). Now median relative successive
   difference over consecutive voiced frames. Threshold 0.06 unchanged.
4. L1 HNR was whole-clip autocorrelation: vibrato/intonation smeared the
   peak, clean voicing scored ~1 dB. Now median over ~128 ms frames
   (tone 25→17, vowel 1.2→3.6, noise stays −10). Threshold 6 dB unchanged;
   strong vibrato still reads low — documented residual, needs labeled data.
5. L3 called a pure sine tone REPLAY_PRERECORDED (pause + "codec cutoff" +
   "loop" trip-wires all fire on content-determined acoustics). Narrowband-
   stationary content (flatness < 0.02, flux < 0.05, crest > 10) now abstains
   from band-limit/loop evidence; pause/noise/clip/tail unaffected.
   Band-limited BROADBAND signals still flag; non-tonal repeats still loop.
6. L3 noise floor read digital-black gaps as −180 dB background. Now
   estimated over audible (≥ −46 dBFS) blocks only.
7. Gateway2 `_build_info` warned (NaN std) on batch-1 inference: std now
   reported 0.0 for single-sample batches (diagnostic value only).

After: vowel L1 20→17 / L3 65→36, phrase L1 57→19 / L3 82→41, tone L3
65→36, all verdicts REPLAY → UNCERTAIN with neural UNCERTAIN. Thresholds,
ramps, weights, fusion rules: unchanged (no labeled data to validate any
retune). Real-model neural path verdicts UNCERTAIN on every probe — weak
discrimination as previously reported, not fixed here.
Observed (not fixed): with the classical branch unrestored (liblinear DLL
blocked), tone scores 0.49 vs 0.82 and noise 0.76 vs 0.36 — single-branch
fallback shifts scores substantially. Production must run the full
checkpoint set (sklearn==1.9.1, unblocked torch stack).
