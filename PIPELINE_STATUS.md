# PIPELINE STATUS — real/AI voice detector for compressed phone audio

Purpose: running log so work resumes from the next unfinished phase after a
long context or session restart. Updated after every phase. No private
filenames, no credential values, no audio data in this file (all committed).

## How to resume
1. Read this file top-to-bottom (current phase = first unchecked box below).
2. Reuse existing work: `data/`, `reports/`, `tools/`, `samples/.phase0_local/`
   (local-only idmap + resume caches; NEVER commit `samples/`).
3. Privacy rules (hard): private clips stay local (never upload to Colab /
   Kaggle / HF / any API, never commit, never send to third parties). Committed
   reports use anonymous ids only (`wa_*`, `pc_*`); real-name map lives at
   `samples/.phase0_local/idmap.json` (git-ignored).
4. Commit after each phase. Frontend (`frontend/`) is finished — do not touch.

## Environment (verified Phase 0, backend venv `.venv/Scripts/python.exe`)
- python 3.12.0, torch 2.14.1+CPU, **CUDA unavailable (CPU only)**,
  RAM 15.6 GB total / ~3.9 GB free, disk C: 41.6 GB free (41.3 at report time).
- ffmpeg: NO system ffmpeg on PATH. `winget install --id Gyan.FFmpeg -e`
  attempted 3x — stalls at "Downloading ..." every time (not installed).
  Using bundled imageio-ffmpeg 7.1 binary (decodes Opus + AMR-NB verified;
  all Phase 0 transcodes real, `simulated=false`). **Still needed: working
  system ffmpeg** (retry winget on a better network or install Gyan build
  manually and add to PATH).
- Credentials (present/missing only, values never printed): ELEVENLABS_API_KEY
  = present (process env, not in `.env`), KAGGLE_USERNAME/KAGGLE_KEY = MISSING,
  HF_TOKEN = MISSING (HF Hub rate limits apply; set one before big pulls).
- HF cache already holds: Bisher, Hemgg, joka13, Gustking, XLS-R-300M,
  caa-speech RawNet2, nllb (from earlier candidate probes — reuse).

## Data on disk (private, local-only)
- `samples/real/whatsapp_opus/`: 86 files (84 `.opus` 16 kHz mono + 2 `.ogg`),
  10.5 min total, 1.2–57.8 s each.
- `samples/real/phone_calls/`: 19 `.amr` (AMR-NB 8 kHz), 16.49 h total;
  one 2.6 s file (skip per brief).
- Public: `data/raw/LibriSpeech` test-clean on disk; `data/phase2_manifest.csv`
  (419 rows, committed) + `data/manifest.csv` (2629 rows, ignored).

## Phase checklist
- [x] **PHASE 0 — checks + CURRENT-model baseline** (2026-10-16, commit pending)
  - Done: env audit above; full baseline over public labels.csv (19) +
    all 86 WhatsApp notes + 281 strided phone windows (5 s / 180 s, cap 25/file,
    1 file skipped) + 6-codec shift probe (Opus 12/24/32k, MP3, AMR-NB, G.711).
  - Tool: `tools/baseline_private_phase0.py`
    (`--only public|whatsapp|phone|codec|md`; resume caches in
    `samples/.phase0_local/`).
  - Artifacts: `reports/baseline.json` + `reports/baseline.md` (regenerated,
    anonymized, PII-scanned clean).
  - HEADLINE RESULT: current production model (rafnet, tau 0.3611/0.8611)
    flags **43/86 WhatsApp notes (FAR 0.50, mean P(AI) 0.8528)** and
    **116/281 phone windows (FAR 0.41, mean 0.8449)** as SYNTHETIC. The 2 .ogg
    notes score 0.8899 / 0.9021 -> SYNTHETIC (History failure REPRODUCED).
    Clean public real FAR 0.0 (n=4). Codec probe: Opus/MP3 deltas ~0.0;
    AMR-NB/G.711 push low-P real clips UP (+0.24 max, no flip in n=10 probe).
  - Bugs filed for Phase 6: (1) API rejects `.opus` uploads (extension gate in
    `src/rafnet/soc/audio_io.py`); (2) one phone window crashed a conv kernel
    on a degenerate segment (recorded in baseline.json, needs guard).
- [x] **DATA UPDATE — private preprocessing** (2026-10-16/17)
  - Tool `tools/prepare_private_data.py` (bundled ffmpeg; energy VAD; 4–6 s
    cuts; QC drops silence/clipped, quarantines tonal; deterministic order).
  - 763 segments: adapt=75 (whatsapp_early 15 + call_A 60, caps 40/file and
    calls ≤4x WhatsApp enforced, 618 surplus quarantined), test=70
    (whatsapp_late 30 incl. the 2 .ogg + whole smaller contact 40),
    quarantine=618. 41 short files skipped (<4 s, incl. the 2.6 s call file).
  - Local-only: `samples/real/manifest.csv`, `manifest_summary.json`,
    `segments/<test|adapt|quarantine>/*.wav` (anon segids, never committed).
  - Committed counts: `reports/data_splits_private.md`.
  - Speaker-leak: MFCC+logreg guesses 4 private groups at 0.676 (chance 0.25)
    = HIGH (channel-driven, expected). Mitigation: adapt share ~2% of human
    train + 3208 public speakers (below).
  - Acoustic gap (`reports/acoustic_gap.md`): private is +11 dB louder,
    −11 dB SNR, half the bandwidth (rolloff 2157 vs 4184 Hz) vs clean public.
    Gain probe: loudness moves P(AI) only ±0.006 — NOT the cause. Training
    must cover noisy + band-limited + codec-degraded real (binding).
- [x] **PHASE 1 — public human data** (2026-10-17)
  - 6978 clips / 3208 verified speakers (FLEURS en/hi/ta, SLR65 Tamil crowd,
    SLR118 GramVaani Hindi telephone spontaneous, Libri dev-clean+other).
  - Tools `tools/fetch_phase1.py` (+resume, budget/disk guards),
    `tools/respeak_phase1.py` (verified speaker ids). Manifest
    `data/manifest_phase1.csv` (local). Report `reports/data_phase1.md`.
  - Budget 4.59/15 GB; C: 38.9 GB free (10 GB floor held). Skipped list in
    report (gated: CV/AI4Bharat/TED-LIUM/VoxCeleb/VCTK-HF/Kaggle; too big:
    VoxPopuli/AMI-full/Libri-train/SBCSAE/GramVaani-train; blocked: parquet
    streaming via pyarrow DLL policy).
- [ ] **PHASE 2 — fake data** (ElevenLabs key present — check quota/terms and
  cache everything; edge-tts/Piper/Coqui/SAPI locally; public fake sets need
  downloads — same >15 GB gate; hold out 2–3 whole generators unseen)
- [ ] **PHASE 3 — codec-fair augmentation** (same random chains both classes;
  contingency table compression-vs-label)
- [ ] **PHASE 4 — model** (verify zero-shot candidates incl. cached XLS-R-300M;
  CPU-only => small train locally or PUBLIC-only notebooks for Colab/Kaggle;
  stack + calibrate + agreement-gated SYNTHETIC/CRITICAL)
- [ ] **PHASE 5 — honest eval** (`reports/model_eval.md/.json`, FAR-first,
  reliability curve, no tuning to look good)
- [ ] **PHASE 6 — integrate + verify** (load at startup, additive fields only,
  determinism 10x + restart, before/after on 2 .ogg + held-out, pytest +
  `npm run build`, retrain commands in final report). Includes `.opus` accept
  fix + degenerate-segment guard from Phase 0 bugs.

## Commands (verified)
- Public+private baseline (chunked, resumable):
  `.venv/Scripts/python tools/baseline_private_phase0.py --only whatsapp`
  `.venv/Scripts/python tools/baseline_private_phase0.py --only phone`
  `.venv/Scripts/python tools/baseline_private_phase0.py --only public`
  `.venv/Scripts/python tools/baseline_private_phase0.py --only codec`
  `.venv/Scripts/python tools/baseline_private_phase0.py --only md`
- Old public-only tool (kept): `.venv/Scripts/python tools/baseline_phase0.py`
- Retrain/eval (Phase 3–5, from prior work, unchanged):
  `.venv/Scripts/python tools/cache_scores_phase3.py --only mo|bisher|rafnet`
  `.venv/Scripts/python tools/train_stacking_phase3.py`
  `.venv/Scripts/python tools/evaluate_phase4.py`

- [x] **PHASE 2 — fake data** (2026-10-17/18)
  - Train 1871 (edge-tts 360/5 voices, Piper 600/4, SAPI 200/2, crudeVC 771/2;
    en+hi+ta+scam scripts) + unseen TEST 235 (ElevenLabs 25 built-in voices,
    quota 0->1919/10000, caps honored; piper-rohan 150; edge-Valluvar 60).
  - Tools `tools/fetch_elevenlabs_test.py`, `tools/synth_phase2_local.py`
    (chunked `--engine/--only/--quota`, disk floor, resume).
  - Manifests `data/manifest_phase2*.csv` (local). Report `reports/data_phase2.md`.
  - Skipped with reasons: XTTS (no hi/ta), ASVspoof/WaveFake/MLAAD/ITW/Kaggle
    (gated/no creds). Honest shortfall: 2106 vs 4000 target, 5 vs 10+ engines.
  - Budget ~4.9/15 GB; C: 37.4 GB free.

- [x] **PHASE 3 — codec-fair augmentation** (2026-10-18)
  - `src/rafnet/data/augment_phase3.py`: 9 codecs x 3 rates x 5 noises x 3
    reverbs + gain + packet-loss, seeded per (clip, epoch), label-blind,
    on-the-fly (tempfiles deleted; no full copies). SLR28 real RIRs skipped
    (1.3 GB, mirror ~300 kB/s, no Range resume); synthetic IRs documented.
  - Proof `tools/verify_augment_phase3.py` -> `reports/augmentation_phase3.*`:
    codec-vs-label chi2 p=0.255, noise-vs-label p=0.372 (independent);
    60 real ffmpeg chains executed, 0 failures, 0 NaN.

## What is next (single next step)
PHASE 2 fakes (~10.4 GB budget left): edge-tts/Piper/XTTS/SAPI locally
+ public fake sets if ungated + ElevenLabs TEST-only (built-in voices,
hard credit cap — check quota endpoint first, cache all, stop cleanly).
Hold out 2–3 whole generators unseen. Then Phase 3 (codec-fair aug,
on-the-fly/small-batch, contingency table).
