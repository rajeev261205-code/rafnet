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
- [ ] **DATA UPDATE — private preprocessing** (next; local compute, no download)
  - Decode (bundled ffmpeg) -> 16 kHz mono WAV + 8 kHz-origin flag; VAD;
    4–6 s segments; drop silence/hold-music/IVR/clipped.
  - Write `samples/real/manifest.csv` (local-only!) with
    source/original/call-id/times/codec/speaker_group (one group per contact,
    "whatsapp" separate). Balance: max 40/file, calls total <= 4x WhatsApp.
  - Split WITHOUT leakage (by group/call): hold out ENTIRE groups as TEST =
    later-half-date WhatsApp + one whole contact's calls + the 2 .ogg. Never
    train/tune/select on them. Rest = <=15% domain-adaptation share.
  - Speaker-leak check: classifier guessing speaker_group from embeddings;
    shrink private share if high.
  - Suggested tool: `tools/prepare_private_data.py` (new). VAD: energy-based
    numpy or `silero-vad` via torch.hub (small download <100 MB, allowed) —
    decide at implementation time; no 15 GB download needed.
- [ ] **PHASE 1 — public human data** (needs downloads; STOP before >15 GB
  without explicit go-ahead; report plan first per user instruction)
  - Common Voice en-IN/hi/ta + en subsets (CC0, per-subset), LibriSpeech/VCTK
    subsets, noisy/conversational if small. Target 4000+ clips / 800+ speakers
    if disk allows (41 GB free — plan sizes first). Manifest: data/manifest.csv
    + licenses. NOTE: Kaggle creds missing (skip Kaggle sets); no HF_TOKEN
    (expect slow Hub pulls).
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

## What is next (single next step)
DATA UPDATE preprocessing (`tools/prepare_private_data.py`): VAD-segment the
86 WhatsApp notes + 18 usable calls into 4–6 s clips, write local-only
`samples/real/manifest.csv` with group-disjoint TEST holdouts, balance caps,
and a speaker-leak check. No download over 15 GB involved; proceed.
