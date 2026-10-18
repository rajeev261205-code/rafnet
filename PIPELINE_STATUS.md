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

## NIGHT PAUSE (2026-10-17/18 evening→night) — resume here
Background embed job is STOPPED. No embed/python job is running. No phase
beyond 4.2a/partial-4.2b was started. Nothing was trained on TEST. Private
clips stayed local. Disk C: ~35 GB free (floor 10 GB held).

WHAT IS DONE (this session):
- CLEAN embedding pass COMPLETE: **7882 / 7882 unique, 0 dupes**.
  Files `data/embed_chunks/chunk_<CC>_clean_part<N>.npz` (40 parts: 10
  chunks x 4 parts). Verified with the command below.
- AUG0 embedding pass (Phase-3 seeded chains, `--epoch 0`) PARTIAL:
  chunks **0 1 2 3 4 5** done = **4730 / 7882 unique** (24 part files).
  Per-chunk rows saved: 0:789, 1:789, 2:788, 3:788, 4:788, 5:788.
  Chunks **6,7,8,9 NOT started** (3152 rows left).
- New tools added (committed, local artifacts git-ignored):
  `tools/train_head_phase4.py` (Phase 4.2b head; SMOKE-TESTED on clean only —
  3-epoch tiny run works end-to-end) and `tools/stack_phase4.py`
  (Phase 4.2c stack; COMPILES, NOT yet run — still a draft).

WHAT IS LEFT:
- aug0 embeddings for chunks 6,7,8,9 (~16 min/chunk; occasional ~5-min slow
  patches from thermal/AV, harmless — save is per-200-row part and resumable).
- Then steps 1–4 below (train head for real, stack, Phase 5 eval, Phase 6).

VERIFY EMBEDDING COUNTS (run from repo root):
```
.\.venv\Scripts\python.exe -c "from pathlib import Path; import numpy as np; ids=[]; [ids.extend(np.load(str(p),allow_pickle=True)['ids'].tolist()) for p in sorted(Path('data/embed_chunks').glob('chunk_*_clean_*.npz'))]; print('clean', len(ids), len(set(ids)))"
.\.venv\Scripts\python.exe -c "from pathlib import Path; import numpy as np; ids=[]; [ids.extend(np.load(str(p),allow_pickle=True)['ids'].tolist()) for p in sorted(Path('data/embed_chunks').glob('chunk_*_aug0_*.npz'))]; print('aug0', len(ids), len(set(ids)))"
```

EXACT RESUME COMMANDS:
Per-chunk in the foreground (~16 min each; output MUST be redirected or the
tool timeout kills python while it blocks writing to a PowerShell pipe):
```
& '.\.venv\Scripts\python.exe' -u tools\embed_phase4.py --chunk 6 --chunks 10 --epoch 0 *> data\embed_chunks\log_aug0_c6.txt
& '.\.venv\Scripts\python.exe' -u tools\embed_phase4.py --chunk 7 --chunks 10 --epoch 0 *> data\embed_chunks\log_aug0_c7.txt
& '.\.venv\Scripts\python.exe' -u tools\embed_phase4.py --chunk 8 --chunks 10 --epoch 0 *> data\embed_chunks\log_aug0_c8.txt
& '.\.venv\Scripts\python.exe' -u tools\embed_phase4.py --chunk 9 --chunks 10 --epoch 0 *> data\embed_chunks\log_aug0_c9.txt
```
Detached background loop (survives shell tool timeouts; what was used here).
Create `%TEMP%\opencode\run_aug0b.cmd` with a `for %%C in (6 7 8 9) do`
loop calling the per-chunk command, appending to
`data\embed_chunks\bg_aug0_all.out` (stdout) and `.err` (stderr), then:
```
$p = Start-Process -FilePath "cmd.exe" -ArgumentList '/c','%TEMP%\opencode\run_aug0b.cmd' -WorkingDirectory "C:\Users\rajee\RAFNet" -WindowStyle Hidden -PassThru; $p.Id
```
Poll with: `Get-Content data\embed_chunks\bg_aug0_all.out -Tail 6`.
`ALL_DONE` appears in the log when all chunks finish. Log files: stdout =
`data/embed_chunks/bg_aug0_all.out`, stderr = `data/embed_chunks/bg_aug0_all.err`.

ENVIRONMENT + PATHS:
- python: `C:\Users\rajee\RAFNet\.venv\Scripts\python.exe` (3.12.0, torch
  2.14.1+cpu, CUDA unavailable; 12 logical cores). workdir = repo root.
- ffmpeg: bundled imageio-ffmpeg 7.1 (no system ffmpeg; winget install stalls).
- encoder: `microsoft/wavlm-base-plus` from local HF cache
  (`%USERPROFILE%\.cache\huggingface\hub\...`, HF_HUB_OFFLINE=1); manual
  zero-mean/unit-var norm (no preprocessor config in cache).
- key files: `tools/embed_phase4.py`, `tools/train_head_phase4.py`,
  `tools/stack_phase4.py`, `tools/make_splits_phase4.py`,
  `tools/zeroshot_val_phase4.py`, `data/splits_phase4.json` (local),
  `data/embed_chunks/` (local). Committed reports = aggregates only.
- data safety: `samples/` + `data/` + `models/` git-ignored; committed files
  carry anonymous ids / counts only (PII-scanned before commit). Private
  clips NEVER leave this PC; TEST split (incl. private groups + unseen
  generators) is never read for training/tuning.

BUG FOUND + FIXED earlier (code committed): `_save_part` restarted part
numbering at 0 every run, so a resumed chunk OVERWROTE the previous run's
part0 instead of appending. Fix in `tools/embed_phase4.py`: part index
starts at max(existing)+1; loader unions all parts and skips done ids.

BEFORE THIS SESSION (committed; git log fd2dfdb, 51f0815, a6a88d5, ...):
- Phases 0–3 + 4.1: disjoint splits (`data/splits_phase4.json`, local):
  train 6509 (4917 real incl. 75 private-adapt + 1592 fake), val 1373,
  test 1347 (incl. 70 private-TEST + 235 unseen fakes). Zero human-speaker /
  private-group overlap (asserted in tool).
- Zero-shot val (500 clips, offline): mo-thecreator AUC 0.66 FAR 0.42 KEEP,
  Bisher AUC 0.62 FAR 0.34 KEEP (`reports/zeroshot_phase4.*`).

## SESSION PAUSE 2026-10-19 00:40 (phase 4.2c/5 partial; resume here)

EMBEDDINGS (all local, `data/embed_chunks/`; git-ignored):
- clean (train+val): 7882/7882 unique DONE (chunks 0..9).
- aug0: 7882/7882 unique DONE (chunks 0..5 earlier; chunks 6..9 finished this
  session, chunk 9 END rc=0). Logs `data/embed_chunks/bg_aug0_all.out|.err`.
- clean TEST: 1347 added ONCE (chunks 0..9, disjoint ids, appended as extra
  parts under the same `chunk_<i>_clean_*` prefixes; loader unions them) ->
  clean total 9229 = 7882 + 1347, all unique.

TRAINED / FITTED (local artifacts; committed reports = aggregates):
- head clean+aug0 (final): `models/head_phase4.pt` + `reports/head_phase4.{json,md}`
  (best epoch 1, val AUC 1.0, FAR 0.0183, band [0.45,0.75]).
  clean-only head also kept: `models/head_phase4_cleanonly.pt` +
  `reports/head_phase4_cleanonly.{json,md}`.
- stack fit on VAL only: `models/stack_phase4.json` + `reports/stack_phase4.md`
  (coef=[8.9958,-0.042,1.3652,0.2068] T=0.3 band=[0.45,0.925] val_auc=1.0
  FAR=0.0018). Features = [head, rafnet, mo, bisher]. NEVER fit on TEST.

SCORING (this is the only job running; detached, logs under `data/`):
- VAL `data/scores_phase4_val.json`: 1373 clips, all 4 models complete (used
  for the VAL fit above).
- TEST `data/scores_phase4_test.json`: IN PROGRESS (file is written
  INCREMENTALLY, flushed every 200 rows, and a re-run RESUMES by skipping any
  model already cached per clip). Current state at pause: head 1347/1347 done,
  rafnet 1347/1347 done (1729 s), mo ~750/1347, bisher 0/1347. So the file is
  PARTIAL (rows have head+rafnet+mo, `bisher` missing) and is NOT final until
  the job prints `wrote scores_phase4_test.json` and exits.
  - Safe to stop now: on resume only the missing mo rows + ALL bisher rows are
    re-scored (est. ~13 min); the expensive rafnet pass is reused.
  - Prefer to WAIT ~13 min (until ~00:53) so the file lands complete with no
    re-run. Do NOT start the Phase 5 evaluator until that line appears.

WHAT IS LEFT (in order):
1. Finish TEST scoring (see resume cmd below) -> complete
   `data/scores_phase4_test.json` (1347 clips, all 4 fields).
2. Phase 5 honest eval (TEST read once, at the end):
   `.\.venv\Scripts\python.exe tools\evaluate_phase5.py` -> FAR-first metrics
   per group (incl. private groups + unseen generators fake-elevenlabs /
   piper:hi-rohan / edge:Ta-ValluvarNeural), reliability bins, missing-row
   honesty -> `reports/model_eval.{json,md}`.
3. Phase 6 verify: `.\.venv\Scripts\python.exe tools\phase6_verify.py`
   (10x determinism + restart, degenerate-segment guard <1600 samples, `.opus`
   accept, v1-vs-v2 before/after on 2 private .ogg + held-out + codec probe)
   -> `reports/phase6_verify.{json,md}`.
4. Update this file with Phase 5/6 results; commit code + reports per phase.

PHASE 6 CODE already in place (compiles; pytest 430 passed, 3 skipped):
`audio_io.py` (.opus accept), `model_adapter.py` (degenerate guard),
`head4.py` (live head; fp16-rounded mean/std to match training -- verified
diff 0.0000 vs cached VAL head scores), `ensemble.py` v2 (additive),
`service.py` (startup head load + additive v2 fields), `routes.py` (additive
v2 fields in the live chunk response), `tools/evaluate_phase5.py`,
`tools/phase6_verify.py`. Frontend build OK (`tsc && vite build`, 2.06 s);
frontend unchanged.

RESUME COMMANDS (PowerShell, workdir = repo root; detached with redirection):
- TEST scoring (resumes; skips head+rafnet already cached):
  `$env:HF_HUB_OFFLINE='1'; Start-Process -FilePath '.\.venv\Scripts\python.exe' -ArgumentList '-u','tools\stack_phase4.py','--split','test' -WorkingDirectory 'C:\Users\rajee\RAFNet' -WindowStyle Hidden -RedirectStandardOutput 'C:\Users\rajee\RAFNet\data\stack_test.out' -RedirectStandardError 'C:\Users\rajee\RAFNet\data\stack_test.err'`
- Phase 5 eval (only after `wrote scores_phase4_test.json` appears):
  `.\.venv\Scripts\python.exe tools\evaluate_phase5.py`
- Phase 6 verify: `.\.venv\Scripts\python.exe tools\phase6_verify.py`
- VAL re-fit (only if needed; already done): `--split val` with same pattern.

ENV/PATHS (unchanged): python `C:\Users\rajee\RAFNet\.venv\Scripts\python.exe`
(3.12, torch 2.14.1+CPU, HF_HUB_OFFLINE=1, CPU only); bundled imageio-ffmpeg;
`samples/` + `data/` + `models/` git-ignored; private clips never leave the PC;
committed reports = aggregates / anonymous ids only.
