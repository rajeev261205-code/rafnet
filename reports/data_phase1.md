# Phase 1 — public human speech (downloaded, ungated sources only)

Tool: `tools/fetch_phase1.py` (`--plan` verifies sizes first; `--do [--only K]`
downloads with resume, per-file disk guards, deletes archives after selection).
Speaker provenance fix: `tools/respeak_phase1.py` (TSV joins, path parsing).
Local manifest: `data/manifest_phase1.csv` (6978 rows, git-ignored).
Audio: `data/phase1/human/<src>/*.wav` (16 kHz mono, ≤10 s, git-ignored).
Budget: **4.59 / 15 GB** (per-file byte log in `data/phase1/download_log.json`).
C: floor 10 GB free enforced before every file (38.9 GB free at end).

## Kept clips (6978) and VERIFIED distinct speakers (3208)

| source | clips | speakers | license | notes |
|---|---|---|---|---|
| fleurs_en_train | 1500 | 318 | CC-BY-4.0 | read speech, TSV-joined ids |
| fleurs_hi_dev | 163 | 122 | CC-BY-4.0 | read Hindi |
| fleurs_hi_test | 241 | 150 | CC-BY-4.0 | read Hindi |
| fleurs_ta_dev | 228 | 152 | CC-BY-4.0 | read Tamil |
| slr65_ta_f/m | 1017 | 50 | CC-BY-SA-4.0 | crowdsourced Tamil volunteers |
| slr118_hi_dev/eval | 2429 | 2367 | GramVaani academic-only | **spontaneous Hindi TELEPHONE (8 kHz mp3), natural background — closest to deployment channel** |
| slr12_dev_clean/other | 1400 | 49 | CC-BY-4.0 | read English (clean + challenging) |

Language mix: English ~3600 (+2620 local LibriSpeech test-clean), Hindi ~2800,
Tamil ~1250. Noisy/conversational coverage: GramVaani telephone (spontaneous,
8 kHz, background noise) + dev-other (challenging read). 131 speaker ids span
dev/test of the same corpus (expected); cross-corpus ids are namespaced.

## Skipped (gated, too big, or technically blocked)
- Common Voice (HF repo is a stub — data removed), ai4bharat/indictts,
  TED-LIUM-HF, VCTK-HF, VoxCeleb-HF, CommonVoice-16 (401, need HF_TOKEN).
- Kaggle sets (no KAGGLE_USERNAME/KEY): ASVspoof (also needs research agreement).
- Too big for budget: VoxPopuli-en full (98.8 GB parquet), AMI-ihm full
  (15.2 GB), LibriSpeech train-100/360/500 (6–30 GB), SBCSAE (6.1 GB),
  GramVaani train-100h (2 GB, revisit if Phase 2 underspends), VCTK zip (10 GB).
- Blocked locally: any HF parquet streaming (`pyarrow.dataset` DLL denied by
  Application Control) — worked around with direct tar/zip + TSV downloads.
- fleurs_ta_test audio fetched but yielded 0 clips under the 2–12 s filter
  (long read sentences); Tamil covered by ta_dev + SLR65 (1245 clips), not re-fetched.

## What this unblocks / next
Phase 2 fakes (budget left ~10.4 GB): edge-tts/Piper/XTTS/SAPI locally +
ElevenLabs TEST-only (built-in voices, hard credit cap, cache all) + public
fake sets if ungated and small. Then Phase 3 codec-fair augmentation
(on-the-fly/small-batch, same chains both classes, contingency table).
