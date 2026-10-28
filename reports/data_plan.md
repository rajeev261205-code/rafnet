# Phase 2 data plan (what exists, what is pending, what you must provide)

## Exists now (verified on disk)

- `data/phase2_manifest.csv` — 419 rows: train 240 (120 real + 120 crude-VC),
  val 80, test 80, my_samples 19. Speaker-disjoint train/val/test (verified
  0 overlap); fake sub-settings disjoint (train ps_up/ps_down, val ts_slow,
  test ts_fast); my_samples generators appear in NO train row (verified).
  Report: `reports/data_splits.md`. Inventory: `reports/data_inventory.*`.
- `data/synth_vc/` — 240 crude-VC wavs (librosa pitch/time warp stopgap).
- `tools/augment_phase2.py` — 9 train-only chains (5 codecs + resample +
  reverb + noise + packet-loss), label-blind, self-tested on 2 real + 2 fake
  (36 cases, `reports/augmentation_check.json`).

## STOPPED: Kaggle (credentials missing)

`KAGGLE_USERNAME` / `KAGGLE_KEY` are not set in this environment, so no
Kaggle download was attempted. To enable Kaggle pulls, set both env vars and
re-run whatever pull script the notebook phase provides. Nothing was
downloaded from Kaggle in this phase.

## Public sets (NOT downloaded — approval gate)

Rule: nothing over 15 GB without your explicit go-ahead. Sizes below are
size-classes from the official dataset pages (verify at download time);
licenses are the access terms each publisher states (read before use).

| set | real/fake | size class | access / license note | use |
|---|---|---|---|---|
| ASVspoof 2019 LA | both | M–L (ask first) | official site, research agreement required | train/val fake diversity + bona fide |
| ASVspoof 2021 LA+DF | both (+codecs) | L (ask first) | official site, research agreement required | codec-robust fake, DF track |
| In-the-Wild | both (celebrity/politician) | M | research use per publishers | unseen-generator test |
| WaveFake | fake (LJSpeech vocoders) | S–M | research per publishers | cheap fake diversity |
| MLAAD | fake (multilingual, incl. Hindi?) | L (ask first) | research per publishers | language coverage |
| Common Voice (en + hi + ta subsets) | real | S per subset (take validated only) | CC0 | Indian-language + noisy real |
| LibriSpeech (have test-clean) | real | on disk | public-domain-derived audiobooks | already in manifest |

Notebook phase (`notebooks/`) will pull an S/M subset first; L sets wait
for your approval.

## Needed from you (nothing else unblocks these)

1. WhatsApp/Opus human voice notes (the History failure) — forward 10–20
   `.ogg`/Opus notes (varied speakers, 3–10 s) into `samples/real/` (they stay
   local-only, never committed). This is the single most important data gap:
   the codec-bias claim cannot be closed without them.
2. Indian English / Hindi / Tamil speech — 20+ clips per language, mic or
   phone, into `samples/real/` (local-only).
3. Phone-line / laptop-mic / noisy-room clips — a few per condition.
4. Your own assistant-voice loopbacks only (`tools/record_assistant_voice.py`);
   never commercial TTS output unless you provide keys AND the terms allow it.
5. Commercial TTS keys — only if you want that family covered.
6. `HF_TOKEN` (optional but recommended) — Hub rate limits without it slowed
   Phase 1 downloads; `winget install --id Gyan.FFmpeg -e` for system ffmpeg.

## Known weaknesses (stated plainly)

- Train fake = ONE crude family (librosa warp). Any "trained head" from this
  alone proves the pipeline, not generalization. Real diversity comes from
  the notebook phase + your clips.
- No Hindi/Tamil, no WhatsApp, no phone-line, no replayed, no VC in train.
- my_samples (19) is the only unseen-generator gate and is tiny.
