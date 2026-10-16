# DATA_NOTES.md — voice-origin data plan, licenses, what exists, what is blocked

`data/` and `models/` are git-ignored. Nothing below was downloaded except
LibriSpeech test-clean (license-clean, §1). No dataset over 15 GB was fetched.

## 1. On disk NOW (measured 2026-10-12)

| Path | Content | License | Size |
|---|---|---|---|
| `data/raw/LibriSpeech/test-clean` | 40 speakers, 2620 clips, real read English speech | CC-BY 4.0 (OpenSLR) | 347 MB |
| `samples/win_tts/` (9 wavs) | MS SAPI David/Zira TTS, EN/HI/TA texts | Generated locally, MS voice SW license | 2 MB |
| `data/raw/` otherwise | empty | — | — |

## 2. Sources to fetch, in order (NOT downloaded — action required)

| # | Dataset | Labels available | License gate | Approx size | Status |
|---|---|---|---|---|---|
| 1 | ASVspoof 2019 LA+PA | bonafide/spoof + replay | Written license request to U. Edinburgh consortium | LA ~20 GB, PA ~30 GB | BLOCKED: needs your form + approval (>15 GB — ask first) |
| 2 | MLAAD (multi-language, 52 TTS systems, 23 languages incl. Hindi) | bonafide + per-system TTS | Check card (research use) | ~100 GB+ full | BLOCKED: too big; plan: download 2–3 language subsets |
| 3 | WaveFake (LJSpeech vocoders) | bonafide + 6 vocoders | MIT (check mirror card) | ~16 GB | BLOCKED: >15 GB — ask first |
| 4 | In-the-Wild | bonafide/spoof, in-the-wild conditions | Check mirror card | ~5 GB | Available via HF mirrors, license unclear — verify first |
| 5 | Fake-or-Real (FoR) | bonafide + commercial TTS | Check card | ~10 GB | Verify license first |
| 6 | Common Voice (hi/ta/en subsets) | bonafide multilingual | CC0 | subset-size dependent | OK to fetch subsets (<15 GB) |
| 7 | ASVspoof 2017 / ReMASC | replay sessions | License / research | varies | Verify per set |
| 8 | Self-made replay (`tools/make_replay.py`) | HUMAN_RECORDED_REPLAY, exact provenance | Own recordings | small | READY (needs speakers+mic) |
| 9 | Local TTS (Piper offline; Coqui XTTS if installable) | generator-labelled fakes | Model licenses (Apache/MPL) | small | Piper OK offline; XTTS needs GPU/time |
| 10 | Commercial TTS | generator-labelled | NO KEYS IN ENV — stopped | — | BLOCKED: needs ELEVENLABS_API_KEY etc + terms check |
| 11 | Assistant voices | own recordings only | Own recordings | small | READY (`tools/record_assistant_voice.py` guide) |

Missing env (checked 2026-10-12): `KAGGLE_USERNAME`, `KAGGLE_KEY`,
`ELEVENLABS_API_KEY` all absent. To create Kaggle keys: kaggle.com →
avatar → Settings → API → Create New Token → save `kaggle.json` to
`%HOMEPATH%\.kaggle\` (or set the two env vars). Then say the word and the
Phase-1 fetch proceeds.

## 3. Terms notes (commercial TTS data generation)

- ElevenLabs: generating clips to train a third-party detector may conflict
  with service terms — read the current ToS before generating; STOP for that
  provider if unclear. No key present, so nothing was generated.
- Same rule for OpenAI/Google/Azure TTS: key + terms check first.
- Assistant voices (ChatGPT voice, Gemini Live, Meta AI, Claude voice):
  record ONLY your own live sessions with the tool + README in
  `samples/assistant/<name>/`. No scraping, no redistribution notes beyond
  personal research use. Note: Claude has no public voice product, so expect
  no "Claude voice" sample to exist.

## 4. Manifest contract (`tools/make_manifest.py`)

`data/manifest.csv` columns: path, label_origin, generator_family,
generator_name, language, speaker_id, source_dataset, split, codec, channel.
Splits are SPEAKER-DISJOINT and GENERATOR-DISJOINT (enforced by the script:
no speaker_id and no generator_name may appear in two splits; violations
fail loudly). Current manifest: LibriSpeech (HUMAN_LIVE-analog bonafide) +
win_tts (AI_TTS/microsoft-sapi-like) only — a diagnosis probe, NOT training
data (n≈2600+9, single-dataset bonafide, one TTS family: unfit for training).
