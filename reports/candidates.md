# Phase 1 candidates — verified strong audio anti-spoofing models

Generated: 2026-10-15. Every entry was verified to exist and (where stated)
to load; every metric is measured on local files and saved in the JSON noted.
Nothing here is assumed — load failures and untested items state their reason.

## Method

- Hub search: `tools/candidates_search.py` (`HfApi.list_models` for
  "deepfake audio", "anti-spoofing", "ASVspoof", "spoof"; full hit list in
  `reports/candidates_search.json`).
- Zero-shot probe: `tools/test_one_candidate.py` — 6 files
  (2 LibriSpeech real + SAPI David/Zira + ChatGPT voice + Windows TTS,
  4 s cap, CPU). Per-model JSON: `reports/zeroshot_<id>.json`.
- Public-real + codec probe: `tools/candidates_public.py` — 6 LibriSpeech
  test-clean clips, speakers 1320/1580 (disjoint from samples/ speakers
  260/4446/5142/6930), plus Opus-24k / MP3-128k / AMR-NB / G.711 transcodes
  of 2 files. JSON: `reports/candidates_public.json`.
- GitHub/backbone/speechbrain: `tools/candidates_github.py`.
  JSON: `reports/candidates_github.json`.

`P(AI)` mapping is recorded per model (most publish `{0: fake-like,
1: real-like}`; the one exception is noted). AUC<0.6 = at chance = dropped.

## HF audio-classification probes (measured, 6-file subset)

| model | license (Hub tag) | params | load | AUC | EER | mean latency (CPU/4 s) | verdict |
|---|---|---|---|---|---|---|---|
| mo-thecreator/Deepfake-audio-detection | apache-2.0 | 94.6 M | ok (2.4 s cached) | 1.0 | 0.0 | ~330 ms | **KEEP — primary** |
| Hemgg/Deepfake-audio-detection | apache-2.0 | 94.6 M | ok | 1.0 | 0.0 | ~340 ms | **DROP — codec bias** (see below) |
| Bisher/wav2vec2_ASV_deepfake_audio_detection | apache-2.0 | 94.6 M | ok | 1.0 | 0.0 | ~340 ms | **KEEP — secondary** |
| MelodyMachine/Deepfake-audio-detection | apache-2.0 | 94.6 M | ok | 1.0 | 0.0 | ~340 ms | reserve (same family as mo-thecreator) |
| MelodyMachine/Deepfake-audio-detection-V2 | apache-2.0 | 94.6 M | ok | 0.75 | 0.25 | ~350 ms | drop (misses SAPI fakes: p≈0.0) |
| MattyB95/AST-ASVspoof2019-Synthetic-Voice-Detection | mit | 86.2 M | ok | 0.75 | 0.25 | ~1520 ms | drop (calls everything spoof) |
| MattyB95/AST-ASVspoof5-Synthetic-Voice-Detection | bsd-3-clause | 86.2 M | ok | 0.5 | 0.5 | ~1500 ms | **DROP — at chance** |
| WWWxp/wav2vec2_spoof_dection1 | apache-2.0 | 94.6 M | ok | 0.5 | 0.5 | ~370 ms | **DROP — at chance** (all p≈0.48) |
| TEC2004/SafeEar-ASV19-spoof-detection | apache-2.0 | — | **FAIL**: no transformers preprocessor (`Can't load feature extractor`) | — | — | — | drop (custom code, not loadable here) |
| joka13/deepfakeaudio | apache-2.0 | — | **FAIL**: same, no preprocessor_config | — | — | — | drop |
| caa-speech-detection-asvspoof2019/rawnet2 | mit | — | **FAIL**: same, custom RawNet2, no preprocessor | — | — | — | drop as HF weight; see GitHub note |
| Gustking/wav2vec2-large-xlsr-deepfake-audio-classification | apache-2.0 | ~317 M (xlsr-large) | **NOT DOWNLOADED**: ~1.2 GB weights, unauthenticated Hub too slow, timed out twice | — | — | — | pending (GPU-notebook only) |

## Public-real + codec probe (the decisive test)

6 LibriSpeech clips, speakers never in samples/; FAR = fraction with P(AI)≥0.5.

| model | public-real P(AI) | FAR | codec deltas (2 files × 4 codecs) | verdict |
|---|---|---|---|---|
| mo-thecreator | 0.0, 0.0001, 0.0008, 0.0, 0.0, 0.0 | **0.0** | max abs +0.014 (AMR) | **KEEP** — no false alarms, codec-stable |
| Bisher | 0.002×6 | **0.0** | max abs 0.0000 | **KEEP** — no false alarms, codec-stable |
| Hemgg | 0.72, 0.81, 0.80, 0.62, 0.63, 0.58 | **1.0** | AMR +0.23, G.711 +0.21 (pushes to 0.95+) | **DROP** — reproduces the reported failure: clean human speech flagged fake, narrowband codecs amplify it |

Hemgg is the cautionary result of this phase: AUC 1.0 on samples/ but
FAR 1.0 on unseen-speaker public speech. Sample-only metrics are not trusted;
the public + codec gate is mandatory (applied again in Phase 4).

## GitHub official repos (verified via api.github.com)

- TakHemlata/SSL_Anti-spoofing — EXISTS, MIT, XLS-R 300M + AASIST backend
  per repo description. No weight files in repo root (code + RawBoost only);
  checkpoints come from the authors' drive/release links and training needs
  ASVspoof data + GPU. Path: GPU notebook (Phase 3), not local CPU.
- clovaai/aasist — EXISTS, MIT, official AASIST. Same: code only in repo,
  weights via releases; needs ASVspoof data + GPU. Path: GPU notebook.
- asvspoof-challenge/asvspoof2021_baseline — 404 at that path (wrong URL,
  not evidence of absence; correct org path to be resolved in the notebook
  phase if needed). No claim made beyond the 404.
- Guessed raw checkpoint URLs for both repos returned 404 — recorded in
  `candidates_github.json`; correct links will be taken from each repo's
  README/releases during the notebook phase, never guessed again.

## Frozen backbones for a trained head (existence verified, weights NOT downloaded)

- facebook/wav2vec2-xls-r-300m — exists, arch Wav2Vec2ForPreTraining,
  pytorch_model.bin present. ~1.2 GB; CPU-load deferred to GPU notebook.
- microsoft/wavlm-large — exists, arch WavLMModel, pytorch_model.bin present.
- openai/whisper-small / whisper-medium — exist, WhisperForConditionalGeneration.
- All Hub pages show no license field (None) — treat as research-use default
  until the notebook phase records each model card's license explicitly.

## Speaker verification (enrollment layer)

- speechbrain/spkrec-ecapa-voxceleb — Hub repo VERIFIED, tag license
  apache-2.0, contains embedding_model.ckpt + hyperparams.yaml.
- speechbrain package is NOT installed in .venv — enrollment biometric
  stays "not installed" (UI must keep showing that). Install is
  `pip install speechbrain` (additive, Phase 5, no inference change until then).

## Shortlist for Phase 3

1. mo-thecreator/Deepfake-audio-detection (apache-2.0, 94.6 M, CPU ~330 ms /
   4 s) — primary. Zero-shot AUC 1.0 samples-probe, FAR 0.0 public-real,
   codec-stable.
2. Bisher/wav2vec2_ASV_deepfake_audio_detection (apache-2.0, 94.6 M,
   CPU ~340 ms) — secondary, independent training lineage, same gates passed.
3. Existing RAFNet branches as the third ensemble input (kept, not trusted
   alone — Phase 0 baseline AUC 0.5).

Ensemble + calibration + abstain band are fit on validation only (Phase 3);
test groups including `my_samples` stay untouched until Phase 4.

## What could not be downloaded and why

- Gustking/wav2vec2-large-xlsr… — ~1.2 GB weights; two unauthenticated
  downloads timed out (Hub rate limits without HF_TOKEN). Retry on GPU host
  with `HF_TOKEN` set, or drop — the two kept models already cover the
  wav2vec2 family.
- SSL_Anti-spoofing / AASIST / RawNet2 weights — no verified URL yet (404s on
  guessed paths); resolve from repo README/releases in the notebook phase.
- System ffmpeg via `winget install --id Gyan.FFmpeg -e` stalled on the
  ~100 MB download; codec work in this phase used the pip-bundled
  imageio-ffmpeg 7.1 binary (verified encoders: libopus, libmp3lame,
  libopencore_amrnb, pcm_alaw). Re-run the winget command to get a
  system-wide ffmpeg.
