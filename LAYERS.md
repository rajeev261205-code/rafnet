# LAYERS.md — RAFNet detection layers

Every analysis (`POST /api/analyze`) and live chunk (`POST /api/live/chunk`)
returns a `layers` array and a fused `verdict`. This file explains what each
layer is, which are heuristic, and the known limitations.

## Layer 1 — Spectral & Vocoder Artifacts (`kind: heuristic`)

Computed in `src/rafnet/soc/forensics.py` (numpy only) from the 16 kHz mono
signal. Score 0–100 = heuristic synthetic-likelihood, or `null` on silence.

Raw metrics (all measured, reported verbatim in `metrics`):

- `flatness` — spectral flatness (geometric/arithmetic mean of the power
  spectrum). Flat spectra look noise-like.
- `highband_ratio` — energy share in 4–8 kHz.
- `hnr_db` — harmonic-to-noise ratio: MEDIAN over per-frame (~128 ms)
  autocorrelation peaks in the pitch band. (Per-frame since 2026-10-12; the
  previous whole-clip estimate smeared vibrato/intonation and scored clean
  sustained voicing ~1 dB. Residual: strong vibrato still reads low — the 6 dB
  trip-wire is unvalidated, see below.)
- `jitter` — LOCAL cycle-to-cycle pitch variation (median relative successive
  F0 difference over consecutive voiced frames; needs ≥ 2 valid steps, else
  `null`). (Local since 2026-10-12; the previous global std/mean measured
  intonation range and flagged expressive speech.)
- `shimmer` — relative spread of frame energy (amplitude-instability proxy).
- `flux` — mean frame-to-frame spectral change (vocoder splicing proxy), with
  the per-frame denominator floored at 1% of clip mean frame energy. (Floored
  since 2026-10-12; without the floor, pause boundaries divided by ~1e-9 and
  ANY speech with pauses read as extreme splicing, e.g. flux 715392 on a
  phrase with gaps. Known limitation: mean-pooling dilutes isolated splices,
  so the 0.5 wire is effectively dormant at utterance scale — documented, not
  tuned, pending labeled data.)
- `rolloff_hz` — 85% spectral rolloff (reported, not scored).

Each metric is mapped 0–100 by a piecewise-linear ramp whose endpoints live
in `src/rafnet/soc/layer_config.py` (`SPECTRAL_METRICS`), then combined by
weights. Evidence strings fire when a metric crosses its trip-wire in
`SPECTRAL_EVIDENCE`, always quoting the measured value and the threshold,
e.g. `spectral flatness 0.42 exceeds heuristic threshold 0.3 (heuristic)`.

**Heuristic — not a validated detector.** The ramps and trip-wires are
engineering defaults without labeled-data validation.

## Layer 2 — Neural Detection (RAFNet) (`kind: model`)

The existing real model outputs, passed through unchanged
(`classical_probability`, `quantum_probability`, `fused_probability`,
`calibrated_probability`, Gateway 1/2 weights, `prediction`, `confidence`).
Layer score = `prob_synthetic × 100`. When the backend runs with
`MOCK_INFERENCE=true` the evidence says so explicitly — and the mock is
intentionally non-diagnostic (constant `UNCERTAIN` at p = 0.5), so mock mode
can never manufacture `REAL`/`SYNTHETIC` verdicts for quiet/loud input.

## Layer 3 — Liveness & Replay (`kind: heuristic`)

Same numpy module. Score 0–100 = heuristic replay/pre-recorded likelihood, or
`null` on silence.

- `pause_fraction` — share of 200 ms blocks below −34 dBFS (absence of
  natural pauses/breaths). Needs ≥ 2 s of audio (`MIN_PAUSE_ANALYSIS_SECONDS`);
  skipped with a note on shorter clips.
- `highfreq_ratio` — energy above 7 kHz; near-zero suggests band-limiting or a
  codec cutoff.
- `loop_score` — max cosine similarity of mean log spectra across
  **non-overlapping** 1 s windows (live mode only compares windows ≥ 1 s
  apart so the rolling buffer cannot self-match). Needs ≥ 2 s of audio.
- `noise_floor_db` — quietest AUDIBLE 200 ms block (blocks below the silence
  floor are digital-black gaps, not room tone; since 2026-10-12 — previously
  any clip with pauses reported −180 dB and tripped the wire on content).
- `clip_fraction` — share of samples past ±0.98 (loudspeaker-style
  saturation proxy).
- `tail_ratio` — quietest/loudest block energy (dry-tail proxy).
- `input_path` — the **input-path signal**: the frontend sends the active
  microphone's device label / sample rate / channels with each live chunk,
  and the backend matches the label against `VIRTUAL_INPUT_KEYWORDS`
  (virtual cable, voicemeeter, stereo mix, loopback, blackhole, obs,
  soundflower, …). A match means *audio is routed from another application* —
  a real, labeled signal, not a voice-authenticity judgment. File uploads
  carry no capture device, so this signal is live-only.

**Heuristic — not a validated detector.** Replay and voice-conversion
detection performance need labeled data to validate (see below).

**Lossy uploads (mp3/aac/m4a/ogg):** these encodings band-limit spectra and
set a codec noise floor by design, so Layer 3 excludes band-limit/codec-cutoff
evidence from the score for lossy input (`bandlimit_excluded: true`,
`source_codec`/`source_lossy` in metrics) and notes it in the evidence. When a
lossy upload shows *only* codec evidence, the verdict is `UNCERTAIN` with
“lossy file: replay cannot be assessed” — never `REPLAY` on that basis.
Genuine repetition/saturation evidence still scores, and live-mic chunks (raw
PCM) plus lossless uploads (wav/flac) always keep full replay analysis.

**Narrowband-stationary tones (since 2026-10-12):** a sustained tone has ~zero
energy above 7 kHz and identical window spectra BY CONSTRUCTION. Scoring those
as a codec cutoff + replay loop made a pure sine wave read
`REPLAY_PRERECORDED`. When content is tonal-stationary (flatness < 0.02,
spectral flux < 0.05, spectral crest > 10 — a content description, not a
detector threshold), band-limit and loop abstain (`tonal_stationary: true`,
`tonal_excluded: true`, recorded in evidence); pause/noise/clipping/tail
evidence is unaffected. Band-limited BROADBAND content (e.g. lowpassed noise:
flat, low crest) still trips the band-limit wire, and non-tonal exact repeats
still trip the loop wire — both covered by unit tests.

## Layer 4 — Voice Enrollment Biometric (`kind: unavailable`)

No speaker-embedding model is installed (`speechbrain`, `resemblyzer`
absent — probed for real in `src/rafnet/soc/speaker.py`). The layer returns
`kind: "unavailable"`, `score: null`, and the UI shows “Not enrolled / not
available”. No similarity is ever invented. `GET /api/enrollment` reports the
live capability probe; enabling enrollment needs an approved open-source
speaker model (e.g. `pip install resemblyzer` plus its model download).

## Verdict fusion (`verdict`)

Rule-based evidence fusion, evaluated in order (rules + defaults in
`src/rafnet/soc/layer_config.py`, all unvalidated):

1. Virtual/loopback input flagged → `EXTERNAL_APP_SUSPECTED`.
2. `prob_synthetic ≥ 0.86` → `AI_SYNTHETIC`.
3. Model `SYNTHETIC` (conf ≥ 0.60) **and** Layer 1 ≥ 50 → `AI_SYNTHETIC`.
4. Lossy upload with only codec evidence → `UNCERTAIN` (“lossy file: replay
   cannot be assessed”).
5. Layer 3 ≥ 60 → `REPLAY_PRERECORDED`.
6. Model `REAL` (conf ≥ 0.60) with both heuristic layers < 50 → `HUMAN_LIVE`.
7. Otherwise `UNCERTAIN` with the model confidence.

`verdict.confidence` is the fusion confidence; `risk_score`/`risk_level` come
from the existing risk engine (`compute_risk`, unchanged). `reasons` names the
layers/evidence that drove the label; `limitations` always includes the
unvalidated-thresholds notice plus dynamic notes (short audio, single window,
unavailable layers).

Live mode keeps a rolling window (`HYSTERESIS_WINDOW = 5`,
`HYSTERESIS_MIN_VOTES = 3`): the session verdict only switches when the new
label holds ≥ 3 of the last 5 chunk votes, so it does not flicker. Chunk
responses carry both the per-chunk `verdict` and the `session_verdict`.

## How to evaluate

1. Collect labeled audio (see `samples/README.md`):
   `samples/human_live/*.wav`, `samples/human_replayed/*.wav`
   (played through speakers and re-recorded), `samples/tts/*.wav`,
   `samples/voice_changed/*.wav`.
2. Run `.venv/Scripts/python tools/evaluate_layers.py` (repo root).
3. Read `reports/layer_eval.md` / `.json`: verdict confusion matrix and
   per-layer accuracy under documented evaluation-time thresholds
   (Layer 1 ≥ 50 = synthetic-like, Layer 3 ≥ 60 = replay-like).
4. `GET /api/evaluation` serves the report; the Model Intelligence page shows
   the measured numbers, or “Not evaluated” when no report exists.

Known limitations: replay and voice-conversion detection need labeled data to
validate — until `samples/` is populated and the report shows measured
numbers, treat Layers 1/3 and the verdict as unvalidated heuristics around a
real (but synthetic-only-trained, see `AI_MODEL_SPEC.md`) neural model.
