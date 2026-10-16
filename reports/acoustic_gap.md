# Acoustic gap — why compressed phone/WhatsApp human speech looks "AI" (measured)

Private real (763 VAD segments, adapt+test) vs clean public real (24 LibriSpeech
5 s cuts). Segment features from `tools/prepare_private_data.py` (librosa,
`samples/real/manifest_summary.json`, local). Aggregates only — no audio, no PII.

| feature | private median (mean) | clean public median | gap |
|---|---|---|---|
| RMS loudness | -15.7 dBFS (-16.9) | -27.1 dBFS | **private ~11 dB louder** (close-mic) |
| speech/silence SNR | 28.4 dB (28.6) | 39.6 dB | **private ~11 dB noisier** |
| spectral rolloff (85%) | 2157 Hz (2188) | 4184 Hz | **private ~half the bandwidth** (Opus voice / AMR-NB 4 kHz cap) |

Gain probe (legacy RAFNet, same clip at 0.25x–4x gain): P(AI) moves only
±0.006 (0.847–0.857, always UNCERTAIN) on both a private and a public clip —
**loudness does NOT explain the false alarms** for this path. (Ensemble/HF
paths re-tested in Phase 4.)

## Reading
Compression alone does not explain it (Opus re-transcode shift ≈ 0 in Phase 0,
yet FAR = 0.50 on WhatsApp). The measured separators are **noise + narrow
bandwidth + channel** (Opus/AMR vs FLAC), i.e. the training distribution
(clean, wideband, quiet read speech) does not cover the deployment channel.
Speaker-leak check on the same segments: logistic/MFCC guesses the 4
private groups at 0.676 CV accuracy (chance 0.25) — HIGH, as expected while
groups differ by channel; mitigation is the tiny adapt share (~2% of a
4000-clip human train) + 800+ public speakers, NOT dropping the data.

## Consequences for training data (binding)
1. Public real MUST include noisy + band-limited + conversational speech, not
   only clean read (LibriSpeech/VCTK/FLEURS): add VoxPopuli/TED-LIUM/VoxCeleb
   class material if ungated and within budget.
2. Phase 3 augmentation must span: Opus 12–32k, AMR-NB, G.711, MP3/AAC,
   8/16 kHz resampling, additive noise to ~25–30 dB SNR, band-limiting to
   ~2–4 kHz rolloff — applied IDENTICALLY to real and fake.
3. Contingency table (Phase 3) must show channel independent of label.
