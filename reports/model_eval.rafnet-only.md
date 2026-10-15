# Model evaluation report (RAFNet voice)

Generated: 2026-10-15T16:08:12.072411+00:00
Model: {"loaded": true, "name": "rafnet", "mock": false}
Thresholds: tau_low=0.3611111111111111 tau_high=0.861111111111111 (configs/decision_thresholds.json)
Files: 8 scored / 8 found (real=4, fake=4)

> Small local sample sets are NOT a general benchmark. Every n is shown; treat all numbers as a snapshot of THESE files only.

## clean — full clip (POST /api/analyze path)
n=8, strict accuracy (UNCERTAIN=wrong)=0.0, coverage=0.0, selective=None, AUC=0.5, EER={"eer": 0.5, "threshold": 0.8238, "far": 0.5, "frr": 0.5}
binary@0.5: {"accuracy": 0.5, "precision": 0.5, "recall": 1.0, "f1": 0.6667, "tp": 4, "tn": 0, "fp": 4, "fn": 0, "n": 8}

Confusion (rows=true, cols=predicted):
| true \ pred | REAL | UNCERTAIN | SYNTHETIC |
| --- | --- | --- | --- |
| real | 0 | 4 | 0 |
| fake | 0 | 4 | 0 |

## mic_sim — laptop-mic simulation (noise + band-limit + MP3)
n=8, strict accuracy (UNCERTAIN=wrong)=0.125, coverage=0.25, selective=0.5, AUC=0.625, EER={"eer": 0.5, "threshold": 0.8209, "far": 0.5, "frr": 0.5}
binary@0.5: {"accuracy": 0.5, "precision": 0.5, "recall": 1.0, "f1": 0.6667, "tp": 4, "tn": 0, "fp": 4, "fn": 0, "n": 8}

Confusion (rows=true, cols=predicted):
| true \ pred | REAL | UNCERTAIN | SYNTHETIC |
| --- | --- | --- | --- |
| real | 0 | 3 | 1 |
| fake | 0 | 3 | 1 |

## Score distributions (P(synthetic); higher = more synthetic-like)
- clean/real: {"n": 4, "mean": 0.8215, "std": 0.0195, "min": 0.7917, "p25": 0.8114, "median": 0.8252, "p75": 0.8352, "max": 0.8439}
```
0.00-0.10 |                                            0
0.10-0.20 |                                            0
0.20-0.30 |                                            0
0.30-0.40 |                                            0
0.40-0.50 |                                            0  <-- UNCERTAIN band
0.50-0.60 |                                            0  <-- UNCERTAIN band
0.60-0.70 |                                            0  <-- UNCERTAIN band
0.70-0.80 | #############                              1  <-- UNCERTAIN band
0.80-0.90 | ########################################   3  <-- UNCERTAIN band
0.90-1.00 |                                            0
```
- clean/fake: {"n": 4, "mean": 0.8228, "std": 0.0168, "min": 0.8077, "p25": 0.8094, "median": 0.8168, "p75": 0.8303, "max": 0.8498}
```
0.00-0.10 |                                            0
0.10-0.20 |                                            0
0.20-0.30 |                                            0
0.30-0.40 |                                            0
0.40-0.50 |                                            0  <-- UNCERTAIN band
0.50-0.60 |                                            0  <-- UNCERTAIN band
0.60-0.70 |                                            0  <-- UNCERTAIN band
0.70-0.80 |                                            0  <-- UNCERTAIN band
0.80-0.90 | ########################################   4  <-- UNCERTAIN band
0.90-1.00 |                                            0
```

UNCERTAIN-band share (fraction of scores inside [0.3611111111111111, 0.861111111111111]):
- clean/real: 1.0
- mic-sim/real: 0.75
- clean/fake: 1.0
- mic-sim/fake: 0.75

Band verdict: UNCERTAIN band [0.3611111111111111, 0.861111111111111] (width 0.5) holds >50% of a class's scores on THESE files — too wide for this data. Recalibrate with --calibrate once n >= 20/class.

Mic-sim chain: room noise SNR 20.0 dB + 50 Hz hum + brickwall 80 Hz-6.8 kHz + MP3 64k round-trip

Accuracy drop under mic-sim (strict): -0.125 (positive = model degrades on laptop-mic conditions).

## Threshold calibration (held-out split)
Tradeoff: maximize strict accuracy with UNCERTAIN counted as incorrect; ties broken by narrower band, then higher coverage
Split: {"calibration_n": 4, "held_out_n": 4, "seed": 42, "method": "stratified per-class shuffle"}
Before (active thresholds, held-out half): {"thresholds": {"tau_low": 0.3611111111111111, "tau_high": 0.861111111111111}, "strict_accuracy": 0.0, "coverage": 0.0, "selective_accuracy": null, "n_certain": 0, "n_total": 4}
After (calibrated thresholds, held-out half): {"thresholds": {"tau_low": 0.45, "tau_high": 0.55}, "strict_accuracy": 0.5, "coverage": 1.0, "selective_accuracy": 0.5, "n_certain": 4, "n_total": 4}
Chosen: tau_low=0.45 tau_high=0.55 (saved=None)
WARNING: n=8 is far below the >= 40 clips needed for stable thresholds — these taus fit THIS sample, not the world. Not a benchmark.

## Live aggregation (single full-clip vs median-of-windows)
Window: 5 chunks, hysteresis majority to switch (3/5), same as live route (same as POST /api/live/chunk).
Single-shot strict accuracy: 0.0 (n=8)
Aggregated strict accuracy: 0.0 (mean windows/file=4.38)
Delta (agg - single): 0.0

## Honest reading
- n=8 (real=4, fake=4): a small local sample, NOT a general benchmark. Add >= 20 clips per class (real speech + TTS + voice-conversion + replayed, disjoint speakers AND generators) before trusting any threshold.
- The model is WEAK on these conditions (stated plainly, numbers untouched).
- Likely causes if weak: synthetic-only training with no real-human speech; 21 generators overlapped across splits (leakage); full-clip embeddings vs 1 s training windows (biggest preprocessing caveat, see ACCURACY_NOTES.md); unvalidated 0.50-wide UNCERTAIN band; laptop-mic noise/band-limit/MP3 shift.
- What would fix it: labeled 16 kHz clips (>= 20/class, disjoint speakers and generators); re-measure with this script; recalibrate thresholds (--calibrate); with explicit approval, refit classical + PCA + gateway2 on windowed embeddings and validate full-clip vs windowed aggregation before changing inference.

## Files
- {"file": "C:\\Users\\rajee\\RAFNet\\samples\\real\\260-123286-0012.flac", "duration_seconds": 2.43, "sample_rate_hz": 16000, "input_lossy": false, "p_synthetic": 0.8323, "prediction": "UNCERTAIN", "confidence": 0.8323, "classical_p": 0.9785071611404419, "quantum_p": 0.3103671371936798, "fused_p": 0.9191942811012268, "mic_sim_method": "room noise SNR 20.0 dB + 50 Hz hum + brickwall 80 Hz-6.8 kHz + MP3 64k round-trip", "p_synthetic_mic": 0.7537, "prediction_mic": "UNCERTAIN", "window_ps": [0.8386, 0.793], "n_windows": 2, "p_synthetic_agg": 0.8158, "label": "real", "y": 0}
- {"file": "C:\\Users\\rajee\\RAFNet\\samples\\real\\4446-2273-0021.flac", "duration_seconds": 5.25, "sample_rate_hz": 16000, "input_lossy": false, "p_synthetic": 0.8439, "prediction": "UNCERTAIN", "confidence": 0.8439, "classical_p": 0.9533776640892029, "quantum_p": 0.6775205135345459, "fused_p": 0.928292453289032, "mic_sim_method": "room noise SNR 20.0 dB + 50 Hz hum + brickwall 80 Hz-6.8 kHz + MP3 64k round-trip", "p_synthetic_mic": 0.9123, "prediction_mic": "SYNTHETIC", "window_ps": [0.8476, 0.8937, 0.9202, 0.815, 0.7991], "n_windows": 5, "p_synthetic_agg": 0.8476, "label": "real", "y": 0}
- {"file": "C:\\Users\\rajee\\RAFNet\\samples\\real\\5142-33396-0059.flac", "duration_seconds": 5.47, "sample_rate_hz": 16000, "input_lossy": false, "p_synthetic": 0.818, "prediction": "UNCERTAIN", "confidence": 0.818, "classical_p": 0.988523542881012, "quantum_p": 0.12583155930042267, "fused_p": 0.9072379469871521, "mic_sim_method": "room noise SNR 20.0 dB + 50 Hz hum + brickwall 80 Hz-6.8 kHz + MP3 64k round-trip", "p_synthetic_mic": 0.8209, "prediction_mic": "UNCERTAIN", "window_ps": [0.9355, 0.8896, 0.8276, 0.7238, 0.8758], "n_windows": 5, "p_synthetic_agg": 0.8758, "label": "real", "y": 0}
- {"file": "C:\\Users\\rajee\\RAFNet\\samples\\real\\6930-81414-0025.flac", "duration_seconds": 2.53, "sample_rate_hz": 16000, "input_lossy": false, "p_synthetic": 0.7917, "prediction": "UNCERTAIN", "confidence": 0.7917, "classical_p": 0.9509821534156799, "quantum_p": 0.20609448850154877, "fused_p": 0.8834913969039917, "mic_sim_method": "room noise SNR 20.0 dB + 50 Hz hum + brickwall 80 Hz-6.8 kHz + MP3 64k round-trip", "p_synthetic_mic": 0.7713, "prediction_mic": "UNCERTAIN", "window_ps": [0.8358, 0.7912, 0.8607], "n_windows": 3, "p_synthetic_agg": 0.8358, "label": "real", "y": 0}
- {"file": "C:\\Users\\rajee\\RAFNet\\samples\\fake\\sapi_david_2.wav", "duration_seconds": 5.15, "sample_rate_hz": 22050, "input_lossy": false, "p_synthetic": 0.8238, "prediction": "UNCERTAIN", "confidence": 0.8238, "classical_p": 0.9846312403678894, "quantum_p": 0.1931469887495041, "fused_p": 0.9121689796447754, "mic_sim_method": "room noise SNR 20.0 dB + 50 Hz hum + brickwall 80 Hz-6.8 kHz + MP3 64k round-trip", "p_synthetic_mic": 0.8274, "prediction_mic": "UNCERTAIN", "window_ps": [0.9166, 0.8438, 0.8512, 0.9184, 0.7845], "n_windows": 5, "p_synthetic_agg": 0.8512, "label": "fake", "y": 1}
- {"file": "C:\\Users\\rajee\\RAFNet\\samples\\fake\\sapi_david_4.wav", "duration_seconds": 4.54, "sample_rate_hz": 22050, "input_lossy": false, "p_synthetic": 0.8099, "prediction": "UNCERTAIN", "confidence": 0.8099, "classical_p": 0.9812623858451843, "quantum_p": 0.12170545756816864, "fused_p": 0.9002370238304138, "mic_sim_method": "room noise SNR 20.0 dB + 50 Hz hum + brickwall 80 Hz-6.8 kHz + MP3 64k round-trip", "p_synthetic_mic": 0.8097, "prediction_mic": "UNCERTAIN", "window_ps": [0.8263, 0.8334, 0.8041, 0.7576, 0.7987], "n_windows": 5, "p_synthetic_agg": 0.8041, "label": "fake", "y": 1}
- {"file": "C:\\Users\\rajee\\RAFNet\\samples\\fake\\sapi_zira_2.wav", "duration_seconds": 5.04, "sample_rate_hz": 22050, "input_lossy": false, "p_synthetic": 0.8498, "prediction": "UNCERTAIN", "confidence": 0.8498, "classical_p": 0.9907416105270386, "quantum_p": 0.33604055643081665, "fused_p": 0.9327318072319031, "mic_sim_method": "room noise SNR 20.0 dB + 50 Hz hum + brickwall 80 Hz-6.8 kHz + MP3 64k round-trip", "p_synthetic_mic": 0.8859, "prediction_mic": "SYNTHETIC", "window_ps": [0.9174, 0.8075, 0.8446, 0.9254, 0.8318], "n_windows": 5, "p_synthetic_agg": 0.8446, "label": "fake", "y": 1}
- {"file": "C:\\Users\\rajee\\RAFNet\\samples\\fake\\sapi_zira_4.wav", "duration_seconds": 4.56, "sample_rate_hz": 22050, "input_lossy": false, "p_synthetic": 0.8077, "prediction": "UNCERTAIN", "confidence": 0.8077, "classical_p": 0.9837805032730103, "quantum_p": 0.09075384587049484, "fused_p": 0.898223340511322, "mic_sim_method": "room noise SNR 20.0 dB + 50 Hz hum + brickwall 80 Hz-6.8 kHz + MP3 64k round-trip", "p_synthetic_mic": 0.8069, "prediction_mic": "UNCERTAIN", "window_ps": [0.8781, 0.8082, 0.8158, 0.7917, 0.7987], "n_windows": 5, "p_synthetic_agg": 0.8082, "label": "fake", "y": 1}
