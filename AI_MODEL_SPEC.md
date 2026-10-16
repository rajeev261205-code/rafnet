# AI_MODEL_SPEC.md — RAFNet (actual supplied model, 2026-10-08)

Source of truth: `src/rafnet/` + `experiments/checkpoints/` + `experiments/*.json/.md`.
`Not available` = not found, NOT invented.

## Identity

RAFNet — Adaptive Audio-Network Fusion for Synthetic Voice Detection in Real-Time.
Version 0.1.0 (pyproject) / pipeline model_version 1.0.0. Framework: PyTorch +
scikit-learn (+optional XGBoost/PennyLane) + HF Transformers (WavLM). Hybrid fusion
classifier: torch encoders/gates + sklearn classical + PCA+VQC/MLP quantum +
calibration + threshold gate. Checkpoints in `experiments/checkpoints/`.

## Architecture (`model.py:34-206`)

audio[B,T] -> AudioEncoder (frozen WavLM-base-plus 768 mean-pool stride 320 +
handcrafted -> 256) = z_a; network[B,D] -> NetworkEncoder (MaskedNorm + BiGRU +
CodecEmbed + MLP = 256) = z_n; Gateway1 MLP(512->128->1) alpha -> z_f;
ClassicalBranch (default logistic) -> p_c; QuantumBranch PCA(256->8) -> VQC 8q/3L
or ClassicalFallback MLP -> p_q; Gateway2 MLP(3->32->1) beta -> p_fused;
Calibrator (temperature/platt/isotonic) -> probs; DecisionGate (tau_low/tau_high)
-> decisions. Output `RAFNetOutput`: probs, decisions, p_c, p_q, p_fused, alpha,
beta, r_a, r_n, z_fused, gate infos, decision info.

## Checkpoints (measured)

classical_logistic.pkl 2465B TRAINED (train_acc 1.0, val/test NOT_MEASURED);
quantum_branch.pkl 16634B PARTIALLY_TRAINED (PCA 4231 samples + fallback MLP 10ep,
VQC NOT trained); gateway2.pth 11019B TRAINED 5ep loss 4.14->3.65;
calibrator.pkl 1977B (ECE NOT_MEASURED); decision_gate.pkl 1638B (0.361/0.861);
embeddings.pt 312505B. Random init (UNTRAINED): audio/handcrafted projections,
NetworkEncoder, Gateway1. WavLM 94M FROZEN from HF Hub (needs download).

## Input

Audio 16kHz mono float32 [B,T], window 1.0s (16000), hop 0.5s, mask ::320, RMS -26dB
clip [-1,1], torchaudio resample. Network input_dim 32 (20 tests), hidden [128,128],
out 256, window 10, 8 codecs+unknown, max_packets 100; 20-dim vector
(packet_loss[0,1], jitter, rtt, bitrate, IAT/size stats, burst, codec one-hot).

## Preprocessing

int16/32768 -> mono -> resample -> RMS norm; WavLM pooling -> Linear->LN->GELU->DO->Linear
256; handcrafted LFCC60+MFCC40 (STFT 512/160/400+mel+dct), CQT84->mean/std/max,
Spectral x4, Prosodic pyin+RMS x8, VoiceQuality HNR/jitter/shimmer x3 -> Linear 128;
concat -> Linear 256 + LN + GELU.

## Output/labels

Manifest 0=real/1=synthetic. Decisions 0=REAL (p<tau_low) / 1=UNCERTAIN / 2=SYNTHETIC
(p>tau_high); defaults 0.3/0.7, optimized 0.361/0.861. p_c/p_q/p_fused/p_calibrated =
P(synthetic); prob_real = 1-p; confidence = max(p,1-p). Diagnostics alpha/beta/r_a/r_n.

## Inference

preprocess 5ms -> audio_enc 30ms GPU/600ms CPU -> net 5ms -> gw1 5ms ->
classical 10ms (0.5 if unfitted) + quantum 15ms -> gw2 3ms -> calib 2ms ->
gate 1ms. Total 76ms GPU / 700ms CPU target; Phase19 CPU measured 1182ms.
Engine: threaded queue + frame-drop. Missing branches -> 0.5/UNCERTAIN.

## Metrics (measured only)

Phase19 n=20: 20 UNCERTAIN (p~0.7); latencies pre 0.2/audio 1180/net 7.6/gw1 4.8/
class 0.1/quant 0.0/gw2 4.0/cal 0.0/gate 11.7/total 1182.1ms. Baselines n=200 synthetic:
1.0 acc (trivial). Gateway2 worse_than_fixed, beta 0.482±0.344, disag corr ~0.
Data 5000 (3478/753/769), 100 speakers disjoint, 21 generators overlapped.

## Limitations

COMPLETED_WITH_CAVEATS / SYSTEM_FUNCTIONAL_BUT_SCIENTIFICALLY_INVALID: untrained
stack, frozen WavLM, fallback quantum, synthetic-only + leakage, tiny eval, CPU over
budget, fusion unvalidated. No accuracy/F1/AUC claims beyond above.

## Runtime verification (2026-10-08, Windows CPU box)

- torch 2.14.1+cpu + torchaudio verified working; `gateway2.pth` loads as an
  OrderedDict with `gate_network` MLP 3→32→32→1 (+ gate stats), matching the
  Gateway2 architecture above.
- Full smoke `scripts/test_model.py` PASSES with sklearn 1.6.1: 440 Hz tone →
  UNCERTAIN @ 0.8216, risk 60/MEDIUM, 1049.8 ms (consistent with Phase19's
  ~1182 ms CPU measurement), mock=False, all 5 checkpoints restored.
- Latency profile `scripts/profile_latency.py` (Windows CPU, synthetic tones,
  3 reps each): 0.5 s → 290–396 ms; 1.0 s → 379–409 ms; 2.0 s → 557–577 ms;
  5.0 s → ~1300 ms; median 483.1 ms. Audio encoder (WavLM CPU) is >99% of every
  run; all other stages < 10 ms combined. Predictions deterministic per duration
  (0.760 / 0.822 / 0.832 / 0.840, all UNCERTAIN). 700 ms CPU target holds up to
  ~2 s clips; 5 s clips exceed it — chunk long uploads for interactive use.
- Environment notes (honest, device-specific): sklearn 1.9.1's DLLs are blocked
  by this machine's Application Control policy, so the smoke runs under sklearn
  1.6.1 with two documented adapter shims — `multi_class='ovr'` on the
  1.9.1-pickled binary LogisticRegression, and fitted-PCA bookkeeping re-derived
  from shipped components/mean. Production MUST use sklearn==1.9.1
  (`requirements-ml.txt`); the shims warn at runtime and change no model weights.
- WavLM `microsoft/wavlm-base-plus` downloads from HF Hub on first boot (248 shards).

## SOC integration

`InferencePipeline.predict_audio_only` (Gateway1 fixed_audio + zero network) for file
upload; `predict_with_network` with telemetry. Load: config -> pipeline -> restore
checkpoints -> decode/mono (resample in AudioPreprocessor) -> forward -> InferenceOutput.
MOCK_INFERENCE=true selects flagged mock adapter (dev only).
