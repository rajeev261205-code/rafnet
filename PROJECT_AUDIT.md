# PROJECT_AUDIT.md — Aegis Voice SOC / RAFNet (full audit 2026-10-08)

Sources: `D:\RAJEEV\DOCUMENTS\SYMPOSIUM & HACKTHON\SAIRAM\rafnet` (authoritative,
mirrored at `C:\Users\rajee\rafnet\rafnet`), Stitch `stitch_aegis_voice_soc_dashboard`
(15 screens), `C:\Users\rajee\aegisvoice-frontend` (Next.js 16 scaffold).
`rafnet.zip` (~560MB) = rafnet tree + venv (venv excluded).

## 1. Architecture

RAFNet = Adaptive Audio-Network Fusion for Synthetic Voice Detection in Real-Time
(`pyproject.toml`, `src/rafnet/model.py:34 class RAFNet(nn.Module)`).

```
audio[B,T] -> AudioEncoder -> z_a[256]
network[B,D] -> NetworkEncoder -> z_n[256]
z_a,z_n -> Gateway1(alpha) -> z_f[256]
z_f -> ClassicalBranch -> p_c ; z_f -> QuantumBranch (PCA->VQC/head) -> p_q
p_c,p_q -> Gateway2(beta) -> p_fused -> Calibrator -> probs -> DecisionGate -> 0=REAL/1=UNCERTAIN/2=SYNTHETIC
```

`RAFNetOutput`: probs, decisions, p_c, p_q, p_fused, alpha, beta, r_a, r_n, z_fused,
gate1_info, gate2_info, decision_info. Inference paths: `inference/pipeline.py`
(batch, 787 lines) and `inference/engine.py` (threaded queue + frame-drop, 1101 lines).

Existing API: FastAPI `RAFNet API 1.0.0` (`api/main.py`), prefix `/api/v1` (`api/routes.py`):
POST /predict, POST /predict/batch, GET /health, /model, /version, /latency, /log,
POST /admin/fit-classical, fit-quantum, fit-calibration, optimize-thresholds,
GET /healthz, /, /docs. No auth, CORS `*` (TODO).

NEW (this session): `src/rafnet/soc/*` -> `/api/analyze` (multipart file upload),
`/api/analysis/{id}`, `/api/dashboard`, `/api/models`, `/api/health` with real
model adapter + risk engine separation. See ARCHITECTURE.md, API.md.

## 2. Frontend

Stitch: 15 static HTML screens (SOC dashboard, live console x3, forensic x3, call
sessions x3, threat history x3, telemetry/forensics) — visual reference only.
Next.js scaffold (`aegisvoice-frontend`, Next 16.4/React 19/Tailwind 4): routes
overview, live, analyze, sessions, history, models, network, fusion, research, settings.
`services/api.ts` was MOCK-ONLY (fake 2s analyze, setInterval realtime) with
`data/mock.ts` demo metrics (98.4% etc, `isDemo:true`) — NOT model outputs.
This session: api.ts rewritten env-driven (`NEXT_PUBLIC_BACKEND_URL`) with typed
`AnalyzeResult`/`SocHealth`, real `POST /api/analyze`, mock fallback explicitly
flagged; Analyze page functional (upload -> real backend -> verdict/confidence/risk).

## 3. ML model (facts only)

Checkpoints `experiments/checkpoints/`: classical_logistic.pkl 2465B (TRAINED),
quantum_branch.pkl 16634B (PCA fitted + fallback MLP, PARTIALLY_TRAINED),
gateway2.pth 11019B (TRAINED 5 epochs), calibrator.pkl 1977B, decision_gate.pkl
1638B (tau 0.361/0.861), embeddings.pt 312505B. WavLM microsoft/wavlm-base-plus 94M
frozen from HF Hub (download required, not vendored). No ONNX/safetensors/TF.
Framework: torch + sklearn (+optional xgboost/pennylane).

Inputs: audio 16kHz mono float32, window 1.0s/hop 0.5s, RMS -26dB; network dim 32
(20 in tests), 8 codecs, window 10. Labels 0=real/1=synthetic; decisions
0=REAL/1=UNCERTAIN/2=SYNTHETIC. Preprocessing: int16 scale, mono, resample, RMS norm,
WavLM pooling + Linear 256, handcrafted LFCC60/MFCC40/CQT84/Spectral/Prosodic/VoiceQuality
-> 128, concat -> 256. Latency Phase19 CPU: total 1182ms (target 700), audio 1180ms.

## 4. Dependencies

torch, torchaudio, transformers, numpy, scipy, scikit-learn, xgboost, pennylane,
pennylane-lightning, librosa, pyyaml, omegaconf, tqdm, matplotlib, pandas (>= versions
in pyproject.toml). Python >=3.10. Frontend: next, react, recharts, framer-motion,
lucide-react, tailwind.

## 5. Metrics (verbatim, NOT production claims)

Phase19 (n=20): 0 REAL/20 UNCERTAIN/0 SYNTHETIC; classical train_acc 1.0, val/test
NOT_MEASURED; Gateway2 loss 4.14->3.65; status COMPLETED_WITH_CAVEATS /
SYSTEM_FUNCTIONAL_BUT_SCIENTIFICALLY_INVALID. Synthetic baselines n=200: 1.0 acc
(trivial fixture). Gateway2 learned worse_than_fixed, beta~0.48 uncorrelated.
Data: 5000 synthetic (3478/753/769), 100 speakers disjoint, 21 generators OVERLAPPED,
10 conditions.

## 6. Problems / gaps

Untrained encoders/gates, frozen WavLM, fallback quantum, synthetic-only data with
generator leakage, tiny eval, CPU over budget, fusion invalidated; API had no file
upload/auth/Supabase/WS/risk separation; frontend mock-only; configs/ empty; no
.env.example/Supabase RLS/Vercel config. Supabase/Auth/WS/persistence remain planned
(DATABASE.md documents schema+RLS; SOCService uses same-shape in-memory store).

## 7. Plan status

Done: audit docs, soc adapter (model_adapter/audio_io/risk_engine/schemas/service/routes),
SOC router mounted in api/main.py, scripts/test_model.py, tests/test_soc.py (5 passed),
frontend api.ts + Analyze + .env.example, frontend build passes, docs
(RISK_ENGINE/ARCHITECTURE/API/DATABASE/DEPLOYMENT). Remaining: provision Supabase +
RLS, Auth pages, WS live, calls/threats/history backed by DB, GPU/heavy-runtime model
smoke (torch missing on this Windows box — script fails honestly, checkpoints verified
present), Vercel + backend deploy.
