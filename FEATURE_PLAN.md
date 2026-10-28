# RAFNet — Feature Plan (Phase 0 Audit, no code changes)

Date: 2026-10-11. Repo root `C:\Users\rajee\rafnet\rafnet`.
Backend entry: `src/rafnet/api/main.py` → app `rafnet.api.main:app` (run with cwd=`src/`).
Frontend: `frontend/` (Vite+React+TS, `vite --port 3000`). `start-local.ps1` starts both.

## 1. `/api/analyze` response fields (exact, `src/rafnet/soc/schemas.py:10-32`, built `src/rafnet/soc/service.py:88-111`)

`AnalyzeResponse`: `id, prediction (REAL|UNCERTAIN|SYNTHETIC), confidence (0-1, model confidence NOT risk), risk_score (0-100), risk_level (LOW|MEDIUM|HIGH|CRITICAL), model_version (e.g. rafnet-1.0.0-phase19), processing_time_ms, audio_duration_seconds, audio_name, prob_synthetic, prob_real, classical_probability|null (p_c), quantum_probability|null (p_q), fused_probability|null (p_fused/Gateway2 out), calibrated_probability|null (p_calibrated), gateway1_audio_weight|null (alpha), gateway1_network_weight|null (1-alpha), gateway2_classical_weight|null (beta), gateway2_quantum_weight|null (1-beta), mock, created_at, reasons[]`.

Model source: `src/rafnet/soc/model_adapter.py VoiceResult` (`prediction, confidence, prob_synthetic, prob_real, classical_probability?, quantum_probability?, fused_probability?, calibrated_probability?, audio_reliability?, network_reliability?, gateway1_audio_weight?, gateway1_network_weight?, gateway2_classical_weight?, gateway2_quantum_weight?, model_version, processing_time_ms, audio_duration_seconds, mock, debug`). Backed by `src/rafnet/inference/pipeline.py InferenceOutput.to_dict()` (`prediction, decision_label, probability_real, probability_synthetic, confidence, audio_reliability?, network_reliability?, gateway1_audio_weight?, gateway1_network_weight?, classical_probability?, quantum_probability?, gateway2_classical_weight?, gateway2_quantum_weight?, fused_probability?, calibrated_probability?, latency_ms, total_latency_ms, model_version, debug_info`).
Risk source: `src/rafnet/soc/risk_engine.py compute_risk()` → `{risk_score, risk_level, reasons}`.

## 2. Mic-chunk (`POST /api/live/chunk`) request/response (exact, `src/rafnet/soc/routes.py:194-274`, `src/rafnet/soc/live.py`)

Request `LiveChunkRequest`: `session_id? (auto-created if missing), sample_rate (=16000 default, 8000-48000), samples[] (non-empty, ≤96000, finite, clamped [-1,1])`.
Buffering response (`buf < 16000`): `{session_id, status:"buffering", buffered_seconds, window_seconds:1.0}`.
Analyzed response: `{session_id, status:"analyzed", prediction, confidence, risk_score, risk_level, model_version, mock, buffered_seconds, analyses}`.
NOTE: chunk response does NOT include `prob_synthetic/real`, `classical/quantum/fused/calibrated`, or gateway weights (only `/api/analyze` does). `POST /api/live/session` → `{session_id, window_samples:16000}`.

## 3. `/ws/live` event schema (server→client only, `src/rafnet/soc/websocket.py:50-106`, `src/rafnet/soc/events.py`)

Connect: `system_status {event,status:connected,model_loaded,mock_mode,model,auth_required}` → replay ≤20 `bus.recent()` → `connected {event,status:subscribed}`. Heartbeat `ping` every 25s.
File analysis: `analysis_started{audio_name}`, `processing{audio_name,duration_seconds}`, `detection{id,prediction,confidence,risk_level}`, `threat_detected{id,risk_score,risk_level}` (HIGH/CRITICAL only), `analysis_completed{id,prediction}`, `error{stage:validation|inference,audio_name}`.
Live: `analysis_started{session_id,source:live}`, `processing{session_id,source:live}`, `detection{session_id,prediction,confidence,risk_level,source:live}`, `threat_detected{session_id,risk_level}` (HIGH/CRITICAL only).
Client sends nothing. Auth only if Supabase JWT env set (`?token=` or `Authorization: Bearer`, else close 4401).

## 4. Available endpoints (all real, `src/rafnet/api/main.py` + `src/rafnet/soc/routes.py`, legacy JSON-frame under `/api/v1` in `src/rafnet/api/routes.py`)

- `POST /api/analyze` (multipart file) ✅ real model inference.
- `GET /api/analysis/{id}`, `GET /api/analysis/{id}/events`, `GET /api/history?limit&offset` → `{items,total}` ✅ (in-memory ring 500 or Supabase).
- `GET /api/threats`, `GET /api/threats/{id}`, `PATCH /api/threats/{id}` ✅ derived from HIGH/CRITICAL analyses.
- `GET /api/calls`, `GET /api/calls/{id}` ✅ derived 1:1 from analyses.
- `GET /api/experiments` ✅ verbatim `experiments/*.json` (`phase19_report.json` + `rafnet_ablation_smoke_*/results.json`, ≤20).
- `GET /api/models` ✅ real (`ModelInfo{name:RAFNet,version,framework,architecture,sample_rate,checkpoints,thresholds:{tau_low:0.361,tau_high:0.861},limitations,mock_mode}`).
- `GET /api/dashboard` ✅ real aggregates (`total_analyses,synthetic/authentic/uncertain_detections,high_risk_events,recent[10],model_loaded,model_version,mock_mode`).
- `GET /api/health` ✅ real (`{status,model_loaded,mock_mode,model}`); `GET /healthz`, `GET /` infra.
- `POST /api/live/session`, `POST /api/live/chunk` ✅ real.
- `GET /api/v1/predict|/predict/batch|/health|/model|/version|/latency|/log|/admin/*` (legacy JSON frames, torch-gated).
- No network-telemetry endpoint (Gateway1 network branch unused in SOC; `predict_audio_only` forces fixed-audio). No speaker-embedding endpoint (none installed — see §7).

## 5. Current frontend routes (`frontend/src/App.tsx:22-36`, nav `frontend/src/components/Layout.tsx:5-14`)

`/login (public)`, `/ (Overview/Dashboard)`, `/live (LiveSOC)`, `/analyze`, `/history`, `/threats`, `/sessions`, `/models (Model Info)`, `/settings`, `*→/`. API client `frontend/src/services/api.ts` mirrors all SOC endpoints; `useLiveSocket.ts` (WS), `useMicChunks.ts` (ScriptProcessor 4096, 1s windows), `useBackendHealth.ts` (15s poll). Types `frontend/src/types/index.ts` mirror backend field-for-field.

## 6. Per-UI-section real-data verdict (does real data exist today?)

| Required UI section | Real data? | Source |
|---|---|---|
| Overview aggregates (analyses, threats, verdict mix, latency, model status) | YES | `GET /api/dashboard` (+ `processing_time_ms` per analysis for measured latency) |
| Live Detection mic→backend 1s + WS events | YES (partial) | `POST /api/live/chunk` + `/ws/live`; chunk lacks branch/gateway fields (Phase 4 adds) |
| Analyzer panel Waveform | YES (partial) | mic/file PCM in browser; spectral/pitch views NOT implemented (Phase 3) |
| Spectrogram / Pitch Track tabs | NO | must be computed in browser (Phase 3); backend not needed |
| Risk gauge + confidence separate | YES | `risk_score/risk_level` vs `confidence` already separate |
| Detection Stack 4 layers w/ evidence | NO | only neural branch exists; heuristic layers + verdict must be added (Phase 4) |
| Verdict {HUMAN_LIVE, AI_SYNTHETIC, REPLAY_PRERECORDED, EXTERNAL_APP_SUSPECTED, UNCERTAIN} + reasons/limitations | NO | only `prediction REAL/UNCERTAIN/SYNTHETIC`; Phase 4 adds |
| Input device selector + virtual-input flag | NO | `enumerateDevices` labels in browser + new backend label-flagging (Phase 4/5) |
| Challenge flow (random phrase → re-analyze) | NO | random phrase is trivially real; re-analysis uses existing endpoint (Phase 5) |
| Suspicious-pattern timeline | NO | derivable from new layers evidence (Phase 5) |
| Model Intelligence | YES | `GET /api/models`; eval report does not exist yet (Phase 6) |
| Fusion & Gateways weights | YES | `classical/quantum/fused/calibrated + gateway1/2 weights` on `/api/analyze` (chunk lacks them — Phase 4 adds) |
| Research (experiments) | YES | `GET /api/experiments` (verbatim JSON; empty-state if no files) |
| History / Threats / Call Sessions tables | YES | `GET /api/history|/threats|/calls` (verdict column added Phase 4/5) |
| Network Monitor | NO | backend has no network-telemetry store; show empty state or drop (Phase 2) |
| Documentation page | YES | repo docs exist (`README, LOCAL_RUN, ARCHITECTURE, API, AI_MODEL_SPEC, RISK_ENGINE, DATABASE, DEPLOYMENT`); render them |
| Audit-log table | YES (real events) | `/api/history` + `/ws/live` event stream; no fake rows |
| Architecture modal | YES (real content) | `ModelInfo.architecture`, thresholds, checkpoints — NOT demo latency/AASIST claims |
| Auth | YES (as-is) | DEV MODE login unless Supabase keys set; keep, label clearly |

## 7. Speaker-embedding / deps check

Installed in `.venv`: `numpy 2.5.3, scipy 1.18.1, torch 2.14.1+cpu, torchaudio, transformers 5.19, scikit-learn 1.9.1, librosa 1.0, pennylane, xgboost`. No speaker-embedding package (`ecapa/x-vector/resemblyzer/speechbrain/pyannote` — 0 hits in `src/`, `pyproject.toml`, `requirements-*.txt`). Layer 4 ships as `kind:"unavailable"` until an open-source model is approved/installed.

## 8. Phase scope (agreed, additive-only)

- Phase 1: rebrand only (no Python package/env renames unless all refs updated).
- Phase 2: theme+shell+shared components+pages with real data only; Network Monitor = empty state or dropped. STOP + checkpoint.
- Phase 3: frontend DSP (waveform/spectrogram/pitch, canvas+rAF, DSP unit tests).
- Phase 4: `src/rafnet/soc/forensics.py` (numpy/scipy) + additive `layers[]`+`verdict` on analyze+chunk, pydantic+TS types, enrollment endpoints (unavailable until model), rolling-window hysteresis for live, pytest DSP tests; 350 existing tests stay green.
- Phase 5: wire 9 capabilities; Detection Stack cards; verdict in tables; challenge flow; no fake mute/2FA controls.
- Phase 6: `tools/evaluate_layers.py` + `reports/layer_eval.{md,json}`; `LAYERS.md`; README/LOCAL_RUN updates.
- Phase 7: pytest + frontend tests + `tsc` + `npm run build`; `start-local.ps1` proof (health, analyze w/ layers+verdict, WS chunk events); Playwright if installable else manual checks; brand-string search clean; final real/heuristic/unavailable report.

## 9. Global rules (binding)

No fake/random data; missing → "Not available". Heuristic outputs labeled `heuristic`, never as validated detectors. Confidence vs risk separate. No API field renames; new output additive. New deps checked first, graceful degrade. Auth unchanged (DEV MODE label).
