# LOCAL_RUN_AUDIT.md — Windows local-run audit (2026-10-11, Phase 1, read-only)

Repo root: `C:\Users\rajee\RAFNet\RAFNet` (same dir as `C:\Users\rajee\rafnet\rafnet`;
Windows path is case-insensitive). Git branch: `fix-ws-live`, status clean.
Machine: Windows + PowerShell, 16 GB RAM (~4 GB free), 51 GB free disk.
Runtimes: Python 3.12.0 (`C:\Program Files\Python312\python.exe`, only version
on `py -0p`); Node v24.21.0, npm 11.19.0.

## 1. Backend layout (exists)

- App entry: `src/rafnet/api/main.py` → `rafnet.api.main:app`. Mounts legacy
  `/api/v1/*` only when torch imports; always mounts SOC `GET/POST /api/*`
  (`src/rafnet/soc/routes.py`) and `WS /ws/live` (`src/rafnet/soc/websocket.py`).
- SOC routes (all under `/api`): `POST /analyze`, `GET /analysis/{id}`,
  `GET /analysis/{id}/events`, `GET /history`, `GET /threats`,
  `PATCH /threats/{id}`, `GET /threats/{id}`, `GET /calls`, `GET /calls/{id}`,
  `GET /experiments`, `POST /live/session`, `POST /live/chunk`,
  `GET /dashboard`, `GET /models`, `GET /health`. Plus `/healthz` and `/`
  on the app root. Legacy: `POST /api/v1/predict`, `/predict/batch`, health/
  model/version/latency/log, admin fit endpoints (torch only).
- Auth: `src/rafnet/soc/auth.py` — optional Supabase JWT. Open by default
  (`SOC_REQUIRE_AUTH=false`); enforced 401/503 when true. WS accepts
  `?token=` or `Authorization: Bearer`.
- WS `/ws/live`: accepts, sends `system_status` (real model state) on connect,
  replays ≤20 bus events, sends `connected`, heartbeat `ping` every 25 s,
  then streams real bus events (`analysis_started/processing/detection/
  threat_detected/analysis_completed/error`). No fake timers.
- Upload contract: WAV/FLAC/OGG via soundfile, MP3/M4A via torchaudio
  fallback; ≤25 MB; allowed suffixes wav/flac/ogg/oga/mp3/m4a/mp4/webm.
- `.gitignore` covers `.venv/`, `venv/`, `.env` (not committed), `*.pt`,
  `*.bin`, `*.onnx`. It does NOT ignore `node_modules/` (fix in Phase 3).

## 2. Checkpoints (exist, real files, committed)

`experiments/checkpoints/` (env `MODEL_CHECKPOINT_DIR`, default
`experiments/checkpoints`; device via `RAFNET_DEVICE=cpu`):
- `classical_logistic.pkl` 2465 B — pickle (`rafnet.classical.branch`), real.
- `quantum_branch.pkl` 16634 B — pickle (`rafnet.quantum.branch`), real.
- `gateway2.pth` 11019 B — torch zip archive (`PK..`, gateway2/data.pkl), real.
- `calibrator.pkl` 1977 B — pickle (`rafnet.calibration.calibrator`), real.
- `decision_gate.pkl` 1638 B — pickle (`rafnet.decision.gate`), real.
All 5 tracked in git. Not LFS stubs. WavLM weights NOT vendored: first boot
downloads `microsoft/wavlm-base-plus` from HF Hub (~94 M params, ~400 MB).

## 3. Environment variables needed

Backend (all read via `os.getenv` with defaults; no central schema file):
`SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, `SUPABASE_ANON_KEY`,
`SOC_REQUIRE_AUTH` (=false), `MODEL_CHECKPOINT_DIR` (=experiments/checkpoints),
`RAFNET_DEVICE` (=cpu), `MOCK_INFERENCE` (=false),
`RISK_HIGH_CONFIDENCE/MEDIUM_CONFIDENCE/CRITICAL_SCORE/HIGH_SCORE/MEDIUM_SCORE`,
`HOST`, `PORT` (=8000), `CORS_ORIGINS` (=`http://localhost:3000,
http://localhost:8000`), `RATE_LIMIT_ANALYZE_PER_MIN`,
`RATE_LIMIT_CHUNK_PER_MIN`, `MAX_REQUEST_BYTES`, `EXPERIMENTS_DIR`,
plus legacy `RAFNET_CLASSICAL_MODEL/QUANTUM_FALLBACK`, `LATENCY_*`,
`AUDIO_WINDOW_SECONDS`, `MAX_QUEUE_SIZE`, `RELOAD`, `LOG_LEVEL`.
Frontend (planned Vite): `VITE_API_URL` (default `http://localhost:8000`),
`VITE_WS_URL` (default `ws://localhost:8000/ws/live`),
`VITE_SUPABASE_URL`, `VITE_SUPABASE_ANON_KEY` (never service_role).
State on disk: `.env` EXISTS (all backend vars set incl. Supabase keys —
values not reproduced here); `.env.example` MISSING; `frontend/.env.local`
and `frontend/.env.example` MISSING.

## 4. What is broken / missing

1. `venv/` is unusable on Windows: Linux-style `bin/` with symlink stubs
   (task notes Python 3.14). Only fix: fresh `.venv` on Python 3.12.
2. System Python 3.12 has NO ML stack: torch/transformers/librosa/sklearn/
   scipy/pandas missing (has fastapi 0.142.4, uvicorn 0.54.0, websockets 17.2,
   pydantic, soundfile 0.14.0, numpy 2.5.3, pytest 9.1.1, httpx 0.28.1).
   So today the app boots SOC-only with `MOCK_INFERENCE=true` only; with
   `MOCK_INFERENCE=false` the real model fails to load (no torch).
3. `SUPABASE_URL` in `.env` is a bare project ref, not an `https://` URL —
   Supabase REST/Auth will fail until fixed or left unconfigured (in-memory
   fallback). `NEXT_PUBLIC_SUPABASE_*` are empty.
4. `tests/conftest.py:6` hardcodes `sys.path.insert(0, '/home/sk/rafnet/src')`
   — nonexistent Linux path on this machine; breaks test collection env.
5. `frontend/` is a 6-file draft, unrunnable: NO `package.json`, no
   `vite.config.ts`/`tsconfig`/`index.html`/`main.tsx`, no Tailwind/router,
   no `node_modules`. And its code disagrees with the backend:
   - `services/api.ts` uses CRA `process.env.REACT_APP_API_URL` (not Vite),
     calls `GET /health` (backend has `/api/health`), expects
     `GET /api/history` → array (backend returns `{items,total}`).
   - `types/index.ts` lacks `risk_level`, probabilities, gateway weights,
     `mock`, `reasons` from the real `AnalyzeResponse` contract.
   - `LiveSOC.tsx` hardcodes `ws://localhost:8000/ws/live`, no reconnect,
     renders `e.timestamp/e.status` which backend events don't send.
   - `App.tsx` has 4 routes only; no Login/History/Threats/Sessions/Models/
     Settings pages.
6. CORS default allows `http://localhost:3000` + `http://localhost:8000` but
   NOT `http://localhost:5173` (Vite default) — required addition for Phase 2.
7. `requirements-ml.txt` pins `scikit-learn==1.9.1`; FINAL_STATUS notes its
   DLLs were blocked by this machine's Application Control policy before
   (adapter shims cover older sklearn). Must re-test on install.
8. Docs referenced a deleted separate frontend scaffold with `NEXT_PUBLIC_*`
   vars — stale; the current UI is `frontend/` (Vite).

## 5. Exact pinned versions

`requirements-soc.txt`: fastapi≥0.110, uvicorn≥0.29, websockets≥12.0,
pydantic≥2.5, python-multipart≥0.0.9, numpy≥1.24, soundfile≥0.12.
`requirements-ml.txt`: torch+torchaudio (CPU index first), then
transformers≥4.30, librosa≥0.10, scikit-learn==1.9.1, scipy≥1.10,
numpy≥1.24, soundfile≥0.12, pyyaml≥6, pandas≥2.0, omegaconf≥2.3, tqdm≥4.65,
matplotlib≥3.7, fastapi/uvicorn/websockets/pydantic/python-multipart (same).
`pyproject.toml`: same ML set + xgboost≥2.0, pennylane(+lightning)≥0.33
(both optional at runtime: xgboost only for xgboost classical model,
pennylane only for non-fallback VQC). Dockerfile base: python:3.11-slim.
Installed now (system): fastapi 0.142.4, uvicorn 0.54.0, websockets 17.2,
numpy 2.5.3, pytest 9.1.1, httpx 0.28.1, soundfile 0.14.0.
