# README.md — Aegis Voice SOC (RAFNet)

AI-powered Security Operations Center for synthetic/AI-generated voice detection:
real RAFNet inference (WavLM + handcrafted → Gateway1 → classical/quantum →
Gateway2 → calibration → decision gate) behind a FastAPI backend and a Next.js SOC UI.

## Roots

- Backend + ML: this repo (`src/rafnet/*`). Legacy JSON-frame API `src/rafnet/api/*`
  (`/api/v1/*`); SOC adapter `src/rafnet/soc/*` (`/api/analyze`, `/api/dashboard`,
  `/api/models`, `/api/history`, `/api/threats`, `/api/calls`, `/api/experiments`,
  `/api/live/*`, `WS /ws/live`). Docs: PROJECT_AUDIT.md, AI_MODEL_SPEC.md,
  ARCHITECTURE.md, API.md, DATABASE.md, RISK_ENGINE.md, DEPLOYMENT.md, FINAL_STATUS.md.
- Frontend: `C:\Users\rajee\aegisvoice-frontend` (Next.js 16). Service
  `src/services/api.ts` calls `NEXT_PUBLIC_BACKEND_URL` with flagged mock fallback.
- Designs: Stitch `stitch_aegis_voice_soc_dashboard/*.html` (reference only).

## Quickstart — backend (SOC-only, no torch)

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows (source .venv/bin/activate on Linux)
pip install -r requirements-soc.txt
cp .env.example .env              # set MODEL_CHECKPOINT_DIR, MOCK_INFERENCE=false
cd src
uvicorn rafnet.api.main:app --reload --port 8000
curl localhost:8000/api/health   # model_loaded:true only if real model loaded
```

Full ML runtime (torch CPU, production needs scikit-learn==1.9.1):
`requirements-ml.txt`. First boot downloads `microsoft/wavlm-base-plus` from HF Hub.

## Quickstart — frontend

```bash
cd C:\Users\rajee\aegisvoice-frontend
cp .env.example .env.local   # NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
npm install
npm run dev                  # or npm run build / npm test
```

Open Analyze → upload WAV → real `POST /api/analyze` → verdict + model
confidence + engine risk + latency. Mock badge shows when backend unreachable and
`NEXT_PUBLIC_MOCK_FALLBACK=true`.

## Verify

```bash
# backend SOC tests (no torch needed) + lint
PYTHONPATH=src python -m pytest tests/test_soc.py -v   # 16 passed
python -m ruff check src/rafnet/soc/ tests/test_soc.py scripts/ supabase/seed.py
# real-model smoke + latency profile (torch venv) + live-server E2E
PYTHONPATH=src MOCK_INFERENCE=false python scripts/test_model.py [--wav sample.wav]
PYTHONPATH=src python scripts/profile_latency.py
BASE_URL=http://localhost:8000 python scripts/e2e_probe.py
# frontend
npm test        # vitest, 5 passed
npm run lint    # eslint, 0 errors
npm run build   # 20 routes
```

## Detection contract

`AnalyzeResponse`: prediction (REAL|UNCERTAIN|SYNTHETIC, model), confidence (model,
NOT risk), risk_score/risk_level/reasons (engine), model_version, processing_time_ms,
audio_duration_seconds, probabilities + gateway weights, mock flag, created_at.

## Honest limitations (see AI_MODEL_SPEC.md / FINAL_STATUS.md)

Synthetic-only training with generator leakage, untrained encoders/gates, frozen WavLM,
fallback (non-quantum) branch, Phase19 n=20 all-UNCERTAIN, CPU latency over the 700 ms
target on long clips. Supabase/Auth/WS persistence planned (DATABASE.md,
supabase/README.md); current store is in-memory with the same shape.
`MOCK_INFERENCE=true` / mock UI badges are dev-only, never production.
