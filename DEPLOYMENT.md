# DEPLOYMENT.md

## Frontend (Vercel)

1. Root: `frontend`. Build: `npm install && npm run build`.
2. Env (Vercel dashboard): `NEXT_PUBLIC_BACKEND_URL` (deployed FastAPI),
   `NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_ANON_KEY`.
3. SPA routing handled by Next.js; no `vercel.json` needed unless rewrites required.
4. No localhost defaults in production; assets under `public/`.

## Backend/ML (persistent Python host — Railway / Render / Fly / VM)

SOC-only (no torch, boots + serves mock/dev): `pip install -r requirements-soc.txt`,
then from `src/`: `uvicorn rafnet.api.main:app --host 0.0.0.0 --port $PORT`.
Full model image: `Dockerfile` (torch CPU + `requirements-ml.txt` + baked WavLM)
or `pip install -e .` from `pyproject.toml` + `fastapi uvicorn soundfile`.
Env: `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, (`SUPABASE_ANON_KEY` for JWT
validation), `SOC_REQUIRE_AUTH`, `MODEL_CHECKPOINT_DIR`, `RAFNET_DEVICE=cpu`,
`MOCK_INFERENCE=false`, `RISK_*` thresholds, `CORS_ORIGINS`,
`RATE_LIMIT_ANALYZE_PER_MIN`, `PORT`.
First boot downloads `microsoft/wavlm-base-plus` (~94M params) from HF Hub —
allow egress + disk; expect 700-1200ms CPU inference (Phase19 measured 1182ms).
Health gate: `GET /api/health` must report `model_loaded:true, mock_mode:false`;
`GET /healthz` mirrors it for load balancers.

## Render (verified live 2026-10-08)

Render auto-detected the `Dockerfile` and deployed SOC-only on the first pass
(image built, `requirements-soc.txt` installed, Uvicorn on `$PORT`, service live).
To switch that service to full inference:

1. **Instance RAM first**: the full image needs **≥ 1 GB** (Starter minimum,
   2 GB+ recommended). torch + WavLM peak well above the 512 MB free tier and
   will OOM there. Upgrade the service before redeploying.
2. Push the updated `Dockerfile` (full-ML image, WavLM baked in) + fixed
   `requirements-ml.txt`; Render rebuilds on push.
3. Environment tab — set: `MOCK_INFERENCE=false`, `MODEL_CHECKPOINT_DIR=
   experiments/checkpoints`, `RAFNET_DEVICE=cpu`, and **`CORS_ORIGINS=
   https://<your-vercel-frontend-url>`** (without this, browsers block every
   frontend call). Add `SUPABASE_*` + `SOC_REQUIRE_AUTH=true` once provisioned.
4. Watch the deploy logs for `RAFNet API started (legacy=on)` and NO
   `No module named 'torch'` warning, then `GET /api/health` must show
   `model_loaded:true, mock_mode:false`.
5. Free-tier note: instances sleep when idle — first request after sleep pays a
   cold start (model already in image, so no re-download, just load time).

## Why not Vercel serverless for the model

torch + WavLM + librosa + long CPU inference + checkpoint loading do not fit
Vercel function limits. Keep model on a persistent host; frontend calls it via
`NEXT_PUBLIC_BACKEND_URL`.

## Supabase

Create project, run `supabase/migrations/001_soc_schema.sql`, enable RLS, wire Auth
(email), store keys per `.env.example` separation. Seed real reports:
`python supabase/seed.py --apply` (dry-run without `--apply`).
