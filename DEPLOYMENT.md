# DEPLOYMENT.md

## Frontend (Vercel)

1. Root: `aegisvoice-frontend`. Build: `npm install && npm run build`.
2. Env (Vercel dashboard): `NEXT_PUBLIC_BACKEND_URL` (deployed FastAPI),
   `NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_ANON_KEY`.
3. SPA routing handled by Next.js; no `vercel.json` needed unless rewrites required.
4. No localhost defaults in production; assets under `public/`.

## Backend/ML (persistent Python host — Railway / Render / Fly / VM)

SOC-only (no torch, boots + serves mock/dev): `pip install -r requirements-soc.txt`,
then from `src/`: `uvicorn rafnet.api.main:app --host 0.0.0.0 --port $PORT`.
Full model image: use `Dockerfile` (uncomment the torch block) or
`pip install -e .` from `pyproject.toml` + `fastapi uvicorn soundfile`.
Env: `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, (`SUPABASE_ANON_KEY` for JWT
validation), `SOC_REQUIRE_AUTH`, `MODEL_CHECKPOINT_DIR`, `RAFNET_DEVICE=cpu`,
`MOCK_INFERENCE=false`, `RISK_*` thresholds, `CORS_ORIGINS`,
`RATE_LIMIT_ANALYZE_PER_MIN`, `PORT`.
First boot downloads `microsoft/wavlm-base-plus` (~94M params) from HF Hub —
allow egress + disk; expect 700-1200ms CPU inference (Phase19 measured 1182ms).
Health gate: `GET /api/health` must report `model_loaded:true, mock_mode:false`;
`GET /healthz` mirrors it for load balancers.

## Why not Vercel serverless for the model

torch + WavLM + librosa + long CPU inference + checkpoint loading do not fit
Vercel function limits. Keep model on a persistent host; frontend calls it via
`NEXT_PUBLIC_BACKEND_URL`.

## Supabase

Create project, run `supabase/migrations/001_soc_schema.sql`, enable RLS, wire Auth
(email), store keys per `.env.example` separation. Seed real reports:
`python supabase/seed.py --apply` (dry-run without `--apply`).
