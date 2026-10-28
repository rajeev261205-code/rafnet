# DEPLOYMENT.md

## Frontend (Vercel)

1. Root: `frontend`. Build: `npm install && npm run build` (Vite).
2. Env (Vercel dashboard) — **Vite exposes only `VITE_*` at build time**:
   `VITE_API_URL` (deployed FastAPI URL, no trailing slash),
   `VITE_WS_URL` (`wss://<api-host>/ws/live`),
   `VITE_SUPABASE_URL`, `VITE_SUPABASE_ANON_KEY`.
3. SPA routing: the Vite build emits `frontend/dist`; add a rewrite to
   `index.html` for client-side routes.
4. Rebuild after changing any `VITE_*` value — they are compiled into the bundle.
   Never put a service-role key in the frontend; only the anon key is public.

## Backend/ML (persistent Python host — Railway / Render / Fly / VM)

SOC-only (no torch, mock/dev or external model): `pip install -r requirements-soc.txt`,
then from `src/`: `uvicorn rafnet.api.main:app --host 0.0.0.0 --port $PORT`.
Full model image: `Dockerfile` (torch CPU + `requirements-ml.txt` + baked WavLM)
or `pip install -e .` from `pyproject.toml` + `fastapi uvicorn soundfile`.

Required production env:

- `RAFNET_ENV=production`
- `SOC_REQUIRE_AUTH=true`
- `SUPABASE_URL`, `SUPABASE_ANON_KEY` (token validation) and
  `SUPABASE_SERVICE_ROLE_KEY` (server-side persistence)
- `SOC_CORS_ORIGINS=https://<your-frontend-host>` (exact origin; never `*`)
- `MODEL_CHECKPOINT_DIR`, `RAFNET_DEVICE=cpu`, `MOCK_INFERENCE=false`,
  `RISK_*` thresholds, `RATE_LIMIT_*`, `PORT`

Production startup fails closed (refuses to boot) when auth is disabled,
Supabase is missing, or CORS contains `*`. The Dockerfile sets
`RAFNET_ENV=production` and `SOC_REQUIRE_AUTH=true` by default; override both
only for a throwaway local demo.

First boot downloads `microsoft/wavlm-base-plus` (~94M params) from HF Hub —
allow egress + disk; expect 700–1200ms CPU inference (Phase19 measured 1182ms).
Health gate: authenticated `GET /api/health` must report
`model_loaded:true, mock_mode:false`; `GET /healthz` is the anonymous
load-balancer probe and returns `{status}` only.

## Render (verified live 2026-10-08)

Render auto-detected the `Dockerfile` and deployed SOC-only on the first pass
(image built, `requirements-soc.txt` installed, Uvicorn on `$PORT`, service live).
To switch that service to full inference:

1. **Instance RAM first**: the full image needs **≥ 1 GB** (Starter minimum,
   2 GB+ recommended). torch + WavLM peak well above the 512 MB free tier and
   will OOM there. Upgrade the service before redeploying.
2. Push the updated `Dockerfile` (full-ML image, WavLM baked in) + fixed
   `requirements-ml.txt`; Render rebuilds on push.
3. Environment tab — set the production env list above. In particular
   `SOC_CORS_ORIGINS=https://<your-vercel-frontend-url>` (without this,
   browsers block every frontend call) and the Supabase keys.
4. Watch the deploy logs for `RAFNet API started` and no unsafe-config startup
   error, then `GET /api/health` must show `model_loaded:true, mock_mode:false`.
5. Free-tier note: instances sleep when idle — first request after sleep pays a
   cold start (model already in image, so no re-download, just load time).

## Why not Vercel serverless for the model

torch + WavLM + librosa + long CPU inference + checkpoint loading do not fit
Vercel function limits. Keep model on a persistent host; the frontend calls it
via `VITE_API_URL`.

## Supabase

Create a project, run all `supabase/migrations/*.sql` in order (`001` owner
columns/RLS, `002` verdict columns, `003` input columns, `004` Auth-user profile
bootstrap, `005` cascade user data on account deletion), keep RLS enabled, and
enable email Auth. The backend uses the
service-role key and therefore scopes
every analysis query by owner in SQL (`user_id=eq.<uid>` or `user_id=is.null`
for anonymous sessions); RLS remains defense-in-depth for any direct client.
Seed real reports: `python supabase/seed.py --apply` (dry-run without `--apply`).

## Verification after deploy

1. `GET /healthz` is public; authenticated `GET /api/health` → real model status.
2. Anonymous `/api/*` request with `SOC_REQUIRE_AUTH=true` → 401.
3. User A's history must never contain user B's analyses; opening B's analysis
   id as A → 404.
4. WS `/ws/live?token=<A token>` replays/streams only A's events.
5. `OPTIONS` preflight from the configured frontend origin succeeds; responses
   include `X-Content-Type-Options: nosniff` and (behind HTTPS) HSTS.
6. WebSocket auth uses `?token=` (browsers cannot set upgrade headers). Configure
   the proxy/CDN access log to redact query strings so access tokens are not
   retained in logs.
