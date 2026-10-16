# LOCAL_RUN.md — run RAFNet on Windows (PowerShell)

## First-time setup (once)

Requirements: Python 3.12 (`py -3.12 --version`), Node 18+ (`node --version`),
~4 GB free disk for torch + WavLM weights, internet (HF Hub download).

```powershell
cd C:\Users\rajee\RAFNet\RAFNet

# 1. Backend env (Python 3.12 only — NOT 3.14; ML wheels lack 3.14 builds)
py -3.12 -m venv .venv
.\.venv\Scripts\pip.exe install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\pip.exe install --no-cache-dir -r requirements-ml.txt
.\.venv\Scripts\pip.exe install --no-cache-dir -e . pytest httpx

# 2. Backend config (placeholders in .env.example; local .env is gitignored)
Copy-Item .env.example .env   # then edit ports/keys if needed; keep MOCK_INFERENCE=false

# 3. Frontend deps + config
cd frontend
npm install
# .env.local already defaults to http://localhost:8000 + ws://localhost:8000/ws/live
cd ..
```

> Machine quirk (verified 2026-10-11): this PC's Application Control (WDAC)
> policy blocks `numpy >= 2.5.4` Windows DLLs (`_umath_linalg` fails with
> "An Application Control policy has blocked this file", CodeIntegrity event
> 3033). `requirements-ml.txt` therefore pins `numpy==2.5.3`. Do NOT upgrade
> numpy until the policy changes. `scikit-learn==1.9.1` (training pin) loads
> fine here.

## Run (every time)

```powershell
cd C:\Users\rajee\RAFNet\RAFNet
.\start-local.ps1
```

This opens two windows (backend :8000, frontend :3000), waits for
`GET /api/health`, and prints the URLs. First backend boot downloads
`microsoft/wavlm-base-plus` from Hugging Face (~400 MB, cached afterwards).

- Frontend: http://127.0.0.1:3000 — log in (DEV MODE unless Supabase keys set),
  then Overview / Live Detection / Analyze / Fusion & Gateways / Research /
  History / Threats / Call Sessions / Model Intelligence / Network Monitor /
  Settings / Documentation.
- Backend docs: http://127.0.0.1:8000/docs
- Health gate: `GET /api/health` must show
  `{"status":"ok","model_loaded":true,"mock_mode":false,"model":"rafnet"}`.

## Useful checks

```powershell
# backend health (real model, no mock)
Invoke-RestMethod http://127.0.0.1:8000/api/health
# backend tests (from repo root, .venv active or via .\.venv\Scripts\python.exe)
.\.venv\Scripts\python.exe -m pytest tests/ -q
# frontend unit tests + production build
cd frontend; npm test; npm run build
```

## Evaluate the detection layers

```powershell
# 1. Collect labeled audio (see samples/README.md):
#    samples/human_live, samples/human_replayed, samples/tts, samples/voice_changed
# 2. Run the evaluator (uses the real model when it loads, layers-only otherwise):
.\.venv\Scripts\python.exe tools/evaluate_layers.py
# 3. Read reports/layer_eval.md (.json for machines) or GET /api/evaluation;
#    the Model Intelligence page shows the measured numbers, else "Not evaluated".
```

## Optional: real Supabase auth + persistence

1. Create a Supabase project, run `supabase/migrations/001_soc_schema.sql`.
2. Backend `.env`: set `SUPABASE_URL=https://<ref>.supabase.co`,
   `SUPABASE_SERVICE_ROLE_KEY`, `SUPABASE_ANON_KEY`, `SOC_REQUIRE_AUTH=true`.
3. Frontend `frontend/.env.local`: set `VITE_SUPABASE_URL`,
   `VITE_SUPABASE_ANON_KEY` (anon key only — never service_role), restart
   `npm run dev`. Without these, the UI uses a labeled DEV MODE login and the
   backend keeps analyses in memory (same record shape).
