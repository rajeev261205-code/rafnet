# Supabase runbook — provision once, then backend + frontend just work

## 1. Create the project

1. https://supabase.com/dashboard → New project (note the DB password).
2. Project Settings → API: copy `Project URL` and `anon public` key; copy the
   `service_role` key (server-only — never put it in frontend env).

## 2. Run the schema (creates tables + RLS + model seed)

SQL Editor → New query → paste `supabase/migrations/001_soc_schema.sql` → Run,
then `supabase/migrations/002_soc_verdict_columns.sql` → Run,
then `supabase/migrations/003_soc_input_columns.sql` → Run.
Verify: Table Editor shows `profiles, analysis_results, calls, threats, models,
experiments, analysis_events`; `models` contains `rafnet-1.0.0-phase19`;
`analysis_results` has `verdict, layers, reasons` columns.

## 3. Seed real experiment reports

```bash
SUPABASE_URL=https://xyz.supabase.co \
SUPABASE_SERVICE_ROLE_KEY=eyJ... \
python supabase/seed.py --apply
```

Without `--apply` it dry-runs (prints 1 model + N experiments, writes nothing).

## 4. Enable Auth (email)

Authentication → Providers → Email → enable. (Confirm-email optional for dev.)

## 5. Wire the apps

Backend `.env` (server-only):
```
SUPABASE_URL=https://xyz.supabase.co
SUPABASE_SERVICE_ROLE_KEY=eyJ...
SUPABASE_ANON_KEY=<anon key>   # only needed for JWT validation
SOC_REQUIRE_AUTH=true          # enforce Bearer JWT in production
```

Frontend `frontend/.env.local` (public only):
```
VITE_API_URL=https://<backend-host>
VITE_WS_URL=wss://<backend-host>/ws/live
VITE_SUPABASE_URL=https://xyz.supabase.co
VITE_SUPABASE_ANON_KEY=<anon key>
```
(Only `VITE_*` vars reach the Vite app — never put a service_role key here.)

Without Supabase env, both sides degrade honestly: backend uses the in-memory
store, frontend offers the labeled dev-mode login.

## 6. Verify

```bash
SUPABASE_URL=... SUPABASE_SERVICE_ROLE_KEY=... SUPABASE_ANON_KEY=... \
python supabase/verify.py
```

Checks REST reachability, all 7 tables, the model seed row, and that RLS hides
`analysis_results` from the anon key. Exit 0 = ready for backend wiring.

- `GET /api/health` → `model_loaded:true, mock_mode:false`
- Register → login → Analyze → History shows the row (RLS: users see only
  their own analyses; anonymous local rows predate RLS).
