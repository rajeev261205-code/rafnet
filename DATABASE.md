# DATABASE.md — Supabase schema (planned; current backend uses in-memory store)

> Status: schema designed, NOT yet provisioned. `SOCService._store` is an
> in-memory ring buffer with the same record shape; swap it for a Supabase
> repository without changing routes or the frontend contract.

## Tables

```sql
create table profiles (
  id uuid primary key references auth.users(id),
  display_name text, role text default 'analyst',
  created_at timestamptz default now(), updated_at timestamptz default now());

create table analysis_results (
  id uuid primary key, user_id uuid references profiles(id),
  audio_name text, prediction text, confidence float,
  risk_score int, risk_level text, model_id text,
  processing_time_ms float, duration_seconds float,
  probs jsonb, gateway jsonb, mock boolean default false,
  created_at timestamptz default now());

create table calls (
  id uuid primary key, user_id uuid references profiles(id),
  caller text, receiver text, duration_seconds float, status text,
  risk_score int, detection_result_id uuid references analysis_results(id),
  created_at timestamptz default now());

create table threats (
  id uuid primary key, user_id uuid references profiles(id),
  analysis_id uuid references analysis_results(id),
  threat_type text, severity text, confidence float, status text default 'open',
  details jsonb, created_at timestamptz default now());

create table models (
  id text primary key, name text, version text, framework text,
  description text, status text, metadata jsonb, created_at timestamptz default now());

create table experiments (
  id uuid primary key, model_id text references models(id),
  name text, metrics jsonb, configuration jsonb, notes text,
  created_at timestamptz default now());

create table analysis_events (
  id uuid primary key default gen_random_uuid(), analysis_id uuid references analysis_results(id),
  event_type text, payload jsonb, created_at timestamptz default now());
```

Seed `models` with one row from `GET /api/models` (real checkpoint sizes);
seed `experiments` from `experiments/phase19_report.json` + ablation `results.json`
(verbatim numbers only).

## RLS (to enforce on provisioning)

- Enable RLS on all user tables; policies: owner can SELECT/INSERT/UPDATE own
  `user_id = auth.uid()` rows; no anonymous writes; `models`/`experiments`
  read-only to authenticated users.
- The backend uses the service-role key (bypasses RLS), so it **also** scopes
  every query in SQL: `user_id=eq.<uid>` for authenticated callers and
  `user_id=is.null` for anonymous sessions. No endpoint fetches all rows and
  post-filters in Python.
- Never disable RLS for dev convenience; never expose `SERVICE_ROLE_KEY` to frontend.

## Env separation

Frontend (Vite, public): `VITE_SUPABASE_URL`, `VITE_SUPABASE_ANON_KEY`.
Backend (server-only): `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`,
`SUPABASE_ANON_KEY` (token validation).
