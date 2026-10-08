-- Aegis Voice SOC schema — run in Supabase SQL editor (or supabase db push).
-- Enables RLS on every user table; owner-only access; models/experiments read-only.

-- profiles ---------------------------------------------------------------
create table if not exists profiles (
  id uuid primary key references auth.users(id) on delete cascade,
  display_name text,
  role text default 'analyst',
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

-- analysis_results -------------------------------------------------------
create table if not exists analysis_results (
  id uuid primary key,
  user_id uuid references profiles(id) on delete set null,
  audio_name text,
  prediction text check (prediction in ('REAL','UNCERTAIN','SYNTHETIC')),
  confidence float check (confidence >= 0 and confidence <= 1),
  risk_score int check (risk_score >= 0 and risk_score <= 100),
  risk_level text check (risk_level in ('LOW','MEDIUM','HIGH','CRITICAL')),
  model_id text,
  processing_time_ms float,
  duration_seconds float,
  probs jsonb default '{}'::jsonb,
  gateway jsonb default '{}'::jsonb,
  mock boolean default false,
  created_at timestamptz default now()
);
create index if not exists idx_analysis_user_created on analysis_results (user_id, created_at desc);
create index if not exists idx_analysis_risk on analysis_results (risk_level, created_at desc);

-- calls ----------------------------------------------------------------
create table if not exists calls (
  id uuid primary key,
  user_id uuid references profiles(id) on delete set null,
  caller text,
  receiver text,
  duration_seconds float,
  status text,
  risk_score int,
  detection_result_id uuid references analysis_results(id) on delete set null,
  created_at timestamptz default now()
);

-- threats ---------------------------------------------------------------
create table if not exists threats (
  id uuid primary key,
  user_id uuid references profiles(id) on delete set null,
  analysis_id uuid references analysis_results(id) on delete cascade,
  threat_type text,
  severity text check (severity in ('LOW','MEDIUM','HIGH','CRITICAL')),
  confidence float,
  status text default 'open',
  details jsonb default '{}'::jsonb,
  created_at timestamptz default now()
);
create index if not exists idx_threats_severity on threats (severity, created_at desc);

-- models ----------------------------------------------------------------
create table if not exists models (
  id text primary key,
  name text,
  version text,
  framework text,
  description text,
  status text,
  metadata jsonb default '{}'::jsonb,
  created_at timestamptz default now()
);

-- experiments ------------------------------------------------------------
create table if not exists experiments (
  id uuid primary key default gen_random_uuid(),
  model_id text references models(id) on delete set null,
  name text,
  metrics jsonb default '{}'::jsonb,
  configuration jsonb default '{}'::jsonb,
  notes text,
  created_at timestamptz default now()
);

-- analysis_events ---------------------------------------------------------
create table if not exists analysis_events (
  id uuid primary key default gen_random_uuid(),
  analysis_id uuid references analysis_results(id) on delete cascade,
  event_type text,
  payload jsonb default '{}'::jsonb,
  created_at timestamptz default now()
);

-- RLS --------------------------------------------------------------------
alter table profiles enable row level security;
alter table analysis_results enable row level security;
alter table calls enable row level security;
alter table threats enable row level security;
alter table models enable row level security;
alter table experiments enable row level security;
alter table analysis_events enable row level security;

-- profiles: owner read/write
drop policy if exists "profiles owner" on profiles;
create policy "profiles owner" on profiles for all using (auth.uid() = id) with check (auth.uid() = id);

-- analysis_results: owner read/write
drop policy if exists "analysis owner" on analysis_results;
create policy "analysis owner" on analysis_results for all using (auth.uid() = user_id) with check (auth.uid() = user_id);

-- calls: owner read/write
drop policy if exists "calls owner" on calls;
create policy "calls owner" on calls for all using (auth.uid() = user_id) with check (auth.uid() = user_id);

-- threats: owner read/write
drop policy if exists "threats owner" on threats;
create policy "threats owner" on threats for all using (auth.uid() = user_id) with check (auth.uid() = user_id);

-- analysis_events: owner via parent analysis
drop policy if exists "events via analysis owner" on analysis_events;
create policy "events via analysis owner" on analysis_events for all using (
  exists (select 1 from analysis_results a where a.id = analysis_events.analysis_id and a.user_id = auth.uid())
) with check (
  exists (select 1 from analysis_results a where a.id = analysis_events.analysis_id and a.user_id = auth.uid())
);

-- models / experiments: read-only to authenticated users (service role writes)
drop policy if exists "models readable" on models;
create policy "models readable" on models for select using (auth.role() = 'authenticated');
drop policy if exists "experiments readable" on experiments;
create policy "experiments readable" on experiments for select using (auth.role() = 'authenticated');

-- Seed: one model row reflecting the REAL supplied model (metadata only)
insert into models (id, name, version, framework, description, status, metadata)
values (
  'rafnet-1.0.0-phase19', 'RAFNet', '1.0.0-phase19', 'PyTorch + scikit-learn',
  'WavLM + Handcrafted -> Gateway1 -> Classical/Quantum -> Gateway2 -> Calibration -> DecisionGate. Synthetic-only training; see AI_MODEL_SPEC.md.',
  'experimental',
  '{"sample_rate": 16000, "tau_low": 0.361, "tau_high": 0.861}'::jsonb
) on conflict (id) do nothing;
