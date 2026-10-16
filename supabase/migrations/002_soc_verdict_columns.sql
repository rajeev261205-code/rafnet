-- RAFNet migration 002 — verdict/layers/reasons on analysis_results.
-- 001 only stored scalar + probs/gateway columns; the app-level record also
-- carries the fused verdict, detection layers, and risk reasons. Without these
-- columns production inserts either 400 (unknown columns) or silently drop
-- the data. Run after 001 in the Supabase SQL editor.
alter table analysis_results add column if not exists verdict jsonb default null;
alter table analysis_results add column if not exists layers jsonb default '[]'::jsonb;
alter table analysis_results add column if not exists reasons jsonb default '[]'::jsonb;
