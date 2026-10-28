-- RAFNet migration 003 — upload provenance on analysis_results.
-- Layer 3 treats lossy uploads (mp3/aac/m4a/ogg) differently from lossless
-- audio and live-mic PCM: codec-cutoff evidence is excluded for lossy input.
-- These columns preserve that provenance per row. Run after 002.
alter table analysis_results add column if not exists input_codec text default null;
alter table analysis_results add column if not exists input_lossy boolean default false;
