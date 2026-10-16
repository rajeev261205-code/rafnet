# RECONCILE.md — three implementations compared (read-only survey, 2026-10-08)

No files from other workstreams were modified for this comparison.

## Option A — Python SOC adapter (this workstream: `src/rafnet/soc/*` + Next.js UI)

- Status: complete, verified live. 18 pytest + full suite 343/3/0, vitest 5/5,
  build 20 routes, ruff clean, eslint 0 errors, real-model smoke PASS,
  real-model HTTP E2E 16/16, latency profiled.
- Model: REAL RAFNet — all 5 shipped checkpoints restored through each branch's
  actual loader; proven predictions (UNCERTAIN @ 0.82–0.83 on tones, matching
  Phase19's known behavior).
- API: 15 SOC routes + WS `/ws/live` + chunk ingestion + triage PATCH + optional
  JWT (owner scoping, WS gate) + rate limits + body guard + versioned docs.
- Persistence: store port (in-memory default, Supabase REST when configured),
  migration + RLS + seed script ready.
- Gaps: needs Supabase project, sklearn 1.9.1 prod runtime, deploy, GPU pass.

## Option B — Node/Express backend (`rafnet-frontend/backend/`, parallel workstream)

- Status: under active development (own package.json, tsconfig, vitest +
  supertest, SQLite via `node:sqlite`, needs Node ≥ 22).
- Model: DSP-heuristic detector (spectral/prosody/artifact proxies), honestly
  flagged mock — NOT the supplied RAFNet model.
- API: same route surface plus `/api/analyses*` aliases, DELETE, report extras;
  deliberately keeps `/api/history` + `/api/analysis/:id` compat with Option A,
  so the Next.js frontend works against either.
- Persistence: real (SQLite) — ahead of Option A's in-memory default, behind a
  provisioned Supabase for multi-user RLS.
- Gaps: no real model; Node ≥ 22 requirement; auth/RLS story unverified by me.

## Option C — `backend/app/*` + `frontend/src/*` scaffold (same repo, idle since 09:49)

- Status: early sketch, ~10 KB total, non-functional. Read-only findings:
  - `main.py` hardcodes `"model_loaded": True` in `/health` (unverifiable claim).
  - `np.frombuffer(upload_bytes, float32)` decodes WAV headers as audio (garbage
    input for real files); no validation, no size limits, CORS `*`.
  - `model_adapter.load()` restores NO shipped checkpoints; `predict` feeds a
    fresh `torch.randn` network tensor per call (nondeterministic) with unfitted
    branches (always 0.5/UNCERTAIN by construction).
  - Bare `except → 500 detail=str(e)` leaks internals; `user_id="system_user"`
    hardcoded (no auth/RLS); frontend is CRA-style stubs (`any` types, 3 endpoints).
- Recommendation: adopt NOTHING from it — Option A fulfills the same layout
  intent with working, tested code. Merging would be a rewrite for zero
  functional gain. Leave the files untouched (another session owns them).

## Recommendation

- Canonical backend for the real-model product: **Option A**.
- Keep **Option B** as the lightweight DSP alternative ONLY if operating without
  the torch stack matters more than real-model fidelity — do not present its
  heuristic verdicts as RAFNet output.
- **Option C**: archive or delete after confirming with its owner; do not build on it.
