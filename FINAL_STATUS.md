# FINAL_STATUS.md — RAFNet (2026-10-08, Phase 25)

## Completed — Phase 25 (this session): Render full-inference image

- User deployed SOC-only to Render live (`rafnet.onrender.com`, verified from
  their logs: image built, service live, `running SOC-only`, `/` 200).
- Rewrote `Dockerfile` as the full-ML production image: torch CPU first (own
  index), `requirements-ml.txt`, WavLM baked in at build time (no re-download
  on ephemeral-disk restarts), `$PORT` handling kept. RAM warning embedded:
  ≥ 1 GB minimum, 2 GB+ recommended — 512 MB free tier will OOM.
- Fixed `requirements-ml.txt`: removed invalid inline `--index-url` syntax
  (torch now installed first by Docker), kept sklearn==1.9.1 training pin,
  added boot-proven deps. `pip install --dry-run` resolves cleanly.
- `DEPLOYMENT.md` gained a Render section from the live log: upgrade RAM,
  push to rebuild, required env (incl. `CORS_ORIGINS` = Vercel URL),
  log lines to expect, health gate, free-tier sleep caveat.
- Verification: requirements dry-run resolves; no product code changed
  (Docker/requirements/docs only), all greens stand.

## Decision recorded — backend posture: C (keep both)

User selected **C**: Python/RAFNet for real inference, Node/Express as
lightweight fallback. No code changes required — already true: the Next.js
frontend speaks the shared contract to either backend (verified in Phase 12
recon), `backend/` is excluded from the Next tsconfig (own build config), and
`RECONCILE.md` documents each option's role. Python remains the canonical
real-model path; Node verdicts must never be presented as RAFNet output.

## Completed — Phase 23 (this session): seed/verify scripts proven, no credentials needed

- Generalized the fake PostgREST to all 7 tables + RLS emulation (non-service
  keys see zero analysis rows, mirroring the owner-only policies).
- New tests: `seed.collect()` asserts real checkpoint sizes (2465/11019/16634)
  and phase19 presence; `seed.rest()` writes through the fake; `verify.main()`
  returns 0 on a seeded fake (incl. the RLS check) and 1 with reported failures
  when the seed row is missing.
- No docker on this box, so the `Dockerfile` stays unvalidated (standard
  python:3.11-slim + pip install shape, low risk, flagged honestly).
- Verification: pytest 23/23, ruff clean.

## Completed — Phase 22 (this session): Supabase adapter proven (and fixed)

- Built a stdlib-only fake PostgREST (`_FakePostgrest` in `test_soc.py`) and
  tested `SupabaseAnalysisStore` save/get/list/order/pagination/count plus
  apikey-header presence and `get_store()` fallback (no env → memory; bad URL →
  memory; fake URL → Supabase adapter).
- The test caught a REAL bug: `save_analysis` never set `method="POST"`, so
  writes went out as GET-with-body — PostgREST would have ignored them while
  the adapter silently echoed the record back. Had this shipped unwitnessed,
  Supabase wiring would have persisted nothing. Fixed + proven.
- Verification: pytest 20/20, ruff clean.

## Completed — Phase 21 (this session): Supabase handoff tooling

- New `supabase/verify.py` (stdlib-only): checks REST reachability, all 7 tables,
  the model seed row, and RLS enforcement (anon key must see zero analysis rows).
  Fails loudly with the exact fix (run the migration SQL). Self-tested
  (clean usage error without env) + ruff clean; referenced from `supabase/README.md`.
- Verification: no product code changed; all greens stand.

## Completed — Phase 20 (this session): reconcile analysis (read-only)

- Surveyed the `backend/app/*` + `frontend/src/*` scaffold without touching it:
  early non-functional sketch (~10 KB) with spec violations (hardcoded
  `model_loaded: True`, WAV-bytes-as-float32 decode, `torch.randn` network input
  per call, zero checkpoint restores, CORS `*`, hardcoded user, stub frontend).
  Full comparison of all three options + recommendation written to `RECONCILE.md`:
  Option A (this workstream) canonical for the real-model product, Option B
  (Node) as lightweight heuristic alternative, Option C not to be built on.
- Verification: no product code changed; all greens stand.

## Completed — Phase 19 (this session): final full verification

- Re-ran everything against the current tree: full venv suite **343 passed,
  3 skipped, 0 failed** · SOC 18/18 (system) · ruff clean · vitest 5/5 ·
  test ports free. Frontend code unchanged since its green 20-route build.
- Verification: all gates green, no code changes needed.

## Completed — Phase 18 (this session): mypy on new code

- Ran mypy (repo config) over `soc/`: fixed every actionable finding in new code
  (interface bodies to `raise NotImplementedError`, `json.loads` Any-return,
  ClassicalBranch/QuantumBranch variable reuse, `_rest`/`tolist` Any-returns,
  sklearn stub ignore). Remaining mypy output is pre-existing supplied-code
  issues + environmental stub gaps — mypy was never gated here (supplied code
  alone reports ~180 errors), so no repo-wide gate was introduced.
- No `.pre-commit-config.yaml` exists in the repo (pre-commit is only a dev
  dependency name); not added — CI wiring is a maintainer decision.
- Verification: mypy clean on `soc/` with full deps, pytest 18/18, ruff clean.

## Completed — Phase 17 (this session): sample search + workstream survey (no code changes)

- Searched both project trees (C: working copy, D: supplied archive, excluding
  venv/node_modules) for any real audio: **zero .wav/.flac/.mp3/.ogg/.m4a files
  exist anywhere**. The dataset is generated synthetically in code
  (`create_synthetic_manifest`, randn fallbacks). Tone-based verification is
  therefore the honest ceiling until real samples are provided —
  `scripts/test_model.py --wav path/to/sample.wav` is ready for that day.
- Surveyed the other scaffolding in this repo: `backend/app/*` +
  `frontend/src/*` under `C:\Users\rajee\rafnet\rafnet` date from 09:49–09:53
  (the earlier concurrent session, idle since) — untouched, as is the Node
   backend in `rafnet-frontend/backend/`. Three structures currently coexist;
  the reconcile decision is still yours.
- Verification: nothing changed, so all greens stand (full suite 343/3/0,
  SOC 18/18 × 3 runtimes, `npm test` 5/5, build 20 routes, ruff clean,
  eslint 0 errors).

## Completed — Phase 16 (this session): final regression + hygiene

- Full suite re-run in the torch venv after all Phase 12–15 changes:
  **343 passed, 3 skipped, 0 failed** (was 341 — the 2 new auth tests included).
- Port hygiene: no listeners left on any test port (8471–8475).
- Verification: full suite 343/3/0, pytest SOC 18/18, ruff clean, eslint
  0 errors, `npm test` 5/5, frontend build 20 routes.

## Completed — Phase 15 (this session): auth-scoping gaps

- `GET /api/analysis/{id}` was the one read not scoped by owner: authenticated
  users could fetch each other's analyses by ID. Now filtered (403-equivalent
  404 on cross-owner access; anonymous legacy rows stay visible). Covered by
  `test_analysis_detail_scoped_by_owner`.
- `/ws/live` now enforces `SOC_REQUIRE_AUTH=true` (Bearer header or `?token=`,
  close 4401 otherwise); the frontend `useLiveEvents` hook sends the Supabase
  token automatically. Covered by `test_websocket_rejects_anonymous_when_auth_required`.
- Docs integrity re-verified (all four core docs intact at expected sizes).
- Verification: pytest 18/18, ruff clean, eslint 0 errors, `npm test` 5/5,
  frontend build 20 routes.

## Completed — Phase 14 (this session): operator probe + README rescue

- `scripts/e2e_probe.py` now covers the triage path live: loud-tone upload →
  HIGH/CRITICAL → threat listed → PATCH acknowledged → status persisted.
  Ran against a fresh mock-mode server: **21/21 checks passed**. (Process hygiene:
  verified the test port is free afterwards instead of bulk-killing PIDs.)
- Rewrote `README.md`: it was a stale stub describing a `backend/`+`frontend/`
  layout that doesn't exist, with wrong install/run commands. Now documents the
  real roots, SOC-only + full-ML setups, and every verify command
  (pytest/ruff/smoke/profile/e2e/vitest/eslint/build) with current counts.
- Verification: e2e_probe 21/21 live; pytest/vitest/build unchanged and green
  (16/16, 5/5, 20 routes).

## Completed — Phase 13 (this session): lint gates green

- Backend ruff on all new code (`soc/`, `test_soc.py`, `scripts/`, `seed.py`):
  fixed 59 auto-fixable (import order), 5 manual (unused vars/imports, `__all__`
  exports) → **all checks passed**. (Supplied code has 145 pre-existing ruff
  findings — untouched.)
- Frontend eslint: fixed all 6 errors in my code (router-based redirects instead
  of `window.location` in render, lazy session/state init, memoized waveform
  bars replacing `Math.random()` in render, pruned dead icon imports). **0 errors**;
  the only 2 warnings left are unused-disable directives in the parallel
  workstream's `backend/` files — not mine, untouched.
- No TODO/FIXME/HACK markers in `soc/`. Parallel workstream still active
  (their `backend/tests`, controllers churning, own vitest) — coexistence holds,
  no files of theirs touched.
- Verification: ruff clean, eslint 0 errors, pytest 16/16, `npm test` 5/5,
  frontend build 20 routes.

## Completed — Phase 12 (this session): legacy engine characterized, not condemned

- Read the engine gating code (`engine.py:670-690`): inference fires only with a
  full audio window AND ≥ 5 telemetry samples. Ran the predicted experiment —
  six rapid sequential frames at `/api/v1/predict`: frames 1–4 504 (priming),
  frames 5–6 **200 with real engine inference** (UNCERTAIN, ~486/~396 ms).
- Those engine verdicts were p=0.5/UNCERTAIN because the engine boots unfitted
  (doesn't restore shipped checkpoints; admin `fit-*` is the runtime path) —
  the documented graceful degradation, verified live rather than assumed.
  `API.md` now records the exact priming behavior instead of "legacy broken".
- `requirements-ml.txt` completed with the boot-proven deps (pandas, omegaconf,
  tqdm, matplotlib). Local venvs used for verification live under
   `%LOCALAPPDATA%\Temp\opencode\` (`rafnet-venv` full ML, `soc-venv` SOC-only);
  recreate via `requirements-ml.txt` / `requirements-soc.txt`.
- Verification: no product code changed this phase (docs + requirements only);
  prior greens stand (full suite 341/16, frontend 20 + 5 vitest).

## Completed — Phase 11 (this session): legacy regression sweep

- Full supplied suite in the torch venv: **341 passed, 3 skipped, 0 failed**
  (incl. all 16 SOC tests) — nothing I changed regressed the supplied code.
- Fixed one genuine pre-existing bug found by that sweep:
  `QuantumBranch.save/load` persisted only PCA components/mean, but sklearn ≥ 1.6
  touches `explained_variance_` in `transform`, breaking every save/load
  round-trip (`test_quantum_branch_serialization` failed). `save` now snapshots
  the full fitted-PCA state, `load`/`load_pca` restore it, old checkpoints fall
  back to derived bookkeeping (same approach as the SOC shim, which stays as the
  backward-compat path). Minimal, additive, covered by the previously-failing test.
- Live legacy check: booted the full app and POSTed a real 1 s frame to
  `/api/v1/predict` → **504 "Inference timeout"** (service timeout 5.0 s; the
  threaded engine wants a filled 2×window buffer + sustained stream, so single
  frames can't return in time on CPU). Untouched supplied code — documented in
  `API.md` as a pre-existing limitation; SOC `/api/analyze` + `/api/live/chunk`
  (direct `InferencePipeline`) are the working inference paths.
- Verification: full suite 341 passed / 3 skipped, pytest SOC 16/16,
  frontend build 20 routes, `npm test` 5/5, compileall clean.

## Completed — Phase 10 (this session): limits + consistency fixes

- Real bug fixed: mic chunks shared the 30/min `/api/analyze` budget and would
  429 mid-scan. Chunks now have their own sustained budget
  (`RATE_LIMIT_CHUNK_PER_MIN`, default 120) via `get_chunk_limiter_dep`;
  covered by `test_chunk_limiter_independent_of_analyze`.
- New `MaxBodySizeMiddleware` (`soc/limits.py`, `MAX_REQUEST_BYTES` default
  32 MB): oversized declared bodies get 413 before buffering (the 25 MB audio
  cap still applies after decode). Covered by `test_body_size_guard`.
- Frontend: analysis detail now goes through `apiService.getAnalysis/
  getAnalysisEvents` (auth token attached, single URL logic); triage buttons
  surface errors instead of failing silently; mic hook drops stale audio past
  3 s backlog so slow backends can't grow memory or skew realtime.
- Docs sweep: no stale counts in README/ARCHITECTURE/DEPLOYMENT/API (only the
  intentionally historical Phase-1 paragraph in PROJECT_AUDIT.md); API.md +
  `.env.example` record the chunk budget and body guard.
- Verification: pytest 16/16, `npm test` 5/5, frontend build 20 routes,
  compileall clean.

## Completed — Phase 9 (this session): install-path proof + frontend tests

- Fresh-venv proof: brand-new venv + only `requirements-soc.txt` (+ pytest/httpx
  test deps) → **pytest 14/14 with no torch installed**. The Dockerfile/SOC-only
  story is verified, not just documented.
- Frontend tests (new): vitest 2 + `vitest.config.ts` + `src/services/api.test.ts`
  (5 tests: flagged mock fallback, unreachable health, config errors on all typed
  endpoints, ws/wss URL mapping) → `npm test` **5/5 pass**; script added.
- Analyze page rejects >25 MB files client-side before upload (backend still
  enforces independently).
- Verification: pytest 14/14 (3 runtimes: system, torch venv, fresh SOC-only),
  `npm test` 5/5, frontend build 20 routes, compileall clean.

## Completed — Phase 8 (this session): dead-end + hardcoded-number sweep

- Threats page: added triage-status filter (any/open/acknowledged/closed) next to
  the severity filter; Mark-buttons update list state in place.
- Sessions page: real loading skeleton (was flashing "no sessions" while loading).
- New `/docs` page (auth-guarded, real content: contract, verified model facts,
  live backend-docs link, known limits) — the nav Documentation item 404'd before.
- AppShell header: hardcoded `Latency: 42ms / Streams: 01 / Threats: 0` replaced
  with live values from `/api/dashboard` (analyses, high-risk threats, model
  version/MOCK; `off` when backend unreachable).
- Live waveform caption now reflects real mic state (capture active/idle) instead
  of a static 48 kHz label.
- Verification: pytest 14/14, frontend build 20 routes, compileall clean.

## Completed — Phase 7 (this session): latency profile + honesty pass

- `scripts/profile_latency.py` on the real model (Windows CPU, 3 reps each):
  0.5 s → 290–396 ms; 1.0 s → 379–409 ms; 2.0 s → 557–577 ms; 5.0 s → ~1300 ms;
  median 483.1 ms. WavLM CPU is >99% of every run; all other stages < 10 ms
  combined. Deterministic per duration (0.760/0.822/0.832/0.840, all UNCERTAIN).
  700 ms target holds to ~2 s clips — recorded in `AI_MODEL_SPEC.md`.
- pytest 14/14 re-verified inside the real torch venv (was: system python only).
- Frontend honesty: Live console's mock-driven DecisionConsole now carries a
  SIMULATED-feed badge (Mic Live Scan above it is the real path); Overview
  recent rows link to full forensics.
- Docs audit: all 15 SOC routes + WS present in code match `API.md`; fixed the
  stale "JWT planned" line (optional JWT shipped in Phase 3).
- Verification: pytest 14/14 (both runtimes), frontend build 19 routes,
  compileall clean.

## Completed — Phase 6 (this session): REAL-MODEL HTTP E2E ✅

- Booted the full FastAPI app on the torch venv (`MOCK_INFERENCE=false`,
  legacy engine on after adding pandas/omegaconf/pyyaml/tqdm/matplotlib) and ran
  `scripts/e2e_probe.py` against it: **16/16 PASS with `model=rafnet,
  mock_mode=False`** — health, healthz, dashboard, file-upload analyze, detail,
  history, threats, calls, models, experiments (9), live session + chunk.
- Real-model upload over HTTP: tone.wav → UNCERTAIN @ 0.8318, risk 60/MEDIUM,
  mock=False, 950.1 ms — matching the direct smoke test. This closes the loop:
  **HTTP → validation → decode → actual RAFNet → risk engine → store → response**.
- Threat triage: `PATCH /api/threats/{id}` (open/acknowledged/closed, 400 on bad
  status, 404 on unknown) + frontend Mark-buttons; `test_threat_triage_workflow`
  forces a CRITICAL via a loud mock tone and walks the full workflow.
- Frontend: Analyze result links to full forensics (`/history/{id}`) + history.
- Recon (read-only) of the parallel Node backend: same route surface (plus
  `/api/analyses*` aliases + DELETE/report extras), compatible record shape, DSP
  heuristic detector (honestly flagged) + SQLite. Frontend works against either.
  Recommendation stands: Python/RAFNet is the real-model path per spec; Node is
  the lightweight alternative. Reconcile before deploying both.
- Verification: pytest 14/14, frontend build 19 routes, compileall clean,
  real-model HTTP E2E 16/16.

## Completed — Phase 5 (this session): REAL MODEL INFERENCE ✅

- Built a torch CPU venv and ran the full stack for real: torch 2.14.1+cpu,
  torchaudio, transformers, librosa, soundfile all import; WavLM
  `microsoft/wavlm-base-plus` downloaded from HF Hub (248 shards) and loaded.
- `gateway2.pth` verified as a real OrderedDict (`gate_network` MLP 3→32→32→1 +
  gate stats), matching the Gateway2 architecture in `AI_MODEL_SPEC.md`.
- Fixed the SOC adapter against the REAL pipeline APIs (found by reading code,
  not guessing): `predict_audio_only(tensor[B,T])` (no sample_rate kwarg),
  branch-level checkpoint restore (`ClassicalBranch.load`, `QuantumBranch.load`,
  gateway2 state subset with `strict=False`, whole-module calibrator/gate pickles
  with fitted checks) replacing the incorrect generic `pipeline.load()` attempt.
- **`scripts/test_model.py` PASSES with the real model**: 440 Hz tone →
  UNCERTAIN @ 0.8216, risk 60/MEDIUM, 1049.8 ms wall (consistent with Phase19's
  ~1182 ms CPU figure), mock=False, all 5 checkpoints restored. The UNCERTAIN
  verdict on a pure tone matches the model's known behavior (Phase19: everything
  lands UNCERTAIN, p~0.7–0.8) — reported as-is, not tuned to look better.
- Environment honesty: sklearn 1.9.1's DLLs are blocked by this machine's
  Application Control policy, so the smoke runs under sklearn 1.6.1 with two
  documented, weight-preserving adapter shims (`multi_class='ovr'` on the
  1.9.1-pickled binary LogisticRegression; fitted-PCA bookkeeping re-derived from
  shipped components/mean). Production MUST use sklearn==1.9.1
  (`requirements-ml.txt`); shims warn at runtime. Details in `AI_MODEL_SPEC.md`.
- Also shipped: `scripts/e2e_probe.py` (operator E2E against a live server),
  `requirements-ml.txt`, `supabase/README.md` runbook, bare auth pages (no SOC
  chrome on /login + /register), ARCHITECTURE boot-mode notes.
- Coexistence note: a parallel workstream is building a Node/Express backend in
   `rafnet-frontend/backend/` (own package.json/tsconfig, `node:sqlite`).
  Untouched except excluding `backend/` from the Next tsconfig (it has its own
  build config) so `next build` stays green. Its `backend/.env` is theirs.
- Verification: pytest 13/13, frontend build 19 routes, compileall clean,
  real-model smoke PASS.

## Completed — Phase 4 (this session)

- Frontend gaps closed: `/threats` page (severity filter + detail + related-analysis
  link), `/history/[id]` analysis detail (verdict, confidence vs risk, forensic
  signals, gateway weights, risk reasons, retained bus events), history rows link to
  detail, Threats added to nav. Fixed two real issues found along the way: missing
  gateway-weight fields in the `AnalyzeResult` type, and AppShell `usePathname`
  needing a Suspense boundary for dynamic routes (frontend build 19 routes).
- Live-server E2E (mock model, honestly flagged): booted uvicorn SOC-only and
  exercised health, healthz, dashboard, analyze (UNCERTAIN 74.5% MEDIUM mock),
  detail, history, threats, calls, models, experiments (9), live session + chunk
  (analyzed) — all 200, E2E OK.
- Security sweep: only `.env.example` files exist (no live secrets on disk);
  backend `.gitignore` added (`.env`, venv, caches); frontend `.gitignore` already
  covers `.env*`; no hardcoded keys/tokens/private keys found in new code; CORS
  locked to localhost by default; auth open-by-default is documented with
  `SOC_REQUIRE_AUTH=true` for production.
- Verification: pytest 13/13, frontend build 19 routes, compileall clean.

## Completed — Phase 3 (this session)

- Boot hardening: legacy torch imports guarded (`api/__init__.py`, `api/main.py`,
  unused torch import removed); full app boots SOC-only without torch
  (`test_full_app_boots_without_torch`), `/healthz` mirrors SOC state, `/api/v1`
  disabled with a logged reason instead of crashing the process.
- Auth: `soc/auth.py` (optional Supabase JWT via Auth API, stdlib-only;
  `SOC_REQUIRE_AUTH` enforces on writes/personal reads, 401/503 fail-closed;
  otherwise anonymous-open with opportunistic `user_id` attach); service + routes
  thread `user_id` through analyze/history/threats/calls/dashboard; frontend sends
  the Supabase session token automatically.
- Chunked live inference: `soc/live.py` (session registry, validation, linear
  16 kHz resample for chunks) + `POST /api/live/session|/api/live/chunk`
  (buffering until 1.0 s window, then SAME model + risk engine, WS events);
  `useMicChunks` hook (mic -> 16 kHz mono -> 1 s POSTs) + Mic Live Scan panel on
  the Live page showing real per-window verdicts; no synthesized detections.
- Deploy: `Dockerfile` (SOC-only base, torch block documented), `requirements-soc.txt`,
  `supabase/seed.py` (dry-run verified: 1 model + 9 experiments from real files).
- Verification: pytest 13/13, frontend build 16 routes, compileall clean.

## Completed — Phase 2 (prior)

## Completed — Phase 2 (this session)

- Persistence: `soc/store.py` (AnalysisStore port + InMemory default + Supabase REST
  adapter opt-in via server-only env, same record shape); `supabase/migrations/
  001_soc_schema.sql` (7 tables + RLS owner-only + read-only models/experiments + model seed).
- Extended API (all real, no mocks): `/api/history`, `/api/threats[/{id}]`,
  `/api/calls[/{id}]` (derived 1:1 sessions + HIGH/CRITICAL threats),
  `/api/experiments` (verbatim phase19 + ablation JSON), `/api/analysis/{id}/events`.
- Live: `soc/events.py` EventBus (analysis_started/processing/detection/
  threat_detected/analysis_completed/error) + `WS /ws/live` (replay 20, heartbeat,
  graceful disconnects) mounted in `api/main.py`.
- Hardening: CORS via `CORS_ORIGINS` (no wildcard default), per-IP rate limit on
  analyze (`RATE_LIMIT_ANALYZE_PER_MIN`, 429 + Retry-After), safe filenames, no
  secret/stack leaks.
- Frontend: Supabase Auth (`lib/supabase.ts`, `lib/auth.tsx`, `/login`, `/register`,
  dev-mode bypass clearly labeled, `RequireAuth` on all SOC pages, user badge + logout
  in AppShell, live backend-health badge); real pages — Overview (dashboard API),
  Models (checkpoint sizes), History (search/filter), Sessions, Research (verbatim
  experiments), Fusion (gateway semantics), Network (honest telemetry status),
  Settings (env/session display); `useLiveEvents` WS hook (reconnect + status) wired
  into Live console with real event feed replacing the fake log.
- Verification: pytest 9/9, `npm run build` 16 routes, compileall clean.

## Completed — Phase 1 (prior)

- Full audit of rafnet (code, checkpoints, experiments, API, tests), Stitch designs
  (15 screens), and Next.js scaffold. Wrote PROJECT_AUDIT.md + AI_MODEL_SPEC.md
  (facts only, no invented metrics).
- SOC adapter `src/rafnet/soc/`: model_adapter (VoiceDetectionModel + RAFNetVoiceModel
  + MockVoiceModel + MOCK_INFERENCE gate), audio_io (validation, 25MB, 8 suffixes,
  soundfile + wav fallback), risk_engine (score/level/reasons, env thresholds),
  schemas (AnalyzeResponse/Dashboard/ModelInfo), service (validate->model->risk->store),
  routes (/api/analyze, /api/analysis/{id}, /api/dashboard, /api/models, /api/health).
  Mounted in `api/main.py` without touching `/api/v1/*`.
- `scripts/test_model.py` (real load + preprocess + infer + risk + timing, honest fail).
- `tests/test_soc.py`: 9 passed (health, 400s, wav contract, models, history/threats/
  calls, verbatim experiments, rate-limit 429, WS live event flow) without torch.
- Frontend: `services/api.ts` env-driven + typed + flagged mock fallback; functional
  Analyze page (upload -> POST /api/analyze -> verdict/confidence/risk/latency);
  `.env.example`; `npm run build` passes (16 routes incl. /login, /register).
- Docs: README.md, ARCHITECTURE.md, API.md, DATABASE.md (schema + RLS plan),
  RISK_ENGINE.md, DEPLOYMENT.md, .env.example (backend) + frontend .env.example.
- Verification: `compileall` clean, pytest 5/5, frontend build ok, checkpoints present,
  model smoke fails honestly (no torch on this Windows box).

## Model

RAFNet (torch + sklearn, WavLM-base-plus frozen 94M via HF + handcrafted ->
Gateway1 -> logistic classical + PCA/fallback-MLP quantum -> Gateway2 MLP ->
temperature calibration -> gate tau 0.361/0.861). Input 16kHz mono, 1.0s window,
RMS -26dB; network 32-dim when present. Checkpoints: classical 2465B, quantum 16634B,
gateway2 11019B, calibrator 1977B, gate 1638B. Audio-only upload uses Gateway1
fixed_audio. No invented I/O, metrics, or behavior.

## Frontend

Functional: Analyze (real upload flow + states + mock badge). Preserved SOC visuals:
Overview, Live (placeholder realtime interval, flagged TODO for WS), Sessions, History,
Models, Network, Fusion, Research, Settings render from existing scaffold. No dead-end
changes beyond Analyze scope; every Analyze button performs a real action.

## Backend

Kept: POST /api/v1/predict, /predict/batch, GET /v1/health, /model, /version, /latency,
/log, POST /v1/admin/*, /healthz, /. Added: POST /api/analyze, GET /api/analysis/{id},
/api/dashboard, /api/models, /api/health. Typed, status-coded, no secret/stack leaks,
server logs. No auth yet (documented).

## Database

Designed, not provisioned: profiles, analysis_results, calls, threats, models,
experiments, analysis_events + RLS (owner-only, no anon writes, models read-only).
Current `SOCService._store` in-memory ring (same shape) for local/demo.

## Authentication

Frontend Supabase Auth + `RequireAuth` guards + dev bypass (labeled); backend optional
JWT (`SOC_REQUIRE_AUTH=false` default = open; `true` = Bearer enforced via Supabase
Auth API, user_id attached + per-user filtering). Remaining: run migration in a real
project and set `SOC_REQUIRE_AUTH=true` there; token-refresh edge cases.

## WebSocket

`/ws/live` real event stream + `useLiveEvents` reconnect + real event feed; chunk
ingestion emits processing/detection/threat events per full window; Mic Live Scan panel
posts real 1 s mic chunks (browser permission required). Remaining: multi-replica
pub/sub (process-local bus + session registry), Supabase persistence for live chunks.

## Testing

- `npm run build`: pass (Next 16.4, 13 routes, 4.8s compile).
- `python -m compileall soc/scripts/test_soc`: clean.
- `pytest tests/test_soc.py`: 5 passed.
- `scripts/test_model.py`: FAIL (honest) — `No module named 'torch'` on this Windows
  Python 3.12; checkpoints verified present (5/5). Rerun in Linux venv with torch +
  HF access for a real PASS.
- API test: TestClient mock-contract passes (prediction/confidence/risk/mock fields,
  retrieval, dashboard increment).

## Deployment

Frontend -> Vercel (`NEXT_PUBLIC_BACKEND_URL` etc). Backend/ML -> persistent Python host
(Railway/Render/Fly/VM, `uvicorn rafnet.api.main:app`), not Vercel serverless (torch +
94M WavLM + 700-1200ms CPU). Supabase for Auth/Postgres. See DEPLOYMENT.md.

## Known limitations

Synthetic-only data + generator overlap; untrained encoders/gates; frozen WavLM;
fallback (non-quantum) branch; Phase19 n=20 all UNCERTAIN; CPU latency over budget;
fusion unvalidated; no auth/RLS-live/WS; in-memory store; Windows box lacks torch.

## Database

Migration + RLS written (`supabase/migrations/001_soc_schema.sql`); runtime uses the
store port (in-memory default, Supabase REST when `SUPABASE_URL` +
`SUPABASE_SERVICE_ROLE_KEY` are set). Remaining: run migration in a real project,
backfill analyses, point backend at it.

## Testing

- Full suite (torch venv): **343 passed, 3 skipped, 0 failed** (all supplied
  tests + all 18 SOC tests).
- `pytest tests/test_soc.py`: 23 passed (+ seed/verify scripts vs fake PostgREST
  with RLS emulation) — in 3 runtimes (system python, torch venv, fresh SOC-only
  venv with no torch).
- `npm test` (vitest): 5 passed (service contract, mock honesty, URL mapping).
- `npm run build`: 20 routes pass. `compileall` clean on soc/api/scripts/seed.
- `scripts/test_model.py`: **PASS with the real model** (UNCERTAIN @ 0.8216,
  1049.8 ms, mock=False).
- `scripts/e2e_probe.py`: **16/16 against the real-model server**
  (`model=rafnet, mock_mode=False`; upload → UNCERTAIN @ 0.8318, 950.1 ms).
- `supabase/seed.py` (dry-run): 1 model (real checkpoint sizes) + 9 experiments.

## Remaining work

Run migration + `seed.py --apply` in a real Supabase project and set
`SOC_REQUIRE_AUTH=true` there; production torch runtime with sklearn==1.9.1
(local shims retire there); latency/GPU pass; multi-replica bus/store
(Redis/Postgres LISTEN); Vercel + backend deploy with health gate
(`model_loaded:true, mock_mode:false`). Reconcile with the parallel Node-backend
workstream (`rafnet-frontend/backend/`: same routes + `/api/analyses*`
aliases, compatible records, DSP-heuristic detector + SQLite, needs Node ≥ 22)
on the canonical backend choice — Python/RAFNet is the real-model path per spec.
