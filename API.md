# API.md — RAFNet backend

Base: `http://localhost:8000`. Interactive docs: `/docs`.

## SOC (new, file-upload oriented)

- `GET /api/health` -> `{status, model_loaded, mock_mode, model}`
  - `model_loaded:true` ONLY if the real pipeline loaded.
- `POST /api/analyze` (multipart `file`) -> `AnalyzeResponse`
  - 400 unsupported/corrupt/oversize/empty; 503 model not loaded; 500 generic.
  - Max 25 MB; suffixes: wav flac ogg oga mp3 m4a mp4 webm.
- `GET /api/analysis/{id}` -> record or 404.
- `GET /api/analysis/{id}/events` -> related live-bus events for the caller's
  own analysis; unknown or foreign ids return 404.
- `GET /api/history?limit=&offset=` -> `{items, total}` newest first.
- `DELETE /api/analyses` -> `{deleted}` — clears the caller's analyses
  (history/threats/calls derive from analyses, so they vanish too). Anonymous
  callers clear anonymous rows only; authenticated callers clear their own rows
  and can never delete another user's or anonymous rows. Backed by
  `Settings > Clear all analysis data` (confirm dialog).
- `GET /api/threats`, `GET /api/threats/{id}` -> HIGH/CRITICAL-derived threat records.
- `POST /api/call-risk` `{transcript, carrier: unknown|jio|airtel|other, language}`
  -> `{risk: low|suspicious|high|insufficient, score, scale, evidence[], explanation,
  actions[], carrier, limitations}`. Transcript-based scam-behavior analysis
  (rule-driven, English + Hindi-English, context-aware against OTP-advisory
  false positives). Separate from voice authenticity: never sees audio; carrier
  recorded, never scored. 422 on bad carrier.
- `POST /api/attribution` `{provenance}` -> `{provider: "unknown", grade,
  evidence[], explanation, limitations}`. No provider fingerprint exists, so
  attribution is honestly unknown; supplied provenance is recorded, not trusted.
- `GET /api/capabilities` -> per-module readiness `{neural, replay, speaker,
  transcription, attribution}` so the UI shows what is actually ready.
- `PATCH /api/threats/{threat_id}` `{status: open|acknowledged|closed}` -> triage
  workflow (process-local overrides; Supabase `threats.status` is the prod home).
- `GET /api/calls`, `GET /api/calls/{id}` -> 1:1 call sessions derived from analyses.
- `GET /api/experiments` -> verbatim `phase19_report.json` + ablation `results.json` (never fabricated).
- `GET /api/dashboard` -> totals + recent + model status (empty state when no analyses).
- `GET /api/models` -> real checkpoint sizes, active tau_low/tau_high
  (from `configs/decision_thresholds.json`, defaults = shipped 0.3611/0.8611),
  `Not available from supplied model metadata` where absent.
- `WS /ws/live` -> real analysis_started/processing/detection/threat_detected/
  analysis_completed/error + replay of last 20 events **for the caller only** +
  25s heartbeat; exponential-backoff reconnect handled client-side
  (`useLiveSocket`, which sends `?token=` when logged in). When auth is required,
  missing/invalid tokens close with 4401; a supplied token is validated even in
  dev mode rather than silently downgraded to anonymous.
- Rate limit: `POST /api/analyze` per-IP budget (`RATE_LIMIT_ANALYZE_PER_MIN`, default 30)
  -> 429 + `Retry-After`. Mic chunks have a separate sustained budget
  (`RATE_LIMIT_CHUNK_PER_MIN`, default 120) so live scans aren't throttled mid-session.
- Body guard: requests declaring `Content-Length` over `MAX_REQUEST_BYTES`
  (default 32 MB) are rejected with 413 before buffering.
- Live chunk ingestion (honest mic path, no fake streaming):
  `POST /api/live/session` -> `{session_id}`; `POST /api/live/chunk`
  `{session_id, sample_rate 8000–48000, samples[]}` -> `buffering` until 1.0 s
  window, then `analyzed` with the SAME model + risk engine as file upload.
  Each analyzed chunk returns the single-window `prediction` PLUS the rolling
  neural aggregation (`aggregated_prediction`, `aggregated_p_synthetic`,
  `aggregated_chunks`/`aggregated_window`, active `thresholds`): median of the
  last 5 chunk scores with majority hysteresis, so flicker never flips the
  session view. Oversize/non-finite payloads -> 400/422.
- Auth: `Authorization: Bearer <supabase access_token>` (validated via Supabase
  Auth API; 401 invalid, 503 unconfigured). Required for the whole SOC API when
  `SOC_REQUIRE_AUTH=true`, which is the default when `RAFNET_ENV=production`
  (the anonymous load-balancer probe remains `GET /healthz`); development
  defaults to anonymous-allowed. A supplied token is always validated.
  Ownership is enforced in every store query: authenticated users
  see their own rows, anonymous callers see anonymous rows, and neither sees the
  other. Frontend sends the session token automatically when Supabase is
  configured.

`AnalyzeResponse`: id, prediction (REAL|UNCERTAIN|SYNTHETIC), confidence (model),
risk_score/risk_level/reasons (engine), model_version, processing_time_ms,
audio_duration_seconds, probabilities + gateway weights, mock flag, created_at.

## Legacy RAFNet JSON-frame API (unchanged)

Prefix `/api/v1`: POST /predict, POST /predict/batch (audio as List[float]),
GET /health, /model, /version, /latency, /log, POST /admin/fit-*.
See `src/rafnet/api/routes.py`. Mounted behind the same Supabase auth dependency
as the SOC API, so enabling `SOC_REQUIRE_AUTH` protects these endpoints too.

Verified streaming behavior (pre-existing, untouched supplied code — live 2026-10-08):
a single JSON frame 504s ("Inference timeout", service timeout 5.0 s) because the
threaded `InferenceEngine` only fires when the audio window is full AND the network
telemetry buffer holds ≥ 5 samples. Six rapid sequential frames: frames 1–4 prime
the buffers (504s), frames 5–6 return **200 with real engine inference**
(UNCERTAIN, ~400–490 ms). Note those engine verdicts were p=0.5/UNCERTAIN because
the engine boots with unfitted classical/quantum branches (it does not restore the
shipped `experiments/checkpoints/*`; runtime fitting is via the admin `fit-*`
endpoints) — the graceful 0.5/UNCERTAIN degradation from `model.forward`, not an
error. The SOC paths (`/api/analyze`, `/api/live/chunk`) call `InferencePipeline`
directly with restored checkpoints and are the recommended inference routes.

## Conventions

Typed Pydantic responses, proper status codes, no stack traces/secrets in
responses, structured server logs. Supabase JWT is required in production
(`RAFNET_ENV=production` defaults `SOC_REQUIRE_AUTH=true`); both the SOC and
legacy routers share the auth dependency. CORS uses explicit origins with
credentials only for non-wildcard lists; API responses carry `nosniff`,
`DENY`, `no-referrer`, `no-store` headers and HSTS in production.
