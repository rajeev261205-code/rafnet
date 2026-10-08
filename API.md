# API.md — Aegis Voice SOC backend

Base: `http://localhost:8000`. Interactive docs: `/docs`.

## SOC (new, file-upload oriented)

- `GET /api/health` -> `{status, model_loaded, mock_mode, model}`
  - `model_loaded:true` ONLY if the real pipeline loaded.
- `POST /api/analyze` (multipart `file`) -> `AnalyzeResponse`
  - 400 unsupported/corrupt/oversize/empty; 503 model not loaded; 500 generic.
  - Max 25 MB; suffixes: wav flac ogg oga mp3 m4a mp4 webm.
- `GET /api/analysis/{id}` -> record or 404.
- `GET /api/analysis/{id}/events` -> related live-bus events.
- `GET /api/history?limit=&offset=` -> `{items, total}` newest first.
- `GET /api/threats`, `GET /api/threats/{id}` -> HIGH/CRITICAL-derived threat records.
- `PATCH /api/threats/{threat_id}` `{status: open|acknowledged|closed}` -> triage
  workflow (process-local overrides; Supabase `threats.status` is the prod home).
- `GET /api/calls`, `GET /api/calls/{id}` -> 1:1 call sessions derived from analyses.
- `GET /api/experiments` -> verbatim `phase19_report.json` + ablation `results.json` (never fabricated).
- `GET /api/dashboard` -> totals + recent + model status (empty state when no analyses).
- `GET /api/models` -> real checkpoint sizes, tau_low 0.361 / tau_high 0.861,
  `Not available from supplied model metadata` where absent.
- `WS /ws/live` -> real analysis_started/processing/detection/threat_detected/
  analysis_completed/error + replay of last 20 events + 25s heartbeat; exponential-backoff
  reconnect handled client-side (`useLiveEvents`, which sends `?token=` when logged in).
  When `SOC_REQUIRE_AUTH=true`, anonymous sockets are closed with 4401.
- Rate limit: `POST /api/analyze` per-IP budget (`RATE_LIMIT_ANALYZE_PER_MIN`, default 30)
  -> 429 + `Retry-After`. Mic chunks have a separate sustained budget
  (`RATE_LIMIT_CHUNK_PER_MIN`, default 120) so live scans aren't throttled mid-session.
- Body guard: requests declaring `Content-Length` over `MAX_REQUEST_BYTES`
  (default 32 MB) are rejected with 413 before buffering.
- Live chunk ingestion (honest mic path, no fake streaming):
  `POST /api/live/session` -> `{session_id}`; `POST /api/live/chunk`
  `{session_id, sample_rate 8000–48000, samples[]}` -> `buffering` until 1.0 s
  window, then `analyzed` with the SAME model + risk engine as file upload.
  Oversize/non-finite payloads -> 400/422.
- Auth (optional): `Authorization: Bearer <supabase access_token>`; enforced only
  when `SOC_REQUIRE_AUTH=true` (validated via Supabase Auth API; 401 invalid,
  503 unconfigured). Otherwise requests are anonymous and open; a valid token
  still attaches `user_id` to saved records. Frontend sends the session token
  automatically when Supabase is configured.

`AnalyzeResponse`: id, prediction (REAL|UNCERTAIN|SYNTHETIC), confidence (model),
risk_score/risk_level/reasons (engine), model_version, processing_time_ms,
audio_duration_seconds, probabilities + gateway weights, mock flag, created_at.

## Legacy RAFNet JSON-frame API (unchanged)

Prefix `/api/v1`: POST /predict, POST /predict/batch (audio as List[float]),
GET /health, /model, /version, /latency, /log, POST /admin/fit-*.
See `src/rafnet/api/routes.py`. No auth (TODO for production).

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
responses, structured server logs. SOC routes carry optional Supabase JWT
(`SOC_REQUIRE_AUTH=true` to enforce); legacy `/api/v1` remains open
(TODO for production).
