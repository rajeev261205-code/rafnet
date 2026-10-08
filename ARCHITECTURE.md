# ARCHITECTURE.md — Aegis Voice SOC (real integration)

```
Vercel (Next.js frontend) --HTTPS--> FastAPI backend --+--> RAFNet InferencePipeline (real model)
                                                      +--> Supabase Postgres (persistent store)
                                                      +--> WebSocket live events
```

## Repos / roots

- Backend + ML: `C:\Users\rajee\rafnet\rafnet` (authoritative; `src/rafnet/*`)
  - Existing JSON-frame API: `src/rafnet/api/*` -> `/api/v1/*` (unchanged)
  - NEW SOC adapter: `src/rafnet/soc/*` -> `/api/analyze`, `/api/analysis/{id}`,
    `/api/dashboard`, `/api/models`, `/api/health`
- Frontend: `C:\Users\rajee\aegisvoice-frontend` (Next.js 16)
  - `src/services/api.ts` -> env-driven (`NEXT_PUBLIC_BACKEND_URL`), typed, mock fallback flagged
- Designs: Stitch `stitch_aegis_voice_soc_dashboard/*.html` (visual reference only)

## Request flow (file upload)

UI -> `services/api.analyzeAudio(file)` -> `POST /api/analyze (multipart)`
-> validate (type/size) -> decode (soundfile/wav fallback) -> mono
-> `RAFNetVoiceModel.predict` (InferencePipeline.predict_audio_only, Gateway1 fixed_audio)
-> `compute_risk` -> store port (in-memory default, Supabase when configured)
-> `AnalyzeResponse` -> UI. Live mic path: `useMicChunks` (16 kHz mono, 1 s POSTs)
-> `POST /api/live/chunk` (buffering until 1.0 s window, then the same predict +
risk path, WS events per window).

Boot modes: with torch the full app serves `/api/v1/*` + SOC; without torch the
legacy stack is disabled and the app boots SOC-only (mock or real-adapter-degraded,
reported honestly by `/api/health`). Production ML runtime: `requirements-ml.txt`
(torch CPU, sklearn==1.9.1 to match the training pickles); SOC-only runtime:
`requirements-soc.txt`. sklearn compat shims in the adapter are documented in
`AI_MODEL_SPEC.md` and only bridge older runtimes.

## Deployment fit

- Frontend: Vercel (static/SSR, `NEXT_PUBLIC_*` only).
- Backend/ML: persistent Python host (Railway/Render/Fly/GPU box) — NOT Vercel
  serverless (94M WavLM + torch + 700ms-1200ms CPU inference unsuitable).
- Supabase: Auth + Postgres; server key stays backend-only.
