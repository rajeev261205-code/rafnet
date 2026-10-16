# MOCK_DIAGNOSIS.md — why the backend runs in MOCK mode (evidence, 2026-10-12)

No code was changed to produce this file.

## 1. Current state (measured)

`GET /api/health` → `{"status":"ok","model_loaded":true,"mock_mode":true,"model":"mock"}`.
Mock is on because the running server was started with `MOCK_INFERENCE=true`
in its environment — the ONLY switch. Every live chunk therefore scores
exactly 0.50 (constant neutral stub ×100).

## 2. Every place mock mode can be enabled (exhaustive grep for MOCK_INFERENCE)

| Location | Value | Effect |
|---|---|---|
| `src/rafnet/soc/model_adapter.py:403` | default `"false"` | Mock is opt-IN, never default |
| `.env` | `MOCK_INFERENCE=false` | Correct for production |
| `.env.example`, `Dockerfile`, `start-local.ps1:41`, `DEPLOYMENT.md` | `false` | Correct |
| `scripts/test_model.py:29`, `profile_latency.py:9` | `setdefault("false")` | Correct |
| `scripts/e2e_probe.py:9` | `true` in a usage comment | Doc example for dev boot only |
| `tests/` | NOTHING sets the env var (test_api.py `mock_inference_service` is an unrelated unittest mock of the legacy service) | No leakage possible; conftest additionally strips Supabase creds |
| `frontend/` | no reference; `NEXT_PUBLIC_MOCK_FALLBACK=true` in `.env` is a dead var (unused in source) | No UI-forced mock |

Conclusion: configuration is correct everywhere. Mock is on ONLY because of
the boot environment of the running server.

## 3. Why the real model does not load here (exact exceptions)

Checkpoint path expected: `experiments/checkpoints/` (`MODEL_CHECKPOINT_DIR`
default; `.env` confirms). Files are REAL (not git-lfs pointers — no
`.gitattributes` LFS entries; magic bytes verified):
`decision_gate.pkl` starts `80 05 95` (pickle proto 5 + FRAME),
`gateway2.pth` starts `50 4b 03 04` (PK zip = torch archive).
Sizes: calibrator 1977, classical 2465, gate 1638, gateway2 11019, quantum 16634.

Per-runtime load results with `MOCK_INFERENCE=false`:
- system python (runs the current server): `ModuleNotFoundError: No module
  named 'torch'` (model_adapter.py:89, importing InferencePipeline).
- repo `.venv` (torch 2.14.1+cpu, sklearn 1.9.1): `ImportError: DLL load
  failed while importing _liblinear: An Application Control policy has
  blocked this file` (via sklearn.linear_model → svm._base). The CLASSICAL
  branch cannot even be imported; nothing is wrong with the checkpoint.
- `aegis-venv` (torch 2.14.1+cpu, sklearn 1.6.1): LOADS — all 5 checkpoints
  restored (classical via the documented multi_class shim, quantum via the
  PCA-attribute shim), verified 2026-10-12 with real LibriSpeech + SAPI TTS
  scores (see DIAGNOSIS.md).

Root cause: ENVIRONMENT (missing torch / OS-blocked sklearn DLL), not
config, not checkpoints. No checkpoint is missing — STOP condition not met.

## 4. Silent-fallback audit

- `load_voice_model()`: refuses silent mock fallback (logs error, returns
  unloaded RAFNet model). ✓ No change needed.
- `GET /api/health`: reports `model_loaded:false` + `degraded` but does NOT
  include the load error text. → FIX: add `error` field.
- Analysis/live endpoints: 503 with clear message when down (tested). ✓
- UI "REAL MODEL"/"AI ACTIVE" labels: NOT FOUND in current source
  (grep 2026-10-12) — the Live header, chips, and Model Intelligence already
  derive from `mock_mode`/`model_loaded`. The running dev server serves this
  source, so any such label is a stale build/tab. → FIX: none needed in code;
  verify visually after restart.

## 5. Fix plan (executed — see §6 for outcomes)

1. Run the local backend on the WORKING runtime (aegis-venv python) with
   `MOCK_INFERENCE=false` → real model loads, health true/false accordingly.
2. `start-local.ps1`: allow `RAFNET_VENV` override + verify torch+sklearn
   imports before boot (fail fast with the exact missing piece).
3. Health: expose `error` (load failure text or null).
4. Verify: health true+false, LibriSpeech vs SAPI scores differ and ≠50%,
   live chunk real score, pytest green.

## 6. Resolution (executed 2026-10-12, same day)

- Backend restarted on the working runtime (aegis-venv) with
  `MOCK_INFERENCE=false`: health →
  `{"status":"ok","model_loaded":true,"mock_mode":false,"model":"rafnet","error":null}`.
- Verified: LibriSpeech human 0.8323 vs SAPI TTS 0.8498 (differ, non-50%,
  mock:false); live chunk real 0.8309 with rolling aggregation.
- `GET /api/health` now exposes `error` (null when loaded).
- "AI Active" chip requires real loaded non-mock model; new red
  "Model not loaded (…)" chip otherwise. No "REAL MODEL" string existed in
  source (stale build/tab if seen).
- `start-local.ps1`: `RAFNET_VENV` override + ML-stack preflight (fail fast).
- pytest 41/41 (`test_soc.py`), concept: tests use in-process stubs with
  explicit service override — they never touch `MOCK_INFERENCE` and cannot
  enable mock for the real app.
