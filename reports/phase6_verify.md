# Phase 6 verification (additive v2 stack + guards)

Generated: 2026-10-26T15:02:11.532580+00:00

## Artifacts
{"rafnet_loaded": true, "head_loaded": true, "head_error": null}

## .opus accept
{"suffix_allowed": true, "lossy": true, "decode_ok": [{"name": ".ogg", "sr": 16000, "dur": 57.78, "is_lossy": true}, {"name": ".ogg", "sr": 16000, "dur": 37.8, "is_lossy": true}], "decode_err": [], "renamed_opus_decode": {"sr": 16000, "dur": 57.78}}

## Degenerate-segment guard (<1600 samples)
{"empty": "ValueError: degenerate audio segment: 0 samples at 16 kHz after preprocessing (minimum 1600 ", "tiny": "ValueError: degenerate audio segment: 800 samples at 16 kHz after preprocessing (minimum 160", "just_under": "ValueError: degenerate audio segment: 1599 samples at 16 kHz after preprocessing (minimum 16"}

## Determinism (10 in-process repeats, full score path)
{"repeats": 10, "unique_digests": 1, "digest": "a23f2b3b1b70a7e4", "scores": {"rafnet": 0.8899, "mo": 0.9999, "bisher": 0.0027, "head": 0.0839, "p_v1": 0.9847, "verdict_v1": "SYNTHETIC", "p_v2": 0.0, "verdict_v2": "REAL"}, "stable": true}

## Determinism (fresh-interpreter restart)
{"rc": 0, "digest": "a23f2b3b1b70a7e4", "in_process_digest": "a23f2b3b1b70a7e4", "match": true}

## Before/after (v1 3-feature vs v2 4-feature with head)
| id | group | y | rafnet | mo | bisher | head | p_v1 | v1 | p_v2 | v2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wa_ogg_1 | private_whatsapp_opus | - | 0.8899 | 0.9999 | 0.0027 | 0.0839 | 0.9847 | SYNTHETIC | 0.0 | REAL |
| wa_ogg_2 | private_whatsapp_opus | - | 0.9021 | 0.0 | 0.0024 | 0.0577 | 0.0568 | UNCERTAIN | 0.0 | REAL |
| fleurs_en_train_00895 | fleurs-en | 0 | 0.8264 | 0.9999 | 0.002 | 0.0138 | 0.9845 | SYNTHETIC | 0.0 | REAL |
| fleurs_en_train_00911 | fleurs-en | 0 | 0.8239 | 0.0 | 0.002 | 0.0593 | 0.0561 | UNCERTAIN | 0.0 | REAL |
| edge-Valluvar-000 | fake-edge | 1 | 0.9098 | 1.0 | 0.0024 | 0.4326 | 0.9847 | SYNTHETIC | 0.011 | REAL |
| edge-Valluvar-001 | fake-edge | 1 | 0.8647 | 1.0 | 0.0073 | 0.7417 | 0.9848 | SYNTHETIC | 0.9917 | SYNTHETIC |

## Codec probe (real clip, Opus/AMR)
| id | codec | p_v1 | v1 | p_v2 | v2 |
| --- | --- | --- | --- | --- | --- |
| fleurs_en_train_00895 | opus_24k | 0.9845 | SYNTHETIC | 0.0 | REAL |
| fleurs_en_train_00895 | amr_nb | error: ValueError: Unsupported audio format '.amr'. Allowed: ['.flac', '.m4a', '.mp3', '.mp4', '.oga', '.ogg', '.opus', |

Private clips stay local; ids are anonymous. v2 is additive: v1 fields and risk gating are unchanged when models/stack_phase4.json is absent.
