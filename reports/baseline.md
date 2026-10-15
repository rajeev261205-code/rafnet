# Phase 0 baseline — CURRENT model on samples/labels.csv

Generated: 2026-10-15T13:48:33.921838+00:00
Model: loaded=true name=rafnet mock=false
Thresholds: tau_low=0.3611111111111111 tau_high=0.861111111111111 (configs/decision_thresholds.json)
Labels: C:\Users\rajee\RAFNet\samples\labels.csv — 19/19 scored (real=4, fake=15)
ffmpeg: C:\Users\rajee\RAFNet\.venv\Lib\site-packages\imageio_ffmpeg\binaries\ffmpeg-win-x86_64-v7.1.exe

> Every number below is measured in this run and saved in reports/baseline.json. Small-n snapshot, NOT a general benchmark.

## Sample inventory (what is actually in samples/)
- clean_human_librispeech: n=4 (real=4, fake=0) strict_acc=0.0 auc=None eer={"eer": null, "threshold": null, "note": "needs both classes"}
- windows_sapi_tts: n=4 (real=0, fake=4) strict_acc=0.0 auc=None eer={"eer": null, "threshold": null, "note": "needs both classes"}
- assistant_voice_chatgpt: n=1 (real=0, fake=1) strict_acc=0.0 auc=None eer={"eer": null, "threshold": null, "note": "needs both classes"}
- windows_tts: n=1 (real=0, fake=1) strict_acc=0.0 auc=None eer={"eer": null, "threshold": null, "note": "needs both classes"}
- windows_sapi_tts_extended: n=9 (real=0, fake=9) strict_acc=0.0 auc=None eer={"eer": null, "threshold": null, "note": "needs both classes"}
- MISSING vs task brief: real/whatsapp_opus/*, real/phone_line/*, real Hindi/Tamil/noisy-room sets, fake voice-conversion set — none present.

## Overall (clean, full-clip POST /api/analyze path)
n=19 strict_accuracy=0.0 coverage=0.0 selective=None AUC=0.5 EER={"eer": 0.4833, "threshold": 0.8236, "far": 0.5, "frr": 0.4667}
binary@0.5: {"accuracy": 0.7895, "precision": 0.7895, "recall": 1.0, "f1": 0.8824, "tp": 15, "tn": 0, "fp": 4, "fn": 0, "n": 19}
confusion: {"real": {"REAL": 0, "UNCERTAIN": 4, "SYNTHETIC": 0}, "fake": {"REAL": 0, "UNCERTAIN": 15, "SYNTHETIC": 0}}

## Per-file scores (clean)
- samples/real/260-123286-0012.flac [real/clean_human_librispeech] p=0.8323 -> UNCERTAIN (conf 0.8323) dur=2.43s sr=16000 lossy=False
- samples/real/4446-2273-0021.flac [real/clean_human_librispeech] p=0.8439 -> UNCERTAIN (conf 0.8439) dur=5.25s sr=16000 lossy=False
- samples/real/5142-33396-0059.flac [real/clean_human_librispeech] p=0.818 -> UNCERTAIN (conf 0.818) dur=5.47s sr=16000 lossy=False
- samples/real/6930-81414-0025.flac [real/clean_human_librispeech] p=0.7917 -> UNCERTAIN (conf 0.7917) dur=2.53s sr=16000 lossy=False
- samples/fake/sapi_david_2.wav [fake/windows_sapi_tts] p=0.8238 -> UNCERTAIN (conf 0.8238) dur=5.15s sr=22050 lossy=False
- samples/fake/sapi_david_4.wav [fake/windows_sapi_tts] p=0.8099 -> UNCERTAIN (conf 0.8099) dur=4.54s sr=22050 lossy=False
- samples/fake/sapi_zira_2.wav [fake/windows_sapi_tts] p=0.8498 -> UNCERTAIN (conf 0.8498) dur=5.04s sr=22050 lossy=False
- samples/fake/sapi_zira_4.wav [fake/windows_sapi_tts] p=0.8077 -> UNCERTAIN (conf 0.8077) dur=4.56s sr=22050 lossy=False
- samples/fake/chatgpt_voice/rafnet_ai_voice_test.wav [fake/assistant_voice_chatgpt] p=0.8603 -> UNCERTAIN (conf 0.8603) dur=25.58s sr=22050 lossy=False
- samples/fake/windows_tts/tts_test.wav [fake/windows_tts] p=0.8262 -> UNCERTAIN (conf 0.8262) dur=6.51s sr=22050 lossy=False
- samples/win_tts/sapi_david_0.wav [fake/windows_sapi_tts_extended] p=0.8239 -> UNCERTAIN (conf 0.8239) dur=6.06s sr=22050 lossy=False
- samples/win_tts/sapi_david_2.wav [fake/windows_sapi_tts_extended] p=0.8238 -> UNCERTAIN (conf 0.8238) dur=5.15s sr=22050 lossy=False
- samples/win_tts/sapi_david_4.wav [fake/windows_sapi_tts_extended] p=0.8099 -> UNCERTAIN (conf 0.8099) dur=4.54s sr=22050 lossy=False
- samples/win_tts/sapi_zira_0.wav [fake/windows_sapi_tts_extended] p=0.8236 -> UNCERTAIN (conf 0.8236) dur=6.02s sr=22050 lossy=False
- samples/win_tts/sapi_zira_1.wav [fake/windows_sapi_tts_extended] p=0.814 -> UNCERTAIN (conf 0.814) dur=5.56s sr=22050 lossy=False
- samples/win_tts/sapi_zira_2.wav [fake/windows_sapi_tts_extended] p=0.8498 -> UNCERTAIN (conf 0.8498) dur=5.04s sr=22050 lossy=False
- samples/win_tts/sapi_zira_3.wav [fake/windows_sapi_tts_extended] p=0.8158 -> UNCERTAIN (conf 0.8158) dur=6.9s sr=22050 lossy=False
- samples/win_tts/sapi_zira_4.wav [fake/windows_sapi_tts_extended] p=0.8077 -> UNCERTAIN (conf 0.8077) dur=4.56s sr=22050 lossy=False
- samples/win_tts/sapi_zira_5.wav [fake/windows_sapi_tts_extended] p=0.8195 -> UNCERTAIN (conf 0.8195) dur=6.83s sr=22050 lossy=False

## Codec stress (real files only: clean p -> codec p, delta)
### opus_24k: {"n": 4, "mean_delta": -0.0009, "max_abs_delta": 0.0036, "mean_p_codec": 0.8206, "simulated": false}
- samples/real/260-123286-0012.flac: 0.8323 -> 0.8287 (delta -0.0036) -> UNCERTAIN
- samples/real/4446-2273-0021.flac: 0.8439 -> 0.8468 (delta 0.0029) -> UNCERTAIN
- samples/real/5142-33396-0059.flac: 0.818 -> 0.8157 (delta -0.0023) -> UNCERTAIN
- samples/real/6930-81414-0025.flac: 0.7917 -> 0.7911 (delta -0.0006) -> UNCERTAIN
### mp3_128k: {"n": 4, "mean_delta": 0.0051, "max_abs_delta": 0.0115, "mean_p_codec": 0.8266, "simulated": false}
- samples/real/260-123286-0012.flac: 0.8323 -> 0.8438 (delta 0.0115) -> UNCERTAIN
- samples/real/4446-2273-0021.flac: 0.8439 -> 0.8479 (delta 0.004) -> UNCERTAIN
- samples/real/5142-33396-0059.flac: 0.818 -> 0.8222 (delta 0.0042) -> UNCERTAIN
- samples/real/6930-81414-0025.flac: 0.7917 -> 0.7924 (delta 0.0007) -> UNCERTAIN
### amr_nb: {"n": 4, "mean_delta": 0.0173, "max_abs_delta": 0.0338, "mean_p_codec": 0.8388, "simulated": false}
- samples/real/260-123286-0012.flac: 0.8323 -> 0.8396 (delta 0.0073) -> UNCERTAIN
- samples/real/4446-2273-0021.flac: 0.8439 -> 0.8565 (delta 0.0126) -> UNCERTAIN
- samples/real/5142-33396-0059.flac: 0.818 -> 0.8518 (delta 0.0338) -> UNCERTAIN
- samples/real/6930-81414-0025.flac: 0.7917 -> 0.8073 (delta 0.0156) -> UNCERTAIN
### g711_alaw: {"n": 4, "mean_delta": 0.0201, "max_abs_delta": 0.0301, "mean_p_codec": 0.8416, "simulated": false}
- samples/real/260-123286-0012.flac: 0.8323 -> 0.8589 (delta 0.0266) -> UNCERTAIN
- samples/real/4446-2273-0021.flac: 0.8439 -> 0.8584 (delta 0.0145) -> UNCERTAIN
- samples/real/5142-33396-0059.flac: 0.818 -> 0.8481 (delta 0.0301) -> UNCERTAIN
- samples/real/6930-81414-0025.flac: 0.7917 -> 0.8008 (delta 0.0091) -> UNCERTAIN

## Calibration table (stated confidence vs actual binary accuracy)
| confidence bin | n | accuracy | avg_confidence | gap (conf-acc) |
| --- | --- | --- | --- | --- |
| 0.5-0.6 | 0 | None | None | None |
| 0.6-0.7 | 0 | None | None | None |
| 0.7-0.8 | 1 | 0.0 | 0.7917 | 0.7917 |
| 0.8-0.9 | 18 | 0.8333 | 0.8256 | -0.0078 |
| 0.9-1.0 | 0 | None | None | None |

_Gap > 0 = overconfident in that bin. Small n: treat as indicative only._

## Honest notes
- n=19 (real=4, fake=15): snapshot of THESE files only, NOT a general benchmark.
- No WhatsApp/Opus human voice notes, phone-line, Hindi/Tamil, noisy-room or voice-conversion clips are present in samples/ — codec bias for those channels is unmeasured until such clips are added.
- win_tts/ duplicates 4 of the samples/fake/*.wav files; per-group counts treat them as listed in labels.csv (duplicates noted).
- ffmpeg present: codec columns are real transcodes (encode + decode back to 16k wav) through the stated codec settings.
- Thresholds unchanged in this phase; strict accuracy counts UNCERTAIN as wrong.
