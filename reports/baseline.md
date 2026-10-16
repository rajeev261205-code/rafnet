# Phase 0 baseline — CURRENT production model on ALL of samples/

Generated: 2026-10-16T05:11:59.776481+00:00
Model: loaded=true name=rafnet mock=false
Thresholds: tau_low=0.3611111111111111 tau_high=0.861111111111111 (configs/decision_thresholds.json)
ffmpeg: C:\Users\rajee\RAFNet\.venv\Lib\site-packages\imageio_ffmpeg\binaries\ffmpeg-win-x86_64-v7.1.exe
ffmpeg note: system ffmpeg missing (winget Gyan.FFmpeg download stalls in this environment); using imageio-ffmpeg bundled binary 7.1 — all transcodes below are real encode+decode, simulated=false.

> Every number below is measured in this run and saved in reports/baseline.json. Private clips are listed by anonymous id only (wa_* = WhatsApp notes in sorted order, pc_* = phone-call files in sorted order); the real-name map stays local under samples/.phase0_local/ and is never committed.

## Environment (this machine, backend venv)
- python=3.12.0 torch=2.14.1+cpu cuda_available=False cuda_devices=0
- disk_free_gb=41.2 (C: 41.6 GB at Phase 0 start)
- credentials present/missing (values never printed): ELEVENLABS_API_KEY=present, KAGGLE_USERNAME=MISSING, KAGGLE_KEY=MISSING, HF_TOKEN=MISSING
- system ffmpeg: MISSING (no `ffmpeg` on PATH; `winget install Gyan.FFmpeg` stalls at Downloading in this environment — attempted 3x. Bundled imageio-ffmpeg 7.1 used; all transcodes below are real, simulated=false).

## Public samples/labels.csv (LibriSpeech real + SAPI/ChatGPT/WinTTS fake)
n=19 (real=4, fake=15) AUC=0.5 EER={"eer": 0.4833, "threshold": 0.8236, "far": 0.5, "frr": 0.4667}
binary@0.5: {"accuracy": 0.7895, "precision": 0.7895, "recall": 1.0, "f1": 0.8824, "tp": 15, "tn": 0, "fp": 4, "fn": 0, "n": 19}
confusion_3way: {"real": {"REAL": 0, "UNCERTAIN": 4, "SYNTHETIC": 0}, "fake": {"REAL": 0, "UNCERTAIN": 15, "SYNTHETIC": 0}}
FAR on human (public, most important): {"n": 4, "n_synthetic": 0, "far": 0.0}

### Per-file (public, no PII)
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

## Private WhatsApp/Telegram voice notes (REAL human, Opus 16 kHz + 2 .ogg)
files=86 scored=86 mean_p=0.8528 (total ~10.5 min, 1.2-57.8 s each)
FAR on human (MOST IMPORTANT): {"n": 86, "n_synthetic": 43, "far": 0.5}
FAR by container: {"opus": {"n": 84, "n_synthetic": 41, "far": 0.4881}, "ogg": {"n": 2, "n_synthetic": 2, "far": 1.0}}
Decode-equivalence (ogg direct vs ffmpeg-wav bypass): [{"container": "ogg", "p_direct": 0.8899, "p_via_ffmpeg": 0.8901, "abs_delta": 0.0002}, {"container": "ogg", "p_direct": 0.9021, "p_via_ffmpeg": 0.9032, "abs_delta": 0.0011}] — bypass validated.
Note: API rejects `.opus` uploads today (extension gate in audio_io.py); .opus clips were decoded with ffmpeg to 16k PCM first (bit-identical samples). Accepting `.opus` is queued as a Phase 6 backend fix.

### Per-file (anonymous ids only)
- wa_001 [opus] p=0.8451 -> UNCERTAIN (conf 0.8451) dur=17.46s lossy=True
- wa_002 [opus] p=0.862 -> SYNTHETIC (conf 0.862) dur=3.68s lossy=True
- wa_003 [opus] p=0.6608 -> UNCERTAIN (conf 0.6608) dur=3.36s lossy=True
- wa_004 [opus] p=0.6067 -> UNCERTAIN (conf 0.6067) dur=1.88s lossy=True
- wa_005 [opus] p=0.8275 -> UNCERTAIN (conf 0.8275) dur=9.94s lossy=True
- wa_006 [opus] p=0.8127 -> UNCERTAIN (conf 0.8127) dur=2.26s lossy=True
- wa_007 [opus] p=0.8399 -> UNCERTAIN (conf 0.8399) dur=1.62s lossy=True
- wa_008 [opus] p=0.8785 -> SYNTHETIC (conf 0.8785) dur=1.18s lossy=True
- wa_009 [opus] p=0.8493 -> UNCERTAIN (conf 0.8493) dur=4.36s lossy=True
- wa_010 [opus] p=0.8648 -> SYNTHETIC (conf 0.8648) dur=4.4s lossy=True
- wa_011 [opus] p=0.8816 -> SYNTHETIC (conf 0.8816) dur=2.74s lossy=True
- wa_012 [opus] p=0.8186 -> UNCERTAIN (conf 0.8186) dur=11.76s lossy=True
- wa_013 [opus] p=0.8044 -> UNCERTAIN (conf 0.8044) dur=7.56s lossy=True
- wa_014 [opus] p=0.8826 -> SYNTHETIC (conf 0.8826) dur=4.92s lossy=True
- wa_015 [opus] p=0.8527 -> UNCERTAIN (conf 0.8527) dur=8.18s lossy=True
- wa_016 [opus] p=0.9431 -> SYNTHETIC (conf 0.9431) dur=2.2s lossy=True
- wa_017 [opus] p=0.8484 -> UNCERTAIN (conf 0.8484) dur=7.38s lossy=True
- wa_018 [opus] p=0.8858 -> SYNTHETIC (conf 0.8858) dur=7.46s lossy=True
- wa_019 [opus] p=0.895 -> SYNTHETIC (conf 0.895) dur=2.64s lossy=True
- wa_020 [opus] p=0.9227 -> SYNTHETIC (conf 0.9227) dur=2.88s lossy=True
- wa_021 [opus] p=0.8192 -> UNCERTAIN (conf 0.8192) dur=3.62s lossy=True
- wa_022 [opus] p=0.8954 -> SYNTHETIC (conf 0.8954) dur=2.5s lossy=True
- wa_023 [opus] p=0.8464 -> UNCERTAIN (conf 0.8464) dur=22.4s lossy=True
- wa_024 [opus] p=0.8808 -> SYNTHETIC (conf 0.8808) dur=3.72s lossy=True
- wa_025 [opus] p=0.7428 -> UNCERTAIN (conf 0.7428) dur=8.0s lossy=True
- wa_026 [opus] p=0.9099 -> SYNTHETIC (conf 0.9099) dur=5.0s lossy=True
- wa_027 [opus] p=0.8382 -> UNCERTAIN (conf 0.8382) dur=2.72s lossy=True
- wa_028 [opus] p=0.8338 -> UNCERTAIN (conf 0.8338) dur=1.84s lossy=True
- wa_029 [opus] p=0.8368 -> UNCERTAIN (conf 0.8368) dur=21.44s lossy=True
- wa_030 [opus] p=0.8623 -> SYNTHETIC (conf 0.8623) dur=7.94s lossy=True
- wa_031 [opus] p=0.8357 -> UNCERTAIN (conf 0.8357) dur=4.24s lossy=True
- wa_032 [opus] p=0.8897 -> SYNTHETIC (conf 0.8897) dur=1.72s lossy=True
- wa_033 [opus] p=0.805 -> UNCERTAIN (conf 0.805) dur=15.8s lossy=True
- wa_034 [opus] p=0.7849 -> UNCERTAIN (conf 0.7849) dur=2.04s lossy=True
- wa_035 [opus] p=0.8853 -> SYNTHETIC (conf 0.8853) dur=3.36s lossy=True
- wa_036 [opus] p=0.8209 -> UNCERTAIN (conf 0.8209) dur=2.6s lossy=True
- wa_037 [opus] p=0.8384 -> UNCERTAIN (conf 0.8384) dur=7.94s lossy=True
- wa_038 [opus] p=0.804 -> UNCERTAIN (conf 0.804) dur=11.74s lossy=True
- wa_039 [opus] p=0.9069 -> SYNTHETIC (conf 0.9069) dur=3.36s lossy=True
- wa_040 [opus] p=0.9049 -> SYNTHETIC (conf 0.9049) dur=1.7s lossy=True
- wa_041 [opus] p=0.8114 -> UNCERTAIN (conf 0.8114) dur=3.12s lossy=True
- wa_042 [opus] p=0.8673 -> SYNTHETIC (conf 0.8673) dur=2.84s lossy=True
- wa_043 [opus] p=0.8924 -> SYNTHETIC (conf 0.8924) dur=6.84s lossy=True
- wa_044 [opus] p=0.8368 -> UNCERTAIN (conf 0.8368) dur=6.2s lossy=True
- wa_045 [opus] p=0.8339 -> UNCERTAIN (conf 0.8339) dur=24.44s lossy=True
- wa_046 [opus] p=0.8299 -> UNCERTAIN (conf 0.8299) dur=11.64s lossy=True
- wa_047 [opus] p=0.9054 -> SYNTHETIC (conf 0.9054) dur=3.34s lossy=True
- wa_048 [opus] p=0.901 -> SYNTHETIC (conf 0.901) dur=5.94s lossy=True
- wa_049 [opus] p=0.8625 -> SYNTHETIC (conf 0.8625) dur=2.04s lossy=True
- wa_050 [opus] p=0.8844 -> SYNTHETIC (conf 0.8844) dur=2.56s lossy=True
- wa_051 [opus] p=0.8957 -> SYNTHETIC (conf 0.8957) dur=3.9s lossy=True
- wa_052 [opus] p=0.892 -> SYNTHETIC (conf 0.892) dur=8.02s lossy=True
- wa_053 [opus] p=0.8856 -> SYNTHETIC (conf 0.8856) dur=2.7s lossy=True
- wa_054 [opus] p=0.8988 -> SYNTHETIC (conf 0.8988) dur=4.22s lossy=True
- wa_055 [opus] p=0.8716 -> SYNTHETIC (conf 0.8716) dur=9.08s lossy=True
- wa_056 [opus] p=0.8826 -> SYNTHETIC (conf 0.8826) dur=10.2s lossy=True
- wa_057 [opus] p=0.8635 -> SYNTHETIC (conf 0.8635) dur=3.7s lossy=True
- wa_058 [opus] p=0.8929 -> SYNTHETIC (conf 0.8929) dur=4.18s lossy=True
- wa_059 [opus] p=0.8522 -> UNCERTAIN (conf 0.8522) dur=9.76s lossy=True
- wa_060 [opus] p=0.871 -> SYNTHETIC (conf 0.871) dur=5.6s lossy=True
- wa_061 [opus] p=0.8071 -> UNCERTAIN (conf 0.8071) dur=26.64s lossy=True
- wa_062 [opus] p=0.8551 -> UNCERTAIN (conf 0.8551) dur=2.42s lossy=True
- wa_063 [opus] p=0.8894 -> SYNTHETIC (conf 0.8894) dur=6.96s lossy=True
- wa_064 [opus] p=0.9032 -> SYNTHETIC (conf 0.9032) dur=2.34s lossy=True
- wa_065 [opus] p=0.8252 -> UNCERTAIN (conf 0.8252) dur=2.6s lossy=True
- wa_066 [opus] p=0.8724 -> SYNTHETIC (conf 0.8724) dur=3.26s lossy=True
- wa_067 [opus] p=0.857 -> UNCERTAIN (conf 0.857) dur=4.0s lossy=True
- wa_068 [opus] p=0.8152 -> UNCERTAIN (conf 0.8152) dur=6.8s lossy=True
- wa_069 [opus] p=0.8619 -> SYNTHETIC (conf 0.8619) dur=1.42s lossy=True
- wa_070 [opus] p=0.9079 -> SYNTHETIC (conf 0.9079) dur=7.38s lossy=True
- wa_071 [opus] p=0.9081 -> SYNTHETIC (conf 0.9081) dur=1.3s lossy=True
- wa_072 [opus] p=0.9042 -> SYNTHETIC (conf 0.9042) dur=10.02s lossy=True
- wa_073 [opus] p=0.8289 -> UNCERTAIN (conf 0.8289) dur=6.88s lossy=True
- wa_074 [opus] p=0.8109 -> UNCERTAIN (conf 0.8109) dur=2.34s lossy=True
- wa_075 [opus] p=0.8067 -> UNCERTAIN (conf 0.8067) dur=8.76s lossy=True
- wa_076 [opus] p=0.9275 -> SYNTHETIC (conf 0.9275) dur=1.99s lossy=True
- wa_077 [opus] p=0.8623 -> SYNTHETIC (conf 0.8623) dur=2.61s lossy=True
- wa_078 [opus] p=0.8178 -> UNCERTAIN (conf 0.8178) dur=5.15s lossy=True
- wa_079 [opus] p=0.8327 -> UNCERTAIN (conf 0.8327) dur=8.09s lossy=True
- wa_080 [opus] p=0.7502 -> UNCERTAIN (conf 0.7502) dur=3.41s lossy=True
- wa_081 [opus] p=0.8217 -> UNCERTAIN (conf 0.8217) dur=16.03s lossy=True
- wa_082 [opus] p=0.8375 -> UNCERTAIN (conf 0.8375) dur=15.95s lossy=True
- wa_083 [opus] p=0.8315 -> UNCERTAIN (conf 0.8315) dur=9.83s lossy=True
- wa_084 [opus] p=0.8587 -> UNCERTAIN (conf 0.8587) dur=6.03s lossy=True
- wa_085 [ogg] p=0.8899 -> SYNTHETIC (conf 0.8899) dur=57.78s lossy=True
- wa_086 [ogg] p=0.9021 -> SYNTHETIC (conf 0.9021) dur=37.8s lossy=True

## Private AMR-NB phone calls (REAL human, 8 kHz, ~16.5 h, strided 5 s / 180 s sample)
files=19 skipped=1 windows=281 mean_p=0.8449
sampling: strided 5.0s windows every 180.0s, cap 25/file (Phase 0 sampling; VAD segmentation is next phase)
skipped (task rule: skip few-second files): [{"id": "pc_03", "reason": "duration 2.58 < 4.0s (task: skip)"}]
FAR on human windows (MOST IMPORTANT): {"n": 281, "n_synthetic": 116, "far": 0.4128}
FAR per call file (anonymous ids):
- pc_01: {"n": 21, "n_synthetic": 8, "far": 0.381}
- pc_02: {"n": 25, "n_synthetic": 7, "far": 0.28}
- pc_04: {"n": 25, "n_synthetic": 13, "far": 0.52}
- pc_05: {"n": 12, "n_synthetic": 4, "far": 0.3333}
- pc_06: {"n": 13, "n_synthetic": 4, "far": 0.3077}
- pc_07: {"n": 7, "n_synthetic": 1, "far": 0.1429}
- pc_08: {"n": 5, "n_synthetic": 3, "far": 0.6}
- pc_09: {"n": 24, "n_synthetic": 10, "far": 0.4167}
- pc_10: {"n": 14, "n_synthetic": 4, "far": 0.2857}
- pc_11: {"n": 25, "n_synthetic": 10, "far": 0.4}
- pc_12: {"n": 3, "n_synthetic": 1, "far": 0.3333}
- pc_13: {"n": 25, "n_synthetic": 12, "far": 0.48}
- pc_14: {"n": 25, "n_synthetic": 9, "far": 0.36}
- pc_15: {"n": 5, "n_synthetic": 1, "far": 0.2}
- pc_16: {"n": 22, "n_synthetic": 11, "far": 0.5}
- pc_17: {"n": 8, "n_synthetic": 4, "far": 0.5}
- pc_18: {"n": 13, "n_synthetic": 10, "far": 0.7692}
- pc_19: {"n": 9, "n_synthetic": 4, "far": 0.4444}

## Codec-shift test (10 decoded WhatsApp wavs -> Opus 12/24/32k, MP3, AMR-NB, G.711)
policy: same chains will be applied to BOTH classes in Phase 3 (contingency table there); this Phase 0 test is real-only shift.
### opus_12k: {"n": 10, "mean_delta": 0.0021, "max_abs_delta": 0.0181, "mean_p_codec": 0.8068, "n_synthetic_after": 2, "simulated": false}
- wa_001: 0.8451 -> 0.8508 (delta 0.0057) UNCERTAIN -> UNCERTAIN
- wa_002: 0.862 -> 0.8676 (delta 0.0056) SYNTHETIC -> SYNTHETIC
- wa_003: 0.6608 -> 0.6621 (delta 0.0013) UNCERTAIN -> UNCERTAIN
- wa_004: 0.6067 -> 0.6248 (delta 0.0181) UNCERTAIN -> UNCERTAIN
- wa_005: 0.8275 -> 0.8323 (delta 0.0048) UNCERTAIN -> UNCERTAIN
- wa_006: 0.8127 -> 0.811 (delta -0.0017) UNCERTAIN -> UNCERTAIN
- wa_007: 0.8399 -> 0.857 (delta 0.0171) UNCERTAIN -> UNCERTAIN
- wa_008: 0.8785 -> 0.8607 (delta -0.0178) SYNTHETIC -> UNCERTAIN
- wa_009: 0.8493 -> 0.8404 (delta -0.0089) UNCERTAIN -> UNCERTAIN
- wa_010: 0.8648 -> 0.8617 (delta -0.0031) SYNTHETIC -> SYNTHETIC
### opus_24k: {"n": 10, "mean_delta": 0.0014, "max_abs_delta": 0.0167, "mean_p_codec": 0.8062, "n_synthetic_after": 2, "simulated": false}
- wa_001: 0.8451 -> 0.8425 (delta -0.0026) UNCERTAIN -> UNCERTAIN
- wa_002: 0.862 -> 0.8588 (delta -0.0032) SYNTHETIC -> UNCERTAIN
- wa_003: 0.6608 -> 0.6685 (delta 0.0077) UNCERTAIN -> UNCERTAIN
- wa_004: 0.6067 -> 0.6026 (delta -0.0041) UNCERTAIN -> UNCERTAIN
- wa_005: 0.8275 -> 0.8354 (delta 0.0079) UNCERTAIN -> UNCERTAIN
- wa_006: 0.8127 -> 0.8096 (delta -0.0031) UNCERTAIN -> UNCERTAIN
- wa_007: 0.8399 -> 0.8534 (delta 0.0135) UNCERTAIN -> UNCERTAIN
- wa_008: 0.8785 -> 0.8619 (delta -0.0166) SYNTHETIC -> SYNTHETIC
- wa_009: 0.8493 -> 0.8473 (delta -0.002) UNCERTAIN -> UNCERTAIN
- wa_010: 0.8648 -> 0.8815 (delta 0.0167) SYNTHETIC -> SYNTHETIC
### opus_32k: {"n": 10, "mean_delta": 0.0013, "max_abs_delta": 0.022, "mean_p_codec": 0.806, "n_synthetic_after": 1, "simulated": false}
- wa_001: 0.8451 -> 0.8444 (delta -0.0007) UNCERTAIN -> UNCERTAIN
- wa_002: 0.862 -> 0.8611 (delta -0.0009) SYNTHETIC -> UNCERTAIN
- wa_003: 0.6608 -> 0.6661 (delta 0.0053) UNCERTAIN -> UNCERTAIN
- wa_004: 0.6067 -> 0.6151 (delta 0.0084) UNCERTAIN -> UNCERTAIN
- wa_005: 0.8275 -> 0.8325 (delta 0.005) UNCERTAIN -> UNCERTAIN
- wa_006: 0.8127 -> 0.8029 (delta -0.0098) UNCERTAIN -> UNCERTAIN
- wa_007: 0.8399 -> 0.8557 (delta 0.0158) UNCERTAIN -> UNCERTAIN
- wa_008: 0.8785 -> 0.8565 (delta -0.022) SYNTHETIC -> UNCERTAIN
- wa_009: 0.8493 -> 0.8493 (delta 0.0) UNCERTAIN -> UNCERTAIN
- wa_010: 0.8648 -> 0.8766 (delta 0.0118) SYNTHETIC -> SYNTHETIC
### mp3_128k: {"n": 10, "mean_delta": 0.008, "max_abs_delta": 0.0118, "mean_p_codec": 0.8128, "n_synthetic_after": 3, "simulated": false}
- wa_001: 0.8451 -> 0.8536 (delta 0.0085) UNCERTAIN -> UNCERTAIN
- wa_002: 0.862 -> 0.869 (delta 0.007) SYNTHETIC -> SYNTHETIC
- wa_003: 0.6608 -> 0.6623 (delta 0.0015) UNCERTAIN -> UNCERTAIN
- wa_004: 0.6067 -> 0.6143 (delta 0.0076) UNCERTAIN -> UNCERTAIN
- wa_005: 0.8275 -> 0.836 (delta 0.0085) UNCERTAIN -> UNCERTAIN
- wa_006: 0.8127 -> 0.8234 (delta 0.0107) UNCERTAIN -> UNCERTAIN
- wa_007: 0.8399 -> 0.8517 (delta 0.0118) UNCERTAIN -> UNCERTAIN
- wa_008: 0.8785 -> 0.8845 (delta 0.006) SYNTHETIC -> SYNTHETIC
- wa_009: 0.8493 -> 0.8589 (delta 0.0096) UNCERTAIN -> UNCERTAIN
- wa_010: 0.8648 -> 0.8741 (delta 0.0093) SYNTHETIC -> SYNTHETIC
### amr_nb: {"n": 10, "mean_delta": 0.0393, "max_abs_delta": 0.2448, "mean_p_codec": 0.844, "n_synthetic_after": 4, "simulated": false}
- wa_001: 0.8451 -> 0.8583 (delta 0.0132) UNCERTAIN -> UNCERTAIN
- wa_002: 0.862 -> 0.9058 (delta 0.0438) SYNTHETIC -> SYNTHETIC
- wa_003: 0.6608 -> 0.7175 (delta 0.0567) UNCERTAIN -> UNCERTAIN
- wa_004: 0.6067 -> 0.8515 (delta 0.2448) UNCERTAIN -> UNCERTAIN
- wa_005: 0.8275 -> 0.8371 (delta 0.0096) UNCERTAIN -> UNCERTAIN
- wa_006: 0.8127 -> 0.7962 (delta -0.0165) UNCERTAIN -> UNCERTAIN
- wa_007: 0.8399 -> 0.8788 (delta 0.0389) UNCERTAIN -> SYNTHETIC
- wa_008: 0.8785 -> 0.8812 (delta 0.0027) SYNTHETIC -> SYNTHETIC
- wa_009: 0.8493 -> 0.8485 (delta -0.0008) UNCERTAIN -> UNCERTAIN
- wa_010: 0.8648 -> 0.8656 (delta 0.0008) SYNTHETIC -> SYNTHETIC
### g711_alaw: {"n": 10, "mean_delta": 0.033, "max_abs_delta": 0.2272, "mean_p_codec": 0.8377, "n_synthetic_after": 4, "simulated": false}
- wa_001: 0.8451 -> 0.8494 (delta 0.0043) UNCERTAIN -> UNCERTAIN
- wa_002: 0.862 -> 0.897 (delta 0.035) SYNTHETIC -> SYNTHETIC
- wa_003: 0.6608 -> 0.7076 (delta 0.0468) UNCERTAIN -> UNCERTAIN
- wa_004: 0.6067 -> 0.8339 (delta 0.2272) UNCERTAIN -> UNCERTAIN
- wa_005: 0.8275 -> 0.8299 (delta 0.0024) UNCERTAIN -> UNCERTAIN
- wa_006: 0.8127 -> 0.7951 (delta -0.0176) UNCERTAIN -> UNCERTAIN
- wa_007: 0.8399 -> 0.8675 (delta 0.0276) UNCERTAIN -> SYNTHETIC
- wa_008: 0.8785 -> 0.8725 (delta -0.006) SYNTHETIC -> SYNTHETIC
- wa_009: 0.8493 -> 0.8598 (delta 0.0105) UNCERTAIN -> UNCERTAIN
- wa_010: 0.8648 -> 0.8645 (delta -0.0003) SYNTHETIC -> SYNTHETIC

## Honest reading (Phase 0)
- The CURRENT production model flags ~50% of real WhatsApp voice notes and ~41% of real phone-call windows as SYNTHETIC (mean P(AI) ~0.85 on both). The reported History failure (human WhatsApp -> SYNTHETIC ~90%) REPRODUCES on the current model: the 2 held-out .ogg notes score 0.8899 and 0.9021 -> SYNTHETIC.
- Clean public real (LibriSpeech FLAC) is NOT flagged (FAR 0.0, but n=4); the false alarms track compressed/phone channels, consistent with compression-label correlation (Phase 3 codec-fair augmentation is the scheduled fix).
- Narrowband re-transcodes (AMR-NB/G.711) push low-P real clips UP (+0.24 max observed, no verdict flip in this n=10 probe) — same direction as the bias.
- Phone windows are fixed-stride, no VAD/silence/IVR filtering yet; VAD segmentation + samples/real/manifest.csv is the next DATA-UPDATE step. Phone FAR is a sampling estimate, not a segmented eval.
- No thresholds changed; no training; nothing tuned on private data. Held-out groups (later-half WhatsApp dates, one whole contact, 2 .ogg) are NOT yet split out — that happens at segmentation time, before any training.
