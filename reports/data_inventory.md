# Data inventory (existing holdings, no downloads)

Generated: 2026-10-15T14:33:34.153070+00:00

data/manifest.csv: 2629 rows, splits={'train': 2206, 'test': 174, 'val': 249}, labels={'HUMAN_LIVE': 2620, 'AI_TTS': 9}, sources={'librispeech': 2620, 'win-sapi': 9}
speakers: 41; speakers spanning splits (LEAKAGE): 0
  none leaked across splits (but single-source REAL-only).
generators: {"('human', 'librispeech')": 2620, "('microsoft-sapi-like', 'win-sapi')": 9}
samples/labels.csv: 19 rows; ('real', 'clean_human_librispeech')=4; ('fake', 'windows_sapi_tts')=4; ('fake', 'assistant_voice_chatgpt')=1; ('fake', 'windows_tts')=1; ('fake', 'windows_sapi_tts_extended')=9
samples missing files: none
data/raw files on disk: 2713

## Gaps vs task brief
- REAL: no Indian English/Hindi/Tamil, no WhatsApp/Opus voice notes, no phone-line (8 kHz), no laptop-mic/noisy-room sets — only LibriSpeech clean + samples/.
- FAKE: only SAPI David/Zira + 1 Windows-TTS + 1 ChatGPT loopback; no open-source TTS/VC diversity (Bark/Coqui/XTTS/YourTTS/RVC/Seed-VC), no commercial TTS (no user keys), no voice conversion at all.
- PUBLIC SETS: none downloaded yet (ASVspoof 2019/2021, In-the-Wild, WaveFake, MLAAD, Common Voice). See reports/data_plan.md for per-set license/size/approval.
- data/manifest.csv is REAL-ONLY (LibriSpeech) — unusable as a spoof train manifest until fake rows are added and splits rebuilt speaker-/generator-disjoint.
