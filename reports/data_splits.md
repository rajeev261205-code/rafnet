# Phase 2 splits (speaker-/generator-disjoint + my_samples)

rows=419 ('my_samples', 'AI_TTS')=15, ('my_samples', 'HUMAN_LIVE')=4, ('test', 'AI_VOICE_CLONE_OR_CONVERSION')=40, ('test', 'HUMAN_LIVE')=40, ('train', 'AI_VOICE_CLONE_OR_CONVERSION')=120, ('train', 'HUMAN_LIVE')=120, ('val', 'AI_VOICE_CLONE_OR_CONVERSION')=40, ('val', 'HUMAN_LIVE')=40

train speakers: 25, val: 5, test: 7 (reserved ['spk-260', 'spk-4446', 'spk-5142', 'spk-6930'] excluded).
train fake settings: ps_up/ps_down; val: ts_slow; test: ts_fast (sub-setting split inside one crude family — stopgap, stated).
my_samples (19 files, SAPI/ChatGPT/Windows-TTS generators) appears in NO train/val/test row.
STOPGAP WARNING: train fake diversity is one crude librosa VC family. Real generator diversity (open TTS/VC, ASVspoof/WaveFake/MLAAD) is Phase-3-notebook + user-clip work (see data_plan.md).
