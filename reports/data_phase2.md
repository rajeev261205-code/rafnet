# Phase 2 — fake (AI) speech (2106 clips)

Local manifests: `data/manifest_phase2.csv` (2081 local, git-ignored),
`data/manifest_phase2_elevenlabs.csv` (25, git-ignored).
Audio: `data/phase2/fake/<engine>/*.wav|mp3` (16 kHz mono except EL mp3,
git-ignored). Median duration 6.3 s. Every file labeled with generator.

## Train fakes (1871, 13 voices / 4 engines, en+hi+ta + mixed)
| engine | voices | clips | notes |
|---|---|---|---|
| edge-tts | en-IN Neerja/Prabhat, hi-IN Swara/Madhur, ta-IN Pallavi | 360 | service synthesis, ±10% rate varied, scam+neutral texts |
| Piper local | en amy/ryan, hi pratham/priyamvada (+rohan unseen) | 600 train | onnx from rhasspy/piper-voices (~0.3 GB, in 15 GB budget) |
| Windows SAPI | David + Zira | 200 | offline, existing tool pattern reused |
| crudeVC | pitch-shift / time-stretch of Phase-1 public real | 771 | librosa, labeled mixed-lang |

## Unseen-generator TEST (235, never train/tune/select)
- ElevenLabs 25 (en 13 / hi 6 / ta 6): built-in voices only (Alice, Adam,
  Bella), multilingual-v2, scam+neutral scripts. Quota: account 0 -> 1919 of
  10000/mo (cap was 25 clips / 3000 chars; billed 1919). Cached locally,
  evaluation only, never redistributed. No cloning, nothing private sent.
- piper:hi-rohan 150, edge:ta-ValluvarNeural 60 (whole voices held out).

## Skipped (with reason)
- XTTS: v2 supports neither Hindi nor Tamil + ~2 GB + slow CPU inference;
  compensated with 5 Piper + 6 edge voices.
- Public fake sets: ASVspoof (research agreement + gated), WaveFake/MLAAD HF
  mirrors (401, need HF_TOKEN), In-the-Wild (gated), all Kaggle (no creds).
- Honest shortfall: 2106 fakes vs 4000 target and 5 train engines vs 10+
  generators — blocked by gated creds + size caps, stated plainly. Next data:
  ASVspoof/WaveFake subsets via credentialed pull when available.

## Budget
Phase 2 downloads (Piper voices ~0.3 GB) + Phase 1 (4.59 GB) = ~4.9/15 GB.
C: 37.4 GB free (10 GB floor held throughout; per-engine disk guards).
