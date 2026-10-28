# Phase 5 — honest TEST evaluation (frozen VAL-fit stack)

Generated: 2026-10-26T14:50:41.685598+00:00
Model: wavlm-head + rafnet + mo + bisher (stacked); stack frozen from VAL (models/stack_phase4.json).
Band: [0.45, 0.925] (UNCERTAIN between). Temperature T=0.3.

> TEST was never used for training, tuning, calibration, or band selection. All thresholds below are the frozen 0.5 decision boundary (no threshold was adjusted on any test group).

## 1. False-alarm rate — human audio called AI (FAR @0.5, lower is better)

| held-out set | segments | FAR | baseline | baseline detail |
| --- | --- | --- | --- | --- |
| held-out WhatsApp | 30 | **0.0333** | 0.5 | 43/86 whole notes (Phase 0) |
| held-out phone calls | 40 | **0.025** | 0.41 | 116/281 5-s windows (Phase 0) |
| the 2 .ogg files | 13 | **0.0** | 1.0 | both whole files SYNTHETIC (0.8899 / 0.9021, Phase 0) |

The two .ogg original notes were split into 13 segments by the private preprocessing (both originals are represented). Baseline numbers are from `reports/baseline.md` (Phase 0) and are file/window-level, not segment-level.

## 2. Detection on unseen generators and ElevenLabs (fixed @0.5)

| generator (unseen in train+val) | clips | detected | median P(AI) | band confusion |
| --- | --- | --- | --- | --- |
| gen-edge:ta-IN-ValluvarNeural | 60 | **0.75** | 0.7938 | {"real": {"REAL": 0, "UNCERTAIN": 0, "SYNTHETIC": 0}, "fake": {"REAL": 14, "UNCERTAIN": 24, "SYNTHETIC": 22}} |
| gen-piper:hi-rohan | 150 | **1.0** | 1.0 | {"real": {"REAL": 0, "UNCERTAIN": 0, "SYNTHETIC": 0}, "fake": {"REAL": 0, "UNCERTAIN": 1, "SYNTHETIC": 149}} |
| gen-elevenlabs | 25 | **0.8** | 0.9995 | {"real": {"REAL": 0, "UNCERTAIN": 0, "SYNTHETIC": 0}, "fake": {"REAL": 4, "UNCERTAIN": 4, "SYNTHETIC": 17}} |

`detected` = 1 - miss rate at 0.5. AUC is undefined for these single-class (synthetic-only) groups. Band confusion is REAL/UNCERTAIN/SYNTHETIC counts per true class; it is reported, not tuned. The stack still separates the private human groups (FAR 2.5-3.3%) and the unseen generators (75-100% detected), but Valluvar and ElevenLabs are visibly weaker than the in-distribution generators.

## 3. Why VAL looked perfect — leakage / shortcut audit

- **Speakers**: human speakers train=2155 / val=462 / test=464; train-val overlap=**0**, train-test overlap=**0** (disjoint).
- **Trigger for this audit**: the VAL-fit stack reported AUC=1.0 / FAR=0.0018, which is implausible for a generalising detector.
- **Generators are NOT disjoint**: VAL uses the *same* 13 fake voices/families as TRAIN (train-val generator overlap **13** of 13 val generators). VAL AUC 1.0 is therefore in-distribution generator memorisation, not generalisation.
- **Unseen on TEST**: gen-edge:ta-IN-ValluvarNeural, gen-elevenlabs, gen-piper:hi-rohan (train-test generator overlap **0**).
- **Dataset id is label-defining** (human clips come from human datasets, fakes from synthetic datasets). It is a metadata shortcut and is never used as an inference feature.
- **Codec** does not explain VAL: every VAL clip is a `.wav`, so the file suffix carries no class separation (only the TEST ElevenLabs clips are `.mp3`).
- **Trivial audio features on VAL** (direction-normalised separability, 0.5 = chance): noise_floor_dbfs=0.6973, rms_dbfs=0.6646, peak=0.6448, snr_db=0.6437, crest_db=0.6321, duration_s=0.6156. No single feature comes close to separating the classes.
- **Shortcut model** (logistic regression on TRAIN duration, loudness, bandwidth, noise floor, flatness, zcr only): val AUC=0.6768 FAR=0.3227; test AUC=0.7733 FAR=0.4308. It is far from VAL AUC 1.0, so cheap acoustic shortcuts do not explain the VAL result.
- **Interpretation**: VAL is easy because its fake generators are seen in TRAIN (the head has 8.9958 of the stack weight, so learned generator/voice artefacts dominate), not because of duration, loudness, bandwidth, noise floor, or codec. The honest generalisation measure is this TEST section, especially the unseen-generator and private rows above.

## 4. Overall TEST and every group

- **Overall FAR on human audio @0.5: 0.0045** (n_human=1112)
- FRR on synthetic audio @0.5: 0.0851 (n_fake=235)
- Coverage inside band: 0.977 | selective accuracy: 0.984
- Overall: AUC=0.9978 EER={"eer": 0.0136, "threshold": 0.011, "far": 0.0144, "frr": 0.0128} acc=0.9814 balanced=0.9552
- Band confusion: {"real": {"REAL": 1107, "UNCERTAIN": 2, "SYNTHETIC": 3}, "fake": {"REAL": 18, "UNCERTAIN": 29, "SYNTHETIC": 188}}

- **fleurs-en** (human, n=213): FAR=0.0 AUC=None conf={"real": {"REAL": 213, "UNCERTAIN": 0, "SYNTHETIC": 0}, "fake": {"REAL": 0, "UNCERTAIN": 0, "SYNTHETIC": 0}}
- **fleurs-hi** (human, n=63): FAR=0.0 AUC=None conf={"real": {"REAL": 63, "UNCERTAIN": 0, "SYNTHETIC": 0}, "fake": {"REAL": 0, "UNCERTAIN": 0, "SYNTHETIC": 0}}
- **fleurs-ta** (human, n=37): FAR=0.0 AUC=None conf={"real": {"REAL": 37, "UNCERTAIN": 0, "SYNTHETIC": 0}, "fake": {"REAL": 0, "UNCERTAIN": 0, "SYNTHETIC": 0}}
- **private-phone** (human, n=40): FAR=0.025 AUC=None conf={"real": {"REAL": 39, "UNCERTAIN": 0, "SYNTHETIC": 1}, "fake": {"REAL": 0, "UNCERTAIN": 0, "SYNTHETIC": 0}}
- **private-whatsapp** (human, n=30): FAR=0.0333 AUC=None conf={"real": {"REAL": 29, "UNCERTAIN": 0, "SYNTHETIC": 1}, "fake": {"REAL": 0, "UNCERTAIN": 0, "SYNTHETIC": 0}}
- **slr118** (human, n=369): FAR=0.0 AUC=None conf={"real": {"REAL": 369, "UNCERTAIN": 0, "SYNTHETIC": 0}, "fake": {"REAL": 0, "UNCERTAIN": 0, "SYNTHETIC": 0}}
- **slr12** (human, n=210): FAR=0.0048 AUC=None conf={"real": {"REAL": 209, "UNCERTAIN": 1, "SYNTHETIC": 0}, "fake": {"REAL": 0, "UNCERTAIN": 0, "SYNTHETIC": 0}}
- **slr65** (human, n=150): FAR=0.0133 AUC=None conf={"real": {"REAL": 148, "UNCERTAIN": 1, "SYNTHETIC": 1}, "fake": {"REAL": 0, "UNCERTAIN": 0, "SYNTHETIC": 0}}
- gen-edge:ta-IN-ValluvarNeural (synthetic, n=60): miss=0.25 AUC=None conf={"real": {"REAL": 0, "UNCERTAIN": 0, "SYNTHETIC": 0}, "fake": {"REAL": 14, "UNCERTAIN": 24, "SYNTHETIC": 22}}
- gen-elevenlabs (synthetic, n=25): miss=0.2 AUC=None conf={"real": {"REAL": 0, "UNCERTAIN": 0, "SYNTHETIC": 0}, "fake": {"REAL": 4, "UNCERTAIN": 4, "SYNTHETIC": 17}}
- gen-piper:hi-rohan (synthetic, n=150): miss=0.0 AUC=None conf={"real": {"REAL": 0, "UNCERTAIN": 0, "SYNTHETIC": 0}, "fake": {"REAL": 0, "UNCERTAIN": 1, "SYNTHETIC": 149}}

## 5. Reliability (calibrated P(AI) bins vs binary accuracy on TEST)
| bin | n | accuracy | avg_p |
| --- | --- | --- | --- |
| 0.0-0.1 | 1113 | 0.991 | 0.0007 |
| 0.1-0.3 | 6 | 0.6667 | 0.1799 |
| 0.3-0.5 | 8 | 0.0 | 0.4174 |
| 0.5-0.7 | 11 | 0.9091 | 0.584 |
| 0.7-0.9 | 15 | 0.9333 | 0.8069 |
| 0.9-1.0 | 194 | 0.9845 | 0.9956 |

## Coverage of the score cache (honesty)
- test rows: 1347; fully scored: 1347; incomplete (excluded from metrics): 0
- missing per model: none

## Honest reading
- TEST was scored once with artifacts frozen from VAL; no threshold, weight, or temperature was chosen or revised after seeing TEST.
- Private clips never left this PC; groups are anonymous labels.
- Private test n=70 (small-n snapshot, not a benchmark).
- Unseen generators (elevenlabs, piper:hi-rohan, edge:ta-ValluvarNeural) were held out of train AND val by construction.
- Baseline FARs in the headline are Phase 0 file/window measurements; the current numbers are 4-6 s segment measurements, so granularity and sample size differ.
