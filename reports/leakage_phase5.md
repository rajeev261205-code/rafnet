# Leakage / shortcut audit (VAL AUC 1.0 sanity check)

Generated: 2026-10-26T14:49:21.258286+00:00

## Split hygiene
- clip-id overlap: {"train_val": 0, "train_test": 0, "val_test": 0}
- human speakers: train=2155 val=462 test=464; train-val overlap=0, train-test overlap=0
- fake generators (voice ids): train=13 val=13 test=3; train-val overlap=13, train-test overlap=0
- VAL fake generators shared with TRAIN: gen-crudeVC:ps, gen-crudeVC:ts, gen-edge:en-IN-NeerjaNeural, gen-edge:en-IN-PrabhatNeural, gen-edge:hi-IN-MadhurNeural, gen-edge:hi-IN-SwaraNeural, gen-edge:ta-IN-PallaviNeural, gen-piper:en-amy, gen-piper:en-ryan, gen-piper:hi-pratham, gen-piper:hi-priyamvada, gen-sapi:david, gen-sapi:zira
- TEST unseen fake generators: gen-edge:ta-IN-ValluvarNeural, gen-elevenlabs, gen-piper:hi-rohan

## Dataset id vs label (metadata shortcut, not usable at inference)
- val: fake-crudeVC=0H/116F, fake-edge=0H/45F, fake-piper=0H/88F, fake-sapi=0H/30F, fleurs-en=263H/0F, fleurs-hi=61H/0F, fleurs-ta=33H/0F, slr118=373H/0F, slr12=195H/0F, slr65=169H/0F
- test: fake-edge=0H/60F, fake-elevenlabs=0H/25F, fake-piper=0H/150F, fleurs-en=213H/0F, fleurs-hi=63H/0F, fleurs-ta=37H/0F, private-phone=40H/0F, private-whatsapp=30H/0F, slr118=369H/0F, slr12=210H/0F, slr65=150H/0F

## Codec vs label (file suffix) 
- val: wav=1094H/279F
- test: mp3=0H/25F, wav=1112H/210F

## Trivial audio features on VAL (direction-normalised separability)
| feature | AUC | separability | human mean | fake mean |
| --- | --- | --- | --- | --- |
| noise_floor_dbfs | 0.6973 | 0.6973 | -62.8579 | -54.5446 |
| rms_dbfs | 0.6646 | 0.6646 | -25.136 | -21.2322 |
| peak | 0.6448 | 0.6448 | 0.5565 | 0.705 |
| snr_db | 0.3563 | 0.6437 | 42.3246 | 37.6349 |
| crest_db | 0.3679 | 0.6321 | 17.9512 | 16.7963 |
| duration_s | 0.3844 | 0.6156 | 7.5187 | 6.7797 |
| flatness | 0.5215 | 0.5215 | 0.0264 | 0.0246 |
| rolloff_hz | 0.4969 | 0.5031 | 1269.1973 | 1246.6953 |
| zcr | 0.5 | 0.5 | 0.117 | 0.1159 |

## Shortcut model (logistic regression on TRAIN trivial audio features only; no dataset id)

| eval set | n | AUC | FAR human @0.5 | detection fake @0.5 |
| --- | --- | --- | --- | --- |
| val | 1373 | 0.6768 | 0.3227 | 0.5735 |
| test | 1347 | 0.7733 | 0.4308 | 0.8553 |

## Reading
- dataset id (source) is label-defining by construction: human clips come from human datasets and synthetic clips from fake datasets, so it must never be used as an inference feature.
- VAL fake generators are the SAME voices/families as TRAIN (only human speakers are disjoint); VAL AUC 1.0 therefore measures in-distribution generator memorisation, not generalisation.
- TEST fake generators and private groups are disjoint from TRAIN; the stack's per-group TEST numbers are the honest generalisation measure.
