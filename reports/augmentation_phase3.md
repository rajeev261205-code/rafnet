# Phase 3 augmentation check — compression independent of label

sampler chains: 10000/class (distribution proof); audio chains executed: 60 (failures=0, nan=0)

## Codec vs label (10k sampled chains per class)
|  | none | opus_12k | opus_24k | opus_32k | amr_nb | g711 | mp3_64k | mp3_128k | aac_32k |
|real| 1057 | 1094 | 1151 | 1155 | 1114 | 1107 | 1127 | 1100 | 1095 |
|fake| 1129 | 1133 | 1150 | 1072 | 1058 | 1102 | 1087 | 1111 | 1158 |
chi2=10.142 p=0.2552 independent=True

## Noise vs label
real [4008, 5992] fake [4070, 5930] chi2=0.798 p=0.3716 independent=True

chi-square p>0.05 => augmentation type independent of label (label-blind by construction; this is the measured proof). Same CODECS/RATES/NOISE distribution feeds Phase-4 training batches.
