# Phase 4.2b head (WavLM embeddings, trained on TRAIN only)

train rows=13018 (real=9834, fake=3184); val rows=1373 (clean)
best epoch=1 val AUC=1.0
band=[0.45, 0.75] (val-only selective>=0.95, max coverage, narrowest)
VAL clean: AUC=1.0 EER={"eer": 0.0, "threshold": 0.7989, "far": 0.0, "frr": 0.0} FAR@0.5=0.0183 binary@0.5={"accuracy": 0.9854, "precision": 0.9331, "recall": 1.0, "f1": 0.9654, "tp": 279, "tn": 1074, "fp": 20, "fn": 0, "n": 1373}
VAL clean confusion(band): {"real": {"REAL": 1065, "UNCERTAIN": 29, "SYNTHETIC": 0}, "fake": {"REAL": 0, "UNCERTAIN": 0, "SYNTHETIC": 279}}
VAL aug0 (robustness, NOT tuned): {"n": 1373, "auc": 0.971, "eer": {"eer": 0.0896, "threshold": 0.6628, "far": 0.0896, "frr": 0.0896}, "binary_at_0.5": {"accuracy": 0.8835, "precision": 0.6413, "recall": 0.9677, "f1": 0.7714, "tp": 270, "tn": 943, "fp": 151, "fn": 9, "n": 1373}, "far_human_at_0.5": 0.138, "band": [0.45, 0.75], "strict_accuracy": 0.8492352512745812, "confusion": {"real": {"REAL": 925, "UNCERTAIN": 105, "SYNTHETIC": 64}, "fake": {"REAL": 7, "UNCERTAIN": 31, "SYNTHETIC": 241}}, "coverage": 0.9009, "selective_accuracy": 0.9426, "n_certain": 1237, "n_total": 1373}

Honest note: val fake shares train generators (in-distribution); the unseen generators live in TEST and are measured only in Phase 5.
VAL is speaker-disjoint from train; TEST (incl. private groups + unseen generators) is never read here.
