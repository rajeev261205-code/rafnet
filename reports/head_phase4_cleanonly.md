# Phase 4.2b head (WavLM embeddings, trained on TRAIN only)

train rows=6509 (real=4917, fake=1592); val rows=1373 (clean)
best epoch=1 val AUC=0.9999
band=[0.45, 0.6] (val-only selective>=0.95, max coverage, narrowest)
VAL clean: AUC=0.9999 EER={"eer": 0.0032, "threshold": 0.532, "far": 0.0027, "frr": 0.0036} FAR@0.5=0.0064 binary@0.5={"accuracy": 0.9949, "precision": 0.9755, "recall": 1.0, "f1": 0.9876, "tp": 279, "tn": 1087, "fp": 7, "fn": 0, "n": 1373}
VAL clean confusion(band): {"real": {"REAL": 1079, "UNCERTAIN": 15, "SYNTHETIC": 0}, "fake": {"REAL": 0, "UNCERTAIN": 16, "SYNTHETIC": 263}}
VAL aug0: n/a

Honest note: val fake shares train generators (in-distribution); the unseen generators live in TEST and are measured only in Phase 5.
VAL is speaker-disjoint from train; TEST (incl. private groups + unseen generators) is never read here.
