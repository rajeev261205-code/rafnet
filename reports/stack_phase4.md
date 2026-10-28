# Phase 4.2c stacked ensemble (fit on VAL only)

features=['head', 'rafnet', 'mo', 'bisher']
coef=[8.9958, -0.042, 1.3652, 0.2068] intercept=-6.5677 T=0.3
band=[0.45, 0.925] selective=1.0 coverage=0.9971
VAL AUC=1.0 EER={"eer": 0.0005, "threshold": 0.9058, "far": 0.0009, "frr": 0.0} FAR@0.5=0.0018 binary@0.5={"accuracy": 0.9985, "precision": 0.9929, "recall": 1.0, "f1": 0.9964, "tp": 279, "tn": 1092, "fp": 2, "fn": 0, "n": 1373}
NLL T=1: 0.0241 -> T=0.3: 0.0034

Stack fit on VAL only (speaker-disjoint); TEST never read for fitting. Band from val selective accuracy >= 0.95.

VAL fake shares train generators (in-distribution); unseen generators + private groups live in TEST (Phase 5).
