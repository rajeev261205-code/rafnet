# Phase 4.2c stacked ensemble (fit on VAL only)

features=['head', 'rafnet', 'mo', 'bisher']
coef=[5.9474, 0.0017, 0.5055, -1.0903] intercept=-3.6223 T=0.2
band=[0.45, 0.55] selective=0.9949 coverage=0.9899
VAL AUC=1.0 EER={"eer": 0.0, "threshold": 0.0334, "far": 0.0, "frr": 0.0} FAR@0.5=0.0 binary@0.5={"accuracy": 0.9899, "precision": 1.0, "recall": 0.95, "f1": 0.9744, "tp": 38, "tn": 159, "fp": 0, "fn": 2, "n": 199}
NLL T=1: 0.152 -> T=0.2: 0.0368

Stack fit on VAL only (speaker-disjoint); TEST never read for fitting. Band from val selective accuracy >= 0.95.

VAL fake shares train generators (in-distribution); unseen generators + private groups live in TEST (Phase 5).
