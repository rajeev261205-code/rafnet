# Phase 4.1 zero-shot candidates on NEW val split (speaker-disjoint)

val subset n=500 (250 real + 250 fake); offline, cached weights.

## mo-thecreator (apache-2.0 (Hub tag))
params=94.6M latency=537.8ms metrics={"n": 500, "errors": 0, "auc": 0.6597, "eer": {"eer": 0.356, "threshold": 0.9999, "far": 0.38, "frr": 0.332}, "far_human": 0.416, "binary_at_0.5": {"accuracy": 0.666, "precision": 0.6426, "recall": 0.748, "f1": 0.6913, "tp": 187, "tn": 146, "fp": 104, "fn": 63, "n": 500}} verdict=KEEP

## bisher (apache-2.0 (Hub tag))
params=94.6M latency=941.3ms metrics={"n": 500, "errors": 0, "auc": 0.6215, "eer": {"eer": 0.426, "threshold": 0.0034, "far": 0.428, "frr": 0.424}, "far_human": 0.336, "binary_at_0.5": {"accuracy": 0.532, "precision": 0.5435, "recall": 0.4, "f1": 0.4608, "tp": 100, "tn": 166, "fp": 84, "fn": 150, "n": 500}} verdict=KEEP

