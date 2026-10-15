# Phase 3 stacking (fit on VAL only)

features=['mo-thecreator/Deepfake-audio-detection', 'Bisher/wav2vec2_ASV_deepfake_audio_detection', 'rafnet']
coef=[3.4875, 1.3679, 0.0719] intercept=-1.4734 T=0.5
band=[0.05, 0.55] selective=1.0 coverage=0.4375
VAL binary@0.5: {"accuracy": 0.9375, "precision": 1.0, "recall": 0.875, "f1": 0.9333, "tp": 35, "tn": 40, "fp": 0, "fn": 5, "n": 80}
VAL AUC=0.9838 EER={"eer": 0.025, "threshold": 0.0568, "far": 0.025, "frr": 0.025}
VAL confusion(band): {"real": {"REAL": 0, "UNCERTAIN": 40, "SYNTHETIC": 0}, "fake": {"REAL": 0, "UNCERTAIN": 5, "SYNTHETIC": 35}}
NLL T=1: 0.2549 -> T=0.5: 0.1939

Grid on VAL: first band with selective accuracy>=0.95, then max coverage, then narrowest. Chosen [0.05, 0.55] selective=1.0 coverage=0.4375. TEST/my_samples untouched.
VAL fake is crude-VC only; band + weights prove the pipeline, not generalization. Phase 4 re-measures on TEST + my_samples.
