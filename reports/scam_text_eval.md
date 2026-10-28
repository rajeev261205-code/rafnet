# Scam-text classifier evaluation (constructed phrase set)

Trained: 2026-10-12T15:50:51.798385+00:00 — seed 7

Data: `scam_data/phrases.csv` — 34 hand-written rows (18 scam / 16 benign; en/hi/ta). Ground truth by construction, NOT field data.

Model: TF-IDF word(1,2)+char(3,5) + LogisticRegression(C=2, balanced).

- held-out test (n=9): acc=0.5556 prec=0.6 rec=0.6 f1=0.6 auc=0.7
- 4-fold CV (n=34): acc=0.8824 f1=0.8947

> These numbers validate the training recipe on a tiny constructed set. They say NOTHING about real-call performance. The backend uses this classifier only as one weak cue (+1 weight at p>=0.75, advisory-guarded); rules carry the decision. Retrain on real labelled transcripts before trusting it.
