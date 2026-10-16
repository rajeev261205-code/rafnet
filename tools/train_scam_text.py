"""Train a tiny scam-vs-benign text classifier on scam_data/phrases.csv.

TF-IDF (word 1-2 grams + char 3-5 grams, handles Hindi/Tamil transliteration)
+ LogisticRegression, stratified held-out test, seed fixed. Saves
models/scam_text.joblib (git-ignored weights: vectorizer + model + metrics)
and writes reports/scam_text_eval.md (committed numbers).

The phrase set is HAND-WRITTEN constructed data (CC0, see scam_data/README).
Metrics are a sanity check on that set — NOT deployment performance.
Needs scikit-learn (aegis-venv has 1.6.1; system python may not).
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_rows(path: Path):
    texts, labels, langs = [], [], []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            texts.append(row["text"].strip())
            labels.append(1 if row["label"].strip().lower() == "scam" else 0)
            langs.append(row.get("language", "?"))
    return texts, labels, langs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="scam_data/phrases.csv")
    ap.add_argument("--out", default="models/scam_text.joblib")
    ap.add_argument("--report", default="reports/scam_text_eval.md")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--test-size", type=float, default=0.25)
    args = ap.parse_args()

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )
        from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
    except ImportError:
        print("scikit-learn is required (not installed in this runtime).")
        return 2

    texts, labels, langs = load_rows(ROOT / args.data)
    print(f"rows={len(texts)} scam={sum(labels)} benign={len(labels)-sum(labels)}")

    vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), analyzer="word",
                          min_df=1, sublinear_tf=True)
    ch = TfidfVectorizer(lowercase=True, analyzer="char", ngram_range=(3, 5),
                         min_df=1, sublinear_tf=True)
    Xw, Xc = vec.fit_transform(texts), ch.fit_transform(texts)
    import numpy as np
    from scipy.sparse import hstack

    X = hstack([Xw, Xc])
    y = np.array(labels)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=args.test_size,
                                          random_state=args.seed, stratify=y)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=2.0)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    proba = clf.predict_proba(Xte)[:, 1]
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=args.seed)
    cv_pred = cross_val_predict(LogisticRegression(max_iter=2000, class_weight="balanced", C=2.0),
                                X, y, cv=cv)
    m = {
        "n": len(texts), "test_n": len(yte), "seed": args.seed,
        "test_accuracy": round(float(accuracy_score(yte, pred)), 4),
        "test_precision": round(float(precision_score(yte, pred, zero_division=0)), 4),
        "test_recall": round(float(recall_score(yte, pred, zero_division=0)), 4),
        "test_f1": round(float(f1_score(yte, pred, zero_division=0)), 4),
        "test_auc": round(float(roc_auc_score(yte, proba)), 4),
        "cv4_accuracy": round(float(accuracy_score(y, cv_pred)), 4),
        "cv4_f1": round(float(f1_score(y, cv_pred, zero_division=0)), 4),
    }
    print(json.dumps(m, indent=1))

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        import joblib

        joblib.dump({"word_vec": vec, "char_vec": ch, "clf": clf,
                     "metrics": m, "trained_at": datetime.now(timezone.utc).isoformat(),
                     "data": str(args.data)}, out)
        print(f"saved {out}")
    except ImportError:
        print("joblib missing — model not saved (report still written).")

    rep = ROOT / args.report
    rep.write_text(
        "# Scam-text classifier evaluation (constructed phrase set)\n\n"
        f"Trained: {datetime.now(timezone.utc).isoformat()} — seed {args.seed}\n\n"
        f"Data: `{args.data}` — {m['n']} hand-written rows "
        f"({sum(labels)} scam / {len(labels)-sum(labels)} benign; en/hi/ta). "
        "Ground truth by construction, NOT field data.\n\n"
        f"Model: TF-IDF word(1,2)+char(3,5) + LogisticRegression(C=2, balanced).\n\n"
        f"- held-out test (n={m['test_n']}): acc={m['test_accuracy']} "
        f"prec={m['test_precision']} rec={m['test_recall']} f1={m['test_f1']} auc={m['test_auc']}\n"
        f"- 4-fold CV (n={m['n']}): acc={m['cv4_accuracy']} f1={m['cv4_f1']}\n\n"
        "> These numbers validate the training recipe on a tiny constructed set. "
        "They say NOTHING about real-call performance. The backend uses this "
        "classifier only as one weak cue (+1 weight at p>=0.75, advisory-guarded); "
        "rules carry the decision. Retrain on real labelled transcripts before "
        "trusting it.\n", encoding="utf-8")
    print(f"wrote {rep}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
