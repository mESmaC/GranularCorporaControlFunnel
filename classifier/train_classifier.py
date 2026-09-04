#!/usr/bin/env python3
"""Train a baseline quality + tag classifier from a GCCF training export.

    py -3 classifier/train_classifier.py gccf-training.jsonl --out classifier/models/gccf-clf.joblib

Input: the JSONL written by GCCF → Classifier → "Export JSONL". Rows with
`label` 0/1 train the good/bad model; every row's `tags` train a multi-label
tagger. Hold-out metrics are printed so you can tell when the signal is dense
enough to trust the model on unseen data.

This is deliberately a small, fast, CPU model (TF-IDF word + char n-grams,
logistic regression). Its job is to bootstrap: once the exported set is large
enough, swap in whatever you like — the worker only needs `predict`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter


def load(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("data", help="training JSONL exported by GCCF")
    ap.add_argument("--out", default="classifier/models/gccf-clf.joblib")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--min-tag-count", type=int, default=5, help="tags seen fewer times are not learned")
    ap.add_argument("--tag-threshold", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    try:
        import joblib
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import classification_report
        from sklearn.model_selection import train_test_split
        from sklearn.multiclass import OneVsRestClassifier
        from sklearn.pipeline import FeatureUnion, Pipeline
        from sklearn.preprocessing import MultiLabelBinarizer
    except ImportError as e:
        print(f"missing dependency: {e}\n  py -3 -m pip install -r classifier/requirements.txt", file=sys.stderr)
        return 2

    rows = load(args.data)
    labelled = [r for r in rows if r.get("label") in (0, 1)]
    print(f"{len(rows)} rows, {len(labelled)} labelled good/bad "
          f"({sum(1 for r in labelled if r['label'] == 1)} good / {sum(1 for r in labelled if r['label'] == 0)} bad)")
    if len(labelled) < 20 or len({r["label"] for r in labelled}) < 2:
        print("need at least 20 labelled rows with both classes; keep annotating", file=sys.stderr)
        return 1

    def features():
        return FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=200_000, sublinear_tf=True)),
        ])

    # ---- verdict model -----------------------------------------------------
    X = [r["text"] for r in labelled]
    y = np.array([int(r["label"]) for r in labelled])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=args.test_size, random_state=args.seed, stratify=y)
    verdict = Pipeline([("f", features()), ("clf", LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced"))])
    verdict.fit(Xtr, ytr)
    print("\n== good/bad hold-out ==")
    print(classification_report(yte, verdict.predict(Xte), target_names=["bad", "good"], digits=3))
    verdict.fit(X, y)  # refit on everything for the shipped model

    # ---- tag model ---------------------------------------------------------
    tag_counts = Counter(t for r in rows for t in r.get("tags", []))
    tag_names = sorted(t for t, c in tag_counts.items() if c >= args.min_tag_count)
    tags_model = None
    if tag_names:
        tagged = [r for r in rows if r.get("text")]
        mlb = MultiLabelBinarizer(classes=tag_names)
        Y = mlb.fit_transform([[t for t in r.get("tags", []) if t in tag_names] for r in tagged])
        keep = [i for i, name in enumerate(tag_names) if 0 < Y[:, i].sum() < len(tagged)]
        if keep:
            tag_names = [tag_names[i] for i in keep]
            Y = Y[:, keep]
            Xt = [r["text"] for r in tagged]
            Xtr, Xte, Ytr, Yte = train_test_split(Xt, Y, test_size=args.test_size, random_state=args.seed)
            tags_model = Pipeline([("f", features()), ("clf", OneVsRestClassifier(LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")))])
            tags_model.fit(Xtr, Ytr)
            print("== tags hold-out ==")
            print(classification_report(Yte, (tags_model.predict_proba(Xte) >= args.tag_threshold).astype(int), target_names=tag_names, digits=3, zero_division=0))
            tags_model.fit(Xt, Y)
        else:
            tag_names = []
    else:
        print(f"no tag seen ≥ {args.min_tag_count} times; skipping tag model")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    joblib.dump({"verdict": verdict, "tags": tags_model, "tag_names": tag_names, "tag_threshold": args.tag_threshold}, args.out)
    print(f"\nwrote {args.out}\nrun with:  py -3 classifier/worker.py --model {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
