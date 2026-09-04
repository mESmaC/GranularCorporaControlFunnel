#!/usr/bin/env python3
"""Train (or finetune) a baseline quality + tag classifier from a GCCF export.

    py -3 classifier/train_classifier.py gccf-training.jsonl --out classifier/models/gccf-clf.joblib
    py -3 classifier/train_classifier.py data.jsonl --out new.joblib --parent old.joblib   # finetune

GCCF's "external trainer" runs this with `{data}` / `{model}` / `{parent_model}`
substituted (also available as GCCF_DATA / GCCF_MODEL_OUT / GCCF_PARENT_MODEL).

Input rows carry `label` (1 good / 0 bad / null), `tags`, and — after an
evaluation round — `eval`: {predicted, score, human, agree, ...}. Rows a model
got wrong during evaluation are up-weighted (`--feedback-weight`), which is the
cheap, honest form of "finetune on the evaluation outputs" for a linear model.
Swap in whatever you like; the only contracts are the JSONL in and the worker
protocol out (see worker.py).

The last stdout line is `{"gccf_metrics": {...}}` so GCCF can store hold-out
metrics next to the model.
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


def prf(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    n = len(y_true)
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    f1 = (2 * prec * rec / (prec + rec)) if prec and rec and prec + rec else None
    return {"n_holdout": n, "tp": tp, "fp": fp, "tn": tn, "fn_": fn,
            "accuracy": (tp + tn) / n if n else None, "precision": prec, "recall": rec, "f1": f1}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="?", default=os.environ.get("GCCF_DATA"), help="training JSONL exported by GCCF")
    ap.add_argument("--out", default=os.environ.get("GCCF_MODEL_OUT", "classifier/models/gccf-clf.joblib"))
    ap.add_argument("--parent", default=os.environ.get("GCCF_PARENT_MODEL") or None, help="finetune: reuse this model's feature space and start from its weights")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--min-tag-count", type=int, default=5, help="tags seen fewer times are not learned")
    ap.add_argument("--tag-threshold", type=float, default=0.5)
    ap.add_argument("--feedback-weight", type=float, default=3.0, help="sample weight for rows a model got wrong in evaluation")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if not args.data:
        print("no training data given", file=sys.stderr)
        return 2

    try:
        import joblib
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.multiclass import OneVsRestClassifier
        from sklearn.pipeline import FeatureUnion, Pipeline
        from sklearn.preprocessing import MultiLabelBinarizer
    except ImportError as e:
        print(f"missing dependency: {e}\n  py -3 -m pip install -r classifier/requirements.txt", file=sys.stderr)
        return 2

    rows = load(args.data)
    labelled = [r for r in rows if r.get("label") in (0, 1)]
    n_good = sum(1 for r in labelled if r["label"] == 1)
    n_bad = len(labelled) - n_good
    fb = [r for r in rows if r.get("eval") and r["eval"].get("agree") is False]
    print(f"{len(rows)} rows, {len(labelled)} labelled ({n_good} good / {n_bad} bad), {len(fb)} with evaluation disagreements")
    if len(labelled) < 20 or n_good == 0 or n_bad == 0:
        print("need at least 20 labelled rows with both classes; keep annotating", file=sys.stderr)
        return 1

    parent = joblib.load(args.parent) if args.parent else None
    if parent:
        print(f"finetuning from {args.parent}")

    def features():
        return FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=200_000, sublinear_tf=True)),
        ])

    def weight(r):
        e = r.get("eval")
        return args.feedback_weight if e and e.get("agree") is False else 1.0

    # ---- feature space: reuse the parent's when finetuning ----------------
    all_text = [r["text"] for r in rows]
    if parent:
        feats = parent["features"]
    else:
        feats = features().fit(all_text)
    X_all = feats.transform(all_text)
    index = {id(r): i for i, r in enumerate(rows)}

    # ---- verdict head -------------------------------------------------------
    li = np.array([index[id(r)] for r in labelled])
    y = np.array([int(r["label"]) for r in labelled])
    w = np.array([weight(r) for r in labelled])
    tr, te = train_test_split(np.arange(len(labelled)), test_size=args.test_size, random_state=args.seed, stratify=y)

    def lr(init=None):
        clf = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced", warm_start=init is not None)
        if init is not None:
            clf.coef_, clf.intercept_, clf.classes_ = init.coef_.copy(), init.intercept_.copy(), init.classes_
        return clf

    clf = lr(parent["verdict"] if parent else None)
    clf.fit(X_all[li[tr]], y[tr], sample_weight=w[tr])
    vm = prf(list(y[te]), list(clf.predict(X_all[li[te]])))
    vm["n_train"] = int(len(tr))
    print("== good/bad hold-out ==", json.dumps(vm))
    verdict = lr(parent["verdict"] if parent else None)
    verdict.fit(X_all[li], y, sample_weight=w)

    # ---- tag heads --------------------------------------------------------
    tag_counts = Counter(t for r in rows for t in r.get("tags", []))
    tag_names = sorted(t for t, c in tag_counts.items() if args.min_tag_count <= c < len(rows))
    tags_model = None
    tag_metrics = []
    if tag_names:
        mlb = MultiLabelBinarizer(classes=tag_names)
        Y = mlb.fit_transform([[t for t in r.get("tags", []) if t in tag_names] for r in rows])
        keep = [i for i in range(len(tag_names)) if 0 < Y[:, i].sum() < len(rows)]
        tag_names = [tag_names[i] for i in keep]
        Y = Y[:, keep]
        if tag_names:
            atr, ate = train_test_split(np.arange(len(rows)), test_size=args.test_size, random_state=args.seed)
            probe = OneVsRestClassifier(LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")).fit(X_all[atr], Y[atr])
            P = (probe.predict_proba(X_all[ate]) >= args.tag_threshold).astype(int)
            for j, name in enumerate(tag_names):
                m = prf(list(Y[ate, j]), list(P[:, j]))
                tag_metrics.append({"name": name, "count": int(tag_counts[name]), "metrics": m})
                print(f"== tag {name} ==", json.dumps(m))
            tags_model = OneVsRestClassifier(LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")).fit(X_all, Y)
    else:
        print(f"no tag seen >= {args.min_tag_count} times; skipping tag model")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    joblib.dump({"features": feats, "verdict": verdict, "tags": tags_model, "tag_names": tag_names, "tag_threshold": args.tag_threshold}, args.out)
    print(f"wrote {args.out}")
    print(json.dumps({"gccf_metrics": {
        "rows": len(rows), "labelled": len(labelled), "good": n_good, "bad": n_bad,
        "feedback_rows": len(fb), "warm_start": bool(parent), "verdict": vm, "tags": tag_metrics,
    }}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
