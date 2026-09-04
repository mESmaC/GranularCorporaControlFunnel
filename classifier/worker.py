#!/usr/bin/env python3
"""GCCF classifier worker.

GCCF launches one copy of this script per worker (up to 8) and speaks a small
JSON-lines protocol over stdin/stdout:

    -> {"id": 12, "text": "...", "path": "shard-1.jsonl", "format": "jsonl"}
    -> {"id": 13, "text": "...", "path": "...", "format": "..."}
    ->                                          (empty line = end of batch)
    <- {"id": 12, "verdict": "good", "score": 0.93, "tags": ["english"]}
    <- {"id": 13, "verdict": "bad",  "score": 0.71, "tags": []}
    <-                                          (empty line = end of batch)

`score` is the confidence in `verdict` (0..1). Items you omit are recorded as
failed. Print diagnostics to stderr only — stdout is the protocol channel.

Usage:
    py -3 classifier/worker.py --model classifier/models/gccf-clf.joblib
    py -3 classifier/worker.py --dummy            # protocol smoke test, no model

To use a torch model on the GPU, replace `SklearnModel` with a class exposing
`predict(items) -> list[(verdict, score, tags)]`; batches arrive whole, so you
can run one forward pass per batch. GCCF_WORKER_INDEX tells you which worker
you are if you want to pin devices (e.g. `cuda:{index % n_gpus}`).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Iterable


def log(msg: str) -> None:
    sys.stderr.write(f"[gccf-worker {os.environ.get('GCCF_WORKER_INDEX', '?')}] {msg}\n")
    sys.stderr.flush()


class DummyModel:
    """Length-based placeholder so the pipeline can be exercised without a model."""

    def predict(self, items):
        out = []
        for it in items:
            text = it.get("text", "")
            good = len(text) >= 80
            out.append(("good" if good else "bad", 0.6, ["dummy"]))
        return out


class SklearnModel:
    def __init__(self, path: str):
        import joblib  # noqa: WPS433 (optional dependency)

        bundle = joblib.load(path)
        self.verdict_clf = bundle["verdict"]
        self.tag_clf = bundle.get("tags")
        self.tag_names = bundle.get("tag_names", [])
        self.tag_threshold = float(bundle.get("tag_threshold", 0.5))
        log(f"loaded {path}: {len(self.tag_names)} tags")

    def predict(self, items):
        texts = [it.get("text", "") for it in items]
        proba = self.verdict_clf.predict_proba(texts)
        classes = list(self.verdict_clf.classes_)
        good_idx = classes.index(1) if 1 in classes else 0
        tag_probs = None
        if self.tag_clf is not None and self.tag_names:
            try:
                tag_probs = self.tag_clf.predict_proba(texts)
            except Exception as e:  # pragma: no cover
                log(f"tag prediction failed: {e}")
        out = []
        for i, _ in enumerate(items):
            p_good = float(proba[i][good_idx])
            verdict = "good" if p_good >= 0.5 else "bad"
            score = p_good if verdict == "good" else 1.0 - p_good
            tags = []
            if tag_probs is not None:
                row = tag_probs[i]
                for j, name in enumerate(self.tag_names):
                    if j < len(row) and float(row[j]) >= self.tag_threshold:
                        tags.append(name)
            out.append((verdict, score, tags))
        return out


def read_batches(stream) -> Iterable[list[dict]]:
    batch: list[dict] = []
    for line in stream:
        line = line.rstrip("\r\n")
        if line.strip() == "":
            if batch:
                yield batch
                batch = []
            continue
        try:
            batch.append(json.loads(line))
        except json.JSONDecodeError as e:
            log(f"bad request line ignored: {e}")
    if batch:
        yield batch


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="joblib bundle written by train_classifier.py")
    ap.add_argument("--dummy", action="store_true", help="no model; length heuristic (protocol test)")
    args = ap.parse_args()

    if args.dummy or not args.model:
        if not args.dummy:
            log("no --model given; using --dummy")
        model = DummyModel()
    else:
        model = SklearnModel(args.model)

    # Binary-safe, line-buffered stdio with UTF-8 regardless of console code page.
    stdin = open(sys.stdin.fileno(), "r", encoding="utf-8", errors="replace", newline="")
    stdout = open(sys.stdout.fileno(), "w", encoding="utf-8", newline="\n")

    for batch in read_batches(stdin):
        try:
            preds = model.predict(batch)
        except Exception as e:
            log(f"batch failed: {e}")
            preds = []
        for item, (verdict, score, tags) in zip(batch, preds):
            stdout.write(json.dumps({"id": item["id"], "verdict": verdict, "score": round(float(score), 4), "tags": list(tags)}) + "\n")
        stdout.write("\n")
        stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
