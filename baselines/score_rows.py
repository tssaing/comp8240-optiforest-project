#!/usr/bin/env python3
"""Save OptIForest's per-row anomaly scores, averaged over several runs, for the LLM judge."""
# Usage: python baselines/score_rows.py --dataset=wiki_edits_nobots_std --repeats=15 --out=results/scores.csv

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

import run_baselines as rb   # same folder, so the detectors are built exactly as in the main runs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, help="CSV name without .csv, read from data/")
    ap.add_argument("--repeats", type=int, default=15)   # as in run_baselines.py
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(os.path.join("data", args.dataset + ".csv"), header=None)
    X, y = df.values[:, :-1], df.values[:, -1]
    threshold = rb.auto_threshold(X)

    runs = []
    for i in range(args.repeats):
        detector = rb.build("L2OPT", i, threshold, 0)
        detector.fit(X)
        s = rb.score(detector, X)
        runs.append(s)
        print(f"run {i + 1}/{args.repeats}: AUC-ROC {roc_auc_score(y, s):.4f}, "
              f"AUC-PR {average_precision_score(y, s):.4f}")

    # Averaged over runs because OptIForest has no random seed.
    mean = np.mean(runs, axis=0)
    out = pd.DataFrame({"row": np.arange(len(y)), "score": mean, "label": y.astype(int)})
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"averaged: AUC-ROC {roc_auc_score(y, mean):.4f}, "
          f"AUC-PR {average_precision_score(y, mean):.4f}, wrote {args.out}")


if __name__ == "__main__":
    main()
