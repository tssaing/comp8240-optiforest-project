#!/usr/bin/env python3
"""Run OptIForest and the three baselines on one dataset and report AUC-ROC and AUC-PR."""
# Usage: python baselines/run_baselines.py --dataset=Ionosphere --detector=all

import argparse
import csv
import importlib.util
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score

from detectors import LSHiForest          # LSHiForest's copy, alongside this file

import warnings
warnings.filterwarnings("ignore")

NUM_TREES = 100          # as in demo.py and the paper
DEFAULT_REPEATS = 15     # as in demo.py

BASELINES = ["L2SH", "iForest", "ECOD"]
ALL = ["L2OPT"] + BASELINES


def load_optiforest():
    # Both packages are called "detectors", so OptIForest's is loaded by path under another name.
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    pkg_dir = os.path.join(root, "detectors")
    spec = importlib.util.spec_from_file_location(
        "optiforest_detectors",
        os.path.join(pkg_dir, "__init__.py"),
        submodule_search_locations=[pkg_dir])
    mod = importlib.util.module_from_spec(spec)
    sys.modules["optiforest_detectors"] = mod
    spec.loader.exec_module(mod)
    return mod.OptIForest


def auto_threshold(X):
    # The paper's rule: 403 above 10,000 rows or 1,000 features, otherwise 55.
    return 403 if (X.shape[0] > 10000 or X.shape[1] > 1000) else 55


def build(name, seed, threshold, branch):
    if name == "L2OPT":
        OptIForest = load_optiforest()
        return OptIForest("L2OPT", NUM_TREES, threshold, branch)
    if name == "L2SH":
        return LSHiForest("L2SH", NUM_TREES)   # the LSHiForest version the paper compares against
    if name == "iForest":
        return IsolationForest(n_estimators=NUM_TREES, random_state=seed)
    if name == "ECOD":
        from pyod.models.ecod import ECOD
        return ECOD()
    raise ValueError(f"unknown detector '{name}'")


def score(detector, X):
    # Higher means more anomalous. ECOD's scores already are, the others are flipped as in demo.py.
    if type(detector).__name__ == "ECOD":
        return detector.decision_scores_
    return -1.0 * detector.decision_function(X)


def drop_constant_columns(X, name):
    # ECOD fails on constant columns (their skewness is 0/0). Distinct values are counted
    # because a standard-deviation test misses columns with tiny floating-point noise.
    keep = np.array([len(np.unique(X[:, j])) > 1 for j in range(X.shape[1])])
    dropped = int((~keep).sum())
    if dropped:
        print(f"\t{dropped} constant column(s) dropped for {name} "
              f"(they make its skewness undefined)")
    return X[:, keep]


def run(name, X, y, repeats, threshold, branch):
    aucs, prs, train_times, test_times = [], [], [], []

    if name == "ECOD":
        X = drop_constant_columns(X, name)

    # ECOD is deterministic, so one run is enough.
    n = 1 if name == "ECOD" else repeats

    for i in range(n):
        detector = build(name, i, threshold, branch)

        t0 = time.time()
        detector.fit(X)
        train_time = time.time() - t0

        t0 = time.time()
        s = score(detector, X)
        test_time = time.time() - t0

        aucs.append(roc_auc_score(y, s))
        prs.append(average_precision_score(y, s))
        train_times.append(train_time)
        test_times.append(test_time)

    return {
        "detector": name,
        "repeats": n,
        "auc_roc": float(np.mean(aucs)),
        "auc_roc_sd": float(np.std(aucs)),
        "auc_pr": float(np.mean(prs)),
        "auc_pr_sd": float(np.std(prs)),
        "train_s": float(np.mean(train_times)),
        "test_s": float(np.mean(test_times)),
    }


def requested(choice):
    names = {"all": ALL, "baselines": BASELINES}.get(choice, [choice])
    out = []
    for n in names:
        if n == "ECOD":
            try:
                import pyod  # noqa: F401
            except ImportError:
                print("ECOD skipped: pyod is not installed "
                      "(pip install pyod to include it)\n")
                continue
        out.append(n)
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True,
                    help="CSV name without .csv, read from data/ as demo.py does")
    ap.add_argument("--detector", default="baselines",
                    choices=["all", "baselines", "L2OPT", "L2SH", "iForest", "ECOD"])
    ap.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    ap.add_argument("--threshold", default="auto",
                    help="OptIForest cut threshold; 'auto' applies the paper's "
                         "rule of 403 for large datasets and 55 for small")
    ap.add_argument("--branch", type=int, default=0,
                    help="OptIForest branching factor; 0 mixes two- and "
                         "three-way splits so the average comes out to e")
    ap.add_argument("--out", default=None,
                    help="append results to this CSV as well as printing them")
    args = ap.parse_args()

    path = os.path.join("data", args.dataset + ".csv")
    df = pd.read_csv(path, header=None)
    X = df.values[:, :-1]
    y = df.values[:, -1]

    threshold = auto_threshold(X) if args.threshold == "auto" else int(args.threshold)

    chance_pr = y.mean()
    print(f"Dataset: {args.dataset}")
    print(f"  {X.shape[0]:,} rows, {X.shape[1]} features, "
          f"{int(y.sum()):,} anomalies ({chance_pr * 100:.2f}%)")
    print(f"  mean row norm {np.linalg.norm(X, axis=1).mean():,.1f}")
    print(f"  chance level: AUC-ROC 0.5000, AUC-PR {chance_pr:.4f}")

    names = requested(args.detector)
    if "L2OPT" in names:
        print(f"  OptIForest threshold {threshold}"
              f"{' (auto)' if args.threshold == 'auto' else ''}, branch {args.branch}")
    print()

    fields = ["dataset", "detector", "repeats", "auc_roc", "auc_roc_sd",
              "auc_pr", "auc_pr_sd", "train_s", "test_s", "wall_s", "threshold"]

    rows = []
    for name in names:
        print(f"{name}:")
        t0 = time.time()
        r = run(name, X, y, args.repeats, threshold, args.branch)
        r["dataset"] = args.dataset
        r["wall_s"] = time.time() - t0
        r["threshold"] = threshold if name == "L2OPT" else ""
        rows.append(r)

        print(f"\tAUC score:\t {r['auc_roc']:.4f}")
        print(f"\tAUC std:\t {r['auc_roc_sd']:.4f}")
        print(f"\tPR score:\t {r['auc_pr']:.4f}")
        print(f"\tPR std:\t\t {r['auc_pr_sd']:.4f}")
        print(f"\tTraining time:\t {r['train_s']:.2f}")
        print(f"\tTesting time:\t {r['test_s']:.2f}")
        print(f"\trepeats:\t {r['repeats']}\n")

        # Saved after each detector, so a later failure does not lose earlier results.
        if args.out:
            new = not os.path.exists(args.out)
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            with open(args.out, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                if new:
                    w.writeheader()
                w.writerow({k: r[k] for k in fields})

    if args.out:
        print(f"wrote {len(rows)} row(s) to {args.out}")


if __name__ == "__main__":
    main()
