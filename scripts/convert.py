"""Convert anomaly detection datasets (.npz, .mat, UCI .data, GAAL .txt) to CSV, label in the last column."""
# Usage: python scripts/convert.py ../raw/6_cardio.npz --outdir data/benchmarks [--standardise | --divide K]

import argparse
import csv
import os
import re
from collections import Counter

import numpy as np


def clean_name(path):
    base = os.path.basename(path)
    base = re.sub(r"\.(npz|mat|data|csv|txt)$", "", base)
    base = re.sub(r"^\d+_", "", base)
    return base


def load_npz(path):
    d = np.load(path, allow_pickle=True)
    if "X" not in d or "y" not in d:
        raise KeyError(f"expected arrays X and y, found {d.files}")
    return np.asarray(d["X"], float), np.asarray(d["y"], float).ravel()


def load_mat(path):
    from scipy.io import loadmat
    m = loadmat(path)
    if "X" not in m or "y" not in m:
        keys = [k for k in m if not k.startswith("__")]
        raise KeyError(f"expected arrays X and y, found {keys}")
    return np.asarray(m["X"], float), np.asarray(m["y"], float).ravel()


def looks_like_gaal(rows):
    # GAAL files are: id, "out"/"nor", then the features.
    return len(rows[0]) >= 3 and rows[0][1].strip().strip('"') in ("out", "nor")


def load_gaal(path):
    # GAAL format (github.com/leibinghe/GAAL-based-outlier-detection): id, label, then features.
    rows = [r for r in csv.reader(open(path)) if r]
    labels = [r[1].strip().strip('"') for r in rows]
    X = np.array([[float(v) for v in r[2:]] for r in rows])
    y = np.array([1.0 if lab == "out" else 0.0 for lab in labels])
    return X, y


def load_uci(path, anomaly_class):
    rows = [r for r in csv.reader(open(path)) if r]
    labels = [r[-1].strip() for r in rows]
    counts = Counter(labels)

    if anomaly_class is None:
        listing = ", ".join(f"{k} ({v})" for k, v in counts.most_common())
        raise ValueError(
            f"class values found: {listing}. "
            f"Re-run with --anomaly to say which one is the anomaly."
        )

    if anomaly_class not in counts:
        raise ValueError(
            f"'{anomaly_class}' is not a class value here. "
            f"Found: {', '.join(counts)}"
        )

    X = np.array([[float(v) for v in r[:-1]] for r in rows])
    y = np.array([1.0 if lab == anomaly_class else 0.0 for lab in labels])
    return X, y


def standardise(X):
    # Rescale each column to mean 0 and standard deviation 1. Constant columns stay at 0.
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    constant = std == 0
    safe_std = np.where(constant, 1.0, std)
    out = (X - mean) / safe_std
    return out, int(constant.sum())


def divide(X, k):
    # Divide every value by k, which is the same as using bin width 4k on the raw data.
    # Choose k from the row norm (aim for about 6) before seeing any results.
    return X / float(k)


def convert(path, outdir, name=None, anomaly_class=None, do_standardise=False,
            divide_by=None):
    ext = os.path.splitext(path)[1].lower()
    is_gaal = False

    if ext == ".npz":
        X, y = load_npz(path)
    elif ext == ".mat":
        X, y = load_mat(path)
    elif ext in (".data", ".csv", ".txt"):
        rows = [r for r in csv.reader(open(path)) if r]
        if looks_like_gaal(rows):
            X, y = load_gaal(path)
            is_gaal = True
        else:
            X, y = load_uci(path, anomaly_class)
    else:
        raise ValueError(f"unsupported extension '{ext}'")

    if len(X) != len(y):
        raise ValueError(f"{len(X)} rows but {len(y)} labels")

    out_name = name or clean_name(path)
    if is_gaal and not name:
        out_name += "_gaal"

    before = np.linalg.norm(X, axis=1).mean()
    n_constant = 0
    if do_standardise:
        X, n_constant = standardise(X)
        if not name:
            out_name += "_std"
    elif divide_by is not None:
        X = divide(X, divide_by)
        if not name:
            out_name += "_div" + f"{divide_by:g}".replace(".", "p")

    out_path = os.path.join(outdir, out_name + ".csv")

    os.makedirs(outdir, exist_ok=True)
    np.savetxt(out_path, np.hstack([X, y.reshape(-1, 1)]),
               delimiter=",", fmt="%.10g")

    print(f"{out_name:20} {X.shape[0]:>8,} rows  {X.shape[1]:>5} features  "
          f"{int(y.sum()):>6,} anomalies ({y.mean() * 100:5.2f}%)  -> {out_path}")

    after = np.linalg.norm(X, axis=1).mean()
    if do_standardise:
        print(f"{'':20} standardised: mean row norm {before:,.1f} -> {after:,.1f}")
        if n_constant:
            print(f"{'':20} {n_constant} constant column(s) left as zeros")
    elif divide_by is not None:
        print(f"{'':20} divided by {divide_by:g}: mean row norm "
              f"{before:,.1f} -> {after:,.1f}")
        print(f"{'':20} equivalent to E2LSH bin width {4 * divide_by:g} "
              f"on the raw data")
    else:
        print(f"{'':20} mean row norm {after:,.1f}"
              f"{f'   (large: consider --divide {round(after / 6)})' if after > 20 else ''}")

    if y.mean() > 0.5:
        print(f"{'':20} WARNING: anomalies are the majority class here")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="one or more dataset files")
    ap.add_argument("--outdir", default="data", help="output directory (default: data)")
    ap.add_argument("--name", default=None,
                    help="output name, single input file only")
    ap.add_argument("--anomaly", default=None,
                    help="for .data files, the class value to treat as anomalous")
    ap.add_argument("--standardise", action="store_true",
                    help="rescale each feature column to mean 0, sd 1; "
                         "adds a _std suffix to the output name")
    ap.add_argument("--divide", type=float, default=None, metavar="K",
                    help="divide every feature by K, preserving geometry; "
                         "adds a _divK suffix. Equivalent to setting E2LSH's "
                         "bin width to 4K on the raw data")
    args = ap.parse_args()

    if args.name and len(args.files) > 1:
        ap.error("--name works with a single input file only")

    if args.standardise and args.divide is not None:
        ap.error("--standardise and --divide do different things; pick one")

    for path in args.files:
        try:
            convert(path, args.outdir, args.name, args.anomaly,
                    args.standardise, args.divide)
        except Exception as exc:
            print(f"{os.path.basename(path):20} FAILED: {exc}")


if __name__ == "__main__":
    main()
