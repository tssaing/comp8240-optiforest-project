"""
Convert anomaly detection datasets into the CSV format demo.py expects:
no header row, all columns numeric, label in the final column.

Handles four formats:

    .npz    ADBench           arrays named X and y
    .mat    PyOD / ODDS       arrays named X and y
    .data   UCI               comma separated, class letter in the last column
    .txt    GAAL              id, "out"/"nor", then features; detected automatically
                              and given a _gaal suffix so it cannot overwrite
                              another version of the same benchmark

Examples:

    python convert.py ../raw/6_cardio.npz --outdir data/benchmarks
    python convert.py ../raw/ionosphere.data --anomaly b --outdir data/benchmarks
    python convert.py ../raw/35_SpamBase.npz --outdir data/benchmarks --divide 50
    python convert.py ../raw/SpamBase.txt --outdir data/benchmarks

For .data files, --anomaly names the class value that counts as an anomaly.
Everything else becomes normal. Without it the script lists the class values
it found and stops, so you can choose. It has no effect on the other formats.

Leading numbers and file extensions are stripped from output names, so
6_cardio.npz becomes cardio.csv.

Output goes to ./data unless --outdir says otherwise. Converted data is split
by provenance, so --outdir should normally be given:

    data/benchmarks     the paper's datasets
    data/new            the five existing datasets not in the paper
    data/constructed    datasets built for this study

Two rescaling flags are available, --standardise and --divide. They do
different things and only one can be used at a time. The reasoning for each is
with the standardise() and divide() functions below.
"""

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
    """GAAL files are: id, "out"/"nor", then the features."""
    return len(rows[0]) >= 3 and rows[0][1].strip().strip('"') in ("out", "nor")


def load_gaal(path):
    """Read a GAAL-format file.

    github.com/leibinghe/GAAL-based-outlier-detection, Data/ folder.
    Column 0 is a row id, column 1 is the label as "out" or "nor", and the
    rest are features. These files are normalised to [0, 1], unlike the ADBench
    and PyOD versions of the same benchmarks.
    """
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
    """Rescale each column to mean 0 and standard deviation 1 (a z-score).

    Columns with zero variance are left as zeros rather than divided by zero.
    Ionosphere has one such column, constant at 0 in the UCI original.

    Not recommended, and kept only because the negative result is part of this
    study. On SpamBase, standardising brought the run down from over seven
    hours to 45 minutes but dropped AUC-ROC to 0.5558 against a chance level of
    0.5. Rescaling each column separately gives every feature equal weight, and
    on SpamBase the informative features are the large-valued ones, so the
    signal is diluted across the rest.

    GAAL's copy of SpamBase is min-max normalised per column and fails the same
    way, at 0.5829. Two independent per-column rescalings, the same outcome.

    The paper does not state that it standardises its data, so any use of this
    is a documented deviation and must be reported as one.
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    constant = std == 0
    safe_std = np.where(constant, 1.0, std)
    out = (X - mean) / safe_std
    return out, int(constant.sum())


def divide(X, k):
    """Divide every feature by a single constant.

    Unlike standardising, this preserves the geometry exactly: all relative
    distances stay in proportion, so features that dominated before still
    dominate. Only the overall scale changes.

    Why this matters. OptIForest hashes points with E2LSH, whose bin width is
    fixed at 4 in detectors/opt.py. The number of hash buckets a dataset falls
    into grows with the size of its values, and the merge step in
    HierTree.build_hire is roughly quartic in the number of nodes it has to
    merge, so datasets with large values cost orders of magnitude more time.

    Dividing the data by k is mathematically identical to leaving the data
    alone and setting E2LSH's bin width to 4k, since the offset B ~ U(0, w)
    scales with it. Bin width is a real LSH hyperparameter that has to be
    matched to the scale of the data, and the published code hard-codes it at
    4, a value the paper never states. So this is a way of testing bin width
    without modifying detectors/opt.py.

    Choose k from the row norm alone, before running anything. Target a mean
    row norm near 6, which is where the datasets that reproduce sit. A k picked
    after seeing accuracy results is a tuned parameter and cannot be reported
    as a reproduction.
    """
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
