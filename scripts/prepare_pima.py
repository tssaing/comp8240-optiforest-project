"""
Prepare Pima for OptIForest.

The dataset uses 0 to stand in for a value that was not recorded. This script
reports which columns are affected and how many rows carry one, and optionally
drops those rows so the effect can be measured rather than assumed.

    python prepare_pima.py ../raw/29_Pima.npz
    python prepare_pima.py ../raw/29_Pima.npz --drop-missing

Reporting only by default. With --drop-missing it writes an .npz that
convert.py reads directly, so scaling and CSV formatting stay in one place:

    python scripts/convert.py ../raw/pima_no_missing.npz --outdir data/new --divide 32

Use the same divisor for both versions, so the row subset is the only thing
that differs between the two runs.
"""

import argparse
import os

import numpy as np

COLUMNS = ["pregnancies", "glucose", "blood pressure", "skin thickness",
           "insulin", "BMI", "diabetes pedigree", "age"]

# Columns where 0 cannot be a real measurement: glucose, blood pressure, skin
# thickness, insulin, BMI. Pregnancies is left out because 0 pregnancies is
# ordinary, and diabetes pedigree and age have no zeros at all.
MISSING_COLUMNS = [1, 2, 3, 4, 5]


def report(X, y):
    print(f"{'column':20} {'zeros':>6} {'%':>7}   {'min > 0':>9} {'max':>9}")
    print("-" * 56)
    for j, name in enumerate(COLUMNS):
        z = int((X[:, j] == 0).sum())
        nonzero = X[X[:, j] > 0, j]
        flag = " *" if j in MISSING_COLUMNS and z else ""
        print(f"{name:20} {z:>6} {z / len(X) * 100:>6.1f}%   "
              f"{nonzero.min():>9.4g} {X[:, j].max():>9.4g}{flag}")
    print("\n* = 0 cannot be a real measurement in this column")

    missing = (X[:, MISSING_COLUMNS] == 0).any(axis=1)
    print(f"\nrows with at least one missing value: {int(missing.sum()):,} "
          f"of {len(X):,} ({missing.mean() * 100:.1f}%)")
    print(f"rows complete: {int((~missing).sum()):,}")

    print(f"\nmissing among normals  : {missing[y == 0].mean() * 100:5.1f}%")
    print(f"missing among anomalies: {missing[y == 1].mean() * 100:5.1f}%")
    return missing


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", help="ADBench 29_Pima.npz")
    ap.add_argument("--drop-missing", action="store_true",
                    help="drop rows with a missing value; writes "
                         "pima_no_missing.npz alongside the input")
    args = ap.parse_args()

    d = np.load(args.file, allow_pickle=True)
    X = np.asarray(d["X"], float)
    y = np.asarray(d["y"], float).ravel()

    print(f"loaded {X.shape[0]:,} rows, {X.shape[1]} features, "
          f"{int(y.sum()):,} anomalies ({y.mean() * 100:.2f}%)\n")

    missing = report(X, y)

    # Default is to keep the zeros. Dropping the affected rows halves the
    # dataset, and imputing would invent nearly half the insulin column, which
    # in anomaly detection pulls those rows toward the centre of the
    # distribution. --drop-missing measures how much the zeros matter.
    if not args.drop_missing:
        print("\nNo file written. The dataset is used as distributed; convert it "
              "with:\n\n    python scripts/convert.py "
              f"{args.file} --outdir data/new --divide 32\n")
        return

    Xc, yc = X[~missing], y[~missing]
    out = os.path.join(os.path.dirname(args.file), "pima_no_missing.npz")
    np.savez(out, X=Xc, y=yc)

    norm_full = np.linalg.norm(X, axis=1).mean()
    norm_complete = np.linalg.norm(Xc, axis=1).mean()

    print(f"\nwrote {out}")
    print(f"  {Xc.shape[0]:,} rows x {Xc.shape[1]} features, "
          f"{int(yc.sum()):,} anomalies ({yc.mean() * 100:.2f}%)")
    print(f"  mean row norm {norm_complete:,.1f} against {norm_full:,.1f} for "
          f"the full set")
    print(f"  dividing both by 32 gives {norm_complete / 32:.2f} and "
          f"{norm_full / 32:.2f}")
    print(f"\n    python scripts/convert.py {out} --outdir data/new --divide 32\n")


if __name__ == "__main__":
    main()
