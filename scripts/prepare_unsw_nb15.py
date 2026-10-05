#!/usr/bin/env python3
"""Build an anomaly detection CSV from UNSW-NB15.

    python scripts/prepare_unsw_nb15.py --rows-frac=0.10 --standardise

Drops id and attack_cat, collapses proto to its top 6 values plus other,
one-hot encodes the three text columns, then samples rows to the wanted size
and anomaly rate. Label goes last, no header, all numeric.
"""

import argparse
import os
import numpy as np
import pandas as pd

TEXT_COLS = ["proto", "service", "state"]


def encode(df):
    top6 = df["proto"].value_counts().head(6).index
    df["proto"] = np.where(df["proto"].isin(top6), df["proto"], "other")
    df = df.drop(columns=["id", "attack_cat"])
    label = df.pop("label")
    X = pd.get_dummies(df, columns=TEXT_COLS, dtype=float).astype(float)
    return X, label


def sample(X, label, rows_frac, anomaly_rate, seed):
    n_total = round(len(X) * rows_frac)
    n_anom = round(n_total * anomaly_rate)
    n_norm = n_total - n_anom
    anom = label.index[label == 1]
    norm = label.index[label == 0]
    if n_anom > len(anom) or n_norm > len(norm):
        raise SystemExit(f"not enough rows: need {n_norm} normal and {n_anom} anomalous")
    rng = np.random.default_rng(seed)
    keep = np.concatenate([rng.choice(norm, n_norm, replace=False),
                           rng.choice(anom, n_anom, replace=False)])
    keep = np.sort(keep)
    return X.loc[keep], label.loc[keep]


def standardise(X):
    sd = X.std(axis=0)
    return (X - X.mean(axis=0)) / sd.where(sd > 0, 1.0)


def report(X, label):
    A = X.values
    norms = np.linalg.norm(A, axis=1)
    share = (A ** 2).sum(axis=0)
    share = share / share.sum()
    top3 = np.argsort(-share)[:3]
    print(f"{len(X):,} rows, {X.shape[1]} features, "
          f"{int(label.sum())} anomalies ({100 * label.mean():.2f}%)")
    print(f"mean row norm {norms.mean():,.2f}, divisor rule suggests {norms.mean() / 6:,.0f}")
    print("top 3 columns: " + ", ".join(
        f"{X.columns[j]} {100 * share[j]:.1f}%" for j in top3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="../raw/UNSW_NB15_training-set.csv")
    ap.add_argument("--rows-frac", type=float, default=0.10)
    ap.add_argument("--anomaly-rate", type=float, default=0.05)
    ap.add_argument("--standardise", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="data/new")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    X, label = encode(pd.read_csv(args.input))
    X, label = sample(X, label, args.rows_frac, args.anomaly_rate, args.seed)
    if args.standardise:
        X = standardise(X)
    report(X, label)

    name = args.name or "unsw_r%d_a%d%s" % (round(args.rows_frac * 100),
                                            round(args.anomaly_rate * 100),
                                            "_std" if args.standardise else "")
    out = os.path.join(args.outdir, name + ".csv")
    X.assign(label=label).to_csv(out, header=False, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
