#!/usr/bin/env python3
"""Prepare Credit Card: keep all 492 frauds, sample the normal rows, optionally standardise."""
# Usage: python scripts/prepare_creditcard.py --rows-frac=0.10 --standardise

import argparse
import os
import numpy as np
import pandas as pd


def sample(df, rows_frac, seed):
    n_total = round(len(df) * rows_frac)
    fraud = df.index[df["Class"] == 1]
    normal = df.index[df["Class"] == 0]
    n_norm = n_total - len(fraud)
    if n_norm > len(normal) or n_norm < 1:
        raise SystemExit(f"cannot keep {len(fraud)} frauds inside {n_total} rows")
    rng = np.random.default_rng(seed)
    keep = np.sort(np.concatenate([fraud, rng.choice(normal, n_norm, replace=False)]))
    out = df.loc[keep]
    return out.drop(columns=["Class"]).astype(float), out["Class"]


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
          f"{int(label.sum())} anomalies ({100 * label.mean():.3f}%)")
    print(f"mean row norm {norms.mean():,.2f}, divisor rule suggests {norms.mean() / 6:,.0f}")
    print("top 3 columns: " + ", ".join(
        f"{X.columns[j]} {100 * share[j]:.1f}%" for j in top3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="../raw/creditcard.csv")
    ap.add_argument("--rows-frac", type=float, default=0.10)
    ap.add_argument("--standardise", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="data/new")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    X, label = sample(pd.read_csv(args.input), args.rows_frac, args.seed)
    if args.standardise:
        X = standardise(X)
    report(X, label)

    name = args.name or "creditcard_r%d%s" % (round(args.rows_frac * 100),
                                              "_std" if args.standardise else "")
    out = os.path.join(args.outdir, name + ".csv")
    X.assign(label=label).to_csv(out, header=False, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
