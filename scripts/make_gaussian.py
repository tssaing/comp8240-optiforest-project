#!/usr/bin/env python3
"""Build a synthetic dataset: Gaussian clusters of normal points plus anomalies scattered at random."""
# Usage: python scripts/make_gaussian.py --clusters=5 --dims=5 --cluster-sd=3 --anomaly-rate=0.05 --scale=1

import argparse
import os
import numpy as np

SPREAD = 10.0


def generate(rows, clusters, dims, anomaly_rate, cluster_sd, seed):
    rng = np.random.default_rng(seed)
    n_anom = round(rows * anomaly_rate)
    n_norm = rows - n_anom

    centres = rng.uniform(-SPREAD, SPREAD, size=(clusters, dims))
    which = rng.integers(0, clusters, size=n_norm)
    normal = centres[which] + rng.normal(0, cluster_sd, size=(n_norm, dims))

    low, high = normal.min(axis=0), normal.max(axis=0)
    anomalies = rng.uniform(low, high, size=(n_anom, dims))

    X = np.vstack([normal, anomalies])
    y = np.concatenate([np.zeros(n_norm), np.ones(n_anom)])
    order = rng.permutation(len(X))
    return X[order], y[order]


def report(X, y):
    norms = np.linalg.norm(X, axis=1)
    print(f"{len(X):,} rows, {X.shape[1]} features, "
          f"{int(y.sum())} anomalies ({100 * y.mean():.2f}%)")
    print(f"mean row norm {norms.mean():,.2f}, divisor rule suggests {norms.mean() / 6:,.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=5000)
    ap.add_argument("--clusters", type=int, default=5)
    ap.add_argument("--dims", type=int, default=10)
    ap.add_argument("--anomaly-rate", type=float, default=0.05)
    ap.add_argument("--cluster-sd", type=float, default=1.0)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="data/constructed")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    X, y = generate(args.rows, args.clusters, args.dims, args.anomaly_rate,
                    args.cluster_sd, args.seed)
    X = X * args.scale
    report(X, y)

    name = args.name or "gauss_k%d_d%d_sd%g_a%d_s%g" % (args.clusters, args.dims,
                                                        args.cluster_sd,
                                                        round(args.anomaly_rate * 100),
                                                        args.scale)
    out = os.path.join(args.outdir, name + ".csv")
    np.savetxt(out, np.column_stack([X, y]), delimiter=",", fmt="%.6f")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
