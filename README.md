# A Reproducibility Study of OptIForest

This project reproduces results from **OptIForest: Optimal Isolation Forest for
Anomaly Detection** (Xiang et al., IJCAI 2023), compares the method against
three of the paper's baselines, and applies it to datasets it has not previously
been tested on.

COMP8240 Applications of Data Science, Macquarie University.

---

## Source code

### OptIForest

The core implementation is the authors' original code, taken **unmodified** from
[github.com/xiagll/OptIForest](https://github.com/xiagll/OptIForest):

- `detectors/` — the OptIForest implementation
- `demo.py` — the authors' entry point for a single experiment
- `data/ad.csv` — the sample dataset bundled with the repository
- `README_original.md` — the authors' own README

**Paper:** Haolong Xiang, Xuyun Zhang, Hongsheng Hu, Lianyong Qi, Wanchun Dou,
Mark Dras, Amin Beheshti and Xiaolong Xu. OptIForest: Optimal Isolation Forest
for Anomaly Detection. *IJCAI-23*, pages 2379–2387, 2023.

### LSHiForest

**Not included in this repository.** LSHiForest is marked "Copyright reserved"
by its author with no licence granting redistribution, so it is not committed
here.

Fetch it from
[github.com/xuyun-zhang/LSHiForest](https://github.com/xuyun-zhang/LSHiForest)
and copy its `src/detectors` folder into `baselines/detectors/` before building
the image. The code is used unmodified.

Its author, Xuyun Zhang, is also a co-author of the OptIForest paper, and the
OptIForest code is built on this framework.

**Paper:** Xuyun Zhang, Wanchun Dou, Qiang He, Rui Zhou, Christopher Leckie,
Ramamohanarao Kotagiri and Zoran Salcic. LSHiForest: A Generic Framework for
Fast Tree Isolation Based Ensemble Anomaly Analysis. *ICDE 2017*, pages 983–994.

### iForest and ECOD

From scikit-learn and PyOD respectively. No code is included here.

---

## Setup

Fetch LSHiForest first, as above, then build. The Docker image copies the
repository in, so `baselines/detectors/` must be present before the build.

The original code needs Python 3.8 with `numpy==1.20.1` and
`scikit-learn==0.22.1`, neither of which installs on current Python versions.
The Dockerfile pins that environment.

    docker build -t optiforest .

PyOD is installed alongside for the ECOD baseline. Its dependencies are pinned
and installed with `--no-deps` so they cannot pull a newer numpy or
scikit-learn over the pins above. The build checks the pins are intact and
imports ECOD, so it fails rather than producing a quietly broken image.

---

## Data preparation

`scripts/convert.py` converts ADBench `.npz`, PyOD `.mat`, UCI `.data` and GAAL
`.txt` files into the format `demo.py` expects: no header row, all columns
numeric, label in the final column.

    python scripts/convert.py ../raw/6_cardio.npz --outdir data/benchmarks

It reports the **mean row norm** of every dataset it converts.

Two rescaling options:

`--divide K` divides every feature by a constant. This shrinks the data without
changing its shape, and has the same effect as widening E2LSH's hash buckets,
which the code fixes at 4.

`--standardise` rescales each column to mean 0 and standard deviation 1.

These are not interchangeable. Which one works depends on the dataset.

---

## Usage

OptIForest alone, as the authors intended:

    docker run --rm -v "$PWD/data/benchmarks:/app/data" optiforest \
        python demo.py --dataset=NAME --threshold=55 --branch=0

All four detectors, writing results to CSV:

    docker run --rm \
        -v "$PWD/data/benchmarks:/app/data" \
        -v "$PWD/results/benchmarks:/app/results" \
        optiforest \
        python baselines/run_baselines.py --dataset=NAME --detector=all \
        --out=/app/results/NAME.csv

`run_baselines.py` uses the same protocol as `demo.py`: 100 trees, 15 repeats,
both AUC-ROC and AUC-PR. ECOD runs once because it is deterministic. The cut
threshold follows the paper's rule, 403 for datasets over 10,000 rows or 1,000
features and 55 otherwise.

---

Environment: MacBook Pro 16-inch 2019, Intel Core i9, macOS. Docker, Python 3.8.
The code is single-threaded and uses one core.
