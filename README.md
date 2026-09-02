# A Reproducibility Study of OptIForest

This project reproduces the results of **OptIForest: Optimal Isolation Forest
for Anomaly Detection** (Xiang et al., IJCAI 2023), then applies the method to
datasets it has not previously been tested on.

---

## Source code

The core implementation is the authors' original code, taken unmodified from
[github.com/xiagll/OptIForest](https://github.com/xiagll/OptIForest):

- `detectors/` — the OptIForest implementation
- `demo.py` — the entry point for a single experiment
- `data/ad.csv` — the sample dataset bundled with the repository
- `test/` — the authors' unit tests
- `README_original.md` — the authors' own README, kept for reference

**Paper:** Haolong Xiang, Xuyun Zhang, Hongsheng Hu, Lianyong Qi, Wanchun Dou,
Mark Dras, Amin Beheshti and Xiaolong Xu. OptIForest: Optimal Isolation Forest
for Anomaly Detection. *IJCAI-23*, pages 2379–2387, 2023.

---

## Setup

The original code needs Python 3.8 with `numpy==1.20.1` and
`scikit-learn==0.22.1`, neither of which installs on current Python versions.
The Dockerfile pins that environment so the code runs anywhere.

Build the image:

    docker build -t optiforest .

Run on the bundled dataset:

    docker run --rm optiforest

Run on a different dataset:

    docker run --rm -v "$PWD/data:/app/data" optiforest \
        python demo.py --dataset=NAME --threshold=403 --branch=0

The threshold is 403 for large datasets and 55 for small ones, following the
paper (e^6 and e^4 respectively). `--branch=0` lets the algorithm mix two-way
and three-way splits so the average branching factor comes out to e.

### Data format

Input CSVs need no header row, all columns numeric, and the label in the final
column (0 normal, 1 anomaly). This matches `demo.py`, which reads features with
`values[:, :-1]` and labels with `values[:, -1]`.

---

## Reproduction

`demo.py` runs each configuration 15 times and reports the mean.

| Dataset | AUC-ROC here | Paper | AUC-PR here | Paper |
| --- | --- | --- | --- | --- |
| AD | 0.7707 ± 0.0071 | 77.4 ± 0.7 | 0.4236 ± 0.0360 | 43.8 ± 3.1 |

Both fall inside the paper's reported standard deviation.

Environment: MacBook Pro (Intel Core i9), Docker, Python 3.8. About 2.5 minutes
per run, so roughly 37 minutes for the full 15.
