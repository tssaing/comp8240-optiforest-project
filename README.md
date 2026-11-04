# A Reproducibility Study of OptIForest

This project reproduces results from **OptIForest: Optimal Isolation Forest for
Anomaly Detection** (Xiang et al., IJCAI 2023), compares the method against
three of the paper's baselines, and applies it to datasets it has not been tested
on before, including two built for this study. A local language model is used as
a judge on the Wikipedia dataset.

COMP8240 Applications of Data Science, Macquarie University.

---

## What is in this repository

| Folder or file | What it holds |
| --- | --- |
| `detectors/`, `demo.py`, `test/` | OptIForest and its tests, the authors' code, unmodified |
| `baselines/run_baselines.py` | Runs OptIForest and the three baselines on one dataset |
| `baselines/score_rows.py` | Saves OptIForest's anomaly score for every row, for the LLM judge |
| `scripts/` | Data preparation, the Wikipedia collection and the LLM judge |
| `data/ad.csv` | The authors' sample dataset |
| `data/constructed/` | The two datasets built for this study |
| `results/` | One CSV of results per dataset |
| `Dockerfile` | The pinned Python environment |

The other datasets are not included, because they belong to other people and
some are large. The step-by-step section says where to download each one.

---

## Source code

### OptIForest

The core implementation is the authors' original code, taken **unmodified** from
[github.com/xiagll/OptIForest](https://github.com/xiagll/OptIForest):

- `detectors/`: the OptIForest implementation
- `demo.py`: the authors' entry point for a single experiment
- `data/ad.csv`: the sample dataset bundled with the repository
- `README_original.md`: the authors' own README

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

1. Fetch LSHiForest as described above.

2. Build the Docker image. All four methods run inside it.

        docker build -t optiforest .

   The original code needs Python 3.8 with `numpy==1.20.1` and
   `scikit-learn==0.22.1`, which do not install on current Python versions. The
   Dockerfile pins them. PyOD, for ECOD, is installed with `--no-deps` so it
   cannot replace them, and the build fails if a pin has moved.

3. The data scripts and the LLM judge run on your own machine, with Python 3,
   numpy and pandas. scipy is also needed for Arrhythmia's `.mat` file.

4. For the LLM judge only, install [Ollama](https://ollama.com) and download the
   model:

        ollama pull qwen2.5:7b

Run every command from the repository folder. Docker only sees the folders
mounted with `-v`, so each command mounts the data and results folders it needs.

---

## Quick start

The synthetic dataset is included, so this works straight after setup. It runs
each method once and takes about 4 minutes.

    docker run --rm -v "$PWD/data/constructed:/app/data" optiforest \
        python baselines/run_baselines.py --dataset=gauss_k5_d5_sd3_a5_s0.4 \
        --detector=all --repeats=1

Example output:

    Dataset: gauss_k5_d5_sd3_a5_s0.4
      5,000 rows, 5 features, 250 anomalies (5.00%)
      mean row norm 6.2
      chance level: AUC-ROC 0.5000, AUC-PR 0.0500
      OptIForest threshold 55 (auto), branch 0

    L2OPT:
        AUC score:       0.9824
        AUC std:         0.0000
        PR score:        0.9055
        PR std:          0.0000
        Training time:   95.68
        Testing time:    98.81
        repeats:         1

    L2SH:
        AUC score:       0.9822
        AUC std:         0.0000
        PR score:        0.8963
        PR std:          0.0000
        Training time:   4.51
        Testing time:    15.85
        repeats:         1

    iForest:
        AUC score:       0.9612
        AUC std:         0.0000
        PR score:        0.8384
        PR std:          0.0000
        Training time:   0.09
        Testing time:    0.10
        repeats:         1

    ECOD:
        AUC score:       0.8552
        AUC std:         0.0000
        PR score:        0.5303
        PR std:          0.0000
        Training time:   0.01
        Testing time:    0.00
        repeats:         1

`L2OPT` is OptIForest and `L2SH` is LSHiForest, the names the authors' code
uses. AUC score is AUC-ROC and PR score is AUC-PR.

**AUC-ROC** measures how well anomalies are ranked above normal points: 0.5 is
chance and 1 is perfect. **AUC-PR** is stricter when anomalies are rare, and its
chance level equals the anomaly rate.

OptIForest and LSHiForest have no random seed, so your numbers will differ a
little from the example. The results in
`results/constructed/gauss_k5_d5_sd3_a5_s0.4.csv` are the average of 15 runs,
which is the default. Add `--out` to save results to a CSV.

The authors' own script also runs straight after setup, on their sample data.
It does 15 runs and takes about 40 minutes:

    docker run --rm optiforest

---

## Step by step

### 1. The paper's datasets

| Dataset | Source | File |
| --- | --- | --- |
| AD | Included in this repository | `data/ad.csv` |
| Ionosphere | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/52/ionosphere) | `ionosphere.data` |
| Arrhythmia | [PyOD's example data](https://github.com/yzhao062/pyod/tree/master/notebooks/data) | `arrhythmia.mat` |
| Vowel, Waveform, cardio, SpamBase, mnist, satellite, shuttle | [ADBench](https://github.com/Minqi824/ADBench) | `40_vowels.npz` and so on |

The ADBench files are in its `adbench/datasets/Classical` folder. The commands
below expect downloaded files in a `raw` folder next to the repository
(`../raw/`).

Convert each file to a CSV with the label in the last column. `convert.py`
reports the **mean row norm**, which shows whether the data needs rescaling.

    python3 scripts/convert.py ../raw/6_cardio.npz --outdir data/benchmarks
    python3 scripts/convert.py ../raw/35_SpamBase.npz --outdir data/benchmarks --divide 50
    python3 scripts/convert.py ../raw/35_SpamBase.npz --outdir data/benchmarks --standardise
    python3 scripts/convert.py ../raw/40_vowels.npz --outdir data/benchmarks --name Vowel

The output is named after the input file, for example `cardio.csv`. Vowel and
Ionosphere were given the paper's names with `--name`.

There are two ways to rescale:

- `--divide K` divides every value by K. This shrinks the data without changing
  its shape, and has the same effect as making OptIForest's hash bins K times
  wider. The code fixes the bin width at 4.
- `--standardise` rescales each column to mean 0 and standard deviation 1.

Which one works depends on the dataset. The divisor used for each dataset is in
its results file name, for example `SpamBase_div50.csv`.

Then run all four methods:

    docker run --rm \
        -v "$PWD/data/benchmarks:/app/data" \
        -v "$PWD/results/benchmarks:/app/results" \
        optiforest \
        python baselines/run_baselines.py --dataset=SpamBase_div50 --detector=all \
        --out=/app/results/SpamBase_div50.csv

`run_baselines.py` follows `demo.py`: 100 trees, 15 repeats, AUC-ROC and AUC-PR.
ECOD runs once because it gives the same answer every time. OptIForest's cut
threshold follows the paper's rule: 403 for datasets over 10,000 rows or 1,000
features, otherwise 55.

AD is already in the Docker image, so it needs no data folder mounted:

    docker run --rm -v "$PWD/results/benchmarks:/app/results" optiforest \
        python baselines/run_baselines.py --dataset=ad --detector=all \
        --out=/app/results/ad.csv

### 2. Existing datasets new to OptIForest

| Dataset | Source | Prepared with |
| --- | --- | --- |
| Pendigits, PageBlocks | [ADBench](https://github.com/Minqi824/ADBench) | `convert.py` |
| Pima | [ADBench](https://github.com/Minqi824/ADBench) | `prepare_pima.py`, then `convert.py` |
| UNSW-NB15 | [UNSW Canberra](https://research.unsw.edu.au/projects/unsw-nb15-dataset) | `prepare_unsw_nb15.py` |
| Credit Card | [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) | `prepare_creditcard.py` |

    python3 scripts/convert.py ../raw/29_Pima.npz --outdir data/new --standardise
    python3 scripts/prepare_pima.py ../raw/29_Pima.npz --drop-missing
    python3 scripts/convert.py ../raw/pima_no_missing.npz --outdir data/new --standardise
    python3 scripts/prepare_unsw_nb15.py --rows-frac=0.10 --standardise
    python3 scripts/prepare_creditcard.py --rows-frac=0.10 --standardise

`prepare_pima.py` reports the missing values, which Pima records as 0, and
`--drop-missing` writes a copy without those rows. UNSW-NB15 is 68% attacks, so
its script samples the attacks down to 5%. Both the UNSW-NB15 and Credit Card
scripts keep a fraction of the rows (`--rows-frac`).

Run the methods as in step 1, mounting `data/new` and `results/new` instead.

### 3. Datasets built for this study

Both are already in `data/constructed/`.

**Synthetic Gaussian data.** Normal points in Gaussian clusters, with anomalies
scattered at random:

    python3 scripts/make_gaussian.py --clusters=5 --dims=5 --cluster-sd=3 \
        --anomaly-rate=0.05 --scale=0.4

The random seed is fixed, so this rebuilds the file in the repository exactly.
The file name gives the settings: `gauss_k5_d5_sd3_a5_s0.4` is 5 clusters, 5
dimensions, cluster spread 3, 5% anomalies and scale 0.4.

**Wikipedia edits.** Edits from Wikipedia's public API, leaving out edits marked as
made by bots, taken from between 3 and 30 days before the collection so the label
has had time to appear. The label is Wikipedia's own `mw-reverted` tag for an edit
that was later undone.

    python3 scripts/wiki_collect.py --target=20000 --name=wiki_edits_nobots

This writes four files:

| File | What it holds |
| --- | --- |
| `wiki_edits_nobots.csv` | Seven numbers per edit, such as its size and whether the editor was registered, plus the label |
| `wiki_edits_nobots_std.csv` | The same, standardised. The methods run on this file. |
| `wiki_edits_nobots_info.csv` | Each row's edit ID, article title, time and edit summary |
| `wiki_edits_nobots_sample.json` | 50 raw records as the API returns them |

No usernames are kept. Edit summaries often name another editor, for example
"Reverted 1 edit by ...", so those names are replaced with `[user]`. The seven
numbers are worked out before this, so it does not change the data.

Wikipedia changes all the time, so a new collection gives different edits. The
files used in the study are the ones included here.

**An earlier collection, with bots.** `wiki_edits_with_bots.csv` and
`wiki_edits_with_bots_std.csv` are an earlier collection of 20,000 edits, made on
6 October 2026 with an earlier version of `wiki_collect.py` that kept edits marked
as bot edits. Its results are in `results/constructed/wiki_edits_with_bots_std.csv`
and are used in the report only for comparison. That earlier version did not keep
edit IDs or summaries, so there is no `_info.csv` for it, and its raw sample is not
published because it contains usernames. To build a comparable collection with
the current script:

    python3 scripts/wiki_collect.py --include-bots --name=wiki_edits_with_bots_new

Run the methods as in step 1, mounting `data/constructed` and
`results/constructed`.

### 4. The LLM judge (Wikipedia only)

First save OptIForest's score for every edit, averaged over 15 runs. This takes
about 1.5 hours:

    docker run --rm \
        -v "$PWD/data/constructed:/app/data" \
        -v "$PWD/results/constructed:/app/results" \
        optiforest \
        python baselines/score_rows.py --dataset=wiki_edits_nobots_std \
        --out=/app/results/wiki_edits_nobots_scores.csv

Then, with Ollama running:

    python3 scripts/llm_judge.py --top=50 --random=50 --out=results/constructed/wiki_judge.csv

The judge takes OptIForest's 50 most unusual edits and 50 random ones, fetches
what each edit changed, and asks the model to rate it from 1 (good-faith) to 5
(vandalism) with a one-sentence reason. The model is not told who made the edit.
A rating of 4 or 5 counts as harmful, and the script compares that with the
reverted tag.

It takes about 35 minutes and needs an internet connection to fetch the edits.
Each rating is saved as it goes, so if the run stops, the same command carries on
where it left off.

Example output:

    1/100 top    score 1  reverted 0  14.0s  Department of Government Efficiency
    2/100 top    score 2  reverted 0  14.5s  Emerico Lobo de Mesquita
    ...

    100 edits rated
      harmful and reverted 7, harmful only 11, reverted only 9, neither 73
      agreement 0.80, Cohen's kappa 0.29
      top   : 50 edits, mean LLM score 1.96, rated harmful 13, reverted 14
      random: 50 edits, mean LLM score 1.44, rated harmful 5, reverted 2

---

## Results

| Folder | Datasets |
| --- | --- |
| `results/benchmarks/` | The paper's datasets |
| `results/new/` | Existing datasets new to OptIForest |
| `results/constructed/` | The Gaussian and Wikipedia datasets, plus OptIForest's per-edit scores and the LLM judge's ratings |

Each results file has one row per method: the AUC-ROC and AUC-PR averaged over
the runs with their standard deviations, the training and testing time per run
in seconds, and OptIForest's threshold.

---

Environment: MacBook Pro 16-inch 2019, Intel Core i9, macOS. Docker, Python 3.8.
The code is single-threaded and uses one core.
