# Explainable Machine Learning for Transparent Student Performance Prediction

Reproducible code for: *Explainable Machine Learning for Transparent Student
Performance Prediction: A Reproducible Framework for Data-Driven Learning
Analytics* (Altuwayjiri, submitted to PeerJ Computer Science as an AI
Application article).

## Description

This repository contains the full, executable pipeline behind every number,
table, and figure in the manuscript: data preprocessing, hyperparameter
search for three classifiers (Logistic Regression, Random Forest, and a
histogram-based gradient-boosting ensemble), repeated cross-validation with
paired significance testing, hand-implemented Kernel SHAP and LIME
explainers, a systematic SHAP-LIME agreement study, and an independent
permutation-importance cross-check. Every result in the paper is produced by
re-running the scripts in `experiments/` with the fixed random seed reported
in the manuscript (seed = 42).

The pipeline evaluates whether post-hoc explainability tools (SHAP, LIME)
can be layered onto a tuned, cross-validated classifier for an educational
early-warning task without degrading predictive accuracy, and whether the
two explainers agree with one another and with an independent
permutation-importance baseline — reporting the answer honestly rather than
through a single, potentially favorable illustrative case.

## Dataset Information

This study uses the public **UCI Student Performance dataset** (Mathematics
cohort), originally curated by Cortez & Silva (2008).

- Source / DOI: https://doi.org/10.24432/C5TG7T
- Dataset URL: https://archive.ics.uci.edu/dataset/320/student+performance
- File used: `data/student-mat.csv` (395 students, 33 raw columns,
  semicolon-delimited)
- License: the dataset is distributed by the UCI Machine Learning
  Repository for research use; see the dataset page for full terms.

No new data were collected for this study. The CSV file is included in
`data/` for convenience and exact reproducibility; it can also be
downloaded fresh from the URL above.

## Code Information

```
.
├── README.md
├── requirements.txt
├── data/
│   └── student-mat.csv          # UCI Student Performance (Math), as used
├── src/
│   ├── preprocess.py            # Encoding protocol (-> 39 features, 395 rows)
│   ├── models.py                # LR / RF / HistGB, RandomizedSearchCV,
│   │                             #   repeated CV, paired significance tests
│   └── shap_lime_impl.py        # Hand-implemented Kernel SHAP and LIME
├── experiments/
│   ├── 01_full_pipeline.py      # Table 2, Table 3, Table 4, Section 4.4
│   ├── 02_explainability.py     # Figure 3, Table 6, Section 4.6 agreement
│   └── 03_build_report_artifacts.py
│                                 # Consolidated run: regenerates every
│                                 #   number and figure (Figures 2 & 3) used
│                                 #   in the manuscript in one pass
└── outputs/                     # Created on run: full_report.json,
                                  #   figure2_confusion_matrices.png,
                                  #   figure3_global_shap.png
```

`experiments/03_build_report_artifacts.py` is the canonical, single-command
entry point: it reproduces every reported number and both figures in one
deterministic run and writes `outputs/full_report.json`, which is the exact
source of every statistic quoted in the manuscript's Results section.

## Usage Instructions

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Reproduce every number and figure in the manuscript (Tables 2-4, 6; Figures 2-3)
python3 experiments/03_build_report_artifacts.py

# Or run the two stages separately:
python3 experiments/01_full_pipeline.py      # classification + significance tests
python3 experiments/02_explainability.py     # SHAP/LIME + permutation importance
```

All scripts are run from the repository root (so that `data/student-mat.csv`
resolves correctly) and write their output to `outputs/`.

## Requirements

- Python 3.11+ (developed and verified on Python 3.13.16)
- See `requirements.txt` for exact pinned package versions:
  scikit-learn, numpy, pandas, scipy, matplotlib.
- No GPU is required. Developed and verified on a 2-core, 8 GB RAM Linux
  x86-64 machine; the full pipeline (both experiment scripts) runs in a few
  minutes on that hardware.

## Methodology

- **Preprocessing** (`src/preprocess.py`): binary yes/no fields mapped to
  0/1; two-level nominal fields and multi-level nominal fields both
  one-hot encoded with a dropped reference level (`pandas.get_dummies(...,
  drop_first=True)`), reproducing the manuscript's reported 39-feature
  matrix exactly. Intermediate grades G1/G2 are dropped to avoid target
  leakage; the target is `Y = 1` if `G3 >= 10` else `0`.
- **Model search** (`src/models.py`): `RandomizedSearchCV` (5-fold
  stratified CV, scoring = ROC-AUC) on an 80/20 stratified split, 7
  candidates for Logistic Regression's regularization strength and 25
  candidates each for Random Forest and the gradient-boosting ensemble.
  The resulting tuned hyperparameters are then evaluated with repeated
  stratified 5-fold cross-validation (5 repeats, 25 folds, shared fold
  partitions) on the full 395-student dataset — the manuscript's primary
  performance metric — with paired t-tests and Wilcoxon signed-rank tests
  across the 25 shared folds.
- **Explainability** (`src/shap_lime_impl.py`): Kernel SHAP and LIME are
  implemented directly from their published formulations (Lundberg & Lee,
  2017; Ribeiro et al., 2016), not via the third-party `shap`/`lime`
  packages. SHAP coalitions are weighted per the standard Kernel SHAP
  kernel and solved by weighted least squares; LIME perturbs around each
  instance using background-sample substitution and fits a locally
  weighted ridge regression surrogate.
- **Agreement quantification**: top-5 feature-set Jaccard similarity and
  Spearman rank correlation between |SHAP| and |LIME| attributions,
  computed per-instance across 50 held-out test instances, plus an
  independent permutation-importance cross-check (`scikit-learn`,
  30 repetitions, scored on held-out ROC-AUC) compared against the SHAP
  global ranking via Spearman correlation.
- A fixed random seed (42) is used throughout for the train/test split,
  cross-validation folds, hyperparameter search, and SHAP/LIME sampling,
  and is reported alongside every result for exact re-execution.

## Citations

If you use this code or build on this pipeline, please cite the
manuscript (full citation to be added upon acceptance) and the underlying
dataset:

> Cortez, P., & Silva, A. M. G. (2008). Using data mining to predict
> secondary school student performance. In *Proceedings of 5th Annual
> Future Business Technology Conference*, EUROSIS.

Methods implemented here are drawn from:

> Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting
> model predictions. *Advances in Neural Information Processing Systems*.
>
> Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust
> you?": Explaining the predictions of any classifier. *Proceedings of the
> 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data
> Mining*.

## License & Contribution Guidelines

Code: released under the MIT License (see `LICENSE`). The dataset retains
its original UCI Machine Learning Repository terms; see the dataset DOI
page for details. Issues and pull requests are welcome once the repository
is public; please open an issue describing the proposed change before
submitting a pull request.
