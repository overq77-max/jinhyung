# Adolescent Suicidality in South Korea: Explainable Machine Learning and Correlation Network Analysis

Code repository accompanying the manuscript:

**"Explainable machine learning and correlation-based network analysis of self-reported adolescent suicidality in South Korea: a repeated cross-sectional study, 2015–2024"**

The analysis uses Korean Youth Risk Behavior Survey (KYRBS) data from 2015–2024. The repository contains preprocessing, survey-weighted machine-learning utilities, model evaluation code, and descriptive correlation-network analysis.

> Raw KYRBS microdata are **not distributed in this repository**. Users must obtain the data independently from the Korea Disease Control and Prevention Agency (KDCA) and comply with the applicable data-use requirements.

---

## Repository structure

```text
jinhyung/
├── README.md
└── kyrbs-suicide-risk/
    ├── requirements.txt
    └── src/
        ├── preprocess_kyrbs.py
        ├── train_weighted_xgb.py
        ├── build_corr_network.py
        ├── revision_table2_models.py
        ├── revision_threshold_sensitivity.py
        └── revision_figure2_roc.py
```

Generated analysis outputs are intentionally not committed as source files.

---

## Installation

```bash
git clone https://github.com/overq77-max/jinhyung.git
cd jinhyung/kyrbs-suicide-risk
pip install -r requirements.txt
```

Place the locally obtained input data in the analysis working directory before running the scripts. File names and variable mappings may need to be aligned with the locally harmonized KYRBS dataset.

---

## Primary analysis

### 1. Data preprocessing

`src/preprocess_kyrbs.py`

Prepares harmonized analysis variables and derived features used by the downstream analysis.

### 2. Weighted XGBoost analysis

`src/train_weighted_xgb.py`

The script:

- constructs the outcome-specific analysis sample;
- uses the across-wave adjusted survey weight;
- performs a PSU-grouped train/test split;
- applies training-set-fitted imputation;
- tunes XGBoost with grouped cross-validation;
- reports AUROC, average precision (AP), positive-class precision/recall/F1, weighted F1, accuracy, and Brier score at the default threshold of 0.50; and
- exports the **aligned held-out labels, probabilities, and survey weights** required for the peer-review threshold analysis.

Change `TARGET_COL` to run the three primary outcomes:

```python
TARGET_COL = "M_SUI_CON"   # suicidal ideation
TARGET_COL = "M_SUI_PLAN"  # suicidal planning
TARGET_COL = "M_SUI_ATT"   # suicide attempt
```

Survey year is not used as an individual-level predictor in the primary classification model. Temporal robustness and survey-year-specific structural non-availability are addressed separately in the manuscript analyses.

Variables with substantial structural non-availability across the full 2015–2024 period (for example, loneliness before 2020) should not be inserted into the full-period primary model. Such variables are evaluated in restricted-wave sensitivity analyses.

---

## Peer-review revision analyses

The following scripts document analyses added or updated during revision for **BMC Public Health**.

### Revised Table 2 model comparison

`src/revision_table2_models.py`

Provides functions for evaluating XGBoost, Random Forest, and Logistic Regression on the **same outcome-specific held-out sample**. Reported metrics include:

- AUROC
- AP/AUPRC
- positive-class precision
- positive-class recall (sensitivity)
- positive-class F1
- weighted F1
- accuracy

The module is intended to be imported after the outcome-specific preprocessing and train/test split have been created. This design helps prevent accidental mixing of probabilities, labels, or survey weights from different outcomes or splits.

### Supplementary Table S12: threshold sensitivity analysis

`src/revision_threshold_sensitivity.py`

Compares:

1. default threshold = 0.50;
2. threshold maximizing **Youden's J**; and
3. threshold maximizing **survey-weighted positive-class F1**.

For each threshold the script reports sensitivity, specificity, PPV, NPV, positive-class F1, balanced accuracy, and accuracy.

Example:

```bash
python src/revision_threshold_sensitivity.py \
  outputs_ideation/predictions_ideation.csv \
  --outcome "Suicidal ideation" \
  --output outputs_revision/threshold_sensitivity_ideation.csv
```

The alternative thresholds are exploratory because threshold selection and evaluation use the same held-out predictions. They should **not** be interpreted as clinically validated screening or triage cutoffs.

### Revised Figure 2: ROC curves

`src/revision_figure2_roc.py`

Generates an outcome-specific ROC panel using the same fitted XGBoost, Random Forest, and Logistic Regression models evaluated for revised Table 2. This keeps the plotted AUROC values synchronized with the table.

### Revised Figure 4: descriptive correlation network

`src/build_corr_network.py`

The revised visualization:

- calculates survey-weighted pairwise Pearson correlations;
- retains edges with **|r| >= 0.10** only to reduce visual complexity;
- maps retained edge width **continuously** to absolute correlation magnitude;
- uses edge color to distinguish positive and negative correlations; and
- treats the network as an exploratory map of **marginal cross-sectional associations**, not a causal or conditional-dependence network.

Alternative graphical models for mixed/categorical data (e.g., GGM, polychoric/tetrachoric approaches, or Ising models) are discussed in the manuscript as future directions rather than implemented here.

---

## Reproducibility safeguards

Because the three suicide-related outcomes have different outcome-specific samples, all downstream metrics must use labels, probabilities, and survey weights generated from the **same fitted model and held-out split**. The revision scripts include length/alignment checks to help detect accidental reuse of predictions from another outcome.

The primary model uses a default classification threshold of 0.50. Alternative threshold selection is kept in a separate revision script so that exploratory threshold optimization is not confused with the primary model evaluation.

---

## Requirements

See `kyrbs-suicide-risk/requirements.txt`.

Main Python dependencies are:

- pandas
- numpy
- scikit-learn
- xgboost
- networkx
- matplotlib

---

## Data availability

KYRBS data are available from the Korea Disease Control and Prevention Agency (KDCA). The repository provides analysis code but does not redistribute the raw survey microdata.

---

## License

MIT License  
© CHA University
