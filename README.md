# Adolescent Suicidality in South Korea: Explainable Machine Learning and Correlation Network Analysis

Code repository accompanying the manuscript **“Explainable machine learning and correlation-based network analysis of self-reported adolescent suicidality in South Korea: a repeated cross-sectional study, 2015–2024.”**

The analysis uses Korean Youth Risk Behavior Survey (KYRBS) data from 2015–2024. Raw KYRBS microdata are not distributed in this repository.

## Source-of-truth alignment

The public scripts have been reconciled against the outcome-specific analysis notebooks for suicidal ideation, suicidal planning, suicide attempt, and the correlation-network analysis. They preserve the conventions that materially affect the analysis: outcome-specific coding, PSU-grouped splitting, survey weighting, the primary full-period feature set, and the final numbered correlation-network construction.

Important variable conventions:

- Suicidal ideation: `M_SUI_CON`, coded 2→0 and 3→1.
- Suicidal planning: accepts `M_SUI_PLN` or `M_SUI_PLAN`, coded 1→0 and 2→1.
- Suicide attempt: `M_SUI_ATT`, coded 1→0 and 2→1.
- The primary prediction models use `V_TRT`.
- `V_TRT_BIN` is a network-specific binary standardization used for the correlation matrix.
- Loneliness (`M_LON`) is excluded from the full 2015–2024 primary model because of structural wave non-availability and is examined separately in the 2020–2024 sensitivity analysis.

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
        ├── revision_figure2_roc.py
        └── revision_loneliness_sensitivity.py
```

## Installation

```bash
git clone https://github.com/overq77-max/jinhyung.git
cd jinhyung/kyrbs-suicide-risk
pip install -r requirements.txt
```

Place the locally harmonized `analysis_variables_only.csv` in the working directory.

## Primary XGBoost analysis

`src/train_weighted_xgb.py` runs the three primary outcomes with the across-wave adjusted survey weight, PSU-preserving train/test splitting, training-set-fitted most-frequent-value imputation after numeric encoding, and 5-fold `StratifiedGroupKFold` tuning. It reports default-threshold (0.50) AUROC, average precision, positive-class precision/recall/F1, weighted F1, accuracy, and Brier score, and exports aligned held-out labels/probabilities/weights.

```bash
python src/train_weighted_xgb.py --outcome all
```

## Revised Table 2

`src/revision_table2_models.py` evaluates XGBoost, Random Forest, and Logistic Regression on the same outcome-specific held-out sample. It reports AUROC, AP/AUPRC, positive-class precision/recall/F1, weighted F1, and accuracy. The same fitted models should be passed to the Figure 2 ROC utility so plotted AUROC values remain synchronized with Table 2.

## Supplementary Table S12

`src/revision_threshold_sensitivity.py` compares the default 0.50 threshold, the threshold maximizing Youden’s J, and the threshold maximizing survey-weighted positive-class F1. It reports sensitivity, specificity, PPV, NPV, positive-class F1, balanced accuracy, and accuracy. Alternative thresholds are exploratory because threshold selection and evaluation use the same held-out predictions; they are not clinically validated screening or triage cutoffs.

## Revised Figure 2

`src/revision_figure2_roc.py` generates outcome-specific ROC curves from the same XGBoost, Random Forest, and Logistic Regression fits used for revised Table 2.

## 2020–2024 loneliness sensitivity analysis

`src/revision_loneliness_sensitivity.py` restricts the harmonized data to 2020–2024, adds `M_LON` to the primary predictor set, refits the outcome-specific XGBoost models using the same PSU-preserving split and survey-weighting framework, and exports aligned predictions and SHAP summaries. These results are supportive sensitivity evidence rather than full-period primary evidence.

```bash
python src/revision_loneliness_sensitivity.py --outcome all
```

## Revised Figure 4: correlation network

`src/build_corr_network.py` reproduces the final descriptive network logic:

- all three suicidality outcomes are included in the same weighted correlation matrix;
- `M_SUI_PLN` and `M_SUI_PLAN` are handled as aliases for suicidal planning;
- `V_TRT_BIN` is generated from `V_TRT` when necessary;
- `PR_BI` is removed from the final displayed network, matching the original node-9 deletion and renumbering;
- edges with `|r| < 0.10` are omitted only for visual filtering;
- all retained edges are solid and edge width is `10 × |r|` continuously;
- positive/negative correlations are shown separately by edge color; and
- weighted betweenness/closeness use `distance = 1 / |r|`.

The Pearson network is a descriptive map of marginal cross-sectional associations. It is not interpreted as a causal, temporal, or conditional-independence network.

## Reproducibility safeguards

The three outcomes have different outcome-specific samples. Labels, probabilities, survey weights, model fits, threshold analyses, and ROC curves must therefore come from the same outcome and the same held-out split. Generated outputs are intentionally not committed as source files. Raw KYRBS data must be obtained independently from the Korea Disease Control and Prevention Agency (KDCA) under the applicable data-use requirements.

## Requirements

Main dependencies: pandas, numpy, scikit-learn, xgboost, shap, networkx, matplotlib.

## License

MIT License  
© CHA University
