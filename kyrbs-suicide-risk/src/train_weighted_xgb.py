"""
Weighted XGBoost model training for KYRBS suicidal ideation/plan/attempt.
All comments fully translated to English for public GitHub release.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
from collections import OrderedDict

from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit, RandomizedSearchCV, StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, precision_recall_curve, average_precision_score,
    brier_score_loss, confusion_matrix, make_scorer
)
from sklearn.calibration import calibration_curve

from xgboost import XGBClassifier


# --------------------------------------------------------
# Configuration
# --------------------------------------------------------
DATA_CSV = "analysis_variables_only.csv"

YEAR_COL = "YEAR"
WEIGHT_COL = "W"
PSU_COL = "PSU_ID"

# Change this to M_SUI_PLAN or M_SUI_ATT for other outcomes
TARGET_COL = "M_SUI_CON"

FEATURE_COLS = [
    "M_SAD", "M_STR", "M_LON", "M_SLP_EN", "V_TRT_BIN",
    "PR_HT", "PR_BI", "SEX",
    "AS_DG_LT", "RH_DG_LT", "ECZ_DG_LT",
    "PA_TOT", "PA_VIG_D", "PA_MSC",
    "WEEKDAY_SLEEP_CATEGORY", "AC_LT", "TC_LT", "DR_HAB_PUR", "S_SI",
    "F_BR", "F_FRUIT", "F_FASTFOOD", "F_CAFF_A", "F_WAT",
    "E_SES", "E_S_RCRD", "CTYPE",
]

RANDOM_STATE = 42
N_JOBS = -1


# --------------------------------------------------------
# Utilities
# --------------------------------------------------------

def recode_target_2_3_to_01(series: pd.Series) -> pd.Series:
    """
    Convert suicidal variables: 2 = No, 3 = Yes → 0/1.
    """
    return pd.to_numeric(series, errors="coerce").map({2: 0, 3: 1})


def build_core_dataset(df: pd.DataFrame):
    """
    Build core analysis dataset using:
    - shift-adjusted target
    - probability weights W / number of years
    - PSU-based grouping
    """
    n_years = df[YEAR_COL].nunique()
    w_new = pd.to_numeric(df[WEIGHT_COL], errors="coerce") / max(n_years, 1)

    y = recode_target_2_3_to_01(df[TARGET_COL])

    core_mask = (
        y.notna()
        & w_new.notna()
        & df[YEAR_COL].notna()
        & df[PSU_COL].notna()
    )

    df_core = df.loc[core_mask].copy()
    y_core = y.loc[core_mask]
    w_core = w_new.loc[core_mask]
    g_core = df_core[PSU_COL]

    X_core = df_core[FEATURE_COLS].copy()

    return X_core, y_core, w_core, g_core


def split_with_class_check(
    X, y, w, groups,
    test_size: float = 0.2,
    max_tries: int = 200,
    seed: int = RANDOM_STATE,
):
    """
    GroupShuffleSplit based on PSU.
    Ensures both train/test contain both class 0 and 1.
    Retries with different seeds until successful.
    """
    rng = np.random.RandomState(seed)
    for _ in range(max_tries):
        rs = int(rng.randint(0, 10_000_000))
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=rs)
        tr, te = next(gss.split(X, y, groups=groups))
        if y.iloc[tr].nunique() == 2 and y.iloc[te].nunique() == 2:
            return (
                X.iloc[tr], X.iloc[te],
                y.iloc[tr], y.iloc[te],
                w.iloc[tr], w.iloc[te],
                groups.iloc[tr], groups.iloc[te],
            )
    raise ValueError("Failed to produce a split containing both classes 0 and 1 in train/test.")


def weighted_brier_score(y_true, y_prob, sample_weight=None) -> float:
    """Weighted Brier score."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    if sample_weight is None:
        sample_weight = np.ones_like(y_true)
    sample_weight = np.asarray(sample_weight, dtype=float)
    return np.average((y_prob - y_true) ** 2, weights=sample_weight)


def make_base_xgb() -> XGBClassifier:
    """Base XGBoost model used for hyperparameter search."""
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        tree_method="hist",
        n_estimators=500,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1.0,
        n_jobs=N_JOBS,
    )

def fit_with_cv(
    X_train, y_train, w_train, groups_train
):
    """
    Fit model with hyperparameter tuning using:
    - StratifiedGroupKFold
    - RandomizedSearchCV
    - Weighted AUROC as scoring metric
    """
    base_model = make_base_xgb()

    param_dist = {
        "max_depth": [3, 4, 5],
        "learning_rate": [0.02, 0.03, 0.05],
        "n_estimators": [400, 600, 800],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "min_child_weight": [1.0, 2.0, 5.0],
    }

    def weighted_roc_auc(y_true, y_pred, sample_weight=None):
        fpr, tpr, _ = roc_curve(y_true, y_pred, sample_weight=sample_weight)
        return auc(fpr, tpr)

    scorer = make_scorer(weighted_roc_auc, needs_proba=True)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=30,
        scoring=scorer,
        n_jobs=N_JOBS,
        cv=cv.split(X_train, y_train, groups_train),
        verbose=1,
        random_state=RANDOM_STATE,
        refit=True,
    )
    search.fit(X_train, y_train, sample_weight=w_train)

    return search.best_estimator_, {
        "best_params": search.best_params_,
        "best_score": float(search.best_score_),
    }


def evaluate_model(model, X_test, y_test, w_test):
    """
    Evaluate model performance:
    - AUROC / AUPRC
    - accuracy / precision / recall / F1 (default 0.5 threshold)
    - optimal threshold maximizing weighted F1
    - Brier score
    - confusion matrices
    - calibration curve
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred_default = (y_proba >= 0.5).astype(int)

    fpr, tpr, _ = roc_curve(y_test, y_proba, sample_weight=w_test)
    auc_roc = auc(fpr, tpr)

    precision, recall, thr_pr = precision_recall_curve(y_test, y_proba, sample_weight=w_test)
    auprc = average_precision_score(y_test, y_proba, sample_weight=w_test)

    # Find best threshold
    f1_scores = []
    for thr in thr_pr:
        if 0 < thr < 1:
            pred_thr = (y_proba >= thr).astype(int)
            f1_scores.append((
                thr,
                f1_score(
                    y_test, pred_thr,
                    average="weighted", zero_division=0,
                    sample_weight=w_test
                )
            ))

    best_thr, best_f1 = max(f1_scores, key=lambda x: x[1]) if f1_scores else (0.5, f1_score(
        y_test, y_pred_default, average="weighted", zero_division=0, sample_weight=w_test))

    y_pred_opt = (y_proba >= best_thr).astype(int)

    cm_default = confusion_matrix(y_test, y_pred_default)
    cm_opt = confusion_matrix(y_test, y_pred_opt)

    frac_pos, mean_pred = calibration_curve(y_test, y_proba, n_bins=10, strategy="quantile")

    metrics = OrderedDict([
        ("auc_roc", float(auc_roc)),
        ("auprc", float(auprc)),
        ("best_threshold", float(best_thr)),
        ("f1_opt", float(best_f1)),
        ("brier", float(weighted_brier_score(y_test, y_proba, sample_weight=w_test))),
    ])

    return {
        "metrics": metrics,
        "confusion_default": cm_default.tolist(),
        "confusion_opt": cm_opt.tolist(),
        "calibration_curve": {
            "mean_pred": mean_pred.tolist(),
            "frac_pos": frac_pos.tolist(),
        },
    }


def main():
    outdir = Path("outputs_suicidal_ideation")
    outdir.mkdir(exist_ok=True)

    df = pd.read_csv(DATA_CSV, encoding="utf-8-sig", keep_default_na=False, na_values=[], low_memory=False)
    X_core, y_core, w_core, g_core = build_core_dataset(df)

    X_tr_raw, X_te_raw, y_tr, y_te, w_tr, w_te, g_tr, g_te = split_with_class_check(
        X_core, y_core, w_core, g_core, test_size=0.2
    )

    imp = SimpleImputer(strategy="most_frequent")
    X_tr = pd.DataFrame(imp.fit_transform(X_tr_raw), columns=X_core.columns)
    X_te = pd.DataFrame(imp.transform(X_te_raw), columns=X_core.columns)

    model, cv_results = fit_with_cv(X_tr, y_tr, w_tr, g_tr)
    eval_results = evaluate_model(model, X_te, y_te, w_te)

    # Save artifacts
    model.save_model(outdir / "xgb_model.json")

    import pickle
    with open(outdir / "imputer.pkl", "wb") as f:
        pickle.dump(imp, f)

    with open(outdir / "cv_results.json", "w", encoding="utf-8") as f:
        json.dump(cv_results, f, indent=2)

    with open(outdir / "eval_results.json", "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)

    print("[Training completed]")
    print("Best CV score:", cv_results)
    print("Evaluation:", eval_results["metrics"])


if __name__ == "__main__":
    main()
