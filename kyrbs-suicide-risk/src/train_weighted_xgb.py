"""Weighted XGBoost model training for KYRBS suicide-related outcomes.

This primary training script reports default-threshold discrimination and
classification metrics and exports held-out predictions. Peer-review-specific
alternative threshold analyses are implemented separately in
revision_threshold_sensitivity.py so the primary analysis remains distinct
from the exploratory threshold sensitivity analysis.
"""

from __future__ import annotations

import json
import pickle
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import (
    GroupShuffleSplit,
    RandomizedSearchCV,
    StratifiedGroupKFold,
)
from xgboost import XGBClassifier


DATA_CSV = "analysis_variables_only.csv"
YEAR_COL = "YEAR"
WEIGHT_COL = "W"
PSU_COL = "PSU_ID"

# Change to M_SUI_PLAN or M_SUI_ATT for the other primary outcomes.
TARGET_COL = "M_SUI_CON"

# IMPORTANT: variables with substantial survey-year structural non-availability
# (e.g., loneliness before 2020) should not be inserted into the full-period
# primary model. Such variables are evaluated in restricted-wave sensitivity
# analyses as described in the manuscript.
FEATURE_COLS = [
    "M_SAD", "M_STR", "M_SLP_EN", "V_TRT_BIN",
    "PR_HT", "PR_BI", "SEX",
    "AS_DG_LT", "RH_DG_LT", "ECZ_DG_LT",
    "PA_TOT", "PA_VIG_D", "PA_MSC",
    "WEEKDAY_SLEEP_CATEGORY", "AC_LT", "TC_LT", "DR_HAB_PUR", "S_SI",
    "F_BR", "F_FRUIT", "F_FASTFOOD", "F_CAFF_A", "F_WAT",
    "E_SES", "E_S_RCRD", "CTYPE",
]

RANDOM_STATE = 42
N_JOBS = -1


def recode_target_2_3_to_01(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").map({2: 0, 3: 1})


def build_core_dataset(df: pd.DataFrame):
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
    return (
        df_core[FEATURE_COLS].copy(),
        y.loc[core_mask],
        w_new.loc[core_mask],
        df_core[PSU_COL],
    )


def split_with_class_check(
    X, y, w, groups, test_size=0.2, max_tries=200, seed=RANDOM_STATE
):
    rng = np.random.RandomState(seed)
    for _ in range(max_tries):
        rs = int(rng.randint(0, 10_000_000))
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=rs)
        tr, te = next(gss.split(X, y, groups=groups))
        if y.iloc[tr].nunique() == 2 and y.iloc[te].nunique() == 2:
            return (
                X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te],
                w.iloc[tr], w.iloc[te], groups.iloc[tr], groups.iloc[te],
            )
    raise ValueError("Could not produce train/test sets containing both classes.")


def weighted_brier_score(y_true, y_prob, sample_weight=None):
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    if sample_weight is None:
        sample_weight = np.ones_like(y_true)
    return float(np.average((y_prob - y_true) ** 2, weights=sample_weight))


def make_base_xgb():
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


def fit_with_cv(X_train, y_train, w_train, groups_train):
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
    """Evaluate the primary model at the prespecified default threshold 0.50."""
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.50).astype(int)

    if not (len(y_test) == len(y_proba) == len(w_test)):
        raise ValueError("Test labels, probabilities, and weights are misaligned.")

    fpr, tpr, _ = roc_curve(y_test, y_proba, sample_weight=w_test)
    auc_roc = auc(fpr, tpr)
    ap = average_precision_score(y_test, y_proba, sample_weight=w_test)

    metrics = OrderedDict([
        ("threshold", 0.50),
        ("auc_roc", float(auc_roc)),
        ("average_precision", float(ap)),
        ("positive_precision", float(precision_score(
            y_test, y_pred, average="binary", pos_label=1,
            sample_weight=w_test, zero_division=0
        ))),
        ("positive_recall", float(recall_score(
            y_test, y_pred, average="binary", pos_label=1,
            sample_weight=w_test, zero_division=0
        ))),
        ("positive_f1", float(f1_score(
            y_test, y_pred, average="binary", pos_label=1,
            sample_weight=w_test, zero_division=0
        ))),
        ("weighted_f1", float(f1_score(
            y_test, y_pred, average="weighted",
            sample_weight=w_test, zero_division=0
        ))),
        ("accuracy", float(accuracy_score(y_test, y_pred, sample_weight=w_test))),
        ("brier", weighted_brier_score(y_test, y_proba, sample_weight=w_test)),
    ])

    cm_weighted = confusion_matrix(y_test, y_pred, sample_weight=w_test)
    frac_pos, mean_pred = calibration_curve(
        y_test, y_proba, n_bins=10, strategy="quantile"
    )
    return {
        "metrics": metrics,
        "confusion_default_weighted": cm_weighted.tolist(),
        "calibration_curve": {
            "mean_pred": mean_pred.tolist(),
            "frac_pos": frac_pos.tolist(),
        },
        "y_proba": y_proba,
        "y_pred": y_pred,
    }


def main():
    outcome_suffix = {
        "M_SUI_CON": "ideation",
        "M_SUI_PLAN": "planning",
        "M_SUI_ATT": "attempt",
    }.get(TARGET_COL, TARGET_COL.lower())
    outdir = Path(f"outputs_{outcome_suffix}")
    outdir.mkdir(exist_ok=True)

    df = pd.read_csv(DATA_CSV, encoding="utf-8-sig", low_memory=False)
    X_core, y_core, w_core, g_core = build_core_dataset(df)
    X_tr_raw, X_te_raw, y_tr, y_te, w_tr, w_te, g_tr, _ = split_with_class_check(
        X_core, y_core, w_core, g_core, test_size=0.2
    )

    imp = SimpleImputer(strategy="most_frequent")
    X_tr = pd.DataFrame(imp.fit_transform(X_tr_raw), columns=X_core.columns)
    X_te = pd.DataFrame(imp.transform(X_te_raw), columns=X_core.columns)

    model, cv_results = fit_with_cv(X_tr, y_tr, w_tr, g_tr)
    eval_results = evaluate_model(model, X_te, y_te, w_te)

    model.save_model(outdir / "xgb_model.json")
    with open(outdir / "imputer.pkl", "wb") as f:
        pickle.dump(imp, f)
    with open(outdir / "cv_results.json", "w", encoding="utf-8") as f:
        json.dump(cv_results, f, indent=2)

    # JSON excludes arrays used for downstream revision analyses.
    with open(outdir / "eval_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "metrics": eval_results["metrics"],
            "confusion_default_weighted": eval_results["confusion_default_weighted"],
            "calibration_curve": eval_results["calibration_curve"],
        }, f, indent=2)

    # Export the exact aligned held-out observations used by Table S12.
    pd.DataFrame({
        "y_true": np.asarray(y_te).ravel(),
        "y_prob": np.asarray(eval_results["y_proba"]).ravel(),
        "sample_weight": np.asarray(w_te).ravel(),
    }).to_csv(outdir / f"predictions_{outcome_suffix}.csv", index=False)

    print("[Training completed]")
    print("Best CV score:", cv_results)
    print("Evaluation:", eval_results["metrics"])
    print(f"Aligned predictions saved in: {outdir}")


if __name__ == "__main__":
    main()
