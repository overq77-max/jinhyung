"""Primary survey-weighted XGBoost analysis for the KYRBS study.

This script is aligned to the three uploaded Scientific Reports revision
notebooks. The full-period primary models use V_TRT (not the network-only
V_TRT_BIN variable), PSU-preserving train/test splitting, survey weights, and
5-fold StratifiedGroupKFold tuning. Alternative decision thresholds remain a
separate exploratory revision analysis.
"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, auc, average_precision_score,
                             confusion_matrix, f1_score, make_scorer,
                             precision_score, recall_score, roc_curve)
from sklearn.model_selection import GroupShuffleSplit, RandomizedSearchCV, StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

DATA_CSV = "analysis_variables_only.csv"
YEAR_COL = "YEAR"
WEIGHT_COL = "W"
PSU_COL = "PSU_ID"
RANDOM_STATE = 42
N_JOBS = -1

FEATURE_COLS = [
    "M_SAD", "M_STR", "M_SLP_EN", "V_TRT", "PR_HT", "PR_BI", "SEX",
    "AS_DG_LT", "RH_DG_LT", "ECZ_DG_LT", "PA_TOT", "PA_VIG_D", "PA_MSC",
    "WEEKDAY_SLEEP_CATEGORY", "AC_LT", "TC_LT", "DR_HAB_PUR", "S_SI",
    "F_BR", "F_FRUIT", "F_FASTFOOD", "F_CAFF_A", "F_WAT",
    "E_SES", "E_S_RCRD", "CTYPE",
]

OUTCOMES = {
    "ideation": {"candidates": ["M_SUI_CON"], "mapping": {2: 0, 3: 1}},
    "planning": {"candidates": ["M_SUI_PLN", "M_SUI_PLAN"], "mapping": {1: 0, 2: 1}},
    "attempt": {"candidates": ["M_SUI_ATT"], "mapping": {1: 0, 2: 1}},
}


def resolve_target(df: pd.DataFrame, outcome: str):
    spec = OUTCOMES[outcome]
    for col in spec["candidates"]:
        if col in df.columns:
            y = pd.to_numeric(df[col], errors="coerce").map(spec["mapping"])
            return col, y
    raise KeyError(f"No target column found for {outcome}: {spec['candidates']}")


def robust_encode_col(s: pd.Series) -> pd.Series:
    """Mirror the notebook's defensive numeric/categorical encoding."""
    s2 = s.replace({"A": 1, "B": 2, "C": 3, "High": 1, "Middle": 2,
                    "Low": 3, "None": 1, "Experienced": 2})
    num = pd.to_numeric(s2, errors="coerce")
    if num.notna().sum() == 0 and (s2.dtype == "object" or str(s2.dtype).startswith("string")):
        missing = s2.isna()
        le = LabelEncoder()
        enc = pd.Series(le.fit_transform(s2.fillna("<<NA>>").astype(str)), index=s2.index, dtype=float)
        enc[missing] = np.nan
        return enc
    return num.astype(float)


def build_core_dataset(df: pd.DataFrame, outcome: str):
    target_col, y = resolve_target(df, outcome)
    n_years = df[YEAR_COL].nunique()
    w_new = pd.to_numeric(df[WEIGHT_COL], errors="coerce") / max(n_years, 1)
    mask = y.notna() & w_new.notna() & df[YEAR_COL].notna() & df[PSU_COL].notna()
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = pd.DataFrame({c: robust_encode_col(df.loc[mask, c]) for c in available})
    return X, y.loc[mask].astype(int), w_new.loc[mask], df.loc[mask, PSU_COL], target_col


def split_with_class_check(X, y, w, groups, test_size=0.2, max_tries=200, seed=RANDOM_STATE):
    rng = np.random.RandomState(seed)
    for _ in range(max_tries):
        rs = int(rng.randint(0, 10_000_000))
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=rs).split(X, y, groups=groups))
        if y.iloc[tr].nunique() == 2 and y.iloc[te].nunique() == 2:
            return X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te], w.iloc[tr], w.iloc[te], groups.iloc[tr], groups.iloc[te]
    raise ValueError("Could not produce PSU-grouped train/test sets containing both classes.")


def weighted_brier(y_true, y_prob, sample_weight):
    return float(np.average((np.asarray(y_prob)-np.asarray(y_true))**2, weights=np.asarray(sample_weight)))


def make_base_xgb():
    return XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                         random_state=RANDOM_STATE, tree_method="hist", n_jobs=N_JOBS)


def fit_with_cv(X_train, y_train, w_train, groups_train):
    params = {
        "max_depth": [3, 4, 5], "learning_rate": [0.02, 0.03, 0.05],
        "n_estimators": [400, 600, 800], "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 1.0], "min_child_weight": [1.0, 2.0, 5.0],
    }
    def weighted_auc(y_true, y_pred, sample_weight=None):
        fpr, tpr, _ = roc_curve(y_true, y_pred, sample_weight=sample_weight)
        return auc(fpr, tpr)
    scorer = make_scorer(weighted_auc, needs_proba=True)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(make_base_xgb(), params, n_iter=30, scoring=scorer,
                                cv=cv.split(X_train, y_train, groups_train), n_jobs=N_JOBS,
                                random_state=RANDOM_STATE, refit=True, verbose=1)
    search.fit(X_train, y_train, sample_weight=w_train)
    return search.best_estimator_, {"best_params": search.best_params_, "best_score": float(search.best_score_)}


def evaluate_model(model, X_test, y_test, w_test):
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.50).astype(int)
    fpr, tpr, _ = roc_curve(y_test, prob, sample_weight=w_test)
    metrics = OrderedDict([
        ("threshold", 0.50), ("auc_roc", float(auc(fpr, tpr))),
        ("average_precision", float(average_precision_score(y_test, prob, sample_weight=w_test))),
        ("positive_precision", float(precision_score(y_test, pred, sample_weight=w_test, zero_division=0))),
        ("positive_recall", float(recall_score(y_test, pred, sample_weight=w_test, zero_division=0))),
        ("positive_f1", float(f1_score(y_test, pred, sample_weight=w_test, zero_division=0))),
        ("weighted_f1", float(f1_score(y_test, pred, average="weighted", sample_weight=w_test, zero_division=0))),
        ("accuracy", float(accuracy_score(y_test, pred, sample_weight=w_test))),
        ("brier", weighted_brier(y_test, prob, w_test)),
    ])
    cm = confusion_matrix(y_test, pred, labels=[0,1], sample_weight=w_test)
    frac_pos, mean_pred = calibration_curve(y_test, prob, n_bins=10, strategy="quantile")
    return metrics, cm, mean_pred, frac_pos, prob


def run_outcome(df: pd.DataFrame, outcome: str):
    outdir = Path(f"outputs_{outcome}"); outdir.mkdir(exist_ok=True)
    X, y, w, groups, target_col = build_core_dataset(df, outcome)
    Xtr0, Xte0, ytr, yte, wtr, wte, gtr, _ = split_with_class_check(X, y, w, groups)
    imp = SimpleImputer(strategy="most_frequent")
    Xtr = pd.DataFrame(imp.fit_transform(Xtr0), columns=X.columns)
    Xte = pd.DataFrame(imp.transform(Xte0), columns=X.columns)
    model, cv_results = fit_with_cv(Xtr, ytr, wtr, gtr)
    metrics, cm, mean_pred, frac_pos, prob = evaluate_model(model, Xte, yte, wte)
    model.save_model(outdir / "xgb_model.json")
    with open(outdir / "imputer.pkl", "wb") as f: pickle.dump(imp, f)
    with open(outdir / "cv_results.json", "w") as f: json.dump(cv_results, f, indent=2)
    with open(outdir / "eval_results.json", "w") as f:
        json.dump({"target_column": target_col, "metrics": metrics,
                   "confusion_default_weighted": cm.tolist(),
                   "calibration_curve": {"mean_pred": mean_pred.tolist(), "frac_pos": frac_pos.tolist()}}, f, indent=2)
    pd.DataFrame({"y_true": np.asarray(yte), "y_prob": prob,
                  "sample_weight": np.asarray(wte)}).to_csv(outdir / f"predictions_{outcome}.csv", index=False)
    print(f"[{outcome}] target={target_col}; n={len(y)}; metrics={dict(metrics)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outcome", choices=["ideation", "planning", "attempt", "all"], default="all")
    parser.add_argument("--data", default=DATA_CSV)
    args = parser.parse_args()
    df = pd.read_csv(args.data, encoding="utf-8-sig", low_memory=False)
    outcomes = OUTCOMES if args.outcome == "all" else [args.outcome]
    for outcome in outcomes: run_outcome(df, outcome)

if __name__ == "__main__":
    main()
