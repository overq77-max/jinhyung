"""2020-2024 loneliness sensitivity analysis for the revised manuscript.

This script mirrors the primary outcome-specific XGBoost workflow while:
  * restricting the harmonized data to survey years 2020-2024; and
  * adding M_LON as an additional predictor.

It exports aligned held-out predictions and SHAP values/plots for each outcome.
The analysis is supportive/sensitivity evidence because loneliness is
structurally unavailable in the harmonized file before 2020.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.impute import SimpleImputer

from train_weighted_xgb import (
    FEATURE_COLS,
    OUTCOMES,
    build_core_dataset,
    evaluate_model,
    fit_with_cv,
    split_with_class_check,
)

YEARS = [2020, 2021, 2022, 2023, 2024]
LONELINESS_COL = "M_LON"


def run_loneliness_outcome(df: pd.DataFrame, outcome: str, output_root: Path):
    subset = df[df["YEAR"].isin(YEARS)].copy()
    if LONELINESS_COL not in subset.columns:
        raise KeyError("M_LON is required for the loneliness sensitivity analysis.")

    # Temporarily extend the primary feature set used by build_core_dataset.
    import train_weighted_xgb as primary
    original_features = list(primary.FEATURE_COLS)
    try:
        primary.FEATURE_COLS = original_features + [LONELINESS_COL]
        X, y, w, groups, target_col = build_core_dataset(subset, outcome)
    finally:
        primary.FEATURE_COLS = original_features

    Xtr0, Xte0, ytr, yte, wtr, wte, gtr, _ = split_with_class_check(
        X, y, w, groups
    )

    # This matches the finalized primary code: all encoded predictors are
    # imputed from the training partition using the most frequent value.
    imputer = SimpleImputer(strategy="most_frequent")
    Xtr = pd.DataFrame(imputer.fit_transform(Xtr0), columns=X.columns)
    Xte = pd.DataFrame(imputer.transform(Xte0), columns=X.columns)

    model, cv_results = fit_with_cv(Xtr, ytr, wtr, gtr)
    metrics, _, _, _, prob = evaluate_model(model, Xte, yte, wte)

    outdir = output_root / outcome
    outdir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        "y_true": np.asarray(yte),
        "y_prob": np.asarray(prob),
        "sample_weight": np.asarray(wte),
    }).to_csv(outdir / f"predictions_{outcome}_2020_2024_loneliness.csv", index=False)

    with open(outdir / "performance.json", "w", encoding="utf-8") as f:
        json.dump({
            "outcome": outcome,
            "target_column": target_col,
            "years": YEARS,
            "n_analysis": int(len(y)),
            "metrics": dict(metrics),
            "cv": cv_results,
        }, f, indent=2)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(Xte)
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    np.save(outdir / "shap_values.npy", np.asarray(shap_values))
    Xte.to_csv(outdir / "X_test_for_shap.csv", index=False)

    plt.figure()
    shap.summary_plot(shap_values, Xte, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(outdir / f"shap_summary_{outcome}_loneliness.png", dpi=600, bbox_inches="tight")
    plt.close()

    print(f"[{outcome}] 2020-2024 + M_LON: n={len(y)}; {dict(metrics)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="analysis_variables_only.csv")
    parser.add_argument(
        "--outcome",
        choices=["ideation", "planning", "attempt", "all"],
        default="all",
    )
    parser.add_argument("--output", default="outputs_loneliness_sensitivity")
    args = parser.parse_args()

    df = pd.read_csv(args.data, encoding="utf-8-sig", low_memory=False)
    outcomes = list(OUTCOMES) if args.outcome == "all" else [args.outcome]
    for outcome in outcomes:
        run_loneliness_outcome(df, outcome, Path(args.output))


if __name__ == "__main__":
    main()
