"""Utilities for reproducing revised manuscript Table 2.

The functions in this module evaluate XGBoost, Random Forest, and Logistic
Regression on the SAME held-out test set and report survey-weighted metrics.
They are intended to be imported after the manuscript's outcome-specific data
preparation and train/test split have been created.

The module deliberately does not recreate preprocessing or choose an outcome:
those steps must remain identical to the primary analysis. This avoids mixing
predictions from different outcome-specific samples.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    auc,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)


def evaluate_model(
    model,
    model_name,
    X_train,
    y_train,
    w_train,
    X_test,
    y_test,
    w_test,
    outcome_name,
    threshold=0.50,
):
    """Fit one model and return the revised Table 2 metrics and predictions."""
    clf = clone(model)
    clf.fit(X_train, y_train, sample_weight=w_train)

    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    if not (len(y_test) == len(y_prob) == len(w_test)):
        raise ValueError("Test labels, probabilities, and weights are misaligned.")

    fpr, tpr, _ = roc_curve(y_test, y_prob, sample_weight=w_test)
    auroc = auc(fpr, tpr)
    ap = average_precision_score(y_test, y_prob, sample_weight=w_test)

    precision_curve, recall_curve, _ = precision_recall_curve(
        y_test, y_prob, sample_weight=w_test
    )
    order = np.argsort(recall_curve)
    auprc_trapezoid = auc(recall_curve[order], precision_curve[order])

    row = {
        "Outcome": outcome_name,
        "Model": model_name,
        "Threshold": threshold,
        "AUROC": auroc,
        "AP/AUPRC": ap,
        "AUPRC_trapezoid": auprc_trapezoid,
        "PR_baseline_prevalence": np.average(y_test, weights=w_test),
        "Positive precision": precision_score(
            y_test, y_pred, average="binary", pos_label=1,
            sample_weight=w_test, zero_division=0
        ),
        "Positive recall": recall_score(
            y_test, y_pred, average="binary", pos_label=1,
            sample_weight=w_test, zero_division=0
        ),
        "Positive F1": f1_score(
            y_test, y_pred, average="binary", pos_label=1,
            sample_weight=w_test, zero_division=0
        ),
        "Weighted precision": precision_score(
            y_test, y_pred, average="weighted",
            sample_weight=w_test, zero_division=0
        ),
        "Weighted recall": recall_score(
            y_test, y_pred, average="weighted",
            sample_weight=w_test, zero_division=0
        ),
        "Weighted F1": f1_score(
            y_test, y_pred, average="weighted",
            sample_weight=w_test, zero_division=0
        ),
        "Accuracy": accuracy_score(y_test, y_pred, sample_weight=w_test),
        "Brier": brier_score_loss(y_test, y_prob, sample_weight=w_test),
    }
    return row, clf, y_prob, y_pred


def evaluate_table2_models(
    xgb_best,
    X_train,
    y_train,
    w_train,
    X_test,
    y_test,
    w_test,
    outcome_name,
):
    """Evaluate the three models used in revised Table 2."""
    models = {
        "XGBoost": xgb_best,
        "Random Forest": RandomForestClassifier(
            n_estimators=600, n_jobs=-1, random_state=42
        ),
        "Logistic Regression": LogisticRegression(
            solver="lbfgs", max_iter=2000, random_state=42
        ),
    }

    rows = []
    fitted_models = {}
    predicted_probs = {}
    predicted_labels = {}

    for model_name, model in models.items():
        row, fitted, prob, pred = evaluate_model(
            model, model_name,
            X_train, y_train, w_train,
            X_test, y_test, w_test,
            outcome_name, threshold=0.50,
        )
        rows.append(row)
        fitted_models[model_name] = fitted
        predicted_probs[model_name] = prob
        predicted_labels[model_name] = pred

    full = pd.DataFrame(rows)
    paper = full[[
        "Outcome", "Model", "AUROC", "AP/AUPRC",
        "Positive precision", "Positive recall", "Positive F1",
        "Weighted F1", "Accuracy",
    ]].copy()
    return paper, full, fitted_models, predicted_probs, predicted_labels


def save_table2_results(paper, full, outcome_name, output_dir="outputs_revision"):
    """Save manuscript-facing and full metric tables."""
    output_dir = pd.io.common.stringify_path(output_dir)
    from pathlib import Path
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    suffix = outcome_name.lower().replace(" ", "_").replace("/", "_")
    paper.to_csv(outdir / f"table2_{suffix}_for_paper.csv", index=False, encoding="utf-8-sig")
    full.to_csv(outdir / f"table2_{suffix}_full.csv", index=False, encoding="utf-8-sig")
