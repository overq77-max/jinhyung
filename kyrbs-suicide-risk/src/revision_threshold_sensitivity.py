"""Threshold sensitivity analysis used for the BMC Public Health revision.

The script compares three decision thresholds for a fitted binary classifier:
1. Default threshold = 0.50
2. Threshold maximizing Youden's J (sensitivity + specificity - 1)
3. Threshold maximizing survey-weighted positive-class F1

Input CSV columns
-----------------
y_true         : binary observed outcome (0/1)
y_prob         : predicted probability for class 1
sample_weight  : survey weight corresponding to the same held-out observations

Example
-------
python src/revision_threshold_sensitivity.py \
    outputs/predictions_ideation.csv \
    --outcome "Suicidal ideation" \
    --output outputs/threshold_sensitivity_ideation.csv

Important
---------
Thresholds are selected and evaluated on the same held-out predictions and are
therefore exploratory. They should not be interpreted as clinically validated
screening or triage cutoffs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_curve,
)


def _as_aligned_arrays(y_true, y_prob, sample_weight):
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    sample_weight = np.asarray(sample_weight, dtype=float).ravel()

    if not (len(y_true) == len(y_prob) == len(sample_weight)):
        raise ValueError(
            "y_true, y_prob, and sample_weight must contain the same number "
            "of observations. Do not combine probabilities from a different "
            "train/test split or outcome."
        )
    if not np.all(np.isfinite(y_prob)):
        raise ValueError("y_prob contains NaN or infinite values.")
    if not np.all(np.isfinite(sample_weight)):
        raise ValueError("sample_weight contains NaN or infinite values.")
    if set(np.unique(y_true)) - {0, 1}:
        raise ValueError("y_true must be binary and coded as 0/1.")
    return y_true.astype(int), y_prob, sample_weight


def calculate_metrics(y_true, y_prob, sample_weight, threshold, outcome, method):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
        sample_weight=sample_weight,
    ).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan

    return {
        "Outcome": outcome,
        "Method": method,
        "Threshold": float(threshold),
        "Sensitivity": float(sensitivity),
        "Specificity": float(specificity),
        "PPV": float(ppv),
        "NPV": float(npv),
        "Positive_F1": float(
            f1_score(
                y_true,
                y_pred,
                average="binary",
                pos_label=1,
                sample_weight=sample_weight,
                zero_division=0,
            )
        ),
        "Balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred, sample_weight=sample_weight)
        ),
        "Accuracy": float(
            accuracy_score(y_true, y_pred, sample_weight=sample_weight)
        ),
    }


def analyze_thresholds(y_true, y_prob, sample_weight, outcome):
    y_true, y_prob, sample_weight = _as_aligned_arrays(
        y_true, y_prob, sample_weight
    )

    # Youden's J.
    fpr, tpr, roc_thresholds = roc_curve(
        y_true, y_prob, sample_weight=sample_weight
    )
    finite = np.isfinite(roc_thresholds)
    j_values = (tpr - fpr)[finite]
    finite_thresholds = roc_thresholds[finite]
    threshold_youden = finite_thresholds[np.argmax(j_values)]

    # Positive-class F1. precision_recall_curve returns one fewer threshold
    # than precision/recall values, so the last precision/recall pair is
    # intentionally excluded here.
    precision, recall, pr_thresholds = precision_recall_curve(
        y_true, y_prob, sample_weight=sample_weight
    )
    precision = precision[:-1]
    recall = recall[:-1]
    denominator = precision + recall
    f1_values = np.divide(
        2 * precision * recall,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0,
    )
    threshold_f1 = pr_thresholds[np.argmax(f1_values)]

    rows = [
        calculate_metrics(
            y_true, y_prob, sample_weight, 0.50, outcome, "Default (0.50)"
        ),
        calculate_metrics(
            y_true, y_prob, sample_weight, threshold_youden, outcome, "Youden J"
        ),
        calculate_metrics(
            y_true,
            y_prob,
            sample_weight,
            threshold_f1,
            outcome,
            "Max positive-class F1",
        ),
    ]
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions_csv", type=Path)
    parser.add_argument("--outcome", required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("threshold_sensitivity.csv")
    )
    args = parser.parse_args()

    df = pd.read_csv(args.predictions_csv)
    required = {"y_true", "y_prob", "sample_weight"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    result = analyze_thresholds(
        df["y_true"], df["y_prob"], df["sample_weight"], args.outcome
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, encoding="utf-8-sig")

    print(result.round(4).to_string(index=False))
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
