"""Generate an outcome-specific ROC panel for revised Figure 2.

Use the fitted_models returned by revision_table2_models.evaluate_table2_models
so the curves and AUROC values are based on exactly the same fitted models and
held-out sample as revised Table 2.
"""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score


def plot_roc_panel(
    fitted_models,
    X_test,
    y_test,
    w_test,
    output_path,
    title="ROC Curve: Model Comparison",
):
    model_names = ["XGBoost", "Random Forest", "Logistic Regression"]
    missing = [m for m in model_names if m not in fitted_models]
    if missing:
        raise ValueError(f"Missing fitted models: {missing}")

    fig, ax = plt.subplots(figsize=(7, 6))
    auc_values = {}

    for model_name in model_names:
        y_prob = fitted_models[model_name].predict_proba(X_test)[:, 1]
        if not (len(y_test) == len(y_prob) == len(w_test)):
            raise ValueError("Test labels, probabilities, and weights are misaligned.")

        fpr, tpr, _ = roc_curve(y_test, y_prob, sample_weight=w_test)
        auc_value = roc_auc_score(y_test, y_prob, sample_weight=w_test)
        auc_values[model_name] = float(auc_value)
        ax.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"{model_name} (AUC = {auc_value:.4f})",
        )

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, label="No Skill")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return auc_values
