"""
Week 4: Result analysis — metrics, plots, and LLM interpretation.

Computes regression metrics (R2, MSE, RMSE, MAE) or classification metrics
(accuracy, precision, recall, F1). Generates predicted vs actual / residuals
for regression, or confusion matrix for classification. Prompts the LLM for
natural-language interpretation.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from aims_agent.agent import Agent


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: Literal["regression", "classification"] = "regression",
) -> dict[str, float]:
    """
    Compute performance metrics.

    For regression: R2, MSE, RMSE, MAE.
    For classification: accuracy, precision (macro), recall (macro), F1 (macro).
    """
    if task_type == "classification":
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
        )

        return {
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "precision": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4),
            "f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        }

    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score,
    )

    return {
        "R2": round(float(r2_score(y_true, y_pred)), 4),
        "MSE": round(float(mean_squared_error(y_true, y_pred)), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 4),
    }


def plot_results(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_dir: str | Path = "results",
    filename: str = "model_performance.png",
    task_type: Literal["regression", "classification"] = "regression",
) -> str:
    """
    Generate and save performance plots.

    Regression: predicted vs actual, residuals.
    Classification: confusion matrix.
    Returns the full path of the saved file.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_path = Path(save_dir) / filename

    if task_type == "classification":
        from sklearn.metrics import ConfusionMatrixDisplay

        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, colorbar=True)
        ax.set_title("Confusion Matrix")
        plt.tight_layout()
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].scatter(y_true, y_pred, alpha=0.6, color="steelblue")
        lo = min(y_true.min(), y_pred.min())
        hi = max(y_true.max(), y_pred.max())
        axes[0].plot([lo, hi], [lo, hi], "r--", label="Ideal")
        axes[0].set_xlabel("Actual")
        axes[0].set_ylabel("Predicted")
        axes[0].set_title("Predicted vs Actual")
        axes[0].legend()

        residuals = y_true - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.6, color="coral")
        axes[1].axhline(0, color="black", linestyle="--")
        axes[1].set_xlabel("Predicted")
        axes[1].set_ylabel("Residual")
        axes[1].set_title("Residuals")

        plt.tight_layout()

    plt.savefig(save_path, dpi=150)
    plt.close()
    return str(save_path)


def interpret_from_metrics(
    metrics: dict[str, float],
    model_name: str,
    task_type: Literal["regression", "classification"] = "regression",
) -> str:
    """Generate a short interpretation from metrics without calling the LLM (for --no-llm mode)."""
    lines = [
        f"Model: {model_name} | Task: {task_type}",
        f"Metrics: {metrics}",
    ]
    if task_type == "classification":
        acc = metrics.get("accuracy", 0)
        f1 = metrics.get("f1", 0)
        if acc >= 0.85:
            lines.append("Performance: good. High accuracy suggests the model generalizes well.")
        elif acc >= 0.65:
            lines.append("Performance: moderate. Consider feature engineering or trying other models.")
        else:
            lines.append("Performance: limited. Try more features, class balance, or different algorithms.")
        lines.append("Suggestion: try hyperparameter tuning (GridSearchCV) or ensemble methods.")
    else:
        r2 = metrics.get("R2", 0)
        rmse = metrics.get("RMSE", 0)
        if r2 >= 0.7:
            lines.append("Performance: good. R2 indicates strong predictive capability.")
        elif r2 >= 0.4:
            lines.append("Performance: moderate. RMSE reflects typical prediction error.")
        else:
            lines.append("Performance: limited. Consider more features or nonlinear models.")
        lines.append("Suggestion: try feature selection or regularization (Ridge/Lasso).")
    return "\n".join(lines)


def interpret_with_llm(
    agent: "Agent",
    metrics: dict[str, float],
    model_name: str,
    task_type: Literal["regression", "classification"] = "regression",
) -> str:
    """Ask the LLM to interpret the metrics in 2–3 short paragraphs."""
    task_desc = "classification" if task_type == "classification" else "regression"
    prompt = f"""As a materials science ML expert, interpret this model evaluation:

Model: {model_name}
Task: {task_desc}
Metrics: {metrics}

In 2–3 short paragraphs:
1. How good is this performance (good / moderate / poor)?
2. What do these metrics mean for materials property prediction?
3. One concrete improvement suggestion."""

    return agent.call_llm(prompt)


__all__ = ["compute_metrics", "plot_results", "interpret_with_llm", "interpret_from_metrics"]
