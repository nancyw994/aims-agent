"""
Distribution analysis — summarize data distributions for LLM-based model recommendation.

Computes target and feature statistics (class balance, skewness, etc.) and formats
a summary for the LLM to use when recommending which ML model to use.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal

import numpy as np
import pandas as pd


def _safe_skew(series: pd.Series) -> float:
    """Compute skewness, return 0 if insufficient data."""
    try:
        return float(series.skew()) if len(series.dropna()) >= 3 else 0.0
    except Exception:
        return 0.0


def analyze_distribution(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    task_type: Literal["regression", "classification"] = "regression",
) -> Dict[str, Any]:
    """
    Analyze distributions of target and features.

    Returns a dict with:
      - target_stats: value_counts (classification) or describe (regression)
      - feature_stats: per-column stats
      - summary_text: formatted string for LLM prompt
    """
    stats: Dict[str, Any] = {}
    target_series = df[target]

    # Target distribution 
    if task_type == "classification":
        vc = target_series.value_counts().sort_index()
        counts = vc.to_dict()
        total = len(target_series)
        ratios = {str(k): round(v / total, 3) for k, v in counts.items()}
        max_ratio = max(ratios.values()) if ratios else 0
        min_ratio = min(ratios.values()) if ratios else 0
        imbalance_ratio = max_ratio / min_ratio if min_ratio > 0 else float("inf")

        stats["target_stats"] = {
            "type": "classification",
            "n_classes": int(target_series.nunique()),
            "counts": counts,
            "ratios": ratios,
            "imbalance_ratio": round(imbalance_ratio, 2),
            "balanced": imbalance_ratio < 2.0,
        }
    else:
        target_numeric = pd.to_numeric(target_series, errors="coerce").dropna()
        desc = target_numeric.describe()
        skew = _safe_skew(target_numeric)
        stats["target_stats"] = {
            "type": "regression",
            "count": int(desc.get("count", 0)),
            "mean": round(float(desc.get("mean", 0)), 4),
            "std": round(float(desc.get("std", 0)), 4),
            "min": round(float(desc.get("min", 0)), 4),
            "25%": round(float(desc.get("25%", 0)), 4),
            "50%": round(float(desc.get("50%", 0)), 4),
            "75%": round(float(desc.get("75%", 0)), 4),
            "max": round(float(desc.get("max", 0)), 4),
            "skewness": round(skew, 3),
        }

    # Feature distributions
    feature_stats: Dict[str, Any] = {}
    for col in features:
        if col not in df.columns:
            continue
        s = df[col]
        if s.dtype in ("object", "string", "bool") or s.dtype.name == "category":
            vc = s.value_counts()
            feature_stats[col] = {
                "dtype": "categorical",
                "n_unique": int(s.nunique()),
                "counts": vc.head(10).to_dict(),
            }
        else:
            snum = pd.to_numeric(s, errors="coerce").dropna()
            if len(snum) == 0:
                feature_stats[col] = {"dtype": "numeric", "note": "all null"}
            else:
                skew = _safe_skew(snum)
                feature_stats[col] = {
                    "dtype": "numeric",
                    "mean": round(float(snum.mean()), 4),
                    "std": round(float(snum.std()), 4),
                    "min": round(float(snum.min()), 4),
                    "max": round(float(snum.max()), 4),
                    "n_unique": int(snum.nunique()),
                    "skewness": round(skew, 3),
                }
    stats["feature_stats"] = feature_stats

    # Build summary text for LLM 
    stats["summary_text"] = _format_for_llm(stats, task_type)

    return stats


def _format_for_llm(stats: Dict[str, Any], task_type: str) -> str:
    """Format distribution stats into a concise text for the LLM."""
    lines = ["=== Data Distribution Summary ==="]

    ts = stats["target_stats"]
    if task_type == "classification":
        lines.append(f"Target: {ts['n_classes']} classes.")
        lines.append(f"Class counts: {ts['counts']}")
        lines.append(f"Class ratios: {ts['ratios']}")
        if ts.get("imbalance_ratio"):
            lines.append(
                f"Class imbalance ratio (max/min): {ts['imbalance_ratio']} "
                f"({'balanced' if ts.get('balanced') else 'imbalanced — consider class_weight or oversampling'})"
            )
    else:
        lines.append(
            f"Target: mean={ts['mean']}, std={ts['std']}, "
            f"range=[{ts['min']}, {ts['max']}], skewness={ts['skewness']}"
        )
        if abs(ts.get("skewness", 0)) > 1:
            lines.append("Target is skewed — consider log transform or robust scaler.")

    lines.append("\nFeatures:")
    for col, fs in stats["feature_stats"].items():
        if fs.get("dtype") == "categorical":
            lines.append(f"  {col}: categorical, {fs['n_unique']} unique values")
        else:
            s = fs.get("skewness", 0)
            skew_note = " (skewed)" if abs(s) > 0.5 else ""
            lines.append(
                f"  {col}: mean={fs.get('mean')}, std={fs.get('std')}, "
                f"range=[{fs.get('min')}, {fs.get('max')}], skew={s}{skew_note}"
            )

    lines.append(
        "\nBased on this distribution, recommend models suited to: "
        "class imbalance (if any), skewed features, high-dimensionality, or nonlinearity."
    )
    return "\n".join(lines)


def plot_distribution(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    task_type: Literal["regression", "classification"],
    save_dir: str | Path = "results",
    filename: str = "data_distribution.png",
) -> str:
    """
    Plot target and feature distributions; save to file.

    Returns the full path of the saved file.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_path = Path(save_dir) / filename

    # Determine grid: 1 row for target + features
    n_features = min(len(features), 6)  # cap at 6 feature subplots
    n_cols = 3
    n_rows = 1 + (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes_flat = axes.flatten()

    idx = 0

    # Target
    ax = axes_flat[idx]
    idx += 1
    if task_type == "classification":
        df[target].value_counts().sort_index().plot(kind="bar", ax=ax, color="steelblue")
        ax.set_title(f"Target: {target}")
        ax.set_ylabel("Count")
    else:
        df[target].hist(ax=ax, bins=30, color="steelblue", edgecolor="white")
        ax.set_title(f"Target: {target}")
        ax.set_ylabel("Count")

    # Features
    for col in features[:6]:
        if idx >= len(axes_flat):
            break
        ax = axes_flat[idx]
        idx += 1
        s = df[col]
        if s.dtype in ("object", "string", "bool") or s.dtype.name == "category":
            s.value_counts().head(10).plot(kind="bar", ax=ax, color="coral")
        else:
            s.dropna().hist(ax=ax, bins=25, color="coral", edgecolor="white")
        ax.set_title(col[:20] + ("..." if len(col) > 20 else ""))
        ax.set_ylabel("Count")

    for j in range(idx, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return str(save_path)


__all__ = ["analyze_distribution", "plot_distribution"]
