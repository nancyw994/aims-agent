"""
LLM-assisted task type suggestion (regression vs classification) with heuristic fallback.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from aims_agent.agent import Agent

from aims_agent.data_interface import DatasetBundle, get_metadata


def _heuristic_task_type(df: pd.DataFrame, target: str) -> tuple[str, str]:
    """Rule-based fallback when LLM is unavailable or parsing fails."""
    if target not in df.columns:
        return "regression", "Target column missing; defaulting to regression."
    y = df[target]
    n = len(y.dropna())
    if n == 0:
        return "regression", "No non-null target values; defaulting to regression."
    n_unique = int(y.nunique(dropna=True))
    if pd.api.types.is_bool_dtype(y):
        return "classification", "Boolean target -> classification."
    if pd.api.types.is_object_dtype(y) or str(y.dtype) == "category":
        return (
            "classification" if n_unique <= 50 else "regression",
            f"Categorical/object target with {n_unique} unique values.",
        )
    # Numeric
    if n_unique <= 15 and n_unique < max(10, int(0.05 * n)):
        return (
            "classification",
            f"Numeric target with few distinct values ({n_unique}); treating as classification.",
        )
    return "regression", f"Numeric target with {n_unique} distinct values; regression."


def suggest_task_type(
    agent: "Agent",
    bundle: DatasetBundle,
    motivation: str,
    background_knowledge: str | None = None,
) -> tuple[str, str]:
    """
    Ask LLM whether the task is regression or classification.

    Returns:
        (task_type, reason) where task_type is 'regression' or 'classification'.
    """
    metadata = get_metadata(bundle)
    target = metadata["target"]
    df = bundle.df
    y = df[target] if target in df.columns else pd.Series(dtype=float)

    n_nonnull = int(y.notna().sum())
    n_unique = int(y.nunique(dropna=True)) if n_nonnull else 0
    dtype_str = str(y.dtype)
    sample_vals = y.dropna().head(8).tolist()

    prompt = f"""You are an Machine Learning & Statistical Modeling expert in materials science. Based on the dataset information, 
                    provided background knowledge and user's motivation, decide the supervised learning task type.

User goal: {motivation}
"""
    if background_knowledge:
        prompt += f"\nBackground / domain context:\n{background_knowledge.strip()}\n"

    prompt += f"""
Dataset:
- Target column name: {target}
- Target dtype: {dtype_str}
- Non-null count: {n_nonnull}
- Unique values (approx): {n_unique}
- Sample target values: {sample_vals}

Rules:
- Use "regression" when predicting a continuous quantity (stress, elongation %, energy, etc.).
- Use "classification" when predicting discrete labels/categories (reaction type, phase, pass/fail, low/med/high bins if encoded as few classes).

Return ONLY a JSON object, no markdown, no other text:
{{"task_type": "regression" or "classification", "reason": "one short sentence"}}
"""
    try:
        response = agent.call_llm(prompt)
        cleaned = re.sub(r"```(?:json)?\s*", "", response).strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start == -1 or end <= start:
            raise ValueError("No JSON object in response")
        data = json.loads(cleaned[start:end])
        tt = str(data.get("task_type", "")).strip().lower()
        reason = str(data.get("reason", "LLM suggestion")).strip()
        if tt in ("regression", "classification"):
            return tt, reason
    except Exception as e:
        print(f"[TaskType] LLM suggestion failed ({e}); using heuristic.")

    return _heuristic_task_type(df, target)


__all__ = ["suggest_task_type", "_heuristic_task_type"]
