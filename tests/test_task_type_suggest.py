"""Tests for aims_agent.task_type_suggest (heuristic only)."""

from __future__ import annotations

import pandas as pd

from aims_agent.task_type_suggest import _heuristic_task_type


def test_heuristic_numeric_many_unique_regression():
    df = pd.DataFrame({"y": pd.Series(range(100), dtype=float)})
    t, _ = _heuristic_task_type(df, "y")
    assert t == "regression"


def test_heuristic_few_unique_classification():
    df = pd.DataFrame({"y": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]})
    t, reason = _heuristic_task_type(df, "y")
    assert t == "classification"
    assert "distinct" in reason.lower() or "few" in reason.lower()
