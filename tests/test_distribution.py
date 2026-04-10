"""Tests for aims_agent.distribution."""

from __future__ import annotations

import numpy as np
import pandas as pd

from aims_agent.distribution import analyze_distribution


def test_analyze_distribution_regression():
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "a": rng.normal(size=50),
            "b": rng.uniform(0, 1, size=50),
            "y": rng.normal(size=50),
        }
    )
    out = analyze_distribution(df, ["a", "b"], "y", task_type="regression")
    assert "summary_text" in out
    assert "target_stats" in out
    assert out["target_stats"]["type"] == "regression"
    assert len(out["summary_text"]) > 20
