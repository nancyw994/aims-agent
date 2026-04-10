"""Tests for aims_agent.model_trainer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from aims_agent.model_trainer import ModelTrainer


@pytest.fixture
def tiny_df():
    rng = np.random.default_rng(0)
    n = 40
    return pd.DataFrame(
        {
            "f1": rng.normal(size=n),
            "f2": rng.normal(size=n),
            "y": rng.normal(size=n),
        }
    )


def test_trainer_regression_fit_predict(tiny_df):
    trainer = ModelTrainer(
        LinearRegression,
        task_type="regression",
        use_hyperparameter_tuning=False,
    )
    trainer.prepare_data(tiny_df, ["f1", "f2"], "y", test_size=0.25, random_state=0)
    trainer.train()
    y_true, y_pred = trainer.predict()
    assert len(y_true) == len(y_pred)
    assert np.isfinite(y_pred).all()
