"""Tests for aims_agent.validator."""

from __future__ import annotations

import numpy as np
import pytest

from aims_agent.validator import (
    validate_dl_training_trace,
    validate_estimator_contract,
    validate_training_result,
    validate_training_result_detailed,
)


def test_validate_ok_regression():
    yt = np.array([1.0, 2.0, 3.0])
    yp = np.array([1.1, 1.9, 3.2])
    ok, msg = validate_training_result(yt, yp, task_type="regression")
    assert ok and msg == "ok"


def test_validate_length_mismatch():
    ok, msg = validate_training_result(np.array([1, 2]), np.array([1.0]), task_type="regression")
    assert not ok
    assert "mismatch" in msg


def test_validate_nan_predictions():
    ok, _ = validate_training_result(
        np.array([1.0, 2.0]),
        np.array([1.0, np.nan]),
        task_type="regression",
    )
    assert not ok


def test_validate_bad_metric():
    ok, msg = validate_training_result(
        np.array([1.0, 2.0]),
        np.array([1.0, 2.0]),
        metrics={"R2": float("nan")},
        task_type="regression",
    )
    assert not ok
    assert "R2" in msg


def test_validate_training_result_detailed_code():
    out = validate_training_result_detailed(
        np.array([1.0, 2.0]),
        np.array([1.0, np.nan]),
        task_type="regression",
    )
    assert not out.ok
    assert out.code == "nan_inf_predictions"


def test_validate_regression_constant_pred_heuristic():
    yt = np.linspace(0, 10, 20)
    yp = np.ones(20) * 3.0
    ok, msg = validate_training_result(yt, yp, task_type="regression")
    assert not ok
    assert "constant" in msg.lower()


def test_validate_dl_training_trace_ok():
    ok, msg = validate_dl_training_trace([2.0, 1.6, 1.3], [0.5, 0.3, 0.2])
    assert ok
    assert msg == "ok"


def test_validate_dl_training_trace_loss_not_decreasing():
    ok, msg = validate_dl_training_trace([1.0, 1.1, 1.2], [0.2, 0.2, 0.2])
    assert not ok
    assert "decrease" in msg


def test_validate_dl_training_trace_bad_gradients():
    ok, msg = validate_dl_training_trace([1.2, 1.0], [0.0, 0.0])
    assert not ok
    assert "backward" in msg or "zero" in msg


def test_validate_estimator_contract_ok():
    class GoodEstimator:
        def fit(self, X, y):
            return self

        def predict(self, X):
            x = np.asarray(X)
            n = x.shape[0] if x.ndim > 1 else len(x)
            return np.zeros(n, dtype=float)

    out = validate_estimator_contract(GoodEstimator, task_type="regression")
    assert out.ok
    assert out.code == "ok"


def test_validate_estimator_contract_missing_predict():
    class BadEstimator:
        def fit(self, X, y):
            return self

    out = validate_estimator_contract(BadEstimator, task_type="regression")
    assert not out.ok
    assert out.code == "missing_interface"
