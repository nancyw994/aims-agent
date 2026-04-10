"""Tests for aims_agent.results_analyzer."""

from __future__ import annotations

import numpy as np

from aims_agent.results_analyzer import compute_metrics, interpret_from_metrics


def test_compute_metrics_regression():
    y = np.array([1.0, 2.0, 3.0])
    p = np.array([1.1, 1.9, 3.05])
    m = compute_metrics(y, p, task_type="regression")
    assert "R2" in m
    assert "RMSE" in m


def test_compute_metrics_classification():
    y = np.array([0, 1, 0, 1])
    p = np.array([0, 1, 0, 0])
    m = compute_metrics(y, p, task_type="classification")
    assert "accuracy" in m
    assert "f1" in m


def test_interpret_from_metrics_nonempty():
    text = interpret_from_metrics({"R2": 0.8}, "RF", task_type="regression")
    assert "RF" in text
    assert len(text) > 10
