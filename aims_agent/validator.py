"""
Post-training validation: shapes, finiteness, and basic metric sanity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np


@dataclass
class ValidationOutcome:
    ok: bool
    code: str
    message: str


def _out(ok: bool, code: str, message: str) -> ValidationOutcome:
    return ValidationOutcome(ok=ok, code=code, message=message)


def validate_estimator_contract(
    model_class: type,
    *,
    task_type: Literal["regression", "classification"] = "regression",
    n_features: int = 3,
) -> ValidationOutcome:
    """
    Lightweight post-load validation for generated estimators.
    Ensures fit/predict interface works and predict returns 1d finite output.
    """
    fit_fn = getattr(model_class, "fit", None)
    pred_fn = getattr(model_class, "predict", None)
    if not callable(fit_fn) or not callable(pred_fn):
        return _out(False, "missing_interface", "model class missing callable fit/predict")
    try:
        model = model_class()
    except Exception as e:
        return _out(False, "init_error", f"failed to instantiate model: {e}")

    n = 8
    x = np.random.RandomState(42).randn(n, n_features)
    y = np.random.RandomState(1).randn(n) if task_type == "regression" else np.random.randint(0, 2, size=n)
    try:
        model.fit(x, y)
    except Exception as e:
        return _out(False, "fit_error", f"fit failed in contract validation: {e}")
    try:
        yp = np.asarray(model.predict(x))
    except Exception as e:
        return _out(False, "predict_error", f"predict failed in contract validation: {e}")
    if yp.ndim != 1:
        return _out(False, "predict_shape_error", f"predict output must be 1d, got shape={yp.shape}")
    if yp.shape[0] != n:
        return _out(False, "predict_length_error", f"predict output length {yp.shape[0]} != {n}")
    if not np.isfinite(yp).all():
        return _out(False, "nan_inf_predictions", "predict output contains NaN/Inf")
    return _out(True, "ok", "ok")


def validate_training_result_detailed(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    metrics: dict[str, float] | None = None,
    task_type: Literal["regression", "classification"] = "regression",
) -> ValidationOutcome:
    if y_true is None or y_pred is None:
        return _out(False, "none_array", "y_true or y_pred is None")
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    if yt.size == 0 or yp.size == 0:
        return _out(False, "empty_arrays", "empty prediction arrays")
    if yt.shape[0] != yp.shape[0]:
        return _out(False, "length_mismatch", f"length mismatch: y_true={yt.shape[0]} vs y_pred={yp.shape[0]}")
    if not np.isfinite(yp).all():
        return _out(False, "nan_inf_predictions", "predictions contain NaN or Inf")
    if metrics:
        for k, v in metrics.items():
            if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                return _out(False, "invalid_metric", f"invalid metric {k}={v}")
    if task_type == "classification":
        if np.unique(yp).size == 1 and yt.size > 2:
            return _out(False, "collapsed_predictions", "predictions are constant; model may have collapsed")
    else:
        if np.unique(yp).size == 1 and yt.size > 5 and float(np.std(yt)) > 1e-9:
            return _out(False, "collapsed_predictions", "regression predictions are constant while target varies")
    return _out(True, "ok", "ok")


def validate_training_result(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    metrics: dict[str, float] | None = None,
    task_type: Literal["regression", "classification"] = "regression",
) -> tuple[bool, str]:
    """
    Return (ok, message). Used after ModelTrainer.predict().
    """
    out = validate_training_result_detailed(
        y_true,
        y_pred,
        metrics=metrics,
        task_type=task_type,
    )
    return out.ok, out.message


def validate_dl_training_trace(
    loss_history: list[float] | tuple[float, ...] | None,
    gradient_norms: list[float] | tuple[float, ...] | None = None,
    *,
    min_steps: int = 2,
) -> tuple[bool, str]:
    """
    Validate minimal DL training signals:
    - finite losses
    - enough steps
    - final loss lower than initial loss
    - optional gradient norms are finite and not all zero
    """
    if not loss_history:
        return False, "missing loss_history for deep-learning validation"
    losses = np.asarray(loss_history, dtype=float).ravel()
    if losses.size < min_steps:
        return False, f"loss_history too short: {losses.size} < {min_steps}"
    if not np.isfinite(losses).all():
        return False, "loss_history contains NaN/Inf"
    if float(losses[-1]) >= float(losses[0]):
        return False, "loss did not decrease"

    if gradient_norms is not None and len(gradient_norms) > 0:
        grads = np.asarray(gradient_norms, dtype=float).ravel()
        if not np.isfinite(grads).all():
            return False, "gradient_norms contains NaN/Inf"
        if np.all(np.abs(grads) < 1e-12):
            return False, "gradient_norms are all near zero; backward may be broken"

    return True, "ok"


__all__ = [
    "ValidationOutcome",
    "validate_estimator_contract",
    "validate_training_result_detailed",
    "validate_training_result",
    "validate_dl_training_trace",
]
