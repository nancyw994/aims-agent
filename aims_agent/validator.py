"""
Post-training validation: shapes, finiteness, and basic metric sanity.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np


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
    if y_true is None or y_pred is None:
        return False, "y_true or y_pred is None"
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    if yt.size == 0 or yp.size == 0:
        return False, "empty prediction arrays"
    if yt.shape[0] != yp.shape[0]:
        return False, f"length mismatch: y_true={yt.shape[0]} vs y_pred={yp.shape[0]}"
    if not np.isfinite(yp).all():
        return False, "predictions contain NaN or Inf"
    if metrics:
        for k, v in metrics.items():
            if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                return False, f"invalid metric {k}={v}"
    if task_type == "classification":
        # allow float labels from some models
        if np.unique(yp).size == 1 and yt.size > 2:
            return False, "predictions are constant; model may have collapsed"
    else:
        if np.unique(yp).size == 1 and yt.size > 5 and float(np.std(yt)) > 1e-9:
            return False, "regression predictions are constant while target varies"
    return True, "ok"


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


__all__ = ["validate_training_result", "validate_dl_training_trace"]
