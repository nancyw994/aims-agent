"""Canonical failure-code taxonomy used by validator and self-correction logs."""

from __future__ import annotations

FAILURE_OK = "ok"
FAILURE_MISSING_INTERFACE = "missing_interface"
FAILURE_INIT_ERROR = "init_error"
FAILURE_FIT_ERROR = "fit_error"
FAILURE_PREDICT_ERROR = "predict_error"
FAILURE_PREDICT_SHAPE_ERROR = "predict_shape_error"
FAILURE_PREDICT_LENGTH_ERROR = "predict_length_error"
FAILURE_NAN_INF_PREDICTIONS = "nan_inf_predictions"
FAILURE_INVALID_METRIC = "invalid_metric"
FAILURE_COLLAPSED_PREDICTIONS = "collapsed_predictions"
FAILURE_NONE_ARRAY = "none_array"
FAILURE_EMPTY_ARRAYS = "empty_arrays"
FAILURE_LENGTH_MISMATCH = "length_mismatch"
FAILURE_RUNTIME_EXCEPTION = "runtime_exception"
FAILURE_RETRY_LIMIT_REACHED = "retry_limit_reached"
FAILURE_REPEATED_ERROR_SIGNATURE = "repeated_error_signature"

ALL_FAILURE_CODES = {
    FAILURE_OK,
    FAILURE_MISSING_INTERFACE,
    FAILURE_INIT_ERROR,
    FAILURE_FIT_ERROR,
    FAILURE_PREDICT_ERROR,
    FAILURE_PREDICT_SHAPE_ERROR,
    FAILURE_PREDICT_LENGTH_ERROR,
    FAILURE_NAN_INF_PREDICTIONS,
    FAILURE_INVALID_METRIC,
    FAILURE_COLLAPSED_PREDICTIONS,
    FAILURE_NONE_ARRAY,
    FAILURE_EMPTY_ARRAYS,
    FAILURE_LENGTH_MISMATCH,
    FAILURE_RUNTIME_EXCEPTION,
    FAILURE_RETRY_LIMIT_REACHED,
    FAILURE_REPEATED_ERROR_SIGNATURE,
}


def parse_failure_code_from_message(text: str) -> str:
    """Extract code from '[code] message' format; fallback to runtime_exception."""
    s = (text or "").strip()
    if s.startswith("[") and "]" in s:
        candidate = s[1 : s.index("]")]
        if candidate in ALL_FAILURE_CODES:
            return candidate
    return FAILURE_RUNTIME_EXCEPTION


__all__ = [
    "ALL_FAILURE_CODES",
    "FAILURE_OK",
    "FAILURE_MISSING_INTERFACE",
    "FAILURE_INIT_ERROR",
    "FAILURE_FIT_ERROR",
    "FAILURE_PREDICT_ERROR",
    "FAILURE_PREDICT_SHAPE_ERROR",
    "FAILURE_PREDICT_LENGTH_ERROR",
    "FAILURE_NAN_INF_PREDICTIONS",
    "FAILURE_INVALID_METRIC",
    "FAILURE_COLLAPSED_PREDICTIONS",
    "FAILURE_NONE_ARRAY",
    "FAILURE_EMPTY_ARRAYS",
    "FAILURE_LENGTH_MISMATCH",
    "FAILURE_RUNTIME_EXCEPTION",
    "FAILURE_RETRY_LIMIT_REACHED",
    "FAILURE_REPEATED_ERROR_SIGNATURE",
    "parse_failure_code_from_message",
]
