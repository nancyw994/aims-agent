"""Tests for aims_agent.validator."""

from __future__ import annotations

import json
import types

import numpy as np
import pytest

from aims_agent.agent import Agent
from aims_agent.agents.debug_agent import SelfCorrectionAgent
from aims_agent.specs import CodeGenSpec
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


# ---------------------------------------------------------------------------
# LLM repair tests
# ---------------------------------------------------------------------------

_BROKEN_CODE = """\
import numpy as np

class GeneratedEstimator:
    def fit(self, X, y):
        return self

    def predict(self, X):
        x = np.asarray(X)
        n = x.shape[0] if x.ndim > 1 else len(x)
        return np.zeros((n, 2), dtype=float)  # BUG: 2D output
"""

_FIXED_CODE = """\
import numpy as np

class GeneratedEstimator:
    def fit(self, X, y):
        return self

    def predict(self, X):
        x = np.asarray(X)
        n = x.shape[0] if x.ndim > 1 else len(x)
        return np.zeros(n, dtype=float)
"""

_MOCK_SPEC = CodeGenSpec(
    model_name="CustomReg",
    task_type="regression",
    required_interface="fit_predict",
    import_path_hint=None,
    package_name="scikit-learn",
    constraints=["predict must return 1D array"],
)


def _load_class_from_code(code: str) -> type:
    """Exec generated code string and return the GeneratedEstimator class."""
    mod = types.ModuleType("_test_generated")
    exec(compile(code, "<test_generated>", "exec"), mod.__dict__)  # noqa: S102
    return mod.GeneratedEstimator


def test_validate_estimator_contract_llm_repair_fixes_shape_error():
    """
    Full repair loop:
    1. Bad estimator fails contract with predict_shape_error.
    2. Mock LLM returns corrected code.
    3. Corrected code passes validate_estimator_contract.
    """
    # Step 1: confirm the broken class fails with predict_shape_error
    broken_cls = _load_class_from_code(_BROKEN_CODE)
    bad_out = validate_estimator_contract(broken_cls, task_type="regression")
    assert not bad_out.ok
    assert bad_out.code == "predict_shape_error"

    # Step 2: mock LLM returns the fixed code
    def mock_llm(_prompt: str) -> str:
        return json.dumps({
            "diagnosis": "predict returned 2D array; must be 1D",
            "patch_summary": "Changed np.zeros((n, 2)) to np.zeros(n)",
            "code": _FIXED_CODE,
        })

    agent = Agent(llm_call=mock_llm)
    sc = SelfCorrectionAgent(agent)
    patch = sc.propose_fix(
        spec=_MOCK_SPEC,
        broken_code=_BROKEN_CODE,
        error_message=f"[{bad_out.code}] {bad_out.message}",
        traceback_text="",
        attempt=0,
    )

    assert patch.diagnosis
    assert patch.patch_summary
    assert "GeneratedEstimator" in patch.corrected_code

    # Step 3: corrected code passes contract validation
    fixed_cls = _load_class_from_code(patch.corrected_code)
    fixed_out = validate_estimator_contract(fixed_cls, task_type="regression")
    assert fixed_out.ok, f"Repaired code still fails: {fixed_out.code} — {fixed_out.message}"
    assert fixed_out.code == "ok"


def test_predict_shape_error_failure_code_appears_in_llm_prompt():
    """
    Verify that when predict_shape_error is recorded as a previous failure,
    the failure code is injected verbatim into the prompt sent to the LLM.
    """
    captured: dict[str, str] = {}

    def capture_llm(prompt: str) -> str:
        captured["prompt"] = prompt
        return json.dumps({
            "diagnosis": "shape was wrong",
            "patch_summary": "returned 1D array",
            "code": _FIXED_CODE,
        })

    agent = Agent(llm_call=capture_llm)
    sc = SelfCorrectionAgent(agent)
    sc.propose_fix(
        spec=_MOCK_SPEC,
        broken_code=_BROKEN_CODE,
        error_message="[predict_shape_error] predict output must be 1d, got shape=(8, 2)",
        traceback_text="RuntimeError: ...",
        attempt=1,
        previous_failures=[
            {"attempt": 0, "failure_code": "predict_shape_error", "error_message": "2D output"}
        ],
    )

    assert "predict_shape_error" in captured["prompt"]
    assert "PREVIOUS FAILURES" in captured["prompt"]
