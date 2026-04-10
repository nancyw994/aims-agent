"""Tests for aims_agent.orchestrator (builtin + mock-LLM codegen)."""

from __future__ import annotations

import pytest

from aims_agent.agent import Agent
from aims_agent.model_selector import ModelSuggestion, get_model_suggestion
from aims_agent.orchestrator import resolve_model_class_multi_agent

# Minimal valid module matching CodeGen contract (returned by mock LLM)
_MOCK_GENERATED_ESTIMATOR = '''```python
import numpy as np
from sklearn.linear_model import LinearRegression

class GeneratedEstimator:
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()
        self._m = LinearRegression()
        self._m.fit(X, y)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.asarray(self._m.predict(X)).ravel()
```
'''

_BROKEN_GENERATED_ESTIMATOR = '''```python
class WrongName:
    def fit(self, X, y):
        return self
```
'''

_DEBUG_PATCH_JSON = """{
  "diagnosis": "Generated class name was incorrect.",
  "patch_summary": "Defined GeneratedEstimator with fit/predict.",
  "code": "import numpy as np\\n\\nclass GeneratedEstimator:\\n    def fit(self, X, y):\\n        return self\\n\\n    def predict(self, X):\\n        x = np.asarray(X)\\n        n = x.shape[0] if x.ndim > 1 else len(x)\\n        return np.zeros(n, dtype=float)\\n"
}"""


def test_resolve_builtin_random_forest_no_llm():
    def _no_llm(_: str) -> str:
        raise AssertionError("LLM should not be called for builtin path")

    agent = Agent(llm_call=_no_llm)
    s = get_model_suggestion("RandomForestRegressor", "regression")
    assert s is not None
    meta = {"features": ["a"], "target": "y", "description": "t"}
    r = resolve_model_class_multi_agent(
        agent,
        s,
        "regression",
        meta,
        use_llm=False,
        generated_code_dir="generated_code",
    )
    assert r.execution_path == "builtin"
    assert r.generated_model_wrapper_path == ""
    assert r.model_class.__name__ == "RandomForestRegressor"
    assert r.self_correction_attempts == 0
    assert "No self-correction needed" in r.self_correction_summary


def test_resolve_codegen_with_mock_llm(tmp_path):
    calls: list[str] = []

    def llm(prompt: str) -> str:
        calls.append(prompt)
        return _MOCK_GENERATED_ESTIMATOR

    agent = Agent(llm_call=llm)
    s = ModelSuggestion(
        model_name="UnknownZZ",
        package_name="scikit-learn",
        import_path="nonexistent_xyz_abc.ModCls",
        reason="test",
    )
    meta = {"features": ["f1", "f2"], "target": "t", "description": "d"}
    out_dir = tmp_path / "gen"
    r = resolve_model_class_multi_agent(
        agent,
        s,
        "regression",
        meta,
        use_llm=True,
        generated_code_dir=str(out_dir),
        max_codegen_retries=1,
    )
    assert r.execution_path == "codegen"
    assert r.generated_model_wrapper_path.endswith(".py")
    assert r.model_class.__name__ == "GeneratedEstimator"
    inst = r.model_class()
    assert len(calls) >= 1
    assert r.self_correction_success is True


def test_codegen_self_correction_retry_and_log(tmp_path):
    calls: list[str] = []

    def llm(prompt: str) -> str:
        calls.append(prompt)
        if len(calls) == 1:
            return _BROKEN_GENERATED_ESTIMATOR
        return _DEBUG_PATCH_JSON

    agent = Agent(llm_call=llm)
    s = ModelSuggestion(
        model_name="UnknownZZ2",
        package_name="scikit-learn",
        import_path="nonexistent_xyz_abc.ModCls",
        reason="test retry",
    )
    meta = {"features": ["f1", "f2"], "target": "t", "description": "d"}
    out_dir = tmp_path / "gen"
    r = resolve_model_class_multi_agent(
        agent,
        s,
        "regression",
        meta,
        use_llm=True,
        generated_code_dir=str(out_dir),
        max_codegen_retries=2,
    )
    assert r.execution_path == "codegen"
    assert r.self_correction_success is True
    assert r.self_correction_attempts >= 1
    assert r.self_correction_log_path.endswith(".jsonl")
    assert (tmp_path / "gen").exists()
