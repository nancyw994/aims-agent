from __future__ import annotations

from aims_agent.agent import Agent
from aims_agent.agents.debug_agent import SelfCorrectionAgent, repair_model_module_code
from aims_agent.specs import CodeGenSpec


def test_repair_model_module_code_parses_structured_json():
    def llm(_: str) -> str:
        return """{
  "diagnosis": "Generated class name mismatched required contract.",
  "patch_summary": "Renamed class and implemented fit/predict signatures.",
  "code": "import numpy as np\\n\\nclass GeneratedEstimator:\\n    def fit(self, X, y):\\n        self._n = np.asarray(X).shape[1] if np.asarray(X).ndim > 1 else 1\\n        return self\\n\\n    def predict(self, X):\\n        x = np.asarray(X)\\n        n = x.shape[0] if x.ndim > 1 else len(x)\\n        return np.zeros(n, dtype=float)\\n"
}"""

    agent = Agent(llm_call=llm)
    spec = CodeGenSpec(
        model_name="CustomReg",
        task_type="regression",
        required_interface="fit_predict",
        import_path_hint="",
        package_name="scikit-learn",
        constraints=[],
    )
    out = repair_model_module_code(
        agent,
        spec=spec,
        broken_code="class Bad:\n    pass\n",
        error_message="AttributeError: missing class",
    )
    assert out.diagnosis
    assert out.patch_summary
    assert "class GeneratedEstimator" in out.corrected_code


def test_self_correction_agent_retry_policy():
    agent = Agent(llm_call=lambda _: "{}")
    sc = SelfCorrectionAgent(agent)
    d1 = sc.should_retry(attempt=0, max_retries=2, repeated_error_count=1)
    d2 = sc.should_retry(attempt=2, max_retries=2, repeated_error_count=1)
    d3 = sc.should_retry(attempt=1, max_retries=3, repeated_error_count=2)
    assert d1.retry is True
    assert d2.retry is False
    assert d3.retry is False
