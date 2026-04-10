from __future__ import annotations

import pandas as pd

from aims_agent import (
    Agent,
    CodeGenSpec,
    ModelSuggestion,
    get_model_suggestion,
    repair_model_module_code,
    run_training_phase_multi_agent_with_fallback,
)


def test_dual_mode_direct_debug_module_call():
    def llm(_: str) -> str:
        return """{
  "diagnosis": "Missing contract methods.",
  "patch_summary": "Added GeneratedEstimator with fit/predict.",
  "code": "import numpy as np\\n\\nclass GeneratedEstimator:\\n    def fit(self, X, y):\\n        return self\\n\\n    def predict(self, X):\\n        x = np.asarray(X)\\n        n = x.shape[0] if x.ndim > 1 else len(x)\\n        return np.zeros(n, dtype=float)\\n"
}"""

    agent = Agent(llm_call=llm)
    spec = CodeGenSpec(
        model_name="CustomReg",
        task_type="regression",
        required_interface="fit_predict",
        import_path_hint=None,
        package_name="scikit-learn",
        constraints=["must define GeneratedEstimator"],
    )
    out = repair_model_module_code(
        agent,
        spec=spec,
        broken_code="class Bad:\n    pass\n",
        error_message="AttributeError: missing class",
    )
    assert "GeneratedEstimator" in out.corrected_code
    assert out.diagnosis


def test_dual_mode_orchestrator_training_entry(tmp_path):
    agent = Agent(llm_call=lambda _: "not used")
    primary = ModelSuggestion(
        model_name="UnknownX",
        package_name="scikit-learn",
        import_path="nonexistent.mod.Class",
        reason="force fallback",
    )
    fallback = get_model_suggestion("RandomForestRegressor", "regression")
    assert fallback is not None
    df = pd.DataFrame(
        {
            "f1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "f2": [1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
            "t": [2.1, 2.3, 2.6, 2.8, 3.0, 3.2],
        }
    )
    meta = {"features": ["f1", "f2"], "target": "t", "description": "demo"}
    out = run_training_phase_multi_agent_with_fallback(
        agent,
        suggestion=primary,
        fallback_suggestion=fallback,
        ensure_package_installed_fn=lambda _: True,
        task_type="regression",
        metadata=meta,
        df=df,
        use_llm=False,
        generated_code_dir=str(tmp_path / "gen"),
        max_codegen_retries=0,
        use_hyperparameter_tuning=False,
    )
    assert out.fallback_used is True
    assert out.used_suggestion.model_name == "RandomForestRegressor"
    assert out.training_validation_ok is True
