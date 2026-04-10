"""
Integration tests: real LLM (OpenAI or OpenRouter) + synthetic data.

Run explicitly:
  pytest -m integration

Requires:
  - pip install openai (the SDK is used for both OpenAI and OpenRouter)
  - OPENAI_API_KEY and/or OPENROUTER_API_KEY in the environment (see .env.example)

Note: ``plan_workflow_steps`` swallows LLM errors and falls back to DEFAULT_PLAN, so a
missing ``openai`` package can make a "plan" test look like it passed without any API
call. All tests here use ``_require_llm_environment()`` to skip unless the SDK is importable.
"""

from __future__ import annotations

import os

import pytest

from aims_agent.agent import Agent
from aims_agent.synthetic_loader import SyntheticDataLoader


pytestmark = pytest.mark.integration


def _require_llm_environment():
    """Skip unless OpenAI SDK is installed and an API key is configured."""
    from pathlib import Path

    try:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    except Exception:
        pass

    try:
        import openai  # noqa: F401
    except ImportError:
        pytest.skip(
            "Integration tests require the 'openai' package. "
            "Install: pip install openai  (use the same Python as pytest)"
        )

    if not (os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")):
        pytest.skip("OPENAI_API_KEY or OPENROUTER_API_KEY not set (integration)")


def test_llm_plan_workflow_steps():
    _require_llm_environment()
    from aims_agent.planning import plan_workflow_steps

    agent = Agent()
    meta = {
        "features": ["a"],
        "target": "y",
        "shape": {"rows": 50, "cols": 2},
        "description": "synthetic hardness",
    }
    steps = plan_workflow_steps(
        agent,
        "Predict hardness from composition",
        dataset_metadata=meta,
        include_codegen=False,
    )
    assert isinstance(steps, list)
    assert len(steps) >= 1
    actions = {s["action"] for s in steps}
    assert "select_model" in actions
    assert "train" in actions


def test_llm_pipeline_synthetic_skip_train():
    """
    End-to-end smoke: synthetic data, real LLM for *planning*, fixed sklearn model for selection.

    We use ``fixed_model`` so this does not depend on ``suggest_models`` JSON quality (which
    varies by model and often breaks CI / local runs). Planning still calls the LLM when
    ``use_llm=True``.
    """
    _require_llm_environment()
    agent = Agent()
    loader = SyntheticDataLoader()
    data_config = {"n_samples": 60, "random_seed": 42, "noise_sigma": 3.0}

    result = agent.run_full_pipeline(
        interface=loader,
        data_config=data_config,
        motivation="Predict alloy hardness from composition and process parameters",
        task_type="regression",
        use_llm=True,
        skip_training=True,
        fixed_model="RandomForestRegressor",
        use_hyperparameter_tuning=False,
        multi_agent=False,
    )
    assert result.success, f"pipeline failed: {result.error}"
    assert result.suggestion is not None
    assert result.suggestion.model_name == "RandomForestRegressor"
    assert result.metadata.get("target") == "hardness_HV"


@pytest.mark.skipif(
    not os.environ.get("RUN_LLM_MODEL_SELECT"),
    reason="Set RUN_LLM_MODEL_SELECT=1 to run (calls suggest_models; may flake on JSON)",
)
def test_llm_pipeline_synthetic_llm_model_pick():
    """Optional: full LLM model suggestions (fragile across providers / prompts)."""
    _require_llm_environment()
    agent = Agent()
    loader = SyntheticDataLoader()
    data_config = {"n_samples": 40, "random_seed": 0, "noise_sigma": 2.0}

    result = agent.run_full_pipeline(
        interface=loader,
        data_config=data_config,
        motivation="Predict hardness",
        task_type="regression",
        use_llm=True,
        skip_training=True,
        choose_model_fn=lambda a, m, suggestions: suggestions[0],
        n_model_suggestions=2,
        use_hyperparameter_tuning=False,
        multi_agent=False,
    )
    assert result.success, f"pipeline failed: {result.error}"
    assert result.suggestion is not None
    assert result.suggestion.model_name
