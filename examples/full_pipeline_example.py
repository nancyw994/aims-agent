"""
Week 4: Full ML pipeline example (no API key required).

Runs the end-to-end pipeline with a mock LLM to verify:
- Data ingestion, planning, model selection, training, evaluation, plotting, LLM interpretation.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aims_agent import Agent, get_metadata
from aims_agent.synthetic_loader import SyntheticDataLoader


class MockLLMAgent(Agent):
    """Agent with mock LLM responses for plan, model selection, and interpretation."""

    def __init__(self):
        super().__init__(llm_call=self._mock_call)
        self._call_count = 0

    def _mock_call(self, prompt: str) -> str:
        self._call_count += 1
        pl = prompt.lower()
        if "interpret this model evaluation" in pl or ("interpret" in pl and "metrics" in pl):
            return """The model shows moderate performance. R2 indicates reasonable
predictive capability. For materials property prediction, consider feature engineering
or trying ensemble methods like Gradient Boosting for potential improvements."""
        if "suggest" in pl and "json array" in pl:
            return """[
  {"model_name": "RandomForestRegressor", "package": "scikit-learn",
   "import_path": "sklearn.ensemble.RandomForestRegressor",
   "reason": "Handles nonlinearity well for materials data."}
]"""
        if "steps" in pl or "series of steps" in pl:
            return """1. Load the materials dataset
2. Select features and target variable
3. Choose an appropriate regression model
4. Train the model with cross-validation
5. Evaluate using R2 and MSE"""
        return "Mock response"


def main():
    print("Week 4: Full ML pipeline (mock LLM)")
    print("=" * 50)

    agent = MockLLMAgent()
    loader = SyntheticDataLoader()

    result = agent.run_full_pipeline(
        interface=loader,
        data_config={"n_samples": 150, "random_seed": 42},
        motivation="Predict hardness from alloy composition and processing",
        task_type="regression",
        use_hyperparameter_tuning=True,
        use_randomized_search=False,
        n_model_suggestions=1,
        skip_training=False,
    )

    print("\nPlan steps:")
    for i, step in enumerate(result.steps, 1):
        print(f"  {i}. {step}")

    print("\nModel:", result.suggestion.model_name if result.suggestion else "N/A")
    print("Metrics:", json.dumps(result.metrics, indent=2))
    print("Plot saved:", result.plot_path)
    print("\nLLM interpretation:")
    print(result.interpretation)

    assert result.success, f"Pipeline failed: {result.error}"
    assert result.metrics, "No metrics computed"
    assert "R2" in result.metrics or "MSE" in result.metrics
    assert Path(result.plot_path).exists(), f"Plot not found: {result.plot_path}"

    print("\n" + "=" * 50)
    print("Full pipeline completed successfully.")


if __name__ == "__main__":
    main()
