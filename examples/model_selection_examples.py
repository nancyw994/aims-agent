"""
Multiple model-selection examples (no API key required).

Runs several synthetic scenarios with mock LLM responses to show different
model/package suggestions and dependency handling.
"""

import sys
from pathlib import Path

# Allow importing aims_agent from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aims_agent import (
    ModelSuggestion,
    suggest_model,
    ensure_package_installed,
    get_metadata,
)
from aims_agent.synthetic_loader import SyntheticDataLoader


# Mock LLM responses for different scenarios (no real API call)
MOCK_RESPONSES = [
    "MODEL: RandomForestRegressor\nPACKAGE: scikit-learn",
    "MODEL: XGBRegressor\nPACKAGE: xgboost",
    "MODEL: GradientBoostingRegressor\nPACKAGE: scikit-learn",
    "MODEL: Ridge\nPACKAGE: scikit-learn",
    "MODEL: LGBMRegressor\nPACKAGE: lightgbm",
]


class MockAgent:
    """Returns canned responses in sequence."""

    def __init__(self, responses):
        self._responses = list(responses)
        self._index = 0

    def call_llm(self, prompt):
        r = self._responses[self._index % len(self._responses)]
        self._index += 1
        return r


def main():
    # Load synthetic dataset once; get features and target from metadata
    loader = SyntheticDataLoader()
    bundle = loader.load_dataset({"n_samples": 50, "random_seed": 42})
    metadata = get_metadata(bundle)
    features = metadata["features"]
    target = metadata["target"]

    print("Dataset: features =", features)
    print("         target   =", target)
    print()

    agent = MockAgent(MOCK_RESPONSES)

    for i, expected in enumerate(
        [
            ("RandomForestRegressor", "scikit-learn"),
            ("XGBRegressor", "xgboost"),
            ("GradientBoostingRegressor", "scikit-learn"),
            ("Ridge", "scikit-learn"),
            ("LGBMRegressor", "lightgbm"),
        ]
    ):
        suggestion = suggest_model(agent, features, target)
        assert suggestion.model_name == expected[0] and suggestion.package_name == expected[1]
        print(f"Scenario {i + 1}: {suggestion.model_name} / {suggestion.package_name}")
        try:
            ok = ensure_package_installed(suggestion.package_name)
            print(f"  Package available: {'yes' if ok else 'no (see logs/)'}")
        except Exception as e:
            print(f"  Package available: no ({type(e).__name__})")
    print()
    print("All examples passed (model selection parsing + dependency check).")


if __name__ == "__main__":
    main()
