"""
DisGB Real Dataset Pipeline — Cu Dislocation–Grain Boundary Interactions
"""

import argparse
import json
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aims_agent import Agent
from aims_agent.csv_loader import CSVDataLoader, inspect_file

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_PATH = Path("/Users/nancy/Downloads/DisGB_data/DisGBDatabase_040125.xlsx")

# Features common to both tasks (type will be label-encoded automatically)
FEATURES = ["tx", "ty", "tz", "nx", "ny", "nz", "type", "misorientation angle", "GBE"]

# Task A: classify dislocation–GB reaction outcome
TASK_CLASSIFICATION = {
    "target": "Reaction_Mod",
    "task_type": "classification",
    "motivation": (
        "Predict the outcome label of dislocation–grain boundary interaction in Cu "
        "from crystallographic geometry and grain boundary energy"
    ),
}

# Task B: regress applied stress
TASK_REGRESSION = {
    "target": "Applied stress (MPa)",
    "task_type": "regression",
    "motivation": (
        "Predict the applied stress (MPa) required for dislocation–grain boundary "
        "interaction in Cu given crystallographic descriptors"
    ),
}


def run_task(
    agent: Agent,
    task: dict,
    use_randomized_search: bool = True,
    use_llm: bool = True,
) -> None:
    """Run one task (classification or regression) on the DisGB dataset."""
    task_type = task["task_type"]
    target = task["target"]
    motivation = task["motivation"]

    print(f"\n{'='*60}")
    print(f"Task     : {task_type.upper()}")
    print(f"Target   : {target}")
    print(f"Features : {FEATURES}")
    print(f"Use LLM  : {use_llm}")
    print("=" * 60)

    loader = CSVDataLoader()
    data_config = {
        "path": str(DATA_PATH),
        "features": FEATURES,
        "target": target,
        "source": "DisGBDatabase_040125.xlsx (Fensin et al., LANL)",
        "description": (
            "Cu dislocation–grain boundary interaction database. "
            f"Task: predict '{target}'."
        ),
    }

    result = agent.run_full_pipeline(
        interface=loader,
        data_config=data_config,
        motivation=motivation,
        task_type=task_type,
        use_hyperparameter_tuning=True,
        use_randomized_search=use_randomized_search,
        n_model_suggestions=1,
        skip_training=False,
        use_llm=use_llm,
    )

    if not result.success:
        print(f"\n[ERROR] Pipeline failed: {result.error}")
        return

    print("\nPlan (from LLM):")
    for i, step in enumerate(result.steps, 1):
        print(f"  {i}. {step}")

    if result.suggestion:
        print(f"\nModel : {result.suggestion.model_name}")
        print(f"Reason: {result.suggestion.reason}")

    print(f"\nMetrics ({task_type}):")
    print(json.dumps(result.metrics, indent=2))

    print(f"\nPlot  : {result.plot_path}")

    print("\nInterpretation:")
    print(result.interpretation)


def main():
    parser = argparse.ArgumentParser(description="DisGB real-data pipeline (classification + regression)")
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Do not call the LLM (use default model and metric-based summary). Use when API returns 504 or to run offline.",
    )
    args = parser.parse_args()

    if not DATA_PATH.exists():
        print(f"[ERROR] Data file not found: {DATA_PATH}")
        print("Please download DisGBDatabase_040125.xlsx and place it in Downloads/DisGB_data/")
        sys.exit(1)

    # Quick inspection before running the pipeline
    print("── Dataset Inspection ──────────────────────────────────────")
    inspect_file(DATA_PATH)

    agent = Agent()
    use_llm = not args.no_llm

    # ── Run Classification ────────────────────────────────────────────
    run_task(agent, TASK_CLASSIFICATION, use_randomized_search=True, use_llm=use_llm)

    # ── Run Regression ────────────────────────────────────────────────
    run_task(agent, TASK_REGRESSION, use_randomized_search=True, use_llm=use_llm)


if __name__ == "__main__":
    main()
