import argparse
import json
import sys

from aims_agent.agent import Agent
from aims_agent.synthetic_loader import SyntheticDataLoader
from aims_agent.csv_loader import CSVDataLoader
from aims_agent.model_selector import suggest_models, ModelSuggestion, list_all_models, get_model_suggestion


def _interactive_choose_model(
    agent: Agent,
    metadata: dict,
    suggestions: list,
    task_type: str = "regression",
) -> ModelSuggestion:
    """
    Show numbered model suggestions. User can:
    - Enter 1..N to choose one
    - Type more requirements (e.g. prefer interpretability) to get new suggestions
    - Press Enter to use the first suggestion
    """
    while True:
        print("\nRecommended models:")
        for i, s in enumerate(suggestions, 1):
            print(f"  {i}. {s.model_name} ({s.package_name}) — {s.reason}")

        if not sys.stdin.isatty():
            print("  (non-interactive: using first suggestion)")
            return suggestions[0]

        prompt_msg = (
            "Enter number 1–%d to choose, type more requirements for new suggestions, or Enter to use the first: "
            % len(suggestions)
        )
        try:
            choice = input(prompt_msg).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nUsing first suggestion.")
            return suggestions[0]

        if not choice:
            return suggestions[0]

        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(suggestions):
                return suggestions[idx - 1]
            print("Invalid number. Try again.")
            continue

        # User provided more context; re-ask LLM
        print("\nGetting new suggestions based on your input...")
        new_suggestions = suggest_models(
            agent,
            features=metadata["features"],
            target=metadata["target"],
            n_suggestions=5,
            task_hint=task_type,
            extra_context=choice,
        )
        if new_suggestions:
            suggestions = new_suggestions
        else:
            print("Could not get new suggestions; please choose from the list above.")


def parse_args():
    p = argparse.ArgumentParser(
        description="AI Agent for ML in Materials Science",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Synthetic data, regression (default):
  python -m aims_agent.cli --motivation "Predict hardness"

  # Real CSV / Excel, classification, explicit target:
  python -m aims_agent.cli \\
      --motivation "Predict dislocation-GB reaction" \\
      --data data/DisGBDatabase_040125.xlsx \\
      --target Reaction_Mod \\
      --task-type classification \\
      --randomized-search

  # Real data, regression, explicit features:
  python -m aims_agent.cli \\
      --motivation "Predict applied stress" \\
      --data data/DisGBDatabase_040125.xlsx \\
      --target "Applied stress (MPa)" \\
      --features tx ty tz nx ny nz "misorientation angle" GBE \\
      --task-type regression
""",
    )

    # Data source
    data_grp = p.add_argument_group("Data source")
    data_grp.add_argument(
        "--data",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to CSV or Excel (.xls/.xlsx) file. If not set, use synthetic data.",
    )
    data_grp.add_argument(
        "--target",
        type=str,
        default=None,
        metavar="COLUMN",
        help="Name of the target column in --data. Auto-detected if not set.",
    )
    data_grp.add_argument(
        "--features",
        nargs="+",
        default=None,
        metavar="COL",
        help="Feature column names. All non-target columns are used if not set.",
    )
    data_grp.add_argument(
        "--sheet",
        type=str,
        default="0",
        metavar="NAME_OR_INDEX",
        help="Excel sheet name or 0-based index (default: 0). Ignored for CSV.",
    )

    # Task
    task_grp = p.add_argument_group("Task")
    task_grp.add_argument(
        "--motivation",
        default=None,
        help="User's research goal in natural language. Required unless --list-models.",
    )
    task_grp.add_argument(
        "--task-type",
        choices=["regression", "classification"],
        default="regression",
        help="ML task type (default: regression).",
    )

    # Synthetic data options
    synth_grp = p.add_argument_group("Synthetic data (only used when --data is not set)")
    synth_grp.add_argument("--n-samples", type=int, default=200, metavar="N")
    synth_grp.add_argument("--noise-sigma", type=float, default=5.0, metavar="SIGMA")
    synth_grp.add_argument("--random-seed", type=int, default=None, metavar="SEED")

    # Training options
    train_grp = p.add_argument_group("Training")
    train_grp.add_argument(
        "--skip-train",
        action="store_true",
        help="Stop after model selection (no training or report).",
    )
    train_grp.add_argument(
        "--no-tuning",
        action="store_true",
        help="Disable hyperparameter tuning (train with default args).",
    )
    train_grp.add_argument(
        "--randomized-search",
        action="store_true",
        help="Use RandomizedSearchCV instead of GridSearchCV (faster).",
    )
    train_grp.add_argument(
        "--no-llm",
        action="store_true",
        help="Do not call the LLM (use default plan, default model, and metric-based summary). Use when API is down or to run offline.",
    )
    train_grp.add_argument(
        "--model",
        type=str,
        default=None,
        metavar="NAME",
        help="Use this model directly (skip LLM). E.g. RandomForestClassifier, XGBRegressor. Run --list-models to see all.",
    )
    train_grp.add_argument(
        "--list-models",
        action="store_true",
        help="List all supported ML models and exit.",
    )

    return p.parse_args()


def main():
    """
    End-to-end ML pipeline:
    DATA INTERFACE → PLAN → MODEL SELECT → TRAIN → REPORT
    """
    args = parse_args()

    # List models and exit
    if args.list_models:
        reg = list_all_models("regression")
        clf = list_all_models("classification")
        print("Supported ML models:")
        print("\n  Regression:")
        for m in reg:
            print(f"    {m}")
        print("\n  Classification:")
        for m in clf:
            print(f"    {m}")
        print("\nUse --model NAME to use a model directly (e.g. --model RandomForestClassifier)")
        return

    if not args.motivation:
        import argparse
        argparse.ArgumentParser().error("--motivation is required unless --list-models is used")

    agent = Agent()

    # Build loader + config 
    if args.data:
        loader = CSVDataLoader()
        sheet: int | str = int(args.sheet) if args.sheet.isdigit() else args.sheet
        data_config: dict = {"path": args.data, "sheet_name": sheet}
        if args.target:
            data_config["target"] = args.target
        if args.features:
            data_config["features"] = args.features
    else:
        loader = SyntheticDataLoader()
        data_config = {
            "n_samples": args.n_samples,
            "noise_sigma": args.noise_sigma,
            "random_seed": args.random_seed,
        }

    task_type = args.task_type

    # Wrap interactive chooser to pass task_type
    def choose_model(agent, metadata, suggestions):
        return _interactive_choose_model(agent, metadata, suggestions, task_type=task_type)

    # Run pipeline
    result = agent.run_full_pipeline(
        interface=loader,
        data_config=data_config,
        motivation=args.motivation,
        task_type=task_type,
        use_hyperparameter_tuning=not args.no_tuning,
        use_randomized_search=args.randomized_search,
        n_model_suggestions=5,
        skip_training=args.skip_train,
        choose_model_fn=choose_model,
        use_llm=not args.no_llm,
        fixed_model=args.model,
    )

    # Print results
    print("\nDataset metadata:")
    print(json.dumps(result.metadata, indent=2))

    if result.distribution_summary:
        print("\nData distribution (used for model recommendation):")
        print(result.distribution_summary)
    if result.distribution_plot_path:
        print(f"Distribution plot: {result.distribution_plot_path}")

    print("\nPlan:")
    for i, step in enumerate(result.steps, 1):
        print(f"  {i}. {step}")

    if result.suggestion:
        print("\nModel selection:")
        print(f"  Chosen model : {result.suggestion.model_name}")
        print(f"  Package      : {result.suggestion.package_name}")
        print(f"  Reason       : {result.suggestion.reason}")

    if args.skip_train:
        print("\n(Skipping train/report: --skip-train)")
        return

    if not result.success:
        print(f"\nPipeline error: {result.error}")
        return

    if result.metrics:
        print(f"\nMetrics ({task_type}):")
        print(json.dumps(result.metrics, indent=2))
    if result.plot_path:
        print(f"Plot saved: {result.plot_path}")
    if result.interpretation:
        print("\nLLM interpretation:")
        print(result.interpretation)


if __name__ == "__main__":
    main()
