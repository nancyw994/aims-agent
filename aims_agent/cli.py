import argparse
import json
import sys

from aims_agent.agent import Agent
from aims_agent.report_writer import write_pipeline_report
from aims_agent.synthetic_loader import SyntheticDataLoader
from aims_agent.csv_loader import CSVDataLoader
from aims_agent.matsci_data_ingestor import MaterialsProjectDataIngestor
from aims_agent.data_interface import get_metadata
from aims_agent.model_selector import suggest_models, ModelSuggestion, list_all_models, get_model_suggestion
from aims_agent.task_type_suggest import _heuristic_task_type, suggest_task_type


def _run_integrated_dataset_analysis_from_args(args) -> None:
    from aims_agent.model_strategy_analysis import run_model_strategy_analysis

    print("\nRunning integrated dataset analysis.")
    output_dir = run_model_strategy_analysis(
        data_path=args.data,
        target=args.target,
        output_root=args.report_dir,
        background_knowledge=args.background_knowledge,
        motivation=args.motivation,
        use_llm=not args.no_llm,
    )
    print(f"\nStrategy report: {output_dir / 'strategy_report.html'}")


def _resolve_task_type(
    agent: Agent,
    loader,
    data_config: dict,
    motivation: str,
    background_knowledge: str | None,
    task_type_arg: str,
    *,
    use_llm: bool,
) -> str:
    """
    If task_type_arg is 'auto', load data once, get LLM (or heuristic) suggestion, then let user confirm or override.
    Otherwise return task_type_arg unchanged.
    """
    if task_type_arg != "auto":
        return task_type_arg

    bundle = loader.load_dataset(data_config)
    if use_llm:
        tt, reason = suggest_task_type(agent, bundle, motivation, background_knowledge)
        print(f"\n── Task type (auto) ────────────────────────────────────────")
        print(f"LLM suggests: {tt}")
        print(f"Reason: {reason}")
    else:
        meta = get_metadata(bundle)
        tt, reason = _heuristic_task_type(bundle.df, meta["target"])
        print(f"\n── Task type (auto, --no-llm) ──────────────────────────────")
        print(f"Heuristic: {tt}")
        print(f"Reason: {reason}")

    if not sys.stdin.isatty():
        print("  (non-interactive: using suggestion above)")
        return tt

    print("\nConfirm task type: Enter = accept | 1 = regression | 2 = classification")
    try:
        choice = input("> ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print(f"\nUsing suggested: {tt}")
        return tt
    if not choice:
        return tt
    if choice in ("1", "r", "reg", "regression"):
        print("Using: regression")
        return "regression"
    if choice in ("2", "c", "clf", "classification"):
        print("Using: classification")
        return "classification"
    return tt


def _interactive_choose_model(
    agent: Agent,
    metadata: dict,
    suggestions: list,
    task_type: str = "regression",
) -> ModelSuggestion:
    """
    Show numbered model suggestions. User can:
    - Enter 1..N to choose one
    - Enter an explicit model name (e.g. RandomForestRegressor) to use directly
    - Type more requirements (e.g. prefer interpretability) to get new suggestions
    - Press Enter to use the first suggestion
    """
    valid_models = set(list_all_models(task_type))
    lower_to_model = {m.lower(): m for m in valid_models}

    while True:
        print(f"\nRecommended models (task_type={task_type}):")
        for i, s in enumerate(suggestions, 1):
            print(f"  {i}. {s.model_name} ({s.package_name}) — {s.reason}")

        if not sys.stdin.isatty():
            print("  (non-interactive: using first suggestion)")
            return suggestions[0]

        prompt_msg = (
            "Enter number 1–%d to choose, enter model name to force one, "
            "type more requirements for new suggestions, or Enter to use the first: "
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

        forced_name = lower_to_model.get(choice.lower())
        if forced_name:
            forced = get_model_suggestion(forced_name, task_type)
            if forced:
                print(f"Using user-selected model: {forced.model_name}")
                return forced
            print(f"Model '{forced_name}' is not valid for task_type='{task_type}'.")
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
    data_grp.add_argument(
        "--header-row",
        type=int,
        default=None,
        metavar="N",
        help="Explicit Excel header row index (0-based). If not set, use smart header detection.",
    )
    data_grp.add_argument(
        "--keep-na",
        action="store_true",
        help="Do not drop rows with missing feature values; still drops rows with missing target.",
    )
    data_grp.add_argument(
        "--impute-missing",
        action="store_true",
        help="When used with --keep-na, impute missing feature values (numeric: median, categorical: mode).",
    )
    data_grp.add_argument(
        "--materials-project",
        action="store_true",
        help="Fetch real materials data from the Materials Project summary API instead of using --data/synthetic.",
    )
    data_grp.add_argument(
        "--mp-api-key",
        default=None,
        metavar="KEY",
        help="Materials Project API key. Defaults to MP_API_KEY or MATERIALS_PROJECT_API_KEY.",
    )
    data_grp.add_argument(
        "--mp-chemsys",
        default=None,
        metavar="SYSTEM",
        help="Materials Project chemical system filter, e.g. Li-Fe-O.",
    )
    data_grp.add_argument(
        "--mp-elements",
        nargs="+",
        default=None,
        metavar="EL",
        help="Materials Project element filter, e.g. Li Fe O.",
    )
    data_grp.add_argument(
        "--mp-material-ids",
        nargs="+",
        default=None,
        metavar="MPID",
        help="Specific Materials Project IDs, e.g. mp-149 mp-13.",
    )
    data_grp.add_argument(
        "--mp-fields",
        nargs="+",
        default=None,
        metavar="FIELD",
        help="Materials Project summary fields to request. Defaults to common scalar ML fields.",
    )
    data_grp.add_argument(
        "--mp-limit",
        type=int,
        default=None,
        metavar="N",
        help="Limit Materials Project records after query.",
    )
    data_grp.add_argument(
        "--preprocessing-suggestion",
        default=None,
        metavar="TEXT",
        help="Optional LLM/domain guidance for missing values, outliers, and scaling.",
    )
    data_grp.add_argument(
        "--missing-strategy",
        choices=["drop", "impute"],
        default=None,
        help="Real-data missing feature handling. Overrides --preprocessing-suggestion.",
    )
    data_grp.add_argument(
        "--outlier-strategy",
        choices=["none", "iqr_clip", "iqr_drop"],
        default=None,
        help="Real-data outlier handling. Overrides --preprocessing-suggestion.",
    )
    data_grp.add_argument(
        "--scaling",
        choices=["none", "standard", "minmax"],
        default=None,
        help="Real-data feature scaling. Overrides --preprocessing-suggestion.",
    )
    data_grp.add_argument(
        "--preprocessed-output",
        default=None,
        metavar="PATH",
        help="Optional CSV path where preprocessed real materials data is saved.",
    )

    # Task
    task_grp = p.add_argument_group("Task")
    task_grp.add_argument(
        "--motivation",
        default=None,
        help="User's research goal in natural language. Required unless --list-models.",
    )
    task_grp.add_argument(
        "--background-knowledge",
        default=None,
        metavar="TEXT",
        help=(
            "Optional research abstract, domain context, constraints, prior findings, or hypotheses. "
            "The AI uses this when planning, model selection, and result interpretation."
        ),
    )
    task_grp.add_argument(
        "--task-type",
        choices=["regression", "classification", "auto"],
        default="regression",
        help=(
            "ML task type (default: regression). "
            "Use 'auto' to load data, get an LLM suggestion from motivation + target + background "
            "(or heuristic with --no-llm), then confirm or override interactively before the rest of the pipeline."
        ),
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
    train_grp.add_argument(
        "--use-custom-codegen",
        action="store_true",
        help="Enable custom code generation and execution step.",
    )
    train_grp.add_argument(
        "--custom-code-request",
        type=str,
        default=None,
        metavar="TEXT",
        help="Specific instruction for generated custom component (used with --use-custom-codegen).",
    )
    train_grp.add_argument(
        "--generated-code-dir",
        type=str,
        default="generated_code",
        metavar="DIR",
        help="Directory to save generated Python modules (default: generated_code).",
    )
    train_grp.add_argument(
        "--multi-agent",
        action="store_true",
        help=(
            "Use Execution Path Resolver (builtin / dynamic_import / codegen) before training; "
            "optional model CodeGen + debug retries when no reliable import path."
        ),
    )
    train_grp.add_argument(
        "--max-codegen-retries",
        type=int,
        default=2,
        metavar="N",
        help="Max LLM repair rounds after failed load of generated estimator (default: 2).",
    )

    out_grp = p.add_argument_group("Output")
    out_grp.add_argument(
        "--report-dir",
        default="results",
        metavar="DIR",
        help="Base directory under which a new run_* folder and HTML report will be saved (default: results).",
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

    if args.data:
        _run_integrated_dataset_analysis_from_args(args)
        return

    agent = Agent()

    # Build loader + config 
    if args.materials_project:
        loader = MaterialsProjectDataIngestor()
        data_config = {
            "api_key": args.mp_api_key,
            "chemsys": args.mp_chemsys,
            "elements": args.mp_elements,
            "material_ids": args.mp_material_ids,
            "fields": args.mp_fields,
            "limit": args.mp_limit,
            "target": args.target,
            "features": args.features,
            "preprocessing_suggestion": args.preprocessing_suggestion,
            "output_path": args.preprocessed_output,
        }
        if args.missing_strategy:
            data_config["missing_strategy"] = args.missing_strategy
        if args.outlier_strategy:
            data_config["outlier_strategy"] = args.outlier_strategy
        if args.scaling:
            data_config["scaling"] = args.scaling
    elif args.data:
        loader = CSVDataLoader()
        sheet: int | str = int(args.sheet) if args.sheet.isdigit() else args.sheet
        data_config: dict = {
            "path": args.data,
            "sheet_name": sheet,
            "header_row": args.header_row,
            "drop_na": not args.keep_na,
            "impute_missing": args.impute_missing,
            "auto_recover_small_sample": True,
        }
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

    task_type = _resolve_task_type(
        agent,
        loader,
        data_config,
        args.motivation,
        args.background_knowledge,
        args.task_type,
        use_llm=not args.no_llm,
    )

    # Wrap interactive chooser to pass task_type
    def choose_model(agent, metadata, suggestions):
        return _interactive_choose_model(agent, metadata, suggestions, task_type=task_type)

    # Run pipeline
    result = agent.run_full_pipeline(
        interface=loader,
        data_config=data_config,
        motivation=args.motivation,
        background_knowledge=args.background_knowledge,
        task_type=task_type,
        use_hyperparameter_tuning=not args.no_tuning,
        use_randomized_search=args.randomized_search,
        n_model_suggestions=5,
        skip_training=args.skip_train,
        choose_model_fn=choose_model,
        use_llm=not args.no_llm,
        fixed_model=args.model,
        use_custom_codegen=args.use_custom_codegen,
        custom_code_request=args.custom_code_request,
        generated_code_dir=args.generated_code_dir,
        multi_agent=args.multi_agent,
        max_codegen_retries=args.max_codegen_retries,
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
    if args.multi_agent and (result.execution_path or result.path_reason):
        print("\nExecution path (multi-agent):")
        print(f"  Path   : {result.execution_path or '(n/a)'}")
        print(f"  Reason : {result.path_reason or '(n/a)'}")
        if result.generated_model_wrapper_path:
            print(f"  Generated model module: {result.generated_model_wrapper_path}")
        if result.training_validation_message:
            print(
                f"  Training validation: {'OK' if result.training_validation_ok else 'WARN'} — "
                f"{result.training_validation_message}"
            )
    if result.generated_code_path:
        print("\nCustom code component:")
        print(f"  Module path  : {result.generated_code_path}")
        if result.generated_code_note:
            print(f"  Agent note   : {result.generated_code_note}")
    if args.multi_agent and (result.self_correction_summary or result.self_correction_log_path):
        print("\nSelf-correction:")
        print(f"  Attempts     : {result.self_correction_attempts}")
        if result.self_correction_attempts > 0:
            print(f"  Success      : {result.self_correction_success}")
        else:
            print(f"  Success      : N/A (no correction attempted)")
        if result.self_correction_summary:
            print(f"  Summary      : {result.self_correction_summary}")
        if result.self_correction_log_path:
            print(f"  Log path     : {result.self_correction_log_path}")
        if result.self_correction_report:
            print(f"  Aggregation  : {result.self_correction_report.get('summary', '')}")

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

    # Write plain-language report
    try:
        report_path = write_pipeline_report(result, output_dir=args.report_dir)
        print(f"\nReport saved: {report_path}")
    except Exception as exc:
        print(f"\n[Warning] Could not write report: {exc}")


if __name__ == "__main__":
    main()
