from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Literal, Mapping

from aims_agent.data_interface import DataInterface, DatasetBundle, get_metadata
from aims_agent.dependency_manager import ensure_package_installed
from aims_agent.distribution import analyze_distribution, plot_distribution
from aims_agent.llm import LMF_LLM
from aims_agent.model_selector import ModelSuggestion, get_default_suggestion, get_model_suggestion, load_model_class, suggest_model, suggest_models
from aims_agent.model_trainer import ModelTrainer
from aims_agent.results_analyzer import compute_metrics, interpret_from_metrics, interpret_with_llm, plot_results


@dataclass
class PipelineResult:
    """Result of running the full ML pipeline."""

    steps: List[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    distribution_summary: str = ""
    distribution_plot_path: str = ""
    suggestion: ModelSuggestion | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    plot_path: str = ""
    interpretation: str = ""
    success: bool = True
    error: str = ""


class Agent:
    """
    High-level agent wrapper around the LLM and data interface.

    Orchestrates the full ML pipeline: data ingestion, planning, model selection,
    dependency installation, training, evaluation, plotting, and LLM interpretation.
    """

    def __init__(self, llm_call=None):
        self._llm_call = llm_call if llm_call is not None else LMF_LLM

    def call_llm(self, prompt: str) -> str:
        try:
            return self._llm_call(prompt)
        except Exception as e:
            raise RuntimeError(f"LLM call failed: {e}") from e

    def retrieve_data(self, interface: DataInterface, config: Mapping[str, Any]) -> DatasetBundle:
        """
        Entry point for the DATA INTERFACE step.

        Downstream code should call this method instead of interacting
        with concrete loaders directly.
        """
        return interface.load_dataset(config)

    def select_model_and_ensure_deps(
        self,
        features: List[str],
        target: str,
        *,
        task_hint: str = "regression",
        extra_context: str | None = None,
    ) -> tuple[ModelSuggestion, bool]:
        """
        Query the LLM for a model suggestion, then ensure its package is installed.

        Returns (suggestion, installed_ok). If the package was missing and install
        failed, installed_ok is False; the error is logged by dependency_manager.
        """
        suggestion = suggest_model(
            self,
            features,
            target,
            task_hint=task_hint,
            extra_context=extra_context,
        )
        installed_ok = ensure_package_installed(suggestion.package_name)
        return suggestion, installed_ok

    def run_full_pipeline(
        self,
        interface: DataInterface,
        data_config: Mapping[str, Any],
        motivation: str,
        *,
        task_type: Literal["regression", "classification"] = "regression",
        use_hyperparameter_tuning: bool = True,
        use_randomized_search: bool = False,
        n_model_suggestions: int = 1,
        skip_training: bool = False,
        choose_model_fn: Any | None = None,
        use_llm: bool = True,
        fixed_model: str | None = None,
    ) -> PipelineResult:
        """
        Execute the full ML pipeline based on the LLM plan.

        Dynamically orchestrates: data load → plan → model select → train → analyze.
        The plan steps from the LLM guide the workflow; each step maps to module calls.

        Args:
            interface: Data loader (e.g. SyntheticDataLoader, CSVDataLoader).
            data_config: Config passed to interface.load_dataset().
            motivation: User's research goal (used for planning).
            task_type: "regression" or "classification".
            use_hyperparameter_tuning: If True, use GridSearchCV/RandomizedSearchCV.
            use_randomized_search: If True, use RandomizedSearchCV instead of GridSearchCV.
            n_model_suggestions: Number of model suggestions to request (for interactive choice).
            skip_training: If True, stop after model selection.
            choose_model_fn: Optional callback(agent, metadata, suggestions) -> ModelSuggestion.
            use_llm: If False, use default plan/model/interpretation (no API calls).
            fixed_model: If set, use this model name directly (skip LLM). Must be in list_all_models().

        Returns:
            PipelineResult with steps, metrics, plot path, and LLM interpretation.
        """
        result = PipelineResult()

        try:
            # Step 1: Data ingestion
            bundle = self.retrieve_data(interface, data_config)
            metadata = get_metadata(bundle)
            result.metadata = metadata

            # Step 1b: Distribution analysis
            dist_stats = analyze_distribution(
                bundle.df,
                metadata["features"],
                metadata["target"],
                task_type=task_type,
            )
            result.distribution_summary = dist_stats["summary_text"]
            result.distribution_plot_path = plot_distribution(
                bundle.df,
                metadata["features"],
                metadata["target"],
                task_type=task_type,
            )
            print("\n── Data distribution ─────────────────────────────────────")
            print(result.distribution_summary)
            print(f"Distribution plot: {result.distribution_plot_path}")

            # Step 2: Plan
            if use_llm:
                from aims_agent.planning import plan_workflow_steps

                plan_actions = plan_workflow_steps(self, motivation, dataset_metadata=metadata)
            else:
                plan_actions = [
                    {"action": "select_model", "description": "Select ML model for the task"},
                    {"action": "train", "description": "Split data, train model (with optional hyperparameter tuning)"},
                    {"action": "evaluate", "description": "Evaluate on test set and generate plots"},
                    {"action": "interpret", "description": "Summarize metrics and interpretation"},
                ]
            result.steps = [p.get("description", p.get("action", "")) for p in plan_actions]

            # Step 3: Execute plan
            suggestion = None
            y_true, y_pred = None, None

            for plan_item in plan_actions:
                action = (plan_item.get("action") or "").strip().lower().replace(" ", "_")
                if not action:
                    continue

                # select_model 
                if action == "select_model":
                    if fixed_model:
                        suggestion = get_model_suggestion(fixed_model, task_type)
                        if suggestion is None:
                            from aims_agent.model_selector import list_all_models
                            valid = list_all_models(task_type)
                            raise ValueError(f"Unknown model '{fixed_model}'. Valid: {valid}")
                        result.suggestion = suggestion
                    elif use_llm:
                        extra_ctx = result.distribution_summary
                        if metadata.get("description"):
                            extra_ctx = f"{metadata['description']}\n\n{extra_ctx}"
                        suggestions = suggest_models(
                            self,
                            features=metadata["features"],
                            target=metadata["target"],
                            n_suggestions=max(1, n_model_suggestions),
                            task_hint=task_type,
                            extra_context=extra_ctx,
                        )
                        if choose_model_fn:
                            suggestion = choose_model_fn(self, metadata, suggestions)
                        else:
                            suggestion = suggestions[0]
                        result.suggestion = suggestion
                    else:
                        suggestion = get_default_suggestion(task_type)
                        result.suggestion = suggestion

                    installed_ok = ensure_package_installed(suggestion.package_name)
                    if not installed_ok:
                        fallback = get_default_suggestion(task_type)
                        if fallback.package_name != suggestion.package_name and ensure_package_installed(
                            fallback.package_name
                        ):
                            print(
                                f"[Model] {suggestion.model_name} 不可用（如缺少 libomp），改用 {fallback.model_name}"
                            )
                            suggestion = fallback
                            result.suggestion = suggestion
                        else:
                            result.success = False
                            result.error = f"Failed to install package: {suggestion.package_name}. Mac: brew install libomp"
                            return result
                    continue

                # train
                if action == "train":
                    if skip_training:
                        continue
                    if suggestion is None:
                        continue
                    try:
                        model_class = load_model_class(suggestion)
                    except Exception as e:
                        fallback = get_default_suggestion(task_type)
                        if fallback.package_name != suggestion.package_name and ensure_package_installed(
                            fallback.package_name
                        ):
                            print(
                                f"[Model] 加载 {suggestion.model_name} 失败 ({e})，改用 {fallback.model_name}"
                            )
                            suggestion = fallback
                            result.suggestion = suggestion
                            model_class = load_model_class(suggestion)
                        else:
                            raise
                    trainer = ModelTrainer(
                        model_class,
                        task_type=task_type,
                        use_hyperparameter_tuning=use_hyperparameter_tuning,
                        use_randomized_search=use_randomized_search,
                    )
                    trainer.prepare_data(bundle.df, metadata["features"], metadata["target"])
                    trainer.train()
                    y_true, y_pred = trainer.predict()
                    continue

                # evaluate 
                if action == "evaluate":
                    if y_true is not None and y_pred is not None and suggestion is not None:
                        result.metrics = compute_metrics(y_true, y_pred, task_type=task_type)
                        result.plot_path = plot_results(
                            y_true, y_pred, task_type=task_type
                        )
                    continue

                # interpret
                if action == "interpret":
                    if result.metrics and suggestion is not None:
                        if use_llm:
                            try:
                                result.interpretation = interpret_with_llm(
                                    self, result.metrics, suggestion.model_name, task_type=task_type
                                )
                            except Exception as e:
                                result.interpretation = interpret_from_metrics(
                                    result.metrics, suggestion.model_name, task_type=task_type
                                )
                                result.interpretation = (
                                    f"[LLM 解释失败 ({e})，使用本地摘要]\n\n" + result.interpretation
                                )
                        else:
                            result.interpretation = interpret_from_metrics(
                                result.metrics, suggestion.model_name, task_type=task_type
                            )
                    continue

        except Exception as e:
            result.success = False
            result.error = str(e)
            raise

        return result
