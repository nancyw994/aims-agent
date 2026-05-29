from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, List, Literal, Mapping

import numpy as np

from aims_agent.code_writer import (
    codegen_required_packages,
    dl_backend_candidates,
    dl_backend_required_packages,
    execute_generated_component,
    generate_code_file,
    infer_preferred_dl_backend,
    infer_codegen_mode,
    validate_component_output,
)
from aims_agent.data_interface import DataInterface, DatasetBundle, get_metadata
from aims_agent.dependency_manager import ensure_package_installed
from aims_agent.distribution import analyze_distribution, plot_distribution
from aims_agent.llm import LMF_LLM
from aims_agent.model_selector import ModelSuggestion, get_default_suggestion, get_model_suggestion, load_model_class, suggest_model, suggest_models
from aims_agent.model_trainer import ModelTrainer
from aims_agent.results_analyzer import compute_metrics, interpret_from_metrics, interpret_with_llm, plot_results
from aims_agent.uncertainty_evaluator import UncertaintyEvaluator
from aims_agent.validator import (
    validate_dl_training_trace,
    validate_training_result_detailed,
)


@dataclass
class PipelineResult:
    """Result of running the full ML pipeline."""

    steps: List[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    background_knowledge: str = ""
    distribution_summary: str = ""
    distribution_plot_path: str = ""
    generated_code_path: str = ""
    generated_code_note: str = ""
    suggestion: ModelSuggestion | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    plot_path: str = ""
    interpretation: str = ""
    success: bool = True
    error: str = ""
    # Multi-agent execution path (model wrapper / resolver)
    execution_path: str = ""
    path_reason: str = ""
    generated_model_wrapper_path: str = ""
    training_validation_ok: bool = True
    training_validation_message: str = ""
    self_correction_attempts: int = 0
    self_correction_success: bool = False
    self_correction_log_path: str = ""
    self_correction_summary: str = ""
    self_correction_report: dict = field(default_factory=dict)
    # Report-level context
    motivation: str = ""
    task_type: str = ""


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
        return interface.load_dataset(config)

    def retrieve_real_materials_data(self, config: Mapping[str, Any]) -> DatasetBundle:
        """Load real materials data through the Materials Project/local MatSci ingestor."""
        from aims_agent.matsci_data_ingestor import MaterialsProjectDataIngestor

        return self.retrieve_data(MaterialsProjectDataIngestor(), config)

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
        background_knowledge: str | None = None,
        *,
        task_type: Literal["regression", "classification"] = "regression",
        use_hyperparameter_tuning: bool = True,
        use_randomized_search: bool = False,
        n_model_suggestions: int = 1,
        skip_training: bool = False,
        choose_model_fn: Any | None = None,
        use_llm: bool = True,
        fixed_model: str | None = None,
        use_custom_codegen: bool = False,
        custom_code_request: str | None = None,
        generated_code_dir: str = "generated_code",
        multi_agent: bool = False,
        max_codegen_retries: int = 2,
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
            multi_agent: If True, use Execution Path Resolver (builtin / dynamic_import / codegen)
                and optional model CodeGen + debug retries before training.
            max_codegen_retries: Max LLM repair attempts after failed load of generated estimator.

        Returns:
            PipelineResult with steps, metrics, plot path, and LLM interpretation.
        """
        result = PipelineResult()
        result.motivation = motivation
        result.task_type = task_type

        try:
            # Step 1: Data ingestion
            bundle = self.retrieve_data(interface, data_config)
            metadata = get_metadata(bundle)
            result.metadata = metadata
            result.background_knowledge = background_knowledge or ""

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

                plan_actions = plan_workflow_steps(
                    self,
                    motivation,
                    dataset_metadata=metadata,
                    background_knowledge=background_knowledge,
                    include_codegen=use_custom_codegen,
                )
            else:
                plan_actions = [
                    {"action": "select_model", "description": "Select ML model for the task"},
                    {"action": "codegen", "description": "Generate and execute a custom code component"},
                    {"action": "train", "description": "Split data, train model (with optional hyperparameter tuning)"},
                    {"action": "evaluate", "description": "Evaluate on test set and generate plots"},
                    {"action": "interpret", "description": "Summarize metrics and interpretation"},
                ]
                if not use_custom_codegen:
                    plan_actions = [p for p in plan_actions if p["action"] != "codegen"]
            result.steps = [p.get("description", p.get("action", "")) for p in plan_actions]

            # Step 3: Execute plan
            suggestion = None
            y_true, y_pred = None, None
            y_std = None
            if task_type == "regression" and not skip_training:
                UncertaintyEvaluator.check_availability()

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
                        if background_knowledge:
                            extra_ctx = (
                                f"{extra_ctx}\n\n"
                                "User-provided background knowledge / constraints:\n"
                                f"{background_knowledge.strip()}"
                            )
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

                # codegen
                if action == "codegen":
                    if not use_custom_codegen:
                        continue
                    if not use_llm:
                        continue
                    code_request = (
                        custom_code_request
                        or "Generate a robust custom preprocessing component for this materials dataset."
                    )
                    codegen_mode = infer_codegen_mode(code_request)
                    selected_dl_backend = ""
                    for pkg in codegen_required_packages(codegen_mode):
                        if not ensure_package_installed(pkg):
                            result.success = False
                            result.error = (
                                f"CodeGen mode '{codegen_mode}' requires package '{pkg}', "
                                "but installation failed."
                            )
                            return result
                    if codegen_mode == "deep_learning":
                        preferred = infer_preferred_dl_backend(code_request)
                        install_errors: list[str] = []
                        for backend in dl_backend_candidates(preferred):
                            pkgs = dl_backend_required_packages(backend)
                            if all(ensure_package_installed(pkg) for pkg in pkgs):
                                selected_dl_backend = backend
                                break
                            install_errors.append(f"{backend}: failed to install {pkgs}")
                        if not selected_dl_backend:
                            result.success = False
                            result.error = (
                                "Deep-learning codegen backend install failed. Tried: "
                                + " | ".join(install_errors)
                            )
                            return result
                        code_request = (
                            f"{code_request}\n\nBackend requirement: use {selected_dl_backend}."
                            " Return optional loss_history (list of floats, >=2 steps) and "
                            "optional gradient_norms in the output dict when training is performed."
                        )
                    module_name = f"custom_component_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    result.generated_code_path = generate_code_file(
                        self,
                        request=code_request,
                        dataset_metadata=metadata,
                        background_knowledge=background_knowledge,
                        task_type=task_type,
                        codegen_mode=codegen_mode,
                        output_dir=generated_code_dir,
                        module_name=module_name,
                    )
                    cg_result = execute_generated_component(
                        result.generated_code_path,
                        df=bundle.df,
                        features=list(metadata["features"]),
                        target=metadata["target"],
                        task_type=task_type,
                    )
                    ok, msg = validate_component_output(
                        cg_result,
                        original_features=list(metadata["features"]),
                        row_count=len(bundle.df),
                    )
                    if not ok:
                        result.success = False
                        result.error = f"Generated component output validation failed: {msg}"
                        return result
                    if codegen_mode == "deep_learning":
                        dl_ok, dl_msg = validate_dl_training_trace(
                            cg_result.get("loss_history"),
                            cg_result.get("gradient_norms"),
                            min_steps=2,
                        )
                        if not dl_ok:
                            result.success = False
                            result.error = f"Deep-learning component validation failed: {dl_msg}"
                            return result
                    if "features" in cg_result and isinstance(cg_result["features"], list):
                        valid_features = [f for f in cg_result["features"] if f in bundle.df.columns]
                        if valid_features:
                            metadata["features"] = valid_features
                    note = str(cg_result.get("note", "")).strip()
                    backend_note = f", backend={selected_dl_backend}" if selected_dl_backend else ""
                    result.generated_code_note = f"[mode={codegen_mode}{backend_note}] {note}".strip()
                    continue

                # train
                if action == "train":
                    if skip_training:
                        continue
                    if suggestion is None:
                        continue
                    if multi_agent:
                        from aims_agent.orchestrator import run_training_phase_multi_agent_with_fallback

                        fallback = get_default_suggestion(task_type)
                        tpr = run_training_phase_multi_agent_with_fallback(
                            self,
                            suggestion=suggestion,
                            fallback_suggestion=fallback,
                            ensure_package_installed_fn=ensure_package_installed,
                            task_type=task_type,
                            metadata=metadata,
                            df=bundle.df,
                            use_llm=use_llm,
                            background_knowledge=background_knowledge,
                            generated_code_dir=generated_code_dir,
                            max_codegen_retries=max(0, max_codegen_retries),
                            use_hyperparameter_tuning=use_hyperparameter_tuning,
                            use_randomized_search=use_randomized_search,
                        )
                        if tpr.fallback_used:
                            print(
                                f"[Multi-agent] primary model failed; falling back to {tpr.used_suggestion.model_name}"
                            )
                            suggestion = tpr.used_suggestion
                            result.suggestion = suggestion
                        mcr = tpr.resolution
                        model_class = mcr.model_class
                        y_true, y_pred = tpr.y_true, tpr.y_pred
                        result.execution_path = mcr.execution_path
                        result.path_reason = mcr.path_reason
                        result.generated_model_wrapper_path = mcr.generated_model_wrapper_path or ""
                        result.self_correction_attempts = mcr.self_correction_attempts
                        result.self_correction_success = mcr.self_correction_success
                        result.self_correction_log_path = mcr.self_correction_log_path
                        result.self_correction_summary = mcr.self_correction_summary
                        if mcr.self_correction_log_path:
                            from aims_agent.self_correction_report import aggregate_self_correction_logs
                            result.self_correction_report = aggregate_self_correction_logs(
                                mcr.self_correction_log_path
                            )
                        result.training_validation_ok = tpr.training_validation_ok
                        result.training_validation_message = (
                            f"[{tpr.training_validation_code}] {tpr.training_validation_message}"
                        )
                        if result.generated_model_wrapper_path:
                            print(
                                f"[Multi-agent] execution_path={result.execution_path} "
                                f"model_module={result.generated_model_wrapper_path}"
                            )
                        else:
                            print(f"[Multi-agent] execution_path={result.execution_path} — {result.path_reason}")
                    else:
                        try:
                            model_class = load_model_class(suggestion)
                        except Exception as e:
                            fallback = get_default_suggestion(task_type)
                            if fallback.package_name != suggestion.package_name and ensure_package_installed(
                                fallback.package_name
                            ):
                                print(
                                    f"[Model] loads {suggestion.model_name} failed ({e})then changes to {fallback.model_name}"
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
                    if task_type == "regression":
                        uq_pred = trainer.predict_with_uncertainty()
                        y_std = uq_pred.get("y_std")
                        if y_std is None or not any(float(v) > 0 for v in y_std):
                            residual_std = float(np.std(np.asarray(y_true) - np.asarray(y_pred), ddof=1))
                            y_std = np.full(len(y_pred), max(residual_std, 1e-8))
                    continue

                # evaluate
                if action == "evaluate":
                    if y_true is not None and y_pred is not None and suggestion is not None:
                        result.metrics = compute_metrics(y_true, y_pred, task_type=task_type)
                        if task_type == "regression":
                            if y_std is None:
                                residual_std = float(np.std(np.asarray(y_true) - np.asarray(y_pred), ddof=1))
                                y_std = np.full(len(y_pred), max(residual_std, 1e-8))
                            uq_summary, _ = UncertaintyEvaluator.evaluate_all(
                                y_true,
                                y_pred,
                                y_std,
                                verbose=False,
                            )
                            uq_coverage = UncertaintyEvaluator.compute_coverage(y_true, y_pred, y_std)
                            result.metrics.update(
                                {
                                    "UQ_Calibration_MAE": uq_summary.get("calibration_mae"),
                                    "UQ_Calibration_RMSE": uq_summary.get("calibration_rmse"),
                                    "UQ_Miscalibration_Area": uq_summary.get("miscalibration_area"),
                                    "UQ_Sharpness": uq_summary.get("sharpness"),
                                    "UQ_NLL": uq_summary.get("nll"),
                                    "UQ_CRPS": uq_summary.get("crps"),
                                    "UQ_Coverage_68": uq_coverage.get(0.68),
                                    "UQ_Coverage_95": uq_coverage.get(0.95),
                                    "UQ_Coverage_99": uq_coverage.get(0.99),
                                }
                            )
                        result.plot_path = plot_results(
                            y_true, y_pred, task_type=task_type
                        )
                        if multi_agent:
                            if not result.training_validation_ok:
                                # attach metric-level checks to existing train-phase validation status
                                out = validate_training_result_detailed(
                                    y_true,
                                    y_pred,
                                    metrics=result.metrics,
                                    task_type=task_type,
                                )
                                result.training_validation_ok = out.ok
                                result.training_validation_message = f"[{out.code}] {out.message}"
                            if not result.training_validation_ok:
                                print(f"[Validator] {result.training_validation_message}")
                    continue

                # interpret
                if action == "interpret":
                    if result.metrics and suggestion is not None:
                        if use_llm:
                            try:
                                result.interpretation = interpret_with_llm(
                                    self,
                                    result.metrics,
                                    suggestion.model_name,
                                    task_type=task_type,
                                    background_knowledge=background_knowledge,
                                )
                            except Exception as e:
                                result.interpretation = interpret_from_metrics(
                                    result.metrics, suggestion.model_name, task_type=task_type
                                )
                                result.interpretation = (
                                    f"[LLM interprets failed ({e})，use local summary]\n\n" + result.interpretation
                                )
                        else:
                            result.interpretation = interpret_from_metrics(
                                result.metrics, suggestion.model_name, task_type=task_type
                            )
                    continue

        except Exception as e:
            result.success = False
            result.error = str(e)
            return result

        return result
