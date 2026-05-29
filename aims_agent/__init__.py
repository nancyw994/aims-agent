"""AI Agent for ML in Materials Science."""

from aims_agent.agent import Agent, PipelineResult
from aims_agent.orchestrator import (
    ModelClassResolution,
    TrainingPhaseResult,
    resolve_model_class_multi_agent,
    run_training_phase_multi_agent,
    run_training_phase_multi_agent_with_fallback,
)
from aims_agent.agents.debug_agent import (
    DebugPatchResult,
    RetryDecision,
    SelfCorrectionAgent,
    repair_model_module_code,
)
from aims_agent.path_resolver import ExecutionPathResolver, PathDecision, PathKind
from aims_agent.specs import CodeGenSpec, build_model_codegen_spec
from aims_agent.validator import (
    ValidationOutcome,
    validate_estimator_contract,
    validate_training_result_detailed,
    validate_training_result,
    validate_dl_training_trace,
)
from aims_agent.failure_codes import ALL_FAILURE_CODES
from aims_agent.self_correction_report import aggregate_self_correction_logs
from aims_agent.llm import LMF_LLM
from aims_agent.planning import plan_steps
from aims_agent.data_interface import (
    DatasetSchema,
    DatasetBundle,
    DataInterface,
    validate_schema,
    get_metadata,
)
from aims_agent.synthetic_loader import SyntheticDataLoader, save_example_csv
from aims_agent.csv_loader import CSVDataLoader
from aims_agent.matsci_data_ingestor import (
    MaterialsProjectDataIngestor,
    PreprocessingReport,
    clean_and_preprocess_materials_data,
    fetch_materials_project_summary,
    load_tabular_materials_file,
    preprocessing_policy_from_text,
)
from aims_agent.data_analyzer import (
    DataProfile,
    FeatureProfile,
    StrategyArtifact,
    analyze_and_formulate_strategy,
    build_strategy_prompt,
    formulate_strategy,
    profile_dataset,
    write_profile_outputs,
)
from aims_agent.pretrained_model_handler import (
    DEFAULT_PRETRAINED_MODEL,
    MODEL_TARGETS,
    MatGLPretrainedModelHandler,
    PretrainedBenchmarkResult,
    PretrainedModelChoice,
    benchmark_pretrained_model,
    fetch_materials_project_structures,
    identify_pretrained_models,
)
from aims_agent.model_selector import (
    ModelSuggestion,
    suggest_model,
    suggest_models,
    load_model_class,
    list_all_models,
    get_model_suggestion,
)
from aims_agent.dependency_manager import ensure_package_installed, INSTALL_LOG
from aims_agent.model_trainer import ModelTrainer
from aims_agent.results_analyzer import compute_metrics, plot_results, interpret_with_llm
from aims_agent.code_writer import (
    codegen_required_packages,
    build_code_generation_prompt,
    dl_backend_candidates,
    dl_backend_required_packages,
    extract_python_code,
    infer_codegen_mode,
    infer_preferred_dl_backend,
    validate_python_syntax,
    validate_component_output,
    save_generated_code,
    generate_code_file,
    load_generated_module,
    execute_generated_component,
)

__all__ = [
    "Agent",
    "PipelineResult",
    "ModelClassResolution",
    "TrainingPhaseResult",
    "resolve_model_class_multi_agent",
    "run_training_phase_multi_agent",
    "run_training_phase_multi_agent_with_fallback",
    "DebugPatchResult",
    "RetryDecision",
    "SelfCorrectionAgent",
    "repair_model_module_code",
    "ExecutionPathResolver",
    "PathDecision",
    "PathKind",
    "CodeGenSpec",
    "build_model_codegen_spec",
    "ValidationOutcome",
    "validate_estimator_contract",
    "validate_training_result_detailed",
    "validate_training_result",
    "validate_dl_training_trace",
    "ALL_FAILURE_CODES",
    "aggregate_self_correction_logs",
    "LMF_LLM",
    "plan_steps",
    "DatasetSchema",
    "DatasetBundle",
    "DataInterface",
    "SyntheticDataLoader",
    "save_example_csv",
    "CSVDataLoader",
    "MaterialsProjectDataIngestor",
    "PreprocessingReport",
    "clean_and_preprocess_materials_data",
    "fetch_materials_project_summary",
    "load_tabular_materials_file",
    "preprocessing_policy_from_text",
    "DataProfile",
    "FeatureProfile",
    "StrategyArtifact",
    "analyze_and_formulate_strategy",
    "build_strategy_prompt",
    "formulate_strategy",
    "profile_dataset",
    "write_profile_outputs",
    "DEFAULT_PRETRAINED_MODEL",
    "MODEL_TARGETS",
    "MatGLPretrainedModelHandler",
    "PretrainedBenchmarkResult",
    "PretrainedModelChoice",
    "benchmark_pretrained_model",
    "fetch_materials_project_structures",
    "identify_pretrained_models",
    "validate_schema",
    "get_metadata",
    "ModelSuggestion",
    "suggest_model",
    "suggest_models",
    "ensure_package_installed",
    "INSTALL_LOG",
    "load_model_class",
    "ModelTrainer",
    "compute_metrics",
    "plot_results",
    "interpret_with_llm",
    "build_code_generation_prompt",
    "infer_codegen_mode",
    "codegen_required_packages",
    "infer_preferred_dl_backend",
    "dl_backend_candidates",
    "dl_backend_required_packages",
    "extract_python_code",
    "validate_python_syntax",
    "validate_component_output",
    "save_generated_code",
    "generate_code_file",
    "load_generated_module",
    "execute_generated_component",
]
