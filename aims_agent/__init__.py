"""AI Agent for ML in Materials Science."""

from aims_agent.agent import Agent, PipelineResult
from aims_agent.orchestrator import ModelClassResolution, resolve_model_class_multi_agent
from aims_agent.path_resolver import ExecutionPathResolver, PathDecision, PathKind
from aims_agent.specs import CodeGenSpec, build_model_codegen_spec
from aims_agent.validator import validate_training_result
from aims_agent.validator import validate_dl_training_trace
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
    "resolve_model_class_multi_agent",
    "ExecutionPathResolver",
    "PathDecision",
    "PathKind",
    "CodeGenSpec",
    "build_model_codegen_spec",
    "validate_training_result",
    "validate_dl_training_trace",
    "LMF_LLM",
    "plan_steps",
    "DatasetSchema",
    "DatasetBundle",
    "DataInterface",
    "SyntheticDataLoader",
    "save_example_csv",
    "CSVDataLoader",
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

