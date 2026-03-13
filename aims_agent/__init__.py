"""AI Agent for ML in Materials Science."""

from aims_agent.agent import Agent, PipelineResult
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

__all__ = [
    "Agent",
    "PipelineResult",
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
]

