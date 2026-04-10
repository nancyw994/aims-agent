"""
Structured specs consumed by the CodeGen agent (model wrapper generation).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from aims_agent.model_selector import ModelSuggestion

from aims_agent.path_resolver import PathDecision


@dataclass
class CodeGenSpec:
    """Contract for generating a sklearn-compatible estimator module."""

    model_name: str
    task_type: str
    required_interface: str
    import_path_hint: str | None
    package_name: str | None
    constraints: list[str] = field(default_factory=list)
    framework: str = "sklearn_compatible"
    reason: str = ""

    def to_prompt_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_model_codegen_spec(
    suggestion: ModelSuggestion,
    task_type: str,
    path_decision: PathDecision,
) -> CodeGenSpec:
    """Build a spec from advisor output and execution-path resolution."""
    constraints = [
        f'Implement a single class named "GeneratedEstimator" with fit(self, X, y) and predict(self, X).',
        "X may be a numpy ndarray or pandas DataFrame; y is 1d array-like.",
        "predict(X) must return a 1d numpy array with one prediction per row of X.",
        "Do not read or write files; no subprocess, os.system, eval, or exec.",
        "Use numpy and scikit-learn only unless the spec explicitly requires otherwise.",
    ]
    if task_type == "classification":
        constraints.append(
            "For classification, predict must return discrete class labels suitable for accuracy/F1."
        )
    else:
        constraints.append("For regression, predict must return floating-point values.")

    return CodeGenSpec(
        model_name=suggestion.model_name,
        task_type=task_type,
        required_interface="fit_predict",
        import_path_hint=(suggestion.import_path or None) or None,
        package_name=suggestion.package_name or None,
        constraints=constraints,
        framework="sklearn_compatible",
        reason=path_decision.reason,
    )


def compact_metadata_for_codegen(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Subset of dataset metadata safe to embed in codegen prompts."""
    return {
        "features": list(metadata.get("features", [])),
        "target": metadata.get("target", ""),
        "shape": metadata.get("shape", {}),
        "dtypes": metadata.get("dtypes", {}),
        "description": (metadata.get("description") or "")[:2000],
    }


__all__ = [
    "CodeGenSpec",
    "build_model_codegen_spec",
    "compact_metadata_for_codegen",
]
