"""
Execution path resolver: builtin map, dynamic import, or codegen.

Rule-first (no LLM): decides how the selected model can be executed.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Literal

from aims_agent.model_selector import MODEL_IMPORT_MAP, ModelSuggestion

PathKind = Literal["builtin", "dynamic_import", "codegen"]


@dataclass
class PathDecision:
    path: PathKind
    reason: str


def try_import_class(import_path: str) -> bool:
    """Return True if import_path (module.Class) can be imported."""
    if not import_path or "." not in import_path:
        return False
    try:
        module_path, class_name = import_path.rsplit(".", 1)
        mod = importlib.import_module(module_path)
        getattr(mod, class_name)
        return True
    except Exception:
        return False


class ExecutionPathResolver:
    """
    Resolve execution path for a ModelSuggestion.

    - builtin: model_name is in MODEL_IMPORT_MAP (preferred integration).
    - dynamic_import: not in map but import_path works (e.g. LLM suggested valid path).
    - codegen: no reliable import; generate sklearn-compatible GeneratedEstimator.
    """

    def resolve(self, suggestion: ModelSuggestion) -> PathDecision:
        if suggestion.model_name in MODEL_IMPORT_MAP:
            return PathDecision(
                "builtin",
                "Model found in MODEL_IMPORT_MAP",
            )
        if try_import_class(suggestion.import_path):
            return PathDecision(
                "dynamic_import",
                "import_path verified via importlib",
            )
        return PathDecision(
            "codegen",
            "Unknown model name or import_path failed; use CodeGen to emit a wrapper",
        )


__all__ = [
    "PathKind",
    "PathDecision",
    "ExecutionPathResolver",
    "try_import_class",
]
