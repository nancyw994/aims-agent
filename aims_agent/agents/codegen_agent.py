"""
CodeGen agent: emit a sklearn-compatible estimator (GeneratedEstimator) as a .py module.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from aims_agent.code_writer import extract_python_code, validate_python_syntax, save_generated_code, load_generated_module
from aims_agent.specs import CodeGenSpec, compact_metadata_for_codegen

if TYPE_CHECKING:
    from aims_agent.agent import Agent

GENERATED_ESTIMATOR_CLASS_NAME = "GeneratedEstimator"


def build_model_estimator_prompt(
    *,
    spec: CodeGenSpec,
    dataset_metadata: Mapping[str, Any],
    background_knowledge: str | None,
) -> str:
    meta = compact_metadata_for_codegen(dataset_metadata)
    prompt = f"""You are an expert Python ML engineer.
Generate ONE Python module that defines a scikit-learn-compatible estimator class named `{GENERATED_ESTIMATOR_CLASS_NAME}`.

Code generation spec (JSON):
{json.dumps(spec.to_prompt_dict(), indent=2)}

Dataset metadata (context only):
{json.dumps(meta, indent=2)}

The user intends to use this conceptual model: {spec.model_name!r}.
"""
    if spec.import_path_hint:
        prompt += f"\nHint: try wrapping or delegating to {spec.import_path_hint!r} if it is importable and appropriate.\n"
    if background_knowledge:
        prompt += f"\nBackground / constraints:\n{background_knowledge.strip()}\n"

    prompt += f"""
Return ONLY Python code in one ```python``` block. No explanations.

Hard requirements:
1) Define class `{GENERATED_ESTIMATOR_CLASS_NAME}` with:
   - def fit(self, X, y): ...
   - def predict(self, X): ...
2) X may be pandas.DataFrame or numpy.ndarray; use numpy.asarray where helpful. Do not require specific column names beyond matching n_features in fit/predict.
3) predict(X) returns a 1-dimensional numpy array with length == number of rows in X.
4) Task type is {spec.task_type!r}. Classification: integer/str labels ok if consistent. Regression: float outputs.
5) No file I/O, subprocess, os.system, eval, exec, or network.
6) Prefer numpy + sklearn only.
"""
    return prompt


def generate_model_estimator_module(
    agent: Any,
    *,
    spec: CodeGenSpec,
    dataset_metadata: Mapping[str, Any],
    background_knowledge: str | None = None,
    output_dir: str | Path = "generated_code",
    module_name: str = "generated_estimator",
) -> str:
    """Call LLM, validate syntax, save module; return file path."""
    prompt = build_model_estimator_prompt(
        spec=spec,
        dataset_metadata=dataset_metadata,
        background_knowledge=background_knowledge,
    )
    response = agent.call_llm(prompt)
    code = extract_python_code(response)
    validate_python_syntax(code)
    return save_generated_code(code, output_dir=output_dir, module_name=module_name)


def load_generated_estimator_class(module_path: str | Path) -> type:
    """Load GeneratedEstimator from a saved module path."""
    mod = load_generated_module(str(module_path))
    cls = getattr(mod, GENERATED_ESTIMATOR_CLASS_NAME, None)
    if cls is None:
        raise AttributeError(
            f"Module {module_path!s} must define class {GENERATED_ESTIMATOR_CLASS_NAME!r}"
        )
    return cls


__all__ = [
    "GENERATED_ESTIMATOR_CLASS_NAME",
    "build_model_estimator_prompt",
    "generate_model_estimator_module",
    "load_generated_estimator_class",
]
