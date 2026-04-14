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
    hint = (
        f"Hint: try wrapping or delegating to {spec.import_path_hint!r} if it is importable and appropriate."
        if spec.import_path_hint
        else "Hint: implement directly with robust sklearn-compatible behavior."
    )
    prompt = f"""SYSTEM:
You are a senior ML engineer generating robust model wrapper code.

TASK:
Generate one complete, executable Python module for the selected model.

INPUT:
Spec:
{json.dumps(spec.to_prompt_dict(), indent=2)}

Dataset metadata (context only):
{json.dumps(meta, indent=2)}

Intent:
The user selected conceptual model {spec.model_name!r}.
{hint}
"""
    if background_knowledge:
        prompt += f"\nAdditional constraints:\n{background_knowledge.strip()}\n"

    prompt += f"""
CONSTRAINTS:
- Must define class `{GENERATED_ESTIMATOR_CLASS_NAME}`
- Must implement fit(self, X, y) and predict(self, X)
- predict must return a 1D numpy array with length == number of rows in X
- Handle both pandas.DataFrame and numpy.ndarray inputs
- fit must return self
- No file I/O, subprocess, os.system, eval, exec, or network
- Prefer numpy + sklearn only
- Keep deterministic defaults (set random_state=42 when applicable)
- Do not rely on hardcoded feature names

SELF-CHECK BEFORE OUTPUT:
- module is syntactically valid Python
- class name is exactly `{GENERATED_ESTIMATOR_CLASS_NAME}`
- methods `fit` and `predict` exist with required signatures
- output shape requirement is satisfied

OUTPUT FORMAT:
Return ONLY Python code in one ```python``` block.
No explanations outside the code block.
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
