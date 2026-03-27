from __future__ import annotations

import ast
import importlib.util
import re
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping


def build_code_generation_prompt(
    *,
    request: str,
    dataset_metadata: Mapping[str, Any],
    background_knowledge: str | None = None,
    task_type: str = "regression",
) -> str:
    """
    Build a strict prompt for generating an executable custom component.
    """
    prompt = f"""You are an expert Python ML engineer.
Generate ONE Python module implementing a custom ML component.

Task request:
{request}

Task type: {task_type}
Dataset metadata:
{dict(dataset_metadata)}
"""
    if background_knowledge:
        prompt += f"\nBackground knowledge / constraints:\n{background_knowledge.strip()}\n"

    prompt += """
Return ONLY Python code in one ```python``` block. No explanations.

Hard requirements:
1) The module MUST define:
   def run_component(df, features, target, task_type="regression"):
       ...
       return {"features": features, "note": "short summary"}
2) Do not read/write files.
3) Do not use subprocess, os.system, eval, exec, network calls, or shell commands.
4) Keep dependencies standard: pandas, numpy, sklearn only (if needed).
5) The function must be robust and not crash on small datasets.
"""
    return prompt


def extract_python_code(text: str) -> str:
    """
    Extract Python code from markdown code blocks; fallback to raw text.
    """
    pattern = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
    matches = pattern.findall(text)
    if matches:
        return matches[0].strip()
    return text.strip()


def validate_python_syntax(code: str) -> None:
    """
    Raise SyntaxError if code is not valid Python.
    """
    ast.parse(code)


def save_generated_code(
    code: str,
    *,
    output_dir: str | Path = "generated_code",
    module_name: str = "custom_component",
) -> str:
    """
    Save generated code to output_dir/module_name.py and return path.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    file_path = output_path / f"{module_name}.py"
    file_path.write_text(code, encoding="utf-8")
    return str(file_path)


def generate_code_file(
    agent: Any,
    *,
    request: str,
    dataset_metadata: Mapping[str, Any],
    background_knowledge: str | None = None,
    task_type: str = "regression",
    output_dir: str | Path = "generated_code",
    module_name: str = "custom_component",
) -> str:
    """
    Ask LLM to generate code, validate syntax, save as .py, and return file path.
    """
    prompt = build_code_generation_prompt(
        request=request,
        dataset_metadata=dataset_metadata,
        background_knowledge=background_knowledge,
        task_type=task_type,
    )
    response = agent.call_llm(prompt)
    code = extract_python_code(response)
    validate_python_syntax(code)
    return save_generated_code(code, output_dir=output_dir, module_name=module_name)


def load_generated_module(module_path: str | Path) -> ModuleType:
    """
    Load a generated Python file as a module.
    """
    path = Path(module_path)
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def execute_generated_component(
    module_path: str | Path,
    *,
    df: Any,
    features: list[str],
    target: str,
    task_type: str = "regression",
) -> dict[str, Any]:
    """
    Execute run_component(...) from generated module.
    """
    module = load_generated_module(module_path)
    fn = getattr(module, "run_component", None)
    if fn is None:
        raise AttributeError("Generated module does not define run_component(df, features, target, task_type)")
    result = fn(df=df, features=features, target=target, task_type=task_type)
    if result is None:
        return {}
    if not isinstance(result, dict):
        raise TypeError("run_component must return a dict")
    return result


__all__ = [
    "build_code_generation_prompt",
    "extract_python_code",
    "validate_python_syntax",
    "save_generated_code",
    "generate_code_file",
    "load_generated_module",
    "execute_generated_component",
]
