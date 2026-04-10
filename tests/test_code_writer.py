"""Tests for aims_agent.code_writer (no LLM)."""

from __future__ import annotations

import pytest

from aims_agent.code_writer import (
    codegen_required_packages,
    dl_backend_candidates,
    dl_backend_required_packages,
    extract_python_code,
    infer_codegen_mode,
    infer_preferred_dl_backend,
    load_generated_module,
    save_generated_code,
    validate_component_output,
    validate_python_syntax,
    build_code_generation_prompt,
)


def test_extract_python_code_from_markdown():
    text = """Here is code:
```python
x = 1 + 2
```
"""
    assert "x = 1 + 2" in extract_python_code(text)


def test_extract_python_code_plain():
    assert extract_python_code("a = 42").strip() == "a = 42"


def test_validate_python_syntax_ok():
    validate_python_syntax("def f():\n    return 1\n")


def test_validate_python_syntax_bad():
    with pytest.raises(SyntaxError):
        validate_python_syntax("def f((: invalid")


def test_save_and_load_module(tmp_path):
    code = """
def hello():
    return "hi"
"""
    path = save_generated_code(code, output_dir=tmp_path, module_name="tmod")
    mod = load_generated_module(path)
    assert mod.hello() == "hi"


def test_infer_codegen_mode():
    assert infer_codegen_mode("build pytorch two-layer neural network") == "deep_learning"
    assert infer_codegen_mode("custom preprocessing only") == "standard"


def test_codegen_required_packages():
    assert codegen_required_packages("standard") == []
    assert "torch" in codegen_required_packages("deep_learning")


def test_infer_preferred_dl_backend():
    assert infer_preferred_dl_backend("build tensorflow model") == "tensorflow"
    assert infer_preferred_dl_backend("build pytorch model") == "torch"


def test_dl_backend_candidates():
    assert dl_backend_candidates("torch") == ["torch", "tensorflow"]
    assert dl_backend_candidates("tensorflow") == ["tensorflow", "torch"]


def test_dl_backend_required_packages():
    assert dl_backend_required_packages("torch") == ["torch"]
    assert dl_backend_required_packages("tensorflow") == ["tensorflow-cpu"]


def test_build_prompt_deep_learning_contains_constraints():
    p = build_code_generation_prompt(
        request="Implement torch regressor",
        dataset_metadata={"features": ["a"], "target": "y"},
        task_type="regression",
        codegen_mode="deep_learning",
    )
    assert "Deep-learning mode" in p
    assert "torch" in p.lower()


def test_validate_component_output_predictions_ok():
    ok, msg = validate_component_output(
        {"features": ["a"], "predictions": [1.0, 2.0, 3.0]},
        original_features=["a"],
        row_count=3,
    )
    assert ok
    assert msg == "ok"


def test_validate_component_output_predictions_bad_shape():
    ok, msg = validate_component_output(
        {"predictions": [1.0, 2.0]},
        original_features=["a"],
        row_count=3,
    )
    assert not ok
    assert "mismatch" in msg
