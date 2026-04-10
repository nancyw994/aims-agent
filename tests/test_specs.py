"""Tests for aims_agent.specs."""

from __future__ import annotations

from aims_agent.model_selector import ModelSuggestion
from aims_agent.path_resolver import PathDecision
from aims_agent.specs import (
    CodeGenSpec,
    build_model_codegen_spec,
    compact_metadata_for_codegen,
)


def test_build_model_codegen_spec_fields():
    s = ModelSuggestion(
        model_name="X",
        package_name="pkg",
        import_path="mod.Cls",
        reason="r",
    )
    dec = PathDecision("codegen", "because")
    spec = build_model_codegen_spec(s, "regression", dec)
    assert spec.model_name == "X"
    assert spec.task_type == "regression"
    assert spec.required_interface == "fit_predict"
    assert spec.import_path_hint == "mod.Cls"
    assert spec.reason == "because"
    assert "GeneratedEstimator" in " ".join(spec.constraints)
    assert any("regression" in c.lower() or "floating" in c.lower() for c in spec.constraints)


def test_build_model_codegen_spec_classification_constraint():
    s = ModelSuggestion("C", "sklearn", "sklearn.svm.SVC", "r")
    spec = build_model_codegen_spec(s, "classification", PathDecision("codegen", "x"))
    assert any("classification" in c for c in spec.constraints)


def test_compact_metadata_for_codegen_truncates_description():
    meta = {
        "features": ["a", "b"],
        "target": "y",
        "shape": {"rows": 10, "cols": 3},
        "dtypes": {"a": "float64"},
        "description": "x" * 3000,
    }
    out = compact_metadata_for_codegen(meta)
    assert len(out["description"]) <= 2000
    assert out["features"] == ["a", "b"]
    assert out["target"] == "y"


def test_codegen_spec_to_prompt_dict():
    spec = CodeGenSpec(
        model_name="m",
        task_type="regression",
        required_interface="fit_predict",
        import_path_hint=None,
        package_name=None,
        constraints=["c1"],
        reason="r",
    )
    d = spec.to_prompt_dict()
    assert d["model_name"] == "m"
    assert d["constraints"] == ["c1"]
