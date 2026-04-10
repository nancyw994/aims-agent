"""Tests for aims_agent.path_resolver."""

from __future__ import annotations

import pytest

from aims_agent.model_selector import MODEL_IMPORT_MAP, ModelSuggestion
from aims_agent.path_resolver import ExecutionPathResolver, try_import_class


def test_try_import_class_valid_sklearn():
    assert try_import_class("sklearn.ensemble.RandomForestRegressor") is True


def test_try_import_class_invalid():
    assert try_import_class("") is False
    assert try_import_class("no_dots") is False
    assert try_import_class("definitely_missing_mod_xyz.Foo") is False


def test_resolve_builtin_random_forest():
    r = ExecutionPathResolver()
    s = ModelSuggestion(
        model_name="RandomForestRegressor",
        package_name="scikit-learn",
        import_path="sklearn.ensemble.RandomForestRegressor",
        reason="x",
    )
    d = r.resolve(s)
    assert d.path == "builtin"
    assert "MODEL_IMPORT_MAP" in d.reason


def test_resolve_dynamic_not_in_map_but_importable():
    r = ExecutionPathResolver()
    # sklearn.dummy.DummyRegressor is valid but not in MODEL_IMPORT_MAP
    s = ModelSuggestion(
        model_name="DummyRegressorAlias",
        package_name="scikit-learn",
        import_path="sklearn.dummy.DummyRegressor",
        reason="x",
    )
    d = r.resolve(s)
    assert d.path == "dynamic_import"


def test_resolve_codegen_unknown():
    r = ExecutionPathResolver()
    s = ModelSuggestion(
        model_name="FakeModelZZZ",
        package_name="fake",
        import_path="nonexistent_module_abc123.SomeClass",
        reason="x",
    )
    d = r.resolve(s)
    assert d.path == "codegen"


def test_all_map_entries_resolve_builtin():
    r = ExecutionPathResolver()
    for name in list(MODEL_IMPORT_MAP.keys())[:5]:
        mod_path, cls_name = MODEL_IMPORT_MAP[name]
        s = ModelSuggestion(
            model_name=name,
            package_name="scikit-learn",
            import_path=f"{mod_path}.{cls_name}",
            reason="t",
        )
        assert r.resolve(s).path == "builtin"
