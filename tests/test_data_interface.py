"""Tests for aims_agent.data_interface."""

from __future__ import annotations

import pandas as pd
import pytest

from aims_agent.data_interface import DatasetBundle, DatasetSchema, get_metadata, validate_schema


def test_get_metadata():
    df = pd.DataFrame({"x": [1, 2], "y": [0.1, 0.2]})
    schema = DatasetSchema(
        features=["x"],
        target="y",
        units={},
        source="test",
        description="unit test",
    )
    bundle = DatasetBundle(df=df, schema=schema)
    meta = get_metadata(bundle)
    assert meta["features"] == ["x"]
    assert meta["target"] == "y"
    assert meta["shape"]["rows"] == 2


def test_validate_schema_missing_column():
    df = pd.DataFrame({"x": [1]})
    schema = DatasetSchema(
        features=["x", "missing"],
        target="y",
        units={},
        source="s",
        description="d",
    )
    with pytest.raises(ValueError, match="Missing feature"):
        validate_schema(df, schema)
