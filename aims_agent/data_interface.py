from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Any, Tuple

import pandas as pd


@dataclass
class DatasetSchema:
    """Schema and metadata for a tabular materials dataset."""

    # Required by spec
    features: List[str]
    target: str
    units: Dict[str, str]
    source: str
    description: str

    shape: Tuple[int, int] | None = None
    dtypes: Dict[str, str] | None = None


@dataclass
class DatasetBundle:
    """Container for a dataset and its schema."""

    df: pd.DataFrame
    schema: DatasetSchema


class DataInterface:
    """
    Abstract data access layer.

    Downstream components should depend only on this interface and on
    DatasetBundle / DatasetSchema, never on concrete loaders or file paths.
    """

    def load_dataset(self, config: Mapping[str, Any]) -> DatasetBundle: 
        raise NotImplementedError


def validate_schema(df: pd.DataFrame, schema: DatasetSchema) -> None:
    """
    Validate that `df` conforms to `schema`.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas.DataFrame")

    missing_feature_cols = [c for c in schema.features if c not in df.columns]
    if missing_feature_cols:
        raise ValueError(f"Missing feature columns in DataFrame: {missing_feature_cols}")

    if schema.target not in df.columns:
        raise ValueError(f"Target column '{schema.target}' not found in DataFrame")

    # Units check is skipped when units dict is empty (real datasets often lack units)
    if schema.units:
        required_cols = list(schema.features) + [schema.target]
        missing_units = [c for c in required_cols if c not in schema.units]
        if missing_units:
            raise ValueError(f"Missing units for columns: {missing_units}")

    if not schema.source:
        raise ValueError("schema.source must be a non-empty string")

    if not schema.description:
        raise ValueError("schema.description must be a non-empty string")


def get_metadata(bundle: DatasetBundle) -> Dict[str, Any]:
    """
    Return rich metadata suitable for LLM consumption.
    """

    df = bundle.df
    schema = bundle.schema

    shape = schema.shape or df.shape
    dtypes = schema.dtypes or {c: str(dt) for c, dt in df.dtypes.items()}

    return {
        "features": schema.features,
        "target": schema.target,
        "units": schema.units,
        "source": schema.source,
        "description": schema.description,
        "shape": {"rows": shape[0], "cols": shape[1]},
        "dtypes": dtypes,
    }


__all__ = [
    "DatasetSchema",
    "DatasetBundle",
    "DataInterface",
    "validate_schema",
    "get_metadata",
]

