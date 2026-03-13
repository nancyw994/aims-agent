"""
Tabular data loader for CSV and Excel files (.csv, .xls, .xlsx).

Supports both synthetic-compatible CSV files and real-world materials datasets.
Key capabilities:
- Auto-detects file format by extension (CSV, XLS, XLSX).
- Auto-infers features (all columns except target) when not specified.
- Auto-encodes categorical (object/string) columns via LabelEncoder.
- Units are optional; set to {} for real datasets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Any

import numpy as np
import pandas as pd

from aims_agent.data_interface import (
    DataInterface,
    DatasetBundle,
    DatasetSchema,
    validate_schema,
)

_SYNTHETIC_FEATURES = [
    "Al_conc_wt_pct",
    "Ti_conc_wt_pct",
    "V_conc_wt_pct",
    "process_temperature_C",
    "process_time_h",
]
_SYNTHETIC_TARGET = "hardness_HV"
_SYNTHETIC_UNITS: Dict[str, str] = {
    "Al_conc_wt_pct": "wt.%",
    "Ti_conc_wt_pct": "wt.%",
    "V_conc_wt_pct": "wt.%",
    "process_temperature_C": "°C",
    "process_time_h": "h",
    "hardness_HV": "HV",
}


def _read_file(path: Path, sheet_name: int | str = 0) -> pd.DataFrame:
    """Read CSV or Excel file; raise on unsupported extension."""
    suffix = path.suffix.lower()
    if suffix in {".xls", ".xlsx"}:
        try:
            return pd.read_excel(path, sheet_name=sheet_name)
        except ImportError as e:
            raise ImportError(
                "Reading Excel files requires openpyxl. "
                "Install it with: pip install openpyxl"
            ) from e
    elif suffix == ".csv":
        return pd.read_csv(path)
    else:
        try:
            return pd.read_csv(path)
        except Exception:
            raise ValueError(
                f"Unsupported file extension '{suffix}'. "
                "Expected .csv, .xls, or .xlsx."
            )


def _encode_categoricals(df: pd.DataFrame, feature_cols: List[str]) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Label-encode any object/string/category columns in feature_cols.

    Returns the modified DataFrame and a dict mapping column name ->
    {type: "label_encoded", classes: [...]}.
    """
    from sklearn.preprocessing import LabelEncoder

    df = df.copy()
    encoding_info: Dict[str, Any] = {}
    for col in feature_cols:
        if col in df.columns and df[col].dtype == object or (
            col in df.columns and hasattr(df[col], "dtype") and pd.api.types.is_string_dtype(df[col])
        ):
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoding_info[col] = {
                "type": "label_encoded",
                "classes": list(le.classes_),
            }
    return df, encoding_info


def _infer_features_and_target(
    df: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[List[str], str]:
    """
    Determine feature columns and target column.

    Priority:
    1. Explicit config['features'] and config['target'].
    2. If only config['target'] given: features = all other columns.
    3. If neither given and columns match synthetic schema: use synthetic defaults.
    4. Otherwise: target = last column, features = all others.
    """
    all_cols = list(df.columns)
    target = config.get("target")
    features = config.get("features")

    if features and target:
        return list(features), target

    if target and not features:
        features = [c for c in all_cols if c != target]
        return features, target

    # No target given — try synthetic defaults, then fall back to last column
    if _SYNTHETIC_TARGET in all_cols and all(c in all_cols for c in _SYNTHETIC_FEATURES):
        return _SYNTHETIC_FEATURES, _SYNTHETIC_TARGET

    target = all_cols[-1]
    features = all_cols[:-1]
    return features, target


class CSVDataLoader(DataInterface):
    """
    Load a tabular dataset from a CSV or Excel file.

    Config keys:
      path (str)         : Path to the file (required). Supports .csv, .xls, .xlsx.
      target (str)       : Name of the target/label column. Auto-detected if not set.
      features (list)    : List of feature column names. Auto-inferred if not set.
      sheet_name (int|str): Excel sheet (default 0). Ignored for CSV.
      units (dict)       : Optional column→unit mapping. Defaults to {} for real datasets.
      source (str)       : Optional data source label.
      description (str)  : Optional free-text description.
      drop_na (bool)     : Drop rows with NaN values (default True).
      encode_categoricals (bool): Label-encode string/object feature columns (default True).
    """

    def load_dataset(self, config: Mapping[str, Any]) -> DatasetBundle:
        path_str = config.get("path")
        if not path_str:
            raise ValueError("config['path'] is required for CSVDataLoader")
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        # ── Read file ──────────────────────────────────────────────────
        sheet_name = config.get("sheet_name", 0)
        df = _read_file(path, sheet_name=sheet_name)

        # ── Drop rows with missing values ──────────────────────────────
        if config.get("drop_na", True):
            before = len(df)
            df = df.dropna().reset_index(drop=True)
            dropped = before - len(df)
            if dropped > 0:
                print(f"[DataLoader] Dropped {dropped} rows with NaN values ({before} → {len(df)})")

        # ── Infer features / target ────────────────────────────────────
        features, target = _infer_features_and_target(df, config)

        # ── Encode categorical feature columns ─────────────────────────
        encoding_info: Dict[str, Any] = {}
        if config.get("encode_categoricals", True):
            df, encoding_info = _encode_categoricals(df, features)
            if encoding_info:
                print(f"[DataLoader] Label-encoded categorical columns: {list(encoding_info.keys())}")
                for col, info in encoding_info.items():
                    print(f"  {col}: {info['classes']}")

        # ── Build schema ───────────────────────────────────────────────
        units = config.get("units", {})

        # Use synthetic units if dataset matches synthetic schema; else empty
        if not units:
            if target == _SYNTHETIC_TARGET and features == _SYNTHETIC_FEATURES:
                units = _SYNTHETIC_UNITS

        source = config.get("source", path.name)
        description = config.get(
            "description",
            f"Dataset loaded from '{path.name}' | features: {features} | target: '{target}'",
        )

        schema = DatasetSchema(
            features=features,
            target=target,
            units=units,
            source=source,
            description=description,
            shape=df.shape,
            dtypes={c: str(dt) for c, dt in df.dtypes.items()},
        )
        validate_schema(df, schema)
        return DatasetBundle(df=df, schema=schema)


def inspect_file(path: str | Path, sheet_name: int | str = 0) -> None:
    """
    Print a quick summary of a CSV or Excel file: shape, columns, dtypes, and head.
    Useful for exploring an unknown dataset before running the pipeline.
    """
    path = Path(path)
    df = _read_file(path, sheet_name=sheet_name)
    print(f"File : {path}")
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print("\nColumns & dtypes:")
    for col, dtype in df.dtypes.items():
        n_unique = df[col].nunique()
        null_pct = df[col].isna().mean() * 100
        print(f"  {col:<35} {str(dtype):<12} {n_unique:>6} unique  {null_pct:.1f}% null")
    print("\nSample (first 3 rows):")
    print(df.head(3).to_string())


__all__ = ["CSVDataLoader", "inspect_file"]
