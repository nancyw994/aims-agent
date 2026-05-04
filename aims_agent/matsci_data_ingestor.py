"""
Real materials-science data ingestion and preprocessing.

The Materials Project path uses the official ``mp-api`` client when available.
Local CSV/JSON loading is also supported so ingestion configs can be tested and
replayed without a live API key.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from aims_agent.data_interface import (
    DataInterface,
    DatasetBundle,
    DatasetSchema,
    validate_schema,
)


DEFAULT_MP_FIELDS = [
    "material_id",
    "formula_pretty",
    "formation_energy_per_atom",
    "band_gap",
    "energy_above_hull",
    "density",
    "volume",
    "nsites",
    "is_stable",
]

DEFAULT_TARGET_PRIORITY = [
    "formation_energy_per_atom",
    "band_gap",
    "energy_above_hull",
]

NON_FEATURE_COLUMNS = {
    "material_id",
    "formula_pretty",
    "pretty_formula",
    "formula_anonymous",
    "chemsys",
}

MP_UNITS = {
    "formation_energy_per_atom": "eV/atom",
    "band_gap": "eV",
    "energy_above_hull": "eV/atom",
    "density": "g/cm^3",
    "volume": "Angstrom^3",
    "nsites": "count",
}


@dataclass
class PreprocessingReport:
    """Audit trail for real-data preprocessing decisions."""

    target: str
    features: list[str]
    original_shape: tuple[int, int]
    final_shape: tuple[int, int]
    missing_strategy: str
    dropped_rows: int = 0
    imputed_columns: list[str] = field(default_factory=list)
    encoded_columns: list[str] = field(default_factory=list)
    outlier_strategy: str = "none"
    clipped_columns: list[str] = field(default_factory=list)
    scaling: str = "none"
    scaled_columns: list[str] = field(default_factory=list)

    def summary(self) -> str:
        parts = [
            f"target={self.target}",
            f"features={self.features}",
            f"shape={self.original_shape}->{self.final_shape}",
            f"missing={self.missing_strategy}",
            f"outliers={self.outlier_strategy}",
            f"scaling={self.scaling}",
        ]
        if self.dropped_rows:
            parts.append(f"dropped_rows={self.dropped_rows}")
        if self.imputed_columns:
            parts.append(f"imputed={self.imputed_columns}")
        if self.encoded_columns:
            parts.append(f"encoded={self.encoded_columns}")
        if self.clipped_columns:
            parts.append(f"clipped={self.clipped_columns}")
        return " | ".join(parts)


def _as_records_from_json(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "records", "materials", "docs"):
            value = data.get(key)
            if isinstance(value, list):
                return value
        return [data]
    raise ValueError(f"Unsupported JSON payload in {path}: expected object or list")


def load_tabular_materials_file(path: str | Path) -> pd.DataFrame:
    """Load a local CSV or JSON materials dataset into a DataFrame."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".json":
        return pd.DataFrame(_as_records_from_json(path))
    raise ValueError(f"Unsupported materials data file '{path}'. Expected .csv or .json.")


def _scalarize(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "as_dict"):
        value = value.as_dict()
    elif hasattr(value, "model_dump"):
        value = value.model_dump()
    elif hasattr(value, "dict"):
        value = value.dict()
    try:
        return json.dumps(value, default=str, sort_keys=True)
    except TypeError:
        return str(value)


def _doc_to_record(doc: Any, fields: list[str]) -> dict[str, Any]:
    record: dict[str, Any] = {}
    for field_name in fields:
        if isinstance(doc, Mapping):
            value = doc.get(field_name)
        else:
            value = getattr(doc, field_name, None)
        record[field_name] = _scalarize(value)
    return record


def fetch_materials_project_summary(config: Mapping[str, Any]) -> pd.DataFrame:
    """
    Query the Materials Project summary endpoint and return tabular records.

    Expected config keys include ``api_key`` or env ``MP_API_KEY`` /
    ``MATERIALS_PROJECT_API_KEY``, plus optional ``fields``, ``material_ids``,
    ``chemsys``, ``elements``, ``band_gap``, ``formation_energy_per_atom``,
    ``energy_above_hull``, ``is_stable``, and ``limit``.
    """

    if config.get("load_env", True):
        load_dotenv()
    api_key = (
        config.get("api_key")
        or os.getenv("MP_API_KEY")
        or os.getenv("MATERIALS_PROJECT_API_KEY")
    )
    if not api_key:
        raise ValueError(
            "Materials Project API key required. Set MP_API_KEY, "
            "MATERIALS_PROJECT_API_KEY, or config['api_key']."
        )

    try:
        from mp_api.client import MPRester
    except ImportError as exc:
        raise ImportError(
            "Materials Project ingestion requires the optional mp-api package. "
            "Install it with: pip install mp-api"
        ) from exc

    fields = list(config.get("fields") or DEFAULT_MP_FIELDS)
    search_kwargs: dict[str, Any] = {"fields": fields, "all_fields": False}
    for key in (
        "material_ids",
        "chemsys",
        "elements",
        "band_gap",
        "formation_energy_per_atom",
        "energy_above_hull",
        "is_stable",
    ):
        if key in config and config[key] is not None:
            search_kwargs[key] = config[key]

    limit = config.get("limit")
    if limit is not None:
        search_kwargs["chunk_size"] = int(limit)
        search_kwargs["num_chunks"] = 1

    with MPRester(api_key) as mpr:
        docs = mpr.materials.summary.search(**search_kwargs)

    if limit is not None:
        docs = docs[: int(limit)]
    return pd.DataFrame(_doc_to_record(doc, fields) for doc in docs)


def _infer_target(df: pd.DataFrame, explicit_target: str | None = None) -> str:
    if explicit_target:
        if explicit_target not in df.columns:
            raise ValueError(f"Target column '{explicit_target}' not found in dataset.")
        return explicit_target
    for candidate in DEFAULT_TARGET_PRIORITY:
        if candidate in df.columns:
            return candidate
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if numeric_cols:
        return numeric_cols[-1]
    raise ValueError("Could not infer a numeric target column. Pass config['target'].")


def _infer_features(df: pd.DataFrame, target: str, explicit_features: Any = None) -> list[str]:
    if explicit_features:
        missing = [c for c in explicit_features if c not in df.columns]
        if missing:
            raise ValueError(f"Feature columns not found in dataset: {missing}")
        return list(explicit_features)
    return [
        c
        for c in df.columns
        if c != target
        and c not in NON_FEATURE_COLUMNS
        and pd.api.types.is_numeric_dtype(df[c])
    ]


def _coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_bool_dtype(df[col]):
            df[col] = df[col].astype(int)
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        try:
            df[col] = pd.to_numeric(df[col])
        except (TypeError, ValueError):
            pass
    return df


def preprocessing_policy_from_text(text: str | None) -> dict[str, str]:
    """
    Convert a plain-language LLM suggestion into conservative preprocessing knobs.

    This intentionally recognizes only simple keywords; free-form LLM guidance is
    kept advisory unless it maps cleanly to supported deterministic operations.
    """

    policy = {
        "missing_strategy": "impute",
        "outlier_strategy": "iqr_clip",
        "scaling": "none",
    }
    if not text:
        return policy
    lower = text.lower()
    if "drop" in lower and "missing" in lower:
        policy["missing_strategy"] = "drop"
    if "no outlier" in lower or "do not clip" in lower:
        policy["outlier_strategy"] = "none"
    elif "remove outlier" in lower or "drop outlier" in lower:
        policy["outlier_strategy"] = "iqr_drop"
    if "standard" in lower or "z-score" in lower or "zscore" in lower:
        policy["scaling"] = "standard"
    elif "minmax" in lower or "min-max" in lower or "normalize" in lower:
        policy["scaling"] = "minmax"
    return policy


def clean_and_preprocess_materials_data(
    df: pd.DataFrame,
    *,
    target: str | None = None,
    features: list[str] | None = None,
    missing_strategy: str = "impute",
    outlier_strategy: str = "iqr_clip",
    scaling: str = "none",
    encode_categoricals: bool = True,
) -> tuple[pd.DataFrame, list[str], str, PreprocessingReport]:
    """Clean a materials DataFrame and return ML-ready data plus an audit report."""

    from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

    original_shape = df.shape
    df = _coerce_numeric_columns(df)
    target = _infer_target(df, target)

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset.")

    before_target = len(df)
    df = df[df[target].notna()].reset_index(drop=True)
    dropped_rows = before_target - len(df)

    if features is None:
        features = _infer_features(df, target)
    else:
        features = _infer_features(df, target, features)

    if not features:
        raise ValueError("No usable feature columns found after preprocessing.")

    encoded_columns: list[str] = []
    if encode_categoricals:
        for col in list(features):
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].fillna("missing").astype(str))
                encoded_columns.append(col)

    selected_cols = [*features, target]
    imputed_columns: list[str] = []
    if missing_strategy == "drop":
        before = len(df)
        df = df.dropna(subset=selected_cols).reset_index(drop=True)
        dropped_rows += before - len(df)
    elif missing_strategy == "impute":
        for col in features:
            if df[col].isna().any():
                if pd.api.types.is_numeric_dtype(df[col]):
                    fill_value = df[col].median()
                else:
                    mode = df[col].mode(dropna=True)
                    fill_value = mode.iloc[0] if not mode.empty else "missing"
                df[col] = df[col].fillna(fill_value)
                imputed_columns.append(col)
    else:
        raise ValueError("missing_strategy must be 'drop' or 'impute'.")

    clipped_columns: list[str] = []
    numeric_features = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    if outlier_strategy in {"iqr_clip", "iqr_drop"}:
        for col in numeric_features:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            if not np.isfinite(iqr) or iqr == 0:
                continue
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            mask = (df[col] < lower) | (df[col] > upper)
            if not mask.any():
                continue
            clipped_columns.append(col)
            if outlier_strategy == "iqr_clip":
                df[col] = df[col].clip(lower=lower, upper=upper)
            else:
                before = len(df)
                df = df.loc[~mask].reset_index(drop=True)
                dropped_rows += before - len(df)
    elif outlier_strategy != "none":
        raise ValueError("outlier_strategy must be 'none', 'iqr_clip', or 'iqr_drop'.")

    scaled_columns: list[str] = []
    if scaling != "none":
        scaler_cls = {"standard": StandardScaler, "minmax": MinMaxScaler}.get(scaling)
        if scaler_cls is None:
            raise ValueError("scaling must be 'none', 'standard', or 'minmax'.")
        if numeric_features:
            scaler = scaler_cls()
            df[numeric_features] = scaler.fit_transform(df[numeric_features])
            scaled_columns = numeric_features.copy()

    report = PreprocessingReport(
        target=target,
        features=features,
        original_shape=original_shape,
        final_shape=df.shape,
        missing_strategy=missing_strategy,
        dropped_rows=dropped_rows,
        imputed_columns=imputed_columns,
        encoded_columns=encoded_columns,
        outlier_strategy=outlier_strategy,
        clipped_columns=clipped_columns,
        scaling=scaling,
        scaled_columns=scaled_columns,
    )
    return df, features, target, report


class MaterialsProjectDataIngestor(DataInterface):
    """
    DataInterface implementation for Materials Project summary data.

    Config keys:
      source_type: "materials_project", "csv", or "json" (default: infer from path)
      path: optional local CSV/JSON export to preprocess instead of querying API
      api_key: optional Materials Project API key (or use MP_API_KEY env var)
      fields/query filters: passed to Materials Project summary search
      target/features: supervised-learning target and feature columns
      preprocessing_suggestion: optional LLM text mapped to deterministic knobs
      missing_strategy/outlier_strategy/scaling: explicit preprocessing controls
      output_path: optional CSV path for the preprocessed dataset
    """

    def load_dataset(self, config: Mapping[str, Any]) -> DatasetBundle:
        policy = preprocessing_policy_from_text(config.get("preprocessing_suggestion"))
        missing_strategy = config.get("missing_strategy", policy["missing_strategy"])
        outlier_strategy = config.get("outlier_strategy", policy["outlier_strategy"])
        scaling = config.get("scaling", policy["scaling"])

        path = config.get("path")
        if path:
            df = load_tabular_materials_file(path)
            source = config.get("source", str(path))
        else:
            df = fetch_materials_project_summary(config)
            source = config.get("source", "Materials Project summary API")

        processed, features, target, report = clean_and_preprocess_materials_data(
            df,
            target=config.get("target"),
            features=config.get("features"),
            missing_strategy=missing_strategy,
            outlier_strategy=outlier_strategy,
            scaling=scaling,
            encode_categoricals=config.get("encode_categoricals", True),
        )

        output_path = config.get("output_path")
        if output_path:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            processed.to_csv(out, index=False)

        units = {col: MP_UNITS.get(col, "") for col in [*features, target]}
        schema = DatasetSchema(
            features=features,
            target=target,
            units=units,
            source=source,
            description=config.get(
                "description",
                f"Real materials data ingested from {source}. Preprocessing: {report.summary()}",
            ),
            shape=processed.shape,
            dtypes={c: str(dt) for c, dt in processed.dtypes.items()},
        )
        validate_schema(processed, schema)
        return DatasetBundle(df=processed, schema=schema)


__all__ = [
    "DEFAULT_MP_FIELDS",
    "MaterialsProjectDataIngestor",
    "PreprocessingReport",
    "clean_and_preprocess_materials_data",
    "fetch_materials_project_summary",
    "load_tabular_materials_file",
    "preprocessing_policy_from_text",
]
