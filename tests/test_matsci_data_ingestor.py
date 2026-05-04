from __future__ import annotations

import json

import pandas as pd
import pytest

from aims_agent.matsci_data_ingestor import (
    MaterialsProjectDataIngestor,
    clean_and_preprocess_materials_data,
    preprocessing_policy_from_text,
)


def test_preprocessing_policy_from_text_maps_supported_keywords():
    policy = preprocessing_policy_from_text(
        "Drop missing rows, remove outliers, and use standard scaling."
    )
    assert policy["missing_strategy"] == "drop"
    assert policy["outlier_strategy"] == "iqr_drop"
    assert policy["scaling"] == "standard"


def test_clean_and_preprocess_materials_data_imputes_clips_and_scales():
    df = pd.DataFrame(
        {
            "material_id": ["a", "b", "c", "d", "e"],
            "formation_energy_per_atom": [-1.0, -1.2, -0.9, -1.1, -1.05],
            "band_gap": [1.1, None, 1.3, 40.0, 1.2],
            "density": [4.0, 4.2, 4.1, 4.3, 4.2],
            "is_stable": [True, False, True, False, True],
        }
    )

    processed, features, target, report = clean_and_preprocess_materials_data(
        df,
        target="formation_energy_per_atom",
        missing_strategy="impute",
        outlier_strategy="iqr_clip",
        scaling="standard",
    )

    assert target == "formation_energy_per_atom"
    assert "material_id" not in features
    assert "band_gap" in features
    assert processed[features].isna().sum().sum() == 0
    assert report.imputed_columns == ["band_gap"]
    assert "band_gap" in report.clipped_columns
    assert set(report.scaled_columns) == set(features)


def test_materials_project_ingestor_loads_local_json_and_writes_output(tmp_path):
    source = tmp_path / "mp_export.json"
    output = tmp_path / "preprocessed.csv"
    source.write_text(
        json.dumps(
            [
                {
                    "material_id": "mp-1",
                    "formula_pretty": "AB",
                    "formation_energy_per_atom": -1.0,
                    "band_gap": 1.1,
                    "energy_above_hull": 0.0,
                    "density": 4.0,
                    "volume": 10.0,
                    "nsites": 2,
                },
                {
                    "material_id": "mp-2",
                    "formula_pretty": "AC",
                    "formation_energy_per_atom": -1.4,
                    "band_gap": None,
                    "energy_above_hull": 0.05,
                    "density": 4.4,
                    "volume": 12.0,
                    "nsites": 3,
                },
            ]
        ),
        encoding="utf-8",
    )

    bundle = MaterialsProjectDataIngestor().load_dataset(
        {
            "path": str(source),
            "target": "formation_energy_per_atom",
            "missing_strategy": "impute",
            "outlier_strategy": "none",
            "scaling": "none",
            "output_path": str(output),
        }
    )

    assert bundle.schema.target == "formation_energy_per_atom"
    assert "band_gap" in bundle.schema.features
    assert output.exists()
    assert len(bundle.df) == 2


def test_materials_project_ingestor_requires_api_key_for_live_query(monkeypatch):
    monkeypatch.delenv("MP_API_KEY", raising=False)
    monkeypatch.delenv("MATERIALS_PROJECT_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key required"):
        MaterialsProjectDataIngestor().load_dataset({"limit": 1, "load_env": False})
