from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from aims_agent.data_analyzer import (
    analyze_and_formulate_strategy,
    build_strategy_prompt,
    formulate_strategy,
    profile_dataset,
)
from aims_agent.data_interface import DatasetBundle, DatasetSchema


def _bundle() -> DatasetBundle:
    rng = np.random.default_rng(7)
    x1 = np.linspace(0, 10, 40)
    x2 = x1 * 1.01
    x3 = rng.normal(size=40)
    y = 2.0 * x1 + rng.normal(scale=0.2, size=40)
    df = pd.DataFrame(
        {
            "band_gap": x1,
            "density": x2,
            "volume": x3,
            "formation_energy_per_atom": y,
        }
    )
    schema = DatasetSchema(
        features=["band_gap", "density", "volume"],
        target="formation_energy_per_atom",
        units={},
        source="unit test",
        description="synthetic MatSci profile test",
    )
    return DatasetBundle(df=df, schema=schema)


def test_profile_dataset_writes_plots_and_detects_correlations(tmp_path):
    profile = profile_dataset(_bundle(), output_dir=tmp_path)
    assert profile.row_count == 40
    assert "band_gap" in profile.correlations["target_correlations"]
    assert profile.plot_paths
    assert all((tmp_path / path.split("/")[-1]).exists() for path in profile.plot_paths)
    assert any("highly correlated" in risk for risk in profile.risks)
    assert "MatSci Data Profile" in profile.summary_text


def test_build_strategy_prompt_contains_required_json_keys(tmp_path):
    profile = profile_dataset(_bundle(), output_dir=tmp_path)
    prompt = build_strategy_prompt(profile)
    assert "recommended_models" in prompt
    assert "validation_plan" in prompt
    assert "band_gap" in prompt


def test_formulate_strategy_parses_llm_json(tmp_path):
    profile = profile_dataset(_bundle(), output_dir=tmp_path)

    class FakeAgent:
        def call_llm(self, _: str) -> str:
            return json.dumps(
                {
                    "key_features": ["band_gap: strongest relationship"],
                    "risks": ["density is redundant with band_gap"],
                    "preprocessing": ["standardize numeric descriptors"],
                    "recommended_models": ["RandomForestRegressor", "Ridge"],
                    "validation_plan": ["5-fold CV with RMSE and MAE"],
                    "scientific_rationale": "Formation energy depends on electronic and structural descriptors.",
                }
            )

    strategy = formulate_strategy(profile, agent=FakeAgent(), use_llm=True, output_dir=tmp_path)
    assert strategy.recommended_models[0] == "Ridge"
    assert len(strategy.recommended_models) == 5
    assert "ElasticNet" in strategy.recommended_models
    assert "RandomForestRegressor" in strategy.recommended_models
    assert "GradientBoostingRegressor" in strategy.recommended_models
    assert "Formation energy" in strategy.llm_interpretation


def test_analyze_and_formulate_strategy_writes_outputs(tmp_path):
    _, strategy, paths = analyze_and_formulate_strategy(
        _bundle(),
        use_llm=False,
        output_dir=tmp_path,
        run_context={
            "api": "Materials Project summary API",
            "dataset": "Li-Fe-O",
            "source": "Materials Project API",
            "mode": "live API ingestion",
            "task_type": "regression",
            "target": "formation_energy_per_atom",
            "llm": "OpenAI / gpt-4o-mini",
            "model_mode": "LLM-guided",
            "preprocessing": {"missing": "drop", "outlier": "iqr_clip", "scaling": "standard"},
        },
    )
    assert strategy.target == "formation_energy_per_atom"
    run_dir = tmp_path / Path(paths["run_dir"]).name
    assert run_dir.exists()
    assert (run_dir / "profile.json").exists()
    assert (run_dir / "strategy.json").exists()
    assert (run_dir / "strategy_report.html").exists()
    report_text = (run_dir / "strategy_report.html").read_text(encoding="utf-8")
    assert "<h2>Dataset Summary</h2>" in report_text
    assert "<h2>User Inputs</h2>" in report_text
    assert "<h2>Data Distribution</h2>" in report_text
    assert "five model families" in report_text
    assert "Fit to this dataset" in report_text
    assert '<figure><img src="data_distribution.png"' in report_text
    assert '<figure><img src="histograms.png"' in report_text
    assert '<figure><img src="correlation_heatmap.png"' in report_text
    assert '<figure><img src="target_relationships.png"' in report_text
    assert set(paths) == {"run_dir", "profile", "strategy", "report"}
