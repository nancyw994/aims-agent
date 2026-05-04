from __future__ import annotations

import json

import numpy as np
import pandas as pd
from pymatgen.core import Lattice, Structure

from aims_agent.pretrained_model_handler import (
    MatGLPretrainedModelHandler,
    benchmark_pretrained_model,
    identify_pretrained_models,
)


class FakeModel:
    def predict_structure(self, structure, **_):
        return float(len(structure)) * -0.5


def _structure_json(n: int = 2) -> str:
    if n == 1:
        struct = Structure(Lattice.cubic(3.0), ["Li"], [[0, 0, 0]])
    else:
        struct = Structure(Lattice.cubic(4.0), ["Li", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    return json.dumps(struct.as_dict())


def test_identify_pretrained_models_heuristic_matches_target():
    choices = identify_pretrained_models(
        target="formation_energy_per_atom",
        available_models=["MEGNet-MP-2019.4.1-BandGap-mfi", "M3GNet-Eform-MP-2018.6.1"],
        use_llm=False,
    )
    assert choices[0].model_name == "M3GNet-Eform-MP-2018.6.1"
    assert choices[0].requires_structure is True


def test_matgl_handler_predicts_structure_with_injected_model():
    handler = MatGLPretrainedModelHandler(model_loader=lambda _: FakeModel())
    pred = handler.predict_structure(_structure_json(2))
    assert pred == -1.0


def test_benchmark_pretrained_model_writes_predictions_and_metrics(tmp_path):
    df = pd.DataFrame(
        {
            "material_id": ["a", "b", "c"],
            "structure": [_structure_json(1), _structure_json(2), _structure_json(2)],
            "formation_energy_per_atom": [-0.5, -1.1, -0.9],
        }
    )
    result = benchmark_pretrained_model(
        df,
        target="formation_energy_per_atom",
        output_dir=tmp_path,
        model_loader=lambda _: FakeModel(),
    )
    assert result.n_samples == 3
    assert "RMSE" in result.metrics
    assert "RMSE" in result.baseline_metrics
    assert (tmp_path / "pretrained_predictions.csv").exists()
    assert np.isfinite(result.metrics["RMSE"])
