"""Tests for aims_agent.synthetic_loader."""

from __future__ import annotations

from aims_agent.synthetic_loader import SyntheticDataLoader


def test_synthetic_load_default_shape():
    loader = SyntheticDataLoader()
    bundle = loader.load_dataset({"n_samples": 30, "random_seed": 0})
    assert bundle.df.shape[0] == 30
    assert "hardness_HV" in bundle.df.columns
    assert bundle.schema.target == "hardness_HV"
    assert len(bundle.schema.features) >= 3
