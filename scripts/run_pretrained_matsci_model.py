#!/usr/bin/env python
"""Run a pre-trained MatSci model benchmark on structure data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aims_agent.pretrained_model_handler import (
    DEFAULT_PRETRAINED_MODEL,
    benchmark_pretrained_model,
    fetch_materials_project_structures,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark a pre-trained MatSci model.")
    parser.add_argument("--data", default=None, help="CSV/JSON file containing structure and target columns.")
    parser.add_argument("--config", default=None, help="JSON config for Materials Project structure fetch.")
    parser.add_argument("--model", default=DEFAULT_PRETRAINED_MODEL, help="MatGL pre-trained model name.")
    parser.add_argument("--target", default="formation_energy_per_atom", help="Target property column.")
    parser.add_argument("--structure-col", default="structure", help="Column containing pymatgen Structure JSON.")
    parser.add_argument("--output-dir", default="results/pretrained_model", help="Output directory.")
    parser.add_argument("--limit", type=int, default=12, help="Live fetch limit when no data file is supplied.")
    parser.add_argument("--chemsys", default="Li-Fe-O", help="Materials Project chemical system for live fetch.")
    return parser.parse_args()


def _load_json_or_csv(path: str):
    import pandas as pd

    p = Path(path)
    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            for key in ("data", "records", "materials", "docs"):
                if isinstance(data.get(key), list):
                    data = data[key]
                    break
        return pd.DataFrame(data)
    return pd.read_csv(p)


def _build_fetch_config(args: argparse.Namespace) -> dict:
    config = {
        "chemsys": args.chemsys,
        "limit": args.limit,
        "fields": [
            "material_id",
            "formula_pretty",
            "formation_energy_per_atom",
            "band_gap",
            "structure",
        ],
    }
    if args.config:
        config.update(json.loads(Path(args.config).read_text(encoding="utf-8")))
    return config


def main() -> None:
    args = parse_args()
    if args.data:
        df = _load_json_or_csv(args.data)
    else:
        df = fetch_materials_project_structures(_build_fetch_config(args))

    result = benchmark_pretrained_model(
        df,
        target=args.target,
        model_name=args.model,
        structure_col=args.structure_col,
        output_dir=args.output_dir,
    )
    print("Pre-trained MatSci benchmark complete")
    print(json.dumps(
        {
            "model": result.model_name,
            "target": result.target,
            "n_samples": result.n_samples,
            "metrics": result.metrics,
            "baseline_metrics": result.baseline_metrics,
            "prediction_path": result.prediction_path,
            "failed_records": result.failed_records[:5],
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
