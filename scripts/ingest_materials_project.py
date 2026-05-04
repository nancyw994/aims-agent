#!/usr/bin/env python
"""Reproducible Materials Project/local MatSci ingestion entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aims_agent.data_interface import get_metadata
from aims_agent.matsci_data_ingestor import MaterialsProjectDataIngestor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest and preprocess Materials Project data.")
    parser.add_argument(
        "--config",
        default="examples/materials_project_ingestion_config.json",
        help="JSON config with Materials Project query/preprocessing options.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional override for preprocessed CSV output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if args.output:
        config["output_path"] = args.output

    loader = MaterialsProjectDataIngestor()
    bundle = loader.load_dataset(config)
    metadata = get_metadata(bundle)

    print("Ingestion complete")
    print(json.dumps(metadata, indent=2))
    if config.get("output_path"):
        print(f"Preprocessed dataset saved: {config['output_path']}")
    else:
        print(bundle.df.head().to_string(index=False))


if __name__ == "__main__":
    main()
