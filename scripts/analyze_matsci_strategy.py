#!/usr/bin/env python
"""Run data profiling and ML strategy formulation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aims_agent.agent import Agent
from aims_agent.csv_loader import CSVDataLoader
from aims_agent.data_analyzer import analyze_and_formulate_strategy
from aims_agent.matsci_data_ingestor import MaterialsProjectDataIngestor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile MatSci data and write an ML strategy.")
    parser.add_argument(
        "--config",
        default=None,
        help="Optional JSON config for MaterialsProjectDataIngestor or CSVDataLoader.",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="CSV/JSON data path. Overrides config['path'] when provided.",
    )
    parser.add_argument("--target", default=None, help="Target column.")
    parser.add_argument("--features", nargs="+", default=None, help="Feature columns.")
    parser.add_argument(
        "--source",
        choices=["auto", "matsci", "csv"],
        default="auto",
        help="Loader to use. auto uses MatSci loader for JSON and CSV loader for CSV.",
    )
    parser.add_argument(
        "--task-type",
        choices=["regression", "classification"],
        default="regression",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Parent directory under which a new run folder will be created.",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Use deterministic heuristic strategy instead of calling the LLM.",
    )
    return parser.parse_args()


def _load_config(args: argparse.Namespace) -> dict:
    config: dict = {}
    if args.config:
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.data:
        config["path"] = args.data
    if args.target:
        config["target"] = args.target
    if args.features:
        config["features"] = args.features
    return config


def _llm_backend_label(use_llm: bool) -> str:
    if not use_llm:
        return "disabled"
    openai_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    router_key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
    if openai_key:
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return f"OpenAI / {model}"
    if router_key:
        model = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")
        return f"OpenRouter / {model}"
    return "enabled, but no API key detected"


def _dataset_label(config: dict) -> str:
    if config.get("chemsys"):
        return f"Materials Project chemical system {config['chemsys']}"
    if config.get("material_ids"):
        return f"Materials Project IDs {', '.join(config['material_ids'])}"
    if config.get("elements"):
        return f"Materials Project element filter {', '.join(config['elements'])}"
    if config.get("path"):
        return str(config["path"])
    return "synthetic or unspecified dataset"


def _run_context(args: argparse.Namespace, config: dict, use_llm: bool) -> dict:
    source = "Materials Project API" if args.source == "matsci" or "chemsys" in config or "elements" in config else "CSV/JSON file" if args.source == "csv" or config.get("path") else "synthetic fallback"
    mode = "live API ingestion" if source == "Materials Project API" else "local file ingestion" if source == "CSV/JSON file" else "synthetic generation"
    preprocessing = {
        "missing": config.get("missing_strategy", "inherit from ingestor"),
        "outlier": config.get("outlier_strategy", "inherit from ingestor"),
        "scaling": config.get("scaling", "inherit from ingestor"),
    }
    if config.get("preprocessing_suggestion"):
        preprocessing["suggestion"] = config["preprocessing_suggestion"]
    return {
        "api": "Materials Project summary API" if source == "Materials Project API" else "N/A",
        "dataset": _dataset_label(config),
        "source": source,
        "mode": mode,
        "task_type": args.task_type,
        "target": config.get("target", "formation_energy_per_atom"),
        "llm": _llm_backend_label(use_llm),
        "model_mode": "LLM-guided" if use_llm else "deterministic heuristic",
        "preprocessing": preprocessing,
    }


def _choose_loader(args: argparse.Namespace, config: dict):
    if args.source == "matsci":
        return MaterialsProjectDataIngestor()
    if args.source == "csv":
        return CSVDataLoader()
    return MaterialsProjectDataIngestor()


def main() -> None:
    args = parse_args()
    config = _load_config(args)
    if not config.get("path") and not args.config:
        config["path"] = "data/materials_project_li_fe_o_preprocessed.csv"
        config.setdefault("target", "formation_energy_per_atom")

    loader = _choose_loader(args, config)
    bundle = loader.load_dataset(config)
    agent = None if args.no_llm else Agent()
    profile, strategy, paths = analyze_and_formulate_strategy(
        bundle,
        agent=agent,
        use_llm=not args.no_llm,
        task_type=args.task_type,
        output_dir=args.output_dir,
        run_context=_run_context(args, config, use_llm=not args.no_llm),
    )

    print("MatSci strategy complete")
    print(f"Rows: {profile.row_count}, features: {len(profile.feature_profiles)}")
    print(f"Top models: {', '.join(strategy.recommended_models[:3])}")
    print(json.dumps(paths, indent=2))


if __name__ == "__main__":
    main()
