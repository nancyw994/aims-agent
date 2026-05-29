#!/usr/bin/env python
"""LLM-guided robustness loop for tabular materials-property conclusions.

Each iteration writes machine-readable uncertainty notes, asks the LLM for the
next bounded experiment plan, runs that plan, and stops when the conclusion is
strong or the iteration budget is exhausted.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import pandas as pd

from aims_agent.llm import LMF_LLM
from run_spall_robustness import DEFAULT_DATA, DEFAULT_TARGET, run_robustness


ALLOWED_MISSING = ["median", "knn", "drop_sparse_cols", "drop_rows"]
ALLOWED_OUTLIER = ["keep", "clip_iqr", "drop_iqr"]
ALLOWED_MODELS = ["Gradient Boosting", "Random Forest", "XGBoost", "Ridge"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Use LLM to iterate robustness experiments until strong.")
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--output-root", default="results")
    parser.add_argument(
        "--drop-columns",
        nargs="*",
        default=None,
        help="Columns to exclude from feature candidates if present. Defaults to robustness workflow defaults.",
    )
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--min-folds", type=int, default=3)
    parser.add_argument("--max-folds", type=int, default=5)
    parser.add_argument("--min-repeats", type=int, default=2)
    parser.add_argument("--max-repeats", type=int, default=4)
    return parser.parse_args()


def read_csv_head(path: Path, n: int = 8) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return pd.read_csv(path).head(n).to_dict(orient="records")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_model_evidence(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "robustness_summary.csv"
    if not summary_path.exists():
        return {"models": [], "ranking_basis": "robustness_summary.csv missing"}
    df = pd.read_csv(summary_path)
    if df.empty:
        return {"models": [], "ranking_basis": "robustness_summary.csv empty"}

    rows = []
    for model, group in df.groupby("model"):
        best = group.sort_values(["robustness_score", "rmse_mean"]).iloc[0]
        top5_hits = int((df.head(5)["model"] == model).sum())
        rows.append(
            {
                "model": model,
                "best_rank": int(df.index[df["model"].eq(model)][0]) + 1,
                "top5_count": top5_hits,
                "best_strategy": {
                    "missing_strategy": best["missing_strategy"],
                    "outlier_strategy": best["outlier_strategy"],
                },
                "best_robustness_score": float(best["robustness_score"]),
                "best_rmse_mean": float(best["rmse_mean"]),
                "best_rmse_std": float(best["rmse_std"]),
                "best_r2_mean": float(best["r2_mean"]),
                "best_uq_miscalibration_area_mean": float(best["uq_miscalibration_area_mean"]),
                "best_uq_sharpness_mean": float(best["uq_sharpness_mean"]),
                "best_uq_coverage_95_mean": float(best["uq_coverage_95_mean"]),
                "best_cv_instability": float(best["cv_instability"]),
                "mean_rank_across_strategies": float(group.index.to_series().add(1).mean()),
                "n_strategy_rows": int(len(group)),
            }
        )
    model_df = pd.DataFrame(rows).sort_values(
        ["best_robustness_score", "best_rmse_mean", "best_uq_miscalibration_area_mean"]
    )
    evidence = {
        "ranking_basis": (
            "Models are compared by robustness_score first, then RMSE and UQ metrics. "
            "robustness_score combines RMSE, uncertainty-toolbox miscalibration, sharpness, and CV instability."
        ),
        "final_model_authority": "Metrics choose the final model. LLM only proposes the next candidate experiment set.",
        "model_evidence": model_df.to_dict(orient="records"),
        "current_best_model": str(model_df.iloc[0]["model"]) if not model_df.empty else None,
    }
    (run_dir / "model_selection_evidence.json").write_text(
        json.dumps(evidence, indent=2, default=str),
        encoding="utf-8",
    )
    return evidence


def build_uncertainty_notes(run_dir: Path, iteration: int, previous_plan: dict | None) -> dict[str, Any]:
    summary = read_json(run_dir / "robustness_summary.json")
    model_evidence = summarize_model_evidence(run_dir)
    notes = {
        "iteration": iteration,
        "run_dir": str(run_dir),
        "decision_sequence": [
            "profile data and run current robustness experiment",
            "write uncertainty_notes.json and model_selection_evidence.json",
            "LLM reads evidence and proposes only the next experiment candidates",
            "code runs repeated CV and uncertainty-toolbox validation",
            "metrics, not the LLM, choose the final model by robustness_score",
        ],
        "previous_plan": previous_plan or {},
        "conclusion_strength": summary.get("conclusion_strength"),
        "reasons": summary.get("reasons", []),
        "best": summary.get("best", {}),
        "model_selection_evidence": model_evidence,
        "top_strategy_rows": read_csv_head(run_dir / "robustness_summary.csv", n=12),
        "stable_features": read_csv_head(run_dir / "feature_stability.csv", n=15),
        "available_next_actions": {
            "missing_strategies": ALLOWED_MISSING,
            "outlier_strategies": ALLOWED_OUTLIER,
            "models": ALLOWED_MODELS,
            "folds_range": "3-5",
            "repeats_range": "2-4",
        },
        "strong_criteria": {
            "cv_instability": "<= 0.10 for the best strategy",
            "uq_miscalibration_area_mean": "<= 0.10 for the best strategy",
            "feature_stability": "top feature top10_frequency >= 0.70",
            "model_stability": "top-ranked strategies should preferably converge on the same exact model",
        },
    }
    path = run_dir / "uncertainty_notes.json"
    path.write_text(json.dumps(notes, indent=2, default=str), encoding="utf-8")
    return notes


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}") + 1
    if start < 0 or end <= start:
        raise ValueError(f"LLM did not return a JSON object: {text[:500]}")
    return json.loads(cleaned[start:end])


def clamp_int(value: Any, low: int, high: int, default: int) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError):
        return default
    return min(max(out, low), high)


def clean_choice_list(values: Any, allowed: list[str], default: list[str]) -> list[str]:
    if not isinstance(values, list):
        return default
    cleaned = []
    for value in values:
        if value in allowed and value not in cleaned:
            cleaned.append(value)
    return cleaned or default


def request_next_plan(notes: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    prompt = f"""You are controlling a bounded robustness workflow for materials ML.

Goal: make the conclusion strength become "strong" if the data supports it.

You must choose only from the allowed actions in the JSON notes. Do not invent new code,
new models, new preprocessing names, or external data. Prefer changes that directly
address the listed weak/moderate reasons. If the evidence cannot become strong with
the available data/actions, say so and choose the best diagnostic next run.

Model recommendation rule:
- You are not the final model selector.
- Recommend candidate models only from model_selection_evidence and available_next_actions.
- Justify each recommended model using evidence fields such as best_robustness_score,
  best_rmse_mean, best_uq_miscalibration_area_mean, best_cv_instability, top5_count,
  and mean_rank_across_strategies.
- The final model will be chosen later by repeated CV + uncertainty-toolbox metrics.

Return ONLY valid JSON:
{{
  "continue": true,
  "reasoning": "short explanation",
  "model_recommendation_basis": [
    {{
      "model": "allowed model name",
      "why_include": "evidence-based reason",
      "risk_to_check": "metric or failure mode to test"
    }}
  ],
  "next_config": {{
    "missing_strategies": ["..."],
    "outlier_strategies": ["..."],
    "models": ["..."],
    "folds": 3,
    "repeats": 2
  }},
  "expected_effect": "what should improve and why",
  "stop_if_not_improved": "condition"
}}

Uncertainty notes:
{json.dumps(notes, indent=2, default=str)}
"""
    response = LMF_LLM(prompt)
    plan = extract_json_object(response)
    cfg = plan.get("next_config") if isinstance(plan.get("next_config"), dict) else {}
    plan["next_config"] = {
        "missing_strategies": clean_choice_list(
            cfg.get("missing_strategies"),
            ALLOWED_MISSING,
            ["median", "drop_sparse_cols"],
        ),
        "outlier_strategies": clean_choice_list(
            cfg.get("outlier_strategies"),
            ALLOWED_OUTLIER,
            ["clip_iqr", "drop_iqr"],
        ),
        "models": clean_choice_list(
            cfg.get("models"),
            ALLOWED_MODELS,
            ["Gradient Boosting", "Random Forest", "XGBoost"],
        ),
        "folds": clamp_int(cfg.get("folds"), args.min_folds, args.max_folds, args.folds),
        "repeats": clamp_int(cfg.get("repeats"), args.min_repeats, args.max_repeats, args.repeats),
    }
    # Guardrails: the LLM may choose a diagnostic-only run that is not viable
    # for this sparse dataset. Keep its idea, but add enough controls to run.
    missing = plan["next_config"]["missing_strategies"]
    if missing == ["drop_rows"]:
        plan["next_config"]["missing_strategies"] = ["drop_rows", "median", "drop_sparse_cols"]
    models = plan["next_config"]["models"]
    if models == ["Ridge"]:
        plan["next_config"]["models"] = ["Ridge", "Gradient Boosting", "Random Forest"]
    plan["continue"] = bool(plan.get("continue", True))
    if not isinstance(plan.get("model_recommendation_basis"), list):
        selected = set(plan["next_config"]["models"])
        evidence_rows = notes.get("model_selection_evidence", {}).get("model_evidence", [])
        plan["model_recommendation_basis"] = [
            {
                "model": row.get("model"),
                "why_include": (
                    f"Selected by sanitized next_config; evidence rank={row.get('best_rank')}, "
                    f"score={row.get('best_robustness_score')}, "
                    f"miscalibration={row.get('best_uq_miscalibration_area_mean')}, "
                    f"cv_instability={row.get('best_cv_instability')}."
                ),
                "risk_to_check": "whether repeated CV instability and UQ miscalibration improve",
            }
            for row in evidence_rows
            if row.get("model") in selected
        ]
    return plan


def namespace_for_run(args: argparse.Namespace, cfg: dict[str, Any], output_root: Path) -> argparse.Namespace:
    return SimpleNamespace(
        data=args.data,
        target=args.target,
        output_root=str(output_root),
        folds=cfg["folds"],
        repeats=cfg["repeats"],
        missing_strategies=cfg["missing_strategies"],
        outlier_strategies=cfg["outlier_strategies"],
        models=cfg["models"],
        drop_columns=args.drop_columns,
    )


def main() -> None:
    args = parse_args()
    loop_root = Path(args.output_root) / datetime.now().strftime("llm_strong_loop_%Y%m%d_%H%M%S")
    loop_root.mkdir(parents=True, exist_ok=True)

    plan_history: list[dict[str, Any]] = []
    current_plan = {
        "continue": True,
        "reasoning": "initial broad robustness baseline",
        "next_config": {
            "missing_strategies": ["median", "knn", "drop_sparse_cols"],
            "outlier_strategies": ["keep", "clip_iqr", "drop_iqr"],
            "models": ["Gradient Boosting", "Random Forest", "XGBoost"],
            "folds": args.folds,
            "repeats": args.repeats,
        },
        "expected_effect": "establish baseline robustness risks",
    }

    final_notes = {}
    for iteration in range(1, args.max_iterations + 1):
        cfg = current_plan["next_config"]
        run_args = namespace_for_run(args, cfg, loop_root)
        print(f"\n=== LLM robustness iteration {iteration}/{args.max_iterations} ===")
        print(json.dumps(cfg, indent=2))
        run_dir = run_robustness(run_args, run_label=f"iteration_{iteration:02d}")
        notes = build_uncertainty_notes(run_dir, iteration, current_plan)
        final_notes = notes
        plan_history.append({"iteration": iteration, "run_dir": str(run_dir), "plan": current_plan, "notes": notes})

        if notes.get("conclusion_strength") == "strong":
            print("\nConclusion reached strong; stopping.")
            break
        if iteration >= args.max_iterations:
            print("\nIteration budget exhausted before reaching strong.")
            break

        next_plan = request_next_plan(notes, args)
        (run_dir / "llm_next_experiment_plan.json").write_text(
            json.dumps(next_plan, indent=2, default=str),
            encoding="utf-8",
        )
        (run_dir / "llm_next_plan.json").write_text(json.dumps(next_plan, indent=2, default=str), encoding="utf-8")
        if not next_plan.get("continue", True):
            print("\nLLM recommended stopping.")
            current_plan = next_plan
            break
        current_plan = next_plan

    loop_summary = {
        "loop_root": str(loop_root),
        "max_iterations": args.max_iterations,
        "final_conclusion_strength": final_notes.get("conclusion_strength"),
        "final_reasons": final_notes.get("reasons", []),
        "history": [
            {
                "iteration": item["iteration"],
                "run_dir": item["run_dir"],
                "plan": item["plan"],
                "conclusion_strength": item["notes"].get("conclusion_strength"),
                "reasons": item["notes"].get("reasons", []),
                "best": item["notes"].get("best", {}),
            }
            for item in plan_history
        ],
    }
    (loop_root / "llm_strong_loop_summary.json").write_text(
        json.dumps(loop_summary, indent=2, default=str),
        encoding="utf-8",
    )
    print("\nLLM strong loop summary:")
    print(json.dumps(loop_summary, indent=2, default=str))
    print(f"\nOutputs written to: {loop_root}")


if __name__ == "__main__":
    main()
