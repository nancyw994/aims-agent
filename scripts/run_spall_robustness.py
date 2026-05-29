#!/usr/bin/env python
"""Robustness workflow for tabular materials-property model conclusions.

Runs multiple missing-data and outlier policies, repeated CV, uncertainty-toolbox
evaluation, and feature-importance stability summaries.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from aims_agent.model_strategy_analysis import (
    DROP_COLUMNS,
    TARGET,
    clean_column_name,
    coerce_numeric_like_columns,
    estimate_model_prediction_std,
    infer_target_column,
    load_tabular_dataset,
    root_mean_squared_error,
)
from aims_agent.uncertainty_evaluator import UncertaintyEvaluator


RANDOM_STATE = 42
DEFAULT_DATA = "data/real_data/Spall_Strength_Database_AliShargh(Processed).csv"
DEFAULT_TARGET = "Spall (Gpa)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run materials-property robustness and UQ validation.")
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--output-root", default="results")
    parser.add_argument(
        "--drop-columns",
        nargs="*",
        default=sorted(DROP_COLUMNS),
        help="Columns to exclude from feature candidates if present.",
    )
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--missing-strategies",
        nargs="+",
        default=["median", "knn", "drop_sparse_cols"],
        choices=["median", "knn", "drop_sparse_cols", "drop_rows"],
    )
    parser.add_argument(
        "--outlier-strategies",
        nargs="+",
        default=["keep", "clip_iqr", "drop_iqr"],
        choices=["keep", "clip_iqr", "drop_iqr"],
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["Gradient Boosting", "Random Forest", "XGBoost"],
        choices=["Gradient Boosting", "Random Forest", "XGBoost", "Ridge"],
    )
    return parser.parse_args()


def load_tabular_frame(path: str | Path, target: str) -> tuple[pd.DataFrame, str]:
    df = load_tabular_dataset(path)
    df.columns = [clean_column_name(c) for c in df.columns]
    target = infer_target_column(df, target)
    df = coerce_numeric_like_columns(df, target)
    df = df.dropna(subset=[target]).reset_index(drop=True)
    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df.dropna(subset=[target]).reset_index(drop=True)
    likely_mpa_mask = df[target] > 50 if "spall" in target.lower() else pd.Series(False, index=df.index)
    if likely_mpa_mask.any():
        df.loc[likely_mpa_mask, target] = df.loc[likely_mpa_mask, target] / 1000.0
    return df, target


def apply_missing_policy(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    strategy: str,
) -> tuple[pd.DataFrame, list[str], dict]:
    work = df.copy()
    selected = [*features, target]
    report = {"missing_strategy": strategy, "rows_before": len(work), "features_before": len(features)}
    if strategy == "drop_rows":
        work = work.dropna(subset=selected).reset_index(drop=True)
    elif strategy == "drop_sparse_cols":
        keep_features = [
            col for col in features
            if col in work.columns and float(work[col].isna().mean()) <= 0.40
        ]
        dropped = [col for col in features if col not in keep_features]
        features = keep_features
        report["dropped_sparse_features"] = dropped
    elif strategy not in {"median", "knn"}:
        raise ValueError(f"Unsupported missing strategy: {strategy}")
    report["rows_after_missing_policy"] = len(work)
    report["features_after_missing_policy"] = len(features)
    report["missing_fraction_after_policy"] = float(work[[*features, target]].isna().mean().mean())
    return work, features, report


def iqr_bounds(series: pd.Series) -> tuple[float, float] | None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean) < 4:
        return None
    q1, q3 = clean.quantile([0.25, 0.75])
    iqr = q3 - q1
    if not np.isfinite(iqr) or iqr == 0:
        return None
    return float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)


def apply_outlier_policy(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    strategy: str,
) -> tuple[pd.DataFrame, dict]:
    work = df.copy()
    numeric_cols = [
        col for col in [*features, target]
        if col in work.columns and pd.api.types.is_numeric_dtype(work[col])
    ]
    report = {"outlier_strategy": strategy, "rows_before": len(work)}
    if strategy == "keep":
        report["rows_after_outlier_policy"] = len(work)
        return work, report

    mask = pd.Series(False, index=work.index)
    bounds_by_col = {}
    for col in numeric_cols:
        bounds = iqr_bounds(work[col])
        if bounds is None:
            continue
        low, high = bounds
        bounds_by_col[col] = [low, high]
        values = pd.to_numeric(work[col], errors="coerce")
        col_mask = (values < low) | (values > high)
        if strategy == "clip_iqr":
            work[col] = values.clip(lower=low, upper=high)
        else:
            mask = mask | col_mask.fillna(False)

    if strategy == "drop_iqr":
        work = work.loc[~mask].reset_index(drop=True)
        report["dropped_outlier_rows"] = int(mask.sum())
    elif strategy != "clip_iqr":
        raise ValueError(f"Unsupported outlier strategy: {strategy}")

    report["rows_after_outlier_policy"] = len(work)
    report["iqr_bounds"] = bounds_by_col
    return work, report


def build_preprocessor(X: pd.DataFrame, missing_strategy: str) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [col for col in X.columns if col not in numeric_cols]
    numeric_imputer = KNNImputer(n_neighbors=5) if missing_strategy == "knn" else SimpleImputer(strategy="median")
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=3)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", numeric_imputer), ("scaler", StandardScaler())]), numeric_cols),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", encoder)]), categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def model_registry() -> dict[str, object]:
    return {
        "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(n_estimators=250, random_state=RANDOM_STATE, n_jobs=1),
        "XGBoost": XGBRegressor(
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=1,
            tree_method="hist",
            eval_metric="rmse",
            n_estimators=200,
        ),
        "Ridge": Ridge(),
    }


def collapse_encoded_feature(name: str) -> str:
    name = name.replace("num__", "").replace("cat__", "")
    for prefix in ["Sample", "Treatment", "Flyer", "Flyer (processed)", "Type of experiment", "Spall direction"]:
        if name == prefix or name.startswith(prefix + "_"):
            return prefix
    return name


def model_feature_importance(model, feature_names: np.ndarray) -> pd.DataFrame:
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        values = np.abs(np.ravel(model.coef_))
    else:
        return pd.DataFrame(columns=["feature", "importance"])
    rows = pd.DataFrame({"encoded_feature": feature_names, "importance": values})
    rows["feature"] = rows["encoded_feature"].map(collapse_encoded_feature)
    return rows.groupby("feature", as_index=False)["importance"].sum()


def finite(value, default: float = np.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def evaluate_fold(
    model_name: str,
    model_template,
    X: pd.DataFrame,
    y: pd.Series,
    train_idx,
    test_idx,
    missing_strategy: str,
) -> tuple[dict, pd.DataFrame]:
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    preprocessor = build_preprocessor(X_train, missing_strategy)
    X_train_enc = preprocessor.fit_transform(X_train)
    X_test_enc = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out()

    model = clone(model_template)
    model.fit(X_train_enc, y_train)
    y_pred = np.asarray(model.predict(X_test_enc), dtype=float)
    rmse = root_mean_squared_error(y_test, y_pred)
    y_std = estimate_model_prediction_std(model, X_test_enc, fallback_std=max(rmse, 1e-8))

    summary, _ = UncertaintyEvaluator.evaluate_all(y_test, y_pred, y_std, verbose=False)
    coverage = UncertaintyEvaluator.compute_coverage(y_test, y_pred, y_std)

    importances = model_feature_importance(model, feature_names)
    importances["model"] = model_name
    row = {
        "model": model_name,
        "rmse": rmse,
        "mse": mean_squared_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
        "uq_calibration_mae": finite(summary.get("calibration_mae")),
        "uq_miscalibration_area": finite(summary.get("miscalibration_area")),
        "uq_sharpness": finite(summary.get("sharpness")),
        "uq_nll": finite(summary.get("nll")),
        "uq_crps": finite(summary.get("crps")),
        "uq_coverage_68": finite(coverage.get(0.68)),
        "uq_coverage_95": finite(coverage.get(0.95)),
    }
    return row, importances


def aggregate_results(fold_results: pd.DataFrame) -> pd.DataFrame:
    grouped = fold_results.groupby(["missing_strategy", "outlier_strategy", "model"])
    rows = []
    for key, group in grouped:
        missing, outlier, model = key
        rmse_mean = float(group["rmse"].mean())
        rmse_std = float(group["rmse"].std(ddof=1))
        cal = float(group["uq_miscalibration_area"].mean())
        sharp = float(group["uq_sharpness"].mean())
        cv_instability = rmse_std / max(rmse_mean, 1e-8)
        score = 0.60 * rmse_mean + 0.25 * cal + 0.10 * sharp + 0.05 * cv_instability
        rows.append(
            {
                "missing_strategy": missing,
                "outlier_strategy": outlier,
                "model": model,
                "robustness_score": score,
                "rmse_mean": rmse_mean,
                "rmse_std": rmse_std,
                "r2_mean": float(group["r2"].mean()),
                "uq_miscalibration_area_mean": cal,
                "uq_sharpness_mean": sharp,
                "uq_coverage_95_mean": float(group["uq_coverage_95"].mean()),
                "cv_instability": cv_instability,
                "n_folds": int(len(group)),
            }
        )
    return pd.DataFrame(rows).sort_values(["robustness_score", "rmse_mean"]).reset_index(drop=True)


def aggregate_feature_stability(importance_rows: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    if importance_rows.empty:
        return pd.DataFrame()
    ranked = importance_rows.copy()
    ranked["rank"] = ranked.groupby(["scenario", "model", "fold"])["importance"].rank(ascending=False, method="min")
    all_scenarios = ranked[["scenario", "model", "fold"]].drop_duplicates()
    n_runs = len(all_scenarios)
    rows = []
    for feature, group in ranked.groupby("feature"):
        top = group[group["rank"] <= top_k]
        rows.append(
            {
                "feature": feature,
                "mean_importance": float(group["importance"].mean()),
                "std_importance": float(group["importance"].std(ddof=1)) if len(group) > 1 else 0.0,
                "top10_frequency": float(len(top[["scenario", "model", "fold"]].drop_duplicates()) / max(n_runs, 1)),
                "mean_rank_when_present": float(top["rank"].mean()) if not top.empty else np.nan,
                "n_observations": int(len(group)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["top10_frequency", "mean_importance"], ascending=[False, False]
    ).reset_index(drop=True)


def conclusion_strength(summary: pd.DataFrame, feature_stability: pd.DataFrame) -> tuple[str, list[str]]:
    best = summary.iloc[0]
    top_freq = float(feature_stability.iloc[0]["top10_frequency"]) if not feature_stability.empty else 0.0
    risks = []
    if best["cv_instability"] > 0.10:
        risks.append(f"best strategy CV instability remains {best['cv_instability']:.1%}")
    if best["uq_miscalibration_area_mean"] > 0.10:
        risks.append(f"best strategy mean UQ miscalibration area is {best['uq_miscalibration_area_mean']:.3f}")
    if top_freq < 0.70:
        risks.append(f"top feature appears in top-10 only {top_freq:.1%} of fold/scenario/model runs")
    model_counts = summary.head(5)["model"].value_counts()
    if len(model_counts) > 1:
        risks.append("top-ranked strategies do not all select the same exact model")
    if not risks:
        return "strong", ["low CV instability, acceptable UQ calibration, and stable top features"]
    if len(risks) <= 2:
        return "moderate", risks
    return "weak", risks


def write_report(
    output_dir: Path,
    summary: pd.DataFrame,
    feature_stability: pd.DataFrame,
    scenario_reports: list[dict],
    strength: str,
    reasons: list[str],
) -> None:
    top = summary.head(12)
    top_features = feature_stability.head(15) if not feature_stability.empty else pd.DataFrame()
    lines = [
        "# Materials Property Robustness Report",
        "",
        f"Conclusion strength after robustness workflow: **{strength}**",
        "",
        "Reasons:",
        *(f"- {reason}" for reason in reasons),
        "",
        "## Best Strategy / Model Combinations",
        "",
        top.to_markdown(index=False),
        "",
        "## Stable Features",
        "",
        top_features.to_markdown(index=False) if not top_features.empty else "No feature importance data available.",
        "",
        "## Scenario Audit",
        "",
        pd.DataFrame(scenario_reports).to_markdown(index=False),
    ]
    (output_dir / "robustness_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_robustness(args: argparse.Namespace, *, run_label: str | None = None) -> Path:
    UncertaintyEvaluator.check_availability()
    label = run_label or datetime.now().strftime("robustness_%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / label
    output_dir.mkdir(parents=True, exist_ok=True)

    df, target = load_tabular_frame(args.data, args.target)
    drop_columns = set(getattr(args, "drop_columns", sorted(DROP_COLUMNS)) or [])
    features = [col for col in df.columns if col not in (drop_columns | {target})]
    registry = model_registry()
    rkf = RepeatedKFold(n_splits=args.folds, n_repeats=args.repeats, random_state=RANDOM_STATE)

    fold_rows = []
    importance_parts = []
    scenario_reports = []
    for missing_strategy in args.missing_strategies:
        missing_df, missing_features, missing_report = apply_missing_policy(df, features, target, missing_strategy)
        for outlier_strategy in args.outlier_strategies:
            scenario_df, outlier_report = apply_outlier_policy(missing_df, missing_features, target, outlier_strategy)
            scenario = f"missing={missing_strategy}|outlier={outlier_strategy}"
            scenario_reports.append({"scenario": scenario, **missing_report, **outlier_report})
            if len(scenario_df) < max(args.folds * 2, 20):
                continue
            X = scenario_df[missing_features].copy()
            y = scenario_df[target].astype(float)
            for model_name in args.models:
                print(f"Scenario {scenario} | model {model_name}", flush=True)
                model_template = registry[model_name]
                for fold, (train_idx, test_idx) in enumerate(rkf.split(X, y), 1):
                    row, importances = evaluate_fold(
                        model_name,
                        model_template,
                        X,
                        y,
                        train_idx,
                        test_idx,
                        missing_strategy,
                    )
                    row.update(
                        {
                            "scenario": scenario,
                            "missing_strategy": missing_strategy,
                            "outlier_strategy": outlier_strategy,
                            "fold": fold,
                            "n_rows": len(scenario_df),
                            "n_features": len(missing_features),
                        }
                    )
                    fold_rows.append(row)
                    if not importances.empty:
                        importances["scenario"] = scenario
                        importances["missing_strategy"] = missing_strategy
                        importances["outlier_strategy"] = outlier_strategy
                        importances["fold"] = fold
                        importance_parts.append(importances)

    fold_results = pd.DataFrame(fold_rows)
    if fold_results.empty:
        raise RuntimeError("No robustness folds were evaluated.")
    fold_results.to_csv(output_dir / "robustness_fold_results.csv", index=False)

    summary = aggregate_results(fold_results)
    summary.to_csv(output_dir / "robustness_summary.csv", index=False)

    importance_rows = pd.concat(importance_parts, ignore_index=True) if importance_parts else pd.DataFrame()
    if not importance_rows.empty:
        importance_rows.to_csv(output_dir / "robustness_feature_importance_long.csv", index=False)
    feature_stability = aggregate_feature_stability(importance_rows)
    feature_stability.to_csv(output_dir / "feature_stability.csv", index=False)

    strength, reasons = conclusion_strength(summary, feature_stability)
    payload = {
        "data": str(args.data),
        "target": target,
        "folds": args.folds,
        "repeats": args.repeats,
        "missing_strategies": args.missing_strategies,
        "outlier_strategies": args.outlier_strategies,
        "models": args.models,
        "drop_columns": sorted(drop_columns),
        "conclusion_strength": strength,
        "reasons": reasons,
        "best": summary.iloc[0].to_dict(),
    }
    (output_dir / "robustness_summary.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    write_report(output_dir, summary, feature_stability, scenario_reports, strength, reasons)

    print("\nBest robustness rows:")
    print(summary.head(10).to_string(index=False))
    print("\nStable features:")
    print(feature_stability.head(15).to_string(index=False))
    print(f"\nConclusion strength: {strength}")
    for reason in reasons:
        print(f"- {reason}")
    print(f"\nOutputs written to: {output_dir}")
    return output_dir


def main() -> None:
    args = parse_args()
    run_robustness(args)


if __name__ == "__main__":
    main()
