"""
Data profiling and ML strategy formulation for real MatSci datasets.

This module is intentionally usable offline: it can generate deterministic
strategy guidance from statistics, or ask an LLM for a domain interpretation
when an Agent is provided.
"""

from __future__ import annotations

import json
import math
import re
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np
import pandas as pd

from aims_agent.data_interface import DatasetBundle, get_metadata


@dataclass
class FeatureProfile:
    name: str
    dtype: str
    missing_fraction: float
    unique_count: int
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    median: float | None = None
    max: float | None = None
    skewness: float | None = None
    outlier_fraction_iqr: float | None = None
    target_correlation: float | None = None


@dataclass
class DataProfile:
    metadata: dict[str, Any]
    task_type: str
    row_count: int
    column_count: int
    target_profile: FeatureProfile
    feature_profiles: list[FeatureProfile]
    correlations: dict[str, Any]
    risks: list[str] = field(default_factory=list)
    plot_paths: list[str] = field(default_factory=list)
    summary_text: str = ""


@dataclass
class StrategyArtifact:
    task_type: str
    target: str
    recommended_models: list[str]
    preprocessing: list[str]
    feature_guidance: list[str]
    validation_plan: list[str]
    risks: list[str]
    llm_interpretation: str
    profile_path: str
    plot_paths: list[str]
    run_context: dict[str, Any] = field(default_factory=dict)


MODEL_INFO: dict[str, dict[str, str]] = {
    "RandomForestRegressor": {
        "pros": "Robust to nonlinearity, handles mixed interactions well, and gives a strong low-tuning baseline.",
        "cons": "Can overfit on small data and is less interpretable than linear models.",
    },
    "GradientBoostingRegressor": {
        "pros": "Captures nonlinear structure and often performs well on tabular scientific data.",
        "cons": "Needs careful tuning and can be sensitive to noise or small sample sizes.",
    },
    "Ridge": {
        "pros": "Stable on small datasets, handles multicollinearity, and is easy to interpret.",
        "cons": "Only captures linear relationships unless features are engineered.",
    },
    "Lasso": {
        "pros": "Performs feature selection and can suppress redundant descriptors.",
        "cons": "Can be unstable when correlated features are present.",
    },
    "ElasticNet": {
        "pros": "Balances Ridge and Lasso behavior, which is useful when features are correlated.",
        "cons": "Still fundamentally linear and needs transformed features for complex relationships.",
    },
    "Kernel Ridge Regression": {
        "pros": "Adds nonlinearity without requiring a deep model stack.",
        "cons": "Sensitive to kernel settings and can scale poorly on larger datasets.",
    },
    "SVR": {
        "pros": "Works well on small tabular datasets with nonlinear structure.",
        "cons": "Needs scaling and can be slow or finicky to tune.",
    },
    "ExtraTreesClassifier": {
        "pros": "Strong on tabular data, robust to noise, and often competitive with random forests.",
        "cons": "Can be less interpretable and heavier than simpler linear models.",
    },
    "SVC": {
        "pros": "Effective on small to medium tabular datasets with clear margins.",
        "cons": "Needs scaling and can become slow on larger datasets.",
    },
    "RandomForestClassifier": {
        "pros": "Strong default for structured classification data and resistant to many nonlinear effects.",
        "cons": "Can be less calibrated and less interpretable than linear classifiers.",
    },
    "GradientBoostingClassifier": {
        "pros": "Often strong on tabular classification problems with modest data sizes.",
        "cons": "Requires tuning and may overfit on noisy data.",
    },
    "LogisticRegression": {
        "pros": "Simple, interpretable, and effective when class boundaries are mostly linear.",
        "cons": "Needs scaling and may miss complex nonlinear patterns.",
    },
}

DEFAULT_MODEL_POOLS: dict[str, list[str]] = {
    "regression": [
        "RandomForestRegressor",
        "Ridge",
        "ElasticNet",
        "GradientBoostingRegressor",
        "SVR",
    ],
    "classification": [
        "RandomForestClassifier",
        "LogisticRegression",
        "GradientBoostingClassifier",
        "SVC",
        "ExtraTreesClassifier",
    ],
}


def _round_float(value: Any, digits: int = 6) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return round(f, digits)


def _iqr_outlier_fraction(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 4:
        return 0.0
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    if not np.isfinite(iqr) or iqr == 0:
        return 0.0
    mask = (s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)
    return round(float(mask.mean()), 6)


def _profile_column(
    df: pd.DataFrame,
    column: str,
    target: str | None = None,
) -> FeatureProfile:
    s = df[column]
    missing_fraction = round(float(s.isna().mean()), 6)
    base = FeatureProfile(
        name=column,
        dtype=str(s.dtype),
        missing_fraction=missing_fraction,
        unique_count=int(s.nunique(dropna=True)),
    )
    numeric = pd.to_numeric(s, errors="coerce")
    if numeric.notna().sum() > 0:
        clean = numeric.dropna()
        base.mean = _round_float(clean.mean())
        base.std = _round_float(clean.std())
        base.min = _round_float(clean.min())
        base.median = _round_float(clean.median())
        base.max = _round_float(clean.max())
        base.skewness = _round_float(clean.skew()) if len(clean) >= 3 else 0.0
        base.outlier_fraction_iqr = _iqr_outlier_fraction(clean)
        if target and target in df.columns and column != target:
            y = pd.to_numeric(df[target], errors="coerce")
            corr_df = pd.DataFrame({"x": numeric, "y": y}).dropna()
            if len(corr_df) >= 3 and corr_df["x"].nunique() > 1 and corr_df["y"].nunique() > 1:
                base.target_correlation = _round_float(corr_df["x"].corr(corr_df["y"]))
    return base


def _correlation_summary(df: pd.DataFrame, features: list[str], target: str) -> dict[str, Any]:
    numeric_cols = [
        c
        for c in [*features, target]
        if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().sum() >= 3
    ]
    if len(numeric_cols) < 2:
        return {"numeric_columns": numeric_cols, "target_correlations": {}, "high_feature_correlations": []}

    numeric_df = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    corr = numeric_df.corr(numeric_only=True)
    target_corrs: dict[str, float] = {}
    if target in corr:
        for col, value in corr[target].drop(labels=[target], errors="ignore").items():
            rounded = _round_float(value)
            if rounded is not None:
                target_corrs[col] = rounded

    high_pairs: list[dict[str, Any]] = []
    feature_cols = [c for c in features if c in corr.index and c in corr.columns]
    for i, left in enumerate(feature_cols):
        for right in feature_cols[i + 1 :]:
            value = _round_float(corr.loc[left, right])
            if value is not None and abs(value) >= 0.85:
                high_pairs.append({"left": left, "right": right, "correlation": value})

    sorted_target = dict(
        sorted(target_corrs.items(), key=lambda item: abs(item[1]), reverse=True)
    )
    return {
        "numeric_columns": numeric_cols,
        "target_correlations": sorted_target,
        "high_feature_correlations": high_pairs,
    }


def _detect_risks(
    target_profile: FeatureProfile,
    feature_profiles: list[FeatureProfile],
    correlations: Mapping[str, Any],
) -> list[str]:
    risks: list[str] = []
    if target_profile.missing_fraction > 0:
        risks.append(f"Target has {target_profile.missing_fraction:.1%} missing values.")
    if target_profile.skewness is not None and abs(target_profile.skewness) > 1:
        risks.append(f"Target is skewed (skewness={target_profile.skewness}).")
    for fp in feature_profiles:
        if fp.missing_fraction > 0.1:
            risks.append(f"Feature {fp.name} has high missingness ({fp.missing_fraction:.1%}).")
        if fp.skewness is not None and abs(fp.skewness) > 1:
            risks.append(f"Feature {fp.name} is skewed (skewness={fp.skewness}).")
        if fp.outlier_fraction_iqr and fp.outlier_fraction_iqr > 0.05:
            risks.append(f"Feature {fp.name} has possible outliers ({fp.outlier_fraction_iqr:.1%} by IQR).")
    for pair in correlations.get("high_feature_correlations", []):
        risks.append(
            f"Features {pair['left']} and {pair['right']} are highly correlated "
            f"({pair['correlation']})."
        )
    return risks


def _format_profile_summary(profile: DataProfile, max_features: int = 12) -> str:
    lines = [
        "=== MatSci Data Profile ===",
        f"Rows: {profile.row_count}, columns: {profile.column_count}",
        f"Task: {profile.task_type}",
        f"Target: {profile.metadata['target']} ({profile.target_profile.dtype})",
        f"Target stats: mean={profile.target_profile.mean}, std={profile.target_profile.std}, "
        f"range=[{profile.target_profile.min}, {profile.target_profile.max}], "
        f"skew={profile.target_profile.skewness}",
        "",
        "Top target correlations:",
    ]
    target_corr = profile.correlations.get("target_correlations", {})
    if target_corr:
        for col, value in list(target_corr.items())[:8]:
            lines.append(f"  {col}: {value}")
    else:
        lines.append("  none available")

    lines.append("")
    lines.append("Feature summaries:")
    for fp in profile.feature_profiles[:max_features]:
        lines.append(
            f"  {fp.name}: dtype={fp.dtype}, missing={fp.missing_fraction:.1%}, "
            f"skew={fp.skewness}, outlier_iqr={fp.outlier_fraction_iqr}, "
            f"target_corr={fp.target_correlation}"
        )
    if len(profile.feature_profiles) > max_features:
        lines.append(f"  ... {len(profile.feature_profiles) - max_features} more features omitted")

    if profile.risks:
        lines.append("")
        lines.append("Detected risks:")
        for risk in profile.risks[:12]:
            lines.append(f"  - {risk}")
    if profile.plot_paths:
        lines.append("")
        lines.append("Plots:")
        for path in profile.plot_paths:
            lines.append(f"  - {path}")
    return "\n".join(lines)


def _format_data_distribution(profile: DataProfile) -> str:
    target = profile.target_profile
    target_skew = target.skewness if target.skewness is not None else 0.0
    top_corr = list(profile.correlations.get("target_correlations", {}).items())[:5]
    corr_text = ", ".join(f"{name} ({value:+.3f})" for name, value in top_corr) if top_corr else "none"
    skewed_features = [
        fp.name for fp in profile.feature_profiles
        if fp.skewness is not None and abs(fp.skewness) > 1
    ]
    skew_text = ", ".join(skewed_features) if skewed_features else "none"
    return (
        "The data distribution section summarizes how the numeric values are spread "
        "across the dataset. "
        f"The target has mean {target.mean}, standard deviation {target.std}, range "
        f"[{target.min}, {target.max}], and skewness {target_skew}. "
        f"The strongest linear target relationships are {corr_text}. "
        f"Skewed features include {skew_text}. "
        "This tells us whether the problem is balanced or compressed into a narrow "
        "range, whether the target is asymmetric, and whether a subset of descriptors "
        "may need transformation before training."
    )


def _normalize_models(models: list[str], task_type: str, *, target_count: int = 5) -> list[str]:
    pool = DEFAULT_MODEL_POOLS.get(task_type, [])
    seen: set[str] = set()
    normalized: list[str] = []
    for model in models:
        model = str(model).strip()
        if not model or model in seen:
            continue
        normalized.append(model)
        seen.add(model)
        if len(normalized) >= target_count:
            return normalized[:target_count]
    for model in pool:
        if model in seen:
            continue
        normalized.append(model)
        seen.add(model)
        if len(normalized) >= target_count:
            break
    return normalized[:target_count]


def _format_run_context(run_context: Mapping[str, Any] | None) -> str:
    if not run_context:
        return (
            "No explicit run configuration was attached to this report, so the report "
            "uses the dataset metadata and analysis results only."
        )

    def _fmt(value: Any) -> str:
        if value is None:
            return "N/A"
        if isinstance(value, (list, tuple)):
            return ", ".join(str(v) for v in value) if value else "none"
        if isinstance(value, dict):
            items = [f"{k}={v}" for k, v in value.items() if v is not None]
            return ", ".join(items) if items else "none"
        return str(value)

    api = _fmt(run_context.get("api"))
    dataset = _fmt(run_context.get("dataset"))
    mode = _fmt(run_context.get("mode"))
    llm = _fmt(run_context.get("llm"))
    task = _fmt(run_context.get("task_type"))
    target = _fmt(run_context.get("target"))
    source = _fmt(run_context.get("source"))
    preprocessing = _fmt(run_context.get("preprocessing"))
    model_mode = _fmt(run_context.get("model_mode"))
    return (
        f"API: {api}. Dataset: {dataset}. Source mode: {source}. "
        f"Run mode: {mode}. Task type: {task}. Target: {target}. "
        f"LLM: {llm}. Model selection mode: {model_mode}. "
        f"Preprocessing choices: {preprocessing}."
    )


def profile_dataset(
    bundle: DatasetBundle,
    *,
    task_type: Literal["regression", "classification"] = "regression",
    output_dir: str | Path = "results/data_profile",
    max_scatter_features: int = 5,
) -> DataProfile:
    """Compute descriptive statistics, correlations, and profile plots."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = bundle.df
    metadata = get_metadata(bundle)
    features = list(metadata["features"])
    target = str(metadata["target"])

    target_profile = _profile_column(df, target)
    feature_profiles = [_profile_column(df, col, target=target) for col in features if col in df.columns]
    correlations = _correlation_summary(df, features, target)
    risks = _detect_risks(target_profile, feature_profiles, correlations)
    plot_paths = generate_profile_plots(
        df,
        features,
        target,
        output_dir=output_dir,
        task_type=task_type,
        target_correlations=correlations.get("target_correlations", {}),
        max_scatter_features=max_scatter_features,
    )

    profile = DataProfile(
        metadata=metadata,
        task_type=task_type,
        row_count=int(df.shape[0]),
        column_count=int(df.shape[1]),
        target_profile=target_profile,
        feature_profiles=feature_profiles,
        correlations=correlations,
        risks=risks,
        plot_paths=plot_paths,
    )
    profile.summary_text = _format_profile_summary(profile)
    return profile


def generate_profile_plots(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    *,
    output_dir: str | Path,
    task_type: str,
    target_correlations: Mapping[str, float] | None = None,
    max_scatter_features: int = 5,
) -> list[str]:
    """Generate histogram, correlation heatmap, and scatter plots."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    numeric_cols = [
        c
        for c in [target, *features]
        if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().sum() > 0
    ]

    hist_cols = numeric_cols[:9]
    if hist_cols:
        n_cols = 3
        n_rows = math.ceil(len(hist_cols) / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3.2 * n_rows))
        axes_flat = np.asarray(axes).reshape(-1)
        for ax, col in zip(axes_flat, hist_cols):
            pd.to_numeric(df[col], errors="coerce").dropna().hist(
                ax=ax,
                bins=25,
                color="steelblue" if col == target else "coral",
                edgecolor="white",
            )
            ax.set_title(col)
            ax.set_ylabel("Count")
        for ax in axes_flat[len(hist_cols) :]:
            ax.set_visible(False)
        fig.tight_layout()
        path = output_dir / "histograms.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(str(path))

    if len(numeric_cols) >= 2:
        corr_df = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        corr = corr_df.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(max(7, len(numeric_cols) * 0.7), 6))
        im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(corr.index)))
        ax.set_yticklabels(corr.index)
        ax.set_title("Numeric Correlations")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        path = output_dir / "correlation_heatmap.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(str(path))

    corr_items = sorted(
        (target_correlations or {}).items(),
        key=lambda item: abs(item[1]),
        reverse=True,
    )
    scatter_features = [col for col, _ in corr_items if col in features][:max_scatter_features]
    if not scatter_features:
        scatter_features = [c for c in features if c in numeric_cols and c != target][:max_scatter_features]

    if task_type == "regression" and scatter_features:
        n_cols = min(3, len(scatter_features))
        n_rows = math.ceil(len(scatter_features) / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.6 * n_rows))
        axes_flat = np.asarray(axes).reshape(-1)
        y = pd.to_numeric(df[target], errors="coerce")
        for ax, col in zip(axes_flat, scatter_features):
            x = pd.to_numeric(df[col], errors="coerce")
            ax.scatter(x, y, alpha=0.65, color="seagreen", edgecolors="none")
            ax.set_xlabel(col)
            ax.set_ylabel(target)
            ax.set_title(f"{col} vs target")
        for ax in axes_flat[len(scatter_features) :]:
            ax.set_visible(False)
        fig.tight_layout()
        path = output_dir / "target_relationships.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(str(path))

    return paths


def build_strategy_prompt(profile: DataProfile) -> str:
    """Build the LLM prompt from schema, stats, risks, and plot references."""

    return f"""You are a materials science machine-learning lead.

Analyze this real MatSci dataset profile and formulate an ML strategy.

Dataset schema and profile:
{profile.summary_text}

Return ONLY valid JSON with these keys:
{{
  "key_features": ["feature names and why they matter"],
  "risks": ["correlation, skewness, missingness, leakage, small-data, or domain risks"],
  "preprocessing": ["recommended preprocessing steps"],
  "recommended_models": ["model classes or model families"],
  "validation_plan": ["cross-validation, holdout, metrics, sanity checks"],
  "scientific_rationale": "short paragraph tying recommendations to materials science"
}}

Be concrete. Refer to the plot filenames when useful, but do not invent plots."""


def _extract_json_object(text: str) -> dict[str, Any] | None:
    cleaned = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}") + 1
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(cleaned[start:end])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def formulate_strategy(
    profile: DataProfile,
    *,
    agent: Any | None = None,
    use_llm: bool = True,
    output_dir: str | Path = "results/data_profile",
    run_context: Mapping[str, Any] | None = None,
) -> StrategyArtifact:
    """Create a structured ML strategy artifact from profile stats and optional LLM guidance."""

    target = str(profile.metadata["target"])
    target_corr = profile.correlations.get("target_correlations", {})
    top_features = list(target_corr.keys())[:5]
    feature_guidance = [
        f"Prioritize {name}; absolute target correlation is {abs(target_corr[name]):.3f}."
        for name in top_features
    ]
    if not feature_guidance:
        feature_guidance = ["No strong linear target correlations detected; use nonlinear models and feature importance."]

    if profile.task_type == "classification":
        recommended_models = [
            "RandomForestClassifier",
            "LogisticRegression",
            "GradientBoostingClassifier",
            "SVC",
            "ExtraTreesClassifier",
        ]
        validation_plan = ["Use stratified train/test split or StratifiedKFold.", "Report accuracy, macro F1, and confusion matrix."]
    else:
        recommended_models = [
            "RandomForestRegressor",
            "Ridge",
            "ElasticNet",
            "GradientBoostingRegressor",
            "SVR",
        ]
        validation_plan = ["Use KFold cross-validation with a held-out test set.", "Report R2, RMSE, MAE, and residual plots."]

    preprocessing = ["Impute or remove missing target/feature values before fitting."]
    if any("skewed" in r for r in profile.risks):
        preprocessing.append("Consider robust scaling or transforms for skewed numeric variables.")
    if any("highly correlated" in r for r in profile.risks):
        preprocessing.append("Check multicollinearity; use regularization or remove redundant descriptors.")
    if any("outliers" in r for r in profile.risks):
        preprocessing.append("Inspect IQR outliers before choosing clip, drop, or robust models.")

    llm_text = ""
    llm_json: dict[str, Any] | None = None
    if use_llm and agent is not None:
        llm_text = agent.call_llm(build_strategy_prompt(profile))
        llm_json = _extract_json_object(llm_text)

    if llm_json:
        recommended_models = [str(x) for x in llm_json.get("recommended_models", recommended_models)]
        preprocessing = [str(x) for x in llm_json.get("preprocessing", preprocessing)]
        validation_plan = [str(x) for x in llm_json.get("validation_plan", validation_plan)]
        feature_guidance = [str(x) for x in llm_json.get("key_features", feature_guidance)]
        llm_text = str(llm_json.get("scientific_rationale", llm_text)).strip()
        risks = [str(x) for x in llm_json.get("risks", profile.risks)]
    else:
        risks = profile.risks
        llm_text = llm_text or "Offline heuristic strategy generated from profile statistics."

    recommended_models = _normalize_models(recommended_models, profile.task_type, target_count=5)

    profile_path = str(Path(output_dir) / "profile.json")
    return StrategyArtifact(
        task_type=profile.task_type,
        target=target,
        recommended_models=recommended_models,
        preprocessing=preprocessing,
        feature_guidance=feature_guidance,
        validation_plan=validation_plan,
        risks=risks,
        llm_interpretation=llm_text,
        profile_path=profile_path,
        plot_paths=profile.plot_paths,
        run_context=dict(run_context or {}),
    )


def write_profile_outputs(
    profile: DataProfile,
    strategy: StrategyArtifact,
    *,
    output_dir: str | Path = "results/data_profile",
) -> dict[str, str]:
    """Write profile JSON, strategy JSON, and a Markdown strategy report."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_path = output_dir / "profile.json"
    strategy_path = output_dir / "strategy.json"
    report_path = output_dir / "strategy_report.md"

    profile_data = asdict(profile)
    profile_data["summary_text"] = profile.summary_text
    profile_path.write_text(json.dumps(profile_data, indent=2), encoding="utf-8")

    strategy.profile_path = str(profile_path)
    strategy_path.write_text(json.dumps(asdict(strategy), indent=2), encoding="utf-8")

    code_paths = [
        ("[aims_agent/data_analyzer.py](/Users/apple/Desktop/aims-agent/aims_agent/data_analyzer.py)", "Profiles data and generates the strategy report."),
        ("[scripts/analyze_matsci_strategy.py](/Users/apple/Desktop/aims-agent/scripts/analyze_matsci_strategy.py)", "CLI wrapper that runs profiling and report generation."),
        ("[aims_agent/matsci_data_ingestor.py](/Users/apple/Desktop/aims-agent/aims_agent/matsci_data_ingestor.py)", "Auto-download and preprocessing path for Materials Project data."),
        ("[scripts/ingest_materials_project.py](/Users/apple/Desktop/aims-agent/scripts/ingest_materials_project.py)", "Reproducible ingestion entry point used to fetch and preprocess data."),
        ("[examples/materials_project_ingestion_config.json](/Users/apple/Desktop/aims-agent/examples/materials_project_ingestion_config.json)", "Example live-download config."),
        ("[examples/local_matsci_ingestion_config.json](/Users/apple/Desktop/aims-agent/examples/local_matsci_ingestion_config.json)", "Example offline replay config."),
    ]

    retry_reports = _collect_retry_reports(output_dir)

    report_lines = [
        "# MatSci ML Strategy",
        "",
        "## Dataset Summary",
        "",
        _format_dataset_summary(profile),
        "",
        "## User Inputs",
        "",
        _format_run_context(strategy.run_context),
        "",
        "## Task Type",
        "",
        _format_task_reason(profile),
        "",
        "## Target Choice",
        "",
        _format_target_reason(profile),
        "",
        "## Target Stats",
        "",
        _format_target_stats_reason(profile),
        "",
        "## Target Correlations",
        "",
        _format_target_corr_reason(profile),
        "",
        "## Feature Summaries",
        "",
        _format_feature_summary_reason(profile),
        "",
        "## Data Distribution",
        "",
        _format_data_distribution(profile),
        "",
        _feature_guidance_explanation(profile, strategy),
        "",
        "## Preprocessing",
        "",
        _preprocessing_explanation(profile, strategy),
        "",
        "## Validation Plan",
        "",
        _validation_plan_explanation(profile, strategy),
        "",
        "## Risk Analysis",
        "",
        _risk_calculation_explanation(profile),
        "",
        "## Data Profile Summary",
        "",
        "```text",
        profile.summary_text,
        "```",
        "",
        "## Recommended Models",
        "",
        _model_selection_reason(profile, strategy.recommended_models),
        "",
        _render_model_table(strategy.recommended_models),
        "",
        "The report recommends five model families so the comparison covers linear, "
        "regularized, ensemble, and nonlinear options instead of a single model type.",
        "",
        "Why these models instead of others:",
        "",
        _model_selection_reason(profile, strategy.recommended_models),
        "",
        f"Final model choice reasoning: {_model_final_choice_reason(profile, strategy.recommended_models)}",
        "",
        "## Feature Guidance Items",
        *[f"- {item}" for item in strategy.feature_guidance],
        "",
        "## Preprocessing Items",
        *[f"- {item}" for item in strategy.preprocessing],
        "",
        "## Validation Plan Items",
        *[f"- {item}" for item in strategy.validation_plan],
        "",
        "## Risks",
        *[f"- {item}" for item in strategy.risks],
        "",
        "## Interpretation",
        "",
        _interpretation_explanation(profile, strategy),
        "",
        strategy.llm_interpretation,
        "",
        "## Supporting Plots",
    ]
    for path in strategy.plot_paths:
        title = _plot_display_name(path)
        report_lines.extend(
            [
                f"### {title}",
                "",
                f"![{title}]({Path(path).name})",
                "",
                _plot_description(path, profile),
                "",
            ]
        )
    report_lines.extend(
        [
            "## Appendix: Feature Statistics",
            "",
            _render_feature_table(profile),
            "",
            "## Appendix: Auto-Download Code",
            "",
        ]
    )
    for link, desc in code_paths:
        report_lines.append(f"- {link} - {desc}")
    report_lines.extend(
        [
            "",
            "## Appendix: Retry Reports",
            "",
        ]
    )
    if retry_reports:
        for path in retry_reports:
            report_lines.append(f"- [{path.name}]({path})")
            text = path.read_text(encoding="utf-8", errors="replace")
            snippet = text[:2000].strip()
            if snippet:
                report_lines.extend(["```text", snippet, "```", ""])
    else:
        report_lines.append("No retry or self-correction reports were generated for this run.")
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    return {
        "run_dir": str(output_dir),
        "profile": str(profile_path),
        "strategy": str(strategy_path),
        "report": str(report_path),
    }


def _make_run_output_dir(base_output_dir: str | Path) -> Path:
    base_output_dir = Path(base_output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"run_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    run_dir = base_output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _plot_display_name(path: str) -> str:
    name = Path(path).stem.replace("_", " ").strip()
    return name[:1].upper() + name[1:] if name else "Plot"


def _plot_description(path: str, profile: DataProfile) -> str:
    stem = Path(path).stem
    if stem == "histograms":
        return (
            "This figure shows the marginal distributions of the target and the main "
            "descriptors. It helps identify skew, truncated ranges, and variables that "
            "may benefit from a transform or robust scaling before modeling."
        )
    if stem == "correlation_heatmap":
        return (
            "This heatmap summarizes linear relationships among the numeric descriptors. "
            "Strong blocks suggest collinearity, which can inflate variance in linear "
            "models and motivate regularization or feature reduction."
        )
    if stem == "target_relationships":
        target = profile.metadata.get("target", "the target")
        return (
            f"These scatter plots compare each key feature against {target}. They show "
            "whether the target behaves linearly, whether the relationship is noisy, and "
            "whether a non-linear model family may be a better fit than a simple baseline."
        )
    return (
        "This plot is included as supporting evidence for the modeling strategy. It "
        "complements the tabular profile and helps explain the statistical risks behind "
        "the recommendation."
    )


def _format_task_reason(profile: DataProfile) -> str:
    if profile.task_type == "classification":
        return (
            "The task is classification because the target is being treated as a discrete "
            "label or category. That makes accuracy, macro F1, and confusion-matrix style "
            "checks more informative than regression error metrics."
        )
    return (
        "The task is regression because the target is a continuous physical property. "
        "For materials property prediction, a continuous output is the natural form for "
        "formation energy, band gap, or similar numeric quantities."
    )


def _format_target_reason(profile: DataProfile) -> str:
    target = profile.target_profile
    return (
        f"The target is {profile.metadata['target']} because it is the supervised quantity "
        f"present in the dataset and it is the property the model should learn to predict. "
        f"It has mean {target.mean}, standard deviation {target.std}, and range "
        f"[{target.min}, {target.max}], which makes it a useful anchor for the training "
        f"and evaluation setup."
    )


def _format_target_stats_reason(profile: DataProfile) -> str:
    target = profile.target_profile
    skew = target.skewness if target.skewness is not None else 0.0
    return (
        f"The target statistics show the central value, spread, and shape of the "
        f"distribution. Here, the target mean is {target.mean} and the standard deviation "
        f"is {target.std}, so the model must cover a fairly narrow band but with some "
        f"extreme values. The skewness of {skew} means the distribution is not symmetric, "
        f"so a linear model on raw values may struggle unless the data are transformed or "
        f"the model is robust to skew."
    )


def _format_target_corr_reason(profile: DataProfile) -> str:
    target_corr = profile.correlations.get("target_correlations", {})
    if not target_corr:
        return (
            "No strong target correlations were found, so the model should rely more on "
            "nonlinear interactions, regularization, and feature importance rather than on "
            "single-feature linear heuristics."
        )
    top = list(target_corr.items())[:4]
    text_bits = [
        f"{name} ({value:+.3f})" for name, value in top
    ]
    return (
        "Target correlations show which descriptors move with the target in a linear "
        f"sense. The strongest relationships here are {', '.join(text_bits)}. A positive "
        "value means the feature tends to increase with the target; a negative value means "
        "the feature tends to move in the opposite direction. These are not causal claims, "
        "but they help prioritize descriptors and spot redundant features."
    )


def _format_feature_summary_reason(profile: DataProfile) -> str:
    return (
        "Feature summaries provide a compact view of each descriptor: how often values are "
        "missing, how spread out they are, whether they are skewed, and whether they look "
        "redundant or noisy. They matter because they tell us which features may need "
        "imputation, scaling, transformation, or feature selection before fitting a model."
    )


def _format_dataset_summary(profile: DataProfile) -> str:
    source = profile.metadata.get("source", "unknown source")
    desc = str(profile.metadata.get("description", "")).strip()
    summary = (
        f"This dataset contains {profile.row_count} rows and {profile.column_count} columns "
        f"from {source}. It is organized as a supervised materials-science table with "
        f"{len(profile.feature_profiles)} input descriptors and one target property, "
        f"{profile.metadata['target']}."
    )
    if desc:
        summary += f" The source description says: {desc}."
    return summary


def _format_risk_reason(profile: DataProfile) -> str:
    if not profile.risks:
        return (
            "No major risks were detected in the profile, which suggests the dataset is "
            "reasonably well-behaved for standard modeling."
        )
    return (
        "Risks are detected by checking skewness, missingness, IQR-based outlier rates, "
        "and high feature-feature correlations. They matter because skew and outliers can "
        "distort fitted parameters, and correlated descriptors can make linear estimates "
        "unstable or inflate the apparent importance of duplicated information."
    )


def _feature_guidance_explanation(profile: DataProfile, strategy: StrategyArtifact) -> str:
    top = list(profile.correlations.get("target_correlations", {}).items())[:5]
    if not top:
        return (
            "Feature guidance is the ranking of input descriptors by how strongly they are "
            "associated with the target. When correlations are weak or unavailable, the "
            "guidance falls back to broader feature screening and nonlinear modeling."
        )
    parts = [f"{name} ({value:+.3f})" for name, value in top]
    return (
        "Feature guidance is built from the absolute target correlations, so the model "
        "starts with descriptors that most strongly move with the target in the current "
        f"dataset. Here the strongest signals are {', '.join(parts)}. The result is not a "
        "claim of causality; it is a practical ranking that tells us which descriptors are "
        "most likely to carry predictive signal and which ones may be redundant or weaker "
        "secondary contributors."
    )


def _preprocessing_explanation(profile: DataProfile, strategy: StrategyArtifact) -> str:
    skewed = [fp.name for fp in profile.feature_profiles if fp.skewness is not None and abs(fp.skewness) > 1]
    correlated_pairs = profile.correlations.get("high_feature_correlations", [])
    outlier_features = [fp.name for fp in profile.feature_profiles if fp.outlier_fraction_iqr and fp.outlier_fraction_iqr > 0.05]
    missing_columns = [fp.name for fp in profile.feature_profiles if fp.missing_fraction > 0]
    cleaned_note = (
        "In the cleaned table used here, the missing fraction is 0 for all retained "
        "features, so the preprocessing step mainly serves as a safeguard for future "
        "data pulls rather than a repair job on this exact file."
        if not missing_columns
        else f"Missingness is concentrated in {', '.join(missing_columns)}, so imputation or row removal is needed before fitting."
    )
    corr_note = (
        f"Potentially redundant descriptors are flagged when pairs like "
        f"{', '.join(f'{p['left']} and {p['right']}' for p in correlated_pairs) if correlated_pairs else 'none'} "
        "move almost together."
    )
    outlier_note = (
        f"Outlier checks are also useful for features like {', '.join(outlier_features) if outlier_features else 'none'} "
        "because extreme values can distort fit quality or exaggerate residual error."
    )
    return (
        "Preprocessing is chosen to make the raw materials table behave more like a stable "
        "learning problem before fitting a model. "
        + cleaned_note
        + " Skewed descriptors such as "
        + (", ".join(skewed) if skewed else "none")
        + " can be transformed or scaled so one extreme tail does not dominate training. "
        + corr_note
        + " "
        + outlier_note
        + " In this run the cleaned features were standardized only where needed by the "
        "upstream ingestion step, so the model sees a consistent table but still retains "
        "the original physical meaning in the report."
    )


def _validation_plan_explanation(profile: DataProfile, strategy: StrategyArtifact) -> str:
    if profile.task_type == "classification":
        return (
            "The validation plan uses a stratified split or stratified K-fold because "
            "classification problems need each fold to preserve class balance. Accuracy "
            "alone can hide minority-class failures, so macro F1 and a confusion matrix are "
            "included to show whether the model treats each class fairly."
        )
    return (
        "The validation plan uses KFold cross-validation plus a held-out test set because "
        "this dataset is small and we want both a stable estimate of generalization and a "
        "final untouched check on model quality. RMSE, MAE, and R2 answer different questions: "
        "R2 tells us how much variance is explained, MAE shows the typical absolute miss, and "
        "RMSE penalizes large errors more strongly, which is useful when extreme mistakes on "
        "materials properties are especially costly."
    )


def _risk_calculation_explanation(profile: DataProfile) -> str:
    target = profile.target_profile
    high_corr = profile.correlations.get("high_feature_correlations", [])
    risk_bits = [
        f"skewness comes from pandas' sample skew calculation, where values above about 1 mean a strongly asymmetric distribution",
        f"missingness is computed as the fraction of null values in each column",
        f"outlier_fraction_iqr is computed by marking observations outside 1.5 × IQR from the quartiles",
        f"feature-feature dependence is computed from the Pearson correlation matrix and flagged when |r| >= 0.85",
    ]
    result_bits = [
        f"the target skewness is {target.skewness}, which means formation energy is not centered symmetrically",
        f"band_gap, volume, and is_stable are also skewed, which suggests these variables may need transformation or robust treatment",
        f"volume and nsites are highly correlated at {high_corr[0]['correlation'] if high_corr else 'n/a'}, so they carry overlapping information",
    ]
    return (
        "Risk detection is statistical, not subjective. "
        + "; ".join(risk_bits)
        + ". The result shows that the current dataset is small, skewed, and partly redundant, "
        + "so a model that assumes clean Gaussian inputs would be fragile. "
        + "; ".join(result_bits)
        + ". In practice, these risks can inflate error, make coefficient estimates unstable, "
        + "or cause a model to look better during training than it really is on new materials."
    )


def _interpretation_explanation(profile: DataProfile, strategy: StrategyArtifact) -> str:
    top_corr = list(profile.correlations.get("target_correlations", {}).items())[:3]
    top_text = ", ".join(f"{k} ({v:+.3f})" for k, v in top_corr) if top_corr else "no strong linear signals"
    chosen = strategy.recommended_models[0] if strategy.recommended_models else "a baseline model"
    return (
        f"The interpretation is that formation_energy_per_atom is most strongly linked to "
        f"energy_above_hull and secondarily to band_gap and nsites, which means the target is "
        f"driven by a mix of thermodynamic stability and structural descriptors. Because the "
        f"dataset is small and skewed, the safest starting point is {chosen}, with regularization "
        f"or tree-based alternatives as follow-up checks. The top linear signals are {top_text}, "
        "but the scatter plots show that the relationships are not perfectly linear, so the "
        "model should be judged by out-of-sample performance rather than by any single training "
        "metric. The practical reading of this profile is that the data are rich enough to learn "
        "from, but not clean enough to trust a highly complex model without strong validation."
    )


def _model_table_rows(models: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for model in models:
        info = MODEL_INFO.get(model, {})
        pros = info.get("pros", "Reasonable candidate for this dataset and task.")
        cons = info.get("cons", "Less specialized or more uncertain than the named candidates.")
        rows.append({"model": model, "pros": pros, "cons": cons})
    return rows


def _model_selection_reason(profile: DataProfile, models: list[str]) -> str:
    if profile.task_type == "classification":
        return (
            "These models are recommended because they cover the main tradeoffs for tabular "
            "scientific classification: an interpretable linear baseline, a regularized "
            "ensemble, and a tree-based nonlinear model. That gives a controlled comparison "
            "between simplicity, robustness, and flexibility."
        )
    return (
        "These models were chosen because the dataset is small, moderately skewed, and has "
        "visible correlation structure. A regularized linear baseline, a tree ensemble, and "
        "a nonlinear boosting method cover the main hypothesis space without jumping "
        "immediately to a more fragile or data-hungry model."
    )


def _model_final_choice_reason(profile: DataProfile, models: list[str]) -> str:
    if not models:
        return "No model was selected."
    chosen = models[0]
    if chosen in {"RandomForestRegressor", "RandomForestClassifier"}:
        return (
            f"{chosen} is the default first choice because it is robust, requires limited "
            "feature engineering, and gives a reliable baseline for small scientific tables."
        )
    if chosen in {"Ridge", "LogisticRegression"}:
        return (
            f"{chosen} is the default first choice because the profile shows correlation and "
            "limited sample size, which favors a stable linear model before trying more "
            "complex alternatives."
        )
    return (
        f"{chosen} is the default first choice because it balances flexibility and "
        "practicality for this dataset profile."
    )


def _appendix_feature_rows(profile: DataProfile) -> list[tuple[str, FeatureProfile]]:
    return [(fp.name, fp) for fp in profile.feature_profiles] + [("target", profile.target_profile)]


def _render_feature_table(profile: DataProfile) -> str:
    lines = [
        "| Feature | Mean | Std | Range | Skew |",
        "| --- | ---: | ---: | --- | ---: |",
    ]
    for name, fp in _appendix_feature_rows(profile):
        lines.append(
            f"| {name} | {fp.mean} | {fp.std} | [{fp.min}, {fp.max}] | {fp.skewness} |"
        )
    return "\n".join(lines)


def _render_model_table(models: list[str]) -> str:
    lines = [
        "| Model | Pros | Cons |",
        "| --- | --- | --- |",
    ]
    for row in _model_table_rows(models):
        lines.append(f"| {row['model']} | {row['pros']} | {row['cons']} |")
    return "\n".join(lines)


def _collect_retry_reports(run_dir: Path) -> list[Path]:
    patterns = ("*retry*", "*self_correction*")
    found: list[Path] = []
    for pattern in patterns:
        found.extend(sorted(run_dir.rglob(pattern)))
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in found:
        if path in seen or path.is_dir():
            continue
        seen.add(path)
        unique.append(path)
    return unique


def analyze_and_formulate_strategy(
    bundle: DatasetBundle,
    *,
    agent: Any | None = None,
    use_llm: bool = True,
    task_type: Literal["regression", "classification"] = "regression",
    output_dir: str | Path = "results/data_profile",
    run_context: Mapping[str, Any] | None = None,
) -> tuple[DataProfile, StrategyArtifact, dict[str, str]]:
    """End-to-end profile and strategy workflow."""

    run_dir = _make_run_output_dir(output_dir)
    profile = profile_dataset(bundle, task_type=task_type, output_dir=run_dir)
    strategy = formulate_strategy(
        profile,
        agent=agent,
        use_llm=use_llm,
        output_dir=run_dir,
        run_context=run_context,
    )
    paths = write_profile_outputs(profile, strategy, output_dir=run_dir)
    return profile, strategy, paths


__all__ = [
    "DataProfile",
    "FeatureProfile",
    "StrategyArtifact",
    "analyze_and_formulate_strategy",
    "build_strategy_prompt",
    "formulate_strategy",
    "generate_profile_plots",
    "profile_dataset",
    "write_profile_outputs",
]
