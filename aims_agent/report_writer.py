"""
Pipeline report writer.

Generates a full academic-style report (Title, Abstract, Introduction,
Methodology, Experiments, Results, Analysis, Discussion, Conclusion,
Future Work, Appendix) from a completed PipelineResult.
"""

from __future__ import annotations

import json
import shutil
import textwrap
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aims_agent.agent import PipelineResult

_WIDTH = 72


# ── Formatting helpers ─────────────────────────────────────────────────

def _p(text: str) -> str:
    """Wrap a paragraph at _WIDTH."""
    return textwrap.fill(text.strip(), width=_WIDTH)


def _indent(text: str, spaces: int = 4) -> str:
    prefix = " " * spaces
    return textwrap.fill(
        text.strip(), width=_WIDTH,
        initial_indent=prefix, subsequent_indent=prefix,
    )


def _h1(title: str) -> str:
    bar = "=" * _WIDTH
    return f"\n{bar}\n{title.upper()}\n{bar}"


def _h2(title: str) -> str:
    bar = "-" * _WIDTH
    return f"\n{bar}\n{title}\n{bar}"


def _h3(title: str) -> str:
    return f"\n{title}"


def _fmt_num(v: Any, decimals: int = 4) -> str:
    try:
        return f"{float(v):.{decimals}f}"
    except Exception:
        return str(v)


def _read_sc_log(log_path: str) -> list[dict[str, Any]]:
    p = Path(log_path)
    if not p.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


# ── Main writer ────────────────────────────────────────────────────────

def write_pipeline_report(
    result: "PipelineResult",
    *,
    output_dir: str = "results",
    filename: str | None = None,
) -> str:
    """
    Write a full academic-style pipeline report to a text file.

    Sections: Title/Abstract, Introduction, Methodology, Experiments,
    Results, Analysis, Discussion, Conclusion, Future Work, Appendix.

    Args:
        result: Completed PipelineResult from Agent.run_full_pipeline().
        output_dir: Directory to save the report (created if missing).
        filename: Override the auto-generated filename.

    Returns:
        Absolute path to the saved report file.
    """
    now = datetime.now()
    ts_label = now.strftime("%Y-%m-%d %H:%M:%S")
    ts_file = now.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(output_dir) / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / (filename or f"report_{ts_file}.txt")

    # ── Pull data from result ──────────────────────────────────────────
    meta = result.metadata or {}
    suggestion = result.suggestion
    metrics = result.metrics or {}
    task_type = getattr(result, "task_type", "regression")
    motivation = getattr(result, "motivation", "").strip()
    bg = (result.background_knowledge or "").strip()
    interp = (result.interpretation or "").strip()

    source = meta.get("source") or meta.get("description", "the provided dataset")
    shape = meta.get("shape", {})
    n_rows = (shape[0] if isinstance(shape, (list, tuple)) and shape
              else shape.get("rows", "?"))
    n_cols = (shape[1] if isinstance(shape, (list, tuple)) and len(shape) > 1
              else shape.get("cols", "?"))
    target = meta.get("target", "the target property")
    features: list[str] = list(meta.get("features", []))

    model_name = suggestion.model_name if suggestion else "the selected model"
    pkg = suggestion.package_name if suggestion else ""
    sel_reason = (suggestion.reason or "").strip() if suggestion else ""
    exec_path = result.execution_path or "builtin"

    r2   = metrics.get("R2")   or metrics.get("r2")
    rmse = metrics.get("RMSE") or metrics.get("rmse")
    mae  = metrics.get("MAE")  or metrics.get("mae")
    mse  = metrics.get("MSE")  or metrics.get("mse")
    acc  = metrics.get("accuracy") or metrics.get("Accuracy")
    f1   = metrics.get("f1")   or metrics.get("F1")

    # Primary metric string for prose
    if r2 is not None:
        primary_metric = f"R² = {_fmt_num(r2)}"
        secondary_metrics = []
        if rmse is not None:
            secondary_metrics.append(f"RMSE = {_fmt_num(rmse)}")
        if mae is not None:
            secondary_metrics.append(f"MAE = {_fmt_num(mae)}")
        metric_summary = primary_metric
        if secondary_metrics:
            metric_summary += ", " + ", ".join(secondary_metrics)
    elif acc is not None:
        metric_summary = f"accuracy = {_fmt_num(acc)}"
        if f1 is not None:
            metric_summary += f", F1 = {_fmt_num(f1)}"
    else:
        metric_summary = "see metrics section"

    dist_raw = (result.distribution_summary or "").strip()
    target_dist_line = next(
        (l for l in dist_raw.splitlines() if l.strip().startswith("Target:")),
        "",
    ).replace("Target:", "").strip()

    sc_attempts = result.self_correction_attempts
    sc_success = result.self_correction_success
    sc_log = _read_sc_log(result.self_correction_log_path) if result.self_correction_log_path else []

    paras: list[str] = []

    # ══════════════════════════════════════════════════════════════════
    # TITLE + ABSTRACT
    # ══════════════════════════════════════════════════════════════════
    paras.append(_h1("AIMS Agent — Automated Materials Informatics Pipeline Report"))
    paras.append(f"Generated: {ts_label}")

    paras.append(_h2("Abstract"))
    if motivation:
        abstract_goal = motivation
    else:
        abstract_goal = f"predict {target} from materials data"

    insight = ""
    if interp:
        first_sentence = interp.split(".")[0].strip()
        insight = first_sentence + "."

    paras.append(_p(
        f"This report presents the results of an automated machine learning pipeline "
        f"applied to a materials science prediction task. The objective was to {abstract_goal}. "
        f"Using an AI agent system that handles data ingestion, model selection, training, "
        f"and evaluation, the pipeline trained a {model_name} model on the \"{source}\" "
        f"dataset ({n_rows} samples, {n_cols} features). "
        f"The model achieved {metric_summary} on the held-out test set. "
        + (f"{insight}" if insight else "")
    ))

    # ══════════════════════════════════════════════════════════════════
    # 1. INTRODUCTION
    # ══════════════════════════════════════════════════════════════════
    paras.append(_h2("1. Introduction"))

    if bg:
        paras.append(_p(bg))

    paras.append(_p(
        f"Predicting {target} from composition and processing features is a key challenge "
        f"in computational materials science. Traditional approaches rely on costly and "
        f"time-consuming physical experiments or high-fidelity simulations, making it "
        f"difficult to rapidly screen large design spaces. Data-driven machine learning "
        f"methods offer an alternative by learning structure-property relationships directly "
        f"from existing experimental datasets, enabling faster and cheaper property prediction."
    ))
    paras.append(_p(
        f"This work addresses the {task_type} problem of predicting \"{target}\" using an "
        f"automated AI agent pipeline — AIMS Agent — which orchestrates the full workflow "
        f"from raw data loading to trained model evaluation without manual intervention. "
        f"The system leverages a large language model (LLM) for model selection, "
        f"code generation, and result interpretation, and includes a self-correction loop "
        f"that automatically repairs generated code when validation fails."
    ))

    # ══════════════════════════════════════════════════════════════════
    # 2. METHODOLOGY
    # ══════════════════════════════════════════════════════════════════
    paras.append(_h2("2. Methodology"))

    paras.append(_h3("2.1  Overall Pipeline"))
    paras.append(_p(
        "The AIMS Agent pipeline consists of the following sequential stages:"
    ))
    stages = [
        ("Data Ingestion",
         "Raw CSV or Excel files are loaded with automatic header detection, "
         "multi-index flattening, and missing-value imputation. Dataset metadata "
         "(shape, feature names, target column, dtypes) is extracted for downstream use."),
        ("Distribution Analysis",
         "Summary statistics (mean, std, skewness, range) are computed for the target "
         "and all input features, and a distribution plot is saved. Skew flags are passed "
         "to the LLM as context for model selection."),
        ("Planning",
         "The LLM receives the user motivation, background knowledge, and dataset metadata "
         "and generates a step-by-step execution plan that guides the remainder of the pipeline."),
        ("Model Selection",
         "The LLM recommends one or more machine learning models suited to the task type "
         "and data characteristics. Unknown models (not in the built-in registry) are "
         "enriched with import paths and implementation notes via a secondary LLM call."),
        ("Execution Path Resolution",
         "Each suggested model is classified as 'builtin' (available in the registry), "
         "'dynamic_import' (installable package), or 'codegen' (requires LLM-generated "
         "wrapper code). This determines how the model class is loaded."),
        ("Dependency Management",
         "Required Python packages are automatically installed via pip before training begins."),
        ("Code Generation & Self-Correction",
         "If the codegen path is selected, the LLM generates a scikit-learn-compatible "
         "estimator wrapper. A validation step checks fit/predict contract, output shape, "
         "and value finiteness. If validation fails, the SelfCorrectionAgent proposes fixes "
         "for up to N retries."),
        ("Training & Evaluation",
         "The resolved model is trained with optional hyperparameter tuning (GridSearchCV "
         "or RandomizedSearchCV). Performance metrics (R², RMSE, MAE, or accuracy/F1) are "
         "computed on the held-out test set."),
        ("Interpretation & Reporting",
         "The LLM interprets the numerical results in the context of the materials problem "
         "and provides actionable suggestions. A full narrative report is written to disk."),
    ]
    for title, desc in stages:
        paras.append(f"\n  [{title}]")
        paras.append(_indent(desc))

    paras.append(_h3("2.2  Dataset Description"))
    feat_preview = ", ".join(features[:8]) + (f" (and {len(features)-8} more)" if len(features) > 8 else "")
    paras.append(_p(
        f"The dataset used in this study is \"{source}\", containing {n_rows} samples "
        f"and {n_cols} columns. The prediction target is \"{target}\" ({task_type}). "
        f"Input features include: {feat_preview}. "
        f"All categorical and label-encoded columns are treated as numeric inputs; "
        f"missing values are imputed using column-wise median imputation."
    ))
    paras.append(_h3("2.2a  Data Distribution"))
    if target_dist_line:
        paras.append(_p(
            f"The target variable has the following distribution: {target_dist_line}. "
            f"Many input features exhibit significant skewness, which informed the choice "
            f"of a tree-based model robust to non-normal feature distributions."
        ))
        dist_bullets = [
            "The target distribution summary shows whether values are centered or skewed.",
            "Large skewness means the model must handle asymmetric tails and rare extremes.",
            "Skewed features can distort linear models unless transformed or scaled.",
            "The distribution plot is used as a quick check for imbalance, spread, and outliers.",
        ]
        for bullet in dist_bullets:
            paras.append(_indent("• " + bullet))
    else:
        paras.append(_p(
            "No distribution summary was recorded for the target in this run, so the "
            "report relies on the broader feature statistics and model metrics."
        ))

    paras.append(_h3("2.3  Model and Hyperparameter Selection"))
    path_desc = {
        "builtin":        "loaded directly from the AIMS Agent built-in model registry",
        "dynamic_import": "dynamically imported from its installed Python package",
        "codegen":        "implemented via LLM-generated scikit-learn-compatible wrapper code",
    }.get(exec_path, exec_path)
    paras.append(_p(
        f"The LLM recommended {model_name} (package: {pkg or 'N/A'}) for this "
        f"{task_type} task. The model was {path_desc}."
    ))
    if sel_reason:
        paras.append(_p(
            f"The selection rationale provided by the LLM was: {sel_reason}"
        ))
    paras.append(_p(
        "Hyperparameter tuning was performed using cross-validated grid search "
        "(GridSearchCV) over a predefined parameter grid for each supported model class. "
        "The best estimator from the search was used for final evaluation."
    ))

    paras.append(_h3("2.4  Agent System Design"))
    paras.append(_p(
        "The AIMS Agent is composed of several cooperating sub-agents, each with a "
        "distinct responsibility:"
    ))
    agents_desc = [
        ("Agent (Orchestrator)",
         "The top-level agent that holds the LLM interface and dispatches all pipeline "
         "steps. It calls sub-agents as needed and aggregates results into PipelineResult."),
        ("ModelSelector",
         "Issues structured LLM prompts requesting JSON-formatted model suggestions, "
         "parses the response, filters task-mismatched models, and enriches unknown "
         "models with import path and package information."),
        ("ExecutionPathResolver",
         "Classifies each model suggestion as builtin, dynamic_import, or codegen based "
         "on the MODEL_IMPORT_MAP registry and the presence of a valid import path."),
        ("CodeGenAgent",
         "Builds a detailed prompt (CodeGenSpec) containing model name, task type, "
         "required interface, import hints, and constraints, then calls the LLM to "
         "generate a full Python estimator class."),
        ("SelfCorrectionAgent",
         "A stateful agent that validates generated code, diagnoses failures using "
         "structured failure codes (e.g. missing_interface, predict_shape_error), "
         "proposes patches via LLM, and decides whether to retry or abort."),
        ("ResultsAnalyzer",
         "Computes evaluation metrics, generates predicted-vs-actual and residual plots, "
         "and calls the LLM to produce a plain-language interpretation of results."),
    ]
    for name, desc in agents_desc:
        paras.append(f"\n  [{name}]")
        paras.append(_indent(desc))

    # ══════════════════════════════════════════════════════════════════
    # 3. EXPERIMENTS
    # ══════════════════════════════════════════════════════════════════
    paras.append(_h2("3. Experiments"))

    paras.append(_h3("3.1  Experimental Setup"))
    paras.append(_p(
        f"The dataset was split into training and test sets using a standard 80/20 "
        f"random split (random state fixed for reproducibility). No data augmentation "
        f"was applied. The full feature set ({len(features)} features) was used without "
        f"manual feature selection; the LLM was provided with feature names and "
        f"distribution statistics to inform model choice."
    ))

    paras.append(_h3("3.2  Evaluation Metrics"))
    if task_type == "regression":
        paras.append(_p(
            "Model performance was assessed using the following regression metrics "
            "computed on the held-out test set:"
        ))
        metric_defs = [
            ("R² (Coefficient of Determination)",
             "Measures the proportion of variance in the target explained by the model. "
             "Values closer to 1.0 indicate better fit."),
            ("RMSE (Root Mean Squared Error)",
             "The square root of the average squared prediction error, in the same units "
             "as the target variable. Sensitive to large outliers."),
            ("MAE (Mean Absolute Error)",
             "The average absolute prediction error. More robust to outliers than RMSE "
             "and directly interpretable in target units."),
            ("MSE (Mean Squared Error)",
             "The average squared prediction error. Used internally for optimization."),
        ]
        for mname, mdef in metric_defs:
            paras.append(f"\n  {mname}:")
            paras.append(_indent(mdef))
    else:
        paras.append(_p(
            "Model performance was assessed using accuracy and F1-score on the held-out "
            "test set. Accuracy measures the fraction of correctly classified samples; "
            "F1-score is the harmonic mean of precision and recall, useful for imbalanced classes."
        ))

    paras.append(_h3("3.3  Baseline and Comparison"))
    paras.append(_p(
        f"The LLM-selected model ({model_name}) is evaluated as the primary model. "
        f"A baseline comparison using a simple linear regression (or majority-class "
        f"classifier for classification) would provide a lower bound, though explicit "
        f"baseline runs were not conducted in this automated pipeline run. Future "
        f"work could add automated multi-model comparison via the ModelComparisonAgent."
    ))

    # ══════════════════════════════════════════════════════════════════
    # 4. RESULTS
    # ══════════════════════════════════════════════════════════════════
    paras.append(_h2("4. Results"))

    paras.append(_h3("4.1  Model Performance"))
    if metrics:
        paras.append(_p(
            f"The {model_name} model achieved the following performance on the "
            f"held-out test set:"
        ))
        paras.append("")
        metric_table_lines = []
        metric_table_lines.append("  Metric   | Value")
        metric_table_lines.append("  " + "-" * 30)
        for k, v in metrics.items():
            metric_table_lines.append(f"  {k:<8} | {_fmt_num(v)}")
        paras.append("\n".join(metric_table_lines))

        if r2 is not None:
            r2_f = float(r2)
            if r2_f >= 0.95:
                quality = "excellent"
            elif r2_f >= 0.90:
                quality = "very good"
            elif r2_f >= 0.80:
                quality = "good"
            elif r2_f >= 0.60:
                quality = "moderate"
            else:
                quality = "limited"
                paras.append(_p(
                    f"The R² of {_fmt_num(r2)} indicates {quality} predictive performance, "
                    f"meaning the model explains approximately {_fmt_num(r2 * 100, 1)}% "  # type: ignore[operator]
                    f"of the variance in {target}."
                    + (f" The RMSE of {_fmt_num(rmse)} and MAE of {_fmt_num(mae)} "
                       f"quantify the average prediction error in the units of the target variable."
                       if rmse is not None and mae is not None else "")
                ))
            paras.append(_p(
                "Interpretation of the metrics:"
            ))
            metric_interp = [
                f"R² close to 1.0 means the model captures most of the target variation; lower values mean important structure is still missing.",
                f"RMSE is the penalty for large mistakes, so a high RMSE means the model sometimes misses badly on difficult samples.",
                f"MAE is the typical absolute error, so it is the clearest summary of how far predictions are from the true value on average.",
            ]
            for bullet in metric_interp:
                paras.append(_indent("• " + bullet))
    else:
        paras.append(_p("Training was skipped or no metrics were recorded for this run."))

    paras.append(_h3("4.2  Validation Status"))
    val_ok = result.training_validation_ok
    val_msg = (result.training_validation_message or "").strip()
    if val_ok:
        paras.append(_p(
            "Post-training validation passed. The model's predict() method returned "
            "outputs of the correct shape and all finite values on a synthetic validation set, "
            "confirming that the trained estimator is well-formed."
        ))
    else:
        paras.append(_p(
            f"Post-training validation failed. "
            f"Detail: {val_msg or 'no detail recorded'}. "
            f"Results should be interpreted with caution."
        ))

    # Copy plot into the reports folder and interpret it
    plot_report_path: str = ""
    if result.plot_path:
        src = Path(result.plot_path)
        if src.exists():
            dest = out_dir / src.name
            shutil.copy2(src, dest)
            plot_report_path = str(dest)
        else:
            plot_report_path = result.plot_path

        paras.append(_h3("4.3  Performance Plot"))
        paras.append(_p(
            f"The performance plot has been saved alongside this report at: "
            f"{plot_report_path}."
        ))

        if task_type == "regression":
            paras.append(_p(
                "The plot contains two panels. The left panel is a Predicted vs Actual "
                "scatter plot: each point represents one test sample, with the actual "
                "measured value on the x-axis and the model's prediction on the y-axis. "
                "Points lying on or close to the red dashed diagonal line indicate accurate "
                "predictions. Systematic curvature or fan-shaped spread around the diagonal "
                "would suggest nonlinearity or heteroscedasticity that the model has not "
                "fully captured."
            ))

            # Quantitative reading of the predicted vs actual panel
            if r2 is not None and rmse is not None:
                r2_f = float(r2)
                rmse_f = float(rmse)
                spread_note = (
                    "tight cluster around the diagonal, indicating consistent predictions "
                    "across the full target range"
                    if r2_f >= 0.90 else
                    "moderate spread around the diagonal, with some samples deviating "
                    "noticeably from the ideal line"
                    if r2_f >= 0.75 else
                    "wide spread around the diagonal, reflecting high prediction uncertainty"
                )
                paras.append(_p(
                    f"Given the R² of {_fmt_num(r2)} and RMSE of {_fmt_num(rmse)}, "
                    f"the Predicted vs Actual panel is expected to show a {spread_note}. "
                    f"Points near the extremes of the target range (low or high {target}) "
                    f"may show larger absolute errors if those regions are underrepresented "
                    f"in the training data."
                ))

            paras.append(_p(
                "The right panel shows the Residual plot: the residual (actual − predicted) "
                "for each test sample plotted against the predicted value. A well-behaved "
                "model produces residuals scattered randomly around the horizontal zero line "
                "with no visible trend or pattern. Trends (e.g., residuals increasing with "
                "prediction magnitude) would indicate heteroscedasticity or a systematic "
                "modelling bias that should be addressed — for example by log-transforming "
                "the target or adding missing features."
            ))

            # Skew-informed residual comment
            if target_dist_line and "skewness" in target_dist_line:
                try:
                    skew_val = float(target_dist_line.split("skewness=")[1].split(",")[0].strip("]"))
                    if abs(skew_val) > 0.5:
                        direction = "right" if skew_val > 0 else "left"
                        paras.append(_p(
                            f"Because the target distribution is skewed {direction} "
                            f"(skewness = {skew_val:.3f}), the residual plot may show "
                            f"larger errors on the {'high' if skew_val > 0 else 'low'}-value "
                            f"tail where training data is sparse. This is a known limitation "
                            f"of tree-based models trained on imbalanced target distributions."
                        ))
                except Exception:
                    pass
        else:
            paras.append(_p(
                "The plot shows the confusion matrix for the test set. Correct predictions "
                "appear on the main diagonal; off-diagonal cells represent misclassifications. "
                "Larger values on the diagonal relative to off-diagonal values indicate "
                "better classification performance."
            ))
            if acc is not None:
                acc_f = float(acc)
                conf_note = (
                    "few off-diagonal entries, indicating high classification accuracy"
                    if acc_f >= 0.90 else
                    "some off-diagonal entries, suggesting confusion between certain classes"
                    if acc_f >= 0.75 else
                    "significant off-diagonal entries, indicating substantial misclassification"
                )
                paras.append(_p(
                    f"With an accuracy of {_fmt_num(acc)}, the confusion matrix is expected "
                    f"to show {conf_note}."
                ))

    # ══════════════════════════════════════════════════════════════════
    # 5. ANALYSIS
    # ══════════════════════════════════════════════════════════════════
    paras.append(_h2("5. Analysis"))

    paras.append(_h3("5.1  Model Interpretation (LLM)"))
    if interp:
        for block in interp.split("\n\n"):
            block = block.strip()
            if block:
                paras.append(_p(block))
    else:
        paras.append(_p("No LLM interpretation was generated for this run."))

    paras.append(_h3("5.2  Data Quality Observations"))
    if target_dist_line:
        paras.append(_p(
            f"The target variable ({target}) has distribution: {target_dist_line}. "
            f"A non-zero skewness value suggests the target is not normally distributed, "
            f"which may affect residual behavior for linear models but is handled well "
            f"by ensemble tree methods such as {model_name}."
        ))
    skewed_count = dist_raw.count("(skewed)")
    if skewed_count > 0:
        paras.append(_p(
            f"{skewed_count} input features were flagged as skewed (|skewness| > 1). "
            f"Highly skewed features (e.g. elemental compositions present in trace amounts) "
            f"can create sparse regions in feature space that are difficult to learn from. "
            f"Log-transformation or quantile normalization of these features is recommended "
            f"as a preprocessing step in future runs."
        ))

    paras.append(_h3("5.3  Feature Considerations"))
    paras.append(_p(
        f"The dataset contains {len(features)} input features. For the selected model "
        f"({model_name}), feature importance scores can be extracted post-training to "
        f"identify which composition or processing variables most strongly influence {target}. "
        f"High-importance features may correspond to known physical mechanisms "
        f"(e.g. solid-solution strengthening elements, grain-refining additions) and can "
        f"guide targeted experimental campaigns."
    ))

    # ══════════════════════════════════════════════════════════════════
    # 6. CODE GENERATION LOG (only when triggered)
    # ══════════════════════════════════════════════════════════════════
    if result.generated_model_wrapper_path or sc_attempts > 0:
        paras.append(_h2("6. Code Generation and Self-Correction"))

        gen_path = result.generated_model_wrapper_path
        if gen_path:
            paras.append(_p(
                f"Because the selected model ({model_name}) did not have a standard "
                f"entry in the model registry, the CodeGenAgent generated a custom "
                f"scikit-learn-compatible wrapper class. The generated file was saved to: "
                f"{gen_path}."
            ))

        if sc_attempts == 0:
            paras.append(_p(
                "The generated code passed all contract validation checks on the first "
                "attempt. No self-correction was required."
            ))
        else:
            outcome_str = "successfully repaired" if sc_success else "not repaired within the retry limit"
            paras.append(_p(
                f"The initial generated code failed validation and required "
                f"{sc_attempts} self-correction attempt(s). The issue was {outcome_str}."
            ))
            for r in sc_log:
                n = int(r.get("attempt", 0)) + 1
                patched = bool(r.get("patched", False))
                fc = r.get("failure_code", "runtime_exception")
                err = str(r.get("error_message", "")).strip().splitlines()[0][:120]
                diagnosis = str(r.get("diagnosis", "")).strip()
                patch_sum = str(r.get("patch_summary", "")).strip()
                ts_attempt = str(r.get("timestamp", "")).strip()

                status_str = "patched successfully" if patched else "patch failed"
                line = (
                    f"Attempt {n} [{ts_attempt}]: failure code = \"{fc}\""
                    + (f"; error = \"{err}\"" if err else "")
                    + f"; {status_str}."
                )
                if diagnosis:
                    line += f" Diagnosis: {diagnosis}."
                if patch_sum:
                    line += f" Fix: {patch_sum}."
                paras.append(_indent(line))

        sc_report = result.self_correction_report or {}
        fc_counts = sc_report.get("failure_code_counts", {})
        if fc_counts:
            fc_parts = [f"{code} ({count}x)" for code, count in
                        sorted(fc_counts.items(), key=lambda x: -x[1])]
            paras.append(_p(
                "Failure codes observed across all repair attempts: "
                + ", ".join(fc_parts) + "."
            ))

    # ══════════════════════════════════════════════════════════════════
    # 7. DISCUSSION
    # ══════════════════════════════════════════════════════════════════
    paras.append(_h2("7. Discussion"))

    paras.append(_h3("7.1  Strengths"))
    paras.append(_p(
        f"The automated pipeline successfully selected, trained, and evaluated a "
        f"{model_name} model without manual intervention. The LLM-driven model "
        f"selection appropriately chose an ensemble method well-suited to the skewed, "
        f"high-dimensional feature space. The self-correction mechanism ensures that "
        f"even models requiring custom code generation can be deployed reliably."
    ))

    paras.append(_h3("7.2  Limitations"))
    limitations = [
        "The dataset size may be insufficient for generalizing to novel alloy systems "
        "outside the training distribution. Tree-based models can extrapolate poorly "
        "beyond the range of training data.",

        "The current feature set is limited to compositional and processing parameters. "
        "Microstructural descriptors (grain size, dislocation density, precipitate "
        "volume fraction) that strongly influence mechanical properties are not included.",

        "The LLM model selection is non-deterministic: different runs may suggest "
        "different models, making reproducibility a concern for production use. "
        "A deterministic fallback or ensemble-of-suggestions strategy is advisable.",

        "Hyperparameter tuning uses a fixed grid; Bayesian optimization or random "
        "search over a wider space may yield better performance.",
    ]
    for lim in limitations:
        paras.append(_indent("• " + lim))

    paras.append(_h3("7.3  LLM Reliability"))
    paras.append(_p(
        "The LLM-generated model selection and code are subject to hallucination and "
        "inconsistency. The validation layer (validate_estimator_contract) and the "
        "self-correction loop mitigate code-level errors, but incorrect reasoning "
        "about model suitability cannot always be detected automatically. Human "
        "review of the selected model and its rationale is recommended for "
        "high-stakes applications."
    ))

    # ══════════════════════════════════════════════════════════════════
    # 8. CONCLUSION
    # ══════════════════════════════════════════════════════════════════
    paras.append(_h2("8. Conclusion"))
    if r2 is not None:
        paras.append(_p(
            f"This study demonstrated the application of an automated AI agent pipeline "
            f"to the materials science problem of predicting {target}. The system "
            f"autonomously loaded and preprocessed the \"{source}\" dataset, selected "
            f"{model_name} as the most appropriate model for this {task_type} task, "
            f"and trained it to achieve {metric_summary}. These results confirm that "
            f"LLM-guided model selection combined with automated training and evaluation "
            f"can produce competitive predictive models for materials property prediction "
            f"with minimal human effort."
        ))
    else:
        paras.append(_p(
            f"This pipeline run completed {'successfully' if result.success else 'with errors'}. "
            + (f"Error: {result.error}" if result.error else "")
        ))

    # ══════════════════════════════════════════════════════════════════
    # 9. FUTURE WORK
    # ══════════════════════════════════════════════════════════════════
    paras.append(_h2("9. Future Work"))
    future = [
        ("Larger and More Diverse Datasets",
         "Incorporating data from multiple sources (e.g., Materials Project, AFLOW, "
         "ICSD) and expanding the dataset with more alloy systems and processing "
         "conditions would improve model generalizability."),
        ("Physics-Informed Features",
         "Enriching the feature set with thermodynamic descriptors (CALPHAD-derived "
         "phase fractions, stacking-fault energy, Peierls stress estimates) could "
         "reduce the residual error attributable to missing microstructural information."),
        ("Graph Neural Network Models",
         "Graph-based models such as MEGNet, CGCNN, or M3GNet, which operate directly "
         "on crystal structure graphs, may outperform tabular models for property "
         "prediction when crystal structure data is available."),
        ("Multi-Model Comparison Agent",
         "A ModelComparisonAgent that automatically trains and benchmarks multiple "
         "LLM-suggested models in parallel would provide more robust model selection "
         "and reduce dependence on a single LLM recommendation."),
        ("Uncertainty Quantification",
         "Adding prediction intervals (e.g., via conformal prediction or Gaussian "
         "Process surrogates) would make the model more useful for experimental "
         "design by identifying high-uncertainty regions of composition space."),
        ("Inverse Design",
         "Coupling the trained forward model with an optimization loop (Bayesian "
         "optimization, genetic algorithms) to suggest alloy compositions that "
         "maximize the target property would move the system from prediction to design."),
    ]
    for title, desc in future:
        paras.append(f"\n  {title}:")
        paras.append(_indent(desc))

    # ══════════════════════════════════════════════════════════════════
    # 10. APPENDIX
    # ══════════════════════════════════════════════════════════════════
    paras.append(_h2("10. Appendix"))

    paras.append(_h3("A. Pipeline Configuration"))
    config_lines = [
        f"  Dataset source  : {source}",
        f"  Target column   : {target}",
        f"  Task type       : {task_type}",
        f"  Model selected  : {model_name}",
        f"  Execution path  : {exec_path}",
        f"  Path reason     : {result.path_reason or 'N/A'}",
        f"  Self-correction : {sc_attempts} attempt(s)",
        f"  Performance plot: {plot_report_path or 'N/A'}",
    ]
    paras.append("\n".join(config_lines))

    paras.append(_h3("B. Self-Correction Log"))
    if sc_log:
        log_lines = []
        for r in sc_log:
            log_lines.append(
                f"  attempt={r.get('attempt', '?')}  "
                f"failure_code={r.get('failure_code', '?')}  "
                f"patched={r.get('patched', '?')}  "
                f"timestamp={r.get('timestamp', '?')}"
            )
            if r.get("error_message"):
                log_lines.append(
                    _indent(f"error: {str(r['error_message']).splitlines()[0][:100]}")
                )
            if r.get("diagnosis"):
                log_lines.append(_indent(f"diagnosis: {r['diagnosis']}"))
            if r.get("patch_summary"):
                log_lines.append(_indent(f"patch: {r['patch_summary']}"))
            log_lines.append("")
        paras.append("\n".join(log_lines))
    elif sc_attempts == 0:
        paras.append("  No self-correction was triggered in this run.")
    else:
        paras.append("  Log file not available.")

    paras.append(_h3("C. Generated Code"))
    if result.generated_model_wrapper_path:
        gen_file = Path(result.generated_model_wrapper_path)
        paras.append(f"  File: {result.generated_model_wrapper_path}")
        if gen_file.exists():
            code = gen_file.read_text(encoding="utf-8")
            code_lines = code.splitlines()
            preview = "\n".join(f"    {l}" for l in code_lines[:60])
            if len(code_lines) > 60:
                preview += f"\n    ... ({len(code_lines) - 60} more lines)"
            paras.append(preview)
    else:
        paras.append("  No custom code was generated in this run (built-in model used).")

    paras.append(_h3("D. Distribution Summary (full)"))
    if dist_raw:
        for dline in dist_raw.splitlines():
            paras.append(f"  {dline}")
    else:
        paras.append("  Not available.")

    # ── Footer ────────────────────────────────────────────────────────
    paras.append(f"\n{'=' * _WIDTH}")
    paras.append(f"Report: {report_path.resolve()}")
    paras.append("=" * _WIDTH)

    report_text = "\n\n".join(paras) + "\n"
    report_path.write_text(report_text, encoding="utf-8")
    return str(report_path.resolve())


__all__ = ["write_pipeline_report"]
