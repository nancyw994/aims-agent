"""
Intelligent model selection via LLM.

Takes processed features and target as input, queries the Agent (LLM) to suggest
a suitable ML model and Python package from standard libraries. Requires the LLM
to return JSON with model name, package, import path, and reason.
"""

from __future__ import annotations

import importlib
import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from aims_agent.agent import Agent

from aims_agent.dependency_manager import ensure_package_installed


@dataclass
class ModelSuggestion:
    """Holds the LLM's recommended model information."""

    model_name: str
    package_name: str
    import_path: str
    reason: str
    
    # Uncertainty quantification metadata
    uq_capability: str = "none"  # "native", "ensemble", "calibration", "none"
    uq_quality: float = 0.0      # 0.0-1.0, expected UQ calibration quality
    heteroscedastic: bool = False  # Supports input-dependent uncertainty

DEFAULT_SUGGESTION = ModelSuggestion(
    model_name="RandomForestRegressor",
    package_name="scikit-learn",
    import_path="sklearn.ensemble.RandomForestRegressor",
    reason="Default fallback: random forest works well for most regression tasks",
    uq_capability="ensemble",
    uq_quality=0.7,
    heteroscedastic=False,
)

DEFAULT_CLASSIFICATION_SUGGESTION = ModelSuggestion(
    model_name="RandomForestClassifier",
    package_name="scikit-learn",
    import_path="sklearn.ensemble.RandomForestClassifier",
    reason="Default fallback: random forest works well for multi-class classification",
    uq_capability="ensemble",
    uq_quality=0.7,
    heteroscedastic=False,
)


def get_default_suggestion(task_type: str) -> "ModelSuggestion":
    """Return a default model suggestion without calling the LLM (for --no-llm mode)."""
    if task_type == "classification":
        return DEFAULT_CLASSIFICATION_SUGGESTION
    return DEFAULT_SUGGESTION


def enrich_suggestion_with_uq_metadata(suggestion: ModelSuggestion) -> ModelSuggestion:
    """
    Enrich a model suggestion with UQ capability metadata from MODEL_UQ_CAPABILITY.
    
    If the model is not in the capability map, returns the suggestion unchanged.
    """
    if suggestion.model_name in MODEL_UQ_CAPABILITY:
        meta = MODEL_UQ_CAPABILITY[suggestion.model_name]
        suggestion.uq_capability = meta["uq_capability"]
        suggestion.uq_quality = meta["uq_quality"]
        suggestion.heteroscedastic = meta["heteroscedastic"]
    
    return suggestion


def select_uq_aware_model(
    n_samples: int,
    n_features: int,
    task_type: str = "regression",
    use_case: str = "exploration",
    target_multimodal: bool = False,
    needs_interpretability: bool = False,
) -> ModelSuggestion:
    """
    Intelligent UQ-aware model selection based on data characteristics and use case.
    
    Args:
        n_samples: Number of training samples
        n_features: Number of features
        task_type: "regression" or "classification"
        use_case: "exploration", "screening", "active_learning", "production"
        target_multimodal: Whether target distribution is multimodal
        needs_interpretability: Whether model interpretability is required
    
    Returns:
        ModelSuggestion with UQ metadata
    """
    
    # High-UQ use cases (screening, active learning)
    if use_case in ["screening", "active_learning"]:
        
        # Small datasets → Gaussian Process (gold standard)
        if n_samples < 1000:
            model_name = "GaussianProcessRegressor" if task_type == "regression" else "GaussianProcessClassifier"
            
        # Medium datasets → Probabilistic NN
        elif 1000 <= n_samples <= 50000:
            if task_type == "regression":
                model_name = "ProbabilisticNN"
            else:
                # For classification, fall back to ensemble
                model_name = "RandomForestClassifier"
        
        # Large datasets → Ensemble (scalable)
        else:
            model_name = "RandomForestRegressor" if task_type == "regression" else "RandomForestClassifier"
    
    # Exploration / baseline (balance accuracy + speed)
    elif use_case == "exploration":
        
        if needs_interpretability:
            # Interpretable models
            model_name = "RandomForestRegressor" if task_type == "regression" else "RandomForestClassifier"
        
        elif n_samples < 10000:
            # Medium data → GradientBoosting
            model_name = "GradientBoostingRegressor" if task_type == "regression" else "GradientBoostingClassifier"
        
        else:
            # Large data → XGBoost (fastest)
            model_name = "XGBRegressor" if task_type == "regression" else "XGBClassifier"
    
    # Production (need reliability + speed)
    elif use_case == "production":
        
        if n_samples < 5000:
            model_name = "RandomForestRegressor" if task_type == "regression" else "RandomForestClassifier"
        else:
            model_name = "XGBRegressor" if task_type == "regression" else "XGBClassifier"
    
    # Default fallback
    else:
        model_name = "RandomForestRegressor" if task_type == "regression" else "RandomForestClassifier"
    
    # Build suggestion
    if model_name == "ProbabilisticNN":
        # Custom model
        suggestion = ModelSuggestion(
            model_name="ProbabilisticNN",
            package_name="aims-agent",
            import_path="aims_agent.probabilistic_models.pnn.ProbabilisticNNWrapper",
            reason=f"Probabilistic NN for {n_samples} samples: native distribution prediction with heteroscedastic uncertainty",
            uq_capability="native",
            uq_quality=0.9,
            heteroscedastic=True,
        )
    elif model_name in MODEL_IMPORT_MAP:
        module_path, class_name = MODEL_IMPORT_MAP[model_name]
        package = "scikit-learn"
        if "xgboost" in module_path:
            package = "xgboost"
        elif "lightgbm" in module_path:
            package = "lightgbm"
        elif "catboost" in module_path:
            package = "catboost"
        
        meta = MODEL_UQ_CAPABILITY.get(model_name, {})
        
        suggestion = ModelSuggestion(
            model_name=model_name,
            package_name=package,
            import_path=f"{module_path}.{class_name}",
            reason=meta.get("best_for", f"Selected for {use_case} use case with {n_samples} samples"),
            uq_capability=meta.get("uq_capability", "none"),
            uq_quality=meta.get("uq_quality", 0.0),
            heteroscedastic=meta.get("heteroscedastic", False),
        )
    else:
        # Unknown model, use default
        suggestion = get_default_suggestion(task_type)
    
    return suggestion

# Maps model class name -> (module path, class name) for reliable import path and dynamic load
MODEL_IMPORT_MAP: dict[str, tuple[str, str]] = {
    # ── Probabilistic Models (Native Uncertainty) ─────────────────────
    "ProbabilisticNN": ("aims_agent.probabilistic_models.pnn", "ProbabilisticNNWrapper"),
    "ProbabilisticNNWrapper": ("aims_agent.probabilistic_models.pnn", "ProbabilisticNNWrapper"),
    
    # ── Regression ────────────────────────────────────────────────────
    "RandomForestRegressor": ("sklearn.ensemble", "RandomForestRegressor"),
    "GradientBoostingRegressor": ("sklearn.ensemble", "GradientBoostingRegressor"),
    "DecisionTreeRegressor": ("sklearn.tree", "DecisionTreeRegressor"),
    "ExtraTreesRegressor": ("sklearn.ensemble", "ExtraTreesRegressor"),
    "Ridge": ("sklearn.linear_model", "Ridge"),
    "Lasso": ("sklearn.linear_model", "Lasso"),
    "ElasticNet": ("sklearn.linear_model", "ElasticNet"),
    "LinearRegression": ("sklearn.linear_model", "LinearRegression"),
    "SVR": ("sklearn.svm", "SVR"),
    "SGDRegressor": ("sklearn.linear_model", "SGDRegressor"),
    "KNeighborsRegressor": ("sklearn.neighbors", "KNeighborsRegressor"),
    "MLPRegressor": ("sklearn.neural_network", "MLPRegressor"),
    "AdaBoostRegressor": ("sklearn.ensemble", "AdaBoostRegressor"),
    "BaggingRegressor": ("sklearn.ensemble", "BaggingRegressor"),
    "XGBRegressor": ("xgboost", "XGBRegressor"),
    "LGBMRegressor": ("lightgbm", "LGBMRegressor"),
    "CatBoostRegressor": ("catboost", "CatBoostRegressor"),
    
    # ── Gaussian Process (Native Uncertainty) ─────────────────────────
    "GaussianProcessRegressor": ("sklearn.gaussian_process", "GaussianProcessRegressor"),
    "GaussianProcessClassifier": ("sklearn.gaussian_process", "GaussianProcessClassifier"),
    
    # ── Classification ────────────────────────────────────────────────
    "RandomForestClassifier": ("sklearn.ensemble", "RandomForestClassifier"),
    "GradientBoostingClassifier": ("sklearn.ensemble", "GradientBoostingClassifier"),
    "DecisionTreeClassifier": ("sklearn.tree", "DecisionTreeClassifier"),
    "ExtraTreesClassifier": ("sklearn.ensemble", "ExtraTreesClassifier"),
    "LogisticRegression": ("sklearn.linear_model", "LogisticRegression"),
    "SVC": ("sklearn.svm", "SVC"),
    "SGDClassifier": ("sklearn.linear_model", "SGDClassifier"),
    "KNeighborsClassifier": ("sklearn.neighbors", "KNeighborsClassifier"),
    "MLPClassifier": ("sklearn.neural_network", "MLPClassifier"),
    "AdaBoostClassifier": ("sklearn.ensemble", "AdaBoostClassifier"),
    "BaggingClassifier": ("sklearn.ensemble", "BaggingClassifier"),
    "GaussianNB": ("sklearn.naive_bayes", "GaussianNB"),
    "BernoulliNB": ("sklearn.naive_bayes", "BernoulliNB"),
    "XGBClassifier": ("xgboost", "XGBClassifier"),
    "LGBMClassifier": ("lightgbm", "LGBMClassifier"),
    "CatBoostClassifier": ("catboost", "CatBoostClassifier"),
}

REGRESSION_MODELS = sorted([
    m for m in MODEL_IMPORT_MAP
    if m.endswith("Regressor") or m in ("Ridge", "Lasso", "ElasticNet", "LinearRegression", "SVR", "ProbabilisticNN")
])
CLASSIFICATION_MODELS = sorted([
    m for m in MODEL_IMPORT_MAP
    if m.endswith("Classifier") or m in ("GaussianNB", "BernoulliNB", "SVC", "LogisticRegression")
])


# ============================================================================
# Uncertainty Quantification Capability Metadata
# ============================================================================

MODEL_UQ_CAPABILITY = {
    # ── Native Distribution Predictors (Highest Quality) ──────────────────
    "GaussianProcessRegressor": {
        "uq_capability": "native",
        "uq_quality": 1.0,
        "heteroscedastic": True,
        "best_for": "<5K samples, gold-standard calibrated UQ, interpretable kernel",
        "recommended_use_cases": ["screening", "active_learning", "small_data"],
    },
    "GaussianProcessClassifier": {
        "uq_capability": "native",
        "uq_quality": 0.95,
        "heteroscedastic": True,
        "best_for": "<2K samples, probabilistic classification",
        "recommended_use_cases": ["screening", "active_learning", "small_data"],
    },
    "ProbabilisticNN": {
        "uq_capability": "native",
        "uq_quality": 0.9,
        "heteroscedastic": True,
        "best_for": "500-50K samples, scalable native distribution prediction",
        "recommended_use_cases": ["screening", "active_learning", "production"],
    },
    
    # ── Ensemble Methods (High Quality via Variance) ──────────────────────
    "RandomForestRegressor": {
        "uq_capability": "ensemble",
        "uq_quality": 0.7,
        "heteroscedastic": False,
        "best_for": "General purpose, robust, interpretable feature importance",
        "recommended_use_cases": ["exploration", "baseline", "feature_analysis"],
    },
    "RandomForestClassifier": {
        "uq_capability": "ensemble",
        "uq_quality": 0.7,
        "heteroscedastic": False,
        "best_for": "General purpose classification with uncertainty",
        "recommended_use_cases": ["exploration", "baseline"],
    },
    "GradientBoostingRegressor": {
        "uq_capability": "ensemble",
        "uq_quality": 0.75,
        "heteroscedastic": False,
        "best_for": "High accuracy + reasonable uncertainty via estimators",
        "recommended_use_cases": ["exploration", "baseline"],
    },
    "GradientBoostingClassifier": {
        "uq_capability": "ensemble",
        "uq_quality": 0.75,
        "heteroscedastic": False,
        "best_for": "High accuracy classification with uncertainty",
        "recommended_use_cases": ["exploration", "baseline"],
    },
    "ExtraTreesRegressor": {
        "uq_capability": "ensemble",
        "uq_quality": 0.7,
        "heteroscedastic": False,
        "best_for": "Fast ensemble with uncertainty",
        "recommended_use_cases": ["exploration"],
    },
    "ExtraTreesClassifier": {
        "uq_capability": "ensemble",
        "uq_quality": 0.7,
        "heteroscedastic": False,
        "best_for": "Fast ensemble classification",
        "recommended_use_cases": ["exploration"],
    },
    
    # ── Can Add Calibration (Medium Quality) ──────────────────────────────
    "XGBRegressor": {
        "uq_capability": "calibration",
        "uq_quality": 0.5,
        "heteroscedastic": False,
        "best_for": "Highest accuracy, add uncertainty via conformal prediction",
        "recommended_use_cases": ["exploration", "accuracy_first"],
    },
    "XGBClassifier": {
        "uq_capability": "calibration",
        "uq_quality": 0.5,
        "heteroscedastic": False,
        "best_for": "High accuracy, requires probability calibration",
        "recommended_use_cases": ["exploration", "accuracy_first"],
    },
    "LGBMRegressor": {
        "uq_capability": "calibration",
        "uq_quality": 0.5,
        "heteroscedastic": False,
        "best_for": "Fast training, add uncertainty via calibration",
        "recommended_use_cases": ["exploration", "large_data"],
    },
    "LGBMClassifier": {
        "uq_capability": "calibration",
        "uq_quality": 0.5,
        "heteroscedastic": False,
        "best_for": "Fast classification, requires calibration",
        "recommended_use_cases": ["exploration", "large_data"],
    },
    "CatBoostRegressor": {
        "uq_capability": "calibration",
        "uq_quality": 0.55,
        "heteroscedastic": False,
        "best_for": "Handles categorical features well, add uncertainty",
        "recommended_use_cases": ["exploration", "categorical_features"],
    },
    "MLPRegressor": {
        "uq_capability": "calibration",
        "uq_quality": 0.4,
        "heteroscedastic": False,
        "best_for": "Neural network baseline, consider ProbabilisticNN instead",
        "recommended_use_cases": ["exploration"],
    },
    
    # ── No Native UQ (Low/No Quality) ─────────────────────────────────────
    "LinearRegression": {
        "uq_capability": "none",
        "uq_quality": 0.2,
        "heteroscedastic": False,
        "best_for": "Interpretability, assumes homoscedastic Gaussian noise",
        "recommended_use_cases": ["exploration", "interpretability"],
    },
    "Ridge": {
        "uq_capability": "none",
        "uq_quality": 0.25,
        "heteroscedastic": False,
        "best_for": "Regularized linear model, limited UQ",
        "recommended_use_cases": ["exploration", "interpretability"],
    },
    "Lasso": {
        "uq_capability": "none",
        "uq_quality": 0.25,
        "heteroscedastic": False,
        "best_for": "Feature selection, limited UQ",
        "recommended_use_cases": ["exploration", "feature_selection"],
    },
    "SVR": {
        "uq_capability": "none",
        "uq_quality": 0.3,
        "heteroscedastic": False,
        "best_for": "Non-linear regression, requires separate calibration",
        "recommended_use_cases": ["exploration"],
    },
    "SVC": {
        "uq_capability": "none",
        "uq_quality": 0.3,
        "heteroscedastic": False,
        "best_for": "Non-linear classification, requires Platt scaling",
        "recommended_use_cases": ["exploration"],
    },
    "LogisticRegression": {
        "uq_capability": "calibration",
        "uq_quality": 0.6,
        "heteroscedastic": False,
        "best_for": "Simple probabilistic classifier, reasonably calibrated",
        "recommended_use_cases": ["exploration", "baseline"],
    },
    "KNeighborsRegressor": {
        "uq_capability": "none",
        "uq_quality": 0.4,
        "heteroscedastic": True,
        "best_for": "Local uncertainty via neighbor variance, not well-calibrated",
        "recommended_use_cases": ["exploration"],
    },
}


def list_all_models(task_type: str = "all") -> List[str]:
    """
    Return all supported model names. Use for programmatic access.

    Args:
        task_type: "regression", "classification", or "all"

    Returns:
        List of model class names.
    """
    if task_type == "regression":
        return REGRESSION_MODELS.copy()
    if task_type == "classification":
        return CLASSIFICATION_MODELS.copy()
    return sorted(MODEL_IMPORT_MAP.keys())


def get_model_suggestion(model_name: str, task_type: str) -> ModelSuggestion | None:
    """
    Get a ModelSuggestion by model name (no LLM). Returns None if unknown or mismatched task type.
    """
    if model_name not in MODEL_IMPORT_MAP:
        return None
    if task_type == "regression" and model_name not in REGRESSION_MODELS:
        return None
    if task_type == "classification" and model_name not in CLASSIFICATION_MODELS:
        return None
    
    module_path, class_name = MODEL_IMPORT_MAP[model_name]
    
    # Determine package name
    if "aims_agent" in module_path:
        pkg = "aims-agent"
    elif model_name.startswith("XGB"):
        pkg = "xgboost"
    elif model_name.startswith("LGBM"):
        pkg = "lightgbm"
    elif model_name.startswith("CatBoost"):
        pkg = "catboost"
    elif "gaussian_process" in module_path:
        pkg = "scikit-learn"
    else:
        pkg = "scikit-learn"
    
    suggestion = ModelSuggestion(
        model_name=model_name,
        package_name=pkg,
        import_path=f"{module_path}.{class_name}",
        reason=f"User-specified model: {model_name}",
    )
    
    # Enrich with UQ metadata
    return enrich_suggestion_with_uq_metadata(suggestion)


def suggest_models(
    agent: "Agent",
    features: List[str],
    target: str,
    *,
    n_suggestions: int = 5,
    task_hint: str = "regression",
    extra_context: str | None = None,
) -> List[ModelSuggestion]:
    """
    Ask the LLM to recommend several ML models with reasons; the user can then choose one.
    LLM can suggest ANY model from sklearn, xgboost, lightgbm, catboost, or other ML libraries.
    Returns a list of ModelSuggestion (or [DEFAULT_SUGGESTION] if parsing fails).
    """
    features_str = ", ".join(features)
    prompt = f"""You are an ML and Statistical Modeling expert in materials science with expertise in uncertainty quantification. Based on the dataset and its distribution below, recommend {n_suggestions} different ML models that are well-suited. You may choose from scikit-learn, xgboost, lightgbm, catboost, or any other standard Python ML library in Python. Do NOT be limited to a fixed list—pick the best models for this data and task.

CRITICAL: Prioritize models with native uncertainty quantification capability:

**Tier 1 - Native Distribution Predictors (BEST for uncertainty-critical tasks):**
- ProbabilisticNN (for 500-50K samples): Predicts full Gaussian distributions N(mu(x), sigma²(x)) with heteroscedastic uncertainty
  - Import: aims_agent.probabilistic_models.pnn.ProbabilisticNNWrapper
  - Package: aims-agent (custom)
- GaussianProcessRegressor (for <5K samples): Gold-standard calibrated uncertainty
  - Import: sklearn.gaussian_process.GaussianProcessRegressor

**Tier 2 - Ensemble Methods (Good uncertainty via variance):**
- RandomForest, GradientBoosting, ExtraTrees: Provide uncertainty through estimator variance

**Tier 3 - Standard Models (Require calibration):**
- XGBoost, LightGBM: High accuracy but need post-hoc uncertainty calibration

Input features: {features_str}
Target variable: {target}
Task type: {task_hint}
"""
    if extra_context:
        prompt += f"\n{extra_context}\n"

    prompt += """
Return ONLY a JSON array of exactly """ + str(n_suggestions) + """ objects. No other text, no markdown.
Each object MUST have all four fields:
  - model_name: the class name (e.g. RandomForestClassifier)
  - package: the pip install name (e.g. scikit-learn for sklearn, xgboost, lightgbm, catboost)
  - import_path: CRITICAL — full Python import path (e.g. sklearn.ensemble.RandomForestClassifier). Without import_path we cannot load the model. Always provide it.
  - reason: 1-2 sentences with professional reasoning based on task type, data characteristics, and uncertainty quantification capability

Task-type constraint:
- If Task type is regression, only suggest regression models.
- If Task type is classification, only suggest classification models.
- Prioritize models that support uncertainty estimation (ensembles, probabilistic models).

Example for regression:
[
  {"model_name": "ProbabilisticNN", "package": "aims-agent", "import_path": "aims_agent.probabilistic_models.pnn.ProbabilisticNNWrapper", "reason": "Native Gaussian distribution prediction with heteroscedastic uncertainty for materials property prediction."},
  {"model_name": "RandomForestRegressor", "package": "scikit-learn", "import_path": "sklearn.ensemble.RandomForestRegressor", "reason": "Robust ensemble with uncertainty via estimator variance."},
  {"model_name": "GradientBoostingRegressor", "package": "scikit-learn", "import_path": "sklearn.ensemble.GradientBoostingRegressor", "reason": "High accuracy with reasonable uncertainty estimates."}
]

Example for classification:
[
  {"model_name": "RandomForestClassifier", "package": "scikit-learn", "import_path": "sklearn.ensemble.RandomForestClassifier", "reason": "Robust to skewed features with probability estimates."},
  {"model_name": "XGBClassifier", "package": "xgboost", "import_path": "xgboost.XGBClassifier", "reason": "Handles class imbalance via scale_pos_weight."}
]
"""

    response = agent.call_llm(prompt)
    suggestions = _parse_responses(response)
    suggestions = _filter_suggestions_for_task(suggestions, task_hint)

    if not suggestions:
        print("[ModelSelector] Could not parse LLM response, using default model.")
        return [get_default_suggestion(task_hint)]

    # Enrich suggestions with UQ metadata
    suggestions = [enrich_suggestion_with_uq_metadata(s) for s in suggestions]

    return suggestions


def _filter_suggestions_for_task(
    suggestions: List[ModelSuggestion],
    task_type: str,
) -> List[ModelSuggestion]:
    """
    Remove only models that are *known* to belong to the wrong task type.
    Unknown models (not in MODEL_IMPORT_MAP) are kept so they can reach
    the dynamic_import or codegen execution paths.
    """
    if task_type not in {"regression", "classification"}:
        return suggestions
    wrong_task = set(
        CLASSIFICATION_MODELS if task_type == "regression" else REGRESSION_MODELS
    )
    filtered = [s for s in suggestions if s.model_name not in wrong_task]
    removed = len(suggestions) - len(filtered)
    if removed:
        print(f"[ModelSelector] Removed {removed} task-mismatched model suggestion(s).")
    return filtered


def enrich_unknown_suggestion(
    agent: "Agent",
    suggestion: ModelSuggestion,
    task_type: str,
) -> ModelSuggestion:
    """
    When a model is not in MODEL_IMPORT_MAP and has no import_path, ask the LLM
    for its full import path, pip package, and key implementation notes.
    Returns an enriched ModelSuggestion so codegen has richer context.
    """
    prompt = f"""You are a Python ML expert. The user wants to use the model "{suggestion.model_name}" for a {task_type} task.

Provide the following as JSON (no other text):
{{
  "import_path": "full.python.import.ClassName",
  "package_name": "pip-install-name",
  "implementation_notes": "1-2 sentences on key fit/predict implementation details"
}}

- import_path: exact Python dotted path (e.g. sklearn.gaussian_process.GaussianProcessRegressor)
- package_name: pip package to install (e.g. scikit-learn, ngboost, gplearn)
- implementation_notes: important constraints for the wrapper (output shape, required kwargs, etc.)

Return ONLY valid JSON."""

    try:
        response = agent.call_llm(prompt)
        cleaned = re.sub(r"```(?:json)?\s*", "", response).strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start != -1 and end > start:
            data = json.loads(cleaned[start:end])
            import_path = str(data.get("import_path", "")).strip()
            package_name = str(data.get("package_name", "")).strip()
            notes = str(data.get("implementation_notes", "")).strip()
            enriched = ModelSuggestion(
                model_name=suggestion.model_name,
                package_name=package_name or suggestion.package_name,
                import_path=import_path or suggestion.import_path,
                reason=f"{suggestion.reason} | {notes}".strip(" |") if notes else suggestion.reason,
            )
            print(
                f"[ModelSelector] Enriched '{suggestion.model_name}': "
                f"import_path={enriched.import_path!r}, package={enriched.package_name!r}"
            )
            return enriched
    except Exception as e:
        print(f"[ModelSelector] Could not enrich suggestion for '{suggestion.model_name}': {e}")
    return suggestion


def suggest_model(
    agent: "Agent",
    features: List[str],
    target: str,
    *,
    task_hint: str = "regression",
    extra_context: str | None = None,
) -> ModelSuggestion:
    """
    Ask the LLM to recommend a single ML model. Returns first of suggest_models(..., n_suggestions=1).
    """
    suggestions = suggest_models(
        agent, features, target, n_suggestions=1, task_hint=task_hint, extra_context=extra_context
    )
    return suggestions[0]


# Maps top-level import module -> pip package name
_IMPORT_TO_PIP: dict[str, str] = {
    "sklearn": "scikit-learn",
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
    "catboost": "catboost",
}


def _package_from_import_path(import_path: str) -> str:
    """Derive pip package name from import path (e.g. sklearn.ensemble.X -> scikit-learn)."""
    if not import_path or "." not in import_path:
        return import_path or "scikit-learn"
    top = import_path.split(".")[0]
    return _IMPORT_TO_PIP.get(top, top)


def _normalize_package_name(model_name: str, package_name: str, import_path: str | None = None) -> str:
    """
    If LLM put import_path in the package field (e.g. contains '.'), correct it.
    Also normalize 'sklearn' -> 'scikit-learn'.
    """
    if package_name.count(".") >= 1:
        print(f"[ModelSelector] package field looks wrong: '{package_name}', correcting...")
        if model_name in MODEL_IMPORT_MAP:
            package_name = "scikit-learn"
        else:
            package_name = package_name.split(".")[0]
    if package_name.strip().lower() == "sklearn":
        package_name = "scikit-learn"
    return package_name


def _parse_responses(response: str) -> List[ModelSuggestion]:
    """
    Extract a JSON array of model suggestions from LLM response.
    Handles markdown code blocks and corrects import_path via MODEL_IMPORT_MAP.
    """
    cleaned = re.sub(r"```(?:json)?\s*", "", response).strip()
    start = cleaned.find("[")
    if start == -1:
        # Fallback: single object
        single = _parse_response(response)
        return [single] if single else []
    # Find matching closing bracket
    depth = 0
    end = -1
    for i in range(start, len(cleaned)):
        if cleaned[i] == "[":
            depth += 1
        elif cleaned[i] == "]":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end == -1:
        print("[ModelSelector] No JSON array found in response.")
        return []

    try:
        arr = json.loads(cleaned[start:end])
    except json.JSONDecodeError as e:
        print(f"[ModelSelector] JSON parse error: {e}")
        return []

    if not isinstance(arr, list):
        return []

    required = {"model_name", "package", "reason"}
    out: List[ModelSuggestion] = []
    for idx, data in enumerate(arr):
        if not isinstance(data, dict):
            continue
        missing = required - set(data.keys())
        if missing:
            print(f"[ModelSelector] Item {idx+1} missing fields: {missing}")
            continue
        model_name = data["model_name"]
        package_name = data["package"]
        reason = data.get("reason", "No reason given.")
        if model_name in MODEL_IMPORT_MAP:
            module_path, class_name = MODEL_IMPORT_MAP[model_name]
            import_path = f"{module_path}.{class_name}"
            package_name = _normalize_package_name(model_name, package_name)
        elif data.get("import_path"):
            import_path = data["import_path"]
            package_name = _package_from_import_path(import_path)
        else:
            # Unknown model with no import_path: pass through for codegen.
            # ExecutionPathResolver will route it to LLM-generated wrapper.
            import_path = ""
            package_name = _normalize_package_name(model_name, package_name)
            print(f"[ModelSelector] Item {idx+1}: '{model_name}' not in map and no import_path — will use codegen.")
        out.append(
            ModelSuggestion(
                model_name=model_name,
                package_name=package_name,
                import_path=import_path,
                reason=reason,
            )
        )
    return out


def _parse_response(response: str) -> ModelSuggestion | None:
    """
    Extract a single JSON object from LLM response.
    """
    cleaned = re.sub(r"```(?:json)?\s*", "", response).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}") + 1
    if start == -1 or end == 0:
        return None

    try:
        data = json.loads(cleaned[start:end])
    except json.JSONDecodeError:
        return None

    required = {"model_name", "package", "reason"}
    if required - set(data.keys()):
        return None
    model_name = data["model_name"]
    package_name = data["package"]
    reason = data.get("reason", "No reason given.")
    if model_name in MODEL_IMPORT_MAP:
        module_path, class_name = MODEL_IMPORT_MAP[model_name]
        import_path = f"{module_path}.{class_name}"
        package_name = _normalize_package_name(model_name, package_name)
    elif data.get("import_path"):
        import_path = data["import_path"]
        package_name = _package_from_import_path(import_path)
    else:
        # Unknown model, no import_path: pass through for codegen.
        import_path = ""
        package_name = _normalize_package_name(model_name, package_name)
        print(f"[ModelSelector] '{model_name}' not in map and no import_path — will use codegen.")
    return ModelSuggestion(
        model_name=model_name,
        package_name=package_name,
        import_path=import_path,
        reason=reason,
    )


def load_model_class(suggestion: ModelSuggestion):
    """
    Ensure the required package is installed, then dynamically load the model class.

    Uses MODEL_IMPORT_MAP when the model is known; otherwise uses suggestion.import_path.
    Returns the class (uninstantiated).
    """
    ok = ensure_package_installed(suggestion.package_name)
    if not ok:
        raise RuntimeError(f"Failed to install package: {suggestion.package_name}")

    if suggestion.model_name in MODEL_IMPORT_MAP:
        module_path, class_name = MODEL_IMPORT_MAP[suggestion.model_name]
    else:
        module_path, class_name = suggestion.import_path.rsplit(".", 1)

    module = importlib.import_module(module_path)
    model_class = getattr(module, class_name)
    print(f"[ModelSelector] Successfully loaded: {class_name}")
    return model_class


__all__ = [
    "ModelSuggestion",
    "suggest_model",
    "suggest_models",
    "load_model_class",
    "get_default_suggestion",
    "get_model_suggestion",
    "list_all_models",
    "enrich_unknown_suggestion",
    "DEFAULT_SUGGESTION",
    "DEFAULT_CLASSIFICATION_SUGGESTION",
    "MODEL_IMPORT_MAP",
]


if __name__ == "__main__":
    # Quick test: load a known model without calling the LLM
    test_suggestion = ModelSuggestion(
        model_name="RandomForestRegressor",
        package_name="scikit-learn",
        import_path="sklearn.ensemble.RandomForestRegressor",
        reason="test",
    )
    cls = load_model_class(test_suggestion)
    model = cls()
    print(f"Instantiation successful: {model}")
