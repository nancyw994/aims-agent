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

DEFAULT_SUGGESTION = ModelSuggestion(
    model_name="RandomForestRegressor",
    package_name="scikit-learn",
    import_path="sklearn.ensemble.RandomForestRegressor",
    reason="Default fallback: random forest works well for most regression tasks",
)

DEFAULT_CLASSIFICATION_SUGGESTION = ModelSuggestion(
    model_name="RandomForestClassifier",
    package_name="scikit-learn",
    import_path="sklearn.ensemble.RandomForestClassifier",
    reason="Default fallback: random forest works well for multi-class classification",
)


def get_default_suggestion(task_type: str) -> "ModelSuggestion":
    """Return a default model suggestion without calling the LLM (for --no-llm mode)."""
    if task_type == "classification":
        return DEFAULT_CLASSIFICATION_SUGGESTION
    return DEFAULT_SUGGESTION

# Maps model class name -> (module path, class name) for reliable import path and dynamic load
MODEL_IMPORT_MAP: dict[str, tuple[str, str]] = {
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
    if m.endswith("Regressor") or m in ("Ridge", "Lasso", "ElasticNet", "LinearRegression", "SVR")
])
CLASSIFICATION_MODELS = sorted([
    m for m in MODEL_IMPORT_MAP
    if m.endswith("Classifier") or m in ("GaussianNB", "BernoulliNB", "SVC", "LogisticRegression")
])


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
    Get a ModelSuggestion by model name (no LLM). Returns None if unknown.
    """
    if model_name not in MODEL_IMPORT_MAP:
        return None
    module_path, class_name = MODEL_IMPORT_MAP[model_name]
    pkg = "scikit-learn"
    if model_name.startswith("XGB"):
        pkg = "xgboost"
    elif model_name.startswith("LGBM"):
        pkg = "lightgbm"
    elif model_name.startswith("CatBoost"):
        pkg = "catboost"
    return ModelSuggestion(
        model_name=model_name,
        package_name=pkg,
        import_path=f"{module_path}.{class_name}",
        reason=f"User-specified model: {model_name}",
    )


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
    prompt = f"""You are an ML and Statistical Modeling expert in materials science. Based on the dataset and its distribution below, recommend {n_suggestions} different ML models that are well-suited. You may choose from scikit-learn, xgboost, lightgbm, catboost, or any other standard Python ML library in Python. Do NOT be limited to a fixed list—pick the best models for this data and task.

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
  - reason: one sentence why this model fits

Example:
[
  {"model_name": "RandomForestClassifier", "package": "scikit-learn", "import_path": "sklearn.ensemble.RandomForestClassifier", "reason": "Robust to skewed features."},
  {"model_name": "XGBClassifier", "package": "xgboost", "import_path": "xgboost.XGBClassifier", "reason": "Handles class imbalance via scale_pos_weight."}
]
"""

    response = agent.call_llm(prompt)
    suggestions = _parse_responses(response)

    if not suggestions:
        print("[ModelSelector] Could not parse LLM response, using default model.")
        return [DEFAULT_SUGGESTION]

    return suggestions


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
            package_name = _package_from_import_path(import_path)  # derive from import_path; don't trust LLM
        else:
            print(f"[ModelSelector] Item {idx+1} missing import_path and model '{model_name}' not in map, skipping.")
            continue
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
        return None
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
