from __future__ import annotations

import json
import math
import re
import traceback
import warnings
from datetime import datetime
from html import escape
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.feature_selection import RFE, SelectFromModel, SelectKBest, VarianceThreshold, f_regression, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.base import clone
from sklearn.tree import DecisionTreeRegressor
from xgboost import DMatrix
from xgboost import XGBRegressor

from aims_agent.data_analyzer import formulate_strategy, profile_dataset, _render_report_html
from aims_agent.data_interface import DatasetBundle, DatasetSchema
from aims_agent.code_writer import extract_python_code, load_generated_module, save_generated_code, validate_python_syntax
from aims_agent.uncertainty_evaluator import UncertaintyEvaluator


RANDOM_STATE = 42
DATA_PATH = Path("data/real_data/Spall_Strength_Database_AliShargh(Processed).csv")
OUTPUT_ROOT = Path("results")
OUTPUT_DIR = OUTPUT_ROOT / datetime.now().strftime("run_%Y%m%d_%H%M%S")
TARGET = "Spall (Gpa)"

DROP_COLUMNS = {
    "Synthesis",
    "Refereces",
}

MATERIAL_PARAMETER_KEYWORDS = (
    "temperature",
    "yield",
    "ultimate",
    "kic",
    "hardness",
    "b (gpa",
    "g (gpa",
    "e (gpa",
    "mu",
    "melting",
    "thickness",
    "diameter",
    "grain",
    "density",
    "sound speed",
)


def clean_column_name(name: str) -> str:
    return re.sub(r"\s+", " ", name).strip()


def parse_numeric(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)
    text = str(value).strip()
    if not text:
        return np.nan

    text = text.replace(",", "")
    text = text.replace("−", "-")
    text = text.replace("?", "-")
    text = re.sub(r"\s+", "", text)

    if "±" in text:
        text = text.split("±", 1)[0]

    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if not match:
        return np.nan
    try:
        return float(match.group(0))
    except ValueError:
        return np.nan


def coerce_numeric_like_columns(df: pd.DataFrame, target: str) -> pd.DataFrame:
    cleaned = df.copy()
    for col in cleaned.columns:
        if col == target:
            cleaned[col] = cleaned[col].map(parse_numeric)
            continue
        if cleaned[col].dtype == "object":
            parsed = cleaned[col].map(parse_numeric)
            non_null_original = cleaned[col].notna().sum()
            parsed_ratio = parsed.notna().sum() / max(non_null_original, 1)
            if parsed_ratio >= 0.65:
                cleaned[col] = parsed
    return cleaned


def load_tabular_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".xls", ".xlsx"}:
        return pd.read_excel(path)
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1")


def infer_target_column(df: pd.DataFrame, requested_target: str | None) -> str:
    if requested_target:
        if requested_target not in df.columns:
            raise ValueError(f"Target column {requested_target!r} not found. Columns: {df.columns.tolist()}")
        return requested_target
    if TARGET in df.columns:
        return TARGET
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        return numeric_cols[-1]
    return df.columns[-1]


def root_mean_squared_error(y_true, y_pred) -> float:
    return math.sqrt(mean_squared_error(y_true, y_pred))


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [col for col in X.columns if col not in numeric_cols]

    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True, min_frequency=3)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore")

    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", encoder),
                    ]
                ),
                categorical_cols,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )


def model_registry():
    return {
        "AdaBoost": AdaBoostRegressor(random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1),
        "XGBoost": XGBRegressor(
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=1,
            tree_method="hist",
            eval_metric="rmse",
        ),
        "Ridge": Ridge(),
        "Lasso": Lasso(random_state=RANDOM_STATE, max_iter=10000),
        "ElasticNet": ElasticNet(random_state=RANDOM_STATE, max_iter=10000),
        "SVR": SVR(),
        "KNeighborsRegressor": KNeighborsRegressor(),
        "LinearRegression": LinearRegression(),
    }


def model_spaces():
    return {name: (model, {}) for name, model in model_registry().items()}


def supported_hyperparameter_schema() -> dict:
    return {
        "AdaBoost": {
            "n_estimators": "integer list, typical 25-500",
            "learning_rate": "float list, typical 0.005-1.0",
            "loss": "subset of ['linear', 'square', 'exponential']",
            "estimator_max_depth": "integer list for DecisionTreeRegressor base estimator, typical 1-6",
        },
        "Gradient Boosting": {
            "n_estimators": "integer list, typical 50-800",
            "learning_rate": "float list, typical 0.005-0.2",
            "max_depth": "integer list, typical 1-6",
            "subsample": "float list in (0, 1]",
            "min_samples_leaf": "integer list, typical 1-20",
        },
        "Random Forest": {
            "n_estimators": "integer list, typical 100-1000",
            "max_depth": "list of integers and/or null",
            "min_samples_split": "integer list, typical 2-20",
            "min_samples_leaf": "integer list, typical 1-20",
            "max_features": "list using 'sqrt', 'log2', or floats in (0, 1]",
        },
        "XGBoost": {
            "n_estimators": "integer list, typical 50-800",
            "learning_rate": "float list, typical 0.005-0.2",
            "max_depth": "integer list, typical 1-8",
            "subsample": "float list in (0, 1]",
            "colsample_bytree": "float list in (0, 1]",
            "reg_lambda": "float list >= 0",
            "min_child_weight": "float or integer list >= 0",
        },
        "Ridge": {
            "alpha": "positive float list",
            "solver": "subset of ['auto', 'svd', 'cholesky', 'lsqr', 'sag', 'saga']",
        },
        "Lasso": {
            "alpha": "positive float list",
            "selection": "subset of ['cyclic', 'random']",
        },
        "ElasticNet": {
            "alpha": "positive float list",
            "l1_ratio": "float list in [0, 1]",
            "selection": "subset of ['cyclic', 'random']",
        },
        "SVR": {
            "C": "positive float list",
            "epsilon": "non-negative float list",
            "kernel": "subset of ['rbf', 'linear', 'poly', 'sigmoid']",
            "gamma": "subset of ['scale', 'auto'] or positive float list",
        },
        "KNeighborsRegressor": {
            "n_neighbors": "positive integer list",
            "weights": "subset of ['uniform', 'distance']",
            "p": "subset of [1, 2]",
        },
        "LinearRegression": {
            "fit_intercept": "boolean list",
        },
    }


MODEL_FAMILY_TO_MODELS = {
    "linear_baseline": ["LinearRegression"],
    "regularized_linear": ["Ridge", "ElasticNet", "Lasso"],
    "tree_bagging": ["Random Forest"],
    "gradient_boosting": ["Gradient Boosting", "XGBoost"],
    "kernel_methods": ["SVR"],
    "local_similarity": ["KNeighborsRegressor"],
}


def default_model_recommendations() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": "AdaBoost",
                "raw_model": "AdaBoostRegressor",
                "package": "scikit-learn",
                "import_path": "sklearn.ensemble.AdaBoostRegressor",
                "recommended_by": "fallback",
                "execution_status": "supported",
                "reason": "Default comparison model: boosting baseline for tabular regression.",
            },
            {
                "model": "Gradient Boosting",
                "raw_model": "GradientBoostingRegressor",
                "package": "scikit-learn",
                "import_path": "sklearn.ensemble.GradientBoostingRegressor",
                "recommended_by": "fallback",
                "execution_status": "supported",
                "reason": "Default comparison model: robust gradient-boosted trees for tabular regression.",
            },
            {
                "model": "Random Forest",
                "raw_model": "RandomForestRegressor",
                "package": "scikit-learn",
                "import_path": "sklearn.ensemble.RandomForestRegressor",
                "recommended_by": "fallback",
                "execution_status": "supported",
                "reason": "Default comparison model: stable bagged-tree baseline for nonlinear features.",
            },
            {
                "model": "XGBoost",
                "raw_model": "XGBRegressor",
                "package": "xgboost",
                "import_path": "xgboost.XGBRegressor",
                "recommended_by": "fallback",
                "execution_status": "supported",
                "reason": "Default comparison model: strong boosted-tree model for structured data.",
            },
        ]
    )


def deterministic_dataset_profile(df: pd.DataFrame, features: list[str], target: str) -> dict:
    numeric_features = [col for col in features if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    categorical_features = [col for col in features if col in df.columns and col not in numeric_features]
    n_rows = int(len(df))
    n_features = int(len(features))
    missing_fraction = float(df[[*features, target]].isna().mean().mean())

    target_series = pd.to_numeric(df[target], errors="coerce")
    target_skew = float(target_series.skew()) if target_series.notna().sum() > 2 else 0.0

    numeric_df = df[numeric_features].apply(pd.to_numeric, errors="coerce") if numeric_features else pd.DataFrame()
    feature_skew = float(numeric_df.skew(numeric_only=True).abs().median()) if not numeric_df.empty else 0.0

    high_corr_pairs = 0
    mean_abs_spearman_minus_pearson = 0.0
    if len(numeric_features) >= 2:
        pearson = numeric_df.corr(method="pearson").abs()
        upper = pearson.where(np.triu(np.ones(pearson.shape), k=1).astype(bool))
        high_corr_pairs = int((upper >= 0.85).sum().sum())
        spearman = numeric_df.corr(method="spearman").abs()
        diff = (spearman - pearson).abs().where(np.triu(np.ones(pearson.shape), k=1).astype(bool))
        mean_abs_spearman_minus_pearson = float(diff.stack().mean()) if not diff.stack().empty else 0.0

    outlier_fractions = []
    for col in numeric_features:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(series) < 4:
            continue
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0:
            continue
        outlier_fractions.append(float(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).mean()))
    outlier_ratio = float(np.mean(outlier_fractions)) if outlier_fractions else 0.0

    feature_to_sample_ratio = float(n_features / max(n_rows, 1))
    estimated_nonlinearity = float(
        min(
            1.0,
            0.35 * (abs(target_skew) > 1.0)
            + 0.25 * (feature_skew > 1.0)
            + 0.25 * (mean_abs_spearman_minus_pearson > 0.08)
            + 0.15 * (len(categorical_features) > 0),
        )
    )

    return {
        "sample_size": n_rows,
        "feature_dimensionality": n_features,
        "numeric_feature_count": len(numeric_features),
        "categorical_feature_count": len(categorical_features),
        "feature_to_sample_ratio": feature_to_sample_ratio,
        "missingness": missing_fraction,
        "target_skewness": target_skew,
        "median_abs_feature_skewness": feature_skew,
        "high_multicollinearity_pairs": high_corr_pairs,
        "outlier_ratio": outlier_ratio,
        "estimated_nonlinearity": estimated_nonlinearity,
        "target_distribution": {
            "mean": float(target_series.mean()),
            "std": float(target_series.std()),
            "min": float(target_series.min()),
            "median": float(target_series.median()),
            "max": float(target_series.max()),
        },
    }


def deterministic_family_scores(profile: dict) -> dict[str, float]:
    n_rows = profile["sample_size"]
    n_features = profile["feature_dimensionality"]
    ratio = profile["feature_to_sample_ratio"]
    missing = profile["missingness"]
    skew = abs(profile["target_skewness"])
    high_corr = profile["high_multicollinearity_pairs"]
    outliers = profile["outlier_ratio"]
    nonlinearity = profile["estimated_nonlinearity"]
    categorical = profile["categorical_feature_count"]

    scores = {
        "linear_baseline": 0.25,
        "regularized_linear": 0.35,
        "tree_bagging": 0.35,
        "gradient_boosting": 0.35,
        "kernel_methods": 0.20,
        "local_similarity": 0.15,
    }
    if n_rows < 500 or ratio > 0.10 or high_corr > 0:
        scores["regularized_linear"] += 0.35
    if nonlinearity >= 0.35 or categorical > 0:
        scores["tree_bagging"] += 0.30
        scores["gradient_boosting"] += 0.35
    if n_rows >= 50:
        scores["gradient_boosting"] += 0.20
    if outliers > 0.03 or missing > 0.05:
        scores["tree_bagging"] += 0.15
    if n_rows < 1000 and n_features < 80 and nonlinearity >= 0.25:
        scores["kernel_methods"] += 0.25
        scores["local_similarity"] += 0.10
    if skew > 1.0:
        scores["gradient_boosting"] += 0.10
        scores["regularized_linear"] += 0.05

    return {key: round(float(value), 3) for key, value in scores.items()}


def family_confidence(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.50:
        return "medium"
    return "low"


def deterministic_recommendations_from_profile(profile: dict, n_models: int = 5) -> pd.DataFrame:
    scores = deterministic_family_scores(profile)
    family_order = sorted(scores, key=lambda family: (-scores[family], family))
    rows = []
    seen = set()
    for family in family_order:
        for model in MODEL_FAMILY_TO_MODELS[family]:
            if model in seen:
                continue
            rows.append(
                {
                    "model": model,
                    "raw_model": model,
                    "package": "xgboost" if model == "XGBoost" else "scikit-learn",
                    "import_path": "",
                    "recommended_by": "deterministic_profile_mapping",
                    "execution_status": "supported",
                    "model_family": family,
                    "family_score": scores[family],
                    "confidence": family_confidence(scores[family]),
                    "reason": (
                        f"Selected deterministically from dataset profile via family '{family}' "
                        f"(score={scores[family]})."
                    ),
                }
            )
            seen.add(model)
            if len(rows) >= n_models:
                return pd.DataFrame(rows).reset_index(drop=True)
    return pd.DataFrame(rows).reset_index(drop=True)


def supported_feature_selection_methods() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "method": "RandomForest_SelectFromModel",
                "family": "embedded_tree_importance",
                "recommended_use": "Nonlinear feature selector for mixed tabular data.",
                "background": "Fits a Random Forest and keeps features with above-threshold impurity importance. It preserves original feature names and can capture nonlinear feature effects.",
            },
            {
                "method": "PCA_95Variance",
                "family": "unsupervised_projection",
                "recommended_use": "Dimensionality reduction for correlated or high-dimensional inputs.",
                "background": "PCA converts correlated numeric/encoded inputs into orthogonal principal components. It is dimensionality reduction rather than direct feature selection, so interpretability shifts from original columns to components.",
            },
            {
                "method": "SelectKBest_FRegression",
                "family": "univariate_filter",
                "recommended_use": "Fast baseline that ranks features by individual relationship with the target.",
                "background": "Uses univariate F-tests for regression. It is simple and deterministic, but it can miss feature interactions and nonlinear effects.",
            },
            {
                "method": "Lasso_SelectFromModel",
                "family": "embedded_regularized_linear",
                "recommended_use": "Sparse linear embedded selector for multicollinearity and interpretability.",
                "background": "Lasso uses L1 regularization to shrink weak coefficients to zero. It is interpretable for mostly linear signals, but less suitable when the target depends strongly on nonlinear interactions.",
            },
            {
                "method": "GradientBoosting_SelectFromModel",
                "family": "embedded_boosting_importance",
                "recommended_use": "Nonlinear embedded selector when boosted-tree interactions are expected.",
                "background": "Fits gradient-boosted trees and selects features using model importance. It can capture nonlinear effects but may be less stable on small noisy datasets.",
            },
            {
                "method": "ElasticNet_SelectFromModel",
                "family": "embedded_regularized_linear",
                "recommended_use": "Sparse linear selector that balances L1 sparsity and L2 shrinkage.",
                "background": "ElasticNet combines Lasso and Ridge penalties, making it useful when predictors are correlated and a sparse but less brittle linear selector is desired.",
            },
            {
                "method": "MutualInfo_SelectKBest",
                "family": "univariate_nonlinear_filter",
                "recommended_use": "Filter method for nonlinear individual feature-target associations.",
                "background": "Ranks features by mutual information with the target. It can detect nonlinear univariate associations but does not model feature interactions.",
            },
            {
                "method": "RFE_Ridge",
                "family": "wrapper_recursive_elimination",
                "recommended_use": "Wrapper selector for compact, interpretable linear feature subsets.",
                "background": "Recursive feature elimination repeatedly fits a Ridge model and removes weaker features. It is more computationally expensive than filters but directly evaluates a supervised estimator.",
            },
            {
                "method": "VarianceThreshold",
                "family": "unsupervised_filter",
                "recommended_use": "Basic preprocessing filter for near-constant encoded features.",
                "background": "Removes features with very low variance. It is target-agnostic and should usually be considered a lightweight cleanup step rather than a complete feature-selection strategy.",
            },
            {
                "method": "PCA_90Variance",
                "family": "unsupervised_projection",
                "recommended_use": "More aggressive PCA dimensionality reduction for high redundancy.",
                "background": "Retains enough principal components to explain 90% of feature variance. It can reduce dimensionality more strongly than PCA_95Variance but may discard predictive low-variance signals.",
            },
            {
                "method": "PCA_99Variance",
                "family": "unsupervised_projection",
                "recommended_use": "Conservative PCA dimensionality reduction when preserving variance is important.",
                "background": "Retains enough principal components to explain 99% of variance. It reduces collinearity while keeping more information, but may preserve many components.",
            },
        ]
    )


def deterministic_feature_selection_recommendations(profile: dict) -> pd.DataFrame:
    method_info = supported_feature_selection_methods().set_index("method").to_dict(orient="index")
    rows = [
        {
            "method": "RandomForest_SelectFromModel",
            "score": 0.45 + 0.25 * (profile["estimated_nonlinearity"] >= 0.35) + 0.15 * (profile["categorical_feature_count"] > 0),
            "reason": "Deterministic fallback: nonlinear signal and categorical interactions make tree-based embedded selection plausible.",
        },
        {
            "method": "PCA_95Variance",
            "score": 0.25 + 0.30 * (profile["high_multicollinearity_pairs"] > 0) + 0.15 * (profile["feature_to_sample_ratio"] > 0.10),
            "reason": "Deterministic fallback: multicollinearity or high feature-to-sample ratio can make projection methods useful.",
        },
        {
            "method": "SelectKBest_FRegression",
            "score": 0.35 + 0.15 * (profile["sample_size"] < 500),
            "reason": "Deterministic fallback: univariate filtering is a fast baseline for small to moderate tabular datasets.",
        },
        {
            "method": "Lasso_SelectFromModel",
            "score": 0.30 + 0.25 * (profile["high_multicollinearity_pairs"] > 0) + 0.10 * (profile["feature_to_sample_ratio"] > 0.10),
            "reason": "Deterministic fallback: sparse linear selection is useful when multicollinearity and interpretability matter.",
        },
        {
            "method": "GradientBoosting_SelectFromModel",
            "score": 0.35 + 0.25 * (profile["estimated_nonlinearity"] >= 0.35),
            "reason": "Deterministic fallback: boosted-tree importance is plausible when nonlinear interactions are likely.",
        },
        {
            "method": "ElasticNet_SelectFromModel",
            "score": 0.30 + 0.20 * (profile["high_multicollinearity_pairs"] > 0),
            "reason": "Deterministic fallback: ElasticNet balances sparsity and correlated predictors.",
        },
        {
            "method": "MutualInfo_SelectKBest",
            "score": 0.30 + 0.20 * (profile["estimated_nonlinearity"] >= 0.35),
            "reason": "Deterministic fallback: mutual information can detect nonlinear univariate target associations.",
        },
        {
            "method": "RFE_Ridge",
            "score": 0.25 + 0.20 * (profile["feature_to_sample_ratio"] < 0.15),
            "reason": "Deterministic fallback: RFE is feasible when the original feature-to-sample ratio is moderate.",
        },
    ]
    df = pd.DataFrame(rows)
    df["family"] = df["method"].map(lambda method: method_info[method]["family"])
    df["recommended_use"] = df["method"].map(lambda method: method_info[method]["recommended_use"])
    df["background"] = df["method"].map(lambda method: method_info[method]["background"])
    df["score"] = df["score"].clip(upper=1.0).round(3)
    df["confidence"] = df["score"].map(family_confidence)
    df["recommended_by"] = "deterministic_fallback"
    return df.sort_values(["score", "method"], ascending=[False, True]).reset_index(drop=True)


def feature_selection_key_metrics(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    X_train_encoded,
    encoded_feature_names: np.ndarray,
) -> dict:
    numeric_features = [col for col in features if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    categorical_features = [col for col in features if col in df.columns and col not in numeric_features]
    numeric_df = df[numeric_features].apply(pd.to_numeric, errors="coerce") if numeric_features else pd.DataFrame()
    y = pd.to_numeric(df[target], errors="coerce")

    high_corr_pairs = 0
    max_abs_corr = 0.0
    median_abs_corr = 0.0
    if len(numeric_features) >= 2:
        corr = numeric_df.corr(method="pearson").abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
        high_corr_pairs = int((upper >= 0.85).sum())
        max_abs_corr = float(upper.max()) if not upper.empty else 0.0
        median_abs_corr = float(upper.median()) if not upper.empty else 0.0

    target_corr = {}
    if numeric_features:
        joined = numeric_df.copy()
        joined[target] = y
        corr_to_target = joined.corr(numeric_only=True)[target].drop(labels=[target], errors="ignore").abs()
        target_corr = {
            "max_abs_feature_target_corr": float(corr_to_target.max()) if not corr_to_target.empty else 0.0,
            "median_abs_feature_target_corr": float(corr_to_target.median()) if not corr_to_target.empty else 0.0,
            "features_with_abs_corr_ge_0_3": int((corr_to_target >= 0.30).sum()) if not corr_to_target.empty else 0,
            "top_target_correlated_features": corr_to_target.sort_values(ascending=False).head(8).round(4).to_dict(),
        }
    else:
        target_corr = {
            "max_abs_feature_target_corr": 0.0,
            "median_abs_feature_target_corr": 0.0,
            "features_with_abs_corr_ge_0_3": 0,
            "top_target_correlated_features": {},
        }

    X_dense = to_dense_array(X_train_encoded)
    variances = np.var(X_dense, axis=0)
    low_variance_ratio = float((variances < 1e-8).mean()) if len(variances) else 0.0
    encoded_to_original_ratio = float(len(encoded_feature_names) / max(len(features), 1))

    return {
        "sample_size": int(len(df)),
        "original_feature_count": int(len(features)),
        "encoded_feature_count": int(len(encoded_feature_names)),
        "numeric_feature_count": int(len(numeric_features)),
        "categorical_feature_count": int(len(categorical_features)),
        "encoded_to_original_feature_ratio": encoded_to_original_ratio,
        "feature_to_sample_ratio_original": float(len(features) / max(len(df), 1)),
        "feature_to_sample_ratio_encoded": float(len(encoded_feature_names) / max(len(df), 1)),
        "missingness": float(df[[*features, target]].isna().mean().mean()),
        "high_multicollinearity_pairs_abs_corr_ge_0_85": high_corr_pairs,
        "max_abs_feature_feature_corr": max_abs_corr,
        "median_abs_feature_feature_corr": median_abs_corr,
        "low_variance_encoded_feature_ratio": low_variance_ratio,
        "target_linear_signal": target_corr,
        "estimated_nonlinearity": deterministic_dataset_profile(df, features, target)["estimated_nonlinearity"],
        "method_implications": {
            "PCA_95Variance": "More attractive when multicollinearity or encoded dimensionality is high, but weaker for direct feature interpretability.",
            "PCA_90Variance": "More aggressive projection when redundancy is high and compactness matters.",
            "PCA_99Variance": "More conservative projection when redundancy exists but information loss is a concern.",
            "RandomForest_SelectFromModel": "More attractive when estimated nonlinearity or categorical interactions are important.",
            "GradientBoosting_SelectFromModel": "More attractive when nonlinear boosted-tree interactions are expected.",
            "SelectKBest_FRegression": "More attractive when individual feature-target linear signals are strong and interactions are less central.",
            "MutualInfo_SelectKBest": "More attractive when individual feature-target associations may be nonlinear.",
            "Lasso_SelectFromModel": "More attractive when sparsity, interpretability, and multicollinearity control are important.",
            "ElasticNet_SelectFromModel": "More attractive when correlated predictors make pure Lasso unstable.",
            "RFE_Ridge": "More attractive when a supervised wrapper is affordable and interpretability matters.",
            "VarianceThreshold": "More attractive as a lightweight cleanup step for near-constant encoded features.",
        },
    }


def llm_feature_selection_recommendations(
    profile: dict,
    feature_selection_metrics: dict,
    features: list[str],
    target: str,
    motivation: str | None,
    background_knowledge: str | None,
    *,
    use_llm: bool,
    output_dir: Path,
) -> pd.DataFrame:
    executable_catalog = supported_feature_selection_methods()
    fallback = deterministic_feature_selection_recommendations(profile)
    if not use_llm:
        return fallback
    try:
        from aims_agent.agent import Agent

        prompt = f"""You are a scientific ML assistant choosing feature-selection or dimensionality-reduction methods.

Analyze the deterministic dataset profile and the feature-selection key metrics, then recommend feature-selection or dimensionality-reduction methods for this dataset.
You may recommend methods outside the executable catalog if scientifically justified, but only executable-catalog methods will be evaluated by this run. Return JSON only.

Dataset target: {target}
Feature names: {", ".join(features[:80])}
Motivation: {motivation or "N/A"}
Background knowledge: {background_knowledge or "N/A"}

Deterministic dataset profile:
{json.dumps(profile, indent=2)}

Feature-selection key metrics:
{json.dumps(feature_selection_metrics, indent=2)}

Executable method catalog:
{executable_catalog.to_json(orient="records", indent=2)}

Return ONLY a JSON array with 4-8 recommended methods, ranked from most to least suitable.
Each item must have:
{{
  "method": "method name",
  "family": "method family",
  "confidence": "low|medium|high",
  "reason": "why this method is suitable or less suitable for this dataset",
  "background": "brief scientific background of the method in this context"
}}
"""
        response = Agent().call_llm(prompt)
        cleaned = re.sub(r"```(?:json)?\s*", "", response).replace("```", "").strip()
        start = cleaned.find("[")
        end = cleaned.rfind("]") + 1
        parsed = json.loads(cleaned[start:end])
        catalog_methods = executable_catalog["method"].tolist()
        catalog_set = set(catalog_methods)
        rows = []
        seen = set()
        for item in parsed:
            method = str(item.get("method", "")).strip()
            if not method or method in seen:
                continue
            if method in catalog_set:
                info = executable_catalog[executable_catalog["method"].eq(method)].iloc[0].to_dict()
                execution_status = "supported"
            else:
                info = {
                    "family": str(item.get("family", "llm_external_recommendation")).strip(),
                    "recommended_use": "LLM-recommended method not currently implemented in the local evaluator.",
                    "background": str(item.get("background", "")).strip(),
                }
                execution_status = "unsupported_not_evaluated"
            rows.append(
                {
                    "method": method,
                    "family": info["family"],
                    "confidence": str(item.get("confidence", "medium")).strip().lower(),
                    "recommended_use": info["recommended_use"],
                    "background": str(item.get("background", info["background"])).strip() or info["background"],
                    "reason": str(item.get("reason", "")).strip(),
                    "recommended_by": "llm",
                    "execution_status": execution_status,
                }
            )
            seen.add(method)
        if not any(row["execution_status"] == "supported" for row in rows):
            for row in fallback.head(4).to_dict(orient="records"):
                row["execution_status"] = "supported"
                row["recommended_by"] = "deterministic_completion"
                rows.append(row)
        recommendations = pd.DataFrame(rows)
        (output_dir / "llm_feature_selection_recommendations.json").write_text(
            json.dumps(recommendations.to_dict(orient="records"), indent=2),
            encoding="utf-8",
        )
        return recommendations
    except Exception as exc:
        fallback = fallback.copy()
        fallback["reason"] = "LLM feature-selection recommendation unavailable; deterministic fallback used."
        fallback["llm_error"] = str(exc)
        return fallback


def to_dense_array(matrix):
    return matrix.toarray() if hasattr(matrix, "toarray") else matrix


def build_feature_selector(method: str, n_features: int):
    if method == "RandomForest_SelectFromModel":
        return SelectFromModel(
            RandomForestRegressor(n_estimators=250, random_state=RANDOM_STATE, n_jobs=1),
            threshold="median",
        )
    if method == "PCA_95Variance":
        return PCA(n_components=0.95, random_state=RANDOM_STATE)
    if method == "PCA_90Variance":
        return PCA(n_components=0.90, random_state=RANDOM_STATE)
    if method == "PCA_99Variance":
        return PCA(n_components=0.99, random_state=RANDOM_STATE)
    if method == "SelectKBest_FRegression":
        return SelectKBest(score_func=f_regression, k=min(20, max(1, n_features)))
    if method == "MutualInfo_SelectKBest":
        return SelectKBest(
            score_func=lambda X, y: mutual_info_regression(X, y, random_state=RANDOM_STATE),
            k=min(20, max(1, n_features)),
        )
    if method == "Lasso_SelectFromModel":
        return SelectFromModel(
            Lasso(alpha=0.01, random_state=RANDOM_STATE, max_iter=10000),
            threshold="median",
        )
    if method == "ElasticNet_SelectFromModel":
        return SelectFromModel(
            ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=RANDOM_STATE, max_iter=10000),
            threshold="median",
        )
    if method == "GradientBoosting_SelectFromModel":
        return SelectFromModel(
            GradientBoostingRegressor(random_state=RANDOM_STATE),
            threshold="median",
        )
    if method == "RFE_Ridge":
        return RFE(
            estimator=Ridge(),
            n_features_to_select=min(20, max(1, n_features)),
            step=0.2,
        )
    if method == "VarianceThreshold":
        return VarianceThreshold(threshold=1e-8)
    raise ValueError(f"Unsupported feature selection method: {method}")


def feature_selector_codegen_prompt(
    *,
    method: str,
    reason: str,
    feature_selection_metrics: dict,
    previous_code: str = "",
    error_message: str = "",
) -> str:
    repair_context = ""
    if previous_code or error_message:
        repair_context = f"""
Previous generated code failed.
Error:
{error_message}

Previous code:
```python
{previous_code}
```
"""
    return f"""You are a senior Python ML engineer.

Generate one complete Python module implementing the feature-selection method: {method}

LLM recommendation reason:
{reason or "N/A"}

Feature-selection key metrics:
{json.dumps(feature_selection_metrics, indent=2)}
{repair_context}

Hard requirements:
- Return ONLY Python code in one ```python``` block.
- Define class GeneratedFeatureSelector.
- GeneratedFeatureSelector must implement fit(self, X, y=None), transform(self, X), and fit_transform(self, X, y=None).
- fit must return self.
- transform must return a 2D numpy array with the same number of rows as X.
- Handle numpy arrays, scipy sparse matrices, and pandas DataFrames.
- Use only numpy, scipy, and sklearn.
- No file I/O, subprocess, os.system, eval, exec, networking, or hardcoded feature names.
- Keep deterministic defaults with random_state=42 when relevant.
- Keep runtime lightweight for a few hundred rows and up to a few hundred encoded features.
- If implementing a complex named method is not feasible, approximate it with a robust sklearn-compatible selector that matches the method's scientific intent.
"""


def generate_feature_selector_from_llm(
    *,
    method: str,
    reason: str,
    feature_selection_metrics: dict,
    output_dir: Path,
    max_retries: int = 1,
):
    from aims_agent.agent import Agent

    agent = Agent()
    code = ""
    error_message = ""
    generated_dir = output_dir / "generated_feature_selectors"
    module_base = re.sub(r"[^A-Za-z0-9_]+", "_", method).strip("_").lower() or "generated_feature_selector"
    log_path = generated_dir / f"self_correction_feature_selector_{module_base}.jsonl"
    for attempt in range(max_retries + 1):
        prompt = feature_selector_codegen_prompt(
            method=method,
            reason=reason,
            feature_selection_metrics=feature_selection_metrics,
            previous_code=code,
            error_message=error_message,
        )
        try:
            response = agent.call_llm(prompt)
            code = extract_python_code(response)
            validate_python_syntax(code)
            path = save_generated_code(
                code,
                output_dir=generated_dir,
                module_name=f"{module_base}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            )
            module = load_generated_module(path)
            cls = getattr(module, "GeneratedFeatureSelector", None)
            if cls is None:
                raise AttributeError("Generated module must define class GeneratedFeatureSelector")
            selector = cls()
            for attr in ("fit", "transform", "fit_transform"):
                if not hasattr(selector, attr):
                    raise AttributeError(f"GeneratedFeatureSelector missing {attr}")
            selector._aims_generated_feature_selector_path = str(path)
            selector._aims_generated_feature_selector_method = method
            return selector, str(path), attempt
        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            generated_dir.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(
                    json.dumps(
                        {
                            "method": method,
                            "attempt": attempt,
                            "error": error_message,
                            "patched": attempt < max_retries,
                        }
                    )
                    + "\n"
                )
            if attempt >= max_retries:
                raise RuntimeError(error_message) from exc
    raise RuntimeError(error_message or "Feature selector codegen failed")


def selector_feature_names(selector, method: str, encoded_feature_names: np.ndarray) -> np.ndarray:
    if method.startswith("PCA_"):
        return np.asarray([f"PC{i + 1}" for i in range(int(selector.n_components_))])
    if hasattr(selector, "get_support"):
        return encoded_feature_names[selector.get_support()]
    if hasattr(selector, "n_features_out_"):
        return np.asarray([f"{method}_feature_{i + 1}" for i in range(int(selector.n_features_out_))])
    return encoded_feature_names


def compare_feature_selection_methods(
    recommendations: pd.DataFrame,
    X_train_encoded,
    y_train,
    encoded_feature_names: np.ndarray,
    output_dir: Path,
    feature_selection_metrics: dict,
    *,
    use_llm: bool,
) -> tuple[pd.DataFrame, object, str, np.ndarray]:
    output_dir.mkdir(parents=True, exist_ok=True)
    X_dense = to_dense_array(X_train_encoded)
    cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    fitted_selectors = {}
    recommendation_meta = recommendations.reset_index().set_index("method").to_dict(orient="index")
    for method in recommendations["method"].tolist():
        meta = recommendation_meta.get(method, {})
        try:
            generated_path = ""
            codegen_attempts = 0
            status = "evaluated"
            if meta.get("execution_status") == "unsupported_not_evaluated":
                if not use_llm:
                    rows.append(
                        {
                            "method": method,
                            "llm_rank": int(meta.get("index", 0)) + 1,
                            "llm_confidence": meta.get("confidence", ""),
                            "llm_reason": meta.get("reason", ""),
                            "n_selected_features": 0,
                            "cv_r2_mean": np.nan,
                            "cv_r2_std": np.nan,
                            "cv_rmse_mean": np.nan,
                            "cv_rmse_std": np.nan,
                            "status": "unsupported_not_evaluated",
                            "generated_selector_path": "",
                            "codegen_attempts": 0,
                        }
                    )
                    continue
                selector, generated_path, codegen_attempts = generate_feature_selector_from_llm(
                    method=method,
                    reason=str(meta.get("reason", "")),
                    feature_selection_metrics=feature_selection_metrics,
                    output_dir=output_dir,
                    max_retries=1,
                )
                status = "codegen_evaluated"
            else:
                selector = build_feature_selector(method, X_dense.shape[1])
            X_selected = selector.fit_transform(X_dense, y_train)
            if not hasattr(selector, "n_features_out_"):
                selector.n_features_out_ = int(X_selected.shape[1])
            model = GradientBoostingRegressor(random_state=RANDOM_STATE)
            scores = cross_validate(
                model,
                X_selected,
                y_train,
                cv=cv,
                scoring={"r2": "r2", "rmse": "neg_root_mean_squared_error"},
                n_jobs=1,
            )
            n_selected = X_selected.shape[1]
            rows.append(
                {
                    "method": method,
                    "llm_rank": int(meta.get("index", 0)) + 1,
                    "llm_confidence": meta.get("confidence", ""),
                    "llm_reason": meta.get("reason", ""),
                    "n_selected_features": int(n_selected),
                    "cv_r2_mean": float(np.mean(scores["test_r2"])),
                    "cv_r2_std": float(np.std(scores["test_r2"])),
                    "cv_rmse_mean": float(np.mean(-scores["test_rmse"])),
                    "cv_rmse_std": float(np.std(-scores["test_rmse"])),
                    "status": status,
                    "generated_selector_path": generated_path,
                    "codegen_attempts": codegen_attempts,
                }
            )
            fitted_selectors[method] = selector
        except Exception as exc:
            rows.append(
                {
                    "method": method,
                    "llm_rank": int(meta.get("index", 0)) + 1,
                    "llm_confidence": meta.get("confidence", ""),
                    "llm_reason": meta.get("reason", ""),
                    "n_selected_features": 0,
                    "cv_r2_mean": np.nan,
                    "cv_r2_std": np.nan,
                    "cv_rmse_mean": np.nan,
                    "cv_rmse_std": np.nan,
                    "status": "codegen_failed" if meta.get("execution_status") == "unsupported_not_evaluated" else f"failed: {exc}",
                    "generated_selector_path": "",
                    "codegen_attempts": 0,
                }
            )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output_dir / "feature_selection_method_comparison.csv", index=False)
    valid = comparison[comparison["status"].isin(["evaluated", "codegen_evaluated"])].sort_values(["cv_r2_mean", "cv_rmse_mean"], ascending=[False, True])
    best_method = valid.iloc[0]["method"] if not valid.empty else "RandomForest_SelectFromModel"
    best_selector = fitted_selectors.get(best_method)
    if best_selector is None:
        best_selector = build_feature_selector(best_method, X_dense.shape[1]).fit(X_dense, y_train)
    selected_names = selector_feature_names(best_selector, best_method, encoded_feature_names)
    return comparison, best_selector, str(best_method), selected_names


def write_llm_family_interpretation(
    profile: dict,
    features: list[str],
    target: str,
    motivation: str | None,
    background_knowledge: str | None,
    output_dir: Path,
    *,
    use_llm: bool,
) -> None:
    scores = deterministic_family_scores(profile)
    payload = {
        "dataset_profile": profile,
        "deterministic_family_scores": scores,
        "note": "Final model candidates are selected by deterministic profile mapping, not by free-form LLM recommendation.",
    }
    if use_llm:
        try:
            from aims_agent.agent import Agent

            prompt = f"""You are a scientific reasoning assistant for materials ML.

Interpret the deterministic dataset profile and explain modeling challenges.
Do not recommend specific model names. Do not change the candidate model list.

Dataset target: {target}
Feature names: {", ".join(features[:80])}
Motivation: {motivation or "N/A"}
Background knowledge: {background_knowledge or "N/A"}

Deterministic profile:
{json.dumps(profile, indent=2)}

Deterministic model-family scores:
{json.dumps(scores, indent=2)}

Return ONLY valid JSON with:
{{
  "dataset_interpretation": "paragraph",
  "modeling_challenges": ["..."],
  "suitable_model_families": [
    {{"family": "regularized_linear|tree_bagging|gradient_boosting|kernel_methods|local_similarity|linear_baseline", "confidence": "low|medium|high", "reason": "..."}}
  ],
  "uncertainty": "paragraph explaining uncertainty and limits"
}}"""
            response = Agent().call_llm(prompt)
            cleaned = re.sub(r"```(?:json)?\s*", "", response).replace("```", "").strip()
            start = cleaned.find("{")
            end = cleaned.rfind("}") + 1
            parsed = json.loads(cleaned[start:end])
            if isinstance(parsed, dict):
                payload["llm_interpretation"] = parsed
        except Exception as exc:
            payload["llm_error"] = str(exc)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "llm_model_family_analysis.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def llm_recommended_models(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    motivation: str | None,
    background_knowledge: str | None,
    *,
    use_llm: bool,
    output_dir: Path | None = None,
    n_models: int = 5,
) -> pd.DataFrame:
    profile = deterministic_dataset_profile(df, features, target)
    recommendations = deterministic_recommendations_from_profile(profile, n_models=n_models)
    if output_dir is not None:
        write_llm_family_interpretation(
            profile,
            features,
            target,
            motivation,
            background_knowledge,
            output_dir,
            use_llm=use_llm,
        )
    return recommendations


def unprefix_params(params: dict) -> dict:
    return {k.replace("model__", "", 1): v for k, v in params.items()}


def normalize_hyperparameter_space(model_name: str, params: dict) -> dict:
    allowed = set(supported_hyperparameter_schema().get(model_name, {}))
    normalized = {}
    for key, values in (params or {}).items():
        key = str(key).replace("model__", "", 1)
        if key not in allowed:
            continue
        if not isinstance(values, list):
            values = [values]
        cleaned = []
        for value in values:
            if key == "estimator_max_depth":
                try:
                    cleaned.append(DecisionTreeRegressor(max_depth=int(value), random_state=RANDOM_STATE))
                except (TypeError, ValueError):
                    continue
            elif key in {"n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "n_neighbors", "p"}:
                if value is None and key == "max_depth":
                    cleaned.append(None)
                    continue
                try:
                    cleaned.append(int(value))
                except (TypeError, ValueError):
                    continue
            elif key in {
                "learning_rate",
                "subsample",
                "colsample_bytree",
                "reg_lambda",
                "min_child_weight",
                "alpha",
                "l1_ratio",
                "C",
                "epsilon",
                "gamma",
                "max_features",
            }:
                if value in {"scale", "auto", "sqrt", "log2"}:
                    cleaned.append(value)
                    continue
                try:
                    cleaned.append(float(value))
                except (TypeError, ValueError):
                    continue
            elif key == "fit_intercept":
                cleaned.append(bool(value))
            else:
                cleaned.append(value)
        if cleaned:
            normalized["estimator" if key == "estimator_max_depth" else key] = cleaned
    return normalized


def llm_hyperparameter_search_spaces(
    selected_model_names: list[str],
    dataset_profile: dict,
    feature_selection_metrics: dict,
    motivation: str | None,
    background_knowledge: str | None,
    *,
    use_llm: bool,
    output_dir: Path,
) -> dict[str, dict]:
    if not use_llm:
        spaces = {model_name: {} for model_name in selected_model_names}
        (output_dir / "llm_hyperparameter_search_spaces.json").write_text(json.dumps(spaces, indent=2), encoding="utf-8")
        return spaces
    schema = {name: supported_hyperparameter_schema()[name] for name in selected_model_names if name in supported_hyperparameter_schema()}
    try:
        from aims_agent.agent import Agent

        prompt = f"""You are choosing hyperparameter search spaces for a scientific tabular regression workflow.

Do not choose models. Only choose hyperparameter values for the listed models.
Use the dataset profile and feature-selection metrics to keep the spaces scientifically reasonable and computationally moderate.
Return JSON only.

Selected models: {selected_model_names}
Motivation: {motivation or "N/A"}
Background knowledge: {background_knowledge or "N/A"}

Dataset profile:
{json.dumps(dataset_profile, indent=2)}

Feature-selection metrics:
{json.dumps(feature_selection_metrics, indent=2)}

Supported hyperparameter schema:
{json.dumps(schema, indent=2)}

Return ONLY a JSON object mapping each selected model name to an object of parameter lists.
Use unprefixed sklearn parameter names. For AdaBoost, use estimator_max_depth instead of estimator.
Example:
{{
  "Random Forest": {{"n_estimators": [200, 500], "max_depth": [null, 6, 12]}},
  "Ridge": {{"alpha": [0.1, 1.0, 10.0]}}
}}
"""
        response = Agent().call_llm(prompt)
        cleaned = re.sub(r"```(?:json)?\s*", "", response).replace("```", "").strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        parsed = json.loads(cleaned[start:end])
        spaces = {}
        for model_name in selected_model_names:
            spaces[model_name] = normalize_hyperparameter_space(model_name, parsed.get(model_name, {}))
        (output_dir / "llm_hyperparameter_search_spaces.json").write_text(
            json.dumps({k: {pk: [str(v) for v in vals] for pk, vals in space.items()} for k, space in spaces.items()}, indent=2),
            encoding="utf-8",
        )
        return spaces
    except Exception as exc:
        spaces = {model_name: {} for model_name in selected_model_names}
        (output_dir / "llm_hyperparameter_search_spaces.json").write_text(
            json.dumps({"error": str(exc), "spaces": spaces}, indent=2),
            encoding="utf-8",
        )
        return spaces


class DefaultFitResult:
    def __init__(self, estimator, X, y):
        self.best_estimator_ = estimator
        self.best_params_ = {}
        self.best_score_ = np.nan
        pred = estimator.predict(X)
        rmse = root_mean_squared_error(y, pred)
        self.cv_results_ = {
            "mean_test_score": np.array([-rmse]),
            "std_test_score": np.array([0.0]),
        }

    def predict(self, X):
        return self.best_estimator_.predict(X)


def finite_or_nan(value) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return np.nan
    return out if np.isfinite(out) else np.nan


def estimate_model_prediction_std(model, X, fallback_std: float) -> np.ndarray:
    """
    Produce per-sample predictive standard deviations for UQ evaluation.

    Native or ensemble uncertainty is used when available. Models without a
    distributional interface get a constant validation-derived fallback so
    uncertainty-toolbox is still run for every candidate.
    """
    fallback_std = max(float(fallback_std), 1e-8)
    n_samples = len(X)
    y_std = None

    if hasattr(model, "predict_distribution"):
        try:
            dist = model.predict_distribution(X)
            y_std = np.asarray(dist.get("std"), dtype=float).reshape(-1)
        except Exception:
            y_std = None

    if y_std is None and "GaussianProcess" in type(model).__name__:
        try:
            _, std = model.predict(X, return_std=True)
            y_std = np.asarray(std, dtype=float).reshape(-1)
        except Exception:
            y_std = None

    if y_std is None and hasattr(model, "estimators_"):
        try:
            estimators = np.asarray(model.estimators_, dtype=object).reshape(-1)
            predictions = np.asarray([est.predict(X) for est in estimators], dtype=float)
            if predictions.ndim == 2 and predictions.shape[1] == n_samples:
                y_std = np.std(predictions, axis=0)
        except Exception:
            y_std = None

    if y_std is None and hasattr(model, "get_booster"):
        try:
            booster = model.get_booster()
            rounds = int(booster.num_boosted_rounds())
            if rounds >= 4:
                dm = DMatrix(X)
                checkpoints = sorted(set(np.linspace(max(2, rounds // 5), rounds, num=min(10, rounds), dtype=int)))
                predictions = np.asarray(
                    [booster.predict(dm, iteration_range=(0, int(r))) for r in checkpoints],
                    dtype=float,
                )
                if predictions.ndim == 2 and predictions.shape[1] == n_samples:
                    y_std = np.std(predictions, axis=0)
        except Exception:
            y_std = None

    if y_std is None or len(y_std) != n_samples or not np.any(np.isfinite(y_std)):
        return np.full(n_samples, fallback_std, dtype=float)

    y_std = np.nan_to_num(y_std, nan=0.0, posinf=0.0, neginf=0.0)
    positive = y_std[y_std > 0]
    if len(positive) == 0:
        return np.full(n_samples, fallback_std, dtype=float)

    # Put ensemble/native dispersion on the validation-error scale.
    scale = fallback_std / max(float(np.mean(positive)), 1e-8)
    y_std = y_std * scale
    floor = max(fallback_std * 0.05, 1e-8)
    return np.maximum(y_std, floor)


def evaluate_uncertainty_for_candidate(
    y_true,
    y_pred,
    y_std,
    *,
    target_std: float,
    test_rmse: float,
    cv_rmse_mean: float,
    cv_rmse_std: float,
) -> dict:
    summary, full_metrics = UncertaintyEvaluator.evaluate_all(y_true, y_pred, y_std, verbose=False)
    coverage = UncertaintyEvaluator.compute_coverage(y_true, y_pred, y_std)

    cal_mae = finite_or_nan(summary.get("calibration_mae"))
    cal_rmse = finite_or_nan(summary.get("calibration_rmse"))
    miscal_area = finite_or_nan(summary.get("miscalibration_area"))
    sharpness = finite_or_nan(summary.get("sharpness"))
    if not np.isfinite(sharpness):
        sharpness = finite_or_nan(full_metrics.get("sharpness", {}).get("sharp"))
    nll = finite_or_nan(summary.get("nll"))
    crps = finite_or_nan(summary.get("crps"))
    interval_score = finite_or_nan(summary.get("interval_score"))

    rmse_norm = float(test_rmse) / max(float(target_std), 1e-8)
    sharpness_norm = sharpness / max(float(target_std), 1e-8) if np.isfinite(sharpness) else 1.0
    cv_instability = float(cv_rmse_std) / max(float(cv_rmse_mean), 1e-8)
    score = (
        0.60 * rmse_norm
        + 0.25 * (miscal_area if np.isfinite(miscal_area) else 1.0)
        + 0.10 * (sharpness_norm if np.isfinite(sharpness_norm) else 1.0)
        + 0.05 * cv_instability
    )

    return {
        "uq_selection_score": float(score),
        "uq_rmse_norm": float(rmse_norm),
        "uq_calibration_mae": cal_mae,
        "uq_calibration_rmse": cal_rmse,
        "uq_miscalibration_area": miscal_area,
        "uq_sharpness": sharpness,
        "uq_sharpness_norm": float(sharpness_norm) if np.isfinite(sharpness_norm) else np.nan,
        "uq_nll": nll,
        "uq_crps": crps,
        "uq_interval_score": interval_score,
        "uq_coverage_68": float(coverage.get(0.68, np.nan)),
        "uq_coverage_95": float(coverage.get(0.95, np.nan)),
        "uq_coverage_99": float(coverage.get(0.99, np.nan)),
        "uq_full_metrics": full_metrics,
    }


def collapse_feature_name(name: str) -> str:
    name = name.replace("num__", "").replace("cat__", "")
    if "_" in name and name.split("_", 1)[0] in {
        "Sample",
        "Treatment",
        "Flyer",
        "Flyer (processed)",
        "Type of experiment",
        "Spall direction",
    }:
        return name.split("_", 1)[0]
    return name


def aggregate_importance(values: np.ndarray, feature_names: np.ndarray) -> pd.DataFrame:
    rows = pd.DataFrame({"encoded_feature": feature_names, "importance": values})
    rows["feature"] = rows["encoded_feature"].map(collapse_feature_name)
    return (
        rows.groupby("feature", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def report_importance_table(material_importance: pd.DataFrame, n: int = 12) -> pd.DataFrame:
    table = material_importance[
        material_importance["source"].astype(str).str.contains("permutation", case=False, na=False)
    ].copy()
    if table.empty:
        table = material_importance.copy()
    cols = ["feature", "importance", "source"]
    if "importance_std" in table.columns:
        cols.insert(2, "importance_std")
    return table.head(n)[cols]


def is_material_parameter(feature: str) -> bool:
    low = feature.lower()
    if low in {"sample", "treatment"}:
        return True
    return any(keyword in low for keyword in MATERIAL_PARAMETER_KEYWORDS)


def uncertainty_strength_label(profile: dict, metrics: pd.DataFrame) -> tuple[str, list[str]]:
    n_rows = profile["sample_size"]
    missing = profile["missingness"]
    skew = abs(profile["target_skewness"])
    outliers = profile["outlier_ratio"]
    best = metrics.iloc[0] if not metrics.empty else pd.Series(dtype=float)
    cv_rmse_mean = float(best.get("cv_rmse_mean", np.nan))
    cv_rmse_std = float(best.get("cv_rmse_std", np.nan))
    cv_relative_std = cv_rmse_std / cv_rmse_mean if cv_rmse_mean and not np.isnan(cv_rmse_mean) else np.nan

    risk_points = 0
    reasons = []
    if n_rows < 100:
        risk_points += 2
        reasons.append("small sample size (<100 rows)")
    elif n_rows < 300:
        risk_points += 1
        reasons.append("moderate sample size (<300 rows)")
    if missing > 0.20:
        risk_points += 2
        reasons.append(f"high missingness ({missing:.1%})")
    elif missing > 0.05:
        risk_points += 1
        reasons.append(f"non-trivial missingness ({missing:.1%})")
    if skew > 1.5:
        risk_points += 2
        reasons.append(f"strong target skewness ({skew:.2f})")
    elif skew > 0.75:
        risk_points += 1
        reasons.append(f"moderate target skewness ({skew:.2f})")
    if outliers > 0.08:
        risk_points += 2
        reasons.append(f"elevated outlier ratio ({outliers:.1%})")
    elif outliers > 0.03:
        risk_points += 1
        reasons.append(f"moderate outlier ratio ({outliers:.1%})")
    if not np.isnan(cv_relative_std):
        if cv_relative_std > 0.25:
            risk_points += 2
            reasons.append(f"high CV RMSE variability ({cv_relative_std:.1%} of mean)")
        elif cv_relative_std > 0.10:
            risk_points += 1
            reasons.append(f"moderate CV RMSE variability ({cv_relative_std:.1%} of mean)")

    if risk_points >= 5:
        label = "weak"
    elif risk_points >= 2:
        label = "moderate"
    else:
        label = "strong"
    if not reasons:
        reasons.append("dataset size, missingness, skewness, outlier rate, and CV variance are all within low-risk ranges")
    return label, reasons


def save_model_mse_plot(metrics: pd.DataFrame, output_dir: Path) -> None:
    if {"cv_mse_mean", "cv_mse_std"}.issubset(metrics.columns):
        chart = metrics.sort_values("cv_mse_mean", ascending=True)
        values = chart["cv_mse_mean"]
        errors = chart["cv_mse_std"]
        ylabel = "CV MSE mean +/- std"
    else:
        chart = metrics.sort_values("test_mse", ascending=True)
        values = chart["test_mse"]
        errors = None
        ylabel = "MSE (Mean Squared Error)"
    palette = ["#376f72", "#6f8f3d", "#b26a3c", "#765d91", "#8b6f47", "#4d668f"]
    plt.figure(figsize=(8.5, 5.2))
    bars = plt.bar(
        chart["model"],
        values,
        yerr=errors,
        capsize=4 if errors is not None else 0,
        color=[palette[i % len(palette)] for i in range(len(chart))],
    )
    plt.ylabel(ylabel)
    plt.xlabel("ML Model")
    plt.title("Model Error Comparison with Cross-Validation Uncertainty")
    plt.xticks(rotation=15, ha="right")
    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison_mse.png", dpi=220)
    plt.close()


def save_target_histogram(y: pd.Series, output_dir: Path) -> None:
    plt.figure(figsize=(8, 5.2))
    plt.hist(y, bins=28, color="#376f72", edgecolor="white", alpha=0.9)
    plt.xlabel(TARGET)
    plt.ylabel("Frequency")
    plt.title(f"Histogram of {TARGET}")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "histogram_target.png", dpi=220)
    plt.close()


def save_target_relationships(df: pd.DataFrame, feature_names: list[str], output_dir: Path) -> list[str]:
    numeric_df = df[[TARGET] + [f for f in feature_names if f in df.columns]].select_dtypes(include=[np.number])
    corr = numeric_df.corr(numeric_only=True)[TARGET].drop(labels=[TARGET], errors="ignore").abs()
    selected = corr.sort_values(ascending=False).head(6).index.tolist()
    if not selected:
        return []

    ncols = 3
    nrows = math.ceil(len(selected) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12.5, 3.8 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, col in zip(axes, selected):
        sub = df[[col, TARGET]].dropna()
        ax.scatter(sub[col], sub[TARGET], s=32, alpha=0.72, color="#376f72", edgecolor="white", linewidth=0.4)
        if len(sub) > 2:
            coef = np.polyfit(sub[col], sub[TARGET], deg=1)
            xs = np.linspace(sub[col].min(), sub[col].max(), 100)
            ax.plot(xs, coef[0] * xs + coef[1], color="#9b3f3f", linewidth=1.6)
        ax.set_xlabel(col)
        ax.set_ylabel(TARGET)
        ax.grid(alpha=0.22)
    for ax in axes[len(selected):]:
        ax.axis("off")
    fig.suptitle(f"Target Relationships: Features vs {TARGET}", y=1.01, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "target_relationships.png", dpi=220, bbox_inches="tight")
    plt.close()
    return selected


def save_correlation_heatmap(df: pd.DataFrame, feature_names: list[str], output_dir: Path) -> pd.DataFrame:
    numeric_df = df[[TARGET] + [f for f in feature_names if f in df.columns]].select_dtypes(include=[np.number])
    corr_to_target = numeric_df.corr(numeric_only=True)[TARGET].drop(labels=[TARGET], errors="ignore").abs()
    selected = [TARGET] + corr_to_target.sort_values(ascending=False).head(13).index.tolist()
    corr = numeric_df[selected].corr(numeric_only=True)
    corr.to_csv(output_dir / "correlation_heatmap_data.csv")

    plt.figure(figsize=(10.5, 8.6))
    image = plt.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(image, label="Pearson correlation")
    labels = [c if len(c) <= 24 else c[:21] + "..." for c in corr.columns]
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
    plt.yticks(range(len(labels)), labels, fontsize=8)
    plt.title("Correlation Heatmap")
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            plt.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7, color="black")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png", dpi=220)
    plt.close()
    return corr


def save_feature_count_analysis(
    best_model,
    best_model_name: str,
    selected_names: np.ndarray,
    selector,
    feature_selection_method: str,
    X_train_encoded,
    X_test_encoded,
    y_train,
    y_test,
    output_dir: Path,
) -> pd.DataFrame:
    if hasattr(selector, "estimator_") and hasattr(selector.estimator_, "feature_importances_"):
        importances = selector.estimator_.feature_importances_
    elif hasattr(selector, "estimator_") and hasattr(selector.estimator_, "coef_"):
        importances = np.abs(np.ravel(selector.estimator_.coef_))
    elif hasattr(selector, "scores_"):
        importances = np.nan_to_num(selector.scores_, nan=0.0, posinf=0.0, neginf=0.0)
    elif feature_selection_method == "PCA_95Variance" and hasattr(selector, "components_"):
        weights = getattr(selector, "explained_variance_ratio_", np.ones(selector.components_.shape[0]))
        importances = np.abs(selector.components_).T @ weights
    else:
        importances = np.asarray(to_dense_array(X_train_encoded)).var(axis=0)
    ranked_indices = np.argsort(importances)[::-1]
    candidate_counts = sorted(set([5, 10, 15, 20, 30, 40, len(selected_names)]))
    candidate_counts = [k for k in candidate_counts if k <= len(ranked_indices)]
    cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    rows = []
    for k in candidate_counts:
        cols = ranked_indices[:k]
        model = clone(best_model)
        cv_scores = cross_validate(
            model,
            X_train_encoded[:, cols],
            y_train,
            cv=cv,
            scoring={"r2": "r2"},
            n_jobs=1,
        )
        model.fit(X_train_encoded[:, cols], y_train)
        pred = model.predict(X_test_encoded[:, cols])
        rows.append(
            {
                "feature_candidates": k,
                "r2": r2_score(y_test, pred),
                "cv_r2_mean": float(np.mean(cv_scores["test_r2"])),
                "cv_r2_std": float(np.std(cv_scores["test_r2"])),
                "model": best_model_name,
            }
        )

    feature_count_df = pd.DataFrame(rows)
    feature_count_df.to_csv(output_dir / "feature_count_r2_analysis.csv", index=False)

    plt.figure(figsize=(7.8, 5))
    plt.errorbar(
        feature_count_df["feature_candidates"],
        feature_count_df["cv_r2_mean"],
        yerr=feature_count_df["cv_r2_std"],
        marker="o",
        linewidth=2.2,
        capsize=4,
        color="#376f72",
        label="CV R2 mean +/- std",
    )
    plt.plot(
        feature_count_df["feature_candidates"],
        feature_count_df["r2"],
        marker="s",
        linewidth=1.4,
        color="#b26a3c",
        alpha=0.85,
        label="Held-out R2",
    )
    plt.xlabel("Features candidates")
    plt.ylabel(r"$R^2$")
    plt.title("Feature Selection with Cross-Validation Uncertainty")
    ymin = min(0, feature_count_df[["r2", "cv_r2_mean"]].min().min() - feature_count_df["cv_r2_std"].max() - 0.05)
    plt.ylim(ymin, 1.02)
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "feature_selection_curve.png", dpi=220)
    plt.close()
    return feature_count_df


def save_hyperparameter_heatmap(search, model_name: str, output_dir: Path) -> pd.DataFrame:
    cv = pd.DataFrame(search.cv_results_)
    cv["mean_cv_rmse"] = -cv["mean_test_score"]
    cv["std_cv_rmse"] = cv["std_test_score"]
    param_cols = [
        col for col in cv.columns
        if col.startswith("param_") and cv[col].nunique(dropna=True) > 1
    ]
    preferred = [
        "param_max_depth",
        "param_learning_rate",
        "param_n_estimators",
        "param_max_features",
        "param_min_samples_leaf",
        "param_estimator",
    ]
    chosen = [col for col in preferred if col in param_cols][:2]
    if len(chosen) < 2:
        chosen.extend([col for col in param_cols if col not in chosen][: 2 - len(chosen)])
    if len(chosen) < 2:
        heatmap_df = cv[["mean_cv_rmse", "std_cv_rmse"]].copy()
        heatmap_df.to_csv(output_dir / "hyperparameter_tuning_heatmap_data.csv", index=False)
        return heatmap_df

    row_param, col_param = chosen
    idx = cv.groupby([row_param, col_param])["mean_cv_rmse"].idxmin()
    heatmap_df = cv.loc[idx, [row_param, col_param, "mean_cv_rmse", "std_cv_rmse"]].sort_values([row_param, col_param])
    heatmap_df.to_csv(output_dir / "hyperparameter_tuning_heatmap_data.csv", index=False)

    pivot = heatmap_df.pivot(index=row_param, columns=col_param, values="mean_cv_rmse")
    std_pivot = heatmap_df.pivot(index=row_param, columns=col_param, values="std_cv_rmse")
    plt.figure(figsize=(7.8, 5.6))
    image = plt.imshow(pivot.values, cmap="viridis_r", aspect="auto")
    plt.colorbar(image, label="CV RMSE (lower is better)")
    plt.xticks(range(len(pivot.columns)), [str(c) for c in pivot.columns])
    plt.yticks(range(len(pivot.index)), [str(i) for i in pivot.index])
    plt.xlabel(col_param.replace("param_", ""))
    plt.ylabel(row_param.replace("param_", ""))
    plt.title(f"Hyperparameter Tuning Heatmap - {model_name}")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.values[i, j]
            if not np.isnan(value):
                std_value = std_pivot.values[i, j]
                label = f"{value:.2f}\n+/-{std_value:.2f}" if not np.isnan(std_value) else f"{value:.2f}"
                plt.text(j, i, label, ha="center", va="center", color="white", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "hyperparameter_tuning_heatmap.png", dpi=220)
    plt.close()

    plt.figure(figsize=(7.8, 5.6))
    image = plt.imshow(std_pivot.values, cmap="magma", aspect="auto")
    plt.colorbar(image, label="CV RMSE std")
    plt.xticks(range(len(std_pivot.columns)), [str(c) for c in std_pivot.columns])
    plt.yticks(range(len(std_pivot.index)), [str(i) for i in std_pivot.index])
    plt.xlabel(col_param.replace("param_", ""))
    plt.ylabel(row_param.replace("param_", ""))
    plt.title(f"Hyperparameter CV Uncertainty - {model_name}")
    for i in range(std_pivot.shape[0]):
        for j in range(std_pivot.shape[1]):
            value = std_pivot.values[i, j]
            if not np.isnan(value):
                plt.text(j, i, f"{value:.2f}", ha="center", va="center", color="white", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_dir / "hyperparameter_tuning_std_heatmap.png", dpi=220)
    plt.close()
    return heatmap_df


def save_actual_vs_predicted(y_test, y_pred, best_model_name: str, output_dir: Path) -> None:
    y_true = np.asarray(y_test, dtype=float)
    y_hat = np.asarray(y_pred, dtype=float)
    residuals = y_true - y_hat
    residual_std = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0
    interval_half_width = 1.96 * residual_std
    interval_df = pd.DataFrame(
        {
            "actual": y_true,
            "predicted": y_hat,
            "residual": residuals,
            "prediction_interval_lower_approx": y_hat - interval_half_width,
            "prediction_interval_upper_approx": y_hat + interval_half_width,
            "interval_method": "holdout_residual_normal_approximation",
        }
    )
    interval_df.to_csv(output_dir / "prediction_intervals_and_residuals.csv", index=False)

    plt.figure(figsize=(6.2, 6))
    plt.errorbar(
        y_true,
        y_hat,
        yerr=interval_half_width if interval_half_width > 0 else None,
        fmt="o",
        markersize=6,
        alpha=0.68,
        color="#376f72",
        ecolor="#8aa7a9",
        elinewidth=1,
        capsize=2,
        markeredgecolor="white",
        markeredgewidth=0.5,
    )
    low = min(float(np.min(y_true)), float(np.min(y_hat - interval_half_width)))
    high = max(float(np.max(y_true)), float(np.max(y_hat + interval_half_width)))
    plt.plot([low, high], [low, high], color="#9b3f3f", linewidth=2, label="Ideal prediction")
    plt.xlabel(f"Actual {TARGET}")
    plt.ylabel(f"Predicted {TARGET}")
    plt.title(f"Actual vs Predicted with Approx. 95% Intervals - {best_model_name}")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "predicted_vs_actual_parity_plot.png", dpi=220)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    axes[0].scatter(y_hat, residuals, s=44, alpha=0.74, color="#376f72", edgecolor="white", linewidth=0.5)
    axes[0].axhline(0, color="#9b3f3f", linewidth=1.7)
    axes[0].axhline(interval_half_width, color="#b26a3c", linewidth=1, linestyle="--")
    axes[0].axhline(-interval_half_width, color="#b26a3c", linewidth=1, linestyle="--")
    axes[0].set_xlabel(f"Predicted {TARGET}")
    axes[0].set_ylabel("Residual (actual - predicted)")
    axes[0].set_title("Residuals vs Predictions")
    axes[0].grid(alpha=0.22)

    axes[1].hist(residuals, bins=min(18, max(6, len(residuals) // 2)), color="#765d91", edgecolor="white", alpha=0.9)
    axes[1].axvline(0, color="#9b3f3f", linewidth=1.7)
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Residual Distribution")
    axes[1].grid(axis="y", alpha=0.22)
    fig.suptitle(f"Residual Analysis - {best_model_name}", y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "residual_analysis.png", dpi=220, bbox_inches="tight")
    plt.close()


def save_epoch_curves(model_name: str, model_template, best_params: dict, X_train_selected, X_test_selected, y_train, y_test, output_dir: Path) -> pd.DataFrame:
    params = dict(best_params)
    params.pop("n_estimators", None)
    n_estimators = int(best_params.get("n_estimators", 100))
    candidate_epochs = [10, 20, 40, 80, 120, 200, 300, 400, n_estimators]
    epochs = sorted({epoch for epoch in candidate_epochs if 1 <= epoch <= n_estimators})
    if not epochs or epochs[-1] != n_estimators:
        epochs.append(n_estimators)

    rows = []
    for epoch in epochs:
        epoch_model = clone(model_template).set_params(**best_params)
        epoch_model.set_params(n_estimators=epoch)
        epoch_model.fit(X_train_selected, y_train)
        train_pred = epoch_model.predict(X_train_selected)
        test_pred = epoch_model.predict(X_test_selected)
        rows.append(
            {
                "epoch": epoch,
                "train_rmse": root_mean_squared_error(y_train, train_pred),
                "test_rmse": root_mean_squared_error(y_test, test_pred),
                "test_r2": r2_score(y_test, test_pred),
            }
        )
    epoch_df = pd.DataFrame(rows)
    epoch_df.to_csv(output_dir / "epoch_curves.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(epoch_df["epoch"], epoch_df["train_rmse"], label="Train RMSE", color="#376f72")
    plt.plot(epoch_df["epoch"], epoch_df["test_rmse"], label="Test RMSE", color="#b26a3c")
    plt.xlabel("Epochs / Boosting rounds")
    plt.ylabel("Loss (RMSE)")
    plt.title(f"Loss vs Epochs - {model_name}")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "loss_vs_epochs.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epoch_df["epoch"], epoch_df["test_r2"], color="#6f8f3d", linewidth=2)
    plt.xlabel("Epochs / Boosting rounds")
    plt.ylabel(r"$R^2$ (Regression accuracy proxy)")
    plt.title(f"Accuracy vs Epochs - {model_name}")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_r2_vs_epochs.png", dpi=220)
    plt.close()
    return epoch_df


def write_report(
    output_dir: Path,
    metrics: pd.DataFrame,
    best_model_name: str,
    best_params: dict,
    shap_df: pd.DataFrame | None,
    material_importance: pd.DataFrame,
    feature_count_df: pd.DataFrame,
    unit_corrections: pd.DataFrame,
    model_recommendations: pd.DataFrame,
) -> None:
    top_material = report_importance_table(material_importance, n=10)
    metric_cols = [
        "model",
        "test_mse",
        "test_rmse",
        "test_r2",
        "cv_rmse_mean",
        "cv_rmse_std",
        "cv_r2_mean",
        "cv_r2_std",
        "uq_selection_score",
        "uq_miscalibration_area",
        "uq_sharpness",
        "uq_coverage_95",
    ]
    metric_cols = [col for col in metric_cols if col in metrics.columns]
    report = [
        "# Machine Learning Strategy Analysis Report",
        "",
        f"Run folder: `{output_dir}`",
        f"Dataset: `{DATA_PATH}`",
        f"Target: `{TARGET}`",
        "",
        "## Data Cleaning",
        "",
        f"- Dataset path: `{DATA_PATH}`",
        f"- Likely MPa target values converted to GPa: {len(unit_corrections)}",
        "",
        "## Model Comparison",
        "",
        "Models compared were selected from the recommendation table below.",
        "",
        model_recommendations.to_markdown(index=False),
        "",
        metrics[metric_cols].to_markdown(index=False),
        "",
        f"Best model: **{best_model_name}**",
        f"Best parameters: `{best_params}`",
        "",
        "![Model MSE comparison](model_comparison_mse.png)",
        "",
        "## Feature Selection",
        "",
        feature_count_df.to_markdown(index=False),
        "",
        "![Feature count R2](feature_selection_curve.png)",
        "",
        "## Hyperparameter Tuning",
        "",
        f"The heatmap shows CV RMSE for sampled hyperparameters of the best model, `{best_model_name}`. Lower is better.",
        "",
        "![Hyperparameter heatmap](hyperparameter_tuning_heatmap.png)",
        "",
        "## Prediction Quality",
        "",
        "![Actual vs predicted](predicted_vs_actual_parity_plot.png)",
        "",
        "## Training Curves",
        "",
        "This is a regression task, so classification accuracy is not defined. The accuracy-style epoch plot uses test-set R2.",
        "",
        "![Loss vs epochs](loss_vs_epochs.png)",
        "",
        "![Accuracy vs epochs](accuracy_r2_vs_epochs.png)",
        "",
        "## Feature Importance / SHAP",
        "",
        top_material.to_markdown(index=False),
        "",
        "![Best model feature importance](feature_importance_material_parameters.png)",
        "",
        "## Conclusion",
        "",
        f"- Best predictive model: **{best_model_name}**.",
        "- The feature-importance table identifies the strongest predictors for the selected target.",
    ]
    if shap_df is None:
        report.append("- Native SHAP package was not installed; model feature importance and permutation importance were used.")
    (output_dir / "report.md").write_text("\n".join(report), encoding="utf-8")


def dataframe_to_html_table(df: pd.DataFrame, float_format: str = "{:.4f}") -> str:
    rows = []
    headers = "".join(f"<th>{escape(str(col))}</th>" for col in df.columns)
    rows.append(f"<tr>{headers}</tr>")
    for _, row in df.iterrows():
        cells = []
        for value in row:
            if isinstance(value, float):
                text = float_format.format(value)
            else:
                text = str(value)
            cells.append(f"<td>{escape(text)}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")
    return "<table>" + "\n".join(rows) + "</table>"


def dataframe_to_markdown(df: pd.DataFrame, float_format: str = ".4f") -> str:
    safe = df.copy()
    object_cols = safe.select_dtypes(include=["object"]).columns
    for col in object_cols:
        safe[col] = (
            safe[col]
            .astype(str)
            .str.replace("\\", "\\\\", regex=False)
            .str.replace("|", "\\|", regex=False)
            .str.replace("\n", " ", regex=False)
        )
    return safe.to_markdown(index=False, floatfmt=float_format)


def build_dataset_bundle(df: pd.DataFrame, features: list[str], target: str, data_path: Path) -> DatasetBundle:
    schema = DatasetSchema(
        features=features,
        target=target,
        units={},
        source=str(data_path),
        description=f"User-provided tabular dataset loaded from {data_path}.",
        shape=df.shape,
        dtypes={col: str(dtype) for col, dtype in df.dtypes.items()},
    )
    return DatasetBundle(df=df, schema=schema)


def build_strategy_context(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    output_dir: Path,
    motivation: str | None,
    background_knowledge: str | None,
    *,
    use_llm: bool,
):
    bundle = build_dataset_bundle(df, features, target, DATA_PATH)
    agent = None
    if use_llm:
        try:
            from aims_agent.agent import Agent

            agent = Agent()
        except Exception:
            agent = None
    profile_dir = output_dir / "dataset_profile"
    profile = profile_dataset(bundle, task_type="regression", output_dir=profile_dir)
    strategy = formulate_strategy(
        profile,
        agent=agent,
        use_llm=use_llm and agent is not None,
        output_dir=profile_dir,
        run_context={
            "motivation": motivation or "",
            "background_knowledge": background_knowledge or "",
            "dataset": str(DATA_PATH),
            "target": target,
            "llm": "enabled" if use_llm and agent is not None else "disabled or unavailable",
            "model_mode": "LLM-guided" if use_llm and agent is not None else "deterministic fallback",
        },
    )
    return profile, strategy


def fallback_graph_explanations(
    target: str,
    best_model_name: str,
    metrics: pd.DataFrame,
    material_importance: pd.DataFrame,
    feature_count_df: pd.DataFrame,
    target_relationship_features: list[str],
) -> dict[str, str]:
    best = metrics.iloc[0]
    top_features = ", ".join(material_importance.head(5)["feature"].astype(str).tolist())
    relationship_text = ", ".join(target_relationship_features) or "the strongest available numeric predictors"
    feature_sort_col = "cv_r2_mean" if "cv_r2_mean" in feature_count_df.columns else "r2"
    best_feature_count = feature_count_df.sort_values(feature_sort_col, ascending=False).iloc[0]
    return {
        "histogram": (
            f"The histogram shows the empirical distribution of `{target}`. Use it to see whether the target is "
            "balanced, skewed, long-tailed, or dominated by outliers; that shape affects the reliability of RMSE "
            "and whether robust preprocessing may be needed."
        ),
        "data_distribution": (
            "The data-distribution plot summarizes the target and feature distributions used by the strategy profiler. "
            "It helps identify skewed variables, sparse ranges, heavy tails, and possible preprocessing needs before "
            "model training."
        ),
        "target_relationships": (
            f"The target-relationship plot compares `{target}` against the most target-correlated numeric features "
            f"({relationship_text}). It shows whether the strongest pairwise signals look linear, nonlinear, clustered, "
            "or outlier-driven before multivariate modeling."
        ),
        "model_comparison": (
            f"`{best_model_name}` is selected as the best evaluated model by the UQ-aware selection score "
            f"({best.get('uq_selection_score', np.nan):.4f}; lower is better), which combines held-out RMSE, "
            "uncertainty-toolbox miscalibration, sharpness, and cross-validation instability. Its held-out RMSE is "
            f"{best['test_rmse']:.4f}; cross-validation gives RMSE {best.get('cv_rmse_mean', np.nan):.4f} +/- "
            f"{best.get('cv_rmse_std', np.nan):.4f} and R2 {best.get('cv_r2_mean', np.nan):.4f} +/- "
            f"{best.get('cv_r2_std', np.nan):.4f}."
        ),
        "parity": (
            "The parity plot compares held-out actual values to predictions and adds approximate prediction intervals "
            "derived from held-out residual variability. The residual-analysis plot should be used with it to check "
            "bias, heteroscedasticity, and target regions where uncertainty is larger."
        ),
        "feature_importance": (
            f"The feature-importance plot identifies which inputs most influence the selected best model. The leading "
            f"signals in this run are {top_features}, so these are the parameters most associated with `{target}` "
            "under the fitted model. Error bars represent permutation-importance variability across repeats, so wide "
            "intervals mean the ranking is less stable."
        ),
        "correlation_heatmap": (
            "The correlation heatmap shows linear relationships among the target and top numeric features. Strong "
            "feature-feature correlations indicate redundant descriptors; strong target correlations indicate useful "
            "linear signals but do not rule out nonlinear effects."
        ),
        "hyperparameter_heatmap": (
            f"The hyperparameter heatmap summarizes cross-validated RMSE across sampled settings for `{best_model_name}`. "
            "Each cell stores both mean and standard deviation across folds; low mean and low standard deviation indicate "
            "a configuration that is both accurate and stable."
        ),
        "feature_selection": (
            f"The feature-selection curve shows CV R2 mean +/- standard deviation as the number of candidate features changes. In this run, the "
            f"best CV R2 mean on that curve is {best_feature_count.get('cv_r2_mean', best_feature_count['r2']):.4f} using {int(best_feature_count['feature_candidates'])} "
            "candidate features, which indicates whether extra descriptors help or add noise."
        ),
        "loss_epochs": (
            f"The loss curve retrains `{best_model_name}` across increasing estimator counts and tracks train/test RMSE. "
            "A falling test curve indicates learning; a widening train-test gap indicates overfitting."
        ),
        "r2_epochs": (
            "The R2-vs-epochs curve is the regression analogue of an accuracy curve. Higher values mean the model explains "
            "more held-out target variance as training capacity increases."
        ),
    }


def llm_graph_explanations(
    profile,
    strategy,
    metrics: pd.DataFrame,
    material_importance: pd.DataFrame,
    feature_count_df: pd.DataFrame,
    target_relationship_features: list[str],
    best_model_name: str,
    motivation: str | None,
    background_knowledge: str | None,
    *,
    use_llm: bool,
) -> dict[str, str]:
    fallback = fallback_graph_explanations(
        str(profile.metadata["target"]),
        best_model_name,
        metrics,
        material_importance,
        feature_count_df,
        target_relationship_features,
    )
    if not use_llm:
        return fallback
    try:
        from aims_agent.agent import Agent

        prompt = f"""You are writing graph explanations for a materials-science ML strategy report.

Dataset profile:
{profile.summary_text}

LLM strategy interpretation:
{strategy.llm_interpretation}

User motivation:
{motivation or "N/A"}

Background knowledge:
{background_knowledge or "N/A"}

Model metrics:
{metrics[["model", "test_mse", "test_rmse", "test_r2", "cv_rmse_mean", "cv_rmse_std", "cv_r2_mean", "cv_r2_std"]].to_string(index=False)}

Top feature importance:
{material_importance.head(12).to_string(index=False)}

Feature selection curve:
{feature_count_df.to_string(index=False)}

Best evaluated model: {best_model_name}
Target relationship features: {", ".join(target_relationship_features)}

Return ONLY valid JSON with exactly these string keys:
histogram, data_distribution, target_relationships, model_comparison, parity, feature_importance,
correlation_heatmap, hyperparameter_heatmap, feature_selection, loss_epochs, r2_epochs.

Each value should be 2-4 sentences explaining what the graph shows in this dataset context,
why it matters, and how to interpret it. Do not invent numbers not shown above."""
        response = Agent().call_llm(prompt)
        cleaned = re.sub(r"```(?:json)?\s*", "", response).replace("```", "").strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        parsed = json.loads(cleaned[start:end])
        if isinstance(parsed, dict):
            return {key: str(parsed.get(key, fallback[key])).strip() or fallback[key] for key in fallback}
    except Exception:
        return fallback
    return fallback


def figure_markdown(image_name: str, title: str, explanation: str) -> list[str]:
    return [
        f"### {title}",
        "",
        f"**What it shows and why it matters:** {explanation}",
        "",
        f'<figure><img src="{image_name}" alt="{title}" style="max-width:100%;height:auto;"><figcaption>{escape(title)}</figcaption></figure>',
        "",
    ]


def optional_figure_markdown(output_dir: Path, image_name: str, title: str, explanation: str) -> list[str]:
    if not (output_dir / image_name).exists():
        return []
    return figure_markdown(image_name, title, explanation)


def collect_self_correction_reports(output_dir: Path) -> list[Path]:
    patterns = ("*retry*", "*self_correction*")
    found: list[Path] = []
    for pattern in patterns:
        found.extend(sorted(output_dir.rglob(pattern)))
    found.extend(sorted((output_dir / "generated_feature_selectors").glob("*.py")) if (output_dir / "generated_feature_selectors").exists() else [])
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in found:
        if path in seen or path.is_dir():
            continue
        seen.add(path)
        unique.append(path)
    return unique


def generated_code_appendix_lines(output_dir: Path) -> list[str]:
    artifacts = [
        "llm_model_family_analysis.json",
        "llm_feature_selection_recommendations.json",
        "llm_hyperparameter_search_spaces.json",
        "feature_selection_key_metrics.json",
        "model_recommendations.csv",
        "feature_selection_recommendations.csv",
        "feature_selection_method_comparison.csv",
        "model_metrics.csv",
        "best_params.json",
        "selected_features.csv",
        "material_parameter_importance.csv",
        "permutation_importance.csv",
        "prediction_intervals_and_residuals.csv",
        "feature_count_r2_analysis.csv",
        "hyperparameter_tuning_heatmap_data.csv",
        "hyperparameter_tuning_std_heatmap.png",
        "residual_analysis.png",
        "uncertainty_assessment.json",
        "epoch_curves.csv",
        "summary.json",
    ]
    existing_artifacts = [name for name in artifacts if (output_dir / name).exists()]
    lines = [
        "## Appendix: Generated Code and Artifacts",
        "",
        "The integrated workflow trains supported estimators through deterministic Python modules. If the LLM recommends a feature-selection method outside the executable catalog, the agent attempts to generate a sklearn-compatible `GeneratedFeatureSelector`; generated modules and self-correction logs are saved in the run folder.",
        "",
        "Relevant code paths:",
        "",
        "- `aims_agent/cli.py` - routes tabular datasets into the integrated analysis workflow.",
        "- `aims_agent/model_strategy_analysis.py` - runs deterministic dataset profiling, model-family mapping, training, evaluation, plots, feature importance, and final report assembly.",
        "- `aims_agent/data_analyzer.py` - profiles the dataset, calls the LLM strategy step, and renders the shared HTML report format.",
        "- `aims_agent/model_selector.py` - contains older LLM model-selection utilities used elsewhere in AIMS Agent; this workflow does not use them for final candidate selection.",
        "- `scripts/analyze_spall_strength_models.py` - compatibility wrapper that calls the integrated analysis module.",
        "",
        "Generated run artifacts:",
    ]
    if existing_artifacts:
        lines.extend([f"- `{name}`" for name in existing_artifacts])
    else:
        lines.append("- No tabular artifacts were found in this run folder.")
    lines.append("")
    return lines


def dataset_profile_appendix_lines(profile) -> list[str]:
    return [
        "## Appendix: Dataset Profile Summary",
        "",
        "```text",
        profile.summary_text,
        "```",
        "",
    ]


def self_correction_appendix_lines(output_dir: Path) -> list[str]:
    reports = collect_self_correction_reports(output_dir)
    lines = [
        "## Appendix: Self-Correction Loop",
        "",
        "The self-correction loop is part of AIMS Agent's code-generation/debugging workflow. This integrated tabular-analysis run did not require generated estimator code or repair retries, so no self-correction loop was invoked for the final model training path.",
        "",
    ]
    if reports:
        lines.append("Detected retry or self-correction reports in this run folder:")
        lines.append("")
        for path in reports:
            lines.append(f"- `{path.relative_to(output_dir)}`")
            snippet = path.read_text(encoding="utf-8", errors="replace")[:2000].strip()
            if snippet:
                lines.extend(["", "```text", snippet, "```", ""])
    else:
        lines.append("No retry or self-correction reports were generated for this run.")
        lines.append("")
    return lines


def write_strategy_html_report(
    output_dir: Path,
    metrics: pd.DataFrame,
    best_model_name: str,
    best_params: dict,
    material_importance: pd.DataFrame,
    feature_count_df: pd.DataFrame,
    target_relationship_features: list[str],
    unit_corrections: pd.DataFrame,
    model_recommendations: pd.DataFrame,
    feature_selection_metrics: dict | None,
    feature_selection_recommendations: pd.DataFrame | None,
    feature_selection_comparison: pd.DataFrame | None,
    selected_feature_selection_method: str | None,
    hyperparameter_spaces: dict | None,
    profile,
    strategy,
    graph_explanations: dict[str, str],
    motivation: str | None = None,
    background_knowledge: str | None = None,
    uncertainty_label: str | None = None,
    uncertainty_reasons: list[str] | None = None,
) -> None:
    top_material = report_importance_table(material_importance, n=12)
    target_features = ", ".join(target_relationship_features)
    best_params_text = json.dumps(best_params, indent=2, default=str)
    uncertainty_reasons = uncertainty_reasons or []
    note = (
        "This is a regression task; classification accuracy is not defined. "
        "The Accuracy / R2 vs Epochs figure uses test-set R2 as the accuracy-style metric."
    )
    metric_cols = [
        "model",
        "test_mse",
        "test_rmse",
        "test_r2",
        "cv_rmse_mean",
        "cv_rmse_std",
        "cv_r2_mean",
        "cv_r2_std",
    ]
    uq_metric_cols = [
        "model",
        "uq_selection_score",
        "test_rmse",
        "test_r2",
        "uq_calibration_mae",
        "uq_miscalibration_area",
        "uq_sharpness",
        "uq_nll",
        "uq_crps",
        "uq_coverage_68",
        "uq_coverage_95",
        "cv_rmse_std",
    ]
    metric_cols = [col for col in metric_cols if col in metrics.columns]
    uq_metric_cols = [col for col in uq_metric_cols if col in metrics.columns]
    report_lines = [
        "# Machine Learning Strategy Report",
        "",
        "## Motivation and Background",
        "",
        f"- Motivation: {motivation or 'N/A'}",
        f"- Background knowledge: {background_knowledge or 'N/A'}",
        "",
        "## Dataset Summary and Analysis",
        "",
        f"- Dataset file: `{DATA_PATH}`",
        f"- Target column: `{TARGET}`",
        f"- Rows: {profile.row_count}",
        f"- Columns: {profile.column_count}",
        f"- Feature count: {len(profile.feature_profiles)}",
        "",
        "### LLM Dataset Analysis",
        "",
        strategy.llm_interpretation,
        "",
        "## Feature Selection Key Metrics",
        "",
        "Before asking the LLM to recommend feature-selection methods, the agent computes deterministic metrics that describe whether the data favors projection, sparse linear selection, univariate filtering, or nonlinear embedded selection.",
        "",
        "```json",
        json.dumps(feature_selection_metrics or {}, indent=2, default=str),
        "```",
        "",
        "## Feature Selection Method Recommendations",
        "",
        "Before model training, the AIMS Agent asks the LLM to analyze the deterministic dataset profile and recommend feature-selection or dimensionality-reduction methods. PCA is included as an unsupervised projection method: it reduces correlated inputs into principal components, but it does not preserve direct original-feature interpretability in the same way as filter or embedded selectors. If the LLM recommends a method outside the executable catalog, the agent attempts to generate a sklearn-compatible selector with codegen and then evaluates it by cross-validation. The final selected method is still chosen by validation results.",
        "",
        dataframe_to_markdown(feature_selection_recommendations) if feature_selection_recommendations is not None else "N/A",
        "",
        "### Feature Selection Method Evaluation",
        "",
        f"Selected feature-selection method: `{selected_feature_selection_method or 'N/A'}`.",
        "",
        dataframe_to_markdown(feature_selection_comparison) if feature_selection_comparison is not None else "N/A",
        "",
        "## Data Cleaning Notes",
        "",
        f"- Likely MPa target values converted to GPa: {len(unit_corrections)}",
        f"- {note}",
        "",
        "## Data Distribution and Target Relationships",
        "",
        f"Target relationship plots use these strongest target-related numeric features: {target_features or 'N/A'}.",
        "",
        *figure_markdown("histogram_target.png", "Target Histogram", graph_explanations["histogram"]),
        *figure_markdown("dataset_profile/data_distribution.png", "Data Distribution", graph_explanations["data_distribution"]),
        *figure_markdown("target_relationships.png", "Target Relationships", graph_explanations["target_relationships"]),
        "## LLM-Guided Model-Family Reasoning",
        "",
        "The AIMS Agent first computes a deterministic dataset profile, then optionally asks the LLM to interpret the profile, modeling challenges, confidence, and uncertainty. The final candidate model list is not freely chosen by the LLM; it is produced by a deterministic mapping from model-family tags to supported estimators, so the same dataset and target produce the same candidates.",
        "",
        dataframe_to_markdown(model_recommendations),
        "",
        "## Model Comparison",
        "",
        graph_explanations["model_comparison"],
        "",
        dataframe_to_markdown(metrics[metric_cols]),
        "",
        "### UQ Evaluation and Model Selection",
        "",
        "Every candidate model is evaluated with uncertainty-toolbox. The final model is selected by `uq_selection_score`, where lower is better; the score combines normalized RMSE, miscalibration area, sharpness, and cross-validation instability. The raw UQ outputs are saved in `uncertainty_model_selection.csv` and `uncertainty_model_selection_full.json`.",
        "",
        dataframe_to_markdown(metrics[uq_metric_cols]) if uq_metric_cols else "N/A",
        "",
        "### LLM Hyperparameter Search Spaces",
        "",
        "The hyperparameter search spaces below are generated by the LLM from the dataset profile, feature-selection metrics, and selected candidate models. If the LLM is disabled or unavailable, the object is empty and models are evaluated with their default estimator settings.",
        "",
        "```json",
        json.dumps(hyperparameter_spaces or {}, indent=2, default=str),
        "```",
        "",
        f"Best evaluated model: **{best_model_name}**",
        "",
        "Best hyperparameters:",
        "",
        "```json",
        best_params_text,
        "```",
        "",
        *figure_markdown("model_comparison_mse.png", "Model Comparison - MSE", graph_explanations["model_comparison"]),
        *figure_markdown("model_rmse_comparison.png", "Model Comparison - RMSE", graph_explanations["model_comparison"]),
        "## Predicted vs Actual (Parity Plot)",
        "",
        *figure_markdown("predicted_vs_actual_parity_plot.png", "Predicted vs Actual", graph_explanations["parity"]),
        *optional_figure_markdown(output_dir, "residual_analysis.png", "Residual Analysis", "Residuals show where the model over- or under-predicts. A centered, pattern-free residual plot supports stronger conclusions; trends, fanning, or heavy-tailed residuals indicate weaker prediction reliability and wider practical uncertainty."),
        "## Feature Importance: Parameters Affecting the Target",
        "",
        graph_explanations["feature_importance"],
        "",
        dataframe_to_markdown(top_material),
        "",
        *figure_markdown("feature_importance_material_parameters.png", "Feature Importance", graph_explanations["feature_importance"]),
        "## Correlation Heatmap",
        "",
        *figure_markdown("correlation_heatmap.png", "Correlation Heatmap", graph_explanations["correlation_heatmap"]),
        "## Hyperparameter Tuning Heatmap",
        "",
        *figure_markdown("hyperparameter_tuning_heatmap.png", "Hyperparameter Tuning Heatmap", graph_explanations["hyperparameter_heatmap"]),
        *optional_figure_markdown(output_dir, "hyperparameter_tuning_std_heatmap.png", "Hyperparameter Tuning CV Standard Deviation", "This companion heatmap shows fold-to-fold variability for the same hyperparameter grid. Low values indicate stable tuning behavior; high values mean the apparent best setting is less certain."),
        "## Feature Selection Curve",
        "",
        dataframe_to_markdown(feature_count_df),
        "",
        *figure_markdown("feature_selection_curve.png", "Feature Selection Curve", graph_explanations["feature_selection"]),
        "## Loss vs Epochs",
        "",
        *figure_markdown("loss_vs_epochs.png", "Loss vs Epochs", graph_explanations["loss_epochs"]),
        "## Accuracy / R2 vs Epochs",
        "",
        *figure_markdown("accuracy_r2_vs_epochs.png", "Accuracy / R2 vs Epochs", graph_explanations["r2_epochs"]),
        "## Why This Model Is Best",
        "",
        graph_explanations["model_comparison"],
        "",
        "The final model choice is based on held-out evaluation after all recommended candidates are trained and tuned. The selected model is therefore not hard-coded; it is the model that performed best on the evaluation metrics in this run.",
        "",
        "## Uncertainty Notes",
        "",
        f"Conclusion strength: {uncertainty_label or 'not assessed'}.",
        "",
        *(f"- {reason}" for reason in uncertainty_reasons),
        "",
        "Strong conclusions require enough samples, low missingness, limited target skew, stable residuals, and low cross-validation variance. Moderate or weak labels mean the model can still be useful, but feature rankings and performance estimates should be treated as less definitive until more data or validation is available.",
        "",
        *dataset_profile_appendix_lines(profile),
        *generated_code_appendix_lines(output_dir),
        *self_correction_appendix_lines(output_dir),
    ]
    report_html = _render_report_html("\n".join(report_lines), title="Machine Learning Strategy Report")
    (output_dir / "strategy_report.html").write_text(report_html, encoding="utf-8")


def run_model_strategy_analysis(
    data_path: str | Path = DATA_PATH,
    target: str | None = None,
    output_root: str | Path = OUTPUT_ROOT,
    motivation: str | None = None,
    background_knowledge: str | None = None,
    use_llm: bool = True,
) -> Path:
    global DATA_PATH, OUTPUT_ROOT, OUTPUT_DIR, TARGET

    DATA_PATH = Path(data_path)
    if target:
        TARGET = target
    OUTPUT_ROOT = Path(output_root)
    OUTPUT_DIR = OUTPUT_ROOT / datetime.now().strftime("run_%Y%m%d_%H%M%S")

    warnings.filterwarnings("ignore", category=UserWarning)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    UncertaintyEvaluator.check_availability()

    df = load_tabular_dataset(DATA_PATH)
    df.columns = [clean_column_name(c) for c in df.columns]
    TARGET = infer_target_column(df, target)

    df = coerce_numeric_like_columns(df, TARGET)
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)

    likely_mpa_mask = df[TARGET] > 50 if "spall" in TARGET.lower() else pd.Series(False, index=df.index)
    correction_cols = [col for col in ["Sample", TARGET] if col in df.columns]
    unit_corrections = df.loc[likely_mpa_mask, correction_cols].copy()
    if not unit_corrections.empty:
        unit_corrections["corrected_spall_gpa"] = unit_corrections[TARGET] / 1000.0
        unit_corrections.to_csv(OUTPUT_DIR / "target_unit_corrections.csv", index=False)
        df.loc[likely_mpa_mask, TARGET] = df.loc[likely_mpa_mask, TARGET] / 1000.0

    features = [col for col in df.columns if col not in (DROP_COLUMNS | {TARGET})]
    X = df[features].copy()
    y = df[TARGET].astype(float)
    target_std = float(y.std(ddof=1)) if len(y) > 1 else 1.0
    deterministic_profile = deterministic_dataset_profile(df, features, TARGET)
    save_target_histogram(y, OUTPUT_DIR)
    target_relationship_features = save_target_relationships(df, features, OUTPUT_DIR)
    save_correlation_heatmap(df, features, OUTPUT_DIR)
    model_recommendations = llm_recommended_models(
        df=df,
        features=features,
        target=TARGET,
        motivation=motivation,
        background_knowledge=background_knowledge,
        use_llm=use_llm,
        output_dir=OUTPUT_DIR,
    )
    model_recommendations.to_csv(OUTPUT_DIR / "model_recommendations.csv", index=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    preprocessor = build_preprocessor(X)
    X_train_encoded = preprocessor.fit_transform(X_train)
    X_test_encoded = preprocessor.transform(X_test)
    encoded_feature_names = preprocessor.get_feature_names_out()

    feature_selection_metrics = feature_selection_key_metrics(
        df,
        features,
        TARGET,
        X_train_encoded,
        encoded_feature_names,
    )
    (OUTPUT_DIR / "feature_selection_key_metrics.json").write_text(
        json.dumps(feature_selection_metrics, indent=2),
        encoding="utf-8",
    )
    feature_selection_recommendations = llm_feature_selection_recommendations(
        deterministic_profile,
        feature_selection_metrics,
        features,
        TARGET,
        motivation,
        background_knowledge,
        use_llm=use_llm,
        output_dir=OUTPUT_DIR,
    )
    feature_selection_recommendations.to_csv(OUTPUT_DIR / "feature_selection_recommendations.csv", index=False)
    feature_selection_comparison, selector, selected_feature_selection_method, selected_names = compare_feature_selection_methods(
        feature_selection_recommendations,
        X_train_encoded,
        y_train,
        encoded_feature_names,
        OUTPUT_DIR,
        feature_selection_metrics,
        use_llm=use_llm,
    )
    X_train_prepared = to_dense_array(X_train_encoded)
    X_test_prepared = to_dense_array(X_test_encoded)
    X_train_selected = selector.transform(X_train_prepared)
    X_test_selected = selector.transform(X_test_prepared)

    results = []
    uq_full_metrics_by_model = {}
    best_estimators = {}
    best_searches = {}

    spaces = model_registry()
    selected_model_names = [
        model for model in model_recommendations["model"].tolist() if model in spaces
    ]
    if not selected_model_names:
        fallback = default_model_recommendations()
        fallback["reason"] = fallback["reason"] + " Used because deterministic profile mapping returned no locally supported models."
        model_recommendations = pd.concat([model_recommendations, fallback], ignore_index=True)
        model_recommendations.to_csv(OUTPUT_DIR / "model_recommendations.csv", index=False)
        selected_model_names = fallback["model"].tolist()
    hyperparameter_spaces = llm_hyperparameter_search_spaces(
        selected_model_names,
        deterministic_profile,
        feature_selection_metrics,
        motivation,
        background_knowledge,
        use_llm=use_llm,
        output_dir=OUTPUT_DIR,
    )
    cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    for model_name in selected_model_names:
        model = spaces[model_name]
        params = hyperparameter_spaces.get(model_name, {})
        print(f"Fitting {model_name}...", flush=True)
        if params:
            search = RandomizedSearchCV(
                model,
                param_distributions=params,
                n_iter=min(12, math.prod([len(v) for v in params.values()])),
                scoring="neg_root_mean_squared_error",
                cv=3,
                random_state=RANDOM_STATE,
                n_jobs=1,
                refit=True,
                verbose=0,
            )
            search.fit(X_train_selected, y_train)
        else:
            fitted = clone(model)
            fitted.fit(X_train_selected, y_train)
            search = DefaultFitResult(fitted, X_train_selected, y_train)
        cv_scores = cross_validate(
            search.best_estimator_,
            X_train_selected,
            y_train,
            cv=cv,
            scoring={
                "mse": "neg_mean_squared_error",
                "rmse": "neg_root_mean_squared_error",
                "r2": "r2",
            },
            n_jobs=1,
        )
        cv_mse = -cv_scores["test_mse"]
        cv_rmse = -cv_scores["test_rmse"]
        cv_r2 = cv_scores["test_r2"]
        y_pred = search.predict(X_test_selected)
        test_rmse = root_mean_squared_error(y_test, y_pred)
        cv_rmse_mean = float(np.mean(cv_rmse))
        cv_rmse_std = float(np.std(cv_rmse))
        y_std = estimate_model_prediction_std(
            search.best_estimator_,
            X_test_selected,
            fallback_std=cv_rmse_mean if np.isfinite(cv_rmse_mean) and cv_rmse_mean > 0 else test_rmse,
        )
        uq_result = evaluate_uncertainty_for_candidate(
            y_test,
            y_pred,
            y_std,
            target_std=target_std,
            test_rmse=test_rmse,
            cv_rmse_mean=cv_rmse_mean,
            cv_rmse_std=cv_rmse_std,
        )
        uq_full_metrics_by_model[model_name] = {
            "full_metrics": uq_result.pop("uq_full_metrics"),
            "prediction_std_summary": {
                "mean": float(np.mean(y_std)),
                "median": float(np.median(y_std)),
                "min": float(np.min(y_std)),
                "max": float(np.max(y_std)),
            },
        }
        row = {
            "model": model_name,
            "test_mse": mean_squared_error(y_test, y_pred),
            "test_rmse": test_rmse,
            "test_r2": r2_score(y_test, y_pred),
            "cv_best_rmse": -search.best_score_,
            "cv_mse_mean": float(np.mean(cv_mse)),
            "cv_mse_std": float(np.std(cv_mse)),
            "cv_rmse_mean": cv_rmse_mean,
            "cv_rmse_std": cv_rmse_std,
            "cv_r2_mean": float(np.mean(cv_r2)),
            "cv_r2_std": float(np.std(cv_r2)),
            "best_params": search.best_params_,
        }
        row.update(uq_result)
        results.append(row)
        best_estimators[model_name] = search.best_estimator_
        best_searches[model_name] = search

    metrics = pd.DataFrame(results).sort_values(["uq_selection_score", "test_rmse"]).reset_index(drop=True)
    metrics.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)
    uq_cols = [
        "model",
        "uq_selection_score",
        "test_rmse",
        "test_r2",
        "uq_calibration_mae",
        "uq_miscalibration_area",
        "uq_sharpness",
        "uq_nll",
        "uq_crps",
        "uq_coverage_68",
        "uq_coverage_95",
        "cv_rmse_mean",
        "cv_rmse_std",
    ]
    metrics[uq_cols].to_csv(OUTPUT_DIR / "uncertainty_model_selection.csv", index=False)
    (OUTPUT_DIR / "uncertainty_model_selection_full.json").write_text(
        json.dumps(uq_full_metrics_by_model, indent=2, default=str),
        encoding="utf-8",
    )
    uncertainty_label, uncertainty_reasons = uncertainty_strength_label(deterministic_profile, metrics)
    (OUTPUT_DIR / "uncertainty_assessment.json").write_text(
        json.dumps(
            {
                "conclusion_strength": uncertainty_label,
                "reasons": uncertainty_reasons,
                "deterministic_dataset_profile": deterministic_profile,
                "best_model_cv_rmse_mean": float(metrics.loc[0, "cv_rmse_mean"]),
                "best_model_cv_rmse_std": float(metrics.loc[0, "cv_rmse_std"]),
                "best_model_cv_r2_mean": float(metrics.loc[0, "cv_r2_mean"]),
                "best_model_cv_r2_std": float(metrics.loc[0, "cv_r2_std"]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    with (OUTPUT_DIR / "best_params.json").open("w") as f:
        json.dump(
            {name: search.best_params_ for name, search in best_searches.items()},
            f,
            indent=2,
            default=str,
        )

    best_model_name = metrics.loc[0, "model"]
    best_model = best_estimators[best_model_name]
    best_y_pred = best_model.predict(X_test_selected)
    pd.DataFrame({"selected_encoded_feature": selected_names}).to_csv(
        OUTPUT_DIR / "selected_features.csv", index=False
    )

    rng = np.random.default_rng(RANDOM_STATE)
    baseline_rmse = root_mean_squared_error(y_test, best_model.predict(X_test_selected))
    permutation_rows = []
    for col in X.columns:
        repeats = []
        for _ in range(20):
            X_perm = X_test.copy()
            X_perm[col] = rng.permutation(X_perm[col].to_numpy())
            X_perm_encoded = preprocessor.transform(X_perm)
            X_perm_selected = selector.transform(to_dense_array(X_perm_encoded))
            rmse = root_mean_squared_error(y_test, best_model.predict(X_perm_selected))
            repeats.append(rmse - baseline_rmse)
        permutation_rows.append(
            {
                "feature": col,
                "importance_mean_rmse_increase": float(np.mean(repeats)),
                "importance_std": float(np.std(repeats)),
            }
        )
    perm_rows = pd.DataFrame(
        permutation_rows
    ).sort_values("importance_mean_rmse_increase", ascending=False)
    perm_rows.to_csv(OUTPUT_DIR / "permutation_importance.csv", index=False)

    model_importance_df = None
    final_model = best_model
    if hasattr(final_model, "feature_importances_"):
        model_importance_df = aggregate_importance(final_model.feature_importances_, selected_names)
        model_importance_df.to_csv(OUTPUT_DIR / "model_feature_importance.csv", index=False)

    shap_df = None
    if best_model_name == "XGBoost":
        xgb_model = best_model
        booster = xgb_model.get_booster()
        contrib = booster.predict(
            DMatrix(X_test_selected),
            pred_contribs=True,
        )
        mean_abs_shap = np.abs(contrib[:, :-1]).mean(axis=0)
        shap_df = aggregate_importance(mean_abs_shap, selected_names)
        shap_df.to_csv(OUTPUT_DIR / "xgboost_shap_contributions.csv", index=False)

    material_sources = []
    if shap_df is not None:
        tmp = shap_df.copy()
        tmp["source"] = f"{best_model_name}_shap_mean_abs"
        material_sources.append(tmp)
    if model_importance_df is not None:
        tmp = model_importance_df.copy()
        tmp["source"] = f"{best_model_name}_model_importance"
        material_sources.append(tmp)
    tmp = perm_rows.rename(columns={"importance_mean_rmse_increase": "importance"})[
        ["feature", "importance", "importance_std"]
    ].copy()
    tmp["source"] = f"{best_model_name}_permutation_rmse_increase"
    material_sources.append(tmp)

    material_importance = pd.concat(material_sources, ignore_index=True)
    material_importance = material_importance[material_importance["feature"].map(is_material_parameter)]
    material_importance.to_csv(OUTPUT_DIR / "material_parameter_importance.csv", index=False)

    save_model_mse_plot(metrics, OUTPUT_DIR)

    plt.figure(figsize=(9, 5))
    chart = metrics.sort_values("cv_rmse_mean", ascending=True) if "cv_rmse_mean" in metrics.columns else metrics.sort_values("test_rmse", ascending=True)
    xerr = chart["cv_rmse_std"] if "cv_rmse_std" in chart.columns else None
    values = chart["cv_rmse_mean"] if "cv_rmse_mean" in chart.columns else chart["test_rmse"]
    plt.barh(chart["model"], values, xerr=xerr, capsize=4 if xerr is not None else 0, color="#2f6f73")
    plt.xlabel("CV RMSE mean +/- std" if xerr is not None else "Test RMSE")
    plt.title("Model Comparison with Cross-Validation Uncertainty")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_rmse_comparison.png", dpi=180)
    plt.close()

    feature_count_df = save_feature_count_analysis(
        best_model=best_model,
        best_model_name=best_model_name,
        selected_names=selected_names,
        selector=selector,
        feature_selection_method=selected_feature_selection_method,
        X_train_encoded=X_train_encoded,
        X_test_encoded=X_test_encoded,
        y_train=y_train,
        y_test=y_test,
        output_dir=OUTPUT_DIR,
    )

    save_hyperparameter_heatmap(best_searches[best_model_name], best_model_name, OUTPUT_DIR)
    save_actual_vs_predicted(y_test, best_y_pred, best_model_name, OUTPUT_DIR)
    best_template = spaces[best_model_name]
    save_epoch_curves(best_model_name, best_template, best_searches[best_model_name].best_params_, X_train_selected, X_test_selected, y_train, y_test, OUTPUT_DIR)

    importance_for_plot = material_importance[
        material_importance["source"].astype(str).str.contains("permutation", case=False, na=False)
    ].copy()
    if importance_for_plot.empty:
        importance_for_plot = shap_df if shap_df is not None else model_importance_df
    if importance_for_plot is not None and not importance_for_plot.empty:
        plt.figure(figsize=(9, 6))
        top = importance_for_plot.head(15).sort_values("importance", ascending=True)
        xerr = top["importance_std"] if "importance_std" in top.columns else None
        plt.barh(
            top["feature"],
            top["importance"],
            xerr=xerr,
            capsize=4 if xerr is not None else 0,
            color="#8063a6",
        )
        plt.xlabel("Permutation RMSE increase mean +/- std" if xerr is not None else "Importance")
        plt.title(f"{best_model_name} Feature Importance with Uncertainty")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "xgboost_shap_top_features.png", dpi=180)
        plt.savefig(OUTPUT_DIR / "feature_importance_material_parameters.png", dpi=220)
        plt.close()

    profile, strategy = build_strategy_context(
        df=df,
        features=features,
        target=TARGET,
        output_dir=OUTPUT_DIR,
        motivation=motivation,
        background_knowledge=background_knowledge,
        use_llm=use_llm,
    )
    graph_explanations = llm_graph_explanations(
        profile,
        strategy,
        metrics,
        material_importance,
        feature_count_df,
        target_relationship_features,
        best_model_name,
        motivation,
        background_knowledge,
        use_llm=use_llm,
    )

    summary = {
        "data_path": str(DATA_PATH),
        "rows_after_target_drop": int(len(df)),
        "target_values_divided_by_1000_as_likely_mpa": int(likely_mpa_mask.sum()),
        "target": TARGET,
        "feature_count_before_encoding": len(features),
        "selected_feature_selection_method": selected_feature_selection_method,
        "best_model": best_model_name,
        "best_model_test_rmse": float(metrics.loc[0, "test_rmse"]),
        "best_model_test_mse": float(metrics.loc[0, "test_mse"]),
        "best_model_test_r2": float(metrics.loc[0, "test_r2"]),
        "best_model_uq_selection_score": float(metrics.loc[0, "uq_selection_score"]),
        "best_model_uq_calibration_mae": float(metrics.loc[0, "uq_calibration_mae"]),
        "best_model_uq_miscalibration_area": float(metrics.loc[0, "uq_miscalibration_area"]),
        "best_model_uq_sharpness": float(metrics.loc[0, "uq_sharpness"]),
        "best_model_uq_coverage_95": float(metrics.loc[0, "uq_coverage_95"]),
        "best_model_cv_rmse_mean": float(metrics.loc[0, "cv_rmse_mean"]),
        "best_model_cv_rmse_std": float(metrics.loc[0, "cv_rmse_std"]),
        "best_model_cv_r2_mean": float(metrics.loc[0, "cv_r2_mean"]),
        "best_model_cv_r2_std": float(metrics.loc[0, "cv_r2_std"]),
        "conclusion_strength": uncertainty_label,
        "uncertainty_reasons": uncertainty_reasons,
        "outputs": sorted(str(p) for p in OUTPUT_DIR.iterdir()),
    }
    with (OUTPUT_DIR / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    write_report(
        output_dir=OUTPUT_DIR,
        metrics=metrics,
        best_model_name=best_model_name,
        best_params=best_searches[best_model_name].best_params_,
        shap_df=shap_df,
        material_importance=material_importance,
        feature_count_df=feature_count_df,
        unit_corrections=unit_corrections,
        model_recommendations=model_recommendations,
    )
    write_strategy_html_report(
        output_dir=OUTPUT_DIR,
        metrics=metrics,
        best_model_name=best_model_name,
        best_params=best_searches[best_model_name].best_params_,
        material_importance=material_importance,
        feature_count_df=feature_count_df,
        target_relationship_features=target_relationship_features,
        unit_corrections=unit_corrections,
        model_recommendations=model_recommendations,
        feature_selection_metrics=feature_selection_metrics,
        profile=profile,
        strategy=strategy,
        graph_explanations=graph_explanations,
        feature_selection_recommendations=feature_selection_recommendations,
        feature_selection_comparison=feature_selection_comparison,
        selected_feature_selection_method=selected_feature_selection_method,
        hyperparameter_spaces=hyperparameter_spaces,
        motivation=motivation,
        background_knowledge=background_knowledge,
        uncertainty_label=uncertainty_label,
        uncertainty_reasons=uncertainty_reasons,
    )

    print(
        metrics[
            [
                "model",
                "test_rmse",
                "test_r2",
                "uq_selection_score",
                "uq_miscalibration_area",
                "uq_sharpness",
                "uq_coverage_95",
                "cv_rmse_mean",
                "cv_rmse_std",
                "cv_r2_mean",
                "cv_r2_std",
            ]
        ].to_string(index=False)
    )
    print(f"\nBest model: {best_model_name}")
    print(f"Outputs written to: {OUTPUT_DIR}")
    if shap_df is not None:
        print(f"\nTop {best_model_name} SHAP features:")
        print(shap_df.head(12).to_string(index=False))
    print("\nTop material parameters:")
    print(material_importance.head(20).to_string(index=False))
    return OUTPUT_DIR


def run_spall_strength_analysis(
    data_path: str | Path = DATA_PATH,
    output_root: str | Path = OUTPUT_ROOT,
) -> Path:
    return run_model_strategy_analysis(data_path=data_path, target="Spall (Gpa)", output_root=output_root)


def main() -> None:
    run_model_strategy_analysis()


if __name__ == "__main__":
    main()
