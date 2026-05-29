"""
Model training and prediction.

Prepares train/test split, trains the model (with optional GridSearchCV or
RandomizedSearchCV), and returns predictions for evaluation.
Supports both regression and classification tasks.
"""

from __future__ import annotations

import re
from typing import Any, List, Literal

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV

DEFAULT_REGRESSION_GRIDS: dict[str, dict[str, List[Any]]] = {
    "RandomForestRegressor": {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5],
    },
    "GradientBoostingRegressor": {
        "n_estimators": [50, 100],
        "max_depth": [3, 5],
        "learning_rate": [0.01, 0.1],
    },
    "Ridge": {"alpha": [0.01, 0.1, 1.0, 10.0]},
    "Lasso": {"alpha": [0.001, 0.01, 0.1, 1.0]},
    "SVR": {"C": [0.1, 1.0, 10.0], "kernel": ["rbf", "linear"]},
}

def _tree_safe_feature_columns(features: List[str]) -> tuple[List[str], dict[str, str]]:
    """
    LightGBM (and some tree backends) reject column names with JSON-special chars
    like []{}'",: — common in materials datasets ("Yield Strength (MPa)", "O (wt.,%)").
    Returns (safe_names_in_order, mapping original_name -> safe_name).
    """
    safe_names: List[str] = []
    orig_to_safe: dict[str, str] = {}
    used: set[str] = set()
    for i, orig in enumerate(features):
        s = str(orig)
        for ch in '[]{}"\',:':
            s = s.replace(ch, "_")
        s = re.sub(r"\s+", "_", s)
        s = re.sub(r"[^A-Za-z0-9_]+", "_", s)
        s = s.strip("_") or f"f_{i}"
        if s[0].isdigit():
            s = "f_" + s
        base = s
        n = 0
        while s in used:
            n += 1
            s = f"{base}__{n}"
        used.add(s)
        safe_names.append(s)
        orig_to_safe[orig] = s
    return safe_names, orig_to_safe


DEFAULT_CLASSIFICATION_GRIDS: dict[str, dict[str, List[Any]]] = {
    "RandomForestClassifier": {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5],
    },
    "LogisticRegression": {"C": [0.01, 0.1, 1.0, 10.0], "max_iter": [1000]},
    "SVC": {"C": [0.1, 1.0, 10.0], "kernel": ["rbf", "linear"]},
}


class ModelTrainer:
    """Handles data split, model training (with optional hyperparameter search), and prediction."""

    def __init__(
        self,
        model_class: type,
        hyperparams: dict[str, List[Any]] | None = None,
        *,
        task_type: Literal["regression", "classification"] = "regression",
        use_hyperparameter_tuning: bool = True,
        use_randomized_search: bool = False,
        n_iter: int = 20,
    ):
        """
        Args:
            model_class: The model class (e.g. RandomForestRegressor), uninstantiated.
            hyperparams: Optional param grid for GridSearchCV/RandomizedSearchCV;
                if None and use_hyperparameter_tuning, use default grid for known models.
            task_type: "regression" or "classification" (affects scoring metric).
            use_hyperparameter_tuning: If False, train with default args (no search).
            use_randomized_search: If True, use RandomizedSearchCV instead of GridSearchCV.
            n_iter: Number of parameter settings sampled for RandomizedSearchCV.
        """
        self.model_class = model_class
        self.hyperparams = hyperparams
        self.task_type = task_type
        self.use_hyperparameter_tuning = use_hyperparameter_tuning
        self.use_randomized_search = use_randomized_search
        self.n_iter = n_iter
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self._feature_rename_map: dict[str, str] = {}

    def _get_scoring(self) -> str:
        """Return the scoring metric for cross-validation."""
        if self.task_type == "classification":
            return "accuracy"
        return "r2"

    def _get_param_grid(self) -> dict[str, List[Any]]:
        """Resolve hyperparameter grid (user-provided, default, or empty)."""
        if not self.use_hyperparameter_tuning:
            return {}
        if self.hyperparams:
            return self.hyperparams
        model_name = self.model_class.__name__
        grids = (
            DEFAULT_CLASSIFICATION_GRIDS
            if self.task_type == "classification"
            else DEFAULT_REGRESSION_GRIDS
        )
        return grids.get(model_name, {})

    def prepare_data(
        self,
        df: pd.DataFrame,
        features: List[str],
        target: str,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> None:
        """Split into train and test sets."""
        n_samples = len(df)
        if n_samples < 2:
            raise ValueError(
                "Dataset has fewer than 2 usable rows after preprocessing. "
                f"n_samples={n_samples}. "
                "Please choose a target/features with fewer missing values, disable aggressive row dropping, "
                "or provide a larger/cleaner dataset."
            )

        X = df[features].copy()
        safe_cols, self._feature_rename_map = _tree_safe_feature_columns(features)
        X.columns = safe_cols
        y = df[target]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    def train(self) -> None:
        """Train the model. If hyperparams available, use GridSearchCV or RandomizedSearchCV."""
        param_grid = self._get_param_grid()
        scoring = self._get_scoring()
        cv_folds = min(5, len(self.y_train))

        if param_grid and cv_folds >= 2:
            base = self.model_class()
            if self.use_randomized_search:
                search = RandomizedSearchCV(
                    base,
                    param_grid,
                    n_iter=min(self.n_iter, self._count_combinations(param_grid)),
                    cv=cv_folds,
                    scoring=scoring,
                    n_jobs=1,
                    random_state=42,
                )
            else:
                search = GridSearchCV(
                    base,
                    param_grid,
                    cv=cv_folds,
                    scoring=scoring,
                    n_jobs=1,
                )
            search.fit(self.X_train, self.y_train)
            self.model = search.best_estimator_
        else:
            self.model = self.model_class()
            self.model.fit(self.X_train, self.y_train)

    def _count_combinations(self, grid: dict[str, List[Any]]) -> int:
        """Count total combinations in a param grid."""
        n = 1
        for v in grid.values():
            n *= len(v)
        return max(n, 1)

    def predict(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (y_true, y_pred) on the test set."""
        y_pred = self.model.predict(self.X_test)
        return self.y_test.values, y_pred

    def predict_with_uncertainty(self) -> dict[str, np.ndarray]:
        """
        Return predictions with uncertainty quantification.

        Supports:
        - ProbabilisticNN: Native distribution prediction
        - Ensemble models (RandomForest, GradientBoosting): Variance across estimators
        - Classification: Prediction entropy
        - GaussianProcess: Native uncertainty

        Returns:
            Dictionary with:
                - 'y_true': True values
                - 'y_pred': Point predictions (mean)
                - 'y_std': Standard deviations (uncertainties)
                - Additional model-specific fields
        """
        model_name = self.model_class.__name__

        result = {
            'y_true': self.y_test.values,
        }

        # ProbabilisticNN: native distribution prediction
        if 'ProbabilisticNN' in model_name or hasattr(self.model, 'predict_distribution'):
            try:
                dist = self.model.predict_distribution(self.X_test.values)
                result.update({
                    'y_pred': dist['mu'],
                    'y_std': dist['std'],
                    'lower_95': dist.get('lower_95'),
                    'upper_95': dist.get('upper_95'),
                })
                return result
            except Exception as e:
                print(f"[ModelTrainer] ProbabilisticNN distribution prediction failed: {e}")
                # Fall through to default prediction

        # GaussianProcess: native uncertainty via return_std=True
        if 'GaussianProcess' in model_name:
            try:
                mu, std = self.model.predict(self.X_test, return_std=True)
                result.update({
                    'y_pred': mu,
                    'y_std': std,
                })
                return result
            except Exception:
                # Fall through if return_std not supported
                pass

        # Ensemble methods: use variance across estimators
        if hasattr(self.model, 'estimators_'):
            # RandomForest, GradientBoosting, etc.
            predictions = np.array([
                estimator.predict(self.X_test)
                for estimator in self.model.estimators_
            ])
            # predictions.shape = (n_estimators, n_samples)

            y_pred = np.mean(predictions, axis=0)
            y_std = np.std(predictions, axis=0)

            result.update({
                'y_pred': y_pred,
                'y_std': y_std,
            })

            return result

        # Classification: use prediction entropy
        if self.task_type == "classification" and hasattr(self.model, 'predict_proba'):
            y_proba = self.model.predict_proba(self.X_test)
            y_pred = self.model.predict(self.X_test)

            # Compute entropy as uncertainty
            proba_clipped = np.clip(y_proba, 1e-10, 1.0)
            entropy = -np.sum(proba_clipped * np.log(proba_clipped), axis=1)

            result.update({
                'y_pred': y_pred,
                'y_std': entropy,  # Use entropy as uncertainty measure
                'y_proba': y_proba,
            })

            return result

        y_pred = self.model.predict(self.X_test)
        result['y_pred'] = y_pred
        result['y_std'] = np.zeros_like(y_pred)

        return result


__all__ = ["ModelTrainer", "DEFAULT_REGRESSION_GRIDS", "DEFAULT_CLASSIFICATION_GRIDS"]
