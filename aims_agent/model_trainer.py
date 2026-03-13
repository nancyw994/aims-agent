"""
Model training and prediction.

Prepares train/test split, trains the model (with optional GridSearchCV or
RandomizedSearchCV), and returns predictions for evaluation.
Supports both regression and classification tasks.
"""

from __future__ import annotations

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
        X = df[features]
        y = df[target]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    def train(self) -> None:
        """Train the model. If hyperparams available, use GridSearchCV or RandomizedSearchCV."""
        param_grid = self._get_param_grid()
        scoring = self._get_scoring()

        if param_grid:
            base = self.model_class()
            if self.use_randomized_search:
                search = RandomizedSearchCV(
                    base,
                    param_grid,
                    n_iter=min(self.n_iter, self._count_combinations(param_grid)),
                    cv=5,
                    scoring=scoring,
                    n_jobs=-1,
                    random_state=42,
                )
            else:
                search = GridSearchCV(
                    base,
                    param_grid,
                    cv=5,
                    scoring=scoring,
                    n_jobs=-1,
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


__all__ = ["ModelTrainer", "DEFAULT_REGRESSION_GRIDS", "DEFAULT_CLASSIFICATION_GRIDS"]
