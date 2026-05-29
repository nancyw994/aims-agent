import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class _CustomRegressorWrapper(BaseEstimator, RegressorMixin):
    """A simple wrapper that mimics a scikit-learn regressor using LinearRegression."""
    def __init__(self):
        self._model = LinearRegression()

    def fit(self, X, y):
        # Validate input
        X, y = check_X_y(X, y, accept_sparse=False, dtype=np.float64, y_numeric=True)
        self._model.fit(X, y)
        return self

    def predict(self, X):
        check_is_fitted(self._model)
        X = check_array(X, accept_sparse=False, dtype=np.float64)
        return self._model.predict(X)

    def score(self, X, y):
        check_is_fitted(self._model)
        X = check_array(X, accept_sparse=False, dtype=np.float64)
        y = check_array(y, ensure_2d=False, dtype=np.float64)
        return self._model.score(X, y)


def run_component(df, features, target, task_type="regression"):
    """
    Runs a custom estimator wrapper for regression tasks.
    Returns a dictionary with the original features and a short note.
    """
    # Guard against empty or invalid dataframes
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return {"features": features, "note": "Input dataframe is empty or invalid."}

    # Ensure requested columns exist
    required_cols = list(features) + [target]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        return {"features": features, "note": f"Missing columns in dataframe: {missing}"}

    # Extract feature matrix and target vector
    X = df[features].values
    y = df[target].values.ravel()

    # Handle too‑small datasets gracefully
    if X.shape[0] < 2:
        return {"features": features, "note": "Not enough samples to fit a model (need at least 2)."}

    try:
        # Attempt to fit using our custom wrapper (as if the selected model could not be loaded directly)
        estimator = _CustomRegressorWrapper()
        estimator.fit(X, y)

        # Optional: compute a quick training score for the note
        train_score = estimator.score(X, y)
        note = (
            f"Fitted custom regressor wrapper on {X.shape[0]} samples. "
            f"Training R^2 = {train_score:.4f}"
        )
    except Exception as exc:
        note = f"Error during model fitting: {exc}"

    return {"features": features, "note": note}