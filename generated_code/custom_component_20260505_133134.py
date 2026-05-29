import pandas as pd
import numpy as np

def run_component(df, features, target, task_type="regression"):
    """
    Attempts to use a standard scikit-learn estimator; if it cannot be loaded,
    falls back to a simple custom estimator. Returns a summary dictionary.
    """
    # Basic validation
    if not isinstance(df, pd.DataFrame):
        return {"features": features, "note": "Input is not a pandas DataFrame"}
    missing = [f for f in features if f not in df.columns] + ([target] if target not in df.columns else [])
    if missing:
        return {"features": features, "note": f"Missing columns: {missing}"}
    if len(df) < 2:
        return {"features": features, "note": "Insufficient data to train model (need at least 2 samples)"}

    # Prepare data
    X = df[features].values
    y = df[target].values.astype(float)

    # Try to load a standard estimator
    try:
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X, y)
        note = "Trained using scikit-learn LinearRegression"
    except Exception:
        # Fallback to a simple custom estimator
        class SimpleRegressor:
            def __init__(self):
                self.mean_ = None
            def fit(self, X, y):
                self.mean_ = np.mean(y) if len(y) > 0 else 0.0
                return self
            def predict(self, X):
                return np.full(shape=(X.shape[0],), fill_value=self.mean_, dtype=float)
        model = SimpleRegressor()
        model.fit(X, y)
        note = "Fell back to custom SimpleRegressor (mean predictor)"

    return {"features": features, "note": note}