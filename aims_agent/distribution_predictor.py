"""
Universal distribution predictor for uncertainty quantification.

This module provides a unified interface to predict distributions
from different types of models (GP, ensembles, probabilistic NNs, etc.).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def predict_distribution(
    model: Any,
    X: np.ndarray,
    model_type: str | None = None
) -> dict[str, np.ndarray] | None:
    """
    Predict distribution (mean, std, confidence intervals) from any model.
    
    Args:
        model: Trained model instance
        X: Input features, shape (n_samples, n_features)
        model_type: Model type hint ("gp", "ensemble", "probabilistic_nn", etc.)
                   If None, will attempt to auto-detect
    
    Returns:
        Dictionary with keys:
            - 'mu': predicted means, shape (n_samples,)
            - 'std': predicted standard deviations, shape (n_samples,)
            - 'var': predicted variances, shape (n_samples,)
            - 'lower_95': lower bound of 95% CI, shape (n_samples,)
            - 'upper_95': upper bound of 95% CI, shape (n_samples,)
        
        Returns None if model doesn't support distribution prediction.
    """
    
    # Auto-detect model type if not provided
    if model_type is None:
        model_type = _detect_model_type(model)
    
    # Route to appropriate prediction function
    if model_type == "probabilistic_nn":
        return _predict_probabilistic_nn(model, X)
    
    elif model_type == "gp":
        return _predict_gaussian_process(model, X)
    
    elif model_type == "ensemble":
        return _predict_ensemble(model, X)
    
    elif model_type == "classification_proba":
        return _predict_classification_entropy(model, X)
    
    else:
        # Model doesn't support native distribution prediction
        return None


def _detect_model_type(model: Any) -> str:
    """Auto-detect model type from class name."""
    
    class_name = model.__class__.__name__
    
    # Probabilistic NN
    if "ProbabilisticNN" in class_name or hasattr(model, "predict_distribution"):
        return "probabilistic_nn"
    
    # Gaussian Process
    if "GaussianProcess" in class_name:
        return "gp"
    
    # Ensemble methods
    if hasattr(model, "estimators_"):
        return "ensemble"
    
    # Classification with probabilities
    if hasattr(model, "predict_proba") and not hasattr(model, "predict"):
        return "classification_proba"
    
    return "unknown"


def _predict_probabilistic_nn(model: Any, X: np.ndarray) -> dict[str, np.ndarray]:
    """Predict from Probabilistic NN (native distribution output)."""
    
    if hasattr(model, "predict_distribution"):
        # ProbabilisticNNWrapper or ProbabilisticTrainer
        return model.predict_distribution(X)
    
    else:
        raise ValueError("Model does not have predict_distribution method")


def _predict_gaussian_process(model: Any, X: np.ndarray) -> dict[str, np.ndarray]:
    """
    Predict from Gaussian Process (native distribution).
    
    GP naturally outputs mean and std via the posterior distribution.
    """
    
    # Check if model has predict method with return_std
    if not hasattr(model, "predict"):
        raise ValueError("GP model does not have predict method")
    
    try:
        # sklearn GP: predict(X, return_std=True)
        mu, std = model.predict(X, return_std=True)
        
        # Flatten if needed
        mu = np.asarray(mu).flatten()
        std = np.asarray(std).flatten()
        
        # Compute derived quantities
        var = std ** 2
        lower_95 = mu - 1.96 * std
        upper_95 = mu + 1.96 * std
        
        return {
            'mu': mu,
            'std': std,
            'var': var,
            'lower_95': lower_95,
            'upper_95': upper_95,
        }
    
    except TypeError:
        # Model doesn't support return_std
        raise ValueError("GP model does not support return_std parameter")


def _predict_ensemble(model: Any, X: np.ndarray) -> dict[str, np.ndarray]:
    """
    Predict from ensemble methods (RF, GB, ExtraTrees).
    
    Uncertainty is estimated from variance across ensemble members.
    """
    
    if not hasattr(model, "estimators_"):
        raise ValueError("Model does not have estimators_ attribute (not an ensemble)")
    
    # Collect predictions from all estimators
    predictions = []
    for estimator in model.estimators_:
        pred = estimator.predict(X)
        predictions.append(pred)
    
    predictions = np.array(predictions)  # shape: (n_estimators, n_samples)
    
    # Compute statistics
    mu = np.mean(predictions, axis=0)
    std = np.std(predictions, axis=0)
    var = np.var(predictions, axis=0)
    
    # 95% confidence intervals
    lower_95 = mu - 1.96 * std
    upper_95 = mu + 1.96 * std
    
    return {
        'mu': mu,
        'std': std,
        'var': var,
        'lower_95': lower_95,
        'upper_95': upper_95,
    }


def _predict_classification_entropy(model: Any, X: np.ndarray) -> dict[str, np.ndarray]:
    """
    Predict from classification models using entropy as uncertainty.
    
    For classification, we use prediction entropy as uncertainty measure.
    """
    
    if not hasattr(model, "predict_proba"):
        raise ValueError("Classification model does not support predict_proba")
    
    # Get class probabilities
    y_proba = model.predict_proba(X)  # shape: (n_samples, n_classes)
    
    # Predicted class
    y_pred = model.predict(X)
    
    # Compute entropy as uncertainty measure
    # H = -sum(p * log(p))
    epsilon = 1e-10
    entropy = -np.sum(
        y_proba * np.log(np.clip(y_proba, epsilon, 1.0)),
        axis=1
    )
    
    # For classification, "std" is entropy
    # We don't have true confidence intervals, so use heuristic
    return {
        'mu': y_pred.astype(float),
        'std': entropy,
        'var': entropy ** 2,
        'lower_95': y_pred.astype(float) - 1.96 * entropy,  # Heuristic
        'upper_95': y_pred.astype(float) + 1.96 * entropy,  # Heuristic
        'y_proba': y_proba,  # Include probabilities
    }


def can_predict_distribution(model: Any) -> bool:
    """
    Check if a model can predict distributions.
    
    Args:
        model: Model instance
    
    Returns:
        True if model supports distribution prediction
    """
    model_type = _detect_model_type(model)
    return model_type in ["probabilistic_nn", "gp", "ensemble", "classification_proba"]


def get_uq_capability(model: Any) -> dict[str, Any]:
    """
    Get UQ capability information for a model.
    
    Args:
        model: Model instance
    
    Returns:
        Dictionary with:
            - 'can_predict_distribution': bool
            - 'model_type': str
            - 'uq_quality_estimate': float (0.0-1.0)
            - 'heteroscedastic': bool
    """
    
    model_type = _detect_model_type(model)
    can_dist = can_predict_distribution(model)
    
    # Estimate UQ quality based on model type
    uq_quality_map = {
        "probabilistic_nn": 0.9,
        "gp": 1.0,
        "ensemble": 0.7,
        "classification_proba": 0.5,
        "unknown": 0.0,
    }
    
    heteroscedastic_map = {
        "probabilistic_nn": True,
        "gp": True,
        "ensemble": False,
        "classification_proba": False,
        "unknown": False,
    }
    
    return {
        "can_predict_distribution": can_dist,
        "model_type": model_type,
        "uq_capability": model_type,  # For backwards compatibility
        "uq_quality_estimate": uq_quality_map.get(model_type, 0.0),
        "heteroscedastic": heteroscedastic_map.get(model_type, False),
    }


__all__ = [
    "predict_distribution",
    "can_predict_distribution",
    "get_uq_capability",
]
