"""
Uncertainty quantification and active learning using uncertainty-toolbox.

This module integrates the uncertainty-toolbox library for professional-grade
uncertainty quantification and calibration analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

try:
    import uncertainty_toolbox as uct
    HAS_UNCERTAINTY_TOOLBOX = True
except ImportError:
    HAS_UNCERTAINTY_TOOLBOX = False
    uct = None

if TYPE_CHECKING:
    from aims_agent.agent import Agent


def check_uncertainty_toolbox():
    """Check if uncertainty-toolbox is available."""
    if not HAS_UNCERTAINTY_TOOLBOX:
        raise ImportError(
            "uncertainty-toolbox is required for uncertainty quantification.\n"
            "Install it with: pip install uncertainty-toolbox"
        )


def compute_uncertainty_metrics(
    y_pred: np.ndarray,
    y_std: np.ndarray,
    y_true: np.ndarray,
) -> dict[str, Any]:
    """
    Compute comprehensive uncertainty quantification metrics using uncertainty-toolbox.
    
    Args:
        y_pred: Point predictions
        y_std: Prediction uncertainties (standard deviations)
        y_true: Ground truth values
    
    Returns:
        Dictionary with all uncertainty metrics including:
        - accuracy: MAE, RMSE, MDAE, MARPD, R2, Correlation
        - avg_calibration: RMSCE, miscalibration area
        - adv_group_calibration: adversarial group calibration errors
        - sharpness: average prediction uncertainty
        - scoring_rule: NLL, CRPS, check score, interval score
    """
    check_uncertainty_toolbox()
    
    # Ensure inputs are numpy arrays
    y_pred = np.asarray(y_pred).flatten()
    y_std = np.asarray(y_std).flatten()
    y_true = np.asarray(y_true).flatten()
    
    # Get all metrics from uncertainty-toolbox (verbose=False to avoid duplicate output)
    all_metrics = uct.metrics.get_all_metrics(y_pred, y_std, y_true, verbose=False)
    
    return all_metrics


def suggest_active_learning_samples(
    y_std: np.ndarray,
    X: np.ndarray | None = None,
    n_samples: int = 10,
    strategy: str = "uncertainty",
) -> dict[str, Any]:
    """
    Suggest samples for active learning based on uncertainty or diversity.
    
    Args:
        y_std: Prediction uncertainties
        X: Feature matrix (optional, for diversity-based selection)
        n_samples: Number of samples to suggest
        strategy: "uncertainty" (exploit) or "diversity" (explore)
    
    Returns:
        Dictionary with suggested sample indices and scores
    """
    result = {
        "strategy": strategy,
        "n_suggested": n_samples,
    }
    
    if strategy == "uncertainty":
        # Select samples with highest uncertainty
        top_indices = np.argsort(y_std)[-n_samples:][::-1]
        result["suggested_indices"] = top_indices.tolist()
        result["suggested_uncertainties"] = y_std[top_indices].tolist()
        result["reason"] = "High uncertainty samples for exploitation (reducing model uncertainty)"
    
    elif strategy == "diversity" and X is not None:
        # Select diverse samples using k-means clustering
        try:
            from sklearn.cluster import KMeans
            
            # Weight by uncertainty
            sample_weights = y_std / np.sum(y_std)
            
            # Cluster to find diverse samples
            n_clusters = min(n_samples, len(X))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans.fit(X, sample_weight=sample_weights)
            
            # Select one high-uncertainty sample from each cluster
            suggested = []
            for i in range(n_clusters):
                cluster_mask = kmeans.labels_ == i
                cluster_indices = np.where(cluster_mask)[0]
                if len(cluster_indices) > 0:
                    cluster_uncertainties = y_std[cluster_indices]
                    best_in_cluster = cluster_indices[np.argmax(cluster_uncertainties)]
                    suggested.append(int(best_in_cluster))
            
            result["suggested_indices"] = suggested[:n_samples]
            result["suggested_uncertainties"] = [float(y_std[i]) for i in result["suggested_indices"]]
            result["reason"] = "Diverse high-uncertainty samples for exploration (improving data coverage)"
        except ImportError:
            # Fallback to uncertainty if sklearn not available
            return suggest_active_learning_samples(y_std, X, n_samples, strategy="uncertainty")
    
    else:
        # Default to uncertainty sampling
        return suggest_active_learning_samples(y_std, X, n_samples, strategy="uncertainty")
    
    return result


def plot_uncertainty_analysis(
    y_pred: np.ndarray,
    y_std: np.ndarray,
    y_true: np.ndarray,
    save_dir: str | Path = "results",
    prefix: str = "uncertainty",
) -> dict[str, str]:
    """
    Generate comprehensive uncertainty visualization using uncertainty-toolbox.
    
    Creates multiple plots:
    - Calibration plot
    - Sharpness plot  
    - Residuals vs uncertainties
    - Ordered prediction intervals
    
    Args:
        y_pred: Point predictions
        y_std: Prediction uncertainties
        y_true: Ground truth values
        save_dir: Directory to save plots
        prefix: Prefix for plot filenames
    
    Returns:
        Dictionary mapping plot names to file paths
    """
    check_uncertainty_toolbox()
    
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    plot_paths = {}
    
    # 1. Calibration plot
    fig, ax = plt.subplots(figsize=(8, 6))
    uct.viz.plot_calibration(y_pred, y_std, y_true, ax=ax)
    path = save_dir / f"{prefix}_calibration.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    plot_paths["calibration"] = str(path)
    
    # 2. Sharpness plot  
    fig, ax = plt.subplots(figsize=(8, 6))
    uct.viz.plot_sharpness(y_std, ax=ax)
    path = save_dir / f"{prefix}_sharpness.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    plot_paths["sharpness"] = str(path)
    
    # 3. Residuals vs uncertainties
    fig, ax = plt.subplots(figsize=(8, 6))
    uct.viz.plot_residuals_vs_stds(y_pred, y_std, y_true, ax=ax)
    path = save_dir / f"{prefix}_residuals_vs_stds.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    plot_paths["residuals_vs_stds"] = str(path)
    
    # 4. Ordered prediction intervals
    fig, ax = plt.subplots(figsize=(10, 6))
    uct.viz.plot_intervals_ordered(y_pred, y_std, y_true, ax=ax)
    path = save_dir / f"{prefix}_intervals_ordered.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    plot_paths["intervals_ordered"] = str(path)
    
    return plot_paths


def interpret_uncertainty_with_llm(
    agent: "Agent",
    metrics: dict[str, Any],
    active_learning_suggestions: dict,
    task_type: str = "regression",
) -> str:
    """
    Use LLM to interpret uncertainty quantification results and provide active learning strategy.
    
    Args:
        agent: AIMS Agent instance
        metrics: Uncertainty metrics from uncertainty-toolbox (nested dict)
        active_learning_suggestions: Active learning sample suggestions
        task_type: "regression" or "classification"
    
    Returns:
        LLM interpretation text
    """
    # Extract nested metrics
    accuracy = metrics.get('accuracy', {})
    avg_cal = metrics.get('avg_calibration', {})
    sharpness = metrics.get('sharpness', {})
    scoring = metrics.get('scoring_rule', {})
    
    prompt = f"""You are a materials science ML expert specializing in uncertainty quantification and active learning.

Task type: {task_type}

Uncertainty Quantification Metrics (from uncertainty-toolbox):

Accuracy Metrics:
- MAE: {accuracy.get('mae', 'N/A')}
- RMSE: {accuracy.get('rmse', 'N/A')}
- R²: {accuracy.get('rsq', 'N/A')}

Calibration Metrics:
- Root Mean Squared Calibration Error: {avg_cal.get('rms_cal', 'N/A')}
- Mean Absolute Calibration Error: {avg_cal.get('ma_cal', 'N/A')}
- Miscalibration Area: {avg_cal.get('miscal_area', 'N/A')}

Sharpness:
- Average Prediction Uncertainty: {sharpness.get('sharpness', 'N/A')} (lower is sharper/more confident)

Proper Scoring Rules:
- Negative Log-Likelihood: {scoring.get('nll', 'N/A')} (lower is better)
- CRPS (Continuous Ranked Probability Score): {scoring.get('crps', 'N/A')} (lower is better)
- Check Score: {scoring.get('check', 'N/A')}
- Interval Score: {scoring.get('interval', 'N/A')}

Active Learning Suggestions:
Strategy: {active_learning_suggestions.get('strategy', 'N/A')}
Suggested samples: {active_learning_suggestions.get('suggested_indices', [])}
Top uncertainties: {active_learning_suggestions.get('suggested_uncertainties', [])}
Reason: {active_learning_suggestions.get('reason', 'N/A')}

Please provide:

1. **Uncertainty Assessment** (2-3 sentences):
   - Is the model well-calibrated? (Low calibration error means high reliability)
   - What does the sharpness tell us about model confidence?
   - Are the scoring rules indicating good uncertainty estimates?

2. **Active Learning Strategy** (2-3 sentences):
   - Should we prioritize high-uncertainty samples (exploitation) or diverse samples (exploration)?
   - How many additional experiments are recommended?
   - What specific material properties or composition ranges need more data?

3. **Risk Assessment** (1-2 sentences):
   - Which predictions are safe to use vs require experimental validation?
   - What uncertainty threshold should we use to filter unreliable predictions?

4. **Next Steps** (bullet points):
   - Concrete actions for improving model reliability
   - Data collection priorities
   - Model improvement suggestions

Keep the response practical and actionable for materials researchers."""
    
    return agent.call_llm(prompt)


def recalibrate_predictions(
    y_pred: np.ndarray,
    y_std: np.ndarray,
    y_true: np.ndarray,
    method: str = "mean_recal",
) -> np.ndarray:
    """
    Recalibrate prediction uncertainties to improve calibration.
    
    Uses a simple scaling approach: multiply all uncertainties by a constant
    factor to minimize calibration error on the validation set.
    
    Args:
        y_pred: Point predictions
        y_std: Original prediction uncertainties  
        y_true: Ground truth values (from validation set)
        method: Recalibration method (currently only "mean_recal" supported)
    
    Returns:
        Recalibrated uncertainties
    """
    check_uncertainty_toolbox()
    
    # Simple recalibration: find optimal scaling factor
    # that minimizes calibration error
    from scipy.optimize import minimize_scalar
    
    def calibration_loss(scale):
        """Compute calibration error for a given scaling factor."""
        y_std_scaled = y_std * scale
        try:
            metrics = compute_uncertainty_metrics(y_pred, y_std_scaled, y_true)
            cal_error = metrics.get('avg_calibration', {}).get('ma_cal', 1.0)
            return cal_error
        except:
            return 1.0  # High penalty if computation fails
    
    # Find optimal scaling factor between 0.1 and 10
    result = minimize_scalar(calibration_loss, bounds=(0.1, 10.0), method='bounded')
    optimal_scale = result.x
    
    # Apply optimal scaling
    y_std_recal = y_std * optimal_scale
    
    return y_std_recal


__all__ = [
    "compute_uncertainty_metrics",
    "suggest_active_learning_samples",
    "plot_uncertainty_analysis",
    "interpret_uncertainty_with_llm",
    "recalibrate_predictions",
    "check_uncertainty_toolbox",
]
