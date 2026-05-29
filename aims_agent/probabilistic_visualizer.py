"""
Visualization tools for probabilistic model predictions and uncertainties.

This module provides plotting functions for uncertainty quantification
using uncertainty-toolbox and custom visualizations.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import uncertainty_toolbox as uct
    HAS_UCT = True
except ImportError:
    HAS_UCT = False


def check_uncertainty_toolbox():
    """Check if uncertainty-toolbox is available."""
    if not HAS_UCT:
        raise ImportError(
            "uncertainty-toolbox required for visualization.\n"
            "Install: pip install uncertainty-toolbox"
        )


def plot_training_history(
    train_losses: list[float],
    val_losses: list[float] | None = None,
    save_path: str | Path | None = None
):
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: Training loss per epoch
        val_losses: Validation loss per epoch (optional)
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    
    if val_losses:
        ax.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        
        # Mark best epoch
        best_epoch = np.argmin(val_losses) + 1
        best_val_loss = min(val_losses)
        ax.axvline(best_epoch, color='green', linestyle='--', alpha=0.7, 
                   label=f'Best Epoch ({best_epoch})')
        ax.plot(best_epoch, best_val_loss, 'g*', markersize=15)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('NLL Loss', fontsize=12)
    ax.set_title('Training History', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_uncertainty_analysis(
    y_true: np.ndarray,
    y_pred_mean: np.ndarray,
    y_pred_std: np.ndarray,
    save_dir: str | Path,
    prefix: str = "uq"
) -> dict[str, str]:
    """
    Generate all uncertainty quantification plots using uncertainty-toolbox.
    
    Args:
        y_true: True values
        y_pred_mean: Predicted means
        y_pred_std: Predicted standard deviations
        save_dir: Directory to save plots
        prefix: Filename prefix
    
    Returns:
        Dictionary mapping plot names to file paths
    """
    check_uncertainty_toolbox()
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    plot_paths = {}
    
    # 1. Calibration plot
    fig, ax = plt.subplots(figsize=(8, 6))
    uct.viz.plot_calibration(y_pred_mean, y_pred_std, y_true, ax=ax)
    path = save_dir / f"{prefix}_calibration.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    plot_paths['calibration'] = str(path)
    
    # 2. Sharpness distribution
    fig, ax = plt.subplots(figsize=(8, 6))
    uct.viz.plot_sharpness(y_pred_std, ax=ax)
    path = save_dir / f"{prefix}_sharpness.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    plot_paths['sharpness'] = str(path)
    
    # 3. Residuals vs uncertainties
    fig, ax = plt.subplots(figsize=(8, 6))
    uct.viz.plot_residuals_vs_stds(y_pred_mean, y_pred_std, y_true, ax=ax)
    path = save_dir / f"{prefix}_residuals_vs_std.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    plot_paths['residuals_vs_std'] = str(path)
    
    # 4. Ordered prediction intervals
    fig, ax = plt.subplots(figsize=(10, 6))
    uct.viz.plot_intervals_ordered(y_pred_mean, y_pred_std, y_true, ax=ax)
    path = save_dir / f"{prefix}_intervals_ordered.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    plot_paths['intervals_ordered'] = str(path)
    
    return plot_paths


def plot_prediction_scatter(
    y_true: np.ndarray,
    y_pred_mean: np.ndarray,
    y_pred_std: np.ndarray,
    save_path: str | Path | None = None
):
    """
    Scatter plot of predictions vs true values with uncertainty.
    
    Points are colored by uncertainty level.
    
    Args:
        y_true: True values
        y_pred_mean: Predicted means
        y_pred_std: Predicted standard deviations
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Scatter with color representing uncertainty
    scatter = ax.scatter(
        y_true, y_pred_mean,
        c=y_pred_std,
        cmap='viridis',
        alpha=0.6,
        s=50,
        edgecolors='black',
        linewidths=0.5
    )
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred_mean.min())
    max_val = max(y_true.max(), y_pred_mean.max())
    ax.plot([min_val, max_val], [min_val, max_val], 
            'r--', linewidth=2, alpha=0.7, label='Perfect Prediction')
    
    ax.set_xlabel('True Values', fontsize=12)
    ax.set_ylabel('Predicted Values', fontsize=12)
    ax.set_title('Predictions vs True Values\n(colored by uncertainty)', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Prediction Uncertainty (Std)', fontsize=11)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_prediction_intervals(
    y_true: np.ndarray,
    y_pred_mean: np.ndarray,
    y_pred_lower: np.ndarray,
    y_pred_upper: np.ndarray,
    indices: np.ndarray | None = None,
    max_samples: int = 50,
    save_path: str | Path | None = None
):
    """
    Plot prediction intervals for a subset of samples.
    
    Args:
        y_true: True values
        y_pred_mean: Predicted means
        y_pred_lower: Lower bound of intervals
        y_pred_upper: Upper bound of intervals
        indices: Sample indices to plot (plots first max_samples if None)
        max_samples: Maximum number of samples to plot
        save_path: Path to save figure
    """
    if indices is None:
        indices = np.arange(min(max_samples, len(y_true)))
    else:
        indices = indices[:max_samples]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(indices))
    
    # Plot intervals
    ax.fill_between(
        x,
        y_pred_lower[indices],
        y_pred_upper[indices],
        alpha=0.3,
        color='lightblue',
        label='95% Prediction Interval'
    )
    
    # Plot predictions and true values
    ax.plot(x, y_pred_mean[indices], 'b-o', label='Predicted Mean', 
            markersize=4, linewidth=1.5)
    ax.plot(x, y_true[indices], 'go', label='True Value', 
            markersize=6, alpha=0.7)
    
    # Highlight points outside interval
    outside = (y_true[indices] < y_pred_lower[indices]) | (y_true[indices] > y_pred_upper[indices])
    if np.any(outside):
        ax.plot(x[outside], y_true[indices][outside], 'rx', 
                markersize=10, markeredgewidth=2, 
                label='Outside 95% CI')
    
    ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Prediction Intervals', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_uncertainty_vs_error(
    y_true: np.ndarray,
    y_pred_mean: np.ndarray,
    y_pred_std: np.ndarray,
    save_path: str | Path | None = None
):
    """
    Plot prediction error vs uncertainty to check calibration.
    
    Well-calibrated models should show correlation between
    prediction error and uncertainty.
    
    Args:
        y_true: True values
        y_pred_mean: Predicted means
        y_pred_std: Predicted standard deviations
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Compute absolute errors
    abs_errors = np.abs(y_true - y_pred_mean)
    
    # Scatter plot
    ax.scatter(y_pred_std, abs_errors, alpha=0.5, s=30)
    
    # Add trendline
    z = np.polyfit(y_pred_std, abs_errors, 1)
    p = np.poly1d(z)
    x_line = np.linspace(y_pred_std.min(), y_pred_std.max(), 100)
    ax.plot(x_line, p(x_line), 'r--', linewidth=2, alpha=0.7, 
            label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
    
    # Compute correlation
    corr = np.corrcoef(y_pred_std, abs_errors)[0, 1]
    
    ax.set_xlabel('Predicted Uncertainty (Std)', fontsize=12)
    ax.set_ylabel('Absolute Prediction Error', fontsize=12)
    ax.set_title(f'Uncertainty vs Error (correlation: {corr:.3f})', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add text box
    textstr = f'Correlation: {corr:.3f}\n'
    if corr > 0.5:
        textstr += 'Good calibration'
    elif corr > 0.3:
        textstr += 'Moderate calibration'
    else:
        textstr += 'Poor calibration'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, 
            fontsize=11, verticalalignment='top', bbox=props)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_visualization_summary(
    y_true: np.ndarray,
    y_pred_mean: np.ndarray,
    y_pred_std: np.ndarray,
    train_history: dict | None = None,
    save_dir: str | Path = "results/probabilistic"
) -> dict[str, str]:
    """
    Create all visualizations for probabilistic model evaluation.
    
    Args:
        y_true: True values
        y_pred_mean: Predicted means
        y_pred_std: Predicted standard deviations
        train_history: Training history dict (optional)
        save_dir: Directory to save all plots
    
    Returns:
        Dictionary mapping plot names to file paths
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    plot_paths = {}
    
    # Training history
    if train_history:
        plot_training_history(
            train_history.get('train_losses', []),
            train_history.get('val_losses'),
            save_path=save_dir / "training_history.png"
        )
        plot_paths['training_history'] = str(save_dir / "training_history.png")
    
    # Uncertainty-toolbox plots
    uct_plots = plot_uncertainty_analysis(
        y_true, y_pred_mean, y_pred_std,
        save_dir=save_dir,
        prefix="uq"
    )
    plot_paths.update(uct_plots)
    
    # Custom plots
    plot_prediction_scatter(
        y_true, y_pred_mean, y_pred_std,
        save_path=save_dir / "prediction_scatter.png"
    )
    plot_paths['prediction_scatter'] = str(save_dir / "prediction_scatter.png")
    
    plot_prediction_intervals(
        y_true, y_pred_mean,
        y_pred_mean - 1.96 * y_pred_std,
        y_pred_mean + 1.96 * y_pred_std,
        save_path=save_dir / "prediction_intervals.png"
    )
    plot_paths['prediction_intervals'] = str(save_dir / "prediction_intervals.png")
    
    plot_uncertainty_vs_error(
        y_true, y_pred_mean, y_pred_std,
        save_path=save_dir / "uncertainty_vs_error.png"
    )
    plot_paths['uncertainty_vs_error'] = str(save_dir / "uncertainty_vs_error.png")
    
    return plot_paths


__all__ = [
    "plot_training_history",
    "plot_uncertainty_analysis",
    "plot_prediction_scatter",
    "plot_prediction_intervals",
    "plot_uncertainty_vs_error",
    "create_visualization_summary",
]
