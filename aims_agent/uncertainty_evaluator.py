"""
Uncertainty quantification evaluator using uncertainty-toolbox.

This module evaluates the quality of predicted uncertainties using
comprehensive metrics for calibration, sharpness, and scoring rules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    import uncertainty_toolbox as uct
    HAS_UCT = True
except ImportError:
    HAS_UCT = False
    uct = None


class UncertaintyEvaluator:
    """
    Evaluator for uncertainty quantification quality.
    
    Uses uncertainty-toolbox to compute calibration, sharpness,
    and proper scoring rules for predicted distributions.
    """
    
    @staticmethod
    def check_availability():
        """Check if uncertainty-toolbox is available."""
        if not HAS_UCT:
            raise ImportError(
                "uncertainty-toolbox is required for uncertainty evaluation.\n"
                "Install it with: pip install uncertainty-toolbox"
            )
    
    @classmethod
    def evaluate_all(
        cls,
        y_true: np.ndarray,
        y_pred_mean: np.ndarray,
        y_pred_std: np.ndarray,
        verbose: bool = False
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Compute all uncertainty quantification metrics.
        
        Args:
            y_true: True values
            y_pred_mean: Predicted means
            y_pred_std: Predicted standard deviations
            verbose: Print metrics table
        
        Returns:
            summary: Dict with key metrics
            full_metrics: Dict with all metrics from uncertainty-toolbox
        """
        cls.check_availability()
        
        # Ensure arrays are proper numpy arrays
        y_true = np.asarray(y_true).flatten()
        y_pred_mean = np.asarray(y_pred_mean).flatten()
        y_pred_std = np.asarray(y_pred_std).flatten()
        
        # Get all metrics from uncertainty-toolbox
        full_metrics = uct.metrics.get_all_metrics(
            y_pred_mean,
            y_pred_std,
            y_true,
            verbose=verbose
        )
        
        # Extract and organize key metrics
        accuracy = full_metrics.get('accuracy', {})
        calibration = full_metrics.get('avg_calibration', {})
        sharpness_dict = full_metrics.get('sharpness', {})
        scoring = full_metrics.get('scoring_rule', {})
        
        summary = {
            # Accuracy metrics
            'rmse': accuracy.get('rmse'),
            'mae': accuracy.get('mae'),
            'r2': accuracy.get('rsq'),
            
            # Calibration metrics
            'calibration_mae': calibration.get('ma_cal'),
            'calibration_rmse': calibration.get('rms_cal'),
            'miscalibration_area': calibration.get('miscal_area'),
            
            # Sharpness
            'sharpness': sharpness_dict.get('sharpness', sharpness_dict.get('sharp')),
            
            # Proper scoring rules
            'nll': scoring.get('nll'),
            'crps': scoring.get('crps'),
            'check_score': scoring.get('check'),
            'interval_score': scoring.get('interval'),
        }
        
        return summary, full_metrics
    
    @staticmethod
    def interpret_metrics(summary: dict[str, Any]) -> str:
        """
        Generate human-readable interpretation of UQ metrics.
        
        Args:
            summary: Summary metrics dict
        
        Returns:
            Multi-line interpretation string
        """
        lines = []
        lines.append("=" * 60)
        lines.append("UNCERTAINTY QUANTIFICATION EVALUATION")
        lines.append("=" * 60)
        
        # Accuracy assessment
        rmse = summary.get('rmse')
        mae = summary.get('mae')
        r2 = summary.get('r2')
        
        lines.append("\n📊 Prediction Accuracy:")
        if rmse is not None:
            lines.append(f"   RMSE: {rmse:.4f}")
        if mae is not None:
            lines.append(f"   MAE:  {mae:.4f}")
        if r2 is not None:
            lines.append(f"   R²:   {r2:.4f}")
            if r2 > 0.9:
                lines.append("   ✓ Excellent prediction accuracy")
            elif r2 > 0.7:
                lines.append("   ✓ Good prediction accuracy")
            else:
                lines.append("   ⚠ Moderate prediction accuracy")
        
        # Calibration assessment
        cal_mae = summary.get('calibration_mae')
        miscal_area = summary.get('miscalibration_area')
        
        lines.append("\n📐 Calibration Quality:")
        if cal_mae is not None:
            lines.append(f"   Calibration Error (MAE): {cal_mae:.4f}")
            if cal_mae < 0.05:
                lines.append("   ✓ EXCELLENT calibration - uncertainties are highly reliable")
                lines.append("     (95% intervals should contain ~95% of true values)")
            elif cal_mae < 0.1:
                lines.append("   ✓ GOOD calibration - uncertainties are mostly reliable")
            elif cal_mae < 0.2:
                lines.append("   ⚠ MODERATE calibration - some miscalibration present")
            else:
                lines.append("   ✗ POOR calibration - uncertainties may be unreliable")
                lines.append("     Consider recalibration or using ensembles")
        
        if miscal_area is not None:
            lines.append(f"   Miscalibration Area: {miscal_area:.4f}")
        
        # Sharpness assessment
        sharpness = summary.get('sharpness')
        
        lines.append("\n🎯 Sharpness (Confidence):")
        if sharpness is not None:
            lines.append(f"   Average Uncertainty: {sharpness:.4f}")
            if sharpness < 0.5:
                lines.append("   ✓ Sharp predictions (model is confident)")
            elif sharpness < 1.0:
                lines.append("   ○ Moderate uncertainty levels")
            else:
                lines.append("   ⚠ Wide uncertainties (model is less confident)")
        else:
            lines.append("   N/A")
        
        # Scoring rules assessment
        nll = summary.get('nll')
        crps = summary.get('crps')
        
        lines.append("\n📈 Probabilistic Scoring Rules:")
        if nll is not None:
            lines.append(f"   NLL (Negative Log-Likelihood): {nll:.4f}")
            if nll < 1.0:
                lines.append("   ✓ Good probabilistic predictions")
            elif nll < 2.0:
                lines.append("   ○ Moderate probabilistic quality")
            else:
                lines.append("   ⚠ Poor probabilistic quality")
        
        if crps is not None:
            lines.append(f"   CRPS (Continuous Ranked Probability Score): {crps:.4f}")
        
        # Overall recommendation
        lines.append("\n" + "=" * 60)
        lines.append("RECOMMENDATIONS:")
        lines.append("=" * 60)
        
        if cal_mae is not None and cal_mae < 0.1:
            lines.append("✓ Model uncertainties are well-calibrated")
            lines.append("  → Safe to use for risk assessment and decision-making")
        else:
            lines.append("⚠ Consider improving calibration:")
            lines.append("  → Try temperature scaling or recalibration")
            lines.append("  → Use ensemble methods (e.g., Deep Ensembles)")
        
        if sharpness is not None and sharpness > 1.0:
            lines.append("\n⚠ High uncertainty levels detected:")
            lines.append("  → Collect more training data in uncertain regions")
            lines.append("  → Consider active learning for efficient data collection")
        
        return "\n".join(lines)
    
    @classmethod
    def identify_high_uncertainty_samples(
        cls,
        y_pred_std: np.ndarray,
        threshold: float | None = None,
        n_top: int = 10
    ) -> dict[str, Any]:
        """
        Identify samples with highest prediction uncertainty.
        
        Args:
            y_pred_std: Predicted standard deviations
            threshold: Uncertainty threshold (uses 75th percentile if None)
            n_top: Number of top uncertain samples to return
        
        Returns:
            Dictionary with high-uncertainty sample info
        """
        y_pred_std = np.asarray(y_pred_std).flatten()
        
        # Set threshold if not provided
        if threshold is None:
            threshold = np.percentile(y_pred_std, 75)
        
        # Find high-uncertainty samples
        high_unc_mask = y_pred_std > threshold
        high_unc_indices = np.where(high_unc_mask)[0]
        
        # Get top N
        top_indices = np.argsort(y_pred_std)[-n_top:][::-1]
        
        return {
            'threshold': float(threshold),
            'n_high_uncertainty': int(np.sum(high_unc_mask)),
            'percentage': float(np.mean(high_unc_mask) * 100),
            'high_uncertainty_indices': high_unc_indices.tolist(),
            'top_n_indices': top_indices.tolist(),
            'top_n_uncertainties': y_pred_std[top_indices].tolist(),
            'mean_uncertainty': float(np.mean(y_pred_std)),
            'median_uncertainty': float(np.median(y_pred_std)),
            'max_uncertainty': float(np.max(y_pred_std)),
        }
    
    @classmethod
    def compute_coverage(
        cls,
        y_true: np.ndarray,
        y_pred_mean: np.ndarray,
        y_pred_std: np.ndarray,
        confidence_levels: list[float] | None = None
    ) -> dict[float, float]:
        """
        Compute empirical coverage at different confidence levels.
        
        Args:
            y_true: True values
            y_pred_mean: Predicted means
            y_pred_std: Predicted standard deviations
            confidence_levels: List of confidence levels (default: [0.68, 0.95, 0.99])
        
        Returns:
            Dictionary mapping confidence level to empirical coverage
        """
        if confidence_levels is None:
            confidence_levels = [0.68, 0.95, 0.99]
        
        y_true = np.asarray(y_true).flatten()
        y_pred_mean = np.asarray(y_pred_mean).flatten()
        y_pred_std = np.asarray(y_pred_std).flatten()
        
        # Map confidence to z-score
        from scipy import stats
        
        coverage = {}
        for conf_level in confidence_levels:
            z = stats.norm.ppf((1 + conf_level) / 2)
            
            lower = y_pred_mean - z * y_pred_std
            upper = y_pred_mean + z * y_pred_std
            
            in_interval = (y_true >= lower) & (y_true <= upper)
            empirical_coverage = np.mean(in_interval)
            
            coverage[conf_level] = float(empirical_coverage)
        
        return coverage
    
    @classmethod
    def save_evaluation_report(
        cls,
        y_true: np.ndarray,
        y_pred_mean: np.ndarray,
        y_pred_std: np.ndarray,
        output_path: str | Path,
        verbose: bool = True
    ):
        """
        Generate and save comprehensive evaluation report.
        
        Args:
            y_true: True values
            y_pred_mean: Predicted means
            y_pred_std: Predicted standard deviations
            output_path: Path to save report
            verbose: Print to console
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Compute all metrics
        summary, full_metrics = cls.evaluate_all(
            y_true, y_pred_mean, y_pred_std, verbose=False
        )
        
        # Get interpretation
        interpretation = cls.interpret_metrics(summary)
        
        # Get high-uncertainty samples
        high_unc_info = cls.identify_high_uncertainty_samples(y_pred_std)
        
        # Get coverage
        coverage = cls.compute_coverage(y_true, y_pred_mean, y_pred_std)
        
        # Build report
        report_lines = [interpretation]
        
        report_lines.append("\n" + "=" * 60)
        report_lines.append("COVERAGE ANALYSIS:")
        report_lines.append("=" * 60)
        for conf_level, empirical in coverage.items():
            expected_pct = conf_level * 100
            empirical_pct = empirical * 100
            diff = abs(empirical_pct - expected_pct)
            
            status = "✓" if diff < 5 else "⚠" if diff < 10 else "✗"
            report_lines.append(
                f"{status} {expected_pct:.0f}% CI: {empirical_pct:.1f}% empirical coverage"
            )
        
        report_lines.append("\n" + "=" * 60)
        report_lines.append("HIGH-UNCERTAINTY SAMPLES:")
        report_lines.append("=" * 60)
        report_lines.append(f"Threshold: {high_unc_info['threshold']:.4f}")
        report_lines.append(f"Samples above threshold: {high_unc_info['n_high_uncertainty']} ({high_unc_info['percentage']:.1f}%)")
        report_lines.append(f"\nTop 10 highest uncertainty samples:")
        for i, (idx, unc) in enumerate(zip(
            high_unc_info['top_n_indices'][:10],
            high_unc_info['top_n_uncertainties'][:10]
        ), 1):
            report_lines.append(f"  {i:2d}. Sample {idx:4d}: uncertainty = {unc:.4f}")
        
        report = "\n".join(report_lines)
        
        # Save to file
        output_path.write_text(report, encoding='utf-8')
        
        if verbose:
            print(report)
            print(f"\n✓ Report saved to: {output_path}")


__all__ = ["UncertaintyEvaluator"]
