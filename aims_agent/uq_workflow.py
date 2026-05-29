"""
UQ-aware ML workflow orchestrator.

This module implements the complete uncertainty-aware workflow:
1. Select model based on UQ requirements
2. Train model
3. Predict distributions
4. Evaluate UQ quality
5. Auto-adjust if UQ is poorly calibrated
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from aims_agent.agent import Agent

from aims_agent.model_selector import (
    ModelSuggestion,
    select_uq_aware_model,
    enrich_suggestion_with_uq_metadata,
)
from aims_agent.distribution_predictor import (
    predict_distribution,
    can_predict_distribution,
    get_uq_capability,
)


class UQWorkflowOrchestrator:
    """
    Orchestrates uncertainty-aware ML workflow.
    
    Workflow:
        Data → UQ-aware Model Selection → Training → Distribution Prediction
        → Uncertainty Evaluation → Report + Recommendations
    """
    
    def __init__(
        self,
        agent: "Agent" = None,
        use_case: str = "exploration",
        uq_importance: str = "medium",
    ):
        """
        Initialize UQ workflow orchestrator.
        
        Args:
            agent: Agent instance (for LLM calls)
            use_case: "exploration", "screening", "active_learning", "production"
            uq_importance: "low", "medium", "high" - how important is UQ calibration
        """
        self.agent = agent
        self.use_case = use_case
        self.uq_importance = uq_importance
        
        self.model_suggestion = None
        self.trained_model = None
        self.distribution_result = None
        self.uq_evaluation = None
        self.recommendations = []
    
    def run_workflow(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        task_type: str = "regression",
        features: list[str] | None = None,
        target: str = "target",
    ) -> dict[str, Any]:
        """
        Run complete UQ-aware ML workflow.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            task_type: "regression" or "classification"
            features: Feature names
            target: Target name
        
        Returns:
            Dictionary with all workflow results
        """
        
        print("\n" + "="*80)
        print("UQ-AWARE ML WORKFLOW")
        print("="*80)
        
        # Step 1: Profile dataset
        profile = self._profile_dataset(X_train, y_train)
        print(f"\n[Step 1/6] Dataset Profile:")
        print(f"   Samples: {profile['n_samples']}")
        print(f"   Features: {profile['n_features']}")
        print(f"   Use case: {self.use_case}")
        print(f"   UQ importance: {self.uq_importance}")
        
        # Step 2: Select UQ-aware model
        self.model_suggestion = select_uq_aware_model(
            n_samples=profile['n_samples'],
            n_features=profile['n_features'],
            task_type=task_type,
            use_case=self.use_case,
        )
        
        print(f"\n[Step 2/6] Model Selection:")
        print(f"   Model: {self.model_suggestion.model_name}")
        print(f"   UQ Capability: {self.model_suggestion.uq_capability}")
        print(f"   UQ Quality: {self.model_suggestion.uq_quality:.2f}")
        print(f"   Heteroscedastic: {self.model_suggestion.heteroscedastic}")
        print(f"   Reason: {self.model_suggestion.reason}")
        
        # Step 3: Train model
        print(f"\n[Step 3/6] Training model...")
        self.trained_model = self._train_model(
            self.model_suggestion,
            X_train, y_train,
            task_type=task_type
        )
        print(f"   ✓ Training complete")
        
        # Step 4: Predict distribution
        print(f"\n[Step 4/6] Predicting distributions...")
        self.distribution_result = self._predict_with_fallback(
            self.trained_model,
            X_test, y_test,
            self.model_suggestion
        )
        
        if self.distribution_result is not None:
            mean_unc = np.mean(self.distribution_result['std'])
            print(f"   ✓ Distribution prediction successful")
            print(f"   Mean uncertainty: {mean_unc:.4f}")
        else:
            print(f"   ⚠ Model doesn't support distribution prediction")
        
        # Step 5: Evaluate UQ quality
        if self.distribution_result is not None:
            print(f"\n[Step 5/6] Evaluating UQ quality...")
            self.uq_evaluation = self._evaluate_uq_quality(
                y_test,
                self.distribution_result['mu'],
                self.distribution_result['std']
            )
            
            if self.uq_evaluation is not None:
                print(f"   Calibration Error: {self.uq_evaluation['calibration_mae']:.4f}")
                print(f"   NLL: {self.uq_evaluation.get('nll', 'N/A')}")
                print(f"   R²: {self.uq_evaluation.get('r2', 'N/A')}")
        else:
            print(f"\n[Step 5/6] Skipping UQ evaluation (no distribution)")
            self.uq_evaluation = None
        
        # Step 6: Generate recommendations
        print(f"\n[Step 6/6] Generating recommendations...")
        self.recommendations = self._generate_recommendations(
            self.model_suggestion,
            self.distribution_result,
            self.uq_evaluation,
            profile
        )
        
        for i, rec in enumerate(self.recommendations, 1):
            print(f"   {i}. {rec}")
        
        # Compile results
        results = {
            'profile': profile,
            'model_suggestion': self.model_suggestion,
            'trained_model': self.trained_model,
            'distribution': self.distribution_result,
            'uq_evaluation': self.uq_evaluation,
            'recommendations': self.recommendations,
        }
        
        print("\n" + "="*80)
        print("✓ WORKFLOW COMPLETE")
        print("="*80)
        
        return results
    
    def _profile_dataset(self, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        """Profile dataset characteristics."""
        return {
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'target_mean': np.mean(y),
            'target_std': np.std(y),
            'target_range': (np.min(y), np.max(y)),
        }
    
    def _train_model(
        self,
        suggestion: ModelSuggestion,
        X_train: np.ndarray,
        y_train: np.ndarray,
        task_type: str = "regression"
    ) -> Any:
        """Train the selected model."""
        
        # Special handling for ProbabilisticNN
        if suggestion.model_name == "ProbabilisticNN":
            from aims_agent.probabilistic_trainer import ProbabilisticTrainer
            
            trainer = ProbabilisticTrainer(
                input_dim=X_train.shape[1],
                hidden_dim=128,
                num_layers=2,
                epochs=100,
                patience=15,
                verbose=False
            )
            trainer.train(X_train, y_train)
            return trainer
        
        # Standard sklearn/xgboost/etc models
        else:
            from aims_agent.model_selector import dynamic_import
            
            try:
                model_class = dynamic_import(suggestion.import_path)
                model = model_class()
                model.fit(X_train, y_train)
                return model
            
            except Exception as e:
                print(f"   ⚠ Failed to train {suggestion.model_name}: {e}")
                raise
    
    def _predict_with_fallback(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        suggestion: ModelSuggestion
    ) -> dict[str, np.ndarray] | None:
        """
        Predict distribution with fallback for models without native UQ.
        """
        
        try:
            # Try native distribution prediction
            dist = predict_distribution(model, X_test)
            
            if dist is not None:
                return dist
        
        except Exception as e:
            print(f"   ⚠ Distribution prediction failed: {e}")
        
        # Fallback: point prediction only
        if hasattr(model, "predict"):
            y_pred = model.predict(X_test)
            print(f"   ⚠ Falling back to point predictions only")
            
            # Return with dummy uncertainty
            return {
                'mu': y_pred,
                'std': np.zeros_like(y_pred),
                'var': np.zeros_like(y_pred),
                'lower_95': y_pred,
                'upper_95': y_pred,
            }
        
        return None
    
    def _evaluate_uq_quality(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_std: np.ndarray
    ) -> dict[str, float] | None:
        """Evaluate UQ quality using uncertainty-toolbox."""
        
        # Check if uncertainty-toolbox is available
        try:
            from aims_agent.uncertainty_evaluator import UncertaintyEvaluator
            
            # Check if there's actual uncertainty
            if np.all(y_std == 0):
                print(f"   ⚠ No uncertainty estimates available")
                return None
            
            summary, _ = UncertaintyEvaluator.evaluate_all(
                y_true, y_pred, y_std, verbose=False
            )
            
            return summary
        
        except ImportError:
            print(f"   ⚠ uncertainty-toolbox not available, skipping UQ evaluation")
            return None
        
        except Exception as e:
            print(f"   ⚠ UQ evaluation failed: {e}")
            return None
    
    def _generate_recommendations(
        self,
        suggestion: ModelSuggestion,
        distribution: dict | None,
        uq_eval: dict | None,
        profile: dict
    ) -> list[str]:
        """Generate actionable recommendations based on results."""
        
        recommendations = []
        
        # Check if UQ is available
        if distribution is None or np.all(distribution['std'] == 0):
            recommendations.append(
                f"⚠ Model '{suggestion.model_name}' doesn't provide uncertainty estimates. "
                f"Consider switching to: GaussianProcess (< 5K samples), "
                f"ProbabilisticNN (500-50K), or Ensemble methods."
            )
            return recommendations
        
        # Check UQ quality
        if uq_eval is not None:
            cal_error = uq_eval.get('calibration_mae', 999)
            
            if cal_error > 0.2:
                recommendations.append(
                    f"✗ POOR CALIBRATION (error={cal_error:.3f}): "
                    f"Uncertainties are unreliable. Recommended actions:\n"
                    f"   - Switch to GaussianProcess (best calibration)\n"
                    f"   - Use Deep Ensemble (robust epistemic uncertainty)\n"
                    f"   - Apply temperature scaling for recalibration\n"
                    f"   - Collect more training data"
                )
            
            elif cal_error > 0.1:
                recommendations.append(
                    f"⚠ MODERATE CALIBRATION (error={cal_error:.3f}): "
                    f"Consider recalibration or using a more sophisticated model."
                )
            
            else:
                recommendations.append(
                    f"✓ GOOD CALIBRATION (error={cal_error:.3f}): "
                    f"Uncertainties are reliable for decision-making."
                )
            
            # Sharpness check
            sharpness = uq_eval.get('sharpness')
            if sharpness is not None and sharpness > 1.0:
                recommendations.append(
                    f"⚠ HIGH UNCERTAINTY (avg={sharpness:.3f}): "
                    f"Model is not confident. Consider:\n"
                    f"   - Active learning to collect data in uncertain regions\n"
                    f"   - Feature engineering to improve predictive power\n"
                    f"   - Using ensemble methods for better coverage"
                )
        
        # Use case specific recommendations
        if self.use_case in ["screening", "active_learning"]:
            high_unc_count = np.sum(distribution['std'] > np.percentile(distribution['std'], 75))
            recommendations.append(
                f"📍 Active Learning: {high_unc_count} samples identified with high uncertainty. "
                f"Prioritize these for experimental validation."
            )
        
        # Model upgrade suggestions
        if suggestion.uq_quality < 0.8 and profile['n_samples'] < 5000:
            recommendations.append(
                f"💡 MODEL UPGRADE: With {profile['n_samples']} samples, "
                f"GaussianProcess could provide better UQ (quality=1.0) than "
                f"{suggestion.model_name} (quality={suggestion.uq_quality:.2f})"
            )
        
        if not recommendations:
            recommendations.append("✓ No issues detected. Results are ready for use.")
        
        return recommendations


def run_uq_aware_workflow(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    use_case: str = "exploration",
    uq_importance: str = "medium",
    task_type: str = "regression",
    agent: "Agent" = None,
) -> dict[str, Any]:
    """
    Convenience function to run UQ-aware workflow.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        use_case: "exploration", "screening", "active_learning", "production"
        uq_importance: "low", "medium", "high"
        task_type: "regression" or "classification"
        agent: Agent instance (optional)
    
    Returns:
        Dictionary with workflow results
    """
    
    orchestrator = UQWorkflowOrchestrator(
        agent=agent,
        use_case=use_case,
        uq_importance=uq_importance
    )
    
    return orchestrator.run_workflow(
        X_train, y_train,
        X_test, y_test,
        task_type=task_type
    )


__all__ = [
    "UQWorkflowOrchestrator",
    "run_uq_aware_workflow",
]
