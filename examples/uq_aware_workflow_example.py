"""
Complete UQ-aware workflow example.

Demonstrates the full uncertainty-aware ML pipeline:
1. Automatic model selection based on data size and use case
2. Training with native distribution prediction
3. Uncertainty quality evaluation
4. Automatic recommendations for improvement
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

from aims_agent.uq_workflow import run_uq_aware_workflow


def create_demo_data(n_samples: int = 1000, noise_level: float = 10.0):
    """Create synthetic regression data with heteroscedastic noise."""
    
    # Generate base data
    X, y = make_regression(
        n_samples=n_samples,
        n_features=5,
        n_informative=3,
        noise=0,  # We'll add custom noise
        random_state=42
    )
    
    # Add heteroscedastic noise (higher noise at extreme values)
    noise_scale = noise_level * (1 + 0.5 * np.abs(X[:, 0]))
    y += np.random.normal(0, noise_scale)
    
    return X, y


def main():
    """Run UQ-aware workflow examples with different use cases."""
    
    print("\n" + "="*80)
    print("UQ-AWARE WORKFLOW DEMONSTRATION")
    print("="*80)
    
    # ========================================================================
    # Example 1: Small dataset for screening (use GP)
    # ========================================================================
    print("\n\n" + "="*80)
    print("EXAMPLE 1: SMALL DATASET FOR MATERIALS SCREENING")
    print("="*80)
    
    X_small, y_small = create_demo_data(n_samples=500, noise_level=5.0)
    X_train, X_test, y_train, y_test = train_test_split(
        X_small, y_small, test_size=0.3, random_state=42
    )
    
    results_screening = run_uq_aware_workflow(
        X_train, y_train,
        X_test, y_test,
        use_case="screening",  # High UQ importance
        uq_importance="high",
        task_type="regression"
    )
    
    print("\n📊 Results Summary:")
    print(f"   Model: {results_screening['model_suggestion'].model_name}")
    print(f"   UQ Capability: {results_screening['model_suggestion'].uq_capability}")
    
    if results_screening['uq_evaluation']:
        print(f"   Calibration Error: {results_screening['uq_evaluation']['calibration_mae']:.4f}")
        print(f"   R²: {results_screening['uq_evaluation'].get('r2', 'N/A')}")
    
    # ========================================================================
    # Example 2: Medium dataset for active learning (use ProbabilisticNN)
    # ========================================================================
    print("\n\n" + "="*80)
    print("EXAMPLE 2: MEDIUM DATASET FOR ACTIVE LEARNING")
    print("="*80)
    
    X_medium, y_medium = create_demo_data(n_samples=2000, noise_level=8.0)
    X_train, X_test, y_train, y_test = train_test_split(
        X_medium, y_medium, test_size=0.3, random_state=42
    )
    
    results_active = run_uq_aware_workflow(
        X_train, y_train,
        X_test, y_test,
        use_case="active_learning",  # High UQ importance
        uq_importance="high",
        task_type="regression"
    )
    
    print("\n📊 Results Summary:")
    print(f"   Model: {results_active['model_suggestion'].model_name}")
    print(f"   UQ Capability: {results_active['model_suggestion'].uq_capability}")
    
    if results_active['uq_evaluation']:
        print(f"   Calibration Error: {results_active['uq_evaluation']['calibration_mae']:.4f}")
        print(f"   R²: {results_active['uq_evaluation'].get('r2', 'N/A')}")
    
    # Identify high-uncertainty samples for validation
    if results_active['distribution']:
        dist = results_active['distribution']
        high_unc_indices = np.argsort(dist['std'])[-10:][::-1]
        print(f"\n   🎯 Top 10 samples for experimental validation:")
        for i, idx in enumerate(high_unc_indices[:5], 1):
            print(f"      {i}. Sample {idx}: uncertainty={dist['std'][idx]:.4f}")
    
    # ========================================================================
    # Example 3: Large dataset for exploration (use XGBoost)
    # ========================================================================
    print("\n\n" + "="*80)
    print("EXAMPLE 3: LARGE DATASET FOR EXPLORATION")
    print("="*80)
    
    X_large, y_large = create_demo_data(n_samples=5000, noise_level=12.0)
    X_train, X_test, y_train, y_test = train_test_split(
        X_large, y_large, test_size=0.3, random_state=42
    )
    
    results_explore = run_uq_aware_workflow(
        X_train, y_train,
        X_test, y_test,
        use_case="exploration",  # Low UQ importance, speed matters
        uq_importance="low",
        task_type="regression"
    )
    
    print("\n📊 Results Summary:")
    print(f"   Model: {results_explore['model_suggestion'].model_name}")
    print(f"   UQ Capability: {results_explore['model_suggestion'].uq_capability}")
    
    # ========================================================================
    # Comparison Summary
    # ========================================================================
    print("\n\n" + "="*80)
    print("COMPARISON: MODEL SELECTION BASED ON USE CASE")
    print("="*80)
    
    comparison_data = [
        {
            "Use Case": "Screening",
            "Data Size": 500,
            "Model": results_screening['model_suggestion'].model_name,
            "UQ Quality": results_screening['model_suggestion'].uq_quality,
            "Reason": "Gold-standard calibrated UQ for critical decisions"
        },
        {
            "Use Case": "Active Learning",
            "Data Size": 2000,
            "Model": results_active['model_suggestion'].model_name,
            "UQ Quality": results_active['model_suggestion'].uq_quality,
            "Reason": "Scalable native distribution prediction"
        },
        {
            "Use Case": "Exploration",
            "Data Size": 5000,
            "Model": results_explore['model_suggestion'].model_name,
            "UQ Quality": results_explore['model_suggestion'].uq_quality,
            "Reason": "Fast exploratory analysis, accuracy-first"
        }
    ]
    
    print(f"\n{'Use Case':<20} {'Size':<8} {'Model':<30} {'UQ Quality':<12} Reason")
    print("-" * 120)
    for row in comparison_data:
        print(f"{row['Use Case']:<20} {row['Data Size']:<8} "
              f"{row['Model']:<30} {row['UQ Quality']:<12.2f} {row['Reason']}")
    
    # ========================================================================
    # Key Takeaways
    # ========================================================================
    print("\n\n" + "="*80)
    print("KEY TAKEAWAYS")
    print("="*80)
    
    print("""
1. **Model Selection is Use-Case Dependent**:
   - Screening/Active Learning → High UQ models (GP, ProbabilisticNN)
   - Exploration → Fast models with reasonable UQ (XGBoost, RandomForest)
   
2. **Data Size Matters**:
   - < 1K samples → GaussianProcess (best calibration)
   - 1K-50K samples → ProbabilisticNN (scalable native UQ)
   - > 50K samples → Ensemble methods (robust)
   
3. **Calibration is Critical**:
   - Always check calibration_error after training
   - If error > 0.15, consider model upgrade or recalibration
   - Well-calibrated models (error < 0.1) are safe for decision-making
   
4. **Automatic Recommendations**:
   - System automatically suggests improvements
   - Identifies high-uncertainty samples for validation
   - Recommends model upgrades when appropriate
""")
    
    print("\n" + "="*80)
    print("✓ ALL EXAMPLES COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
