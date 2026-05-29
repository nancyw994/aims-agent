"""
Quickstart: Using ProbabilisticNN in the integrated AIMS Agent system.

This example shows how ProbabilisticNN is automatically selected and used
in the complete workflow.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Import the integrated system components
from aims_agent.model_selector import select_uq_aware_model
from aims_agent.model_trainer import ModelTrainer
from aims_agent.distribution_predictor import predict_distribution
from aims_agent.uncertainty_evaluator import UncertaintyEvaluator


def main():
    print("\n" + "="*80)
    print("QUICKSTART: ProbabilisticNN in Integrated System")
    print("="*80)
    
    # ========================================================================
    # 1. Prepare Data
    # ========================================================================
    print("\n[Step 1/5] Preparing synthetic materials data...")
    
    np.random.seed(42)
    n_samples = 2000  # Medium-sized dataset
    
    # Simulate materials features
    composition_A = np.random.uniform(0, 1, n_samples)
    composition_B = np.random.uniform(0, 1, n_samples)
    temperature = np.random.uniform(300, 1500, n_samples)
    pressure = np.random.uniform(1, 100, n_samples)
    
    # Target with heteroscedastic noise
    base_property = (
        15 * composition_A ** 1.5
        + 8 * composition_B
        + 0.01 * temperature
        - 0.02 * pressure
    )
    noise_scale = 0.5 + 2.0 * np.abs(composition_A - 0.5)  # Heteroscedastic
    property_value = base_property + np.random.normal(0, noise_scale)
    
    # Create DataFrame
    df = pd.DataFrame({
        'composition_A': composition_A,
        'composition_B': composition_B,
        'temperature': temperature,
        'pressure': pressure,
        'property': property_value
    })
    
    print(f"   ✓ Created {len(df)} samples with 4 features")
    
    # ========================================================================
    # 2. Automatic Model Selection
    # ========================================================================
    print("\n[Step 2/5] Automatic model selection...")
    
    suggestion = select_uq_aware_model(
        n_samples=len(df),
        n_features=4,
        task_type="regression",
        use_case="screening"  # High UQ importance
    )
    
    print(f"   Selected model: {suggestion.model_name}")
    print(f"   UQ capability: {suggestion.uq_capability}")
    print(f"   UQ quality: {suggestion.uq_quality}")
    print(f"   Heteroscedastic: {suggestion.heteroscedastic}")
    print(f"   Reason: {suggestion.reason}")
    
    if suggestion.model_name != "ProbabilisticNN":
        print(f"\n   ⚠️  Note: PNN not selected (got {suggestion.model_name})")
        print(f"   This is OK - selection is based on data size and use case")
    
    # ========================================================================
    # 3. Training with Standard Workflow
    # ========================================================================
    print("\n[Step 3/5] Training model...")
    
    # Import the selected model
    if suggestion.model_name == "ProbabilisticNN":
        from aims_agent.probabilistic_models import ProbabilisticNNWrapper
        model_class = ProbabilisticNNWrapper
    else:
        # For other models, dynamically import
        try:
            import importlib
            module_path, class_name = suggestion.import_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name)
        except Exception as e:
            print(f"   ⚠️  Could not import {suggestion.model_name}: {e}")
            print(f"   Falling back to ProbabilisticNN for this demo")
            from aims_agent.probabilistic_models import ProbabilisticNNWrapper
            model_class = ProbabilisticNNWrapper
    
    # Create trainer
    trainer = ModelTrainer(
        model_class=model_class,
        task_type="regression",
        use_hyperparameter_tuning=False  # Skip for demo speed
    )
    
    # Prepare and train
    features = ['composition_A', 'composition_B', 'temperature', 'pressure']
    trainer.prepare_data(df, features=features, target='property')
    trainer.train()
    
    print(f"   ✓ Training completed")
    
    # ========================================================================
    # 4. Distribution Prediction
    # ========================================================================
    print("\n[Step 4/5] Predicting distributions...")
    
    # Method 1: Using trainer's built-in method
    result = trainer.predict_with_uncertainty()
    
    print(f"   Predictions shape: {result['y_pred'].shape}")
    print(f"   Mean uncertainty: {np.mean(result['y_std']):.4f}")
    print(f"   Max uncertainty: {np.max(result['y_std']):.4f}")
    print(f"   Min uncertainty: {np.min(result['y_std']):.4f}")
    
    # Method 2: Using distribution_predictor (alternative)
    # dist = predict_distribution(trainer.model, trainer.X_test)
    
    # ========================================================================
    # 5. Uncertainty Quality Evaluation
    # ========================================================================
    print("\n[Step 5/5] Evaluating uncertainty quality...")
    
    try:
        summary, full_metrics = UncertaintyEvaluator.evaluate_all(
            result['y_true'],
            result['y_pred'],
            result['y_std'],
            verbose=False
        )
        
        print(f"\n   📊 Accuracy:")
        print(f"      RMSE: {summary['rmse']:.4f}")
        print(f"      MAE: {summary['mae']:.4f}")
        print(f"      R²: {summary.get('r2', 'N/A')}")
        
        print(f"\n   📐 Calibration:")
        cal_error = summary['calibration_mae']
        print(f"      Calibration Error: {cal_error:.4f}")
        
        if cal_error < 0.05:
            print(f"      ✓ EXCELLENT calibration")
        elif cal_error < 0.1:
            print(f"      ✓ GOOD calibration")
        elif cal_error < 0.15:
            print(f"      ⚠ MODERATE calibration")
        else:
            print(f"      ✗ POOR calibration")
        
        print(f"\n   🎯 Probabilistic Scoring:")
        print(f"      NLL: {summary.get('nll', 'N/A')}")
        print(f"      CRPS: {summary.get('crps', 'N/A')}")
        
        # Identify high-uncertainty samples
        high_unc_info = UncertaintyEvaluator.identify_high_uncertainty_samples(
            result['y_std'], n_top=5
        )
        
        print(f"\n   📍 Active Learning:")
        print(f"      High-uncertainty samples: {high_unc_info['n_high_uncertainty']}")
        print(f"      Percentage: {high_unc_info['percentage']:.1f}%")
        print(f"      Top 5 for validation: {high_unc_info['top_n_indices'][:5]}")
        
    except ImportError:
        print("   ⚠️  uncertainty-toolbox not available")
        print("   Install with: pip install uncertainty-toolbox")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*80)
    print("✓ WORKFLOW COMPLETE!")
    print("="*80)
    
    print("\n💡 What happened:")
    print(f"   1. System analyzed {len(df)} samples")
    print(f"   2. Selected {suggestion.model_name} (UQ quality={suggestion.uq_quality})")
    print(f"   3. Trained model with standard workflow")
    print(f"   4. Predicted distributions with uncertainty")
    print(f"   5. Evaluated UQ quality automatically")
    
    print("\n🎯 Key Features:")
    print("   ✓ Automatic model selection based on data size and use case")
    print("   ✓ Native distribution prediction (not post-hoc)")
    print("   ✓ Heteroscedastic uncertainty (varies with input)")
    print("   ✓ Professional UQ evaluation metrics")
    print("   ✓ Active learning sample identification")
    
    print("\n📚 Next Steps:")
    print("   - Validate the top high-uncertainty samples experimentally")
    print("   - Use predictions for materials screening")
    print("   - Iterate with active learning")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
