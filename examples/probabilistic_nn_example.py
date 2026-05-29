"""
Complete example: Probabilistic Neural Network for uncertainty quantification.

This script demonstrates the full workflow:
1. Create/load materials data
2. Train probabilistic neural network
3. Predict distributions
4. Evaluate uncertainty quality
5. Visualize results
6. Generate interpretation
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from aims_agent.probabilistic_trainer import ProbabilisticTrainer
from aims_agent.uncertainty_evaluator import UncertaintyEvaluator
from aims_agent.probabilistic_visualizer import create_visualization_summary


def create_synthetic_materials_data(n_samples: int = 300):
    """
    Create synthetic materials data for demonstration.
    
    Simulates materials properties with input-dependent uncertainty
    (heteroscedastic noise).
    """
    np.random.seed(42)
    
    # Features
    composition_A = np.random.uniform(0, 1, n_samples)
    composition_B = np.random.uniform(0, 1, n_samples)
    temperature = np.random.uniform(300, 1500, n_samples)
    pressure = np.random.uniform(1, 100, n_samples)
    grain_size = np.random.lognormal(2, 0.5, n_samples)
    
    # Target with heteroscedastic noise
    # (uncertainty varies with input - higher at extreme compositions)
    base_hardness = (
        15 * composition_A ** 1.5
        + 8 * composition_B
        + 0.015 * temperature
        - 0.03 * pressure
        + 2 * np.log(grain_size)
    )
    
    # Heteroscedastic noise: higher noise at extreme compositions
    noise_scale = 0.5 + 2.0 * np.abs(composition_A - 0.5)
    noise = np.random.normal(0, noise_scale)
    
    hardness = base_hardness + noise
    
    # Create dataframe
    df = pd.DataFrame({
        'composition_A': composition_A,
        'composition_B': composition_B,
        'temperature': temperature,
        'pressure': pressure,
        'grain_size': grain_size,
        'hardness': hardness
    })
    
    return df


def main():
    """Run the complete probabilistic NN workflow."""
    
    print("\n" + "="*80)
    print("PROBABILISTIC NEURAL NETWORK FOR UNCERTAINTY QUANTIFICATION")
    print("="*80)
    
    # ========================================================================
    # 1. Load Data
    # ========================================================================
    print("\n[Step 1/7] Loading data...")
    df = create_synthetic_materials_data(n_samples=300)
    
    features = ['composition_A', 'composition_B', 'temperature', 'pressure', 'grain_size']
    target = 'hardness'
    
    X = df[features].values
    y = df[target].values
    
    print(f"   ✓ Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"   ✓ Target range: [{y.min():.2f}, {y.max():.2f}]")
    
    # ========================================================================
    # 2. Train Probabilistic NN
    # ========================================================================
    print("\n[Step 2/7] Training Probabilistic Neural Network...")
    
    trainer = ProbabilisticTrainer(
        input_dim=X.shape[1],
        hidden_dim=128,
        num_layers=3,
        dropout_rate=0.1,
        learning_rate=1e-3,
        batch_size=32,
        epochs=200,
        patience=20,
        val_split=0.2,
        verbose=True
    )
    
    # Train
    train_info = trainer.train(X, y)
    
    print(f"\n   ✓ Training complete!")
    print(f"   - Final train loss: {train_info['final_train_loss']:.4f}")
    if train_info['final_val_loss']:
        print(f"   - Final val loss: {train_info['final_val_loss']:.4f}")
        print(f"   - Best epoch: {train_info['best_epoch'] + 1}")
    
    # ========================================================================
    # 3. Predict Distributions
    # ========================================================================
    print("\n[Step 3/7] Predicting distributions...")
    
    # For demo, predict on the full dataset
    # (in practice, use a separate test set)
    dist_result = trainer.predict_distribution(X)
    
    y_pred_mean = dist_result['mu']
    y_pred_std = dist_result['std']
    y_pred_lower = dist_result['lower_95']
    y_pred_upper = dist_result['upper_95']
    
    print(f"   ✓ Predictions generated")
    print(f"   - Mean uncertainty: {np.mean(y_pred_std):.4f}")
    print(f"   - Max uncertainty: {np.max(y_pred_std):.4f}")
    print(f"   - Min uncertainty: {np.min(y_pred_std):.4f}")
    
    # ========================================================================
    # 4. Evaluate Uncertainty Quality
    # ========================================================================
    print("\n[Step 4/7] Evaluating uncertainty quality...")
    
    try:
        summary, full_metrics = UncertaintyEvaluator.evaluate_all(
            y, y_pred_mean, y_pred_std, verbose=True
        )
        
        print("\n   📊 Key Metrics:")
        print(f"   - RMSE: {summary['rmse']:.4f}")
        print(f"   - MAE: {summary['mae']:.4f}")
        print(f"   - R²: {summary.get('r2', 'N/A')}")
        print(f"   - Calibration Error: {summary['calibration_mae']:.4f}")
        print(f"   - NLL: {summary['nll']:.4f}")
        print(f"   - CRPS: {summary['crps']:.4f}")
        
    except ImportError as e:
        print(f"   ⚠️  {e}")
        print("   Skipping uncertainty evaluation")
        summary = None
    
    # ========================================================================
    # 5. Identify High-Uncertainty Samples
    # ========================================================================
    print("\n[Step 5/7] Identifying high-uncertainty samples...")
    
    high_unc_info = UncertaintyEvaluator.identify_high_uncertainty_samples(
        y_pred_std, n_top=10
    )
    
    print(f"   - Threshold: {high_unc_info['threshold']:.4f}")
    print(f"   - High-uncertainty samples: {high_unc_info['n_high_uncertainty']} ({high_unc_info['percentage']:.1f}%)")
    print(f"\n   Top 5 samples for experimental validation:")
    for i, (idx, unc) in enumerate(zip(
        high_unc_info['top_n_indices'][:5],
        high_unc_info['top_n_uncertainties'][:5]
    ), 1):
        print(f"      {i}. Sample {idx}: uncertainty={unc:.4f}")
    
    # ========================================================================
    # 6. Generate Visualizations
    # ========================================================================
    print("\n[Step 6/7] Generating visualizations...")
    
    output_dir = Path("results/probabilistic_nn_example")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        plot_paths = create_visualization_summary(
            y, y_pred_mean, y_pred_std,
            train_history=train_info,
            save_dir=output_dir
        )
        
        print(f"   ✓ Generated {len(plot_paths)} plots:")
        for name, path in plot_paths.items():
            print(f"      - {name}: {Path(path).name}")
        
    except ImportError as e:
        print(f"   ⚠️  Visualization skipped: {e}")
    
    # ========================================================================
    # 7. Save Results
    # ========================================================================
    print("\n[Step 7/7] Saving results...")
    
    # Save predictions
    results_df = pd.DataFrame({
        'true_value': y,
        'pred_mean': y_pred_mean,
        'pred_std': y_pred_std,
        'lower_95': y_pred_lower,
        'upper_95': y_pred_upper,
        'abs_error': np.abs(y - y_pred_mean),
        'in_95_ci': (y >= y_pred_lower) & (y <= y_pred_upper)
    })
    
    # Add features for high-uncertainty analysis
    for i, feat in enumerate(features):
        results_df[feat] = X[:, i]
    
    results_path = output_dir / "predictions_with_distribution.csv"
    results_df.to_csv(results_path, index=False)
    print(f"   ✓ Predictions: {results_path}")
    
    # Save high-uncertainty samples
    high_unc_df = pd.DataFrame({
        'sample_index': high_unc_info['top_n_indices'],
        'uncertainty': high_unc_info['top_n_uncertainties'],
        'priority_rank': range(1, len(high_unc_info['top_n_indices']) + 1)
    })
    high_unc_path = output_dir / "high_uncertainty_samples.csv"
    high_unc_df.to_csv(high_unc_path, index=False)
    print(f"   ✓ High-uncertainty samples: {high_unc_path}")
    
    # Save model
    model_path = output_dir / "probabilistic_nn_model.pkl"
    trainer.save(model_path)
    print(f"   ✓ Model saved: {model_path}")
    
    # Save evaluation report
    if summary:
        report_path = output_dir / "uncertainty_evaluation_report.txt"
        UncertaintyEvaluator.save_evaluation_report(
            y, y_pred_mean, y_pred_std,
            output_path=report_path,
            verbose=False
        )
        print(f"   ✓ Evaluation report: {report_path}")
    
    # Save metrics
    if summary:
        metrics_df = pd.DataFrame([summary])
        metrics_path = output_dir / "uncertainty_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        print(f"   ✓ Metrics: {metrics_path}")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*80)
    print("✓ WORKFLOW COMPLETE!")
    print("="*80)
    
    print(f"\n📁 Results saved in: {output_dir}")
    
    print("\n💡 Key Findings:")
    if summary:
        print(f"   - Model Accuracy: RMSE={summary['rmse']:.4f}, R²={summary.get('r2', 'N/A')}")
        print(f"   - Calibration Quality: MAE={summary['calibration_mae']:.4f}")
        
        if summary['calibration_mae'] < 0.1:
            print("     ✓ Well-calibrated (reliable uncertainties)")
        else:
            print("     ⚠ Poor calibration (consider recalibration)")
    
    print(f"   - Uncertainty Range: [{np.min(y_pred_std):.4f}, {np.max(y_pred_std):.4f}]")
    
    # Coverage check
    coverage_95 = np.mean((y >= y_pred_lower) & (y <= y_pred_upper))
    print(f"   - 95% CI Empirical Coverage: {coverage_95*100:.1f}%")
    if abs(coverage_95 - 0.95) < 0.05:
        print("     ✓ Good coverage")
    else:
        print("     ⚠ Coverage mismatch")
    
    print(f"\n📚 Next Steps:")
    print("   1. Review the calibration plot to assess reliability")
    print("   2. Validate the top high-uncertainty samples experimentally")
    print("   3. Retrain with new data to refine uncertainty estimates")
    print("   4. Use prediction distributions for risk-aware decision making")
    
    # LLM interpretation (if available)
    print("\n" + "="*80)
    print("LLM INTERPRETATION")
    print("="*80)
    
    try:
        from aims_agent.agent import Agent
        agent = Agent()
        
        interpretation = interpret_with_probabilistic_llm(
            agent, summary, high_unc_info
        )
        print(interpretation)
        
        # Save interpretation
        interp_path = output_dir / "llm_interpretation.txt"
        interp_path.write_text(interpretation)
        print(f"\n✓ Interpretation saved: {interp_path}")
        
    except Exception as e:
        print(f"⚠️  LLM interpretation unavailable: {e}")
        print("(Set OPENAI_API_KEY or OPENROUTER_API_KEY to enable)")


def interpret_with_probabilistic_llm(agent, summary: dict, high_unc_info: dict) -> str:
    """Generate LLM interpretation of probabilistic model results."""
    
    prompt = f"""You are a materials science ML expert analyzing a Probabilistic Neural Network.

MODEL TYPE: Probabilistic Neural Network (predicts Gaussian distributions)
- Output format: y | x ~ N(mu(x), sigma²(x))
- This provides NATIVE uncertainty quantification

PERFORMANCE METRICS:
- RMSE: {summary['rmse']:.4f}
- MAE: {summary['mae']:.4f}
- R²: {summary.get('r2', 'N/A')}

UNCERTAINTY QUALITY METRICS:
- Calibration Error (MAE): {summary['calibration_mae']:.4f} (target < 0.1)
- Miscalibration Area: {summary['miscalibration_area']:.4f}
- Sharpness: {summary.get('sharpness', 'N/A')} (average uncertainty)
- NLL: {summary['nll']:.4f} (lower is better)
- CRPS: {summary['crps']:.4f} (lower is better)

HIGH-UNCERTAINTY SAMPLES:
- Threshold: {high_unc_info['threshold']:.4f}
- Percentage above threshold: {high_unc_info['percentage']:.1f}%
- Top uncertain sample indices: {high_unc_info['top_n_indices'][:5]}

TASK: Provide a comprehensive analysis in 4 sections:

1. **Prediction Quality** (2-3 sentences):
   - How good are the point predictions (RMSE, R²)?
   - Is this acceptable for materials property prediction?

2. **Uncertainty Calibration** (3-4 sentences):
   - Are the uncertainties well-calibrated?
   - Can we trust the 95% prediction intervals?
   - Is the model overconfident or underconfident?

3. **High-Uncertainty Analysis** (2-3 sentences):
   - Why might these samples have high uncertainty?
   - What material properties or conditions cause high uncertainty?
   - Should we validate them experimentally?

4. **Recommendations** (3-4 concrete bullet points):
   - How to use this model safely
   - Which predictions to trust vs validate
   - How to improve the model further
   - Active learning strategy for next experiments

Be specific, scientific, and actionable. Reference the actual metric values."""
    
    return agent.call_llm(prompt)


if __name__ == "__main__":
    main()
