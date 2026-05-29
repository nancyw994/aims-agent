"""
Example: Uncertainty Quantification with uncertainty-toolbox

This script demonstrates how to use uncertainty-toolbox for professional-grade
uncertainty quantification and active learning in materials science predictions.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from aims_agent.model_trainer import ModelTrainer
from aims_agent.uncertainty_analysis import (
    compute_uncertainty_metrics,
    suggest_active_learning_samples,
    plot_uncertainty_analysis,
    interpret_uncertainty_with_llm,
    recalibrate_predictions,
)


def create_synthetic_materials_data():
    """Create synthetic materials data for demonstration."""
    from aims_agent.data_interface import DatasetBundle, DatasetSchema
    
    np.random.seed(42)
    n_samples = 200
    
    # Simulate materials properties
    df = pd.DataFrame({
        'composition_A': np.random.uniform(0, 1, n_samples),
        'composition_B': np.random.uniform(0, 1, n_samples),
        'temperature': np.random.uniform(300, 1500, n_samples),
        'pressure': np.random.uniform(1, 100, n_samples),
    })
    
    # Target: synthetic property with nonlinear relationship + noise
    df['hardness'] = (
        10 * df['composition_A'] ** 2 
        + 5 * df['composition_B'] 
        + 0.01 * df['temperature']
        - 0.02 * df['pressure']
        + np.random.normal(0, 1, n_samples)
    )
    
    schema = DatasetSchema(
        features=[col for col in df.columns if col != 'hardness'],
        target='hardness',
        units={'hardness': 'GPa', 'temperature': 'K', 'pressure': 'GPa'},
        source='synthetic',
        description='Synthetic materials hardness data'
    )
    
    return DatasetBundle(df=df, schema=schema)


def main():
    """Run uncertainty quantification example with uncertainty-toolbox."""
    
    print("\n" + "="*80)
    print("Uncertainty Quantification with uncertainty-toolbox")
    print("="*80)
    
    # 1. Load data
    print("\n[1/8] Loading data...")
    bundle = create_synthetic_materials_data()
    print(f"   ✓ Loaded {len(bundle.df)} samples")
    
    # 2. Train model (use ensemble for uncertainty)
    print("\n[2/8] Training RandomForest model...")
    from sklearn.ensemble import RandomForestRegressor
    
    trainer = ModelTrainer(
        RandomForestRegressor,
        task_type="regression",
        use_hyperparameter_tuning=False,  # For speed
    )
    
    features = [col for col in bundle.df.columns if col != 'hardness']
    trainer.prepare_data(bundle.df, features, 'hardness')
    trainer.train()
    print("   ✓ Training complete")
    
    # 3. Predict with uncertainty
    print("\n[3/8] Computing predictions with uncertainty...")
    result = trainer.predict_with_uncertainty()
    
    y_true = result['y_true']
    y_pred = result['y_pred']
    y_std = result['y_std']
    
    print(f"   ✓ Predictions shape: {y_pred.shape}")
    print(f"   ✓ Mean uncertainty: {np.mean(y_std):.4f}")
    print(f"   ✓ Max uncertainty: {np.max(y_std):.4f}")
    
    # 4. Compute comprehensive metrics using uncertainty-toolbox
    print("\n[4/8] Computing uncertainty metrics with uncertainty-toolbox...")
    try:
        metrics = compute_uncertainty_metrics(y_pred, y_std, y_true)
        
        def format_metric(value):
            """Format metric value safely."""
            if value is None or value == 'N/A':
                return 'N/A'
            try:
                return f"{float(value):.4f}"
            except (TypeError, ValueError):
                return str(value)
        
        # Extract nested metrics
        accuracy = metrics.get('accuracy', {})
        avg_cal = metrics.get('avg_calibration', {})
        sharpness_dict = metrics.get('sharpness', {})
        scoring = metrics.get('scoring_rule', {})
        
        print("\n   📊 Accuracy Metrics:")
        print(f"      - MAE: {format_metric(accuracy.get('mae'))}")
        print(f"      - RMSE: {format_metric(accuracy.get('rmse'))}")
        print(f"      - R²: {format_metric(accuracy.get('rsq'))}")
        
        print("\n   📐 Calibration Metrics:")
        print(f"      - Mean Absolute Calibration Error: {format_metric(avg_cal.get('ma_cal'))}")
        print(f"      - Root Mean Squared Calibration Error: {format_metric(avg_cal.get('rms_cal'))}")
        print(f"      - Miscalibration Area: {format_metric(avg_cal.get('miscal_area'))}")
        
        print("\n   🎯 Sharpness:")
        print(f"      - Average: {format_metric(sharpness_dict.get('sharpness'))}")
        
        print("\n   📈 Scoring Rules (lower is better):")
        print(f"      - NLL: {format_metric(scoring.get('nll'))}")
        print(f"      - CRPS: {format_metric(scoring.get('crps'))}")
        
    except ImportError as e:
        print(f"   ⚠️  {e}")
        print("   Install: pip install uncertainty-toolbox")
        return
    
    # 5. Generate visualizations using uncertainty-toolbox
    print("\n[5/8] Generating uncertainty-toolbox visualizations...")
    output_dir = Path("results/uncertainty_toolbox_example")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_paths = plot_uncertainty_analysis(
        y_pred, y_std, y_true,
        save_dir=output_dir,
        prefix="materials"
    )
    
    print("   ✓ Generated plots:")
    for name, path in plot_paths.items():
        print(f"      - {name}: {path}")
    
    # 6. Active learning suggestions
    print("\n[6/8] Generating active learning suggestions...")
    
    al_uncertainty = suggest_active_learning_samples(
        y_std,
        X=trainer.X_test.values,
        n_samples=10,
        strategy="uncertainty"
    )
    
    print(f"\n   Strategy: {al_uncertainty['strategy']}")
    print(f"   Top 5 samples for experimental validation:")
    for i, (idx, unc) in enumerate(zip(
        al_uncertainty['suggested_indices'][:5],
        al_uncertainty['suggested_uncertainties'][:5]
    )):
        print(f"      #{i+1}: Sample {idx}, uncertainty={unc:.4f}")
    
    # 7. Recalibration (optional)
    print("\n[7/8] Recalibrating uncertainties...")
    try:
        y_std_recal = recalibrate_predictions(y_pred, y_std, y_true)
        
        improvement = (np.mean(y_std) - np.mean(y_std_recal)) / np.mean(y_std) * 100
        print(f"   ✓ Original mean uncertainty: {np.mean(y_std):.4f}")
        print(f"   ✓ Recalibrated mean uncertainty: {np.mean(y_std_recal):.4f}")
        print(f"   ✓ Sharpness improvement: {improvement:.1f}%")
        
        # Compute metrics after recalibration
        metrics_recal = compute_uncertainty_metrics(y_pred, y_std_recal, y_true)
        cal_error = metrics_recal.get('avg_calibration', {}).get('ma_cal', 'N/A')
        if isinstance(cal_error, (int, float)):
            print(f"   ✓ Calibration error after recalibration: {cal_error:.4f}")
        else:
            print(f"   ✓ Calibration error after recalibration: {cal_error}")
        
    except Exception as e:
        print(f"   ⚠️  Recalibration failed: {e}")
    
    # 8. LLM interpretation
    print("\n[8/8] Generating LLM interpretation...")
    try:
        from aims_agent.agent import Agent
        
        agent = Agent()
        interpretation = interpret_uncertainty_with_llm(
            agent=agent,
            metrics=metrics,
            active_learning_suggestions=al_uncertainty,
            task_type="regression"
        )
        
        print("\n" + "="*80)
        print("LLM INTERPRETATION")
        print("="*80)
        print(interpretation)
        
        # Save interpretation
        interp_path = output_dir / "llm_interpretation.txt"
        interp_path.write_text(interpretation)
        print(f"\n✓ Saved interpretation: {interp_path}")
        
    except Exception as e:
        print(f"   ⚠️  LLM interpretation failed: {e}")
        print("   (Set OPENAI_API_KEY or OPENROUTER_API_KEY to enable)")
    
    # 9. Save results
    print("\n[9/9] Saving results...")
    
    # Save predictions with uncertainty
    results_df = pd.DataFrame({
        'true_value': y_true,
        'predicted_value': y_pred,
        'uncertainty_std': y_std,
        'absolute_error': np.abs(y_true - y_pred),
    })
    
    results_path = output_dir / "predictions_with_uncertainty.csv"
    results_df.to_csv(results_path, index=False)
    print(f"   ✓ Saved predictions: {results_path}")
    
    # Save active learning suggestions
    al_df = pd.DataFrame({
        'sample_index': al_uncertainty['suggested_indices'],
        'uncertainty': al_uncertainty['suggested_uncertainties'],
        'priority_rank': range(1, len(al_uncertainty['suggested_indices']) + 1),
    })
    
    al_path = output_dir / "active_learning_suggestions.csv"
    al_df.to_csv(al_path, index=False)
    print(f"   ✓ Saved active learning: {al_path}")
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_path = output_dir / "uncertainty_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"   ✓ Saved metrics: {metrics_path}")
    
    print("\n" + "="*80)
    print("✓ Analysis complete!")
    print(f"📁 Results saved in: {output_dir}")
    print("="*80)
    
    print("\n💡 Key Takeaways:")
    print(f"   - Mean uncertainty: {np.mean(y_std):.4f}")
    cal_err = metrics.get('avg_calibration', {}).get('ma_cal', 'N/A')
    if isinstance(cal_err, (int, float)):
        print(f"   - Calibration error: {cal_err:.4f}")
    else:
        print(f"   - Calibration error: {cal_err}")
    print(f"   - Top uncertain sample: #{al_uncertainty['suggested_indices'][0]}")
    print(f"   - Visualizations: {len(plot_paths)} plots generated")
    
    return {
        'metrics': metrics,
        'active_learning': al_uncertainty,
        'output_dir': output_dir,
    }


if __name__ == "__main__":
    # Run the example
    results = main()
    
    print("\n📚 Next Steps:")
    print("   1. Review the generated plots in results/uncertainty_toolbox_example/")
    print("   2. Check the uncertainty metrics CSV")
    print("   3. Validate the high-uncertainty samples experimentally")
    print("   4. Use uncertainty-toolbox's advanced features for deeper analysis")
