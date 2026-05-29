"""
Example: Uncertainty-Aware ML Workflow with AIMS Agent

This script demonstrates the complete uncertainty-aware workflow:
1. Data analysis with uncertainty-aware strategy
2. Model training with ensemble models
3. Prediction with uncertainty quantification
4. LLM interpretation considering uncertainty
5. Active learning sample selection
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from aims_agent.agent import Agent
from aims_agent.data_analyzer import analyze_and_formulate_strategy
from aims_agent.data_interface import DatasetBundle, DatasetSchema
from aims_agent.model_trainer import ModelTrainer
from aims_agent.results_analyzer import compute_metrics, interpret_with_llm
from aims_agent.uncertainty_analysis import (
    compute_uncertainty_metrics,
    suggest_active_learning_samples,
    plot_uncertainty_analysis,
)


def create_example_data():
    """Create synthetic materials data for demonstration."""
    np.random.seed(42)
    n_samples = 150
    
    df = pd.DataFrame({
        'composition_A': np.random.uniform(0, 1, n_samples),
        'composition_B': np.random.uniform(0, 1, n_samples),
        'temperature': np.random.uniform(300, 1500, n_samples),
        'pressure': np.random.uniform(1, 100, n_samples),
        'grain_size': np.random.lognormal(2, 0.5, n_samples),
    })
    
    # Target with some nonlinearity and noise
    df['hardness'] = (
        15 * df['composition_A'] ** 1.5
        + 8 * df['composition_B']
        + 0.015 * df['temperature']
        - 0.03 * df['pressure']
        + 2 * np.log(df['grain_size'])
        + np.random.normal(0, 1.5, n_samples)
    )
    
    schema = DatasetSchema(
        features=['composition_A', 'composition_B', 'temperature', 'pressure', 'grain_size'],
        target='hardness',
        units={'hardness': 'GPa', 'temperature': 'K', 'pressure': 'GPa', 'grain_size': 'nm'},
        source='synthetic',
        description='Synthetic materials hardness data for uncertainty-aware demo'
    )
    
    return DatasetBundle(df=df, schema=schema)


def main():
    """Run the complete uncertainty-aware workflow."""
    
    print("\n" + "="*80)
    print("Uncertainty-Aware ML Workflow with AIMS Agent")
    print("="*80)
    
    # Step 1: Create/Load Data
    print("\n[Step 1/6] Loading data...")
    bundle = create_example_data()
    print(f"   ✓ Loaded {len(bundle.df)} samples with {len(bundle.schema.features)} features")
    
    # Step 2: Analyze Data with Uncertainty-Aware Strategy
    print("\n[Step 2/6] Analyzing data and formulating uncertainty-aware strategy...")
    try:
        agent = Agent()
        profile, strategy = analyze_and_formulate_strategy(
            df=bundle.df,
            target=bundle.schema.target,
            agent=agent,
            use_llm=True,
            output_dir="results/uncertainty_aware_demo"
        )
        print("   ✓ Strategy formulated")
        print(f"   - Recommended models: {', '.join(strategy.recommended_models[:3])}")
        print(f"   - Uncertainty strategy items: {len(strategy.uncertainty_strategy)}")
        print(f"   - Active learning plan items: {len(strategy.active_learning_plan)}")
    except Exception as e:
        print(f"   ⚠️  LLM unavailable, using heuristic strategy: {e}")
        agent = None
    
    # Step 3: Train Model (use ensemble for uncertainty)
    print("\n[Step 3/6] Training ensemble model for uncertainty estimation...")
    trainer = ModelTrainer(
        RandomForestRegressor,
        task_type="regression",
        use_hyperparameter_tuning=False  # For speed
    )
    
    features = bundle.schema.features
    target = bundle.schema.target
    trainer.prepare_data(bundle.df, features, target)
    trainer.train()
    print("   ✓ Model trained successfully")
    
    # Step 4: Predict with Uncertainty
    print("\n[Step 4/6] Computing predictions with uncertainty...")
    result = trainer.predict_with_uncertainty()
    
    y_true = result['y_true']
    y_pred = result['y_pred']
    y_std = result['y_std']
    
    print(f"   ✓ Predictions: {len(y_pred)} samples")
    print(f"   - Mean uncertainty: {np.mean(y_std):.4f}")
    print(f"   - Max uncertainty: {np.max(y_std):.4f}")
    print(f"   - Min uncertainty: {np.min(y_std):.4f}")
    
    # Step 5: Compute Metrics (both accuracy and uncertainty)
    print("\n[Step 5/6] Computing comprehensive metrics...")
    
    # Standard metrics
    metrics = compute_metrics(y_true, y_pred, task_type="regression")
    print(f"   📊 Accuracy: RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}")
    
    # Uncertainty metrics
    try:
        uq_metrics = compute_uncertainty_metrics(y_pred, y_std, y_true)
        
        accuracy = uq_metrics.get('accuracy', {})
        calibration = uq_metrics.get('avg_calibration', {})
        scoring = uq_metrics.get('scoring_rule', {})
        
        print(f"   📐 Calibration: MAE={calibration.get('ma_cal', 'N/A'):.4f}")
        print(f"   📈 Scoring: NLL={scoring.get('nll', 'N/A'):.4f}, CRPS={scoring.get('crps', 'N/A'):.4f}")
    except Exception as e:
        print(f"   ⚠️  Uncertainty metrics unavailable: {e}")
        uq_metrics = None
    
    # Step 6: LLM Interpretation (Uncertainty-Aware)
    print("\n[Step 6/6] Generating uncertainty-aware interpretation...")
    
    if agent:
        try:
            interpretation = interpret_with_llm(
                agent=agent,
                metrics=metrics,
                model_name="RandomForestRegressor",
                task_type="regression",
                uncertainty_metrics=uq_metrics  # New: pass uncertainty metrics
            )
            
            print("\n" + "="*80)
            print("UNCERTAINTY-AWARE INTERPRETATION")
            print("="*80)
            print(interpretation)
            print("="*80)
            
            # Save interpretation
            output_dir = Path("results/uncertainty_aware_demo")
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "interpretation.txt").write_text(interpretation)
            print(f"\n✓ Saved: {output_dir / 'interpretation.txt'}")
            
        except Exception as e:
            print(f"   ⚠️  LLM interpretation failed: {e}")
    else:
        print("   ⚠️  LLM not available (set OPENAI_API_KEY or OPENROUTER_API_KEY)")
    
    # Bonus: Active Learning Suggestions
    print("\n[Bonus] Active learning suggestions...")
    
    al_uncertainty = suggest_active_learning_samples(
        y_std,
        X=trainer.X_test.values,
        n_samples=5,
        strategy="uncertainty"
    )
    
    print(f"\n   🎯 Top 5 samples for experimental validation:")
    for i, (idx, unc) in enumerate(zip(
        al_uncertainty['suggested_indices'][:5],
        al_uncertainty['suggested_uncertainties'][:5]
    )):
        print(f"      #{i+1}: Sample {idx}, uncertainty={unc:.4f}")
    
    # Visualizations
    print("\n[Visualization] Generating uncertainty plots...")
    try:
        plot_paths = plot_uncertainty_analysis(
            y_pred, y_std, y_true,
            save_dir="results/uncertainty_aware_demo",
            prefix="demo"
        )
        print(f"   ✓ Generated {len(plot_paths)} plots")
        for name, path in plot_paths.items():
            print(f"      - {name}: {path}")
    except Exception as e:
        print(f"   ⚠️  Visualization failed: {e}")
    
    print("\n" + "="*80)
    print("✓ Uncertainty-Aware Workflow Complete!")
    print("="*80)
    
    print("\n💡 Key Takeaways:")
    print(f"   - Model: RandomForestRegressor with {len(y_pred)} test samples")
    print(f"   - Accuracy: R²={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}")
    if uq_metrics:
        cal_err = uq_metrics.get('avg_calibration', {}).get('ma_cal', 'N/A')
        if isinstance(cal_err, (int, float)):
            print(f"   - Calibration Error: {cal_err:.4f} (well-calibrated if < 0.1)")
        print(f"   - Mean Uncertainty: {np.mean(y_std):.4f}")
        print(f"   - Top uncertain sample needs validation: #{al_uncertainty['suggested_indices'][0]}")
    
    print("\n📁 Results saved in: results/uncertainty_aware_demo/")
    print("   - strategy_report.html: Full strategy with UQ and AL sections")
    print("   - interpretation.txt: Uncertainty-aware LLM interpretation")
    print("   - demo_*.png: Uncertainty visualization plots")
    
    print("\n📚 Next Steps:")
    print("   1. Review the strategy report for uncertainty strategy")
    print("   2. Validate high-uncertainty samples experimentally")
    print("   3. Retrain model with new data for next AL iteration")
    print("   4. Monitor calibration metrics to ensure reliability")


if __name__ == "__main__":
    main()
