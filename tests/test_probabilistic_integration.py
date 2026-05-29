"""
Integration tests for ProbabilisticNN in the model selection system.

Tests the complete workflow:
1. Model selection recommends ProbabilisticNN
2. Model training works correctly
3. Distribution prediction produces valid results
4. Integration with uncertainty evaluation
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from aims_agent.model_selector import (
    select_uq_aware_model,
    MODEL_UQ_CAPABILITY,
    MODEL_IMPORT_MAP,
    REGRESSION_MODELS,
)
from aims_agent.model_trainer import ModelTrainer
from aims_agent.distribution_predictor import predict_distribution, get_uq_capability
from aims_agent.probabilistic_models import ProbabilisticNNWrapper


def test_pnn_in_model_registry():
    """Test that ProbabilisticNN is registered in the model system."""

    print("\n=== Test 1: PNN in Model Registry ===")

    # Check MODEL_IMPORT_MAP
    assert "ProbabilisticNN" in MODEL_IMPORT_MAP, "ProbabilisticNN not in MODEL_IMPORT_MAP"
    module_path, class_name = MODEL_IMPORT_MAP["ProbabilisticNN"]
    assert module_path == "aims_agent.probabilistic_models.pnn"
    assert class_name == "ProbabilisticNNWrapper"
    print("✓ ProbabilisticNN in MODEL_IMPORT_MAP")

    # Check UQ capability metadata
    assert "ProbabilisticNN" in MODEL_UQ_CAPABILITY, "ProbabilisticNN not in MODEL_UQ_CAPABILITY"
    meta = MODEL_UQ_CAPABILITY["ProbabilisticNN"]
    assert meta["uq_capability"] == "native"
    assert meta["uq_quality"] == 0.9
    assert meta["heteroscedastic"] == True
    print("✓ ProbabilisticNN has UQ metadata")

    # Check in regression models list
    assert "ProbabilisticNN" in REGRESSION_MODELS, "ProbabilisticNN not in REGRESSION_MODELS"
    print("✓ ProbabilisticNN in REGRESSION_MODELS")

    print("✅ Test 1 passed\n")


def test_smart_model_selection():
    """Test that select_uq_aware_model recommends ProbabilisticNN appropriately."""

    print("\n=== Test 2: Smart Model Selection ===")

    # Small dataset + screening → should prefer GP
    suggestion_small = select_uq_aware_model(
        n_samples=500,
        n_features=5,
        use_case="screening"
    )
    print(f"500 samples + screening → {suggestion_small.model_name} (UQ={suggestion_small.uq_quality})")

    # Medium dataset + screening/active_learning → should recommend PNN
    suggestion_medium = select_uq_aware_model(
        n_samples=2000,
        n_features=5,
        use_case="screening"
    )
    print(f"2000 samples + screening → {suggestion_medium.model_name} (UQ={suggestion_medium.uq_quality})")
    assert suggestion_medium.model_name == "ProbabilisticNN", f"Expected ProbabilisticNN, got {suggestion_medium.model_name}"
    assert suggestion_medium.uq_capability == "native"
    assert suggestion_medium.heteroscedastic == True
    print("✓ ProbabilisticNN recommended for medium dataset + high UQ need")

    # Active learning case
    suggestion_al = select_uq_aware_model(
        n_samples=3000,
        n_features=10,
        use_case="active_learning"
    )
    print(f"3000 samples + active_learning → {suggestion_al.model_name} (UQ={suggestion_al.uq_quality})")
    assert suggestion_al.model_name == "ProbabilisticNN"
    print("✓ ProbabilisticNN recommended for active learning")

    print("✅ Test 2 passed\n")


def test_pnn_training_and_prediction():
    """Test that ProbabilisticNN can be trained and make predictions."""

    print("\n=== Test 3: PNN Training and Prediction ===")

    # Create synthetic data
    np.random.seed(42)
    X = np.random.randn(200, 5)
    y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.randn(200) * 0.5

    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
    df['target'] = y

    # Train using ModelTrainer
    trainer = ModelTrainer(
        model_class=ProbabilisticNNWrapper,
        task_type="regression",
        use_hyperparameter_tuning=False  # Skip for speed
    )

    trainer.prepare_data(
        df,
        features=[f'feature_{i}' for i in range(5)],
        target='target',
        test_size=0.3
    )

    print("Training ProbabilisticNN...")
    trainer.train()
    print("✓ Training completed")

    # Test standard prediction
    y_true, y_pred = trainer.predict()
    assert len(y_pred) == len(y_true)
    print(f"✓ Standard prediction: {len(y_pred)} samples")

    # Test uncertainty prediction
    result = trainer.predict_with_uncertainty()
    assert 'y_pred' in result
    assert 'y_std' in result
    assert 'lower_95' in result
    assert 'upper_95' in result
    print(f"✓ Uncertainty prediction: mean_std={np.mean(result['y_std']):.4f}")

    # Verify uncertainty is non-zero
    assert np.mean(result['y_std']) > 0, "Uncertainty should be non-zero"
    print("✓ Non-zero uncertainties")

    print("✅ Test 3 passed\n")


def test_distribution_predictor_integration():
    """Test that distribution_predictor can handle ProbabilisticNN."""

    print("\n=== Test 4: Distribution Predictor Integration ===")

    # Create and train a simple PNN
    np.random.seed(42)
    X_train = np.random.randn(100, 3)
    y_train = np.random.randn(100)
    X_test = np.random.randn(20, 3)

    model = ProbabilisticNNWrapper(
        hidden_dim=32,
        epochs=10,
        verbose=False
    )
    model.fit(X_train, y_train)
    print("✓ Model trained")

    # Test capability detection
    cap = get_uq_capability(model)
    print(f"Detected capability: {cap['model_type']}, can_predict_distribution={cap['can_predict_distribution']}")
    assert cap['can_predict_distribution'] == True
    print("✓ Capability detected correctly")

    # Test distribution prediction
    dist = predict_distribution(model, X_test)
    assert dist is not None, "Distribution prediction returned None"
    assert 'mu' in dist
    assert 'std' in dist
    assert len(dist['mu']) == len(X_test)
    print(f"✓ Distribution predicted: mu={dist['mu'][:3]}, std={dist['std'][:3]}")

    print("✅ Test 4 passed\n")


def test_uq_evaluation_integration():
    """Test that uncertainty evaluation works with PNN predictions."""

    print("\n=== Test 5: UQ Evaluation Integration ===")

    try:
        from aims_agent.uncertainty_evaluator import UncertaintyEvaluator

        # Create synthetic data with known uncertainty
        np.random.seed(42)
        X = np.random.randn(300, 5)
        y = 2 * X[:, 0] + np.random.randn(300) * 0.5

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Train PNN
        model = ProbabilisticNNWrapper(hidden_dim=64, epochs=20, verbose=False)
        model.fit(X_train, y_train)

        # Predict distribution
        dist = model.predict_distribution(X_test)

        # Evaluate UQ quality
        summary, full_metrics = UncertaintyEvaluator.evaluate_all(
            y_test, dist['mu'], dist['std'], verbose=False
        )

        print(f"Calibration Error: {summary['calibration_mae']:.4f}")
        print(f"NLL: {summary.get('nll', 'N/A')}")
        print(f"R²: {summary.get('r2', 'N/A')}")

        # Basic sanity checks
        assert summary['calibration_mae'] is not None
        assert summary['calibration_mae'] >= 0
        print("✓ UQ evaluation completed successfully")

        print("✅ Test 5 passed\n")

    except ImportError:
        print("⚠️  uncertainty-toolbox not available, skipping Test 5\n")


def run_all_tests():
    """Run all integration tests."""

    print("\n" + "="*80)
    print("PROBABILISTIC NN INTEGRATION TESTS")
    print("="*80)

    try:
        test_pnn_in_model_registry()
        test_smart_model_selection()
        test_pnn_training_and_prediction()
        test_distribution_predictor_integration()
        test_uq_evaluation_integration()

        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED!")
        print("="*80)
        print("\nProbabilisticNN is fully integrated into the model selection system.")
        print("\nKey features:")
        print("  ✓ Registered in MODEL_IMPORT_MAP")
        print("  ✓ UQ metadata in MODEL_UQ_CAPABILITY")
        print("  ✓ Automatically recommended by select_uq_aware_model()")
        print("  ✓ Works with ModelTrainer")
        print("  ✓ Supports predict_with_uncertainty()")
        print("  ✓ Compatible with distribution_predictor")
        print("  ✓ Integrates with uncertainty evaluation")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
