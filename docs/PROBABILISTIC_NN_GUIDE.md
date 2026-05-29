# Probabilistic NN Guide

The probabilistic neural-network workflow predicts both a target value and an
input-dependent uncertainty estimate. Use it when prediction intervals, active
learning, or uncertainty-aware model comparison matter.

## Quick Start

```bash
python examples/probabilistic_nn_example.py
```

## Main APIs

- `aims_agent.probabilistic_models.ProbabilisticNNWrapper`
- `aims_agent.probabilistic_trainer.ProbabilisticTrainer`
- `aims_agent.uncertainty_evaluator.UncertaintyEvaluator`
- `aims_agent.probabilistic_visualizer.create_visualization_summary`

## Recommended Workflow

1. Split the dataset using the same protocol as the baseline model workflow.
2. Train the probabilistic model with `ProbabilisticTrainer`.
3. Predict distribution parameters for the validation or test set.
4. Evaluate accuracy and UQ quality with `UncertaintyEvaluator`.
5. Compare calibration, sharpness, NLL, CRPS, RMSE, and coverage before using
   the model for conclusions.

## When To Prefer It

- The dataset is large enough to train a neural model without unstable results.
- The target is expected to have heteroscedastic noise.
- You need prediction intervals rather than only point estimates.
- You want to select follow-up samples by uncertainty.

For small tabular datasets, compare this method against tree ensembles with
repeated CV and UQ calibration before treating its conclusions as strong.
