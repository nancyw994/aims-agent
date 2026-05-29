# Probabilistic NN Implementation

The probabilistic NN implementation lives in:

- `aims_agent/probabilistic_models/`
- `aims_agent/probabilistic_trainer.py`
- `aims_agent/probabilistic_visualizer.py`
- `aims_agent/uncertainty_evaluator.py`

## Model Contract

The model predicts a Gaussian distribution:

```text
y | x ~ Normal(mu(x), sigma(x))
```

The wrapper returns distribution outputs such as:

- `mu`: predictive mean
- `std`: predictive standard deviation
- interval bounds when supported by the caller

## Training Objective

The training objective uses Gaussian negative log likelihood so the model learns
both prediction accuracy and scale estimates. Downstream evaluation should still
check calibration because low point error does not guarantee reliable
uncertainty.

## Evaluation Contract

UQ metrics are computed through `UncertaintyEvaluator`, which wraps
`uncertainty-toolbox` when it is available. Reports should include:

- RMSE and R2 for point-prediction quality
- calibration error and miscalibration area
- sharpness
- NLL and CRPS when available
- empirical coverage at standard confidence levels

## Integration Notes

- Keep probabilistic model logic inside `aims_agent/`.
- Keep runnable demos inside `examples/`.
- Keep generated plots and JSON outputs under `results/`.
- Do not use probabilistic NN results alone as strong evidence on small datasets;
  compare against repeated-CV baselines and robustness strategies.
