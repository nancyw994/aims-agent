# Scripts

This directory contains command-line workflows that call into `aims_agent/`.

## Main Entry Points

- `analyze_matsci_strategy.py`: profile a dataset, ask the LLM or deterministic
  fallback for modeling strategy, evaluate feature-selection methods, compare
  models, run UQ evaluation, and write reports.
- `run_materials_robustness.py`: run robustness experiments across missing-value
  handling, outlier handling, repeated CV, UQ calibration, and feature-stability
  checks.
- `llm_materials_strong_loop.py`: run the LLM-guided evidence loop. Each
  iteration writes JSON evidence, feeds uncertainty notes back to the LLM, and
  asks for the next experiment plan.

## Compatibility Entry Points

- `run_spall_robustness.py` and `llm_spall_strong_loop.py` keep older commands
  working for the original spall-strength dataset. Prefer the generic
  `materials` names for new usage.

## Typical Commands

```bash
python scripts/run_materials_robustness.py \
  --data "data/real_data/Spall_Strength_Database_AliShargh(Processed).csv" \
  --target "Spall Strength" \
  --repeats 3
```

```bash
python scripts/llm_materials_strong_loop.py \
  --data "data/real_data/Spall_Strength_Database_AliShargh(Processed).csv" \
  --target "Spall Strength" \
  --max-iterations 3
```
