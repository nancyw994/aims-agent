# aims-agent

`aims-agent` is an LLM-assisted machine-learning workflow for materials-science
tabular data. It helps a researcher move from a local dataset and a research
question to a reproducible modeling report with data profiling, preprocessing,
feature selection, model selection, validation, uncertainty quantification, and
evidence-strength analysis.

The project is not limited to one dataset or one property. The original
motivation includes predicting metallic-alloy spall strength under high-strain-
rate shock loading, but the workflow is written for general materials-property
prediction tasks.

## What This Project Does

The agent can:

- Load synthetic, CSV, Excel, and Materials Project style tabular data.
- Profile deterministic dataset properties: schema, missingness, outliers,
  skewness, correlations, target relationships, and data-quality risks.
- Ask an LLM for modeling strategy when API credentials are available, or fall
  back to deterministic recommendations when running offline.
- Recommend and evaluate feature-selection or dimensionality-reduction methods.
- Select models from an executable catalog, with optional dependency checks and
  model code generation when a selected model is not directly available.
- Train and evaluate regression or classification models.
- Run uncertainty evaluation with `uncertainty-toolbox` metrics when available.
- Run robustness experiments across missing-value strategies, outlier handling,
  repeated cross-validation, UQ calibration, and feature-stability checks.
- Write machine-readable JSON/CSV outputs and human-readable HTML reports.
- Feed uncertainty notes and validation evidence back to the LLM so it can
  propose the next experiment until the conclusion becomes stronger or the loop
  reaches its iteration limit.

## Project Status

This is an active research-agent prototype. The strongest workflows are the
tabular materials-property pipelines under `aims_agent/` and `scripts/`.
Generated experiment outputs are written to `results/` and should be treated as
local artifacts, not source code.

## Repository Layout

```text
aims_agent/          Reusable Python package and agent logic
scripts/             CLI workflows for analysis, robustness, and LLM loops
examples/            Small runnable examples and config files
data/                Example and real materials datasets
docs/                Project organization and focused technical notes
tests/               Unit and integration tests
generated_code/      Generated estimator/component modules
results/             Local run outputs, ignored by Git
logs/                Local runtime logs, ignored by Git
```

See `docs/REPO_STRUCTURE.md` for organization rules.

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

Optional dependency groups:

```bash
python -m pip install -r requirements-matsci.txt
python -m pip install -r requirements-dl.txt
```

If Matplotlib cannot write to the default user cache on your machine, set a
local writable cache directory before running plot-heavy workflows:

```bash
export MPLCONFIGDIR=/tmp/matplotlib
```

## LLM Configuration

The project can run with or without an LLM.

For LLM-enabled runs, copy `.env.example` to `.env` and set one provider:

```bash
cp .env.example .env
```

Supported environment variables:

```text
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini

OPENROUTER_API_KEY=
OPENROUTER_MODEL=
```

If no provider key is configured, use `--no-llm` where supported. The workflow
will use deterministic fallback logic.

## Quick Start

Run the synthetic full pipeline:

```bash
python -m aims_agent.cli \
  --motivation "Predict hardness from composition and processing"
```

Run offline without LLM calls:

```bash
python -m aims_agent.cli \
  --motivation "Predict hardness from composition and processing" \
  --no-llm
```

List supported model names:

```bash
python -m aims_agent.cli --list-models
```

## Full Pipeline On The Spall Dataset

The included processed spall-strength dataset is:

```text
data/real_data/Spall_Strength_Database_AliShargh(Processed).csv
```

The target column is:

```text
Spall (Gpa)
```

Run the full LLM-assisted pipeline:

```bash
python -m aims_agent.cli \
  --data "data/real_data/Spall_Strength_Database_AliShargh(Processed).csv" \
  --target "Spall (Gpa)" \
  --motivation "Predict the spall strength of metallic alloys under high-strain-rate shock loading using machine learning" \
  --task-type regression \
  --randomized-search
```

Run the same pipeline offline:

```bash
python -m aims_agent.cli \
  --data "data/real_data/Spall_Strength_Database_AliShargh(Processed).csv" \
  --target "Spall (Gpa)" \
  --motivation "Predict the spall strength of metallic alloys under high-strain-rate shock loading using machine learning" \
  --task-type regression \
  --randomized-search \
  --no-llm
```

Outputs are written under `results/run_*/`, including metrics, plots, and the
HTML strategy/report artifact when the strategy workflow is used.

## Strategy, Feature Selection, And Model Selection

For real tabular data, the recommended modeling sequence is:

1. Load the dataset through the data interface.
2. Build a deterministic dataset profile.
3. Ask the LLM to recommend preprocessing, feature selection, and candidate
   models using the dataset profile, research motivation, target column, and
   background knowledge.
4. Validate recommended feature-selection methods by cross-validation.
5. Train candidate models under comparable validation settings.
6. Evaluate point-prediction metrics and UQ metrics.
7. Choose the final model from validation evidence, not from the LLM ranking
   alone.
8. Write evidence files for audit and downstream LLM reflection.

The LLM is a recommender and planner. It does not get final authority over model
choice unless validation and UQ evidence support that choice.

Run dataset strategy analysis:

```bash
python scripts/analyze_matsci_strategy.py \
  --data "data/real_data/Spall_Strength_Database_AliShargh(Processed).csv" \
  --target "Spall (Gpa)" \
  --output-dir results
```

Offline fallback:

```bash
python scripts/analyze_matsci_strategy.py \
  --no-llm \
  --data "data/real_data/Spall_Strength_Database_AliShargh(Processed).csv" \
  --target "Spall (Gpa)" \
  --output-dir results
```

Typical outputs:

```text
results/run_*/profile.json
results/run_*/strategy.json
results/run_*/strategy_report.html
results/run_*/feature_selection_evaluation.csv
results/run_*/uncertainty_model_selection.csv
results/run_*/uncertainty_model_selection_full.json
```

## Robustness And UQ Evidence

For stronger scientific conclusions, do not rely only on one train/test split or
one preprocessing choice. Use the robustness workflow to compare missing-value
strategies, outlier strategies, repeated CV performance, UQ calibration, and
feature-importance stability.

```bash
python scripts/run_materials_robustness.py \
  --data "data/real_data/Spall_Strength_Database_AliShargh(Processed).csv" \
  --target "Spall (Gpa)" \
  --folds 3 \
  --repeats 2 \
  --missing-strategies median knn drop_sparse_cols \
  --outlier-strategies keep clip_iqr drop_iqr \
  --models "Gradient Boosting" "Random Forest" "XGBoost"
```

Key outputs:

```text
results/robustness_*/robustness_summary.json
results/robustness_*/robustness_results.csv
results/robustness_*/model_stability.csv
results/robustness_*/feature_stability.csv
```

The robustness workflow is the preferred way to answer whether a model and its
important features are stable under reasonable data-cleaning choices.

## LLM-Guided Strong Evidence Loop

The LLM-guided loop writes uncertainty notes and model-selection evidence to
JSON, feeds them back to the LLM, asks for the next experiment plan, then runs
the next robustness experiment. The goal is to improve conclusion strength from
weak to moderate or strong when the data supports it.

```bash
python scripts/llm_materials_strong_loop.py \
  --data "data/real_data/Spall_Strength_Database_AliShargh(Processed).csv" \
  --target "Spall (Gpa)" \
  --max-iterations 3 \
  --folds 3 \
  --repeats 2
```

Each iteration writes files like:

```text
results/llm_strong_loop_*/iteration_*/uncertainty_notes.json
results/llm_strong_loop_*/iteration_*/model_selection_evidence.json
results/llm_strong_loop_*/iteration_*/llm_next_plan.json
```

The loop should become "strong" only when the evidence is actually stable:

- repeated-CV metrics are high and low variance;
- calibration and coverage are acceptable;
- the chosen model is stable across preprocessing choices;
- important features are stable across repeated experiments;
- the conclusion is not dependent on one fragile missing/outlier strategy.

## Uncertainty Quantification

UQ evaluation is part of model selection where possible. The project uses
`uncertainty-toolbox` through local wrappers such as
`aims_agent.uncertainty_evaluator.UncertaintyEvaluator`.

Common UQ metrics include:

- calibration error;
- miscalibration area;
- empirical coverage at confidence levels such as 68 percent and 95 percent;
- sharpness;
- negative log likelihood;
- CRPS.

For a standalone UQ example:

```bash
python examples/uncertainty_toolbox_example.py
```

For probabilistic neural-network UQ:

```bash
python examples/probabilistic_nn_example.py
```

Related docs:

- `docs/PROBABILISTIC_NN_GUIDE.md`
- `docs/PROBABILISTIC_NN_IMPLEMENTATION.md`

## Data Interface

All data loaders should conform to the project data-interface contract:

- `load_dataset(config)` returns a `DatasetBundle`.
- `validate_schema(df, schema)` checks declared features, target, units, source,
  and description.
- `get_metadata(bundle)` returns feature, target, unit, source, description,
  shape, and dtype metadata for the LLM and downstream modules.

This keeps the modeling pipeline independent of hardcoded dataset paths or
column names.

## Materials Project Ingestion

The optional Materials Project path supports either local exports or live API
queries through `mp-api`.

Install optional dependencies:

```bash
python -m pip install -r requirements-matsci.txt
```

Offline local-export example:

```bash
python scripts/ingest_materials_project.py \
  --config examples/local_matsci_ingestion_config.json
```

Live query example:

```bash
export MP_API_KEY="..."
python scripts/ingest_materials_project.py \
  --config examples/materials_project_ingestion_config.json
```

Use Materials Project data in the full agent pipeline:

```bash
python -m aims_agent.cli \
  --materials-project \
  --mp-chemsys Li-Fe-O \
  --mp-limit 200 \
  --target formation_energy_per_atom \
  --scaling standard \
  --preprocessed-output data/materials_project_li_fe_o_preprocessed.csv \
  --motivation "Predict formation energy from Materials Project summary descriptors"
```

## Code Generation Policy

Model code generation is used to execute a selected model, not to replace model
selection.

Execution order:

1. Try the builtin model mapping.
2. Try dynamic import for the selected model.
3. Use code generation only if both paths are unavailable or fail.
4. Validate generated estimators before training.
5. Use debug/retry agents only for generated-code repair when enabled.

## Testing

Run the test suite:

```bash
python -m pytest
```

Run a focused subset:

```bash
python -m pytest tests/test_data_analyzer.py tests/test_probabilistic_integration.py
```

## Development Notes

- Put reusable implementation in `aims_agent/`.
- Put runnable workflow scripts in `scripts/`.
- Put small demonstration code in `examples/`.
- Put local run outputs in `results/`.
- Keep datasets in `data/`, but do not write generated experiment artifacts
  there.
- Avoid committing caches, virtual environments, `.DS_Store`, logs, and
  library-generated runtime folders.

## Important Limitations

- LLM recommendations must be validated. They are not evidence by themselves.
- Small tabular datasets can produce fragile model rankings. Prefer repeated CV
  and robustness analysis before making strong scientific claims.
- UQ quality can be weak even when RMSE or R2 looks good. Calibration and
  coverage should be reported alongside point metrics.
- Feature importance is only scientifically persuasive when it is stable across
  preprocessing choices, CV repeats, and reasonable model families.
