# aims-agent

AI Agent for ML in Materials Science. The project provides a **standardized data ingestion interface**: real MatSci datasets can be loaded through CSV/JSON files or external data sources, while the architecture supports future replacement with literature-extracted datasets, experimental CSVs, database APIs, and structured MatSci repositories. The system remains **modular and forward-compatible**.

---

## Data interface & replacement strategy

### Interface contract (`data_interface.py`)

- **`load_dataset(config)`** — Returns a `DatasetBundle` (DataFrame + schema). All data access goes through implementations of `DataInterface`.
- **`validate_schema(df, schema)`** — Ensures the DataFrame has the declared features, target, units, source, and description.
- **`get_metadata(bundle)`** — Returns a dict (features, target, units, source, description, shape, dtypes) for the LLM and downstream modules.

**Schema format:** `features` (list), `target` (str), `units` (dict), `source` (str), `description` (str). Downstream code must rely only on this interface and on `bundle.df` / `bundle.schema`, not on hardcoded column names or file paths.

### Replacement strategy

1. Implement a new loader that conforms to the interface, e.g.:
   ```python
   class MaterialsProjectLoader(DataInterface):
       def load_dataset(self, config):
           # fetch from API / CSV / repo, build DataFrame + DatasetSchema
           return DatasetBundle(df=df, schema=schema)
   ```
2. In the workflow (e.g. `cli.py`), change only the loader instantiation:
   ```python
   loader = MaterialsProjectLoader()  # was SyntheticDataLoader()
   ```
3. No changes are required in the agent, planner, or future codegen/run/report steps. They depend only on `DataInterface` and `DatasetBundle`.

---

## Model Selection & Dependency Management

- **`model_selector.py`** — `suggest_model(agent, features, target)` asks the LLM (via the Agent) for a model name and Python package (e.g. `RandomForestRegressor`, `scikit-learn`). The LLM must return both in the format `MODEL: ...` / `PACKAGE: ...`.
- **`dependency_manager.py`** — `ensure_package_installed(package_name)` checks if the package is importable; if not, runs `pip install <package>` in a subprocess. Installation failures are logged to `logs/dependency_install.log`.
- **Integration** — The workflow is **DATA INTERFACE → PLAN → MODEL SELECT**. `agent.select_model_and_ensure_deps(features, target)` runs the suggestion and install step. Example: LLM suggests `RandomForestRegressor` from `scikit-learn`; the agent installs `scikit-learn` if missing.

### Other examples (model / package pairs the LLM may suggest)

| Model (class name)      | Package (pip install) |
|-------------------------|------------------------|
| `RandomForestRegressor`  | `scikit-learn`         |
| `GradientBoostingRegressor` | `scikit-learn`     |
| `Ridge`, `Lasso`         | `scikit-learn`         |
| `XGBRegressor`          | `xgboost`              |
| `LGBMRegressor`         | `lightgbm`             |
| `CatBoostRegressor`     | `catboost`             |
| `SVR`                   | `scikit-learn`         |

The LLM must reply in this format (other text is ignored; the parser looks for these lines):

```
MODEL: XGBRegressor
PACKAGE: xgboost
```

To try multiple scenarios without using the API, run the demo script:

```bash
python examples/model_selection_examples.py
```

---

## Training & Results Analysis

- **`model_trainer.py`** — `ModelTrainer(model_class, hyperparams?)` prepares train/test split, trains the model (optionally with `GridSearchCV`), and returns `(y_true, y_pred)` from `predict()`.
- **`results_analyzer.py`** — `compute_metrics(y_true, y_pred)` (R2, MSE, RMSE, MAE); `plot_results(...)` saves predicted-vs-actual and residual plots to `results/`; `interpret_with_llm(agent, metrics, model_name)` asks the LLM to interpret the metrics.
- **`model_selector.load_model_class(suggestion)`** — Resolves the model class from a `ModelSuggestion` for training (uses a small import map for common models).

**Full Phase 1 workflow:** DATA INTERFACE → PLAN → MODEL SELECT → TRAIN → REPORT.

```bash
# Run full pipeline (set OPENAI_API_KEY or OPENROUTER_API_KEY in .env)
python -m aims_agent --motivation "Predict hardness from composition and processing"

# Stop after model selection (no training)
python -m aims_agent --motivation "..." --skip-train
```

Outputs: metrics printed to console, plot at `results/model_performance.png`, and LLM interpretation of the results.

---

## Model CodeGen policy (selected-model first)

Model CodeGen is used to **execute the selected model**, not to replace model selection.

Execution order:

1. Try builtin mapping for the selected model.
2. Try dynamic import for the selected model.
3. Only if both paths are unavailable or fail, trigger CodeGen for that same selected model.

In other words, if a selected model already has a working builtin/dynamic path, CodeGen is not called.

---

## Real MatSci Data Ingestion

- **`matsci_data_ingestor.py`** — `MaterialsProjectDataIngestor` loads local CSV/JSON materials exports or fetches live Materials Project summary data through the optional `mp-api` client.
- **Preprocessing** — Missing feature values can be dropped or imputed; outliers can be ignored, IQR-clipped, or IQR-dropped; numeric features can be left raw, standardized, or min-max scaled. A limited `preprocessing_suggestion` parser maps simple LLM guidance onto those deterministic operations.
- **Agent integration** — `Agent.retrieve_real_materials_data(config)` and the CLI `--materials-project` path use the same `DataInterface` contract as synthetic and CSV data.
- **Reproducibility** — `scripts/ingest_materials_project.py` runs from JSON config and can save a preprocessed CSV.

Install optional live-ingestion dependencies:

```bash
pip install -r requirements-matsci.txt
```

Run a live Materials Project query:

```bash
export MP_API_KEY="..."
python scripts/ingest_materials_project.py \
  --config examples/materials_project_ingestion_config.json
```

Run the offline local-export example:

```bash
python scripts/ingest_materials_project.py \
  --config examples/local_matsci_ingestion_config.json
```

Use live Materials Project data in the full agent pipeline:

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

---

## LLM-Driven Data Analysis & Strategy

- **`data_analyzer.py`** — Builds a real-data profile with descriptive statistics, missingness, skewness, IQR outlier checks, target correlations, and high feature-feature correlations.
- **Plots** — Saves histograms, a numeric correlation heatmap, and target relationship scatter plots.
- **Strategy formulation** — Creates a new run folder under `results/` for every execution and writes `strategy.json`, `strategy_report.md`, and all plots there. With an LLM-enabled `Agent`, the strategy prompt includes schema, statistics, risk flags, and plot references. With `--no-llm`, it writes a deterministic heuristic strategy.

Run against the Materials Project output:

```bash
python scripts/analyze_matsci_strategy.py \
  --no-llm \
  --data data/materials_project_li_fe_o_preprocessed.csv \
  --target formation_energy_per_atom \
  --output-dir results
```

Outputs:

- `results/run_*/profile.json`
- `results/run_*/strategy.json`
- `results/run_*/strategy_report.md`
- `results/run_*/histograms.png`
- `results/run_*/correlation_heatmap.png`
- `results/run_*/target_relationshi