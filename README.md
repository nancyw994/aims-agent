# aims-agent

AI Agent for ML in Materials Science. Week 2 delivers a **standardized data ingestion interface**: real MatSci datasets are not yet integrated, but the architecture supports future replacement with literature-extracted datasets, experimental CSVs, database APIs, and structured MatSci repositories. The system remains **modular and forward-compatible**.

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

## Week 3: Model selection & dependency management

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
python examples/week3_model_selection_examples.py
```

---

## Week 4: Training & results analysis

- **`model_trainer.py`** — `ModelTrainer(model_class, hyperparams?)` prepares train/test split, trains the model (optionally with `GridSearchCV`), and returns `(y_true, y_pred)` from `predict()`.
- **`results_analyzer.py`** — `compute_metrics(y_true, y_pred)` (R2, MSE, RMSE, MAE); `plot_results(...)` saves predicted-vs-actual and residual plots to `results/`; `interpret_with_llm(agent, metrics, model_name)` asks the LLM to interpret the metrics.
- **`model_selector.load_model_class(suggestion)`** — Resolves the model class from a `ModelSuggestion` for training (uses a small import map for common models).

**Full Phase 1 workflow:** DATA INTERFACE → PLAN → MODEL SELECT → TRAIN → REPORT.

```bash
# Run full pipeline (requires OPENROUTER_API_KEY in .env)
python -m aims_agent --motivation "Predict hardness from composition and processing"

# Stop after model selection (no training)
python -m aims_agent --motivation "..." --skip-train
```

Outputs: metrics printed to console, plot at `results/model_performance.png`, and LLM interpretation of the results.