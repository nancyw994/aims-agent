# Data

This directory contains local example and real materials datasets used by the
agent workflows.

- `synthetic_materials.csv` is a small synthetic dataset for quick demos.
- `example_materials_project_summary.json` and
  `example_materials_project_preprocessed.csv` support offline ingestion tests.
- `materials_project_li_fe_o_preprocessed.csv` is an example preprocessed
  Materials Project export.
- `real_data/` contains real materials datasets, including the processed
  spall-strength database used by the robustness and LLM-guided evidence loop.

Generated experiment outputs should go to `results/`, not this directory.
