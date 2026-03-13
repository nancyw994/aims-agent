from __future__ import annotations

from typing import Mapping, Any

import numpy as np
import pandas as pd
import os

from aims_agent.data_interface import (
    DataInterface,
    DatasetBundle,
    DatasetSchema,
    validate_schema,
)


class SyntheticDataLoader(DataInterface):
    """
    Synthetic Materials Science–like dataset generator.

    Generates columns such as:
      - Al_conc_wt_pct, Ti_conc_wt_pct, V_conc_wt_pct
      - process_temperature_C, process_time_h
      - hardness_HV (target)

    This is a placeholder implementation that conforms to the data interface;
    downstream code must not depend on any of these column names.
    """

    def load_dataset(self, config: Mapping[str, Any]) -> DatasetBundle:
        # Configuration with sensible defaults
        n_samples = int(config.get("n_samples", 200))
        random_seed = config.get("random_seed")
        noise_sigma = float(config.get("noise_sigma", 5.0))

        if random_seed is not None:
            np.random.seed(int(random_seed))

        # Define columns
        features = [
            "Al_conc_wt_pct",
            "Ti_conc_wt_pct",
            "V_conc_wt_pct",
            "process_temperature_C",
            "process_time_h",
        ]
        target = "hardness_HV"

        # Composition in wt.% (simple bounds for illustration)
        al = np.random.uniform(0.0, 6.0, size=n_samples)
        ti = np.random.uniform(0.0, 4.0, size=n_samples)
        v = np.random.uniform(0.0, 3.0, size=n_samples)

        # Processing parameters
        temp = np.random.uniform(200, 600, size=n_samples)  # °C
        time_h = np.random.uniform(0.5, 8.0, size=n_samples)  # hours

        base_hardness = 150.0
        hardness = (
            base_hardness
            + 2.0 * al
            + 3.0 * ti
            + 1.5 * v
            + 0.05 * temp
            - 10 * np.log(time_h + 1e-9)
            + noise_sigma
        )

        hardness_noisy = hardness + np.random.normal(0.0, noise_sigma, size=n_samples)

        df = pd.DataFrame(
            {
                "Al_conc_wt_pct": al,
                "Ti_conc_wt_pct": ti,
                "V_conc_wt_pct": v,
                "process_temperature_C": temp,
                "process_time_h": time_h,
                "hardness_HV": hardness_noisy,
            }
        )

        schema = DatasetSchema(
            features=features,
            target=target,
            units={
                "Al_conc_wt_pct": "wt.%",
                "Ti_conc_wt_pct": "wt.%",
                "V_conc_wt_pct": "wt.%",
                "process_temperature_C": "°C",
                "process_time_h": "h",
                "hardness_HV": "HV",
            },
            source="synthetic placeholder",
            description=(
                "Synthetic alloy dataset with composition, processing parameters, "
                "and Vickers hardness target, generated from a simple heuristic model "
                "plus Gaussian noise."
            ),
            shape=df.shape,
            dtypes={c: str(dt) for c, dt in df.dtypes.items()},
        )

        # Enforce schema contract
        validate_schema(df, schema)

        return DatasetBundle(df=df, schema=schema)


def save_example_csv(path: str, config: Mapping[str, Any] | None = None) -> None:
    """
    Generate a synthetic dataset and save it as a CSV file in the data folder.

    This is used to provide an example placeholder dataset file
    (e.g., synthetic_materials.csv). The file is always written under data/.
    """
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, os.path.basename(path))
    loader = SyntheticDataLoader()
    bundle = loader.load_dataset(config or {})
    bundle.df.to_csv(out_path, index=False)


__all__ = ["SyntheticDataLoader", "save_example_csv"]

