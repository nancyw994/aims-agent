"""
Pre-trained materials model integration and benchmarking.

The primary implementation targets MatGL property models, which accept
pymatgen ``Structure`` objects and expose ``predict_structure``.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from aims_agent.results_analyzer import compute_metrics


DEFAULT_MATGL_CACHE = "/private/tmp/matgl-cache"
DEFAULT_PRETRAINED_MODEL = "M3GNet-Eform-MP-2018.6.1"
MODEL_TARGETS = {
    "M3GNet-Eform-MP-2018.6.1": "formation_energy_per_atom",
    "MEGNet-MP-2018.6.1-Eform": "formation_energy_per_atom",
    "MEGNet-MP-2019.4.1-BandGap-mfi": "band_gap",
}


@dataclass
class PretrainedModelChoice:
    model_name: str
    library: str
    target: str
    reason: str
    requires_structure: bool = True


@dataclass
class PretrainedBenchmarkResult:
    model_name: str
    library: str
    target: str
    n_samples: int
    prediction_path: str
    metrics: dict[str, float]
    baseline_metrics: dict[str, float]
    failed_records: list[dict[str, Any]] = field(default_factory=list)


def _ensure_matgl_cache(cache_dir: str | Path | None = None) -> str:
    cache = str(cache_dir or os.getenv("MATGL_CACHE") or DEFAULT_MATGL_CACHE)
    Path(cache).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MATGL_CACHE", cache)
    return cache


def _structure_from_value(value: Any):
    from pymatgen.core import Structure

    if isinstance(value, Structure):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValueError("empty structure value")
        value = json.loads(stripped)
    if isinstance(value, Mapping):
        return Structure.from_dict(dict(value))
    raise TypeError(f"Unsupported structure value type: {type(value).__name__}")


def _prediction_to_float(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value)
    if arr.size == 0:
        raise ValueError("empty prediction")
    return float(arr.reshape(-1)[0])


def fetch_materials_project_structures(config: Mapping[str, Any]) -> pd.DataFrame:
    """Fetch Materials Project rows with structure objects for pre-trained models."""

    from aims_agent.matsci_data_ingestor import fetch_materials_project_summary

    fields = list(
        config.get(
            "fields",
            [
                "material_id",
                "formula_pretty",
                "formation_energy_per_atom",
                "band_gap",
                "structure",
            ],
        )
    )
    cfg = dict(config)
    cfg["fields"] = fields
    return fetch_materials_project_summary(cfg)


def identify_pretrained_models(
    *,
    agent: Any | None = None,
    target: str = "formation_energy_per_atom",
    available_models: list[str] | None = None,
    use_llm: bool = True,
) -> list[PretrainedModelChoice]:
    """Identify suitable pre-trained MatSci models, optionally using an LLM."""

    available = available_models or [
        DEFAULT_PRETRAINED_MODEL,
        "MEGNet-MP-2018.6.1-Eform",
        "MEGNet-MP-2019.4.1-BandGap-mfi",
    ]
    if use_llm and agent is not None:
        prompt = f"""You are choosing pre-trained MatSci models for property prediction.

Target property: {target}
Available MatGL models: {available}

Return ONLY JSON as a list of objects with keys:
model_name, library, target, reason, requires_structure.
Prefer a model whose native target matches the requested target."""
        try:
            response = agent.call_llm(prompt)
            start = response.find("[")
            end = response.rfind("]") + 1
            if start != -1 and end > start:
                data = json.loads(response[start:end])
                out = []
                for item in data:
                    out.append(
                        PretrainedModelChoice(
                            model_name=str(item["model_name"]),
                            library=str(item.get("library", "matgl")),
                            target=str(item.get("target", target)),
                            reason=str(item.get("reason", "")),
                            requires_structure=bool(item.get("requires_structure", True)),
                        )
                    )
                if out:
                    return out
        except Exception:
            pass

    matching = [m for m in available if MODEL_TARGETS.get(m) == target]
    model_name = matching[0] if matching else DEFAULT_PRETRAINED_MODEL
    return [
        PretrainedModelChoice(
            model_name=model_name,
            library="matgl",
            target=MODEL_TARGETS.get(model_name, target),
            reason="Heuristic selection: MatGL property model with a structure-level target matching the dataset.",
            requires_structure=True,
        )
    ]


class MatGLPretrainedModelHandler:
    """Load a MatGL pre-trained model and run structure-level predictions."""

    def __init__(
        self,
        model_name: str = DEFAULT_PRETRAINED_MODEL,
        *,
        cache_dir: str | Path | None = None,
        model_loader: Callable[[str], Any] | None = None,
    ):
        self.model_name = model_name
        self.cache_dir = _ensure_matgl_cache(cache_dir)
        self._model_loader = model_loader
        self.model = None

    def load(self) -> Any:
        if self.model is not None:
            return self.model
        if self._model_loader is not None:
            self.model = self._model_loader(self.model_name)
            return self.model

        _ensure_matgl_cache(self.cache_dir)
        try:
            import matgl
        except ImportError as exc:
            raise ImportError(
                "MatGL is required for pre-trained MatSci models. Install with: pip install matgl"
            ) from exc
        self.model = matgl.load_model(self.model_name)
        return self.model

    def predict_structure(self, structure: Any, *, state_attr: Any | None = None) -> float:
        model = self.load()
        struct = _structure_from_value(structure)
        if state_attr is not None:
            raw = model.predict_structure(structure=struct, state_attr=state_attr)
        else:
            raw = model.predict_structure(struct)
        return _prediction_to_float(raw)

    def predict_dataframe(
        self,
        df: pd.DataFrame,
        *,
        structure_col: str = "structure",
        state_attr: Any | None = None,
    ) -> tuple[pd.Series, list[dict[str, Any]]]:
        preds: list[float] = []
        failed: list[dict[str, Any]] = []
        for idx, row in df.iterrows():
            try:
                preds.append(self.predict_structure(row[structure_col], state_attr=state_attr))
            except Exception as exc:
                preds.append(np.nan)
                failed.append({"index": int(idx), "error": f"{type(exc).__name__}: {exc}"})
        return pd.Series(preds, index=df.index, name="pretrained_prediction"), failed


def benchmark_pretrained_model(
    df: pd.DataFrame,
    *,
    target: str,
    model_name: str = DEFAULT_PRETRAINED_MODEL,
    structure_col: str = "structure",
    output_dir: str | Path = "results/pretrained_model",
    cache_dir: str | Path | None = None,
    model_loader: Callable[[str], Any] | None = None,
) -> PretrainedBenchmarkResult:
    """Run pre-trained model inference and compare against a mean baseline."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if structure_col not in df.columns:
        raise ValueError(f"Structure column '{structure_col}' not found.")
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found.")

    handler = MatGLPretrainedModelHandler(
        model_name,
        cache_dir=cache_dir,
        model_loader=model_loader,
    )
    preds, failed = handler.predict_dataframe(df, structure_col=structure_col)
    out_df = df.copy()
    out_df["pretrained_prediction"] = preds
    eval_df = out_df[[target, "pretrained_prediction"]].dropna()
    if eval_df.empty:
        raise ValueError("No valid predictions available for evaluation.")

    y_true = eval_df[target].astype(float).to_numpy()
    y_pred = eval_df["pretrained_prediction"].astype(float).to_numpy()
    metrics = compute_metrics(y_true, y_pred, task_type="regression")
    baseline_pred = np.full_like(y_true, fill_value=float(np.mean(y_true)), dtype=float)
    baseline_metrics = compute_metrics(y_true, baseline_pred, task_type="regression")

    prediction_path = output_dir / "pretrained_predictions.csv"
    metrics_path = output_dir / "pretrained_metrics.json"
    out_df.to_csv(prediction_path, index=False)

    result = PretrainedBenchmarkResult(
        model_name=model_name,
        library="matgl",
        target=target,
        n_samples=int(len(eval_df)),
        prediction_path=str(prediction_path),
        metrics=metrics,
        baseline_metrics=baseline_metrics,
        failed_records=failed,
    )
    metrics_path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    return result


__all__ = [
    "DEFAULT_PRETRAINED_MODEL",
    "MODEL_TARGETS",
    "MatGLPretrainedModelHandler",
    "PretrainedBenchmarkResult",
    "PretrainedModelChoice",
    "benchmark_pretrained_model",
    "fetch_materials_project_structures",
    "identify_pretrained_models",
]
