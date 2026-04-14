from __future__ import annotations

import json
from pathlib import Path

from aims_agent.failure_codes import ALL_FAILURE_CODES
from aims_agent.self_correction_report import aggregate_self_correction_logs
from aims_agent.validator import validate_training_result_detailed


def test_failure_codes_are_standardized():
    out = validate_training_result_detailed([1.0, 2.0], [1.0, float("nan")], task_type="regression")
    assert out.code in ALL_FAILURE_CODES


def test_aggregate_self_correction_logs_counts(tmp_path):
    p = Path(tmp_path) / "sc.jsonl"
    rows = [
        {"step": "load_generated_estimator_class", "failure_code": "predict_shape_error", "patched": True},
        {"step": "load_generated_estimator_class", "failure_code": "predict_shape_error", "patched": False},
        {"step": "runtime_validation_repair", "failure_code": "nan_inf_predictions", "patched": True},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    rep = aggregate_self_correction_logs(p)
    assert rep["total_attempts"] == 3
    assert rep["patched_attempts"] == 2
    assert rep["failure_code_counts"]["predict_shape_error"] == 2
    assert "Top failures" in rep["summary"]
