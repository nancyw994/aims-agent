"""Aggregate self-correction JSONL logs into concise metrics."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


def aggregate_self_correction_logs(log_path: str | Path) -> dict[str, Any]:
    p = Path(log_path)
    if not p.exists() or not p.is_file():
        return {
            "log_path": str(p),
            "total_attempts": 0,
            "patched_attempts": 0,
            "failure_code_counts": {},
            "phase_counts": {},
            "top_failure_codes": [],
            "summary": "No self-correction log found.",
        }

    rows: list[dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue

    failure_counter = Counter(str(r.get("failure_code", "runtime_exception")) for r in rows)
    phase_counter = Counter(str(r.get("step", "unknown")) for r in rows)
    total = len(rows)
    patched = sum(1 for r in rows if bool(r.get("patched")))
    top = failure_counter.most_common(3)
    top_text = ", ".join(f"{k}({v})" for k, v in top) if top else "n/a"
    summary = (
        f"Self-correction attempts={total}, patched={patched}. "
        f"Top failures: {top_text}."
    )
    return {
        "log_path": str(p),
        "total_attempts": total,
        "patched_attempts": patched,
        "failure_code_counts": dict(failure_counter),
        "phase_counts": dict(phase_counter),
        "top_failure_codes": top,
        "summary": summary,
    }


__all__ = ["aggregate_self_correction_logs"]
