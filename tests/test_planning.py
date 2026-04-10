"""Tests for aims_agent.planning (parse logic, no LLM)."""

from __future__ import annotations

import pytest

from aims_agent.planning import _parse_plan_json


def test_parse_plan_json_valid():
    raw = """
```json
[
  {"action": "select_model", "description": "Pick model"},
  {"action": "train", "description": "Train"},
  {"action": "evaluate", "description": "Eval"},
  {"action": "interpret", "description": "Done"}
]
```
"""
    steps = _parse_plan_json(raw)
    assert steps is not None
    assert [s["action"] for s in steps] == ["select_model", "train", "evaluate", "interpret"]


def test_parse_plan_json_filters_unknown_action():
    raw = """[{"action": "select_model", "description": "a"}, {"action": "invalid", "description": "b"}]"""
    steps = _parse_plan_json(raw)
    assert steps is not None
    assert len(steps) == 1
    assert steps[0]["action"] == "select_model"


def test_parse_plan_json_garbage():
    assert _parse_plan_json("no array here") is None
