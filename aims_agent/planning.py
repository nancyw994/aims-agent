"""
LLM-based workflow planning.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING
from typing import Any, List, Mapping

if TYPE_CHECKING:
    from aims_agent.agent import Agent

# Executable actions the pipeline can run (in typical order)
PLAN_ACTIONS = ("select_model", "codegen", "train", "evaluate", "interpret")

DEFAULT_PLAN = [
    {"action": "select_model", "description": "Select ML model based on data distribution"},
    {"action": "codegen", "description": "Generate and execute custom code component if needed"},
    {"action": "train", "description": "Split data, train model with hyperparameter tuning"},
    {"action": "evaluate", "description": "Compute metrics and generate plots on test set"},
    {"action": "interpret", "description": "Summarize results and suggest improvements"},
]


def plan_workflow_steps(
    agent: "Agent",
    motivation: str,
    dataset_metadata: Mapping[str, Any] | None = None,
    background_knowledge: str | None = None,
    include_codegen: bool = False,
) -> List[dict[str, str]]:
    """
    Ask LLM for a structured workflow plan. Returns list of {action, description}.
    Actions must be from PLAN_ACTIONS.
    Falls back to DEFAULT_PLAN if parsing fails.
    """
    if include_codegen:
        actions_header = "select_model, codegen, train, evaluate, interpret"
        actions_rules = """- select_model: choose ML model based on data
- codegen: generate and execute a custom Python ML component when standard libraries are insufficient
- train: split data, train model, optionally tune hyperparameters
- evaluate: compute metrics (R2/MSE for regression, accuracy/F1 for classification), save plots
- interpret: summarize performance and suggest improvements"""
        actions_order = "select_model, codegen, train, evaluate, interpret"
        required_text = "All five are required."
    else:
        actions_header = "select_model, train, evaluate, interpret"
        actions_rules = """- select_model: choose ML model based on data
- train: split data, train model, optionally tune hyperparameters
- evaluate: compute metrics (R2/MSE for regression, accuracy/F1 for classification), save plots
- interpret: summarize performance and suggest improvements"""
        actions_order = "select_model, train, evaluate, interpret"
        required_text = "All four are required."

    prompt = f"""You are a materials science ML expert. The user wants to: {motivation}

Available workflow actions (use exactly these): {actions_header}
{actions_rules}

"""
    if dataset_metadata:
        prompt += "Dataset context (JSON):\n"
        prompt += json.dumps(dataset_metadata, indent=2) + "\n\n"

    if background_knowledge:
        prompt += "User background knowledge and constraints:\n"
        prompt += background_knowledge.strip() + "\n\n"

    prompt += f"""Return ONLY a JSON array of workflow steps. Each step: {{"action": "action_name", "description": "brief human-readable description"}}
Use actions in this order: {actions_order}. {required_text}
Example:
[
  {{"action": "select_model", "description": "Choose model suited to class imbalance"}},
  {{"action": "codegen", "description": "Generate a custom feature transform component from domain constraints"}},
  {{"action": "train", "description": "Train with stratified split and hyperparameter tuning"}},
  {{"action": "evaluate", "description": "Compute accuracy, F1, confusion matrix"}},
  {{"action": "interpret", "description": "Summarize results and next steps"}}
]
"""

    try:
        response = agent.call_llm(prompt)
        steps = _parse_plan_json(response)
        if steps:
            return steps
    except Exception:
        pass
    return DEFAULT_PLAN


def _parse_plan_json(response: str) -> List[dict[str, str]] | None:
    """Extract JSON array of {action, description} from LLM response."""
    cleaned = re.sub(r"```(?:json)?\s*", "", response).strip()
    start = cleaned.find("[")
    if start == -1:
        return None
    depth = 0
    end = -1
    for i in range(start, len(cleaned)):
        if cleaned[i] == "[":
            depth += 1
        elif cleaned[i] == "]":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end == -1:
        return None
    try:
        arr = json.loads(cleaned[start:end])
    except json.JSONDecodeError:
        return None
    if not isinstance(arr, list):
        return None
    out: List[dict[str, str]] = []
    for item in arr:
        if not isinstance(item, dict):
            continue
        action = item.get("action")
        description = item.get("description", "")
        if action in PLAN_ACTIONS:
            out.append({"action": action, "description": str(description)})
    return out if out else None


def plan_steps(
    agent: "Agent",
    motivation: str,
    dataset_metadata: Mapping[str, Any] | None = None,
    background_knowledge: str | None = None,
    include_codegen: bool = False,
) -> List[str]:
    """
    Legacy: returns list of human-readable step descriptions for display.
    Uses plan_workflow_steps and extracts descriptions.
    """
    steps_struct = plan_workflow_steps(
        agent,
        motivation,
        dataset_metadata,
        background_knowledge=background_knowledge,
        include_codegen=include_codegen,
    )
    return [s["description"] for s in steps_struct]
