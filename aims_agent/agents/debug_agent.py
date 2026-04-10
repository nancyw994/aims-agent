"""
Debug agent: repair failing generated model module code using LLM feedback.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from aims_agent.agents.codegen_agent import GENERATED_ESTIMATOR_CLASS_NAME
from aims_agent.code_writer import extract_python_code, validate_python_syntax
from aims_agent.specs import CodeGenSpec


@dataclass
class DebugPatchResult:
    corrected_code: str
    diagnosis: str
    patch_summary: str


def _parse_debug_response(text: str) -> DebugPatchResult:
    """Parse JSON-first debug responses; fallback to plain code block."""
    stripped = text.strip()
    start = stripped.find("{")
    end = stripped.rfind("}") + 1
    if start != -1 and end > start:
        try:
            payload = json.loads(stripped[start:end])
            code = str(payload.get("code", "")).strip()
            diagnosis = str(payload.get("diagnosis", "")).strip()
            patch_summary = str(payload.get("patch_summary", "")).strip()
            if code:
                validate_python_syntax(code)
                return DebugPatchResult(
                    corrected_code=code,
                    diagnosis=diagnosis or "No diagnosis provided.",
                    patch_summary=patch_summary or "No patch summary provided.",
                )
        except Exception:
            pass

    code = extract_python_code(text)
    validate_python_syntax(code)
    return DebugPatchResult(
        corrected_code=code,
        diagnosis="Fallback parse: response was not valid JSON.",
        patch_summary="Patched from extracted python block.",
    )


def repair_model_module_code(
    agent: Any,
    *,
    spec: CodeGenSpec,
    broken_code: str,
    error_message: str,
) -> DebugPatchResult:
    """
    Return corrected code and a compact diagnosis/patch summary.
    """
    prompt = f"""You are an expert Python debugger. The following module failed to load or run.

Error:
{error_message}

Spec (must still satisfy):
{json.dumps(spec.to_prompt_dict(), indent=2)}

Broken code:
```python
{broken_code}
```

Return ONLY a JSON object with keys:
- diagnosis: short root-cause diagnosis (1 sentence)
- patch_summary: concise summary of the concrete fix (1 sentence)
- code: the complete corrected Python module source

The code MUST define class `{GENERATED_ESTIMATOR_CLASS_NAME}` with fit(self, X, y) and predict(self, X).
Do not include markdown fences.
"""
    response = agent.call_llm(prompt)
    return _parse_debug_response(response)


__all__ = ["DebugPatchResult", "repair_model_module_code"]
