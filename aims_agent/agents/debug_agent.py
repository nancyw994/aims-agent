"""
Debug agent: repair failing generated model module code using LLM feedback.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

from aims_agent.agents.codegen_agent import GENERATED_ESTIMATOR_CLASS_NAME
from aims_agent.code_writer import extract_python_code, validate_python_syntax
from aims_agent.specs import CodeGenSpec


@dataclass
class DebugPatchResult:
    corrected_code: str
    diagnosis: str
    patch_summary: str


@dataclass
class RetryDecision:
    retry: bool
    reason: str


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


class SelfCorrectionAgent:
    """LLM-driven self-correction agent for generated model modules."""

    def __init__(self, llm_agent: Any):
        if not llm_agent or not hasattr(llm_agent, "call_llm"):
            raise RuntimeError("SelfCorrectionAgent requires an object with call_llm(prompt)")
        self._llm_agent = llm_agent

    def propose_fix(
        self,
        *,
        spec: CodeGenSpec,
        broken_code: str,
        error_message: str,
        traceback_text: str = "",
        attempt: int = 0,
        recent_patch_summary: str = "",
    ) -> DebugPatchResult:
        """Return corrected code and compact diagnosis/patch summary."""
        prompt = f"""You are an expert Python debugger. The following module failed to load or run.

Error:
{error_message}

Traceback:
{traceback_text}

Attempt index: {attempt}
Recent patch summary: {recent_patch_summary or "N/A"}

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
        response = self._llm_agent.call_llm(prompt)
        return _parse_debug_response(response)

    def should_retry(
        self,
        *,
        attempt: int,
        max_retries: int,
        repeated_error_count: int,
    ) -> RetryDecision:
        """Deterministic retry decision policy."""
        if attempt >= max_retries:
            return RetryDecision(False, "Reached maximum retry limit.")
        if repeated_error_count >= 2:
            return RetryDecision(False, "Repeated identical error signature.")
        return RetryDecision(True, "Retry allowed.")

    def summarize(self, records: list[Mapping[str, Any]]) -> str:
        """Compact summary for pipeline result reporting."""
        if not records:
            return "No correction required."
        patched = sum(1 for r in records if bool(r.get("patched")))
        last = records[-1]
        last_error = str(last.get("error_message", "")).strip()
        if last_error:
            last_error = last_error.splitlines()[0][:160]
        return (
            f"Self-correction ran {len(records)} attempt(s), produced {patched} patch(es). "
            f"Last error: {last_error or 'N/A'}"
        )


def repair_model_module_code(
    agent: Any,
    *,
    spec: CodeGenSpec,
    broken_code: str,
    error_message: str,
) -> DebugPatchResult:
    """Backward-compatible helper that delegates to SelfCorrectionAgent."""
    sc = SelfCorrectionAgent(agent)
    return sc.propose_fix(
        spec=spec,
        broken_code=broken_code,
        error_message=error_message,
    )


__all__ = [
    "DebugPatchResult",
    "RetryDecision",
    "SelfCorrectionAgent",
    "repair_model_module_code",
]
