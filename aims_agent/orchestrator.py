"""
Multi-agent training phase: Execution Path Resolver + optional model CodeGen + debug retries.

Trainer (ModelTrainer) stays deterministic; this module only resolves model_class.
"""

from __future__ import annotations

import json
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from aims_agent.agents.codegen_agent import generate_model_estimator_module, load_generated_estimator_class
from aims_agent.agents.debug_agent import repair_model_module_code
from aims_agent.model_selector import ModelSuggestion, load_model_class
from aims_agent.path_resolver import ExecutionPathResolver, PathDecision
from aims_agent.specs import build_model_codegen_spec


@dataclass
class ModelClassResolution:
    model_class: type
    execution_path: str
    path_reason: str
    generated_model_wrapper_path: str
    self_correction_attempts: int = 0
    self_correction_success: bool = False
    self_correction_log_path: str = ""
    self_correction_summary: str = ""


@dataclass
class DebugAttemptRecord:
    attempt: int
    step: str
    exception_type: str
    error_message: str
    traceback: str
    offending_code: str
    diagnosis: str = ""
    patch_summary: str = ""
    patched: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


def _error_signature(exc: Exception) -> str:
    return f"{type(exc).__name__}:{str(exc).splitlines()[0][:200]}"


def _write_self_correction_log(log_path: Path, record: DebugAttemptRecord) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def _codegen_model_class(
    agent: Any,
    suggestion: ModelSuggestion,
    task_type: str,
    metadata: Mapping[str, Any],
    path_decision: PathDecision,
    *,
    background_knowledge: str | None,
    generated_code_dir: str,
    max_codegen_retries: int,
) -> ModelClassResolution:
    if not agent or not hasattr(agent, "call_llm"):
        raise RuntimeError("CodeGen requires an agent with call_llm")
    spec = build_model_codegen_spec(suggestion, task_type, path_decision)
    module_name = f"generated_estimator_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    file_path_str = generate_model_estimator_module(
        agent,
        spec=spec,
        dataset_metadata=metadata,
        background_knowledge=background_knowledge,
        output_dir=generated_code_dir,
        module_name=module_name,
    )
    code = Path(file_path_str).read_text(encoding="utf-8")
    log_path = Path(generated_code_dir).resolve() / f"self_correction_{module_name}.jsonl"
    repeated_sig_count = 0
    prev_sig = ""
    last_err: str | None = None
    for attempt in range(max_codegen_retries + 1):
        try:
            cls = load_generated_estimator_class(file_path_str)
            summary = (
                "No correction required."
                if attempt == 0
                else f"Self-correction succeeded after {attempt} retry attempt(s)."
            )
            return ModelClassResolution(
                model_class=cls,
                execution_path="codegen",
                path_reason=path_decision.reason,
                generated_model_wrapper_path=file_path_str,
                self_correction_attempts=attempt,
                self_correction_success=True,
                self_correction_log_path=str(log_path),
                self_correction_summary=summary,
            )
        except Exception as e:
            last_err = str(e)
            sig = _error_signature(e)
            repeated_sig_count = repeated_sig_count + 1 if sig == prev_sig else 1
            prev_sig = sig

            tb_text = traceback.format_exc()
            rec = DebugAttemptRecord(
                attempt=attempt,
                step="load_generated_estimator_class",
                exception_type=type(e).__name__,
                error_message=last_err,
                traceback=tb_text,
                offending_code=code,
            )
            if attempt >= max_codegen_retries:
                _write_self_correction_log(log_path, rec)
                break
            if repeated_sig_count >= 2:
                rec.patch_summary = "Stop early due to repeated identical error signature."
                _write_self_correction_log(log_path, rec)
                break

            patch = repair_model_module_code(
                agent,
                spec=spec,
                broken_code=code,
                error_message=f"{last_err}\n\nTraceback:\n{tb_text}",
            )
            code = patch.corrected_code
            rec.diagnosis = patch.diagnosis
            rec.patch_summary = patch.patch_summary
            rec.patched = True
            _write_self_correction_log(log_path, rec)
            Path(file_path_str).write_text(code, encoding="utf-8")

    fail_summary = (
        f"Self-correction failed after {max_codegen_retries + 1} attempt(s): {last_err}"
    )
    raise RuntimeError(
        f"{fail_summary}. See log: {log_path}"
    )


def resolve_model_class_multi_agent(
    agent: Any,
    suggestion: ModelSuggestion,
    task_type: str,
    metadata: Mapping[str, Any],
    *,
    use_llm: bool = True,
    background_knowledge: str | None = None,
    generated_code_dir: str = "generated_code",
    max_codegen_retries: int = 2,
) -> ModelClassResolution:
    """
    Rule-based Execution Path Resolver, then load or CodeGen a model class.

    - builtin / dynamic_import: load_model_class when possible.
    - On load failure with use_llm, fall back to CodeGen.
    - codegen: emit GeneratedEstimator module (with debug retries on load errors).
    """
    resolver = ExecutionPathResolver()
    dec = resolver.resolve(suggestion)
    path = dec.path
    reason = dec.reason

    if path in ("builtin", "dynamic_import"):
        try:
            cls = load_model_class(suggestion)
            return ModelClassResolution(
                model_class=cls,
                execution_path=path,
                path_reason=reason,
                generated_model_wrapper_path="",
                self_correction_summary="No self-correction needed (non-codegen path).",
            )
        except Exception as e:
            if use_llm:
                fallback = PathDecision(
                    "codegen",
                    f"{path} load failed ({e}); falling back to CodeGen",
                )
                return _codegen_model_class(
                    agent,
                    suggestion,
                    task_type,
                    metadata,
                    fallback,
                    background_knowledge=background_knowledge,
                    generated_code_dir=generated_code_dir,
                    max_codegen_retries=max_codegen_retries,
                )
            raise

    if path == "codegen":
        if not use_llm:
            raise RuntimeError(
                "Selected model has no builtin/dynamic import path; CodeGen requires LLM "
                "(remove --no-llm or disable --multi-agent for this run)."
            )
        return _codegen_model_class(
            agent,
            suggestion,
            task_type,
            metadata,
            dec,
            background_knowledge=background_knowledge,
            generated_code_dir=generated_code_dir,
            max_codegen_retries=max_codegen_retries,
        )

    raise RuntimeError(f"Unexpected execution path: {path!r}")


__all__ = ["ModelClassResolution", "resolve_model_class_multi_agent"]
