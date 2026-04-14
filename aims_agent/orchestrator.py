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
from typing import Any, Callable, Mapping

import numpy as np

from aims_agent.agents.codegen_agent import generate_model_estimator_module, load_generated_estimator_class
from aims_agent.agents.debug_agent import SelfCorrectionAgent
from aims_agent.failure_codes import (
    FAILURE_REPEATED_ERROR_SIGNATURE,
    FAILURE_RETRY_LIMIT_REACHED,
    FAILURE_RUNTIME_EXCEPTION,
    parse_failure_code_from_message,
)
from aims_agent.model_selector import ModelSuggestion, enrich_unknown_suggestion, load_model_class
from aims_agent.model_trainer import ModelTrainer
from aims_agent.path_resolver import ExecutionPathResolver, PathDecision
from aims_agent.specs import build_model_codegen_spec
from aims_agent.validator import validate_estimator_contract, validate_training_result_detailed


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
    failure_code: str
    exception_type: str
    error_message: str
    traceback: str
    offending_code: str
    diagnosis: str = ""
    patch_summary: str = ""
    patched: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


@dataclass
class TrainingPhaseResult:
    y_true: np.ndarray
    y_pred: np.ndarray
    resolution: ModelClassResolution
    used_suggestion: ModelSuggestion
    fallback_used: bool
    training_validation_ok: bool
    training_validation_code: str
    training_validation_message: str


def _error_signature(exc: Exception) -> str:
    return f"{type(exc).__name__}:{str(exc).splitlines()[0][:200]}"


def _write_self_correction_log(log_path: Path, record: DebugAttemptRecord) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def _should_trigger_codegen_for_selected_model(
    *,
    selected_model_name: str,
    path: str,
    use_llm: bool,
    load_error: Exception | None = None,
) -> tuple[bool, str]:
    """
    CodeGen policy:
    - Only trigger for the currently selected model.
    - Trigger when resolver path is 'codegen', or builtin/dynamic_import load failed.
    """
    if not use_llm:
        return False, "LLM disabled; codegen unavailable."
    if path == "codegen":
        return True, f"Selected model {selected_model_name!r} requires codegen path."
    if path in ("builtin", "dynamic_import") and load_error is not None:
        return True, (
            f"Selected model {selected_model_name!r} failed to load via {path}: {load_error}. "
            "Falling back to codegen."
        )
    return False, "Selected model has a working non-codegen execution path."


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
    self_correction_agent = SelfCorrectionAgent(agent)
    log_path = Path(generated_code_dir).resolve() / f"self_correction_{module_name}.jsonl"
    repeated_sig_count = 0
    prev_sig = ""
    records: list[DebugAttemptRecord] = []
    last_err: str | None = None
    failure_code = FAILURE_RUNTIME_EXCEPTION
    for attempt in range(max_codegen_retries + 1):
        try:
            cls = load_generated_estimator_class(file_path_str)
            contract = validate_estimator_contract(cls, task_type=task_type)
            if not contract.ok:
                failure_code = contract.code
                raise RuntimeError(f"[{contract.code}] {contract.message}")
            summary = self_correction_agent.summarize([asdict(r) for r in records])
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
                failure_code=failure_code,
                exception_type=type(e).__name__,
                error_message=last_err,
                traceback=tb_text,
                offending_code=code,
            )
            decision = self_correction_agent.should_retry(
                attempt=attempt,
                max_retries=max_codegen_retries,
                repeated_error_count=repeated_sig_count,
            )
            if not decision.retry:
                if attempt >= max_codegen_retries:
                    rec.failure_code = FAILURE_RETRY_LIMIT_REACHED
                elif repeated_sig_count >= 2:
                    rec.failure_code = FAILURE_REPEATED_ERROR_SIGNATURE
                rec.patch_summary = decision.reason
                _write_self_correction_log(log_path, rec)
                break

            patch = self_correction_agent.propose_fix(
                spec=spec,
                broken_code=code,
                error_message=last_err,
                traceback_text=tb_text,
                attempt=attempt,
                recent_patch_summary=records[-1].patch_summary if records else "",
                previous_failures=[asdict(r) for r in records[-3:]],
            )
            code = patch.corrected_code
            rec.diagnosis = patch.diagnosis
            rec.patch_summary = patch.patch_summary
            rec.patched = True
            _write_self_correction_log(log_path, rec)
            records.append(rec)
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
            should_codegen, codegen_reason = _should_trigger_codegen_for_selected_model(
                selected_model_name=suggestion.model_name,
                path=path,
                use_llm=use_llm,
                load_error=e,
            )
            if should_codegen:
                fallback = PathDecision(
                    "codegen",
                    codegen_reason,
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
        should_codegen, codegen_reason = _should_trigger_codegen_for_selected_model(
            selected_model_name=suggestion.model_name,
            path=path,
            use_llm=use_llm,
        )
        if not should_codegen:
            raise RuntimeError(
                "Selected model has no builtin/dynamic import path; CodeGen requires LLM "
                "(remove --no-llm or disable --multi-agent for this run)."
            )
        # Unknown model: ask LLM for import path + implementation notes before codegen.
        # This enriches the spec so codegen has a concrete import hint and constraints.
        if use_llm and not suggestion.import_path:
            suggestion = enrich_unknown_suggestion(agent, suggestion, task_type)
            # Re-resolve: LLM may have provided a valid import_path → try dynamic_import first.
            enriched_dec = resolver.resolve(suggestion)
            if enriched_dec.path in ("builtin", "dynamic_import"):
                try:
                    cls = load_model_class(suggestion)
                    return ModelClassResolution(
                        model_class=cls,
                        execution_path=enriched_dec.path,
                        path_reason=enriched_dec.reason,
                        generated_model_wrapper_path="",
                        self_correction_summary="No self-correction needed (enriched import path).",
                    )
                except Exception:
                    pass  # fall through to codegen with enriched hint
        # Pass enriched reason as additional background knowledge for codegen.
        enriched_bg = background_knowledge or ""
        if suggestion.reason and suggestion.reason not in (enriched_bg or ""):
            enriched_bg = f"{enriched_bg}\n{suggestion.reason}".strip()
        return _codegen_model_class(
            agent,
            suggestion,
            task_type,
            metadata,
            PathDecision("codegen", codegen_reason or dec.reason),
            background_knowledge=enriched_bg or None,
            generated_code_dir=generated_code_dir,
            max_codegen_retries=max_codegen_retries,
        )

    raise RuntimeError(f"Unexpected execution path: {path!r}")


def repair_generated_model_after_runtime_validation(
    agent: Any,
    *,
    model_module_path: str,
    suggestion: ModelSuggestion,
    task_type: str,
    metadata: Mapping[str, Any],
    validation_error: str,
    max_codegen_retries: int = 1,
) -> ModelClassResolution:
    """
    Repair an already-generated module when runtime/post-train validation fails.
    """
    file_path = Path(model_module_path)
    _ = metadata
    if not file_path.exists():
        raise RuntimeError(f"Generated model module not found: {model_module_path}")
    code = file_path.read_text(encoding="utf-8")
    module_name = file_path.stem
    log_path = file_path.parent.resolve() / f"self_correction_runtime_{module_name}.jsonl"
    sc = SelfCorrectionAgent(agent)
    spec = build_model_codegen_spec(
        suggestion,
        task_type,
        PathDecision("codegen", "Runtime validation failed; repair generated module."),
    )
    last_err = validation_error
    failure_code = parse_failure_code_from_message(validation_error)
    for attempt in range(max(1, max_codegen_retries)):
        patch = sc.propose_fix(
            spec=spec,
            broken_code=code,
            error_message=last_err,
            attempt=attempt,
            previous_failures=[
                {
                    "attempt": attempt,
                    "step": "runtime_validation_repair",
                    "failure_code": failure_code,
                    "error_message": last_err,
                }
            ],
        )
        code = patch.corrected_code
        file_path.write_text(code, encoding="utf-8")
        try:
            cls = load_generated_estimator_class(file_path)
            contract = validate_estimator_contract(cls, task_type=task_type)
            if not contract.ok:
                raise RuntimeError(f"[{contract.code}] {contract.message}")
            return ModelClassResolution(
                model_class=cls,
                execution_path="codegen",
                path_reason="runtime_validation_repaired",
                generated_model_wrapper_path=str(file_path),
                self_correction_attempts=attempt + 1,
                self_correction_success=True,
                self_correction_log_path=str(log_path),
                self_correction_summary=f"Runtime validation repaired in {attempt + 1} attempt(s).",
            )
        except Exception as e:
            last_err = str(e)
            rec = DebugAttemptRecord(
                attempt=attempt,
                step="runtime_validation_repair",
                failure_code=failure_code,
                exception_type=type(e).__name__,
                error_message=last_err,
                traceback=traceback.format_exc(),
                offending_code=code,
                patched=True,
            )
            _write_self_correction_log(log_path, rec)
    raise RuntimeError(f"Runtime validation repair failed: {last_err}")


def run_training_phase_multi_agent(
    agent: Any,
    *,
    suggestion: ModelSuggestion,
    task_type: str,
    metadata: Mapping[str, Any],
    df: Any,
    use_llm: bool = True,
    background_knowledge: str | None = None,
    generated_code_dir: str = "generated_code",
    max_codegen_retries: int = 2,
    use_hyperparameter_tuning: bool = True,
    use_randomized_search: bool = False,
) -> TrainingPhaseResult:
    """
    Unified training-phase control: resolve -> train -> validate -> (optional) repair+retrain.
    """
    mcr = resolve_model_class_multi_agent(
        agent,
        suggestion,
        task_type,
        metadata,
        use_llm=use_llm,
        background_knowledge=background_knowledge,
        generated_code_dir=generated_code_dir,
        max_codegen_retries=max(0, max_codegen_retries),
    )

    trainer = ModelTrainer(
        mcr.model_class,
        task_type=task_type,
        use_hyperparameter_tuning=use_hyperparameter_tuning,
        use_randomized_search=use_randomized_search,
    )
    trainer.prepare_data(df, list(metadata["features"]), str(metadata["target"]))
    trainer.train()
    y_true, y_pred = trainer.predict()
    validation = validate_training_result_detailed(y_true, y_pred, task_type=task_type)
    if validation.ok:
        return TrainingPhaseResult(
            y_true=np.asarray(y_true),
            y_pred=np.asarray(y_pred),
            resolution=mcr,
            used_suggestion=suggestion,
            fallback_used=False,
            training_validation_ok=True,
            training_validation_code=validation.code,
            training_validation_message=validation.message,
        )

    # execution-level correction: only for generated modules
    if not (
        use_llm
        and mcr.execution_path == "codegen"
        and mcr.generated_model_wrapper_path
    ):
        return TrainingPhaseResult(
            y_true=np.asarray(y_true),
            y_pred=np.asarray(y_pred),
            resolution=mcr,
            used_suggestion=suggestion,
            fallback_used=False,
            training_validation_ok=False,
            training_validation_code=validation.code,
            training_validation_message=validation.message,
        )

    repaired = repair_generated_model_after_runtime_validation(
        agent,
        model_module_path=mcr.generated_model_wrapper_path,
        suggestion=suggestion,
        task_type=task_type,
        metadata=metadata,
        validation_error=f"[{validation.code}] {validation.message}",
        max_codegen_retries=max(1, max_codegen_retries),
    )
    # aggregate correction stats in one place
    mcr.self_correction_attempts += repaired.self_correction_attempts
    mcr.self_correction_success = repaired.self_correction_success
    if repaired.self_correction_summary:
        mcr.self_correction_summary = repaired.self_correction_summary

    trainer = ModelTrainer(
        repaired.model_class,
        task_type=task_type,
        use_hyperparameter_tuning=use_hyperparameter_tuning,
        use_randomized_search=use_randomized_search,
    )
    trainer.prepare_data(df, list(metadata["features"]), str(metadata["target"]))
    trainer.train()
    y_true, y_pred = trainer.predict()
    final_validation = validate_training_result_detailed(y_true, y_pred, task_type=task_type)
    return TrainingPhaseResult(
        y_true=np.asarray(y_true),
        y_pred=np.asarray(y_pred),
        resolution=mcr,
        used_suggestion=suggestion,
        fallback_used=False,
        training_validation_ok=final_validation.ok,
        training_validation_code=final_validation.code,
        training_validation_message=final_validation.message,
    )


def run_training_phase_multi_agent_with_fallback(
    agent: Any,
    *,
    suggestion: ModelSuggestion,
    fallback_suggestion: ModelSuggestion | None,
    ensure_package_installed_fn: Callable[[str], bool],
    task_type: str,
    metadata: Mapping[str, Any],
    df: Any,
    use_llm: bool = True,
    background_knowledge: str | None = None,
    generated_code_dir: str = "generated_code",
    max_codegen_retries: int = 2,
    use_hyperparameter_tuning: bool = True,
    use_randomized_search: bool = False,
) -> TrainingPhaseResult:
    """
    Training-phase control with built-in fallback model strategy.
    """
    try:
        return run_training_phase_multi_agent(
            agent,
            suggestion=suggestion,
            task_type=task_type,
            metadata=metadata,
            df=df,
            use_llm=use_llm,
            background_knowledge=background_knowledge,
            generated_code_dir=generated_code_dir,
            max_codegen_retries=max_codegen_retries,
            use_hyperparameter_tuning=use_hyperparameter_tuning,
            use_randomized_search=use_randomized_search,
        )
    except Exception:
        if (
            fallback_suggestion is None
            or not ensure_package_installed_fn(fallback_suggestion.package_name)
        ):
            raise
        out = run_training_phase_multi_agent(
            agent,
            suggestion=fallback_suggestion,
            task_type=task_type,
            metadata=metadata,
            df=df,
            use_llm=use_llm,
            background_knowledge=background_knowledge,
            generated_code_dir=generated_code_dir,
            max_codegen_retries=max_codegen_retries,
            use_hyperparameter_tuning=use_hyperparameter_tuning,
            use_randomized_search=use_randomized_search,
        )
        out.used_suggestion = fallback_suggestion
        out.fallback_used = True
        return out


__all__ = [
    "ModelClassResolution",
    "TrainingPhaseResult",
    "resolve_model_class_multi_agent",
    "repair_generated_model_after_runtime_validation",
    "run_training_phase_multi_agent",
    "run_training_phase_multi_agent_with_fallback",
]
