"""Controlled image-prompt and MFLUX model evaluation suite."""

from __future__ import annotations

import inspect
import json
import re
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import Field

from hypergen.domain.models import (
    DomainModel,
    ImageGenerationInputs,
    NonEmptyString,
    PositiveInt,
)
from hypergen.evaluation.contracts import SafeCaseId
from hypergen.evaluation.manifest import (
    EnvironmentProvider,
    RunLifecycle,
    atomic_write_json,
    classify_failure,
    contract_digest,
    default_environment,
)
from hypergen.evaluation.reports import render_reports, render_reports_checked
from hypergen.generation.errors import ModelResponseError
from hypergen.generation.image_prompts import (
    IMAGE_PROMPT_VERSION,
    DerivedRenderPrompt,
    OllamaImagePromptDeriver,
    RenderPromptModelOutput,
    build_image_prompt_request,
)
from hypergen.generation.mflux_generator import (
    MfluxGenerationRequest,
    MfluxGenerationResult,
    MfluxGenerator,
)
from hypergen.generation.ollama_client import OllamaRuntime, OllamaSettings

DEFAULT_OLLAMA_MODELS = ("qwen3.5:4b", "qwen3.5:9b", "qwen3.6:35b")
DEFAULT_MFLUX_MODELS = ("flux2-klein-4b", "flux2-klein-9b")
IMAGE_CASE_VERSION = "image-case-v1"
IMAGE_RESULT_VERSION = "image-result-v1"
HUMAN_RUBRIC_FIELDS = (
    "scene_fidelity",
    "interactive_subject_visibility",
    "hotspot_suitability",
    "composition",
    "style_consistency",
    "absence_of_unwanted_text_or_ui",
)
PROMPT_RUBRIC_FIELDS = (
    "scene_preservation",
    "interaction_to_visual_translation",
    "style_adherence",
    "subject_distinctness",
    "absence_of_labels_or_ui_instructions",
)


class ImageEvaluationCase(DomainModel):
    """One version-controlled input and rubric for the image suite."""

    case_version: Literal["image-case-v1"] = IMAGE_CASE_VERSION
    case_id: SafeCaseId
    inputs: ImageGenerationInputs
    fixed_render_prompt: NonEmptyString
    required_visual_elements: tuple[NonEmptyString, ...] = Field(min_length=1)
    unwanted_artifacts: tuple[NonEmptyString, ...] = Field(default_factory=tuple)


class ImageEvaluationSettings(DomainModel):
    """Controlled settings shared by all image-suite comparisons."""

    output_dir: Path
    case_dir: Path = Path("evals/cases/images")
    ollama_endpoint: NonEmptyString = "http://localhost:11434"
    ollama_models: tuple[NonEmptyString, ...] = DEFAULT_OLLAMA_MODELS
    mflux_models: tuple[NonEmptyString, ...] = DEFAULT_MFLUX_MODELS
    downstream_mflux_model: NonEmptyString = "flux2-klein-4b"
    seed: int = 42
    width: PositiveInt = 1024
    height: PositiveInt = 768
    step_count: PositiveInt = 4
    quantization: int | None = None
    ollama_timeout_seconds: float = Field(default=300.0, gt=0.0, allow_inf_nan=False)
    ollama_num_predict: PositiveInt = 2048
    ollama_context_length: PositiveInt = 8192


OllamaRuntimeFactory = Callable[[OllamaSettings], OllamaRuntime]
MfluxGeneratorFactory = Callable[[], MfluxGenerator]


def default_image_output_dir() -> Path:
    """Return a unique output directory for a live image-suite run."""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    return Path("evals/runs") / f"images-{timestamp}"


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value)


def load_image_cases(case_dir: Path) -> tuple[ImageEvaluationCase, ...]:
    """Load deterministic tracked cases in filename order."""
    paths = sorted(case_dir.glob("*.json"))
    if not paths:
        raise ValueError(f"no image evaluation cases found in {case_dir}")
    cases = tuple(
        ImageEvaluationCase.model_validate_json(path.read_text(encoding="utf-8")) for path in paths
    )
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("image evaluation case IDs must be unique")
    return cases


def _prompt_metrics(prompt: DerivedRenderPrompt) -> dict[str, float | int | None]:
    inference_duration_ns = None
    if prompt.total_duration_ns is not None:
        inference_duration_ns = prompt.total_duration_ns - (prompt.load_duration_ns or 0)
    return {
        "elapsed_seconds": prompt.duration_seconds,
        "total_duration_ns": prompt.total_duration_ns,
        "load_duration_ns": prompt.load_duration_ns,
        "inference_duration_ns": inference_duration_ns,
    }


def _generation_record(result: MfluxGenerationResult, output_dir: Path) -> dict[str, object]:
    return {
        "status": "success",
        "artifact_path": result.output_path.relative_to(output_dir).as_posix(),
        "load_duration_seconds": result.load_duration_seconds,
        "inference_duration_seconds": result.generation_duration_seconds,
        "serialization_duration_seconds": result.serialization_duration_seconds,
        "total_duration_seconds": result.metadata.duration_seconds,
        "metadata": result.metadata.model_dump(mode="json"),
    }


def _request(
    *,
    case: ImageEvaluationCase,
    prompt: str,
    output_path: Path,
    model: str,
    settings: ImageEvaluationSettings,
) -> MfluxGenerationRequest:
    return MfluxGenerationRequest(
        inputs=case.inputs,
        derived_prompt=prompt,
        output_path=output_path,
        model_identifier=model,
        seed=settings.seed,
        width=settings.width,
        height=settings.height,
        step_count=settings.step_count,
        quantization=settings.quantization,
    )


def _write_result(output_dir: Path, result: dict[str, object]) -> Path:
    result_path = output_dir / "image-results.json"
    atomic_write_json(result_path, result)
    return result_path


def _failure_record(error: BaseException, *, stage: str) -> dict[str, object]:
    metadata = error.response_metadata if isinstance(error, ModelResponseError) else {}
    return {
        "status": "failed",
        "failure": {
            "stage": stage,
            "classification": classify_failure(error),
            "error_type": type(error).__name__,
            "message": str(error),
        },
        "elapsed_seconds": metadata.get("elapsed_seconds"),
        "prompt_eval_count": metadata.get("prompt_eval_count"),
        "eval_count": metadata.get("eval_count"),
        "done_reason": metadata.get("done_reason"),
    }


def _blocked_record(*, stage: str, message: str) -> dict[str, object]:
    return {
        "status": "failed",
        "failure": {
            "stage": stage,
            "classification": "blocked_by_prior_stage",
            "error_type": "BlockedStage",
            "message": message,
        },
    }


def _prompt_failure_metrics(record: dict[str, object]) -> dict[str, object]:
    return {
        "elapsed_seconds": record.get("elapsed_seconds"),
        "total_duration_ns": None,
        "load_duration_ns": None,
        "inference_duration_ns": None,
    }


def _execute_image_evaluation(
    settings: ImageEvaluationSettings,
    *,
    runtime_factory: OllamaRuntimeFactory,
    mflux_factory: MfluxGeneratorFactory,
    lifecycle: RunLifecycle,
) -> Path:
    """Execute separate comparison axes inside an initialized run."""
    cases = load_image_cases(settings.case_dir)
    lifecycle.update_contract(
        "image_case",
        {
            "version": IMAGE_CASE_VERSION,
            "sha256": contract_digest([case.model_dump(mode="json") for case in cases]),
        },
    )
    raw_dir = settings.output_dir / "raw" / "prompts"
    prompt_image_dir = settings.output_dir / "images" / "prompt-axis"
    mflux_image_dir = settings.output_dir / "images" / "mflux-axis"
    for directory in (raw_dir, prompt_image_dir, mflux_image_dir):
        directory.mkdir(parents=True)

    prompt_results: list[dict[str, object]] = []
    derived_prompts: dict[tuple[str, str], DerivedRenderPrompt] = {}
    ollama_settings_by_model: dict[str, object] = {}
    downstream_results: list[dict[str, object]] = []
    mflux_results: list[dict[str, object]] = []
    result: dict[str, object] = {
        "result_version": IMAGE_RESULT_VERSION,
        "suite": "images",
        "status": "running",
        "created_at": datetime.now(UTC).isoformat(),
        "settings": settings.model_dump(mode="json"),
        "ollama_effective_settings": ollama_settings_by_model,
        "cases": [case.model_dump(mode="json") for case in cases],
        "prompt_axis": prompt_results,
        "prompt_downstream_axis": downstream_results,
        "mflux_axis": mflux_results,
        "warnings": [],
    }
    _write_result(settings.output_dir, result)
    for model in settings.ollama_models:
        lifecycle.set_stage(f"prompt_axis:{model}:setup")
        runtime = runtime_factory(
            OllamaSettings(
                endpoint=settings.ollama_endpoint,
                model=model,
                think=False,
                temperature=0.0,
                request_timeout_seconds=settings.ollama_timeout_seconds,
                num_predict=settings.ollama_num_predict,
                context_length=settings.ollama_context_length,
            )
        )
        ollama_settings_by_model[model] = asdict(runtime.settings)
        try:
            runtime.require_model()
        except Exception as error:
            failure = _failure_record(error, stage=f"prompt_axis:{model}:setup")
            for case in cases:
                prompt_results.append(
                    {
                        "case_id": case.case_id,
                        "model": model,
                        "prompt_version": IMAGE_PROMPT_VERSION,
                        "source_inputs": case.inputs.model_dump(mode="json"),
                        "cold": failure,
                        "warm": failure,
                        "cold_metrics": _prompt_failure_metrics(failure),
                        "warm_metrics": _prompt_failure_metrics(failure),
                        "rubric": {field: None for field in PROMPT_RUBRIC_FIELDS},
                    }
                )
            _write_result(settings.output_dir, result)
            continue
        deriver = OllamaImagePromptDeriver(runtime)
        for case in cases:
            lifecycle.set_stage(f"prompt_axis:{case.case_id}:{model}")
            model_name = _safe_name(model)
            case_raw_dir = raw_dir / case.case_id
            case_raw_dir.mkdir(exist_ok=True)
            phase_results: dict[str, DerivedRenderPrompt | dict[str, object]] = {}
            try:
                runtime.unload_model()
            except Exception as error:
                phase_results["cold"] = _failure_record(
                    error,
                    stage=f"prompt_axis:{case.case_id}:{model}:cold_reset",
                )
            for phase in ("cold", "warm"):
                if phase in phase_results:
                    continue
                try:
                    prompt = deriver.derive(case.inputs)
                except Exception as error:
                    raw_response = (
                        error.raw_response if isinstance(error, ModelResponseError) else None
                    )
                    if raw_response is not None:
                        (case_raw_dir / f"partial-{model_name}-{phase}.json").write_text(
                            raw_response,
                            encoding="utf-8",
                        )
                    phase_results[phase] = _failure_record(
                        error,
                        stage=f"prompt_axis:{case.case_id}:{model}:{phase}",
                    )
                else:
                    (case_raw_dir / f"{model_name}-{phase}.json").write_text(
                        prompt.raw_response,
                        encoding="utf-8",
                    )
                    phase_results[phase] = prompt
            cold = phase_results["cold"]
            warm = phase_results["warm"]
            usable_prompt = next(
                (prompt for prompt in (cold, warm) if isinstance(prompt, DerivedRenderPrompt)),
                None,
            )
            if usable_prompt is not None:
                derived_prompts[(case.case_id, model)] = usable_prompt
            prompt_results.append(
                {
                    "case_id": case.case_id,
                    "model": model,
                    "prompt_version": IMAGE_PROMPT_VERSION,
                    "source_inputs": case.inputs.model_dump(mode="json"),
                    "cold": (
                        cold.model_dump(mode="json", exclude={"raw_response"})
                        if isinstance(cold, DerivedRenderPrompt)
                        else cold
                    ),
                    "warm": (
                        warm.model_dump(mode="json", exclude={"raw_response"})
                        if isinstance(warm, DerivedRenderPrompt)
                        else warm
                    ),
                    "cold_metrics": (
                        _prompt_metrics(cold)
                        if isinstance(cold, DerivedRenderPrompt)
                        else _prompt_failure_metrics(cold)
                    ),
                    "warm_metrics": (
                        _prompt_metrics(warm)
                        if isinstance(warm, DerivedRenderPrompt)
                        else _prompt_failure_metrics(warm)
                    ),
                    "rubric": {field: None for field in PROMPT_RUBRIC_FIELDS},
                }
            )
            _write_result(settings.output_dir, result)
            if all(
                isinstance(phase_results[phase], DerivedRenderPrompt) for phase in ("cold", "warm")
            ):
                lifecycle.complete_stage(f"prompt_axis:{case.case_id}:{model}")

    prompt_axis_generator = mflux_factory()
    for case in cases:
        for model in settings.ollama_models:
            lifecycle.set_stage(f"prompt_downstream:{case.case_id}:{model}")
            prompt = derived_prompts.get((case.case_id, model))
            if prompt is None:
                downstream_results.append(
                    {
                        "case_id": case.case_id,
                        "ollama_model": model,
                        "mflux_model": settings.downstream_mflux_model,
                        "required_visual_elements": case.required_visual_elements,
                        "unwanted_artifacts": case.unwanted_artifacts,
                        "generation": _blocked_record(
                            stage=f"prompt_downstream:{case.case_id}:{model}",
                            message="No valid derived prompt was available.",
                        ),
                        "rubric": {field: None for field in HUMAN_RUBRIC_FIELDS},
                    }
                )
                _write_result(settings.output_dir, result)
                continue
            output_path = prompt_image_dir / case.case_id / f"{_safe_name(model)}.png"
            output_path.parent.mkdir(exist_ok=True)
            try:
                generated = prompt_axis_generator.generate(
                    _request(
                        case=case,
                        prompt=prompt.prompt,
                        output_path=output_path,
                        model=settings.downstream_mflux_model,
                        settings=settings,
                    )
                )
                generation = _generation_record(generated, settings.output_dir)
            except Exception as error:
                generation = _failure_record(
                    error,
                    stage=f"prompt_downstream:{case.case_id}:{model}",
                )
            downstream_results.append(
                {
                    "case_id": case.case_id,
                    "ollama_model": model,
                    "mflux_model": settings.downstream_mflux_model,
                    "required_visual_elements": case.required_visual_elements,
                    "unwanted_artifacts": case.unwanted_artifacts,
                    "generation": generation,
                    "rubric": {field: None for field in HUMAN_RUBRIC_FIELDS},
                }
            )
            _write_result(settings.output_dir, result)
            if generation["status"] == "success":
                lifecycle.complete_stage(f"prompt_downstream:{case.case_id}:{model}")

    for model in settings.mflux_models:
        for case in cases:
            lifecycle.set_stage(f"mflux_axis:{case.case_id}:{model}")
            mflux_axis_generator = mflux_factory()
            case_model_dir = mflux_image_dir / case.case_id / _safe_name(model)
            case_model_dir.mkdir(parents=True, exist_ok=True)
            generations: dict[str, dict[str, object]] = {}
            for phase in ("cold", "warm"):
                try:
                    generated = mflux_axis_generator.generate(
                        _request(
                            case=case,
                            prompt=case.fixed_render_prompt,
                            output_path=case_model_dir / f"{phase}.png",
                            model=model,
                            settings=settings,
                        )
                    )
                    generations[phase] = _generation_record(
                        generated,
                        settings.output_dir,
                    )
                except Exception as error:
                    generations[phase] = _failure_record(
                        error,
                        stage=f"mflux_axis:{case.case_id}:{model}:{phase}",
                    )
            mflux_results.append(
                {
                    "case_id": case.case_id,
                    "model": model,
                    "fixed_prompt": case.fixed_render_prompt,
                    "required_visual_elements": case.required_visual_elements,
                    "unwanted_artifacts": case.unwanted_artifacts,
                    "cold": generations["cold"],
                    "warm": generations["warm"],
                    "rubric": {field: None for field in HUMAN_RUBRIC_FIELDS},
                }
            )
            _write_result(settings.output_dir, result)
            if all(generations[phase]["status"] == "success" for phase in ("cold", "warm")):
                lifecycle.complete_stage(f"mflux_axis:{case.case_id}:{model}")
    failures = [
        record
        for item in prompt_results
        for record in (item["cold"], item["warm"])
        if record.get("status") == "failed"  # type: ignore[union-attr]
    ]
    failures.extend(
        item["generation"]
        for item in downstream_results
        if item["generation"].get("status") == "failed"  # type: ignore[union-attr]
    )
    failures.extend(
        item[phase]
        for item in mflux_results
        for phase in ("cold", "warm")
        if item[phase].get("status") == "failed"  # type: ignore[union-attr]
    )
    result["status"] = "completed_with_failures" if failures else "success"
    result_path = _write_result(settings.output_dir, result)
    lifecycle.set_stage("reports")
    render_reports_checked(result_path)
    lifecycle.complete_stage("reports")
    return result_path


def run_image_evaluation(
    settings: ImageEvaluationSettings,
    *,
    runtime_factory: OllamaRuntimeFactory = OllamaRuntime,
    mflux_factory: MfluxGeneratorFactory = MfluxGenerator,
    environment_provider: EnvironmentProvider = default_environment,
) -> Path:
    """Run image evaluation with an exception-safe immutable run manifest."""
    lifecycle = RunLifecycle.create(
        run_dir=settings.output_dir,
        suite="images",
        settings=settings.model_dump(mode="json"),
        models={
            "ollama_candidates": list(settings.ollama_models),
            "mflux_candidates": list(settings.mflux_models),
            "fixed_downstream_mflux": settings.downstream_mflux_model,
        },
        contracts={
            "image_prompt": {
                "version": IMAGE_PROMPT_VERSION,
                "sha256": contract_digest(
                    IMAGE_PROMPT_VERSION,
                    inspect.getsource(build_image_prompt_request),
                    RenderPromptModelOutput.model_json_schema(),
                ),
            },
            "image_case": {"version": IMAGE_CASE_VERSION},
        },
        environment_provider=environment_provider,
    )
    try:
        result_path = _execute_image_evaluation(
            settings,
            runtime_factory=runtime_factory,
            mflux_factory=mflux_factory,
            lifecycle=lifecycle,
        )
    except Exception as error:
        if isinstance(error, ModelResponseError) and error.raw_response is not None:
            partial = settings.output_dir / "raw" / "partial-response.json"
            partial.parent.mkdir(parents=True, exist_ok=True)
            partial.write_text(error.raw_response, encoding="utf-8")
        result_path = settings.output_dir / "image-results.json"
        report_failure = lifecycle.manifest["stage"] == "reports"
        failure = {
            "stage": lifecycle.manifest["stage"],
            "classification": ("report_rendering" if report_failure else classify_failure(error)),
            "error_type": type(error).__name__,
            "message": str(error),
        }
        final_status = "completed_with_report_failure" if report_failure else "failed"
        if result_path.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["status"] = final_status
            result["failure"] = failure
            _write_result(settings.output_dir, result)
            if not report_failure:
                try:
                    render_reports(result_path)
                except Exception as report_error:
                    lifecycle.add_warning(f"failure report could not be rendered: {report_error}")
        lifecycle.finalize(
            status=final_status,
            failure=failure,
        )
        raise
    completed_result = json.loads(result_path.read_text(encoding="utf-8"))
    status = completed_result["status"]
    stage_failures = [
        {
            "axis": "prompt",
            "case_id": item["case_id"],
            "model": item["model"],
            "phase": phase,
            **item[phase]["failure"],
        }
        for item in completed_result["prompt_axis"]
        for phase in ("cold", "warm")
        if item[phase].get("status") == "failed"
    ]
    stage_failures.extend(
        {
            "axis": "prompt_downstream",
            "case_id": item["case_id"],
            "model": item["ollama_model"],
            **item["generation"]["failure"],
        }
        for item in completed_result["prompt_downstream_axis"]
        if item["generation"].get("status") == "failed"
    )
    stage_failures.extend(
        {
            "axis": "mflux",
            "case_id": item["case_id"],
            "model": item["model"],
            "phase": phase,
            **item[phase]["failure"],
        }
        for item in completed_result["mflux_axis"]
        for phase in ("cold", "warm")
        if item[phase].get("status") == "failed"
    )
    lifecycle.finalize(
        status=status,
        failure=(
            {"classification": "stage_failures", "stages": stage_failures}
            if stage_failures
            else None
        ),
    )
    return result_path
