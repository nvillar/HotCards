"""Controlled deterministic-prompt MFLUX model evaluation suite."""

from __future__ import annotations

import inspect
import json
import re
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import Field

from hotcards.domain.image_dimensions import (
    AspectRatio,
    ResolutionTier,
    output_dimensions,
)
from hotcards.domain.models import (
    DomainModel,
    GenerateInputs,
    NonEmptyString,
    PositiveInt,
    PresetOutputSize,
)
from hotcards.evaluation.contracts import SafeCaseId
from hotcards.evaluation.manifest import (
    EnvironmentProvider,
    RunLifecycle,
    atomic_write_json,
    classify_failure,
    contract_digest,
    default_environment,
)
from hotcards.evaluation.reports import render_reports, render_reports_checked
from hotcards.evaluation.results import generate_cold, generation_record
from hotcards.generation.errors import ImageGenerationError
from hotcards.generation.image_generation import (
    DIRECT_GENERATION_PROMPT_VERSION,
    compose_generation_prompt,
)
from hotcards.generation.mflux_generator import (
    MfluxGenerateRequest,
    MfluxGenerator,
)

DEFAULT_MFLUX_MODELS = ("flux2-klein-4b", "flux2-klein-9b")
IMAGE_CASE_VERSION = "image-case-v3"
IMAGE_RESULT_VERSION = "image-result-v2"
HUMAN_RUBRIC_FIELDS = (
    "scene_fidelity",
    "interactive_subject_visibility",
    "hotspot_suitability",
    "composition",
    "style_consistency",
    "absence_of_unwanted_text_or_ui",
)


class ImageEvaluationCase(DomainModel):
    """One version-controlled input and rubric for the image suite."""

    case_version: Literal["image-case-v3"] = IMAGE_CASE_VERSION
    case_id: SafeCaseId
    inputs: GenerateInputs
    required_visual_elements: tuple[NonEmptyString, ...] = Field(min_length=1)
    unwanted_artifacts: tuple[NonEmptyString, ...] = Field(default_factory=tuple)


class ImageEvaluationSettings(DomainModel):
    """Controlled settings shared by all MFLUX comparisons."""

    output_dir: Path
    case_dir: Path = Path("evals/cases/images")
    mflux_models: tuple[NonEmptyString, ...] = DEFAULT_MFLUX_MODELS
    seed: int = 42
    tier: ResolutionTier = ResolutionTier.FULL
    aspect_ratio: AspectRatio = AspectRatio.LANDSCAPE
    step_count: PositiveInt = 4
    quantization: int | None = None


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


def _request(
    *,
    case: ImageEvaluationCase,
    render_prompt: str,
    output_path: Path,
    model: str,
    settings: ImageEvaluationSettings,
) -> MfluxGenerateRequest:
    inputs = case.inputs.model_copy(
        update={"output_size": PresetOutputSize(tier=settings.tier)},
    )
    width, height = output_dimensions(
        settings.tier,
        settings.aspect_ratio,
    )
    return MfluxGenerateRequest(
        inputs=inputs,
        render_prompt=render_prompt,
        output_path=output_path,
        model_identifier=model,
        seed=settings.seed,
        aspect_ratio=settings.aspect_ratio,
        width=width,
        height=height,
        step_count=settings.step_count,
        quantization=settings.quantization,
    )


def _write_result(output_dir: Path, result: dict[str, object]) -> Path:
    result_path = output_dir / "image-results.json"
    atomic_write_json(result_path, result)
    return result_path


def _failure_record(error: BaseException, *, stage: str) -> dict[str, object]:
    return {
        "status": "failed",
        "failure": {
            "stage": stage,
            "classification": classify_failure(error),
            "error_type": type(error).__name__,
            "message": str(error),
        },
    }


def _execute_image_evaluation(
    settings: ImageEvaluationSettings,
    *,
    mflux_factory: MfluxGeneratorFactory,
    lifecycle: RunLifecycle,
) -> Path:
    """Evaluate MFLUX candidates with one deterministic prompt per case."""
    cases = load_image_cases(settings.case_dir)
    lifecycle.update_contract(
        "image_case",
        {
            "version": IMAGE_CASE_VERSION,
            "sha256": contract_digest([case.model_dump(mode="json") for case in cases]),
        },
    )
    image_dir = settings.output_dir / "images" / "mflux-axis"
    image_dir.mkdir(parents=True)
    mflux_results: list[dict[str, object]] = []
    result: dict[str, object] = {
        "result_version": IMAGE_RESULT_VERSION,
        "suite": "images",
        "status": "running",
        "created_at": datetime.now(UTC).isoformat(),
        "settings": settings.model_dump(mode="json"),
        "cases": [case.model_dump(mode="json") for case in cases],
        "mflux_axis": mflux_results,
        "warnings": [],
    }
    _write_result(settings.output_dir, result)

    for case in cases:
        render_prompt = compose_generation_prompt(case.inputs)
        for model in settings.mflux_models:
            stage = f"mflux_axis:{case.case_id}:{model}"
            lifecycle.set_stage(stage)
            generator = mflux_factory()
            case_model_dir = image_dir / case.case_id / _safe_name(model)
            case_model_dir.mkdir(parents=True, exist_ok=True)
            generations: dict[str, dict[str, object]] = {}
            for phase in ("cold", "warm"):
                if phase == "warm" and generations["cold"]["status"] != "success":
                    generations[phase] = {
                        "status": "skipped",
                        "failure": {
                            "stage": f"{stage}:warm",
                            "classification": "dependency",
                            "message": "Warm measurement requires a successful cold invocation.",
                        },
                    }
                    continue
                try:
                    request = _request(
                        case=case,
                        render_prompt=render_prompt,
                        output_path=case_model_dir / f"{phase}.png",
                        model=model,
                        settings=settings,
                    )
                    generated = (
                        generate_cold(generator, request)
                        if phase == "cold"
                        else generator.generate(request)
                    )
                    generations[phase] = generation_record(
                        generated,
                        settings.output_dir,
                    )
                except Exception as error:
                    generations[phase] = _failure_record(
                        error,
                        stage=f"{stage}:{phase}",
                    )
            mflux_results.append(
                {
                    "case_id": case.case_id,
                    "model": model,
                    "prompt_version": DIRECT_GENERATION_PROMPT_VERSION,
                    "render_prompt": render_prompt,
                    "required_visual_elements": case.required_visual_elements,
                    "unwanted_artifacts": case.unwanted_artifacts,
                    "cold": generations["cold"],
                    "warm": generations["warm"],
                    "rubric": {field: None for field in HUMAN_RUBRIC_FIELDS},
                }
            )
            _write_result(settings.output_dir, result)
            if all(generations[phase]["status"] == "success" for phase in ("cold", "warm")):
                lifecycle.complete_stage(stage)

    failures = [
        item[phase]
        for item in mflux_results
        for phase in ("cold", "warm")
        if item[phase].get("status") != "success"  # type: ignore[union-attr]
    ]
    successes = len(mflux_results) * 2 - len(failures)
    result["status"] = (
        "failed" if successes == 0 else "completed_with_failures" if failures else "success"
    )
    result_path = _write_result(settings.output_dir, result)
    lifecycle.set_stage("reports")
    render_reports_checked(result_path)
    lifecycle.complete_stage("reports")
    return result_path


def run_image_evaluation(
    settings: ImageEvaluationSettings,
    *,
    mflux_factory: MfluxGeneratorFactory = MfluxGenerator,
    environment_provider: EnvironmentProvider = default_environment,
) -> Path:
    """Run deterministic image evaluation with an exception-safe manifest."""
    lifecycle = RunLifecycle.create(
        run_dir=settings.output_dir,
        suite="images",
        settings=settings.model_dump(mode="json"),
        models={"mflux_candidates": list(settings.mflux_models)},
        contracts={
            "generation_prompt": {
                "version": DIRECT_GENERATION_PROMPT_VERSION,
                "sha256": contract_digest(
                    DIRECT_GENERATION_PROMPT_VERSION,
                    inspect.getsource(compose_generation_prompt),
                ),
            },
            "image_case": {"version": IMAGE_CASE_VERSION},
        },
        environment_provider=environment_provider,
    )
    try:
        result_path = _execute_image_evaluation(
            settings,
            mflux_factory=mflux_factory,
            lifecycle=lifecycle,
        )
    except Exception as error:
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
        lifecycle.finalize(status=final_status, failure=failure)
        raise

    completed_result = json.loads(result_path.read_text(encoding="utf-8"))
    stage_failures = [
        {
            "axis": "mflux",
            "case_id": item["case_id"],
            "model": item["model"],
            "phase": phase,
            **item[phase]["failure"],
        }
        for item in completed_result["mflux_axis"]
        for phase in ("cold", "warm")
        if item[phase].get("status") != "success"
    ]
    lifecycle.finalize(
        status=completed_result["status"],
        failure=(
            {"classification": "stage_failures", "stages": stage_failures}
            if stage_failures
            else None
        ),
    )
    if completed_result["status"] == "failed":
        raise ImageGenerationError(
            f"Image evaluation produced no images; diagnostics were retained at {result_path}"
        )
    return result_path
