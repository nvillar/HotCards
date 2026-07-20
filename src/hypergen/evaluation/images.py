"""Controlled image-prompt and MFLUX model evaluation suite."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, StringConstraints

from hypergen.domain.models import (
    DomainModel,
    ImageGenerationInputs,
    NonEmptyString,
    PositiveInt,
)
from hypergen.generation.image_prompts import (
    IMAGE_PROMPT_VERSION,
    DerivedRenderPrompt,
    OllamaImagePromptDeriver,
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
SafeCaseId = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        pattern=r"^[a-z0-9][a-z0-9._-]*$",
    ),
]
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


def _generation_record(result: MfluxGenerationResult) -> dict[str, object]:
    return {
        "artifact_path": result.output_path.as_posix(),
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


def run_image_evaluation(
    settings: ImageEvaluationSettings,
    *,
    runtime_factory: OllamaRuntimeFactory = OllamaRuntime,
    mflux_factory: MfluxGeneratorFactory = MfluxGenerator,
) -> Path:
    """Run separate Ollama-prompt and MFLUX-model comparison axes."""
    cases = load_image_cases(settings.case_dir)
    settings.output_dir.mkdir(parents=True, exist_ok=False)
    raw_dir = settings.output_dir / "raw" / "prompts"
    prompt_image_dir = settings.output_dir / "images" / "prompt-axis"
    mflux_image_dir = settings.output_dir / "images" / "mflux-axis"
    for directory in (raw_dir, prompt_image_dir, mflux_image_dir):
        directory.mkdir(parents=True)

    prompt_results: list[dict[str, object]] = []
    derived_prompts: dict[tuple[str, str], DerivedRenderPrompt] = {}
    ollama_settings_by_model: dict[str, object] = {}
    for model in settings.ollama_models:
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
        runtime.require_model()
        deriver = OllamaImagePromptDeriver(runtime)
        for case in cases:
            runtime.unload_model()
            cold = deriver.derive(case.inputs)
            warm = deriver.derive(case.inputs)
            derived_prompts[(case.case_id, model)] = cold
            model_name = _safe_name(model)
            case_raw_dir = raw_dir / case.case_id
            case_raw_dir.mkdir(exist_ok=True)
            (case_raw_dir / f"{model_name}-cold.json").write_text(
                cold.raw_response,
                encoding="utf-8",
            )
            (case_raw_dir / f"{model_name}-warm.json").write_text(
                warm.raw_response,
                encoding="utf-8",
            )
            prompt_results.append(
                {
                    "case_id": case.case_id,
                    "model": model,
                    "prompt_version": IMAGE_PROMPT_VERSION,
                    "source_inputs": case.inputs.model_dump(mode="json"),
                    "cold": cold.model_dump(mode="json", exclude={"raw_response"}),
                    "warm": warm.model_dump(mode="json", exclude={"raw_response"}),
                    "cold_metrics": _prompt_metrics(cold),
                    "warm_metrics": _prompt_metrics(warm),
                    "rubric": {field: None for field in PROMPT_RUBRIC_FIELDS},
                }
            )

    prompt_axis_generator = mflux_factory()
    downstream_results: list[dict[str, object]] = []
    for case in cases:
        for model in settings.ollama_models:
            prompt = derived_prompts[(case.case_id, model)]
            output_path = prompt_image_dir / case.case_id / f"{_safe_name(model)}.png"
            output_path.parent.mkdir(exist_ok=True)
            generated = prompt_axis_generator.generate(
                _request(
                    case=case,
                    prompt=prompt.prompt,
                    output_path=output_path,
                    model=settings.downstream_mflux_model,
                    settings=settings,
                )
            )
            downstream_results.append(
                {
                    "case_id": case.case_id,
                    "ollama_model": model,
                    "mflux_model": settings.downstream_mflux_model,
                    "required_visual_elements": case.required_visual_elements,
                    "unwanted_artifacts": case.unwanted_artifacts,
                    "generation": _generation_record(generated),
                    "rubric": {field: None for field in HUMAN_RUBRIC_FIELDS},
                }
            )

    mflux_results: list[dict[str, object]] = []
    for model in settings.mflux_models:
        for case in cases:
            mflux_axis_generator = mflux_factory()
            case_model_dir = mflux_image_dir / case.case_id / _safe_name(model)
            case_model_dir.mkdir(parents=True, exist_ok=True)
            cold = mflux_axis_generator.generate(
                _request(
                    case=case,
                    prompt=case.fixed_render_prompt,
                    output_path=case_model_dir / "cold.png",
                    model=model,
                    settings=settings,
                )
            )
            warm = mflux_axis_generator.generate(
                _request(
                    case=case,
                    prompt=case.fixed_render_prompt,
                    output_path=case_model_dir / "warm.png",
                    model=model,
                    settings=settings,
                )
            )
            mflux_results.append(
                {
                    "case_id": case.case_id,
                    "model": model,
                    "fixed_prompt": case.fixed_render_prompt,
                    "required_visual_elements": case.required_visual_elements,
                    "unwanted_artifacts": case.unwanted_artifacts,
                    "cold": _generation_record(cold),
                    "warm": _generation_record(warm),
                    "rubric": {field: None for field in HUMAN_RUBRIC_FIELDS},
                }
            )

    result = {
        "result_version": IMAGE_RESULT_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "settings": settings.model_dump(mode="json"),
        "ollama_effective_settings": ollama_settings_by_model,
        "cases": [case.model_dump(mode="json") for case in cases],
        "prompt_axis": prompt_results,
        "prompt_downstream_axis": downstream_results,
        "mflux_axis": mflux_results,
    }
    result_path = settings.output_dir / "image-results.json"
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return result_path
