"""Early production-path image generation smoke runner."""

from __future__ import annotations

import inspect
from datetime import UTC, datetime
from pathlib import Path

from pydantic import PositiveInt

from hypergen.domain.models import DomainModel, ImageGenerationInputs, NonEmptyString
from hypergen.evaluation.manifest import (
    EnvironmentProvider,
    RunLifecycle,
    atomic_write_json,
    classify_failure,
    contract_digest,
    default_environment,
)
from hypergen.evaluation.reports import (
    ReportRenderingError,
    render_reports,
    render_reports_checked,
)
from hypergen.generation.errors import GenerationError
from hypergen.generation.image_prompts import IMAGE_PROMPT_VERSION, compose_image_prompt
from hypergen.generation.mflux_generator import (
    MfluxGenerationRequest,
    MfluxGenerator,
)


class SmokeStageError(GenerationError):
    """An expected generation failure annotated with its smoke stage."""

    def __init__(self, stage: str, cause: GenerationError) -> None:
        self.stage = stage
        self.cause = cause
        super().__init__(f"{stage} failed: {cause}")


class SmokeSettings(DomainModel):
    """Explicit effective settings for one smoke run."""

    output_dir: Path
    mflux_model: NonEmptyString = "flux2-klein-4b"
    seed: int = 42
    width: PositiveInt = 1024
    height: PositiveInt = 768
    step_count: PositiveInt = 4
    quantization: int | None = None


def default_smoke_output_dir() -> Path:
    """Return a unique developer-run output directory."""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    return Path("evals/runs") / f"smoke-{timestamp}"


def _write_result(output_dir: Path, result: dict[str, object]) -> Path:
    result_path = output_dir / "smoke-result.json"
    atomic_write_json(result_path, result)
    return result_path


def run_smoke(
    settings: SmokeSettings,
    *,
    mflux_generator: MfluxGenerator | None = None,
    environment_provider: EnvironmentProvider = default_environment,
) -> Path:
    """Exercise cold and warm production image paths and retain results."""
    lifecycle = RunLifecycle.create(
        run_dir=settings.output_dir,
        suite="smoke",
        settings=settings.model_dump(mode="json"),
        models={"mflux": settings.mflux_model},
        contracts={
            "image_prompt": {
                "version": IMAGE_PROMPT_VERSION,
                "sha256": contract_digest(
                    IMAGE_PROMPT_VERSION,
                    inspect.getsource(compose_image_prompt),
                ),
            },
        },
        environment_provider=environment_provider,
    )
    result: dict[str, object] = {
        "result_version": "smoke-result-v3",
        "suite": "smoke",
        "status": "running",
        "mflux_model": settings.mflux_model,
        "settings": {"smoke": settings.model_dump(mode="json")},
        "stages": {},
        "warnings": [],
    }
    result_path = _write_result(settings.output_dir, result)
    stage = "image_generation_setup"
    try:
        mflux = mflux_generator or MfluxGenerator()
        prompt = (
            "A quiet stone castle courtyard at dusk with an arched wooden gate, "
            "a red travel chest, and a leafy tree. "
            "Restrained storybook ink and watercolor illustration with "
            "cool twilight shadows and warm lantern light."
        )
        inputs = ImageGenerationInputs(
            description=prompt,
            image_prompt=prompt,
        )
        render_prompt = compose_image_prompt(inputs)
        result["render_prompt"] = render_prompt
        for phase in ("cold", "warm"):
            stage = f"image_generation_{phase}"
            lifecycle.set_stage(stage)
            generated = mflux.generate(
                MfluxGenerationRequest(
                    inputs=inputs,
                    render_prompt=render_prompt,
                    output_path=settings.output_dir / f"generated-{phase}.png",
                    model_identifier=settings.mflux_model,
                    seed=settings.seed,
                    width=settings.width,
                    height=settings.height,
                    step_count=settings.step_count,
                    quantization=settings.quantization,
                )
            )
            image_stage = result["stages"].setdefault("image_generation", {})  # type: ignore[union-attr]
            image_stage[phase] = generated.model_dump(mode="json")
            result_path = _write_result(settings.output_dir, result)
            lifecycle.complete_stage(stage)
    except Exception as error:
        failure = {
            "stage": stage,
            "classification": classify_failure(error),
            "error_type": type(error).__name__,
            "message": str(error),
        }
        result.update(
            {
                "status": "failed",
                "stage": stage,
                "error_type": type(error).__name__,
                "message": str(error),
                "failure": failure,
            }
        )
        result_path = _write_result(settings.output_dir, result)
        try:
            render_reports(result_path)
        except Exception as report_error:
            lifecycle.add_warning(f"failure report could not be rendered: {report_error}")
        lifecycle.finalize(status="failed", failure=failure)
        if isinstance(error, GenerationError):
            raise SmokeStageError(stage, error) from error
        raise

    result["status"] = "success"
    result_path = _write_result(settings.output_dir, result)
    try:
        lifecycle.set_stage("reports")
        render_reports_checked(result_path)
        lifecycle.complete_stage("reports")
    except ReportRenderingError as error:
        result["status"] = "failed"
        result["failure"] = {
            "stage": "reports",
            "classification": classify_failure(error),
            "error_type": type(error).__name__,
            "message": str(error),
        }
        _write_result(settings.output_dir, result)
        lifecycle.finalize(status="failed", failure=result["failure"])  # type: ignore[arg-type]
        raise
    lifecycle.finalize(status="success")
    return result_path


__all__ = [
    "SmokeSettings",
    "SmokeStageError",
    "default_smoke_output_dir",
    "run_smoke",
]
