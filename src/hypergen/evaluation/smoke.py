"""Early production-path feasibility smoke runner."""

from __future__ import annotations

import inspect
import shutil
from dataclasses import asdict
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Annotated

from pydantic import Field, FiniteFloat, PositiveInt

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
from hypergen.generation.hotspot_prompts import (
    HOTSPOT_PROMPT_VERSION,
    HOTSPOT_SCHEMA_VERSION,
    CardCatalogueEntry,
    HotspotGenerationRequest,
    HotspotModelOutput,
    OllamaHotspotGenerator,
    build_hotspot_prompt,
    build_hotspot_response_schema,
)
from hypergen.generation.image_prompts import IMAGE_PROMPT_VERSION, compose_image_prompt
from hypergen.generation.mflux_generator import (
    MfluxGenerationRequest,
    MfluxGenerator,
)
from hypergen.generation.ollama_client import (
    DEFAULT_OLLAMA_MODEL,
    OllamaRuntime,
    OllamaSettings,
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
    fixture_image: Path
    ollama_endpoint: NonEmptyString = "http://localhost:11434"
    ollama_model: NonEmptyString = DEFAULT_OLLAMA_MODEL
    mflux_model: NonEmptyString = "flux2-klein-4b"
    seed: int = 42
    width: PositiveInt = 1024
    height: PositiveInt = 768
    step_count: PositiveInt = 4
    quantization: int | None = None
    coordinate_extent: PositiveInt = 1000
    ollama_timeout_seconds: Annotated[FiniteFloat, Field(gt=0.0)] = 300.0
    ollama_num_predict: PositiveInt = 2048
    ollama_context_length: PositiveInt = 8192


def default_smoke_output_dir() -> Path:
    """Return a unique developer-run output directory."""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    return Path("evals/runs") / f"smoke-{timestamp}"


def default_smoke_fixture() -> Path:
    """Return the tracked smoke image from the repository."""
    return Path("evals/cases/smoke/courtyard.png")


def _write_result(output_dir: Path, result: dict[str, object]) -> Path:
    result_path = output_dir / "smoke-result.json"
    atomic_write_json(result_path, result)
    return result_path


def _collect_hotspot_warnings(result: dict[str, object]) -> list[str]:
    stages = result.get("stages")
    if not isinstance(stages, dict):
        return []
    hotspot_generation = stages.get("hotspot_generation")
    if not isinstance(hotspot_generation, dict):
        return []
    warnings: set[str] = set()
    for phase in ("cold", "warm"):
        record = hotspot_generation.get(phase)
        if isinstance(record, dict):
            warnings.update(record.get("warnings", []))
    return sorted(warnings)


def run_smoke(
    settings: SmokeSettings,
    *,
    ollama_runtime: OllamaRuntime | None = None,
    mflux_generator: MfluxGenerator | None = None,
    environment_provider: EnvironmentProvider = default_environment,
) -> Path:
    """Exercise cold and warm production paths and write structured results."""
    lifecycle = RunLifecycle.create(
        run_dir=settings.output_dir,
        suite="smoke",
        settings=settings.model_dump(mode="json"),
        models={"ollama": settings.ollama_model, "mflux": settings.mflux_model},
        contracts={
            "image_prompt": {
                "version": IMAGE_PROMPT_VERSION,
                "sha256": contract_digest(
                    IMAGE_PROMPT_VERSION,
                    inspect.getsource(compose_image_prompt),
                ),
            },
            "hotspot_prompt": {
                "version": HOTSPOT_PROMPT_VERSION,
                "schema_version": HOTSPOT_SCHEMA_VERSION,
                "sha256": contract_digest(
                    HOTSPOT_PROMPT_VERSION,
                    inspect.getsource(build_hotspot_prompt),
                    HotspotModelOutput.model_json_schema(),
                    inspect.getsource(build_hotspot_response_schema),
                ),
            },
        },
        environment_provider=environment_provider,
    )
    stage = "initializing"
    try:
        runtime = ollama_runtime or OllamaRuntime(
            OllamaSettings(
                endpoint=settings.ollama_endpoint,
                model=settings.ollama_model,
                think=False,
                temperature=0.0,
                request_timeout_seconds=settings.ollama_timeout_seconds,
                num_predict=settings.ollama_num_predict,
                context_length=settings.ollama_context_length,
            )
        )
        mflux = mflux_generator or MfluxGenerator()
        raw_dir = settings.output_dir / "raw"
        raw_dir.mkdir()
        fixture_dir = settings.output_dir / "artifacts" / "sources"
        fixture_dir.mkdir(parents=True)
        fixture_copy = fixture_dir / settings.fixture_image.name
        shutil.copyfile(settings.fixture_image, fixture_copy)
        lifecycle.update_contract(
            "fixture",
            {
                "path": fixture_copy.relative_to(settings.output_dir).as_posix(),
                "sha256": sha256(fixture_copy.read_bytes()).hexdigest(),
            },
        )
        result: dict[str, object] = {
            "result_version": "smoke-result-v2",
            "suite": "smoke",
            "status": "running",
            "ollama_model": settings.ollama_model,
            "mflux_model": settings.mflux_model,
            "settings": {
                "smoke": settings.model_dump(mode="json"),
                "ollama": asdict(runtime.settings),
            },
            "stages": {},
            "warnings": [],
        }
        _write_result(settings.output_dir, result)
    except Exception as error:
        lifecycle.finalize(
            status="failed",
            failure={
                "stage": stage,
                "classification": classify_failure(error),
                "error_type": type(error).__name__,
                "message": str(error),
            },
        )
        raise
    stage = "ollama_diagnostics"
    try:
        lifecycle.set_stage(stage)
        runtime.require_model(capabilities=frozenset({"vision"}))
        lifecycle.complete_stage(stage)
        stage = "ollama_unload"
        lifecycle.set_stage(stage)
        runtime.unload_model()
        lifecycle.complete_stage(stage)
        inputs = ImageGenerationInputs(
            scene_description=(
                "A quiet stone castle courtyard at dusk with an arched wooden gate, "
                "a red travel chest, and a leafy tree."
            ),
            global_style="Restrained storybook ink and watercolor illustration.",
            card_style="Cool twilight shadows with warm lantern light.",
        )
        interaction_description = (
            "The gate leads to the moonlit garden. The travel chest opens the treasure room."
        )
        render_prompt = compose_image_prompt(inputs)
        result["render_prompt"] = render_prompt

        stage = "image_generation_cold"
        lifecycle.set_stage(stage)
        image_cold = mflux.generate(
            MfluxGenerationRequest(
                inputs=inputs,
                render_prompt=render_prompt,
                output_path=settings.output_dir / "generated-cold.png",
                model_identifier=settings.mflux_model,
                seed=settings.seed,
                width=settings.width,
                height=settings.height,
                step_count=settings.step_count,
                quantization=settings.quantization,
            )
        )
        result["stages"]["image_generation"] = {  # type: ignore[index]
            "cold": image_cold.model_dump(mode="json")
        }
        _write_result(settings.output_dir, result)
        lifecycle.complete_stage(stage)
        stage = "image_generation_warm"
        lifecycle.set_stage(stage)
        image_warm = mflux.generate(
            MfluxGenerationRequest(
                inputs=inputs,
                render_prompt=render_prompt,
                output_path=settings.output_dir / "generated-warm.png",
                model_identifier=settings.mflux_model,
                seed=settings.seed,
                width=settings.width,
                height=settings.height,
                step_count=settings.step_count,
                quantization=settings.quantization,
            )
        )
        result["stages"]["image_generation"]["warm"] = image_warm.model_dump(mode="json")  # type: ignore[index]
        _write_result(settings.output_dir, result)
        lifecycle.complete_stage(stage)

        stage = "hotspot_generation_setup"
        lifecycle.set_stage(stage)
        hotspot_request = HotspotGenerationRequest(
            image_path=fixture_copy,
            interaction_description=interaction_description,
            card_catalogue=(
                CardCatalogueEntry(
                    token="C1",
                    name="Moonlit Garden",
                    description="A walled garden beneath the moon.",
                ),
                CardCatalogueEntry(
                    token="C2",
                    name="Treasure Room",
                    description="A small chamber filled with old maps and a brass coffer.",
                ),
            ),
            coordinate_extent=settings.coordinate_extent,
        )
        hotspot_generator = OllamaHotspotGenerator(runtime)
        lifecycle.complete_stage(stage)
        stage = "hotspot_generation_cold"
        lifecycle.set_stage(stage)
        hotspots_cold = hotspot_generator.generate(hotspot_request)
        (raw_dir / "hotspots-cold.json").write_text(hotspots_cold.raw_response, encoding="utf-8")
        result["stages"]["hotspot_generation"] = {  # type: ignore[index]
            "cold": hotspots_cold.model_dump(mode="json")
        }
        _write_result(settings.output_dir, result)
        lifecycle.complete_stage(stage)
        stage = "hotspot_generation_warm"
        lifecycle.set_stage(stage)
        hotspots_warm = hotspot_generator.generate(hotspot_request)
        (raw_dir / "hotspots-warm.json").write_text(hotspots_warm.raw_response, encoding="utf-8")
        result["stages"]["hotspot_generation"]["warm"] = hotspots_warm.model_dump(mode="json")  # type: ignore[index]
        _write_result(settings.output_dir, result)
        lifecycle.complete_stage(stage)
    except Exception as error:
        raw_response = getattr(error, "raw_response", None)
        if isinstance(raw_response, str):
            (raw_dir / f"partial-{stage}.json").write_text(raw_response, encoding="utf-8")
        warnings = _collect_hotspot_warnings(result)
        result.update(
            {
                "status": "failed",
                "stage": stage,
                "error_type": type(error).__name__,
                "message": str(error),
                "failure": {
                    "stage": stage,
                    "classification": classify_failure(error),
                    "error_type": type(error).__name__,
                    "message": str(error),
                },
                "warnings": warnings,
            }
        )
        result_path = _write_result(settings.output_dir, result)
        try:
            render_reports(result_path)
        except Exception as report_error:
            lifecycle.add_warning(f"failure report could not be rendered: {report_error}")
        lifecycle.finalize(
            status="failed",
            failure=result["failure"],  # type: ignore[arg-type]
            warnings=warnings,
        )
        if isinstance(error, GenerationError):
            raise SmokeStageError(stage, error) from error
        raise

    warnings = _collect_hotspot_warnings(result)
    result["status"] = "success"
    result["warnings"] = warnings
    result_path = _write_result(settings.output_dir, result)
    try:
        lifecycle.set_stage("reports")
        render_reports_checked(result_path)
    except ReportRenderingError as error:
        result.update(
            {
                "status": "completed_with_report_failure",
                "stage": "reports",
                "failure": {
                    "stage": "reports",
                    "classification": "report_rendering",
                    "error_type": type(error).__name__,
                    "message": str(error),
                },
            }
        )
        _write_result(settings.output_dir, result)
        lifecycle.finalize(
            status="completed_with_report_failure",
            failure=result["failure"],  # type: ignore[arg-type]
            warnings=warnings,
        )
        raise
    lifecycle.complete_stage("reports")
    lifecycle.finalize(status="success", warnings=warnings)
    return result_path
