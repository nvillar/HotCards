"""Early production-path feasibility smoke runner."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

from pydantic import Field, FiniteFloat, PositiveInt

from hypergen.domain.models import DomainModel, ImageGenerationInputs, NonEmptyString
from hypergen.generation.errors import GenerationError
from hypergen.generation.hotspot_prompts import (
    CardCatalogueEntry,
    HotspotGenerationRequest,
    OllamaHotspotGenerator,
)
from hypergen.generation.image_prompts import OllamaImagePromptDeriver
from hypergen.generation.mflux_generator import (
    MfluxGenerationRequest,
    MfluxGenerator,
)
from hypergen.generation.ollama_client import OllamaRuntime, OllamaSettings


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
    ollama_model: NonEmptyString = "qwen3.5:9b"
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
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return result_path


def run_smoke(
    settings: SmokeSettings,
    *,
    ollama_runtime: OllamaRuntime | None = None,
    mflux_generator: MfluxGenerator | None = None,
) -> Path:
    """Exercise cold and warm production paths and write structured results."""
    settings.output_dir.mkdir(parents=True, exist_ok=False)
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
    stage = "ollama_diagnostics"
    try:
        runtime.require_model(capabilities=frozenset({"vision"}))
        runtime.unload_model()
        inputs = ImageGenerationInputs(
            scene_description=(
                "A quiet stone castle courtyard at dusk with an arched wooden gate, "
                "a red travel chest, and a leafy tree."
            ),
            interaction_description=(
                "The gate leads to the moonlit garden. The travel chest opens the treasure room."
            ),
            stack_art_direction="Restrained storybook ink and watercolor illustration.",
            card_style="Cool twilight shadows with warm lantern light.",
        )
        prompt_deriver = OllamaImagePromptDeriver(runtime)
        stage = "prompt_derivation_cold"
        prompt_cold = prompt_deriver.derive(inputs)
        stage = "prompt_derivation_warm"
        prompt_warm = prompt_deriver.derive(inputs)
        stage = "hotspot_cold_reset"
        runtime.unload_model()

        stage = "image_generation_cold"
        image_cold = mflux.generate(
            MfluxGenerationRequest(
                inputs=inputs,
                derived_prompt=prompt_cold.prompt,
                output_path=settings.output_dir / "generated-cold.png",
                model_identifier=settings.mflux_model,
                seed=settings.seed,
                width=settings.width,
                height=settings.height,
                step_count=settings.step_count,
                quantization=settings.quantization,
            )
        )
        stage = "image_generation_warm"
        image_warm = mflux.generate(
            MfluxGenerationRequest(
                inputs=inputs,
                derived_prompt=prompt_warm.prompt,
                output_path=settings.output_dir / "generated-warm.png",
                model_identifier=settings.mflux_model,
                seed=settings.seed,
                width=settings.width,
                height=settings.height,
                step_count=settings.step_count,
                quantization=settings.quantization,
            )
        )

        stage = "hotspot_generation_setup"
        hotspot_request = HotspotGenerationRequest(
            image_path=settings.fixture_image,
            interaction_description=inputs.interaction_description,
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
        stage = "hotspot_generation_cold"
        hotspots_cold = hotspot_generator.generate(hotspot_request)
        stage = "hotspot_generation_warm"
        hotspots_warm = hotspot_generator.generate(hotspot_request)
    except GenerationError as error:
        _write_result(
            settings.output_dir,
            {
                "status": "failed",
                "stage": stage,
                "error_type": type(error).__name__,
                "message": str(error),
                "settings": {
                    "smoke": settings.model_dump(mode="json"),
                    "ollama": asdict(runtime.settings),
                },
            },
        )
        raise SmokeStageError(stage, error) from error

    warnings = sorted(set(hotspots_cold.warnings + hotspots_warm.warnings))
    return _write_result(
        settings.output_dir,
        {
            "status": "success",
            "settings": {
                "smoke": settings.model_dump(mode="json"),
                "ollama": asdict(runtime.settings),
            },
            "stages": {
                "prompt_derivation": {
                    "cold": prompt_cold.model_dump(mode="json"),
                    "warm": prompt_warm.model_dump(mode="json"),
                },
                "image_generation": {
                    "cold": image_cold.model_dump(mode="json"),
                    "warm": image_warm.model_dump(mode="json"),
                },
                "hotspot_generation": {
                    "cold": hotspots_cold.model_dump(mode="json"),
                    "warm": hotspots_warm.model_dump(mode="json"),
                },
            },
            "warnings": warnings,
        },
    )
