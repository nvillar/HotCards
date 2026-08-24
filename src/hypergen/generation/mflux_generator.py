"""In-process MFLUX image-generation adapter."""

from __future__ import annotations

import gc
from collections.abc import Callable
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter
from typing import Protocol

from PIL import Image, UnidentifiedImageError
from pydantic import Field, FiniteFloat, model_validator

from hypergen.domain.models import (
    DomainModel,
    ImageGenerationInputs,
    ImageGenerationMetadata,
    NonEmptyString,
    PositiveInt,
)
from hypergen.generation.errors import ImageGenerationError, ModelLoadError


class MfluxModelProtocol(Protocol):
    """Subset of the MFLUX model used by HyperGen."""

    def generate_image(
        self,
        *,
        seed: int,
        prompt: str,
        num_inference_steps: int,
        height: int,
        width: int,
        guidance: float,
        scheduler: str,
    ) -> GeneratedImageProtocol: ...


class MfluxEditModelProtocol(Protocol):
    """Subset of the MFLUX edit model used for image references."""

    def generate_image(
        self,
        *,
        seed: int,
        prompt: str,
        num_inference_steps: int,
        height: int,
        width: int,
        guidance: float,
        scheduler: str,
        image_paths: list[Path],
        use_kv_cache: bool,
    ) -> GeneratedImageProtocol: ...


class GeneratedImageProtocol(Protocol):
    """Generated image value returned by MFLUX."""

    def save(self, path: Path, *, overwrite: bool) -> None: ...


MfluxModelFactory = Callable[[str, int | None], MfluxModelProtocol]
MfluxEditModelFactory = Callable[[str, int | None], MfluxEditModelProtocol]


def _release_model_cache() -> None:
    gc.collect()
    try:
        import mlx.core as mx
    except ImportError:
        return
    mx.clear_cache()


class MfluxGenerationRequest(DomainModel):
    """Effective request for one transient background candidate."""

    inputs: ImageGenerationInputs
    render_prompt: NonEmptyString
    output_path: Path
    model_identifier: NonEmptyString = "flux2-klein-4b"
    seed: int
    width: PositiveInt = 1024
    height: PositiveInt = 768
    step_count: PositiveInt = 4
    quantization: int | None = None
    guidance: FiniteFloat = Field(default=1.0, gt=0.0)
    scheduler: NonEmptyString = "flow_match_euler_discrete"
    reference_image_paths: tuple[Path, ...] = ()

    @model_validator(mode="after")
    def require_matching_reference_inputs(self) -> MfluxGenerationRequest:
        if len(self.inputs.references()) != len(self.reference_image_paths):
            raise ValueError(
                "reference image paths must match captured reference inputs"
            )
        return self


class MfluxGenerationResult(DomainModel):
    """Generated candidate path and complete production metadata."""

    output_path: Path
    metadata: ImageGenerationMetadata
    load_duration_seconds: float = Field(ge=0.0)
    generation_duration_seconds: float = Field(ge=0.0)
    serialization_duration_seconds: float = Field(ge=0.0)


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "unknown"


def _default_model_factory(model_identifier: str, quantization: int | None) -> MfluxModelProtocol:
    from mflux.models.common.config import ModelConfig
    from mflux.models.flux2.variants import Flux2Klein

    configurations = {
        "flux2-klein-4b": ModelConfig.flux2_klein_4b,
        "flux2-klein-9b": ModelConfig.flux2_klein_9b,
        "flux2-klein-9b-kv": ModelConfig.flux2_klein_9b,
    }
    configuration_factory = configurations.get(model_identifier)
    if configuration_factory is None:
        supported = ", ".join(sorted(configurations))
        raise ValueError(
            f"unsupported MFLUX model {model_identifier!r}; choose one of: {supported}"
        )
    return Flux2Klein(
        quantize=quantization,
        model_config=configuration_factory(),
    )


def _default_edit_model_factory(
    model_identifier: str,
    quantization: int | None,
) -> MfluxEditModelProtocol:
    from mflux.models.common.config import ModelConfig
    from mflux.models.flux2.variants import Flux2KleinEdit

    configurations = {
        "flux2-klein-4b": ModelConfig.flux2_klein_4b,
        "flux2-klein-9b": ModelConfig.flux2_klein_9b,
        "flux2-klein-9b-kv": ModelConfig.flux2_klein_9b_kv,
    }
    configuration_factory = configurations.get(model_identifier)
    if configuration_factory is None:
        supported = ", ".join(sorted(configurations))
        raise ValueError(
            f"unsupported MFLUX edit model {model_identifier!r}; "
            f"choose one of: {supported}"
        )
    return Flux2KleinEdit(
        quantize=quantization,
        model_config=configuration_factory(),
    )


class MfluxGenerator:
    """Lazy, cached, synchronous MFLUX adapter for a serialized worker."""

    def __init__(
        self,
        *,
        model_factory: MfluxModelFactory | None = None,
        edit_model_factory: MfluxEditModelFactory | None = None,
    ) -> None:
        self._model_factory = model_factory or _default_model_factory
        self._edit_model_factory = edit_model_factory or _default_edit_model_factory
        self._models: dict[
            tuple[str, int | None, bool],
            MfluxModelProtocol | MfluxEditModelProtocol,
        ] = {}

    def _model_for(
        self,
        request: MfluxGenerationRequest,
    ) -> MfluxModelProtocol | MfluxEditModelProtocol:
        edit = bool(request.reference_image_paths)
        cache_key = (request.model_identifier, request.quantization, edit)
        if cache_key not in self._models:
            if self._models:
                self._models.clear()
                _release_model_cache()
            try:
                factory = self._edit_model_factory if edit else self._model_factory
                self._models[cache_key] = factory(
                    request.model_identifier,
                    request.quantization,
                )
            except (ImportError, MemoryError, OSError, RuntimeError, ValueError) as error:
                raise ModelLoadError(
                    f"Could not load MFLUX model {request.model_identifier!r} "
                    f"with quantization {request.quantization!r}: {error}"
                ) from error
        return self._models[cache_key]

    def generate(self, request: MfluxGenerationRequest) -> MfluxGenerationResult:
        """Generate and save one candidate image through the cached model."""
        if request.output_path.exists():
            raise ImageGenerationError(
                f"Refusing to overwrite existing generated image: {request.output_path}"
            )
        request.output_path.parent.mkdir(parents=True, exist_ok=True)
        started = perf_counter()
        model = self._model_for(request)
        load_duration_seconds = perf_counter() - started
        generated_at = datetime.now(UTC)
        generation_started = perf_counter()
        try:
            generation_arguments = {
                "seed": request.seed,
                "prompt": request.render_prompt,
                "num_inference_steps": request.step_count,
                "height": request.height,
                "width": request.width,
                "guidance": request.guidance,
                "scheduler": request.scheduler,
            }
            if request.reference_image_paths:
                generation_arguments.update(
                    {
                        "image_paths": list(request.reference_image_paths),
                        "use_kv_cache": (
                            request.model_identifier == "flux2-klein-9b-kv"
                        ),
                    }
                )
            image = model.generate_image(  # type: ignore[union-attr]
                **generation_arguments,
            )
            generation_duration_seconds = perf_counter() - generation_started
            serialization_started = perf_counter()
            image.save(request.output_path, overwrite=False)
            serialization_duration_seconds = perf_counter() - serialization_started
        except (AttributeError, MemoryError, OSError, RuntimeError, TypeError, ValueError) as error:
            raise ImageGenerationError(
                f"MFLUX generation failed for {request.model_identifier!r}: {error}"
            ) from error
        duration_seconds = perf_counter() - started
        if not request.output_path.is_file():
            raise ImageGenerationError(
                f"MFLUX reported success but did not create {request.output_path}"
            )
        try:
            with Image.open(request.output_path) as saved_image:
                image_format = saved_image.format
                image_size = saved_image.size
                saved_image.verify()
            with Image.open(request.output_path) as decoded_image:
                decoded_image.load()
        except (OSError, UnidentifiedImageError) as error:
            request.output_path.unlink(missing_ok=True)
            raise ImageGenerationError(
                f"MFLUX produced an unreadable image at {request.output_path}: {error}"
            ) from error
        if image_format != "PNG" or image_size != (request.width, request.height):
            request.output_path.unlink(missing_ok=True)
            raise ImageGenerationError(
                "MFLUX output did not match the requested PNG contract: "
                f"format={image_format!r}, size={image_size!r}, "
                f"expected_size={(request.width, request.height)!r}"
            )
        metadata = ImageGenerationMetadata(
            inputs=request.inputs,
            render_prompt=request.render_prompt,
            model_identifier=request.model_identifier,
            mflux_version=_package_version("mflux"),
            dependency_versions={"mlx": _package_version("mlx")},
            seed=request.seed,
            width=request.width,
            height=request.height,
            step_count=request.step_count,
            quantization=str(request.quantization) if request.quantization is not None else None,
            effective_settings={
                "guidance": request.guidance,
                "scheduler": request.scheduler,
                "reference_count": len(request.reference_image_paths),
                "use_kv_cache": (
                    request.model_identifier == "flux2-klein-9b-kv"
                    and bool(request.reference_image_paths)
                ),
            },
            generated_at=generated_at,
            duration_seconds=duration_seconds,
        )
        return MfluxGenerationResult(
            output_path=request.output_path,
            metadata=metadata,
            load_duration_seconds=load_duration_seconds,
            generation_duration_seconds=generation_duration_seconds,
            serialization_duration_seconds=serialization_duration_seconds,
        )
