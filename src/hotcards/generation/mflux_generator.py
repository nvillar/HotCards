"""Serialized in-process MFLUX image-operation adapter."""

from __future__ import annotations

import gc
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from threading import Event, Lock
from time import perf_counter
from typing import Literal, Protocol, cast

from PIL import Image, UnidentifiedImageError
from pydantic import Field, FiniteFloat, model_validator

from hotcards.domain.image_dimensions import (
    AspectRatio,
    GenerateResolution,
    output_dimensions,
)
from hotcards.domain.models import (
    AcceptedEdit,
    DirectGenerateProvenance,
    DomainModel,
    EditOutputSize,
    EditPreserveOptions,
    EditProvenance,
    GenerateInputs,
    ImageOperationSettings,
    ImageSourceSnapshot,
    NonEmptyString,
    PositiveInt,
    RefineProvenance,
    RefineTransformation,
    StyleSnapshot,
    edit_output_dimensions,
)
from hotcards.generation.errors import (
    ImageGenerationCancelled,
    ImageGenerationError,
    ModelLoadError,
)
from hotcards.generation.image_generation import EDIT_PROMPT_TOKEN_BUDGET


class MfluxCallbackRegistryProtocol(Protocol):
    """MFLUX callback registration used for progress and cancellation."""

    def register(self, callback: object) -> None: ...


class MfluxRegularModelProtocol(Protocol):
    """Subset of regular Flux2Klein used by Generate and Refine."""

    callbacks: MfluxCallbackRegistryProtocol

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
        image_path: Path | None = None,
        image_strength: float | None = None,
    ) -> GeneratedImageProtocol: ...


class MfluxEditModelProtocol(Protocol):
    """Subset of Flux2KleinEdit used by Reference Generate and Edit."""

    callbacks: MfluxCallbackRegistryProtocol
    tokenizers: Mapping[str, object]

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


MfluxRegularModelFactory = Callable[
    [str, int | None],
    MfluxRegularModelProtocol,
]
MfluxEditModelFactory = Callable[[str, int | None], MfluxEditModelProtocol]
MfluxProgressCallback = Callable[[int, int], None]
_DEFAULT_WIDTH, _DEFAULT_HEIGHT = output_dimensions(
    GenerateResolution.RESOLUTION_512,
    AspectRatio.LANDSCAPE,
)


class MfluxCancellationToken:
    """Thread-safe cooperative cancellation shared with one adapter request."""

    def __init__(self) -> None:
        self._event = Event()
        self._publication_lock = Lock()

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self) -> None:
        with self._publication_lock:
            self._event.set()

    def raise_if_cancelled(self, operation: str) -> None:
        if self.is_cancelled:
            raise ImageGenerationCancelled(f"MFLUX {operation} was cancelled")

    def publish_if_active(
        self,
        operation: str,
        publish: Callable[[], MfluxOutputOwnership],
    ) -> MfluxOutputOwnership:
        """Linearize cancellation against final output publication."""
        with self._publication_lock:
            self.raise_if_cancelled(operation)
            return publish()


class _MfluxStepProgress:
    def __init__(self) -> None:
        self._sink: MfluxProgressCallback | None = None
        self._cancellation: MfluxCancellationToken | None = None
        self._operation = "operation"
        self._completed_steps = 0
        self._total_steps = 0
        self._last_reported_steps = -1

    def set_context(
        self,
        *,
        sink: MfluxProgressCallback | None,
        cancellation: MfluxCancellationToken,
        operation: str,
    ) -> None:
        self._sink = sink
        self._cancellation = cancellation
        self._operation = operation

    def clear_context(self) -> None:
        self._sink = None
        self._cancellation = None
        self._operation = "operation"

    def call_before_loop(self, **values: object) -> None:
        self._raise_if_cancelled()
        config = values.get("config")
        total_steps = getattr(config, "num_inference_steps", None)
        if not isinstance(total_steps, int) or total_steps <= 0:
            raise ImageGenerationError(
                "MFLUX reported an invalid inference-step count"
            )
        self._completed_steps = 0
        self._total_steps = total_steps
        self._last_reported_steps = -1
        self._report_steps()

    def call_in_loop(self, **_values: object) -> None:
        self._raise_if_cancelled()
        self._report_steps()
        self._completed_steps = min(
            self._completed_steps + 1,
            self._total_steps,
        )

    def call_after_loop(self, **_values: object) -> None:
        self._raise_if_cancelled()
        self._completed_steps = self._total_steps
        self._report_steps()
        if self._sink is not None:
            self._sink(0, 0)

    def _raise_if_cancelled(self) -> None:
        cancellation = self._cancellation
        if cancellation is not None:
            cancellation.raise_if_cancelled(self._operation)

    def _report_steps(self) -> None:
        if (
            self._sink is not None
            and self._completed_steps != self._last_reported_steps
        ):
            self._sink(self._completed_steps, self._total_steps)
            self._last_reported_steps = self._completed_steps


def _release_model_cache() -> None:
    gc.collect()
    try:
        import mlx.core as mx
    except ImportError:
        return
    mx.clear_cache()


class _MfluxRequest(DomainModel):
    """Shared exact execution settings for one MFLUX operation."""

    output_path: Path
    model_identifier: NonEmptyString = "flux2-klein-4b"
    aspect_ratio: AspectRatio = AspectRatio.LANDSCAPE
    width: PositiveInt = _DEFAULT_WIDTH
    height: PositiveInt = _DEFAULT_HEIGHT
    step_count: PositiveInt = 4
    quantization: int | None = None
    guidance: FiniteFloat = Field(default=1.0, gt=0.0)
    scheduler: NonEmptyString = "flow_match_euler_discrete"

    def require_dimensions(self, resolution: GenerateResolution) -> None:
        expected_dimensions = output_dimensions(
            resolution,
            self.aspect_ratio,
        )
        if (self.width, self.height) != expected_dimensions:
            raise ValueError(
                "image dimensions must match the request resolution and aspect ratio"
            )


class MfluxGenerateRequest(_MfluxRequest):
    """Direct Generate request with zero, one, or two ordered References."""

    operation: Literal["generate"] = "generate"
    inputs: GenerateInputs
    render_prompt: NonEmptyString
    seed: int
    reference_image_paths: tuple[Path, ...] = Field(
        default_factory=tuple,
        max_length=2,
    )

    @model_validator(mode="after")
    def require_generate_contract(self) -> MfluxGenerateRequest:
        self.require_dimensions(self.inputs.resolution)
        if len(self.inputs.references) != len(self.reference_image_paths):
            raise ValueError(
                "reference image paths must match captured reference inputs"
            )
        return self


class MfluxRefineRequest(_MfluxRequest):
    """Regular Flux2Klein img2img request for one current source image."""

    operation: Literal["refine"] = "refine"
    source: ImageSourceSnapshot
    source_image_path: Path
    source_seed: int
    description: str
    style: StyleSnapshot | None = None
    edit_lineage: tuple[AcceptedEdit, ...] = Field(default_factory=tuple)
    render_prompt: NonEmptyString
    resolution: GenerateResolution
    transformation: RefineTransformation
    image_strength: FiniteFloat = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def require_refine_contract(self) -> MfluxRefineRequest:
        self.require_dimensions(self.resolution)
        if self.image_strength != self.transformation.strength:
            raise ValueError(
                f"{self.transformation.value} Refine image strength must be "
                f"{self.transformation.strength:.2f}"
            )
        return self


class MfluxEditRequest(_MfluxRequest):
    """Flux2KleinEdit request for one current source image."""

    operation: Literal["edit"] = "edit"
    source: ImageSourceSnapshot
    source_image_path: Path
    instruction: NonEmptyString
    preserve: EditPreserveOptions
    expanded_prompt: NonEmptyString
    output_size: EditOutputSize
    edit_lineage: tuple[AcceptedEdit, ...] = Field(min_length=1)
    seed: int

    @property
    def accepted_edit(self) -> AcceptedEdit:
        return AcceptedEdit(
            instruction=self.instruction,
            preserve=self.preserve,
            expanded_prompt=self.expanded_prompt,
        )

    @model_validator(mode="after")
    def require_edit_contract(self) -> MfluxEditRequest:
        if (self.width, self.height) != edit_output_dimensions(
            self.output_size,
            self.aspect_ratio,
        ):
            raise ValueError(
                "Edit dimensions must match the selected output size"
            )
        if self.edit_lineage[-1] != self.accepted_edit:
            raise ValueError(
                "Edit request lineage must end with the accepted current Edit"
            )
        return self


type MfluxOperationRequest = (
    MfluxGenerateRequest | MfluxRefineRequest | MfluxEditRequest
)


class MfluxOutputOwnership(DomainModel):
    """Filesystem identity proving ownership of one atomically published output."""

    output_path: Path
    device: int = Field(ge=0)
    inode: int = Field(ge=0)

    def dispose(self) -> None:
        try:
            current = self.output_path.stat(follow_symlinks=False)
        except FileNotFoundError:
            return
        if (current.st_dev, current.st_ino) != (self.device, self.inode):
            return
        try:
            self.output_path.unlink()
        except FileNotFoundError:
            return


class _MfluxResult(DomainModel):
    """Shared timing and output facts for one completed MFLUX operation."""

    output_path: Path
    output_ownership: MfluxOutputOwnership
    queue_duration_seconds: float = Field(ge=0.0)
    load_duration_seconds: float = Field(ge=0.0)
    generation_duration_seconds: float = Field(ge=0.0)
    serialization_duration_seconds: float = Field(ge=0.0)

    @model_validator(mode="after")
    def require_matching_output_ownership(self) -> _MfluxResult:
        if self.output_ownership.output_path != self.output_path:
            raise ValueError("MFLUX result ownership must match its output path")
        return self

    def dispose_output(self) -> None:
        """Remove this result only while its exact published inode is still present."""
        self.output_ownership.dispose()


class MfluxGenerateResult(_MfluxResult):
    provenance: DirectGenerateProvenance


class MfluxRefineResult(_MfluxResult):
    provenance: RefineProvenance


class MfluxEditResult(_MfluxResult):
    provenance: EditProvenance


type MfluxOperationResult = (
    MfluxGenerateResult | MfluxRefineResult | MfluxEditResult
)


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "unknown"


def _default_model_factory(
    model_identifier: str,
    quantization: int | None,
) -> MfluxRegularModelProtocol:
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


class _ModelFamily(StrEnum):
    REGULAR = "regular"
    EDIT = "edit"


@dataclass(slots=True)
class _CachedModel:
    family: _ModelFamily
    model_identifier: str
    quantization: int | None
    factory: object
    model: MfluxRegularModelProtocol | MfluxEditModelProtocol
    progress: _MfluxStepProgress | None = None

    def is_compatible(
        self,
        *,
        family: _ModelFamily,
        model_identifier: str,
        quantization: int | None,
        factory: object,
    ) -> bool:
        return (
            self.family is family
            and self.model_identifier == model_identifier
            and self.quantization == quantization
            and self.factory is factory
        )


_PROCESS_EXECUTION_LOCK = Lock()
_CACHED_MODEL: _CachedModel | None = None
_RELEASE_REQUEST_LOCK = Lock()
_DEFERRED_RELEASES: list[tuple[object, object]] = []
_CACHE_RELEASE_PENDING = False


def _cached_model_is_compatible(
    *,
    family: _ModelFamily,
    model_identifier: str,
    quantization: int | None,
    factory: object,
) -> bool:
    cached = _CACHED_MODEL
    return cached is not None and cached.is_compatible(
        family=family,
        model_identifier=model_identifier,
        quantization=quantization,
        factory=factory,
    )


def _cached_model_uses_factory(
    regular_factory: object,
    edit_factory: object,
) -> bool:
    cached = _CACHED_MODEL
    return cached is not None and (
        cached.factory is regular_factory or cached.factory is edit_factory
    )


def _release_cached_model_locked() -> None:
    """Detach every cache-owned model reference before clearing MLX memory."""
    global _CACHE_RELEASE_PENDING, _CACHED_MODEL
    if _CACHED_MODEL is None:
        if _CACHE_RELEASE_PENDING:
            _release_model_cache()
            _CACHE_RELEASE_PENDING = False
        return

    progress = _CACHED_MODEL.progress
    _CACHED_MODEL.progress = None
    _CACHED_MODEL = None
    _CACHE_RELEASE_PENDING = True
    try:
        if progress is not None:
            progress.clear_context()
    finally:
        progress = None
        _release_model_cache()
        _CACHE_RELEASE_PENDING = False


def _release_requested_cache_locked(
    requests: tuple[tuple[object, object], ...],
) -> None:
    if _CACHE_RELEASE_PENDING:
        _release_cached_model_locked()
    for regular_factory, edit_factory in requests:
        if _cached_model_uses_factory(regular_factory, edit_factory):
            _release_cached_model_locked()
            return


def _process_deferred_release_requests_locked() -> None:
    if _CACHE_RELEASE_PENDING:
        _release_cached_model_locked()
    while True:
        with _RELEASE_REQUEST_LOCK:
            if not _DEFERRED_RELEASES:
                return
            requests = tuple(_DEFERRED_RELEASES)
            _DEFERRED_RELEASES.clear()
        _release_requested_cache_locked(requests)


def _release_process_lock_after_deferred_requests() -> None:
    try:
        _process_deferred_release_requests_locked()
    finally:
        with _RELEASE_REQUEST_LOCK:
            _PROCESS_EXECUTION_LOCK.release()


@dataclass(frozen=True, slots=True)
class _OwnedOutput:
    scope: Path
    candidate: Path

    @classmethod
    def create(cls, target: Path, operation: str) -> _OwnedOutput:
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise ImageGenerationError(
                f"MFLUX {operation} could not prepare output directory "
                f"{target.parent}: {error}"
            ) from error
        if os.path.lexists(target):
            raise ImageGenerationError(
                f"Refusing to overwrite existing MFLUX {operation} output: {target}"
            )
        try:
            scope = Path(
                tempfile.mkdtemp(
                    prefix=".hotcards-mflux-",
                    dir=target.parent,
                )
            )
        except OSError as error:
            raise ImageGenerationError(
                f"MFLUX {operation} could not reserve a private output scope in "
                f"{target.parent}: {error}"
            ) from error
        return cls(scope=scope, candidate=scope / "candidate.png")

    def publish(
        self,
        target: Path,
        operation: str,
    ) -> MfluxOutputOwnership:
        try:
            candidate_identity = self.candidate.stat(follow_symlinks=False)
        except OSError as error:
            raise ImageGenerationError(
                f"MFLUX {operation} could not inspect reserved candidate "
                f"{self.candidate}: {error}"
            ) from error
        try:
            os.link(self.candidate, target)
        except FileExistsError as error:
            raise ImageGenerationError(
                f"MFLUX {operation} output appeared before publication; "
                f"refusing to overwrite {target}"
            ) from error
        except OSError as error:
            if os.path.lexists(target):
                raise ImageGenerationError(
                    f"MFLUX {operation} output appeared before publication; "
                    f"refusing to overwrite {target}"
                ) from error
            raise ImageGenerationError(
                f"MFLUX {operation} could not atomically publish {target}: {error}"
            ) from error
        return MfluxOutputOwnership(
            output_path=target,
            device=candidate_identity.st_dev,
            inode=candidate_identity.st_ino,
        )

    def cleanup(self, operation: str) -> None:
        if not self.scope.exists():
            return
        try:
            shutil.rmtree(self.scope)
        except OSError as error:
            raise ImageGenerationError(
                f"MFLUX {operation} could not clean private output scope "
                f"{self.scope}: {error}"
            ) from error


class MfluxGenerator:
    """One typed adapter over the process-local serialized MFLUX runtime."""

    def __init__(
        self,
        *,
        model_factory: MfluxRegularModelFactory | None = None,
        edit_model_factory: MfluxEditModelFactory | None = None,
    ) -> None:
        self._model_factory = model_factory or _default_model_factory
        self._edit_model_factory = edit_model_factory or _default_edit_model_factory

    def generate(
        self,
        request: MfluxGenerateRequest,
        *,
        progress: MfluxProgressCallback | None = None,
        cancellation: MfluxCancellationToken | None = None,
    ) -> MfluxGenerateResult:
        """Execute plain or Reference-backed Generate."""
        result = self._execute(
            request,
            progress=progress,
            cancellation=cancellation,
        )
        assert isinstance(result, MfluxGenerateResult)
        return result

    def refine(
        self,
        request: MfluxRefineRequest,
        *,
        progress: MfluxProgressCallback | None = None,
        cancellation: MfluxCancellationToken | None = None,
    ) -> MfluxRefineResult:
        """Execute regular Flux2Klein true img2img Refine."""
        result = self._execute(
            request,
            progress=progress,
            cancellation=cancellation,
        )
        assert isinstance(result, MfluxRefineResult)
        return result

    def edit(
        self,
        request: MfluxEditRequest,
        *,
        progress: MfluxProgressCallback | None = None,
        cancellation: MfluxCancellationToken | None = None,
    ) -> MfluxEditResult:
        """Execute one direct Flux2KleinEdit operation."""
        result = self._execute(
            request,
            progress=progress,
            cancellation=cancellation,
        )
        assert isinstance(result, MfluxEditResult)
        return result

    def release(self) -> None:
        """Release now when idle, or defer without waiting for active MFLUX."""
        request = (self._model_factory, self._edit_model_factory)
        with _RELEASE_REQUEST_LOCK:
            if not _PROCESS_EXECUTION_LOCK.acquire(blocking=False):
                if not any(
                    regular is request[0] and edit is request[1]
                    for regular, edit in _DEFERRED_RELEASES
                ):
                    _DEFERRED_RELEASES.append(request)
                return
        try:
            _process_deferred_release_requests_locked()
            _release_requested_cache_locked((request,))
        finally:
            _release_process_lock_after_deferred_requests()

    def _execute(
        self,
        request: MfluxOperationRequest,
        *,
        progress: MfluxProgressCallback | None,
        cancellation: MfluxCancellationToken | None,
    ) -> MfluxOperationResult:
        operation = request.operation
        token = cancellation or MfluxCancellationToken()
        request_started = perf_counter()
        token.raise_if_cancelled(operation)
        self._require_request_paths(request)
        while not _PROCESS_EXECUTION_LOCK.acquire(timeout=0.05):
            token.raise_if_cancelled(operation)
        queue_duration_seconds = perf_counter() - request_started
        progress_callback: _MfluxStepProgress | None = None
        model: MfluxRegularModelProtocol | MfluxEditModelProtocol | None = None
        image: GeneratedImageProtocol | None = None
        owned_output: _OwnedOutput | None = None
        output_ownership: MfluxOutputOwnership | None = None
        prompt_token_count: int | None = None
        try:
            _process_deferred_release_requests_locked()
            token.raise_if_cancelled(operation)
            owned_output = _OwnedOutput.create(request.output_path, operation)
            load_started = perf_counter()
            model, family = self._model_for(request)
            load_duration_seconds = perf_counter() - load_started
            token.raise_if_cancelled(operation)
            if isinstance(request, MfluxEditRequest):
                prompt_token_count = self._formatted_edit_token_count(
                    model,
                    request.expanded_prompt,
                )
                if prompt_token_count > EDIT_PROMPT_TOKEN_BUDGET:
                    raise ImageGenerationError(
                        "Edit Instruction expands to "
                        f"{prompt_token_count} model tokens; the limit is "
                        f"{EDIT_PROMPT_TOKEN_BUDGET}. Shorten the instruction "
                        "or select fewer Preserve options."
                    )
            try:
                progress_callback = self._progress_callback_for(
                    model,
                    register_if_missing=progress is not None or cancellation is not None,
                )
            except (AttributeError, RuntimeError, TypeError) as error:
                raise ImageGenerationError(
                    f"MFLUX {operation} could not register progress callbacks: "
                    f"{error}"
                ) from error
            if progress_callback is not None:
                progress_callback.set_context(
                    sink=progress,
                    cancellation=token,
                    operation=operation,
                )
            generated_at = datetime.now(UTC)
            generation_started = perf_counter()
            try:
                image = model.generate_image(
                    **self._generation_arguments(request),
                )
                token.raise_if_cancelled(operation)
                generation_duration_seconds = perf_counter() - generation_started
                serialization_started = perf_counter()
                image.save(owned_output.candidate, overwrite=False)
                token.raise_if_cancelled(operation)
                serialization_duration_seconds = (
                    perf_counter() - serialization_started
                )
            except ImageGenerationCancelled:
                raise
            except (
                AttributeError,
                MemoryError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as error:
                raise ImageGenerationError(
                    f"MFLUX {operation} failed for "
                    f"{request.model_identifier!r}: {error}"
                ) from error
            self._validate_output(
                owned_output.candidate,
                request=request,
            )
            token.raise_if_cancelled(operation)
            output_ownership = token.publish_if_active(
                operation,
                lambda: owned_output.publish(request.output_path, operation),
            )
            duration_seconds = perf_counter() - request_started
            settings = ImageOperationSettings(
                model_identifier=request.model_identifier,
                mflux_version=_package_version("mflux"),
                dependency_versions={"mlx": _package_version("mlx")},
                seed=self._seed(request),
                width=request.width,
                height=request.height,
                step_count=request.step_count,
                quantization=request.quantization,
                guidance=request.guidance,
                scheduler=request.scheduler,
                use_kv_cache=(
                    family is _ModelFamily.EDIT
                    and request.model_identifier == "flux2-klein-9b-kv"
                ),
                generated_at=generated_at,
                duration_seconds=duration_seconds,
            )
            return self._result(
                request,
                output_ownership=output_ownership,
                settings=settings,
                queue_duration_seconds=queue_duration_seconds,
                load_duration_seconds=load_duration_seconds,
                generation_duration_seconds=generation_duration_seconds,
                serialization_duration_seconds=serialization_duration_seconds,
                prompt_token_count=prompt_token_count,
            )
        except ImageGenerationCancelled as error:
            error.__traceback__ = None
            image = None
            model = None
            if progress_callback is not None:
                progress_callback.clear_context()
                progress_callback = None
            _release_cached_model_locked()
            raise error from None
        finally:
            if progress_callback is not None:
                progress_callback.clear_context()
            image = None
            model = None
            try:
                if owned_output is not None:
                    owned_output.cleanup(operation)
            finally:
                _release_process_lock_after_deferred_requests()

    def _model_for(
        self,
        request: MfluxOperationRequest,
    ) -> tuple[
        MfluxRegularModelProtocol | MfluxEditModelProtocol,
        _ModelFamily,
    ]:
        global _CACHED_MODEL
        family = self._family(request)
        factory = (
            self._model_factory
            if family is _ModelFamily.REGULAR
            else self._edit_model_factory
        )
        if not _cached_model_is_compatible(
            family=family,
            model_identifier=request.model_identifier,
            quantization=request.quantization,
            factory=factory,
        ):
            _release_cached_model_locked()
            try:
                model = factory(
                    request.model_identifier,
                    request.quantization,
                )
            except (
                ImportError,
                MemoryError,
                OSError,
                RuntimeError,
                ValueError,
            ) as error:
                raise ModelLoadError(
                    f"Could not load MFLUX {family.value} model "
                    f"{request.model_identifier!r} with quantization "
                    f"{request.quantization!r} for {request.operation}: {error}"
                ) from error
            _CACHED_MODEL = _CachedModel(
                family=family,
                model_identifier=request.model_identifier,
                quantization=request.quantization,
                factory=factory,
                model=model,
            )
            return model, family
        assert _CACHED_MODEL is not None
        return _CACHED_MODEL.model, family

    @staticmethod
    def _progress_callback_for(
        model: MfluxRegularModelProtocol | MfluxEditModelProtocol,
        *,
        register_if_missing: bool,
    ) -> _MfluxStepProgress | None:
        cached = _CACHED_MODEL
        assert cached is not None and cached.model is model
        if cached.progress is None:
            if not register_if_missing:
                return None
            cached.progress = _MfluxStepProgress()
            model.callbacks.register(cached.progress)
        return cached.progress

    @staticmethod
    def _family(request: MfluxOperationRequest) -> _ModelFamily:
        if isinstance(request, MfluxRefineRequest):
            return _ModelFamily.REGULAR
        if isinstance(request, MfluxEditRequest):
            return _ModelFamily.EDIT
        return (
            _ModelFamily.EDIT
            if request.reference_image_paths
            else _ModelFamily.REGULAR
        )

    @staticmethod
    def _seed(request: MfluxOperationRequest) -> int:
        return (
            request.source_seed
            if isinstance(request, MfluxRefineRequest)
            else request.seed
        )

    @classmethod
    def _generation_arguments(
        cls,
        request: MfluxOperationRequest,
    ) -> dict[str, object]:
        arguments: dict[str, object] = {
            "seed": cls._seed(request),
            "prompt": cls._prompt(request),
            "num_inference_steps": request.step_count,
            "height": request.height,
            "width": request.width,
            "guidance": request.guidance,
            "scheduler": request.scheduler,
        }
        if isinstance(request, MfluxRefineRequest):
            arguments.update(
                {
                    "image_path": request.source_image_path,
                    "image_strength": request.image_strength,
                }
            )
        elif isinstance(request, MfluxEditRequest):
            arguments.update(
                {
                    "image_paths": [request.source_image_path],
                    "use_kv_cache": (
                        request.model_identifier == "flux2-klein-9b-kv"
                    ),
                }
            )
        elif request.reference_image_paths:
            arguments.update(
                {
                    "image_paths": list(request.reference_image_paths),
                    "use_kv_cache": (
                        request.model_identifier == "flux2-klein-9b-kv"
                    ),
                }
            )
        return arguments

    @staticmethod
    def _formatted_edit_token_count(
        model: MfluxRegularModelProtocol | MfluxEditModelProtocol,
        prompt: str,
    ) -> int:
        try:
            tokenizer = cast(MfluxEditModelProtocol, model).tokenizers["qwen3"]
            raw_tokenizer = tokenizer.tokenizer
            formatted_prompt = prompt
            template = getattr(tokenizer, "template", None)
            use_chat_template = getattr(
                tokenizer,
                "use_chat_template",
                False,
            )
            if template:
                formatted_prompt = template.format(prompt)
            elif use_chat_template:
                formatted_prompt = raw_tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                    **getattr(tokenizer, "chat_template_kwargs", {}),
                )
            tokens = raw_tokenizer(
                formatted_prompt,
                padding=False,
                truncation=False,
                add_special_tokens=tokenizer.add_special_tokens,
                return_attention_mask=False,
            )
            input_ids = tokens["input_ids"]
            if input_ids and isinstance(input_ids[0], list):
                input_ids = input_ids[0]
            count = len(input_ids)
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            raise ImageGenerationError(
                f"Could not validate the Edit prompt token count: {error}"
            ) from error
        if count <= 0:
            raise ImageGenerationError(
                "Could not validate the Edit prompt token count"
            )
        return count

    @staticmethod
    def _prompt(request: MfluxOperationRequest) -> str:
        return (
            request.expanded_prompt
            if isinstance(request, MfluxEditRequest)
            else request.render_prompt
        )

    @staticmethod
    def _require_request_paths(request: MfluxOperationRequest) -> None:
        sources: tuple[Path, ...]
        if isinstance(request, (MfluxRefineRequest, MfluxEditRequest)):
            sources = (request.source_image_path,)
        else:
            sources = request.reference_image_paths
        for position, source in enumerate(sources, start=1):
            if not source.is_file():
                raise ImageGenerationError(
                    f"MFLUX {request.operation} source image {position} "
                    f"does not exist: {source}"
                )

    @staticmethod
    def _validate_output(
        output_path: Path,
        *,
        request: MfluxOperationRequest,
    ) -> None:
        if not output_path.is_file():
            raise ImageGenerationError(
                f"MFLUX {request.operation} reported success but did not create "
                f"the reserved candidate {output_path}"
            )
        try:
            with Image.open(output_path) as saved_image:
                image_format = saved_image.format
                image_size = saved_image.size
                saved_image.verify()
            with Image.open(output_path) as decoded_image:
                decoded_image.load()
        except (OSError, UnidentifiedImageError) as error:
            raise ImageGenerationError(
                f"MFLUX {request.operation} produced an unreadable image at "
                f"{output_path}: {error}"
            ) from error
        if image_format != "PNG" or image_size != (
            request.width,
            request.height,
        ):
            raise ImageGenerationError(
                f"MFLUX {request.operation} output did not match the requested "
                f"PNG contract: format={image_format!r}, size={image_size!r}, "
                f"expected_size={(request.width, request.height)!r}"
            )

    @staticmethod
    def _result(
        request: MfluxOperationRequest,
        *,
        output_ownership: MfluxOutputOwnership,
        settings: ImageOperationSettings,
        queue_duration_seconds: float,
        load_duration_seconds: float,
        generation_duration_seconds: float,
        serialization_duration_seconds: float,
        prompt_token_count: int | None,
    ) -> MfluxOperationResult:
        timing = {
            "output_path": request.output_path,
            "output_ownership": output_ownership,
            "queue_duration_seconds": queue_duration_seconds,
            "load_duration_seconds": load_duration_seconds,
            "generation_duration_seconds": generation_duration_seconds,
            "serialization_duration_seconds": serialization_duration_seconds,
        }
        if isinstance(request, MfluxRefineRequest):
            return MfluxRefineResult(
                **timing,
                provenance=RefineProvenance(
                    source=request.source,
                    description=request.description,
                    style=request.style,
                    edit_lineage=request.edit_lineage,
                    render_prompt=request.render_prompt,
                    resolution=request.resolution,
                    transformation=request.transformation,
                    strength=request.image_strength,
                    settings=settings,
                ),
            )
        if isinstance(request, MfluxEditRequest):
            if prompt_token_count is None:
                raise ImageGenerationError(
                    "MFLUX Edit completed without a validated prompt token count"
                )
            return MfluxEditResult(
                **timing,
                provenance=EditProvenance(
                    source=request.source,
                    instruction=request.instruction,
                    preserve=request.preserve,
                    expanded_prompt=request.expanded_prompt,
                    output_size=request.output_size,
                    edit_lineage=request.edit_lineage,
                    prompt_token_count=prompt_token_count,
                    prompt_token_budget=EDIT_PROMPT_TOKEN_BUDGET,
                    settings=settings,
                ),
            )
        return MfluxGenerateResult(
            **timing,
            provenance=DirectGenerateProvenance(
                inputs=request.inputs,
                render_prompt=request.render_prompt,
                settings=settings,
            ),
        )


def dispose_mflux_result(result: object) -> None:
    """Dispose only a typed MFLUX result's identity-verified output."""
    if not isinstance(
        result,
        (MfluxGenerateResult, MfluxRefineResult, MfluxEditResult),
    ):
        raise TypeError("MFLUX result disposer requires a typed MFLUX result")
    result.dispose_output()


__all__ = [
    "dispose_mflux_result",
    "MfluxCancellationToken",
    "MfluxEditRequest",
    "MfluxEditResult",
    "MfluxGenerateRequest",
    "MfluxGenerateResult",
    "MfluxGenerator",
    "MfluxOperationRequest",
    "MfluxOperationResult",
    "MfluxOutputOwnership",
    "MfluxRefineRequest",
    "MfluxRefineResult",
]
