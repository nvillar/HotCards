"""Direct, revision-local background image workflow."""

from __future__ import annotations

import secrets
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from uuid import UUID, uuid4

from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import ValidationError
from PySide6.QtCore import QObject, Signal

from hypergen.application.commands import (
    ActivateRevisionCommand,
    CommandError,
    DeleteRevisionCommand,
    DuplicateRevisionCommand,
    ReplaceRevisionBackgroundCommand,
)
from hypergen.application.document_controller import DocumentController
from hypergen.application.document_session import DocumentSession
from hypergen.application.workers import AdapterWorkers, WorkerOperation
from hypergen.domain.models import (
    Card,
    GeneratedBackground,
    ImageGenerationInputs,
    ImportedBackground,
    Stack,
)
from hypergen.generation.image_prompts import compose_image_prompt
from hypergen.generation.mflux_generator import (
    MfluxGenerationRequest,
    MfluxGenerationResult,
    MfluxGenerator,
)
from hypergen.storage.stack_store import StackStore, StackStoreError


class BackgroundWorkflowError(ValueError):
    """A background operation cannot proceed without losing user intent."""


@dataclass(frozen=True, slots=True)
class BackgroundGenerationSettings:
    """Machine-local effective settings captured before worker submission."""

    mflux_model: str
    step_count: int
    quantization: int | None
    random_seed: bool
    fixed_seed: int


@dataclass(frozen=True, slots=True)
class _GenerationTarget:
    stack_id: UUID
    card_id: UUID
    revision_id: UUID
    description: str
    style_id: UUID | None
    style_prompt: str
    background_id: UUID | None
    bundle_path: Path


GenerationSettingsProvider = Callable[[], BackgroundGenerationSettings]


def _crop_box(
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
    *,
    position_x: float,
    position_y: float,
) -> tuple[int, int, int, int]:
    source_ratio = source_width / source_height
    target_ratio = target_width / target_height
    if source_ratio > target_ratio:
        crop_width = source_height * target_ratio
        left = (source_width - crop_width) * position_x
        return round(left), 0, round(left + crop_width), source_height
    crop_height = source_width / target_ratio
    top = (source_height - crop_height) * position_y
    return 0, round(top), source_width, round(top + crop_height)


def prepare_import_image(
    source_path: Path,
    output_path: Path,
    *,
    width: int,
    height: int,
    position_x: float = 0.5,
    position_y: float = 0.5,
) -> None:
    """Decode, orient, crop-to-fill, and write a correctly sized PNG."""
    if not 0.0 <= position_x <= 1.0 or not 0.0 <= position_y <= 1.0:
        raise BackgroundWorkflowError("crop positions must be between zero and one")
    if output_path.exists():
        raise BackgroundWorkflowError(f"candidate output already exists: {output_path}")
    try:
        with Image.open(source_path) as opened:
            image = ImageOps.exif_transpose(opened)
            image.load()
            crop_box = _crop_box(
                image.width,
                image.height,
                width,
                height,
                position_x=position_x,
                position_y=position_y,
            )
            prepared = image.crop(crop_box).resize(
                (width, height),
                Image.Resampling.LANCZOS,
            )
            if prepared.mode not in {"RGB", "RGBA"}:
                prepared = prepared.convert(
                    "RGBA" if "A" in prepared.getbands() else "RGB"
                )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            prepared.save(output_path, format="PNG")
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as error:
        output_path.unlink(missing_ok=True)
        raise BackgroundWorkflowError(
            f"Could not import image {source_path.name!r}: {error}"
        ) from error


class BackgroundWorkflow(QObject):
    """Generate, import, clear, and manage complete card revisions."""

    busy_changed = Signal(bool)
    progress_changed = Signal(str)
    failed = Signal(object)
    document_changed = Signal(object)
    change_applied = Signal(str, object)

    def __init__(
        self,
        controller: DocumentController,
        session: DocumentSession,
        workers: AdapterWorkers,
        settings_provider: GenerationSettingsProvider,
        *,
        mflux_generator: MfluxGenerator | None = None,
        temporary_directory: Path | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.controller = controller
        self.session = session
        self.workers = workers
        self._settings_provider = settings_provider
        self._mflux_generator = mflux_generator or MfluxGenerator()
        self._owned_temporary_directory = (
            tempfile.TemporaryDirectory(prefix="hypergen-background-")
            if temporary_directory is None
            else None
        )
        self._temporary_directory = (
            Path(self._owned_temporary_directory.name)
            if self._owned_temporary_directory is not None
            else temporary_directory
        )
        assert self._temporary_directory is not None
        self._temporary_directory.mkdir(parents=True, exist_ok=True)
        self._operation: WorkerOperation | None = None
        self._request_id: UUID | None = None
        self._request_target: _GenerationTarget | None = None
        self._pending_image_path: Path | None = None
        self._busy = False

    @property
    def busy(self) -> bool:
        return self._busy

    def generate(self, card_id: UUID) -> WorkerOperation:
        """Generate and directly apply a background to the active revision."""
        self._require_ready(card_id)
        document = self.controller.document
        card = self._card(document, card_id)
        revision = card.active_revision
        style = next(
            (style for style in document.styles if style.id == revision.style_id),
            None,
        )
        inputs = ImageGenerationInputs(
            description=revision.description,
            style_name=style.name if style is not None else None,
            style_prompt=style.prompt if style is not None else "",
        )
        try:
            render_prompt = compose_image_prompt(inputs)
        except ValueError as error:
            raise BackgroundWorkflowError(str(error)) from error
        settings = self._settings_provider()
        request_id = uuid4()
        asset_id = uuid4()
        target = self._target(document, card)
        self._request_id = request_id
        self._request_target = target
        output_path = self._temporary_directory / f"generated-{asset_id}.png"
        request = MfluxGenerationRequest(
            inputs=inputs,
            render_prompt=render_prompt,
            output_path=output_path,
            model_identifier=settings.mflux_model,
            seed=(
                secrets.randbelow(2_147_483_648)
                if settings.random_seed
                else settings.fixed_seed
            ),
            width=document.canvas.width,
            height=document.canvas.height,
            step_count=settings.step_count,
            quantization=settings.quantization,
        )
        self._set_busy(True, "Generating image...")
        operation = self.workers.run_mflux(
            lambda: self._mflux_generator.generate(request),
            stage="generating background image",
        )
        self._operation = operation
        operation.succeeded.connect(
            partial(self._generation_succeeded, request_id, target, asset_id)
        )
        operation.failed.connect(partial(self._operation_failed, request_id))
        return operation

    def import_image(
        self,
        card_id: UUID,
        source_path: Path,
        *,
        position_x: float = 0.5,
        position_y: float = 0.5,
    ) -> Stack:
        """Prepare, store, and directly apply one imported image."""
        self._require_ready(card_id)
        document = self.controller.document
        card = self._card(document, card_id)
        asset_id = uuid4()
        temporary_path = self._temporary_directory / f"import-{asset_id}.png"
        prepare_import_image(
            source_path,
            temporary_path,
            width=document.canvas.width,
            height=document.canvas.height,
            position_x=position_x,
            position_y=position_y,
        )
        try:
            store = self._require_store()
            image_path = store.import_image(
                temporary_path,
                card_id=card.id,
                asset_id=asset_id,
            )
        finally:
            temporary_path.unlink(missing_ok=True)
        background = ImportedBackground(
            id=asset_id,
            image_path=image_path,
            source_filename=source_path.name,
            created_at=datetime.now(UTC),
        )
        return self._apply_background(
            card.id,
            card.active_revision.id,
            background,
            "Image imported",
        )

    def clear_background(self, card_id: UUID) -> Stack:
        """Clear only the active revision's background."""
        card = self._card(self.controller.document, card_id)
        if card.active_revision.background is None:
            raise BackgroundWorkflowError("this revision has no image to clear")
        return self._apply_background(
            card.id,
            card.active_revision.id,
            None,
            "Image cleared",
        )

    def duplicate_revision(self, card_id: UUID, revision_id: UUID) -> Stack:
        previous_token = self.controller.current_undo_token
        changed = self.controller.execute(
            DuplicateRevisionCommand(
                card_id=card_id,
                source_revision_id=revision_id,
            )
        )
        self.progress_changed.emit("Revision duplicated")
        self.document_changed.emit(changed)
        self._emit_change_applied("Revision duplicated", previous_token)
        return changed

    def activate_revision(self, card_id: UUID, revision_id: UUID) -> Stack:
        changed = self.controller.execute(
            ActivateRevisionCommand(card_id=card_id, revision_id=revision_id)
        )
        self.progress_changed.emit("Revision activated")
        self.document_changed.emit(changed)
        return changed

    def delete_revision(self, card_id: UUID, revision_id: UUID) -> Stack:
        previous_token = self.controller.current_undo_token
        changed = self.controller.execute(
            DeleteRevisionCommand(card_id=card_id, revision_id=revision_id)
        )
        self.progress_changed.emit("Revision deleted")
        self.document_changed.emit(changed)
        self._emit_change_applied("Revision deleted", previous_token)
        return changed

    def cancel(self) -> None:
        if self._operation is not None and not self._operation.is_finished:
            self._operation.cancel()
        self._request_id = None
        self._request_target = None
        self._operation = None
        self._discard_pending_image()
        if self._busy:
            self._set_busy(False, "Generation cancelled")

    def close(self) -> None:
        self.cancel()
        if self._owned_temporary_directory is not None:
            self._owned_temporary_directory.cleanup()

    def is_generating_for(self, card_id: UUID) -> bool:
        return (
            self._busy
            and self._request_target is not None
            and self._request_target.card_id == card_id
        )

    def _generation_succeeded(
        self,
        request_id: UUID,
        target: _GenerationTarget,
        asset_id: UUID,
        result: object,
    ) -> None:
        if request_id != self._request_id:
            if isinstance(result, MfluxGenerationResult):
                result.output_path.unlink(missing_ok=True)
            return
        if not isinstance(result, MfluxGenerationResult):
            self._finish_with_error(
                BackgroundWorkflowError("image generation returned an unexpected result")
            )
            return
        self._pending_image_path = result.output_path
        if not self._target_is_current(target):
            self._finish_with_error(
                BackgroundWorkflowError(
                    "the stack, revision, Description, Style, or image changed "
                    "before generation completed"
                )
            )
            return
        try:
            image_path = self._require_store().import_image(
                result.output_path,
                card_id=target.card_id,
                asset_id=asset_id,
            )
            background = GeneratedBackground(
                id=asset_id,
                image_path=image_path,
                generation_metadata=result.metadata,
                created_at=result.metadata.generated_at,
            )
            self._apply_background(
                target.card_id,
                target.revision_id,
                background,
                "Image generated",
            )
        except (CommandError, StackStoreError, ValidationError) as error:
            self._finish_with_error(error)
            return
        self._pending_image_path = None
        result.output_path.unlink(missing_ok=True)
        self._operation = None
        self._request_id = None
        self._request_target = None
        self._set_busy(False, "Image generated")

    def _apply_background(
        self,
        card_id: UUID,
        revision_id: UUID,
        background: GeneratedBackground | ImportedBackground | None,
        message: str,
    ) -> Stack:
        previous_token = self.controller.current_undo_token
        changed = self.controller.execute(
            ReplaceRevisionBackgroundCommand(
                card_id=card_id,
                revision_id=revision_id,
                background=background,
            )
        )
        self.progress_changed.emit(message)
        self.document_changed.emit(changed)
        self._emit_change_applied(message, previous_token)
        return changed

    def _emit_change_applied(
        self,
        message: str,
        previous_token: object,
    ) -> None:
        token = self.controller.current_undo_token
        if token is not None and token != previous_token:
            self.change_applied.emit(message, token)

    def _operation_failed(self, request_id: UUID, failure: object) -> None:
        if request_id == self._request_id:
            self._finish_with_error(failure)

    def _finish_with_error(self, failure: object) -> None:
        self._operation = None
        self._request_id = None
        self._request_target = None
        self._discard_pending_image()
        self._set_busy(False, "Image generation failed")
        self.failed.emit(failure)

    def _discard_pending_image(self) -> None:
        if self._pending_image_path is not None:
            self._pending_image_path.unlink(missing_ok=True)
            self._pending_image_path = None

    def _set_busy(self, busy: bool, progress: str) -> None:
        self._busy = busy
        self.busy_changed.emit(busy)
        self.progress_changed.emit(progress)

    def _require_ready(self, card_id: UUID) -> None:
        if self._busy:
            raise BackgroundWorkflowError("background generation is already running")
        if self.session.store is None:
            raise BackgroundWorkflowError("save the stack before changing an image")
        self._card(self.controller.document, card_id)

    def _require_store(self) -> StackStore:
        store = self.session.store
        if store is None:
            raise BackgroundWorkflowError("save the stack before changing an image")
        return store

    def _target(self, document: Stack, card: Card) -> _GenerationTarget:
        bundle_path = self.session.state.bundle_path
        if bundle_path is None:
            raise BackgroundWorkflowError("save the stack before changing an image")
        revision = card.active_revision
        style = next(
            (style for style in document.styles if style.id == revision.style_id),
            None,
        )
        return _GenerationTarget(
            stack_id=document.id,
            card_id=card.id,
            revision_id=revision.id,
            description=revision.description,
            style_id=revision.style_id,
            style_prompt=style.prompt if style is not None else "",
            background_id=(
                revision.background.id if revision.background is not None else None
            ),
            bundle_path=bundle_path.resolve(),
        )

    def _target_is_current(self, target: _GenerationTarget) -> bool:
        bundle_path = self.session.state.bundle_path
        if bundle_path is None or bundle_path.resolve() != target.bundle_path:
            return False
        document = self.controller.document
        if document.id != target.stack_id:
            return False
        card = next(
            (card for card in document.cards if card.id == target.card_id),
            None,
        )
        if card is None or card.active_revision_id != target.revision_id:
            return False
        revision = card.active_revision
        style = next(
            (style for style in document.styles if style.id == revision.style_id),
            None,
        )
        return (
            revision.description == target.description
            and revision.style_id == target.style_id
            and (style.prompt if style is not None else "") == target.style_prompt
            and (
                revision.background.id
                if revision.background is not None
                else None
            )
            == target.background_id
        )

    @staticmethod
    def _card(document: Stack, card_id: UUID) -> Card:
        card = next(
            (candidate for candidate in document.cards if candidate.id == card_id),
            None,
        )
        if card is None:
            raise BackgroundWorkflowError(f"card {card_id} no longer exists")
        return card


__all__ = [
    "BackgroundGenerationSettings",
    "BackgroundWorkflow",
    "BackgroundWorkflowError",
    "GenerationSettingsProvider",
    "prepare_import_image",
]
