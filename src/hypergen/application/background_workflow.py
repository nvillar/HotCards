"""UI-independent background candidate and revision workflow."""

from __future__ import annotations

import secrets
import tempfile
from collections.abc import Callable, Collection
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Protocol
from uuid import UUID, uuid4

from PIL import Image, ImageOps, UnidentifiedImageError
from PySide6.QtCore import QObject, Signal

from hypergen.application.commands import (
    ActivateRevisionCommand,
    AddImageRevisionCommand,
    DeleteImageRevisionCommand,
)
from hypergen.application.document_controller import DocumentController
from hypergen.application.document_session import DocumentSession
from hypergen.application.workers import AdapterWorkers, WorkerOperation
from hypergen.domain.models import (
    Card,
    DomainModel,
    ImageGenerationInputs,
    ImageGenerationMetadata,
    ImageOrigin,
    ImageRevision,
    RevisionName,
    Stack,
)
from hypergen.generation.image_prompts import compose_image_prompt
from hypergen.generation.mflux_generator import (
    MfluxGenerationRequest,
    MfluxGenerationResult,
    MfluxGenerator,
)
from hypergen.generation.ollama_client import OllamaRuntime, OllamaSettings
from hypergen.generation.revision_naming import (
    OllamaRevisionNamer,
    RevisionNamingRequest,
    RevisionNamingResult,
    unique_revision_name,
)


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
    bundle_path: Path


GenerationSettingsProvider = Callable[[], BackgroundGenerationSettings]
OllamaSettingsProvider = Callable[[], OllamaSettings]


class RevisionNamerProtocol(Protocol):
    def name(self, request: RevisionNamingRequest) -> RevisionNamingResult: ...


RevisionNamerFactory = Callable[[OllamaSettings], RevisionNamerProtocol]


def _default_revision_namer_factory(
    settings: OllamaSettings,
) -> RevisionNamerProtocol:
    return OllamaRevisionNamer(OllamaRuntime(settings))


class BackgroundDraft(DomainModel):
    """One card-local transient background that has not entered the stack."""

    stack_id: UUID
    card_id: UUID
    revision_id: UUID
    name: RevisionName | None = None
    proposed_name: RevisionName | None = None
    image_path: Path
    origin: ImageOrigin
    source_filename: str | None = None
    generation_metadata: ImageGenerationMetadata | None = None
    created_at: datetime


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
    """Decode, orient, crop-to-fill, and write a correctly sized PNG candidate."""
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
            candidate = image.crop(crop_box).resize(
                (width, height),
                Image.Resampling.LANCZOS,
            )
            if candidate.mode not in {"RGB", "RGBA"}:
                candidate = candidate.convert("RGBA" if "A" in candidate.getbands() else "RGB")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            candidate.save(output_path, format="PNG")
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as error:
        output_path.unlink(missing_ok=True)
        raise BackgroundWorkflowError(
            f"Could not import image {source_path.name!r}: {error}"
        ) from error


class BackgroundWorkflow(QObject):
    """Coordinate card-local drafts, workers, storage, and typed commands."""

    drafts_changed = Signal()
    busy_changed = Signal(bool)
    progress_changed = Signal(str)
    failed = Signal(object)
    document_changed = Signal(object)

    def __init__(
        self,
        controller: DocumentController,
        session: DocumentSession,
        workers: AdapterWorkers,
        settings_provider: GenerationSettingsProvider,
        ollama_settings_provider: OllamaSettingsProvider,
        *,
        mflux_generator: MfluxGenerator | None = None,
        revision_namer_factory: RevisionNamerFactory = _default_revision_namer_factory,
        temporary_directory: Path | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.controller = controller
        self.session = session
        self.workers = workers
        self._settings_provider = settings_provider
        self._ollama_settings_provider = ollama_settings_provider
        self._mflux_generator = mflux_generator or MfluxGenerator()
        self._revision_namer_factory = revision_namer_factory
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
        self._drafts: dict[UUID, BackgroundDraft] = {}
        self._operation: WorkerOperation | None = None
        self._request_id: UUID | None = None
        self._request_target: _GenerationTarget | None = None
        self._pending_image_path: Path | None = None
        self._busy = False

    @property
    def drafts(self) -> tuple[BackgroundDraft, ...]:
        return tuple(self._drafts.values())

    @property
    def draft_card_ids(self) -> frozenset[UUID]:
        return frozenset(self._drafts)

    def draft_for(self, card_id: UUID) -> BackgroundDraft | None:
        return self._drafts.get(card_id)

    @property
    def busy(self) -> bool:
        return self._busy

    def generate(
        self,
        card_id: UUID,
        *,
        replace_draft: bool = False,
    ) -> WorkerOperation:
        """Start serialized MFLUX generation from author-controlled text."""
        self._require_ready_for_draft(card_id, replace_draft=replace_draft)
        document = self.controller.document
        card = self._card(document, card_id)
        settings = self._settings_provider()
        ollama_settings = self._ollama_settings_provider()
        inputs = ImageGenerationInputs(
            scene_description=card.scene_description,
            global_style=document.global_style,
            card_style=card.card_style,
        )
        try:
            render_prompt = compose_image_prompt(inputs)
        except ValueError as error:
            raise BackgroundWorkflowError(str(error)) from error
        request_id = uuid4()
        assert self.session.state.bundle_path is not None
        target = _GenerationTarget(
            stack_id=document.id,
            card_id=card_id,
            bundle_path=self.session.state.bundle_path.resolve(),
        )
        self._request_id = request_id
        self._request_target = target
        seed = (
            secrets.randbelow(2_147_483_648)
            if settings.random_seed
            else settings.fixed_seed
        )
        revision_id = uuid4()
        output_path = self._temporary_directory / f"generated-{revision_id}.png"
        request = MfluxGenerationRequest(
            inputs=inputs,
            render_prompt=render_prompt,
            output_path=output_path,
            model_identifier=settings.mflux_model,
            seed=seed,
            width=document.canvas.width,
            height=document.canvas.height,
            step_count=settings.step_count,
            quantization=settings.quantization,
        )
        self._set_busy(True, "Generating background image...")
        operation = self.workers.run_mflux(
            lambda: self._mflux_generator.generate(request),
            stage="generating background image",
        )
        self._operation = operation
        operation.succeeded.connect(
            partial(
                self._image_generated,
                request_id,
                target,
                revision_id,
                ollama_settings,
            )
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
    ) -> BackgroundDraft:
        """Create one transient imported draft at the stack canvas size."""
        self._require_ready_for_draft(card_id, replace_draft=False)
        document = self.controller.document
        self._card(document, card_id)
        revision_id = uuid4()
        output_path = self._temporary_directory / f"import-{revision_id}.png"
        prepare_import_image(
            source_path,
            output_path,
            width=document.canvas.width,
            height=document.canvas.height,
            position_x=position_x,
            position_y=position_y,
        )
        draft = BackgroundDraft(
            stack_id=document.id,
            card_id=card_id,
            revision_id=revision_id,
            image_path=output_path,
            origin=ImageOrigin.IMPORTED,
            source_filename=source_path.name,
            created_at=datetime.now(UTC),
        )
        self._set_draft(draft)
        return draft

    def apply_draft(self, card_id: UUID) -> Stack:
        """Durably copy and atomically apply one card's draft revision."""
        draft = self._drafts.get(card_id)
        if draft is None:
            raise BackgroundWorkflowError("this card has no background draft to accept")
        store = self.session.store
        if store is None:
            raise BackgroundWorkflowError("save the stack before applying a background")
        document = self.controller.document
        if document.id != draft.stack_id:
            raise BackgroundWorkflowError("the background draft belongs to another stack")
        card = self._card(document, draft.card_id)
        revision_name = (
            unique_revision_name(
                draft.proposed_name or draft.name,
                (
                    self._revision_display_name(revision)
                    for revision in card.image_revisions
                ),
            )
            if draft.name is not None
            else None
        )
        image_path = store.import_image(
            draft.image_path,
            card_id=draft.card_id,
            revision_id=draft.revision_id,
        )
        revision = ImageRevision(
            id=draft.revision_id,
            name=revision_name,
            image_path=image_path,
            origin=draft.origin,
            source_filename=draft.source_filename,
            generation_metadata=draft.generation_metadata,
            created_at=draft.created_at,
        )
        changed = self.controller.execute(
            AddImageRevisionCommand(
                card_id=draft.card_id,
                revision=revision,
            )
        )
        self._clear_draft(card_id)
        self.progress_changed.emit("Background revision applied")
        self.document_changed.emit(changed)
        return changed

    def discard_draft(self, card_id: UUID) -> None:
        """Discard only the selected card's transient draft."""
        if card_id in self._drafts:
            self._clear_draft(card_id)
            self.progress_changed.emit("Background draft discarded")

    def discard_all_drafts(self) -> None:
        """Discard all transient drafts in one observable state change."""
        if not self._drafts:
            return
        drafts = tuple(self._drafts.values())
        self._drafts.clear()
        for draft in drafts:
            draft.image_path.unlink(missing_ok=True)
        self.drafts_changed.emit()

    def discard_orphaned_drafts(self, valid_card_ids: Collection[UUID]) -> None:
        """Remove drafts whose cards disappeared through undo or replacement."""
        valid_ids = frozenset(valid_card_ids)
        orphaned_ids = [
            card_id for card_id in self._drafts if card_id not in valid_ids
        ]
        for card_id in orphaned_ids:
            draft = self._drafts.pop(card_id)
            draft.image_path.unlink(missing_ok=True)

    def activate_revision(self, card_id: UUID, revision_id: UUID) -> Stack:
        changed = self.controller.execute(
            ActivateRevisionCommand(card_id=card_id, revision_id=revision_id)
        )
        self.progress_changed.emit("Background revision activated")
        self.document_changed.emit(changed)
        return changed

    def delete_revision(self, card_id: UUID, revision_id: UUID) -> Stack:
        changed = self.controller.execute(
            DeleteImageRevisionCommand(card_id=card_id, revision_id=revision_id)
        )
        self.progress_changed.emit("Background revision deleted")
        self.document_changed.emit(changed)
        return changed

    def cancel(self) -> None:
        if self._operation is not None and not self._operation.is_finished:
            self._operation.cancel()
        self._request_id = None
        self._request_target = None
        self._operation = None
        self._discard_pending_image()
        self._set_busy(False, "Generation cancelled")

    def close(self) -> None:
        self.cancel()
        self.discard_all_drafts()
        if self._owned_temporary_directory is not None:
            self._owned_temporary_directory.cleanup()

    def _image_generated(
        self,
        request_id: UUID,
        target: _GenerationTarget,
        revision_id: UUID,
        ollama_settings: OllamaSettings,
        result: object,
    ) -> None:
        if request_id != self._request_id:
            if isinstance(result, MfluxGenerationResult):
                result.output_path.unlink(missing_ok=True)
            return
        if not self._target_is_current(target):
            if isinstance(result, MfluxGenerationResult):
                result.output_path.unlink(missing_ok=True)
            self._finish_with_error(
                BackgroundWorkflowError(
                    "the stack or card changed before image generation completed"
                )
            )
            return
        if not isinstance(result, MfluxGenerationResult):
            self._finish_with_error(
                BackgroundWorkflowError("image generation returned an unexpected result")
            )
            return
        self._pending_image_path = result.output_path
        card = self._card(self.controller.document, target.card_id)
        existing_names = tuple(
            self._revision_display_name(revision) for revision in card.image_revisions
        )
        request = RevisionNamingRequest(
            image_path=result.output_path,
            render_prompt=result.metadata.render_prompt,
            existing_names=existing_names,
        )
        self.progress_changed.emit("Naming background revision...")
        operation = self.workers.run_ollama(
            lambda: self._revision_namer_factory(ollama_settings).name(request),
            stage="naming background revision",
        )
        self._operation = operation
        operation.succeeded.connect(
            partial(
                self._revision_named,
                request_id,
                target,
                revision_id,
                result,
            )
        )
        operation.failed.connect(partial(self._operation_failed, request_id))

    def _revision_named(
        self,
        request_id: UUID,
        target: _GenerationTarget,
        revision_id: UUID,
        generation_result: MfluxGenerationResult,
        result: object,
    ) -> None:
        if request_id != self._request_id:
            return
        if not self._target_is_current(target):
            self._finish_with_error(
                BackgroundWorkflowError(
                    "the stack or card changed before revision naming completed"
                )
            )
            return
        if not isinstance(result, RevisionNamingResult):
            self._finish_with_error(
                BackgroundWorkflowError("revision naming returned an unexpected result")
            )
            return
        card = self._card(self.controller.document, target.card_id)
        existing_names = (
            self._revision_display_name(revision) for revision in card.image_revisions
        )
        draft = BackgroundDraft(
            stack_id=target.stack_id,
            card_id=target.card_id,
            revision_id=revision_id,
            name=unique_revision_name(result.name, existing_names),
            proposed_name=result.name,
            image_path=generation_result.output_path,
            origin=ImageOrigin.GENERATED,
            generation_metadata=generation_result.metadata,
            created_at=generation_result.metadata.generated_at,
        )
        self._operation = None
        self._request_id = None
        self._request_target = None
        self._pending_image_path = None
        self._set_busy(False, "Background draft ready")
        self._set_draft(draft)

    def _operation_failed(self, request_id: UUID, failure: object) -> None:
        if request_id != self._request_id:
            return
        self._finish_with_error(failure)

    def _finish_with_error(self, failure: object) -> None:
        self._operation = None
        self._request_id = None
        self._request_target = None
        self._discard_pending_image()
        self._set_busy(False, "Background generation failed")
        self.failed.emit(failure)

    def _discard_pending_image(self) -> None:
        if self._pending_image_path is not None:
            self._pending_image_path.unlink(missing_ok=True)
            self._pending_image_path = None

    def _set_draft(self, draft: BackgroundDraft) -> None:
        previous = self._drafts.get(draft.card_id)
        self._drafts[draft.card_id] = draft
        if previous is not None:
            previous.image_path.unlink(missing_ok=True)
        self.drafts_changed.emit()

    def _clear_draft(self, card_id: UUID) -> None:
        draft = self._drafts.pop(card_id, None)
        if draft is not None:
            draft.image_path.unlink(missing_ok=True)
            self.drafts_changed.emit()

    def _set_busy(self, busy: bool, progress: str) -> None:
        self._busy = busy
        self.busy_changed.emit(busy)
        self.progress_changed.emit(progress)

    def _require_ready_for_draft(
        self,
        card_id: UUID,
        *,
        replace_draft: bool,
    ) -> None:
        if self._busy:
            raise BackgroundWorkflowError("background generation is already running")
        if card_id in self._drafts and not replace_draft:
            raise BackgroundWorkflowError("this card already has a background draft")
        if self.session.store is None:
            raise BackgroundWorkflowError("save the stack before creating a background")

    def is_generating_for(self, card_id: UUID) -> bool:
        return (
            self._busy
            and self._request_target is not None
            and self._request_target.card_id == card_id
        )

    def _target_is_current(self, target: _GenerationTarget) -> bool:
        bundle_path = self.session.state.bundle_path
        if bundle_path is None:
            return False
        document = self.controller.document
        return (
            document.id == target.stack_id
            and bundle_path.resolve() == target.bundle_path
            and any(card.id == target.card_id for card in document.cards)
        )

    @staticmethod
    def _revision_display_name(revision: ImageRevision) -> str:
        if revision.name is not None:
            return revision.name
        return "Generated" if revision.origin is ImageOrigin.GENERATED else "Imported"

    @staticmethod
    def _card(document: Stack, card_id: UUID) -> Card:
        card = next((candidate for candidate in document.cards if candidate.id == card_id), None)
        if card is None:
            raise BackgroundWorkflowError(f"card {card_id} no longer exists")
        return card


__all__ = [
    "BackgroundDraft",
    "BackgroundGenerationSettings",
    "BackgroundWorkflow",
    "BackgroundWorkflowError",
    "GenerationSettingsProvider",
    "OllamaSettingsProvider",
    "RevisionNamerFactory",
    "RevisionNamerProtocol",
    "prepare_import_image",
]
