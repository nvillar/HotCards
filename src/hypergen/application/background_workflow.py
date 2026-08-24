"""Direct, revision-local background image workflow."""

from __future__ import annotations

import secrets
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from uuid import UUID, uuid4

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
from hypergen.application.generated_revision_change import GeneratedRevisionChange
from hypergen.application.workers import AdapterWorkers, WorkerOperation
from hypergen.domain.models import (
    Card,
    CardRevision,
    GeneratedBackground,
    ImageGenerationInputs,
    ImageReferenceSnapshot,
    ReferenceRole,
    ResolvedCardReference,
    Stack,
    UnresolvedCardReference,
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
    background_id: UUID | None
    bundle_path: Path
    references: tuple[_GenerationReferenceTarget, ...]


@dataclass(frozen=True, slots=True)
class _GenerationReferenceTarget:
    role: ReferenceRole
    card_id: UUID
    revision_id: UUID
    background_id: UUID
    image_path: str


GenerationSettingsProvider = Callable[[], BackgroundGenerationSettings]


class BackgroundWorkflow(QObject):
    """Generate, clear, and manage complete card revisions."""

    busy_changed = Signal(bool)
    progress_changed = Signal(str)
    failed = Signal(object)
    document_changed = Signal(object)
    change_applied = Signal(str, object)
    generation_applied = Signal(object)

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
        references = self._resolve_references(document, card)
        snapshots = {
            f"{reference.role.value}_reference": ImageReferenceSnapshot(
                card_id=reference.card_id,
                revision_id=reference.revision_id,
                background_id=reference.background_id,
            )
            for reference in references
        }
        inputs = ImageGenerationInputs(
            description=revision.description,
            enriched_description=(
                revision.enriched_description.text
                if revision.enriched_description is not None
                else None
            ),
            **snapshots,
        )
        try:
            render_prompt = compose_image_prompt(inputs)
        except ValueError as error:
            raise BackgroundWorkflowError(str(error)) from error
        settings = self._settings_provider()
        reference_image_paths = tuple(
            self._reference_asset_path(reference)
            for reference in self._unique_references(references)
        )
        request_id = uuid4()
        asset_id = uuid4()
        target = self._target(document, card, references)
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
            reference_image_paths=reference_image_paths,
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
                    "the stack, revision, Description, or image changed "
                    "before generation completed"
                )
            )
            return
        try:
            image_path = self._require_store().store_image_asset(
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
                generated=True,
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
        background: GeneratedBackground | None,
        message: str,
        *,
        generated: bool = False,
    ) -> Stack:
        previous_token = self.controller.current_undo_token
        previous_revision = next(
            revision
            for revision in self._card(self.controller.document, card_id).revisions
            if revision.id == revision_id
        )
        changed = self.controller.execute(
            ReplaceRevisionBackgroundCommand(
                card_id=card_id,
                revision_id=revision_id,
                background=background,
            )
        )
        self.progress_changed.emit(message)
        self.document_changed.emit(changed)
        self._emit_change_applied(
            message,
            previous_token,
            card_id=card_id if generated else None,
            revision_id=revision_id if generated else None,
            previous_revision=previous_revision if generated else None,
        )
        return changed

    def _emit_change_applied(
        self,
        message: str,
        previous_token: object,
        *,
        card_id: UUID | None = None,
        revision_id: UUID | None = None,
        previous_revision: CardRevision | None = None,
    ) -> None:
        token = self.controller.current_undo_token
        if token is not None and token != previous_token:
            if (
                card_id is not None
                and revision_id is not None
                and previous_revision is not None
            ):
                self.generation_applied.emit(
                    GeneratedRevisionChange(
                        message=message,
                        token=token,
                        card_id=card_id,
                        revision_id=revision_id,
                        previous_revision=previous_revision,
                    )
                )
            else:
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

    def _target(
        self,
        document: Stack,
        card: Card,
        references: tuple[_GenerationReferenceTarget, ...],
    ) -> _GenerationTarget:
        bundle_path = self.session.state.bundle_path
        if bundle_path is None:
            raise BackgroundWorkflowError("save the stack before changing an image")
        revision = card.active_revision
        return _GenerationTarget(
            stack_id=document.id,
            card_id=card.id,
            revision_id=revision.id,
            description=(
                revision.enriched_description.text
                if revision.enriched_description is not None
                else revision.description
            ),
            background_id=(
                revision.background.id if revision.background is not None else None
            ),
            bundle_path=bundle_path.resolve(),
            references=references,
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
        if not (
            (
                revision.enriched_description.text
                if revision.enriched_description is not None
                else revision.description
            )
            == target.description
            and (
                revision.background.id
                if revision.background is not None
                else None
            )
            == target.background_id
        ):
            return False
        for reference in target.references:
            assignment = getattr(revision, reference.role.value)
            if not (
                isinstance(assignment, ResolvedCardReference)
                and assignment.target_card_id == reference.card_id
            ):
                return False
            source = next(
                (
                    candidate
                    for candidate in document.cards
                    if candidate.id == reference.card_id
                ),
                None,
            )
            if (
                source is None
                or source.active_revision_id != reference.revision_id
                or source.active_revision.background is None
                or source.active_revision.background.id != reference.background_id
            ):
                return False
        return len(target.references) == sum(
            getattr(revision, role.value) is not None
            for role in ReferenceRole
        )

    def _resolve_references(
        self,
        document: Stack,
        card: Card,
    ) -> tuple[_GenerationReferenceTarget, ...]:
        references: list[_GenerationReferenceTarget] = []
        for role in ReferenceRole:
            assignment = getattr(card.active_revision, role.value)
            if assignment is None:
                continue
            label = role.value.replace("_", " ").title()
            if isinstance(assignment, UnresolvedCardReference):
                name = assignment.target_name or "unknown card"
                raise BackgroundWorkflowError(
                    f"{label} reference {name!r} is unresolved"
                )
            source = next(
                (
                    candidate
                    for candidate in document.cards
                    if candidate.id == assignment.target_card_id
                ),
                None,
            )
            if source is None:
                raise BackgroundWorkflowError(
                    f"{label} reference card no longer exists"
                )
            source_revision = source.active_revision
            if source_revision.background is None:
                raise BackgroundWorkflowError(
                    f"{label} reference card {source.name!r} has no image"
                )
            references.append(
                _GenerationReferenceTarget(
                    role=role,
                    card_id=source.id,
                    revision_id=source_revision.id,
                    background_id=source_revision.background.id,
                    image_path=source_revision.background.image_path,
                )
            )
        return tuple(references)

    def _reference_asset_path(
        self,
        reference: _GenerationReferenceTarget,
    ) -> Path:
        path = self._require_store().asset_path(reference.image_path)
        if not path.is_file():
            raise BackgroundWorkflowError(
                f"{reference.role.value.replace('_', ' ').title()} "
                "reference image is unavailable"
            )
        return path

    @staticmethod
    def _unique_references(
        references: tuple[_GenerationReferenceTarget, ...],
    ) -> tuple[_GenerationReferenceTarget, ...]:
        unique: list[_GenerationReferenceTarget] = []
        seen: set[tuple[UUID, UUID, UUID]] = set()
        for reference in references:
            key = (
                reference.card_id,
                reference.revision_id,
                reference.background_id,
            )
            if key not in seen:
                seen.add(key)
                unique.append(reference)
        return tuple(unique)

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
]
