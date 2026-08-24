"""Direct, reviewable Image Prompt preparation workflow."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Protocol
from uuid import UUID, uuid4

from PySide6.QtCore import QObject, Signal

from hypergen.application.commands import SetRevisionImagePromptCommand
from hypergen.application.document_controller import DocumentController
from hypergen.application.generated_revision_change import GeneratedRevisionChange
from hypergen.application.workers import AdapterWorkers, WorkerOperation
from hypergen.domain.models import (
    Card,
    ImagePrompt,
    ImageReferenceSnapshot,
    ResolvedCardReference,
    Stack,
    UnresolvedCardReference,
)
from hypergen.generation.image_prompt_preparation import (
    IMAGE_PROMPT_PREPARATION_VERSION,
    ImagePromptPreparationRequest,
    ImagePromptPreparationResult,
    OllamaImagePromptPreparer,
)
from hypergen.generation.ollama_client import OllamaRuntime, OllamaSettings


class ImagePromptWorkflowError(ValueError):
    """An Image Prompt operation cannot safely proceed."""


class ImagePromptPreparerProtocol(Protocol):
    def prepare(
        self,
        request: ImagePromptPreparationRequest,
        *,
        reference_image_path: Path | None = None,
    ) -> ImagePromptPreparationResult: ...


ImagePromptPreparerFactory = Callable[
    [OllamaSettings],
    ImagePromptPreparerProtocol,
]
OllamaSettingsProvider = Callable[[], OllamaSettings]
ReferenceImageResolver = Callable[[str], Path]


def _default_preparer_factory(
    settings: OllamaSettings,
) -> ImagePromptPreparerProtocol:
    return OllamaImagePromptPreparer(OllamaRuntime(settings))


@dataclass(frozen=True, slots=True)
class _ReferenceTarget:
    card_id: UUID
    revision_id: UUID
    background_id: UUID
    image_path: str
    asset_path: Path

    @property
    def snapshot(self) -> ImageReferenceSnapshot:
        return ImageReferenceSnapshot(
            card_id=self.card_id,
            revision_id=self.revision_id,
            background_id=self.background_id,
        )


@dataclass(frozen=True, slots=True)
class _ImagePromptTarget:
    stack_id: UUID
    card_id: UUID
    revision_id: UUID
    source_description: str
    reference: _ReferenceTarget | None


class ImagePromptWorkflow(QObject):
    """Prepare and apply one Image Prompt through an undoable command."""

    busy_changed = Signal(bool)
    progress_changed = Signal(str)
    failed = Signal(object)
    document_changed = Signal(object)
    change_applied = Signal(str, object)
    generation_applied = Signal(object)

    def __init__(
        self,
        controller: DocumentController,
        workers: AdapterWorkers,
        settings_provider: OllamaSettingsProvider,
        *,
        reference_image_resolver: ReferenceImageResolver | None = None,
        preparer_factory: ImagePromptPreparerFactory = _default_preparer_factory,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.controller = controller
        self.workers = workers
        self._settings_provider = settings_provider
        self._reference_image_resolver = reference_image_resolver
        self._preparer_factory = preparer_factory
        self._operation: WorkerOperation | None = None
        self._request_id: UUID | None = None
        self._target: _ImagePromptTarget | None = None
        self._busy = False

    @property
    def busy(self) -> bool:
        return self._busy

    def start(self, card_id: UUID) -> WorkerOperation:
        if self._busy:
            raise ImagePromptWorkflowError(
                "Image Prompt preparation is already running"
            )
        document = self.controller.document
        card = self._card(document, card_id)
        revision = card.active_revision
        if not revision.description.strip():
            raise ImagePromptWorkflowError(
                "enter a Description before preparing its Image Prompt"
            )
        target = _ImagePromptTarget(
            stack_id=document.id,
            card_id=card.id,
            revision_id=revision.id,
            source_description=revision.description,
            reference=self._reference_target(document, card),
        )
        settings = self._settings_provider()
        request_id = uuid4()
        self._request_id = request_id
        self._target = target
        self._set_busy(True, "Preparing Image Prompt...")
        operation = self.workers.run_ollama(
            lambda: self._run_preparation(settings, target),
            stage="preparing Image Prompt",
        )
        self._operation = operation
        operation.succeeded.connect(
            partial(self._preparation_succeeded, request_id, target)
        )
        operation.failed.connect(
            partial(
                self._operation_failed,
                request_id,
                "Image Prompt preparation failed",
            )
        )
        return operation

    def cancel(self) -> None:
        if self._operation is not None and not self._operation.is_finished:
            self._operation.cancel()
        self._operation = None
        self._request_id = None
        self._target = None
        if self._busy:
            self._set_busy(False, "Image Prompt preparation cancelled")

    def close(self) -> None:
        self.cancel()

    def _preparation_succeeded(
        self,
        request_id: UUID,
        target: _ImagePromptTarget,
        result: object,
    ) -> None:
        if request_id != self._request_id:
            return
        if not self._target_is_current(target):
            self._finish_with_error(
                ImagePromptWorkflowError(
                    "the stack, card, revision, Description, or Reference "
                    "changed before the Image Prompt was prepared"
                ),
                "Image Prompt preparation failed",
            )
            return
        if not isinstance(result, ImagePromptPreparationResult):
            self._finish_with_error(
                ImagePromptWorkflowError(
                    "Image Prompt preparation returned an unexpected result"
                ),
                "Image Prompt preparation failed",
            )
            return
        previous_token = self.controller.current_undo_token
        previous_revision = next(
            revision
            for revision in self._card(
                self.controller.document,
                target.card_id,
            ).revisions
            if revision.id == target.revision_id
        )
        changed = self.controller.execute(
            SetRevisionImagePromptCommand(
                card_id=target.card_id,
                revision_id=target.revision_id,
                value=ImagePrompt(
                    text=result.image_prompt,
                    source_description=target.source_description,
                    reference=(
                        target.reference.snapshot
                        if target.reference is not None
                        else None
                    ),
                    model_identifier=result.model_identifier,
                    prompt_version=result.prompt_version,
                ),
            )
        )
        self._operation = None
        self._request_id = None
        self._target = None
        self._set_busy(False, "Image Prompt prepared")
        self.document_changed.emit(changed)
        token = self.controller.current_undo_token
        if token is not None and token != previous_token:
            self.generation_applied.emit(
                GeneratedRevisionChange(
                    message="Image Prompt prepared",
                    token=token,
                    card_id=target.card_id,
                    revision_id=target.revision_id,
                    previous_revision=previous_revision,
                )
            )

    def _operation_failed(
        self,
        request_id: UUID,
        progress: str,
        failure: object,
    ) -> None:
        if request_id == self._request_id:
            self._finish_with_error(failure, progress)

    def _finish_with_error(self, failure: object, progress: str) -> None:
        self._operation = None
        self._request_id = None
        self._target = None
        self._set_busy(False, progress)
        self.failed.emit(failure)

    def _set_busy(self, busy: bool, progress: str) -> None:
        self._busy = busy
        self.busy_changed.emit(busy)
        self.progress_changed.emit(progress)

    def _run_preparation(
        self,
        settings: OllamaSettings,
        target: _ImagePromptTarget,
    ) -> ImagePromptPreparationResult:
        return self._preparer_factory(settings).prepare(
            ImagePromptPreparationRequest(
                description=target.source_description,
                has_reference=target.reference is not None,
            ),
            reference_image_path=(
                target.reference.asset_path
                if target.reference is not None
                else None
            ),
        )

    def _target_is_current(self, target: _ImagePromptTarget) -> bool:
        document = self.controller.document
        if document.id != target.stack_id:
            return False
        card = next(
            (candidate for candidate in document.cards if candidate.id == target.card_id),
            None,
        )
        if (
            card is None
            or card.active_revision_id != target.revision_id
            or card.active_revision.description != target.source_description
        ):
            return False
        try:
            return self._reference_target(document, card) == target.reference
        except ImagePromptWorkflowError:
            return False

    def _reference_target(
        self,
        document: Stack,
        card: Card,
    ) -> _ReferenceTarget | None:
        assignment = card.active_revision.reference
        if assignment is None:
            return None
        if isinstance(assignment, UnresolvedCardReference):
            name = assignment.target_name or "unknown card"
            raise ImagePromptWorkflowError(
                f"Reference {name!r} is unresolved"
            )
        assert isinstance(assignment, ResolvedCardReference)
        source = next(
            (
                candidate
                for candidate in document.cards
                if candidate.id == assignment.target_card_id
            ),
            None,
        )
        if source is None:
            raise ImagePromptWorkflowError(
                "Reference card no longer exists"
            )
        source_revision = source.active_revision
        background = source_revision.background
        if background is None:
            raise ImagePromptWorkflowError(
                f"Reference card {source.name!r} has no image"
            )
        if self._reference_image_resolver is None:
            raise ImagePromptWorkflowError(
                "save the stack before using a Reference image"
            )
        try:
            asset_path = self._reference_image_resolver(
                background.image_path
            )
        except (OSError, ValueError) as error:
            raise ImagePromptWorkflowError(str(error)) from error
        if not asset_path.is_file():
            raise ImagePromptWorkflowError(
                f"Reference image for {source.name!r} is unavailable"
            )
        return _ReferenceTarget(
            card_id=source.id,
            revision_id=source_revision.id,
            background_id=background.id,
            image_path=background.image_path,
            asset_path=asset_path,
        )

    @staticmethod
    def _card(document: Stack, card_id: UUID) -> Card:
        card = next(
            (candidate for candidate in document.cards if candidate.id == card_id),
            None,
        )
        if card is None:
            raise ImagePromptWorkflowError(f"card {card_id} no longer exists")
        return card


def image_prompt_reference_snapshot(
    document: Stack,
    card: Card,
) -> ImageReferenceSnapshot | None:
    """Return exact usable Reference provenance for freshness checks."""
    assignment = card.active_revision.reference
    if not isinstance(assignment, ResolvedCardReference):
        return None
    source = next(
        (
            candidate
            for candidate in document.cards
            if candidate.id == assignment.target_card_id
        ),
        None,
    )
    if source is None or source.active_revision.background is None:
        return None
    return ImageReferenceSnapshot(
        card_id=source.id,
        revision_id=source.active_revision.id,
        background_id=source.active_revision.background.id,
    )


__all__ = [
    "IMAGE_PROMPT_PREPARATION_VERSION",
    "ImagePromptPreparerFactory",
    "ImagePromptPreparerProtocol",
    "ImagePromptWorkflow",
    "ImagePromptWorkflowError",
    "OllamaSettingsProvider",
    "ReferenceImageResolver",
    "image_prompt_reference_snapshot",
]
