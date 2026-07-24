"""Identity-bound image-to-Scene replacement workflow."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Protocol
from uuid import UUID, uuid4

from PySide6.QtCore import QObject, Signal

from hypergen.application.commands import EditCardTextCommand
from hypergen.application.document_controller import DocumentController
from hypergen.application.workers import AdapterWorkers, WorkerOperation
from hypergen.domain.models import Card, ImageRevision, Stack
from hypergen.generation.image_description import (
    ImageDescriptionRequest,
    ImageDescriptionResult,
    OllamaImageDescriber,
)
from hypergen.generation.ollama_client import OllamaRuntime, OllamaSettings


class ImageDescriptionWorkflowError(ValueError):
    """An image-description action cannot safely proceed."""


class ImageDescriberProtocol(Protocol):
    def describe(self, request: ImageDescriptionRequest) -> ImageDescriptionResult: ...


ImageDescriberFactory = Callable[[OllamaSettings], ImageDescriberProtocol]
OllamaSettingsProvider = Callable[[], OllamaSettings]
ImagePathResolver = Callable[[str], Path | None]


def _default_describer_factory(settings: OllamaSettings) -> ImageDescriberProtocol:
    return OllamaImageDescriber(OllamaRuntime(settings))


@dataclass(frozen=True, slots=True)
class _DescriptionTarget:
    stack_id: UUID
    card_id: UUID
    revision_id: UUID
    image_path: str
    source_scene: str


class ImageDescriptionWorkflow(QObject):
    """Describe one active revision and replace Scene through one command."""

    busy_changed = Signal(bool)
    progress_changed = Signal(str)
    failed = Signal(object)
    document_changed = Signal(object)

    def __init__(
        self,
        controller: DocumentController,
        workers: AdapterWorkers,
        settings_provider: OllamaSettingsProvider,
        image_path_resolver: ImagePathResolver,
        *,
        describer_factory: ImageDescriberFactory = _default_describer_factory,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.controller = controller
        self.workers = workers
        self._settings_provider = settings_provider
        self._image_path_resolver = image_path_resolver
        self._describer_factory = describer_factory
        self._operation: WorkerOperation | None = None
        self._request_id: UUID | None = None
        self._target: _DescriptionTarget | None = None
        self._busy = False

    @property
    def busy(self) -> bool:
        return self._busy

    def start(self, card_id: UUID) -> WorkerOperation:
        if self._busy:
            raise ImageDescriptionWorkflowError("image description is already running")
        document = self.controller.document
        card = self._card(document, card_id)
        revision = self._active_revision(card)
        image_path = self._image_path_resolver(revision.image_path)
        if image_path is None or not image_path.is_file():
            raise ImageDescriptionWorkflowError(
                "the active background image is unavailable"
            )
        request = ImageDescriptionRequest(image_path=image_path)
        target = _DescriptionTarget(
            stack_id=document.id,
            card_id=card.id,
            revision_id=revision.id,
            image_path=revision.image_path,
            source_scene=card.scene_description,
        )
        settings = self._settings_provider()
        request_id = uuid4()
        self._request_id = request_id
        self._target = target
        self._set_busy(True, "Describing active image...")
        operation = self.workers.run_ollama(
            lambda: self._describer_factory(settings).describe(request),
            stage="describing image",
        )
        self._operation = operation
        operation.succeeded.connect(
            partial(self._description_succeeded, request_id, target)
        )
        operation.failed.connect(partial(self._operation_failed, request_id))
        return operation

    def cancel(self) -> None:
        if not self._busy:
            return
        if self._operation is not None and not self._operation.is_finished:
            self._operation.cancel()
        self._operation = None
        self._request_id = None
        self._target = None
        self._set_busy(False, "Image description cancelled")

    def close(self) -> None:
        self.cancel()

    def _description_succeeded(
        self,
        request_id: UUID,
        target: _DescriptionTarget,
        result: object,
    ) -> None:
        if request_id != self._request_id:
            return
        if not self._target_is_current(target):
            self._finish_with_error(
                ImageDescriptionWorkflowError(
                    "the stack, card, active revision, or Scene changed before "
                    "image description completed"
                )
            )
            return
        if not isinstance(result, ImageDescriptionResult):
            self._finish_with_error(
                ImageDescriptionWorkflowError(
                    "image description returned an unexpected result"
                )
            )
            return
        changed = self.controller.execute(
            EditCardTextCommand(
                card_id=target.card_id,
                field="scene_description",
                value=result.scene,
            )
        )
        self._operation = None
        self._request_id = None
        self._target = None
        self._set_busy(False, "Scene replaced from active image")
        self.document_changed.emit(changed)

    def _operation_failed(self, request_id: UUID, failure: object) -> None:
        if request_id == self._request_id:
            self._finish_with_error(failure)

    def _finish_with_error(self, failure: object) -> None:
        self._operation = None
        self._request_id = None
        self._target = None
        self._set_busy(False, "Image description failed")
        self.failed.emit(failure)

    def _set_busy(self, busy: bool, progress: str) -> None:
        self._busy = busy
        self.busy_changed.emit(busy)
        self.progress_changed.emit(progress)

    def _target_is_current(self, target: _DescriptionTarget) -> bool:
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
            or card.scene_description != target.source_scene
        ):
            return False
        revision = next(
            (
                revision
                for revision in card.image_revisions
                if revision.id == target.revision_id
            ),
            None,
        )
        return revision is not None and revision.image_path == target.image_path

    @staticmethod
    def _card(document: Stack, card_id: UUID) -> Card:
        card = next(
            (candidate for candidate in document.cards if candidate.id == card_id),
            None,
        )
        if card is None:
            raise ImageDescriptionWorkflowError(f"card {card_id} no longer exists")
        return card

    @staticmethod
    def _active_revision(card: Card) -> ImageRevision:
        revision = next(
            (
                revision
                for revision in card.image_revisions
                if revision.id == card.active_revision_id
            ),
            None,
        )
        if revision is None:
            raise ImageDescriptionWorkflowError(
                "apply a background before describing its image"
            )
        return revision


__all__ = [
    "ImageDescriberFactory",
    "ImageDescriberProtocol",
    "ImageDescriptionWorkflow",
    "ImageDescriptionWorkflowError",
    "ImagePathResolver",
    "OllamaSettingsProvider",
]
