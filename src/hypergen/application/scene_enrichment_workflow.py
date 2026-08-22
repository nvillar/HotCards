"""Direct, identity-bound Description enrichment workflow."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Protocol
from uuid import UUID, uuid4

from PySide6.QtCore import QObject, Signal

from hypergen.application.commands import EditRevisionDescriptionCommand
from hypergen.application.document_controller import DocumentController
from hypergen.application.workers import AdapterWorkers, WorkerOperation
from hypergen.domain.models import Card, CardRevision, Stack
from hypergen.generation.image_description import (
    ImageDescriptionRequest,
    ImageDescriptionResult,
    OllamaImageDescriber,
)
from hypergen.generation.ollama_client import OllamaRuntime, OllamaSettings
from hypergen.generation.scene_enrichment import (
    OllamaSceneEnricher,
    SceneEnrichmentRequest,
    SceneEnrichmentResult,
)


class SceneEnrichmentWorkflowError(ValueError):
    """A Description enrichment action cannot safely proceed."""


class SceneEnricherProtocol(Protocol):
    def enrich(self, request: SceneEnrichmentRequest) -> SceneEnrichmentResult: ...


class ImageDescriberProtocol(Protocol):
    def describe(self, request: ImageDescriptionRequest) -> ImageDescriptionResult: ...


SceneEnricherFactory = Callable[[OllamaSettings], SceneEnricherProtocol]
ImageDescriberFactory = Callable[[OllamaSettings], ImageDescriberProtocol]
OllamaSettingsProvider = Callable[[], OllamaSettings]
ImagePathResolver = Callable[[str], Path | None]


def _default_enricher_factory(settings: OllamaSettings) -> SceneEnricherProtocol:
    return OllamaSceneEnricher(OllamaRuntime(settings))


def _default_describer_factory(settings: OllamaSettings) -> ImageDescriberProtocol:
    return OllamaImageDescriber(OllamaRuntime(settings))


@dataclass(frozen=True, slots=True)
class _EnrichmentTarget:
    stack_id: UUID
    card_id: UUID
    revision_id: UUID
    source_description: str
    style_id: UUID | None
    style_prompt: str
    background_id: UUID | None
    background_path: str | None
    resolved_image_path: Path | None


class SceneEnrichmentWorkflow(QObject):
    """Enrich the active revision Description and apply it immediately."""

    busy_changed = Signal(bool)
    progress_changed = Signal(str)
    failed = Signal(object)
    document_changed = Signal(object)
    change_applied = Signal(str, object)

    def __init__(
        self,
        controller: DocumentController,
        workers: AdapterWorkers,
        settings_provider: OllamaSettingsProvider,
        image_path_resolver: ImagePathResolver | None = None,
        *,
        enricher_factory: SceneEnricherFactory = _default_enricher_factory,
        describer_factory: ImageDescriberFactory = _default_describer_factory,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.controller = controller
        self.workers = workers
        self._settings_provider = settings_provider
        self._image_path_resolver = image_path_resolver
        self._enricher_factory = enricher_factory
        self._describer_factory = describer_factory
        self._operation: WorkerOperation | None = None
        self._request_id: UUID | None = None
        self._target: _EnrichmentTarget | None = None
        self._busy = False

    @property
    def busy(self) -> bool:
        return self._busy

    def start(self, card_id: UUID) -> WorkerOperation:
        if self._busy:
            raise SceneEnrichmentWorkflowError(
                "Description enrichment is already running"
            )
        document = self.controller.document
        card = self._card(document, card_id)
        revision = card.active_revision
        style_prompt = self._style_prompt(document, revision)
        image_path = self._readable_image(revision)
        if not revision.description.strip() and image_path is None:
            raise SceneEnrichmentWorkflowError(
                "enter a Description or add a readable image before enriching it"
            )
        target = _EnrichmentTarget(
            stack_id=document.id,
            card_id=card.id,
            revision_id=revision.id,
            source_description=revision.description,
            style_id=revision.style_id,
            style_prompt=style_prompt,
            background_id=(
                revision.background.id if revision.background is not None else None
            ),
            background_path=revision.image_path,
            resolved_image_path=image_path,
        )
        settings = self._settings_provider()
        request_id = uuid4()
        self._request_id = request_id
        self._target = target
        if image_path is not None:
            self._set_busy(True, "Describing image...")
            operation = self.workers.run_ollama(
                lambda: self._describer_factory(settings).describe(
                    ImageDescriptionRequest(image_path=image_path)
                ),
                stage="describing background",
            )
            self._operation = operation
            operation.succeeded.connect(
                partial(self._description_succeeded, request_id, target, settings)
            )
            operation.failed.connect(
                partial(
                    self._operation_failed,
                    request_id,
                    "Image description failed",
                )
            )
            return operation
        self._set_busy(True, "Enriching Description...")
        return self._start_enrichment(request_id, target, settings)

    def cancel(self) -> None:
        if self._operation is not None and not self._operation.is_finished:
            self._operation.cancel()
        self._operation = None
        self._request_id = None
        self._target = None
        if self._busy:
            self._set_busy(False, "Description enrichment cancelled")

    def close(self) -> None:
        self.cancel()

    def _start_enrichment(
        self,
        request_id: UUID,
        target: _EnrichmentTarget,
        settings: OllamaSettings,
        image_description: str | None = None,
    ) -> WorkerOperation:
        request = SceneEnrichmentRequest(
            scene=target.source_description,
            effective_style=target.style_prompt,
            image_description=image_description,
        )
        operation = self.workers.run_ollama(
            lambda: self._enricher_factory(settings).enrich(request),
            stage="enriching Description",
        )
        self._operation = operation
        operation.succeeded.connect(
            partial(self._enrichment_succeeded, request_id, target)
        )
        operation.failed.connect(
            partial(
                self._operation_failed,
                request_id,
                "Description enrichment failed",
            )
        )
        return operation

    def _description_succeeded(
        self,
        request_id: UUID,
        target: _EnrichmentTarget,
        settings: OllamaSettings,
        result: object,
    ) -> None:
        if request_id != self._request_id:
            return
        if not self._target_is_current(target):
            self._finish_with_error(self._stale_error(), "Image description failed")
            return
        if not isinstance(result, ImageDescriptionResult):
            self._finish_with_error(
                SceneEnrichmentWorkflowError(
                    "image description returned an unexpected result"
                ),
                "Image description failed",
            )
            return
        self.progress_changed.emit("Enriching Description...")
        self._start_enrichment(request_id, target, settings, result.scene)

    def _enrichment_succeeded(
        self,
        request_id: UUID,
        target: _EnrichmentTarget,
        result: object,
    ) -> None:
        if request_id != self._request_id:
            return
        if not self._target_is_current(target):
            self._finish_with_error(self._stale_error(), "Description enrichment failed")
            return
        if not isinstance(result, SceneEnrichmentResult):
            self._finish_with_error(
                SceneEnrichmentWorkflowError(
                    "Description enrichment returned an unexpected result"
                ),
                "Description enrichment failed",
            )
            return
        previous_token = self.controller.current_undo_token
        changed = self.controller.execute(
            EditRevisionDescriptionCommand(
                card_id=target.card_id,
                revision_id=target.revision_id,
                value=result.scene,
            )
        )
        self._operation = None
        self._request_id = None
        self._target = None
        self._set_busy(False, "Description enriched")
        self.document_changed.emit(changed)
        token = self.controller.current_undo_token
        if token is not None and token != previous_token:
            self.change_applied.emit("Description enriched", token)

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

    def _target_is_current(self, target: _EnrichmentTarget) -> bool:
        document = self.controller.document
        if document.id != target.stack_id:
            return False
        card = next(
            (candidate for candidate in document.cards if candidate.id == target.card_id),
            None,
        )
        if card is None or card.active_revision_id != target.revision_id:
            return False
        revision = card.active_revision
        if (
            revision.description != target.source_description
            or revision.style_id != target.style_id
            or self._style_prompt(document, revision) != target.style_prompt
            or revision.image_path != target.background_path
            or (
                revision.background.id
                if revision.background is not None
                else None
            )
            != target.background_id
        ):
            return False
        if target.resolved_image_path is None:
            return True
        image_path = self._readable_image(revision)
        return image_path == target.resolved_image_path

    def _readable_image(self, revision: CardRevision) -> Path | None:
        if revision.image_path is None or self._image_path_resolver is None:
            return None
        image_path = self._image_path_resolver(revision.image_path)
        if image_path is None or not image_path.is_file():
            return None
        return image_path

    @staticmethod
    def _style_prompt(document: Stack, revision: CardRevision) -> str:
        style = next(
            (style for style in document.styles if style.id == revision.style_id),
            None,
        )
        return style.prompt if style is not None else ""

    @staticmethod
    def _card(document: Stack, card_id: UUID) -> Card:
        card = next(
            (candidate for candidate in document.cards if candidate.id == card_id),
            None,
        )
        if card is None:
            raise SceneEnrichmentWorkflowError(f"card {card_id} no longer exists")
        return card

    @staticmethod
    def _stale_error() -> SceneEnrichmentWorkflowError:
        return SceneEnrichmentWorkflowError(
            "the stack, card, revision, image, Description, or Style changed "
            "before enrichment completed"
        )


__all__ = [
    "ImageDescriberFactory",
    "ImageDescriberProtocol",
    "ImagePathResolver",
    "OllamaSettingsProvider",
    "SceneEnricherFactory",
    "SceneEnricherProtocol",
    "SceneEnrichmentWorkflow",
    "SceneEnrichmentWorkflowError",
]
