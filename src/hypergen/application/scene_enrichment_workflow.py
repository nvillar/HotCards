"""Transient, identity-bound Description enrichment workflow."""

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
from hypergen.domain.models import Card, DomainModel, ImageRevision, Stack
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
    source_scene: str
    effective_style: str
    active_revision_id: UUID | None = None
    revision_id: UUID | None = None
    revision_image_path: str | None = None
    resolved_image_path: Path | None = None


class SceneEnrichmentDraft(DomainModel):
    """One reviewable rewrite kept outside the authoritative Stack."""

    stack_id: UUID
    card_id: UUID
    source_scene: str
    effective_style: str
    active_revision_id: UUID | None = None
    revision_id: UUID | None = None
    revision_image_path: str | None = None
    resolved_image_path: Path | None = None
    enriched_scene: str
    model_identifier: str
    prompt_version: str


class SceneEnrichmentWorkflow(QObject):
    """Coordinate one review-first Description enrichment operation."""

    draft_changed = Signal()
    busy_changed = Signal(bool)
    progress_changed = Signal(str)
    failed = Signal(object)
    document_changed = Signal(object)

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
        self._draft: SceneEnrichmentDraft | None = None
        self._operation: WorkerOperation | None = None
        self._request_id: UUID | None = None
        self._target: _EnrichmentTarget | None = None
        self._busy = False

    @property
    def draft(self) -> SceneEnrichmentDraft | None:
        return self._draft

    @property
    def busy(self) -> bool:
        return self._busy

    def start(self, card_id: UUID) -> WorkerOperation:
        if self._busy:
            raise SceneEnrichmentWorkflowError(
                "Description enrichment is already running"
            )
        if self._draft is not None:
            raise SceneEnrichmentWorkflowError(
                "accept or discard the current enriched Description before starting another"
            )
        document = self.controller.document
        card = self._card(document, card_id)
        effective_style = (
            card.card_style if card.card_style is not None else document.global_style
        )
        revision, image_path = self._readable_active_image(card)
        if not card.scene_description.strip() and image_path is None:
            raise SceneEnrichmentWorkflowError(
                "enter a Description or apply a readable background before enriching it"
            )
        target = _EnrichmentTarget(
            stack_id=document.id,
            card_id=card.id,
            source_scene=card.scene_description,
            effective_style=effective_style,
            active_revision_id=card.active_revision_id,
            revision_id=revision.id if revision is not None else None,
            revision_image_path=revision.image_path if revision is not None else None,
            resolved_image_path=image_path,
        )
        settings = self._settings_provider()
        request_id = uuid4()
        self._request_id = request_id
        self._target = target
        if image_path is not None:
            self._set_busy(True, "Describing background...")
            operation = self.workers.run_ollama(
                lambda: self._describer_factory(settings).describe(
                    ImageDescriptionRequest(image_path=image_path)
                ),
                stage="describing background",
            )
            self._operation = operation
            operation.succeeded.connect(
                partial(
                    self._description_succeeded,
                    request_id,
                    target,
                    settings,
                )
            )
            operation.failed.connect(
                partial(
                    self._operation_failed,
                    request_id,
                    "Background description failed",
                )
            )
            return operation
        self._set_busy(True, "Enriching Description...")
        return self._start_enrichment(request_id, target, settings)

    def _start_enrichment(
        self,
        request_id: UUID,
        target: _EnrichmentTarget,
        settings: OllamaSettings,
        image_description: str | None = None,
    ) -> WorkerOperation:
        request = SceneEnrichmentRequest(
            scene=target.source_scene,
            effective_style=target.effective_style,
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

    def apply(self, card_id: UUID, enriched_scene: str) -> Stack:
        draft = self._draft
        if draft is None or draft.card_id != card_id:
            raise SceneEnrichmentWorkflowError(
                "this card has no enriched Description to accept"
            )
        if not enriched_scene.strip():
            raise SceneEnrichmentWorkflowError(
                "the enriched Description must not be empty"
            )
        target = _EnrichmentTarget(
            stack_id=draft.stack_id,
            card_id=draft.card_id,
            source_scene=draft.source_scene,
            effective_style=draft.effective_style,
            active_revision_id=draft.active_revision_id,
            revision_id=draft.revision_id,
            revision_image_path=draft.revision_image_path,
            resolved_image_path=draft.resolved_image_path,
        )
        if not self._target_is_current(target):
            self.discard()
            raise SceneEnrichmentWorkflowError(
                "the Description, Style, or background changed before the "
                "enrichment was accepted"
            )
        changed = self.controller.execute(
            EditCardTextCommand(
                card_id=card_id,
                field="scene_description",
                value=enriched_scene,
            )
        )
        self._clear_draft()
        self.progress_changed.emit("Enriched Description accepted")
        self.document_changed.emit(changed)
        return changed

    def discard(self) -> None:
        if self._draft is None:
            return
        self._clear_draft()
        self.progress_changed.emit("Enriched Description discarded")

    def cancel(self) -> None:
        if not self._busy and self._draft is None:
            return
        if self._operation is not None and not self._operation.is_finished:
            self._operation.cancel()
        self._operation = None
        self._request_id = None
        self._target = None
        self._set_busy(False, "Description enrichment cancelled")
        self.discard()

    def close(self) -> None:
        self.cancel()

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
            self._finish_with_error(
                SceneEnrichmentWorkflowError(
                    "the stack, card, active revision, image, Description, or "
                    "Style changed before background description completed"
                ),
                "Background description failed",
            )
            return
        if not isinstance(result, ImageDescriptionResult):
            self._finish_with_error(
                SceneEnrichmentWorkflowError(
                    "background description returned an unexpected result"
                ),
                "Background description failed",
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
            self._finish_with_error(
                SceneEnrichmentWorkflowError(
                    "the stack, card, active revision, image, Description, or "
                    "Style changed before enrichment completed"
                ),
                "Description enrichment failed",
            )
            return
        if not isinstance(result, SceneEnrichmentResult):
            self._finish_with_error(
                SceneEnrichmentWorkflowError(
                    "Description enrichment returned an unexpected result"
                ),
                "Description enrichment failed",
            )
            return
        self._operation = None
        self._request_id = None
        self._target = None
        self._set_busy(False, "Enriched Description ready for review")
        self._draft = SceneEnrichmentDraft(
            stack_id=target.stack_id,
            card_id=target.card_id,
            source_scene=target.source_scene,
            effective_style=target.effective_style,
            active_revision_id=target.active_revision_id,
            revision_id=target.revision_id,
            revision_image_path=target.revision_image_path,
            resolved_image_path=target.resolved_image_path,
            enriched_scene=result.scene,
            model_identifier=result.model_identifier,
            prompt_version=result.prompt_version,
        )
        self.draft_changed.emit()

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

    def _clear_draft(self) -> None:
        self._draft = None
        self.draft_changed.emit()

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
        if card is None or card.scene_description != target.source_scene:
            return False
        effective_style = (
            card.card_style if card.card_style is not None else document.global_style
        )
        if effective_style != target.effective_style:
            return False
        if card.active_revision_id != target.active_revision_id:
            return False
        if target.revision_id is None:
            return True
        if card.active_revision_id != target.revision_id:
            return False
        revision = next(
            (
                candidate
                for candidate in card.image_revisions
                if candidate.id == target.revision_id
            ),
            None,
        )
        if (
            revision is None
            or revision.image_path != target.revision_image_path
            or self._image_path_resolver is None
        ):
            return False
        image_path = self._image_path_resolver(revision.image_path)
        return (
            image_path == target.resolved_image_path
            and image_path is not None
            and image_path.is_file()
        )

    def _readable_active_image(
        self,
        card: Card,
    ) -> tuple[ImageRevision | None, Path | None]:
        if self._image_path_resolver is None:
            return None, None
        revision = next(
            (
                candidate
                for candidate in card.image_revisions
                if candidate.id == card.active_revision_id
            ),
            None,
        )
        if revision is None:
            return None, None
        image_path = self._image_path_resolver(revision.image_path)
        if image_path is None or not image_path.is_file():
            return None, None
        return revision, image_path

    @staticmethod
    def _card(document: Stack, card_id: UUID) -> Card:
        card = next(
            (candidate for candidate in document.cards if candidate.id == card_id),
            None,
        )
        if card is None:
            raise SceneEnrichmentWorkflowError(f"card {card_id} no longer exists")
        return card


__all__ = [
    "OllamaSettingsProvider",
    "ImageDescriberFactory",
    "ImageDescriberProtocol",
    "ImagePathResolver",
    "SceneEnricherFactory",
    "SceneEnricherProtocol",
    "SceneEnrichmentDraft",
    "SceneEnrichmentWorkflow",
    "SceneEnrichmentWorkflowError",
]
