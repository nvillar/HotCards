"""Transient, identity-bound Scene enrichment workflow."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Protocol
from uuid import UUID, uuid4

from PySide6.QtCore import QObject, Signal

from hypergen.application.commands import EditCardTextCommand
from hypergen.application.document_controller import DocumentController
from hypergen.application.workers import AdapterWorkers, WorkerOperation
from hypergen.domain.models import Card, DomainModel, Stack
from hypergen.generation.ollama_client import OllamaRuntime, OllamaSettings
from hypergen.generation.scene_enrichment import (
    OllamaSceneEnricher,
    SceneEnrichmentRequest,
    SceneEnrichmentResult,
)


class SceneEnrichmentWorkflowError(ValueError):
    """A Scene enrichment action cannot safely proceed."""


class SceneEnricherProtocol(Protocol):
    def enrich(self, request: SceneEnrichmentRequest) -> SceneEnrichmentResult: ...


SceneEnricherFactory = Callable[[OllamaSettings], SceneEnricherProtocol]
OllamaSettingsProvider = Callable[[], OllamaSettings]


def _default_enricher_factory(settings: OllamaSettings) -> SceneEnricherProtocol:
    return OllamaSceneEnricher(OllamaRuntime(settings))


@dataclass(frozen=True, slots=True)
class _EnrichmentTarget:
    stack_id: UUID
    card_id: UUID
    source_scene: str
    effective_style: str


class SceneEnrichmentDraft(DomainModel):
    """One reviewable rewrite kept outside the authoritative Stack."""

    stack_id: UUID
    card_id: UUID
    source_scene: str
    effective_style: str
    enriched_scene: str
    model_identifier: str
    prompt_version: str


class SceneEnrichmentWorkflow(QObject):
    """Coordinate one opt-in Scene rewrite and its explicit Apply boundary."""

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
        *,
        enricher_factory: SceneEnricherFactory = _default_enricher_factory,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.controller = controller
        self.workers = workers
        self._settings_provider = settings_provider
        self._enricher_factory = enricher_factory
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
            raise SceneEnrichmentWorkflowError("Scene enrichment is already running")
        if self._draft is not None:
            raise SceneEnrichmentWorkflowError(
                "accept or discard the current enriched Scene before starting another"
            )
        document = self.controller.document
        card = self._card(document, card_id)
        if not card.scene_description.strip():
            raise SceneEnrichmentWorkflowError("enter a Scene before enriching it")
        effective_style = (
            card.card_style if card.card_style is not None else document.global_style
        )
        request = SceneEnrichmentRequest(
            scene=card.scene_description,
            effective_style=effective_style,
        )
        target = _EnrichmentTarget(
            stack_id=document.id,
            card_id=card.id,
            source_scene=card.scene_description,
            effective_style=effective_style,
        )
        settings = self._settings_provider()
        request_id = uuid4()
        self._request_id = request_id
        self._target = target
        self._set_busy(True, "Enriching Scene...")
        operation = self.workers.run_ollama(
            lambda: self._enricher_factory(settings).enrich(request),
            stage="enriching Scene",
        )
        self._operation = operation
        operation.succeeded.connect(
            partial(self._enrichment_succeeded, request_id, target)
        )
        operation.failed.connect(partial(self._operation_failed, request_id))
        return operation

    def apply(self, card_id: UUID, enriched_scene: str) -> Stack:
        draft = self._draft
        if draft is None or draft.card_id != card_id:
            raise SceneEnrichmentWorkflowError(
                "this card has no enriched Scene to accept"
            )
        if not enriched_scene.strip():
            raise SceneEnrichmentWorkflowError("the enriched Scene must not be empty")
        target = _EnrichmentTarget(
            stack_id=draft.stack_id,
            card_id=draft.card_id,
            source_scene=draft.source_scene,
            effective_style=draft.effective_style,
        )
        if not self._target_is_current(target):
            self.discard()
            raise SceneEnrichmentWorkflowError(
                "the Scene or Style changed before the enrichment was accepted"
            )
        changed = self.controller.execute(
            EditCardTextCommand(
                card_id=card_id,
                field="scene_description",
                value=enriched_scene,
            )
        )
        self._clear_draft()
        self.progress_changed.emit("Enriched Scene accepted")
        self.document_changed.emit(changed)
        return changed

    def discard(self) -> None:
        if self._draft is None:
            return
        self._clear_draft()
        self.progress_changed.emit("Enriched Scene discarded")

    def cancel(self) -> None:
        if not self._busy and self._draft is None:
            return
        if self._operation is not None and not self._operation.is_finished:
            self._operation.cancel()
        self._operation = None
        self._request_id = None
        self._target = None
        self._set_busy(False, "Scene enrichment cancelled")
        self.discard()

    def close(self) -> None:
        self.cancel()

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
                    "the stack, card, Scene, or Style changed before enrichment completed"
                )
            )
            return
        if not isinstance(result, SceneEnrichmentResult):
            self._finish_with_error(
                SceneEnrichmentWorkflowError(
                    "Scene enrichment returned an unexpected result"
                )
            )
            return
        self._operation = None
        self._request_id = None
        self._target = None
        self._set_busy(False, "Enriched Scene ready for review")
        self._draft = SceneEnrichmentDraft(
            stack_id=target.stack_id,
            card_id=target.card_id,
            source_scene=target.source_scene,
            effective_style=target.effective_style,
            enriched_scene=result.scene,
            model_identifier=result.model_identifier,
            prompt_version=result.prompt_version,
        )
        self.draft_changed.emit()

    def _operation_failed(self, request_id: UUID, failure: object) -> None:
        if request_id == self._request_id:
            self._finish_with_error(failure)

    def _finish_with_error(self, failure: object) -> None:
        self._operation = None
        self._request_id = None
        self._target = None
        self._set_busy(False, "Scene enrichment failed")
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
        return effective_style == target.effective_style

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
    "SceneEnricherFactory",
    "SceneEnricherProtocol",
    "SceneEnrichmentDraft",
    "SceneEnrichmentWorkflow",
    "SceneEnrichmentWorkflowError",
]
