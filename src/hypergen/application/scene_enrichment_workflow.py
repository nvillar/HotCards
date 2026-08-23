"""Text-only, review-first Description enrichment workflow."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Protocol
from uuid import UUID, uuid4

from PySide6.QtCore import QObject, Signal

from hypergen.application.commands import EditRevisionDescriptionCommand
from hypergen.application.document_controller import DocumentController
from hypergen.application.workers import AdapterWorkers, WorkerOperation
from hypergen.domain.models import Card, Stack
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


SceneEnricherFactory = Callable[[OllamaSettings], SceneEnricherProtocol]
OllamaSettingsProvider = Callable[[], OllamaSettings]


def _default_enricher_factory(settings: OllamaSettings) -> SceneEnricherProtocol:
    return OllamaSceneEnricher(OllamaRuntime(settings))


@dataclass(frozen=True, slots=True)
class _EnrichmentTarget:
    stack_id: UUID
    card_id: UUID
    revision_id: UUID
    source_description: str


@dataclass(frozen=True, slots=True)
class _EnrichmentProposal:
    target: _EnrichmentTarget
    description: str


class SceneEnrichmentWorkflow(QObject):
    """Produce an editable Description proposal and apply it only on request."""

    busy_changed = Signal(bool)
    progress_changed = Signal(str)
    failed = Signal(object)
    proposal_ready = Signal(str)
    proposal_cleared = Signal()
    document_changed = Signal(object)
    change_applied = Signal(str, object)

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
        self._operation: WorkerOperation | None = None
        self._request_id: UUID | None = None
        self._target: _EnrichmentTarget | None = None
        self._proposal: _EnrichmentProposal | None = None
        self._busy = False

    @property
    def busy(self) -> bool:
        return self._busy

    @property
    def has_proposal(self) -> bool:
        return self._proposal is not None

    def start(self, card_id: UUID) -> WorkerOperation:
        if self._busy:
            raise SceneEnrichmentWorkflowError(
                "Description enrichment is already running"
            )
        document = self.controller.document
        card = self._card(document, card_id)
        revision = card.active_revision
        if not revision.description.strip():
            raise SceneEnrichmentWorkflowError(
                "enter a Description before enriching it"
            )
        self.discard_proposal()
        target = _EnrichmentTarget(
            stack_id=document.id,
            card_id=card.id,
            revision_id=revision.id,
            source_description=revision.description,
        )
        settings = self._settings_provider()
        request_id = uuid4()
        self._request_id = request_id
        self._target = target
        self._set_busy(True, "Enriching Description...")
        operation = self.workers.run_ollama(
            lambda: self._enricher_factory(settings).enrich(
                SceneEnrichmentRequest(scene=target.source_description)
            ),
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

    def apply_proposal(self, description: str) -> Stack:
        proposal = self._proposal
        if proposal is None:
            raise SceneEnrichmentWorkflowError(
                "there is no Description enrichment proposal to apply"
            )
        if not description.strip():
            raise SceneEnrichmentWorkflowError(
                "the enriched Description must not be empty"
            )
        if not self._target_is_current(proposal.target):
            self.discard_proposal()
            raise SceneEnrichmentWorkflowError(
                "the Description changed before the enrichment proposal was applied"
            )
        previous_token = self.controller.current_undo_token
        changed = self.controller.execute(
            EditRevisionDescriptionCommand(
                card_id=proposal.target.card_id,
                revision_id=proposal.target.revision_id,
                value=description,
            )
        )
        self._clear_proposal()
        self.document_changed.emit(changed)
        token = self.controller.current_undo_token
        if token is not None and token != previous_token:
            self.change_applied.emit("Description enriched", token)
        return changed

    def discard_proposal(self) -> None:
        if self._proposal is not None:
            self._clear_proposal()

    def cancel(self) -> None:
        if self._operation is not None and not self._operation.is_finished:
            self._operation.cancel()
        self._operation = None
        self._request_id = None
        self._target = None
        self.discard_proposal()
        if self._busy:
            self._set_busy(False, "Description enrichment cancelled")

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
                    "the stack, card, revision, or Description changed "
                    "before enrichment completed"
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
        self._proposal = _EnrichmentProposal(
            target=target,
            description=result.scene,
        )
        self._set_busy(False, "Description enrichment ready for review")
        self.proposal_ready.emit(result.scene)

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

    def _clear_proposal(self) -> None:
        self._proposal = None
        self.proposal_cleared.emit()

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
        return (
            card is not None
            and card.active_revision_id == target.revision_id
            and card.active_revision.description == target.source_description
        )

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
    "SceneEnrichmentWorkflow",
    "SceneEnrichmentWorkflowError",
]
