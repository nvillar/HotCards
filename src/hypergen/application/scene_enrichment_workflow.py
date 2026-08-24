"""Text-only, directly applied Description enrichment workflow."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Protocol
from uuid import UUID, uuid4

from PySide6.QtCore import QObject, Signal

from hypergen.application.commands import SetRevisionEnrichedDescriptionCommand
from hypergen.application.document_controller import DocumentController
from hypergen.application.workers import AdapterWorkers, WorkerOperation
from hypergen.domain.models import (
    Card,
    CardReference,
    EnrichedDescription,
    EnrichmentReferenceSnapshot,
    ReferenceRole,
    ResolvedCardReference,
    Stack,
    UnresolvedCardReference,
)
from hypergen.generation.ollama_client import OllamaRuntime, OllamaSettings
from hypergen.generation.scene_enrichment import (
    OllamaSceneEnricher,
    SceneEnrichmentReference,
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
    references: tuple[_EnrichmentReferenceTarget, ...]


@dataclass(frozen=True, slots=True)
class _EnrichmentReferenceTarget:
    role: ReferenceRole
    assignment: CardReference
    source_revision_id: UUID | None
    source_background_id: UUID | None
    source_generation_description: str | None


class SceneEnrichmentWorkflow(QObject):
    """Derive and apply one Enriched Description through an undoable command."""

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
        if not revision.description.strip():
            raise SceneEnrichmentWorkflowError(
                "enter a Description before enriching it"
            )
        references = self._reference_targets(document, card)
        target = _EnrichmentTarget(
            stack_id=document.id,
            card_id=card.id,
            revision_id=revision.id,
            source_description=revision.description,
            references=references,
        )
        settings = self._settings_provider()
        request_id = uuid4()
        self._request_id = request_id
        self._target = target
        self._set_busy(True, "Enriching Description...")
        operation = self.workers.run_ollama(
            lambda: self._enricher_factory(settings).enrich(
                SceneEnrichmentRequest(
                    scene=target.source_description,
                    references=self._reference_contexts(
                        target.references
                    ),
                )
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
        if not result.scene.strip():
            self._finish_with_error(
                SceneEnrichmentWorkflowError(
                    "Description enrichment returned an empty Description"
                ),
                "Description enrichment failed",
            )
            return
        previous_token = self.controller.current_undo_token
        changed = self.controller.execute(
            SetRevisionEnrichedDescriptionCommand(
                card_id=target.card_id,
                revision_id=target.revision_id,
                value=EnrichedDescription(
                    text=result.scene,
                    source_description=target.source_description,
                    references=_reference_snapshots(target.references),
                    model_identifier=result.model_identifier,
                    prompt_version=result.prompt_version,
                ),
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
        return (
            card is not None
            and card.active_revision_id == target.revision_id
            and card.active_revision.description == target.source_description
            and self._reference_targets(document, card) == target.references
        )

    @staticmethod
    def _reference_targets(
        document: Stack,
        card: Card,
    ) -> tuple[_EnrichmentReferenceTarget, ...]:
        targets: list[_EnrichmentReferenceTarget] = []
        for role in ReferenceRole:
            assignment = getattr(card.active_revision, role.value)
            if assignment is None:
                continue
            if isinstance(assignment, UnresolvedCardReference):
                targets.append(
                    _EnrichmentReferenceTarget(
                        role=role,
                        assignment=assignment,
                        source_revision_id=None,
                        source_background_id=None,
                        source_generation_description=None,
                    )
                )
                continue
            source = next(
                candidate
                for candidate in document.cards
                if candidate.id == assignment.target_card_id
            )
            source_revision = source.active_revision
            background = source_revision.background
            targets.append(
                _EnrichmentReferenceTarget(
                    role=role,
                    assignment=assignment,
                    source_revision_id=source_revision.id,
                    source_background_id=(
                        background.id if background is not None else None
                    ),
                    source_generation_description=(
                        background.generation_metadata.inputs.effective_description
                        if background is not None
                        else None
                    ),
                )
            )
        return tuple(targets)

    @staticmethod
    def _reference_contexts(
        targets: tuple[_EnrichmentReferenceTarget, ...],
    ) -> tuple[SceneEnrichmentReference, ...]:
        groups: list[tuple[tuple[UUID, UUID], list[ReferenceRole], str]] = []
        indexes: dict[tuple[UUID, UUID], int] = {}
        for target in targets:
            if (
                not isinstance(target.assignment, ResolvedCardReference)
                or target.source_background_id is None
                or not target.source_generation_description
            ):
                continue
            key = (
                target.assignment.target_card_id,
                target.source_background_id,
            )
            index = indexes.get(key)
            if index is None:
                indexes[key] = len(groups)
                groups.append(
                    (
                        key,
                        [target.role],
                        target.source_generation_description,
                    )
                )
            else:
                groups[index][1].append(target.role)
        return tuple(
            SceneEnrichmentReference(
                roles=tuple(roles),
                source_description=description,
            )
            for _key, roles, description in groups
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
    "enrichment_reference_snapshots",
]


def _reference_snapshots(
    targets: tuple[_EnrichmentReferenceTarget, ...],
) -> tuple[EnrichmentReferenceSnapshot, ...]:
    return tuple(
        EnrichmentReferenceSnapshot(
            role=target.role,
            card_id=target.assignment.target_card_id,
            revision_id=target.source_revision_id,
            background_id=target.source_background_id,
        )
        for target in targets
        if (
            isinstance(target.assignment, ResolvedCardReference)
            and target.source_revision_id is not None
            and target.source_background_id is not None
            and target.source_generation_description
        )
    )


def enrichment_reference_snapshots(
    document: Stack,
    card: Card,
) -> tuple[EnrichmentReferenceSnapshot, ...]:
    """Return the usable reference-image inputs for enrichment freshness."""
    return _reference_snapshots(
        SceneEnrichmentWorkflow._reference_targets(document, card)
    )
