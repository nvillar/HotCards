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
from hypergen.application.generated_revision_change import GeneratedRevisionChange
from hypergen.application.workers import AdapterWorkers, WorkerOperation
from hypergen.domain.models import (
    Card,
    EnrichedDescription,
    EnrichmentReferenceSnapshot,
    ReferenceRole,
    ResolvedCardReference,
    Stack,
)
from hypergen.generation.ollama_client import OllamaRuntime, OllamaSettings
from hypergen.generation.reference_profiles import (
    REFERENCE_PROFILE_PROMPT_VERSION,
    OllamaReferenceProfiler,
    ReferenceProfileRequest,
    ReferenceProfileResult,
)
from hypergen.generation.scene_enrichment import (
    SCENE_ENRICHMENT_PROMPT_VERSION,
    OllamaSceneEnricher,
    SceneEnrichmentReference,
    SceneEnrichmentRequest,
    SceneEnrichmentResult,
    compose_profiled_scene,
)

ENRICHMENT_WORKFLOW_PROMPT_VERSION = (
    f"{SCENE_ENRICHMENT_PROMPT_VERSION}+{REFERENCE_PROFILE_PROMPT_VERSION}"
)


class SceneEnrichmentWorkflowError(ValueError):
    """A Description enrichment action cannot safely proceed."""


class SceneEnricherProtocol(Protocol):
    def enrich(self, request: SceneEnrichmentRequest) -> SceneEnrichmentResult: ...


class ReferenceProfilerProtocol(Protocol):
    def extract(
        self,
        request: ReferenceProfileRequest,
    ) -> ReferenceProfileResult: ...


SceneEnricherFactory = Callable[[OllamaSettings], SceneEnricherProtocol]
ReferenceProfilerFactory = Callable[
    [OllamaSettings],
    ReferenceProfilerProtocol,
]
OllamaSettingsProvider = Callable[[], OllamaSettings]


def _default_enricher_factory(settings: OllamaSettings) -> SceneEnricherProtocol:
    return OllamaSceneEnricher(OllamaRuntime(settings))


def _default_profiler_factory(
    settings: OllamaSettings,
) -> ReferenceProfilerProtocol:
    return OllamaReferenceProfiler(OllamaRuntime(settings))


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
    card_id: UUID
    revision_id: UUID
    background_id: UUID
    source_description: str


class SceneEnrichmentWorkflow(QObject):
    """Derive and apply one Enriched Description through an undoable command."""

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
        enricher_factory: SceneEnricherFactory = _default_enricher_factory,
        profiler_factory: ReferenceProfilerFactory = _default_profiler_factory,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.controller = controller
        self.workers = workers
        self._settings_provider = settings_provider
        self._enricher_factory = enricher_factory
        self._profiler_factory = profiler_factory
        self._profile_cache: dict[
            tuple[str, str, UUID, ReferenceRole, str],
            ReferenceProfileResult,
        ] = {}
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
        target = _EnrichmentTarget(
            stack_id=document.id,
            card_id=card.id,
            revision_id=revision.id,
            source_description=revision.description,
            references=self._reference_targets(document, card),
        )
        settings = self._settings_provider()
        request_id = uuid4()
        self._request_id = request_id
        self._target = target
        progress = (
            "Preparing references and enriching Description..."
            if target.references
            else "Enriching Description..."
        )
        self._set_busy(True, progress)
        operation = self.workers.run_ollama(
            lambda: self._run_enrichment(settings, target),
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
        previous_revision = next(
            revision
            for revision in self._card(
                self.controller.document,
                target.card_id,
            ).revisions
            if revision.id == target.revision_id
        )
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
            self.generation_applied.emit(
                GeneratedRevisionChange(
                    message="Description enriched",
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

    def _run_enrichment(
        self,
        settings: OllamaSettings,
        target: _EnrichmentTarget,
    ) -> SceneEnrichmentResult:
        profiles = tuple(
            self._profile_reference(settings, reference)
            for reference in target.references
        )
        references = tuple(
            SceneEnrichmentReference(
                role=profile.role,
                capsule=profile.capsule,
            )
            for profile in profiles
        )
        result = self._enricher_factory(settings).enrich(
            SceneEnrichmentRequest(
                scene=target.source_description,
                references=references,
            )
        )
        return result.model_copy(
            update={
                "scene": compose_profiled_scene(result.scene, references),
                "prompt_version": ENRICHMENT_WORKFLOW_PROMPT_VERSION,
            }
        )

    def _profile_reference(
        self,
        settings: OllamaSettings,
        target: _EnrichmentReferenceTarget,
    ) -> ReferenceProfileResult:
        key = (
            settings.model,
            REFERENCE_PROFILE_PROMPT_VERSION,
            target.background_id,
            target.role,
            target.source_description,
        )
        cached = self._profile_cache.get(key)
        if cached is not None:
            return cached
        result = self._profiler_factory(settings).extract(
            ReferenceProfileRequest(
                role=target.role,
                source_description=target.source_description,
            )
        )
        self._profile_cache[key] = result
        return result

    @staticmethod
    def _reference_targets(
        document: Stack,
        card: Card,
    ) -> tuple[_EnrichmentReferenceTarget, ...]:
        targets: list[_EnrichmentReferenceTarget] = []
        for role in ReferenceRole:
            assignment = getattr(card.active_revision, role.value)
            if not isinstance(assignment, ResolvedCardReference):
                continue
            source = next(
                candidate
                for candidate in document.cards
                if candidate.id == assignment.target_card_id
            )
            revision = source.active_revision
            background = revision.background
            if background is None:
                continue
            source_description = (
                background.generation_metadata.inputs.effective_description
            )
            if not source_description.strip():
                continue
            targets.append(
                _EnrichmentReferenceTarget(
                    role=role,
                    card_id=source.id,
                    revision_id=revision.id,
                    background_id=background.id,
                    source_description=source_description,
                )
            )
        return tuple(targets)

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
    "ENRICHMENT_WORKFLOW_PROMPT_VERSION",
    "OllamaSettingsProvider",
    "ReferenceProfilerFactory",
    "ReferenceProfilerProtocol",
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
            card_id=target.card_id,
            revision_id=target.revision_id,
            background_id=target.background_id,
        )
        for target in targets
    )


def enrichment_reference_snapshots(
    document: Stack,
    card: Card,
) -> tuple[EnrichmentReferenceSnapshot, ...]:
    """Return exact usable reference provenance for enrichment freshness."""
    return _reference_snapshots(
        SceneEnrichmentWorkflow._reference_targets(document, card)
    )
