"""Transient AI hotspot generation, editing, and atomic Apply workflow."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Protocol
from uuid import UUID, uuid4

from pydantic import ValidationError
from PySide6.QtCore import QObject, Signal

from hypergen.application.commands import (
    CreateCardCommand,
    ReplaceHotspotSetCommand,
)
from hypergen.application.document_controller import DocumentController
from hypergen.application.workers import AdapterWorkers, WorkerOperation
from hypergen.domain.models import (
    Card,
    DomainModel,
    HotspotGenerationProvenance,
    HotspotSet,
    ImageRevision,
    Interaction,
    NavigateAction,
    Polygon,
    ResolvedCardReference,
    Stack,
    UnresolvedCardReference,
)
from hypergen.generation.hotspot_prompts import (
    CandidateTarget,
    CardCatalogueEntry,
    ExistingCandidateTarget,
    HotspotGenerationRequest,
    HotspotGenerationResult,
    HotspotProposal,
    HotspotReconciliationWarning,
    NewCandidateTarget,
    OllamaHotspotGenerator,
    UnresolvedCandidateTarget,
)
from hypergen.generation.ollama_client import OllamaRuntime, OllamaSettings


class HotspotGenerationWorkflowError(ValueError):
    """A generated-hotspot action cannot safely proceed."""


class HotspotGeneratorProtocol(Protocol):
    def generate(self, request: HotspotGenerationRequest) -> HotspotGenerationResult: ...


HotspotGeneratorFactory = Callable[[OllamaSettings], HotspotGeneratorProtocol]
OllamaSettingsProvider = Callable[[], OllamaSettings]
ImagePathResolver = Callable[[str], Path | None]


def _default_generator_factory(settings: OllamaSettings) -> HotspotGeneratorProtocol:
    return OllamaHotspotGenerator(OllamaRuntime(settings))


@dataclass(frozen=True, slots=True)
class _GenerationTarget:
    stack_id: UUID
    card_id: UUID
    revision_id: UUID
    image_path: str
    interaction_description: str


class HotspotGenerationDraft(DomainModel):
    """Editable candidate set kept outside the authoritative Stack."""

    stack_id: UUID
    card_id: UUID
    revision_id: UUID
    image_path: str
    interaction_description: str
    card_ids_by_token: dict[str, UUID]
    hotspot_set: HotspotSet
    targets_by_interaction_id: dict[UUID, CandidateTarget]
    warnings: tuple[str, ...]
    reconciliation_warnings: tuple[HotspotReconciliationWarning, ...]
    raw_response: str
    provenance: HotspotGenerationProvenance


class HotspotGenerationWorkflow(QObject):
    """Coordinate request-local generation, candidate edits, and atomic Apply."""

    candidate_changed = Signal()
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
        generator_factory: HotspotGeneratorFactory = _default_generator_factory,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.controller = controller
        self.workers = workers
        self._settings_provider = settings_provider
        self._image_path_resolver = image_path_resolver
        self._generator_factory = generator_factory
        self._candidate: HotspotGenerationDraft | None = None
        self._operation: WorkerOperation | None = None
        self._request_id: UUID | None = None
        self._target: _GenerationTarget | None = None
        self._busy = False

    @property
    def candidate(self) -> HotspotGenerationDraft | None:
        return self._candidate

    @property
    def busy(self) -> bool:
        return self._busy

    def start(self, card_id: UUID) -> WorkerOperation:
        if self._busy:
            raise HotspotGenerationWorkflowError("hotspot generation is already running")
        if self._candidate is not None:
            raise HotspotGenerationWorkflowError(
                "apply or discard the current hotspot candidate before generating another"
            )
        document = self.controller.document
        card = self._card(document, card_id)
        revision = self._active_revision(card)
        image_path = self._image_path_resolver(revision.image_path)
        if image_path is None:
            raise HotspotGenerationWorkflowError(
                "the active background image is unavailable"
            )
        catalogue, card_ids_by_token = self._catalogue(document)
        request = HotspotGenerationRequest(
            image_path=image_path,
            interaction_description=card.interaction_description,
            card_catalogue=catalogue,
        )
        target = _GenerationTarget(
            stack_id=document.id,
            card_id=card.id,
            revision_id=revision.id,
            image_path=revision.image_path,
            interaction_description=card.interaction_description,
        )
        settings = self._settings_provider()
        request_id = uuid4()
        self._request_id = request_id
        self._target = target
        self._set_busy(True, "Generating hotspot candidates...")
        operation = self.workers.run_ollama(
            lambda: self._generator_factory(settings).generate(request),
            stage="generating hotspots",
        )
        self._operation = operation
        operation.succeeded.connect(
            partial(
                self._generation_succeeded,
                request_id,
                target,
                card_ids_by_token,
                settings,
            )
        )
        operation.failed.connect(partial(self._operation_failed, request_id))
        return operation

    def apply(self) -> Stack:
        candidate = self._require_candidate()
        target = self._target_for(candidate)
        if not self._target_is_current(target):
            self.discard()
            raise HotspotGenerationWorkflowError(
                "the card, active revision, or interaction intent changed before Apply"
            )
        document = self.controller.document
        known_card_ids = {card.id for card in document.cards}
        interactions = tuple(
            interaction.model_copy(
                update={
                    "action": NavigateAction(
                        target=self._persisted_target(
                            candidate.targets_by_interaction_id[interaction.id],
                            candidate.card_ids_by_token,
                            known_card_ids,
                        )
                    )
                }
            )
            for interaction in candidate.hotspot_set.interactions
        )
        hotspot_set = HotspotSet(
            interactions=interactions,
            generation_provenance=candidate.provenance,
        )
        changed = self.controller.execute(
            ReplaceHotspotSetCommand(
                card_id=candidate.card_id,
                revision_id=candidate.revision_id,
                hotspot_set=hotspot_set,
            )
        )
        self._clear_candidate()
        self.progress_changed.emit("Hotspot candidates applied")
        self.document_changed.emit(changed)
        return changed

    def discard(self) -> None:
        if self._candidate is None:
            return
        self._clear_candidate()
        self.progress_changed.emit("Hotspot candidates discarded")

    def cancel(self) -> None:
        if not self._busy and self._candidate is None:
            return
        if self._operation is not None and not self._operation.is_finished:
            self._operation.cancel()
        self._operation = None
        self._request_id = None
        self._target = None
        self._set_busy(False, "Hotspot generation cancelled")
        self.discard()

    def close(self) -> None:
        self.cancel()

    def rename_interaction(self, interaction_id: UUID, label: str) -> None:
        self._replace_interaction(
            interaction_id,
            lambda interaction: interaction.model_copy(update={"label": label}),
        )

    def set_destination(self, interaction_id: UUID, card_id: UUID | None) -> None:
        candidate = self._require_candidate()
        targets = dict(candidate.targets_by_interaction_id)
        if card_id is None:
            targets[interaction_id] = UnresolvedCandidateTarget()
        else:
            card = self._card(self.controller.document, card_id)
            token = next(
                (
                    token
                    for token, mapped_card_id in candidate.card_ids_by_token.items()
                    if mapped_card_id == card_id
                ),
                None,
            )
            card_ids_by_token = dict(candidate.card_ids_by_token)
            if token is None:
                token = self._next_token(card_ids_by_token)
                card_ids_by_token[token] = card_id
            targets[interaction_id] = ExistingCandidateTarget(
                card_token=token,
                card_name=card.name,
            )
            candidate = candidate.model_copy(
                update={"card_ids_by_token": card_ids_by_token}
            )
        self._candidate = candidate.model_copy(
            update={"targets_by_interaction_id": targets}
        )
        self.candidate_changed.emit()

    def create_destination_card(self, interaction_id: UUID, name: str) -> UUID:
        candidate = self._require_candidate()
        command = CreateCardCommand(name=name)
        changed = self.controller.execute(command)
        token = self._next_token(candidate.card_ids_by_token)
        card_ids_by_token = {
            **candidate.card_ids_by_token,
            token: command.card_id,
        }
        targets = {
            **candidate.targets_by_interaction_id,
            interaction_id: ExistingCandidateTarget(
                card_token=token,
                card_name=name,
            ),
        }
        self._candidate = candidate.model_copy(
            update={
                "card_ids_by_token": card_ids_by_token,
                "targets_by_interaction_id": targets,
            }
        )
        self.document_changed.emit(changed)
        self.candidate_changed.emit()
        return command.card_id

    def reorder_interaction(self, interaction_id: UUID, new_index: int) -> None:
        candidate = self._require_candidate()
        interactions = list(candidate.hotspot_set.interactions)
        old_index = self._interaction_index(interactions, interaction_id)
        if not 0 <= new_index < len(interactions):
            raise HotspotGenerationWorkflowError(
                f"hotspot index {new_index} is out of range"
            )
        interaction = interactions.pop(old_index)
        interactions.insert(new_index, interaction)
        self._set_interactions(tuple(interactions))

    def delete_interaction(self, interaction_id: UUID) -> None:
        candidate = self._require_candidate()
        interactions = tuple(
            interaction
            for interaction in candidate.hotspot_set.interactions
            if interaction.id != interaction_id
        )
        if len(interactions) == len(candidate.hotspot_set.interactions):
            raise HotspotGenerationWorkflowError("candidate hotspot does not exist")
        targets = dict(candidate.targets_by_interaction_id)
        targets.pop(interaction_id)
        self._candidate = candidate.model_copy(
            update={
                "hotspot_set": candidate.hotspot_set.model_copy(
                    update={"interactions": interactions}
                ),
                "targets_by_interaction_id": targets,
            }
        )
        self.candidate_changed.emit()

    def add_interaction(self, polygon: Polygon) -> UUID:
        candidate = self._require_candidate()
        number = len(candidate.hotspot_set.interactions) + 1
        interaction = Interaction(
            label=f"Hotspot {number}",
            action=NavigateAction(target=UnresolvedCardReference()),
            polygons=(polygon,),
        )
        targets = {
            **candidate.targets_by_interaction_id,
            interaction.id: UnresolvedCandidateTarget(),
        }
        self._candidate = candidate.model_copy(
            update={
                "hotspot_set": candidate.hotspot_set.model_copy(
                    update={
                        "interactions": (
                            *candidate.hotspot_set.interactions,
                            interaction,
                        )
                    }
                ),
                "targets_by_interaction_id": targets,
            }
        )
        self.candidate_changed.emit()
        return interaction.id

    def add_polygon(self, interaction_id: UUID, polygon: Polygon) -> None:
        self._replace_interaction(
            interaction_id,
            lambda interaction: interaction.model_copy(
                update={"polygons": (*interaction.polygons, polygon)}
            ),
        )

    def replace_polygon(
        self,
        interaction_id: UUID,
        polygon_index: int,
        polygon: Polygon,
    ) -> None:
        def replace(interaction: Interaction) -> Interaction:
            polygons = list(interaction.polygons)
            if not 0 <= polygon_index < len(polygons):
                raise HotspotGenerationWorkflowError(
                    f"polygon index {polygon_index} is out of range"
                )
            polygons[polygon_index] = polygon
            return interaction.model_copy(update={"polygons": tuple(polygons)})

        self._replace_interaction(interaction_id, replace)

    def delete_polygon(self, interaction_id: UUID, polygon_index: int) -> None:
        def delete(interaction: Interaction) -> Interaction:
            if len(interaction.polygons) == 1:
                raise HotspotGenerationWorkflowError(
                    "cannot delete the only polygon; delete the hotspot instead"
                )
            polygons = list(interaction.polygons)
            if not 0 <= polygon_index < len(polygons):
                raise HotspotGenerationWorkflowError(
                    f"polygon index {polygon_index} is out of range"
                )
            polygons.pop(polygon_index)
            return interaction.model_copy(update={"polygons": tuple(polygons)})

        self._replace_interaction(interaction_id, delete)

    def _generation_succeeded(
        self,
        request_id: UUID,
        target: _GenerationTarget,
        card_ids_by_token: dict[str, UUID],
        settings: OllamaSettings,
        result: object,
    ) -> None:
        if request_id != self._request_id:
            return
        if not self._target_is_current(target):
            self._finish_with_error(
                HotspotGenerationWorkflowError(
                    "the stack, card, revision, or interaction intent changed "
                    "before hotspot generation completed"
                )
            )
            return
        if not isinstance(result, HotspotGenerationResult):
            self._finish_with_error(
                HotspotGenerationWorkflowError(
                    "hotspot generation returned an unexpected result"
                )
            )
            return
        try:
            hotspot_set, targets = self._candidate_set(result.proposals)
        except ValidationError as error:
            self._finish_with_error(
                HotspotGenerationWorkflowError(
                    f"generated hotspot geometry is invalid: {error}"
                )
            )
            return
        provenance = HotspotGenerationProvenance(
            model_identifier=result.model_identifier,
            prompt_version=result.prompt_version,
            schema_version=result.schema_version,
            effective_settings={
                "temperature": settings.temperature,
                "think": settings.think,
                "num_predict": settings.num_predict,
                "context_length": settings.context_length,
            },
            generated_at=datetime.now(UTC),
            duration_seconds=result.duration_seconds,
        )
        warnings = (
            *result.warnings,
            *(warning.message for warning in result.reconciliation_warnings),
        )
        self._operation = None
        self._request_id = None
        self._target = None
        self._set_busy(False, "Hotspot candidates ready for review")
        self._candidate = HotspotGenerationDraft(
            stack_id=target.stack_id,
            card_id=target.card_id,
            revision_id=target.revision_id,
            image_path=target.image_path,
            interaction_description=target.interaction_description,
            card_ids_by_token=card_ids_by_token,
            hotspot_set=hotspot_set,
            targets_by_interaction_id=targets,
            warnings=warnings,
            reconciliation_warnings=result.reconciliation_warnings,
            raw_response=result.raw_response,
            provenance=provenance,
        )
        self.candidate_changed.emit()

    def _operation_failed(self, request_id: UUID, failure: object) -> None:
        if request_id == self._request_id:
            self._finish_with_error(failure)

    def _finish_with_error(self, failure: object) -> None:
        self._operation = None
        self._request_id = None
        self._target = None
        self._set_busy(False, "Hotspot generation failed")
        self.failed.emit(failure)

    def _replace_interaction(
        self,
        interaction_id: UUID,
        replace: Callable[[Interaction], Interaction],
    ) -> None:
        candidate = self._require_candidate()
        interactions = list(candidate.hotspot_set.interactions)
        index = self._interaction_index(interactions, interaction_id)
        interactions[index] = replace(interactions[index])
        interactions[index].model_dump(mode="python", round_trip=True)
        self._set_interactions(tuple(interactions))

    def _set_interactions(self, interactions: tuple[Interaction, ...]) -> None:
        candidate = self._require_candidate()
        self._candidate = candidate.model_copy(
            update={
                "hotspot_set": candidate.hotspot_set.model_copy(
                    update={"interactions": interactions}
                )
            }
        )
        self.candidate_changed.emit()

    def _clear_candidate(self) -> None:
        self._candidate = None
        self.candidate_changed.emit()

    def _set_busy(self, busy: bool, progress: str) -> None:
        self._busy = busy
        self.busy_changed.emit(busy)
        self.progress_changed.emit(progress)

    def _require_candidate(self) -> HotspotGenerationDraft:
        if self._candidate is None:
            raise HotspotGenerationWorkflowError(
                "no hotspot candidate is available"
            )
        return self._candidate

    def _target_is_current(self, target: _GenerationTarget) -> bool:
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
            or card.interaction_description != target.interaction_description
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
    def _candidate_set(
        proposals: tuple[HotspotProposal, ...],
    ) -> tuple[HotspotSet, dict[UUID, CandidateTarget]]:
        interactions: list[Interaction] = []
        targets: dict[UUID, CandidateTarget] = {}
        for proposal in proposals:
            if isinstance(proposal.target, ExistingCandidateTarget):
                target_name = proposal.target.card_name
            elif isinstance(proposal.target, NewCandidateTarget):
                target_name = proposal.target.proposed_name
            else:
                target_name = proposal.target.description or None
            interaction = Interaction(
                label=proposal.label,
                action=NavigateAction(
                    target=UnresolvedCardReference(target_name=target_name)
                ),
                polygons=tuple(
                    Polygon(points=polygon.points)
                    for polygon in proposal.polygons
                ),
            )
            interactions.append(interaction)
            targets[interaction.id] = proposal.target
        return HotspotSet(interactions=tuple(interactions)), targets

    @staticmethod
    def _persisted_target(
        target: CandidateTarget,
        card_ids_by_token: dict[str, UUID],
        known_card_ids: set[UUID],
    ) -> ResolvedCardReference | UnresolvedCardReference:
        if isinstance(target, ExistingCandidateTarget):
            card_id = card_ids_by_token.get(target.card_token)
            if card_id in known_card_ids:
                return ResolvedCardReference(target_card_id=card_id)
            return UnresolvedCardReference(target_name=target.card_name)
        if isinstance(target, NewCandidateTarget):
            return UnresolvedCardReference(target_name=target.proposed_name)
        return UnresolvedCardReference(target_name=target.description or None)

    @staticmethod
    def _catalogue(
        document: Stack,
    ) -> tuple[tuple[CardCatalogueEntry, ...], dict[str, UUID]]:
        entries = tuple(
            CardCatalogueEntry(
                token=f"C{index}",
                name=card.name,
                description=card.scene_description,
            )
            for index, card in enumerate(document.cards, start=1)
        )
        return entries, {
            entry.token: card.id
            for entry, card in zip(entries, document.cards, strict=True)
        }

    @staticmethod
    def _next_token(card_ids_by_token: dict[str, UUID]) -> str:
        index = 1
        while f"C{index}" in card_ids_by_token:
            index += 1
        return f"C{index}"

    @staticmethod
    def _interaction_index(
        interactions: list[Interaction],
        interaction_id: UUID,
    ) -> int:
        for index, interaction in enumerate(interactions):
            if interaction.id == interaction_id:
                return index
        raise HotspotGenerationWorkflowError("candidate hotspot does not exist")

    @staticmethod
    def _target_for(candidate: HotspotGenerationDraft) -> _GenerationTarget:
        return _GenerationTarget(
            stack_id=candidate.stack_id,
            card_id=candidate.card_id,
            revision_id=candidate.revision_id,
            image_path=candidate.image_path,
            interaction_description=candidate.interaction_description,
        )

    @staticmethod
    def _card(document: Stack, card_id: UUID) -> Card:
        card = next(
            (candidate for candidate in document.cards if candidate.id == card_id),
            None,
        )
        if card is None:
            raise HotspotGenerationWorkflowError(f"card {card_id} no longer exists")
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
            raise HotspotGenerationWorkflowError(
                "apply a background before generating hotspots"
            )
        return revision


__all__ = [
    "HotspotGenerationDraft",
    "HotspotGenerationWorkflow",
    "HotspotGenerationWorkflowError",
    "HotspotGeneratorFactory",
    "HotspotGeneratorProtocol",
    "ImagePathResolver",
    "OllamaSettingsProvider",
]
