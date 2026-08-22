"""Direct geometry-only remapping for an active revision's hotspots."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Protocol
from uuid import UUID, uuid4

from PySide6.QtCore import QObject, Signal

from hypergen.application.commands import ReplaceHotspotSetCommand
from hypergen.application.document_controller import DocumentController
from hypergen.application.workers import AdapterWorkers, WorkerOperation
from hypergen.domain.models import Card, HotspotSet, Stack
from hypergen.generation.hotspot_prompts import (
    HotspotRemapRequest,
    HotspotRemapResult,
    OllamaHotspotRemapper,
    RemapHotspotInput,
)
from hypergen.generation.ollama_client import OllamaRuntime, OllamaSettings


class HotspotRemapWorkflowError(ValueError):
    """A hotspot remap cannot safely proceed."""


class HotspotRemapperProtocol(Protocol):
    def remap(self, request: HotspotRemapRequest) -> HotspotRemapResult: ...


HotspotRemapperFactory = Callable[[OllamaSettings], HotspotRemapperProtocol]
OllamaSettingsProvider = Callable[[], OllamaSettings]
ImagePathResolver = Callable[[str], Path | None]


def _default_remapper_factory(
    settings: OllamaSettings,
) -> HotspotRemapperProtocol:
    return OllamaHotspotRemapper(OllamaRuntime(settings))


@dataclass(frozen=True, slots=True)
class _RemapTarget:
    stack_id: UUID
    card_id: UUID
    revision_id: UUID
    background_id: UUID
    image_path: str
    hotspot_set: HotspotSet
    interaction_ids_by_token: dict[str, UUID]


class HotspotRemapWorkflow(QObject):
    """Remap existing hotspot polygons and apply them atomically."""

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
        image_path_resolver: ImagePathResolver,
        *,
        remapper_factory: HotspotRemapperFactory = _default_remapper_factory,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.controller = controller
        self.workers = workers
        self._settings_provider = settings_provider
        self._image_path_resolver = image_path_resolver
        self._remapper_factory = remapper_factory
        self._operation: WorkerOperation | None = None
        self._request_id: UUID | None = None
        self._target: _RemapTarget | None = None
        self._busy = False

    @property
    def busy(self) -> bool:
        return self._busy

    def start(self, card_id: UUID) -> WorkerOperation:
        if self._busy:
            raise HotspotRemapWorkflowError("hotspot remapping is already running")
        document = self.controller.document
        card = self._card(document, card_id)
        revision = card.active_revision
        if revision.background is None:
            raise HotspotRemapWorkflowError(
                "add an image before remapping hotspots"
            )
        image_path = self._image_path_resolver(revision.background.image_path)
        if image_path is None or not image_path.is_file():
            raise HotspotRemapWorkflowError(
                "the active revision image is unavailable"
            )
        hotspot_set = revision.hotspot_set
        if hotspot_set is None or not hotspot_set.interactions:
            raise HotspotRemapWorkflowError(
                "add at least one hotspot before remapping"
            )
        tokens = {
            f"H{index}": interaction.id
            for index, interaction in enumerate(
                hotspot_set.interactions,
                start=1,
            )
        }
        request = HotspotRemapRequest(
            image_path=image_path,
            hotspots=tuple(
                RemapHotspotInput(
                    token=token,
                    label=next(
                        interaction.label
                        for interaction in hotspot_set.interactions
                        if interaction.id == interaction_id
                    ),
                )
                for token, interaction_id in tokens.items()
            ),
        )
        target = _RemapTarget(
            stack_id=document.id,
            card_id=card.id,
            revision_id=revision.id,
            background_id=revision.background.id,
            image_path=revision.background.image_path,
            hotspot_set=hotspot_set,
            interaction_ids_by_token=tokens,
        )
        settings = self._settings_provider()
        request_id = uuid4()
        self._request_id = request_id
        self._target = target
        self._set_busy(True, "Remapping hotspots...")
        operation = self.workers.run_ollama(
            lambda: self._remapper_factory(settings).remap(request),
            stage="remapping hotspots",
        )
        self._operation = operation
        operation.succeeded.connect(
            partial(self._remap_succeeded, request_id, target)
        )
        operation.failed.connect(partial(self._operation_failed, request_id))
        return operation

    def cancel(self) -> None:
        if self._operation is not None and not self._operation.is_finished:
            self._operation.cancel()
        self._operation = None
        self._request_id = None
        self._target = None
        if self._busy:
            self._set_busy(False, "Hotspot remap cancelled")

    def close(self) -> None:
        self.cancel()

    def _remap_succeeded(
        self,
        request_id: UUID,
        target: _RemapTarget,
        result: object,
    ) -> None:
        if request_id != self._request_id:
            return
        if not self._target_is_current(target):
            self._finish_with_error(
                HotspotRemapWorkflowError(
                    "the stack, card, revision, image, or hotspots changed "
                    "before remapping completed"
                )
            )
            return
        if not isinstance(result, HotspotRemapResult):
            self._finish_with_error(
                HotspotRemapWorkflowError(
                    "hotspot remapping returned an unexpected result"
                )
            )
            return
        tokens_by_interaction_id = {
            interaction_id: token
            for token, interaction_id in target.interaction_ids_by_token.items()
        }
        interactions = tuple(
            interaction.model_copy(
                update={
                    "polygons": result.polygons_by_token.get(
                        tokens_by_interaction_id[interaction.id],
                        interaction.polygons,
                    )
                }
            )
            for interaction in target.hotspot_set.interactions
        )
        previous_token = self.controller.current_undo_token
        changed = self.controller.execute(
            ReplaceHotspotSetCommand(
                card_id=target.card_id,
                revision_id=target.revision_id,
                hotspot_set=HotspotSet(
                    interactions=interactions,
                    remap_provenance=result.provenance,
                ),
            )
        )
        unmatched = len(result.unlocated)
        message = (
            "Hotspots remapped"
            if unmatched == 0
            else f"Hotspots remapped; {unmatched} kept existing geometry"
        )
        self._operation = None
        self._request_id = None
        self._target = None
        self._set_busy(False, message)
        self.document_changed.emit(changed)
        token = self.controller.current_undo_token
        if token is not None and token != previous_token:
            self.change_applied.emit(message, token)

    def _operation_failed(self, request_id: UUID, failure: object) -> None:
        if request_id == self._request_id:
            self._finish_with_error(failure)

    def _finish_with_error(self, failure: object) -> None:
        self._operation = None
        self._request_id = None
        self._target = None
        self._set_busy(False, "Hotspot remap failed")
        self.failed.emit(failure)

    def _set_busy(self, busy: bool, progress: str) -> None:
        self._busy = busy
        self.busy_changed.emit(busy)
        self.progress_changed.emit(progress)

    def _target_is_current(self, target: _RemapTarget) -> bool:
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
        return (
            revision.background is not None
            and revision.background.id == target.background_id
            and revision.background.image_path == target.image_path
            and revision.hotspot_set == target.hotspot_set
        )

    @staticmethod
    def _card(document: Stack, card_id: UUID) -> Card:
        card = next(
            (candidate for candidate in document.cards if candidate.id == card_id),
            None,
        )
        if card is None:
            raise HotspotRemapWorkflowError(f"card {card_id} no longer exists")
        return card


__all__ = [
    "HotspotRemapWorkflow",
    "HotspotRemapWorkflowError",
    "HotspotRemapperFactory",
    "HotspotRemapperProtocol",
    "ImagePathResolver",
    "OllamaSettingsProvider",
]
