"""Tests for direct, geometry-only hotspot remapping."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PIL import Image
from PySide6.QtCore import QObject, Signal

from hypergen.application.commands import RenameInteractionCommand
from hypergen.application.document_controller import DocumentController, UndoToken
from hypergen.application.hotspot_remap_workflow import (
    HotspotRemapWorkflow,
    HotspotRemapWorkflowError,
)
from hypergen.domain.models import (
    Card,
    CardRevision,
    HotspotRemapProvenance,
    HotspotSet,
    ImportedBackground,
    Interaction,
    NavigateAction,
    Point,
    Polygon,
    ResolvedCardReference,
    Stack,
    UnresolvedCardReference,
)
from hypergen.generation.hotspot_prompts import (
    HOTSPOT_REMAP_PROMPT_VERSION,
    HOTSPOT_REMAP_SCHEMA_VERSION,
    HotspotRemapRequest,
    HotspotRemapResult,
    UnlocatedHotspot,
)
from hypergen.generation.ollama_client import OllamaSettings


class FakeOperation(QObject):
    succeeded = Signal(object)
    failed = Signal(object)
    finished = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.cancelled = False
        self.finished_state = False

    @property
    def is_finished(self) -> bool:
        return self.finished_state

    def cancel(self) -> None:
        self.cancelled = True
        self.finished_state = True


class FakeWorkers:
    def __init__(self) -> None:
        self.calls: list[object] = []
        self.operations: list[FakeOperation] = []

    def run_ollama(self, work: object, *, stage: str) -> FakeOperation:
        assert stage == "remapping hotspots"
        self.calls.append(work)
        operation = FakeOperation()
        self.operations.append(operation)
        return operation


class RecordingRemapper:
    def __init__(self, result: HotspotRemapResult) -> None:
        self.result = result
        self.requests: list[HotspotRemapRequest] = []

    def remap(self, request: HotspotRemapRequest) -> HotspotRemapResult:
        self.requests.append(request)
        return self.result


def _polygon(offset: float = 0) -> Polygon:
    return Polygon(
        points=(
            Point(x=0.1 + offset, y=0.1),
            Point(x=0.4 + offset, y=0.1),
            Point(x=0.2 + offset, y=0.5),
        )
    )


def _result(
    *,
    polygons_by_token: dict[str, tuple[Polygon, ...]],
    unlocated: tuple[UnlocatedHotspot, ...] = (),
) -> HotspotRemapResult:
    return HotspotRemapResult(
        polygons_by_token=polygons_by_token,
        unlocated=unlocated,
        warnings=(),
        raw_responses=('{"mapped":[]}',),
        provenance=HotspotRemapProvenance(
            model_identifier="test-model",
            prompt_version=HOTSPOT_REMAP_PROMPT_VERSION,
            schema_version=HOTSPOT_REMAP_SCHEMA_VERSION,
            generated_at=datetime.now(UTC),
            duration_seconds=0.1,
        ),
    )


def _setup(
    tmp_path: Path,
    *,
    include_background: bool = True,
    include_hotspots: bool = True,
    result: HotspotRemapResult | None = None,
) -> tuple[
    HotspotRemapWorkflow,
    DocumentController,
    FakeWorkers,
    RecordingRemapper,
    Card,
    tuple[Interaction, ...],
]:
    image = tmp_path / "image.png"
    Image.new("RGB", (64, 48), "navy").save(image)
    destination = Card(name="Destination")
    interactions = (
        Interaction(
            label="Door",
            action=NavigateAction(
                target=ResolvedCardReference(target_card_id=destination.id)
            ),
            polygons=(_polygon(),),
        ),
        Interaction(
            label="Window",
            action=NavigateAction(
                target=UnresolvedCardReference(target_name="Tower")
            ),
        ),
    )
    asset_id = uuid4()
    revision = CardRevision(
        description="A room",
        background=(
            ImportedBackground(
                id=asset_id,
                image_path="assets/image.png",
                source_filename="image.png",
                created_at=datetime.now(UTC),
            )
            if include_background
            else None
        ),
        hotspot_set=(
            HotspotSet(interactions=interactions)
            if include_hotspots
            else None
        ),
    )
    card = Card(
        name="Source",
        revisions=(revision,),
        active_revision_id=revision.id,
    )
    controller = DocumentController(
        Stack(name="Stack", cards=(card, destination))
    )
    workers = FakeWorkers()
    remapper = RecordingRemapper(
        result or _result(polygons_by_token={"H1": (_polygon(0.1),)})
    )
    workflow = HotspotRemapWorkflow(
        controller,
        workers,  # type: ignore[arg-type]
        lambda: OllamaSettings(model="test-model"),
        lambda _relative: image,
        remapper_factory=lambda _settings: remapper,
    )
    return workflow, controller, workers, remapper, card, interactions


def _complete(workers: FakeWorkers) -> None:
    work = workers.calls[-1]
    assert callable(work)
    workers.operations[-1].succeeded.emit(work())


def test_remap_preserves_semantics_and_applies_one_undoable_replacement(
    tmp_path: Path,
) -> None:
    workflow, controller, workers, remapper, card, original = _setup(tmp_path)
    applied_tokens: list[object] = []
    workflow.change_applied.connect(
        lambda _message, token: applied_tokens.append(token)
    )

    workflow.start(card.id)
    _complete(workers)

    request = remapper.requests[0]
    assert [(item.token, item.label) for item in request.hotspots] == [
        ("H1", "Door"),
        ("H2", "Window"),
    ]
    changed = controller.document.cards[0].active_revision.hotspot_set
    assert changed is not None
    assert [item.id for item in changed.interactions] == [
        item.id for item in original
    ]
    assert [item.label for item in changed.interactions] == ["Door", "Window"]
    assert [item.action for item in changed.interactions] == [
        item.action for item in original
    ]
    assert changed.interactions[0].polygons == (_polygon(0.1),)
    assert changed.interactions[1].polygons == ()
    assert changed.remap_provenance is not None
    assert isinstance(applied_tokens[0], UndoToken)

    assert controller.undo_if_current(applied_tokens[0])  # type: ignore[arg-type]
    restored = controller.document.cards[0].active_revision.hotspot_set
    assert restored is not None
    assert restored.interactions == original


def test_unlocated_hotspot_keeps_old_geometry_and_reports_warning(
    tmp_path: Path,
) -> None:
    result = _result(
        polygons_by_token={},
        unlocated=(UnlocatedHotspot(token="H1", reason="not visible"),),
    )
    workflow, controller, workers, _remapper, card, original = _setup(
        tmp_path,
        result=result,
    )
    messages: list[str] = []
    workflow.change_applied.connect(lambda message, _token: messages.append(message))

    workflow.start(card.id)
    _complete(workers)

    hotspot_set = controller.document.cards[0].active_revision.hotspot_set
    assert hotspot_set is not None
    assert hotspot_set.interactions[0].polygons == original[0].polygons
    assert messages == ["Hotspots remapped; 1 kept existing geometry"]


@pytest.mark.parametrize(
    ("include_background", "include_hotspots", "message"),
    [
        (False, True, "add an image"),
        (True, False, "add at least one hotspot"),
    ],
)
def test_remap_requires_image_and_existing_hotspots(
    tmp_path: Path,
    include_background: bool,
    include_hotspots: bool,
    message: str,
) -> None:
    workflow, _controller, _workers, _remapper, card, _original = _setup(
        tmp_path,
        include_background=include_background,
        include_hotspots=include_hotspots,
    )
    with pytest.raises(HotspotRemapWorkflowError, match=message):
        workflow.start(card.id)


def test_failure_cancel_and_stale_result_preserve_applied_set(
    tmp_path: Path,
) -> None:
    workflow, controller, workers, _remapper, card, original = _setup(tmp_path)
    failures: list[object] = []
    workflow.failed.connect(failures.append)

    workflow.start(card.id)
    workers.operations[-1].failed.emit(RuntimeError("vision failed"))
    current = controller.document.cards[0].active_revision.hotspot_set
    assert current is not None
    assert current.interactions == original

    workflow.start(card.id)
    revision = controller.document.cards[0].active_revision
    controller.execute(
        RenameInteractionCommand(
            card_id=card.id,
            revision_id=revision.id,
            interaction_id=original[0].id,
            label="Changed",
        )
    )
    _complete(workers)
    assert "changed before remapping completed" in str(failures[-1])
    current = controller.document.cards[0].active_revision.hotspot_set
    assert current is not None
    assert current.interactions[0].label == "Changed"
    assert current.interactions[0].polygons == original[0].polygons

    workflow.start(card.id)
    operation = workers.operations[-1]
    workflow.cancel()
    operation.succeeded.emit(_result(polygons_by_token={"H1": (_polygon(0.2),)}))
    current = controller.document.cards[0].active_revision.hotspot_set
    assert current is not None
    assert current.interactions[0].polygons == original[0].polygons
