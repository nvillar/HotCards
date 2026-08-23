"""Tests for direct revision-local background changes."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PIL import Image
from PySide6.QtCore import QObject, Signal

from hypergen.application.background_workflow import (
    BackgroundGenerationSettings,
    BackgroundWorkflow,
    BackgroundWorkflowError,
)
from hypergen.application.commands import EditRevisionDescriptionCommand
from hypergen.application.document_controller import DocumentController, UndoToken
from hypergen.application.document_session import DocumentSession
from hypergen.domain.models import (
    Card,
    CardRevision,
    HotspotSet,
    Interaction,
    NavigateAction,
    Stack,
    UnresolvedCardReference,
)
from hypergen.generation.mflux_generator import MfluxGenerator
from hypergen.storage.stack_store import StackStore


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

    def run_mflux(self, operation: object, *, stage: str) -> FakeOperation:
        assert stage == "generating background image"
        self.calls.append(operation)
        handle = FakeOperation()
        self.operations.append(handle)
        return handle


class FakeMfluxImage:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height

    def save(self, path: Path, *, overwrite: bool) -> None:
        assert not overwrite
        Image.new("RGB", (self.width, self.height), "navy").save(path)


class FakeMfluxModel:
    def generate_image(self, **kwargs: object) -> FakeMfluxImage:
        return FakeMfluxImage(
            width=kwargs["width"],  # type: ignore[arg-type]
            height=kwargs["height"],  # type: ignore[arg-type]
        )


def _settings() -> BackgroundGenerationSettings:
    return BackgroundGenerationSettings(
        mflux_model="flux2-klein-4b",
        step_count=4,
        quantization=None,
        random_seed=False,
        fixed_seed=42,
    )


def _bound_workflow(
    tmp_path: Path,
) -> tuple[
    BackgroundWorkflow,
    DocumentController,
    DocumentSession,
    FakeWorkers,
    Card,
]:
    hotspot = Interaction(
        label="Gate",
        action=NavigateAction(target=UnresolvedCardReference()),
    )
    revision = CardRevision(
        description="A garden",
        hotspot_set=HotspotSet(interactions=(hotspot,)),
    )
    card = Card(
        name="Garden",
        revisions=(revision,),
        active_revision_id=revision.id,
    )
    controller = DocumentController(
        Stack(
            name="Stack",
            cards=(card,),
            start_card_id=card.id,
        )
    )
    session = DocumentSession(controller)
    session.create(controller.document, tmp_path / "Stack.hypergen")
    workers = FakeWorkers()
    workflow = BackgroundWorkflow(
        controller,
        session,
        workers,  # type: ignore[arg-type]
        _settings,
        mflux_generator=MfluxGenerator(
            model_factory=lambda *_args: FakeMfluxModel()
        ),
        temporary_directory=tmp_path / "temporary",
    )
    return workflow, controller, session, workers, card


def _complete_generation(workers: FakeWorkers) -> None:
    operation = workers.operations[-1]
    work = workers.calls[-1]
    assert callable(work)
    operation.succeeded.emit(work())


def test_generate_applies_to_active_revision_and_preserves_other_content(
    tmp_path: Path,
) -> None:
    workflow, controller, session, workers, card = _bound_workflow(tmp_path)
    applied: list[tuple[str, object]] = []
    workflow.change_applied.connect(
        lambda message, token: applied.append((message, token))
    )

    workflow.generate(card.id)
    _complete_generation(workers)

    revision = controller.document.cards[0].active_revision
    assert revision.background is not None
    assert revision.background.type == "generated"
    assert revision.description == "A garden"
    assert revision.hotspot_set is not None
    assert revision.hotspot_set.interactions[0].label == "Gate"
    metadata = revision.generation_metadata
    assert metadata is not None
    assert metadata.inputs.description == "A garden"
    assert metadata.inputs.identity_reference is None
    assert session.flush()
    assert StackStore(session.state.bundle_path).load() == controller.document
    assert applied[0][0] == "Image generated"
    assert isinstance(applied[0][1], UndoToken)

    assert controller.undo_if_current(applied[0][1])  # type: ignore[arg-type]
    assert controller.document.cards[0].active_revision.background is None


def test_generation_failure_and_stale_result_preserve_current_revision(
    tmp_path: Path,
) -> None:
    workflow, controller, _session, workers, card = _bound_workflow(tmp_path)
    failures: list[object] = []
    workflow.failed.connect(failures.append)

    workflow.generate(card.id)
    workers.operations[-1].failed.emit(RuntimeError("model failed"))
    assert controller.document.cards[0].active_revision.background is None
    assert not workflow.busy
    assert failures[-1].args == ("model failed",)

    workflow.generate(card.id)
    revision = controller.document.cards[0].active_revision
    controller.execute(
        EditRevisionDescriptionCommand(
            card_id=card.id,
            revision_id=revision.id,
            value="Changed while generating",
        )
    )
    _complete_generation(workers)
    assert controller.document.cards[0].active_revision.background is None
    assert "changed before generation completed" in str(failures[-1])
    assert not list((tmp_path / "temporary").glob("generated-*.png"))

def test_revision_duplicate_activate_delete_round_trip(tmp_path: Path) -> None:
    workflow, controller, _session, _workers, card = _bound_workflow(tmp_path)
    original = card.active_revision

    workflow.duplicate_revision(card.id, original.id)
    changed = controller.document.cards[0]
    assert len(changed.revisions) == 2
    duplicate = changed.active_revision
    assert duplicate.id != original.id
    assert duplicate.description == original.description
    assert duplicate.hotspot_set == original.hotspot_set

    workflow.activate_revision(card.id, original.id)
    assert controller.document.cards[0].active_revision.id == original.id
    workflow.delete_revision(card.id, duplicate.id)
    assert len(controller.document.cards[0].revisions) == 1
    with pytest.raises(BackgroundWorkflowError):
        workflow.clear_background(card.id)


def test_unbound_stack_rejects_image_changes(tmp_path: Path) -> None:
    card = Card(name="Card")
    controller = DocumentController(Stack(name="Stack", cards=(card,)))
    workflow = BackgroundWorkflow(
        controller,
        DocumentSession(controller),
        FakeWorkers(),  # type: ignore[arg-type]
        _settings,
        temporary_directory=tmp_path / "temporary",
    )
    with pytest.raises(BackgroundWorkflowError, match="save the stack"):
        workflow.generate(card.id)
