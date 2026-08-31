"""Tests for direct revision-local background changes."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PIL import Image
from PySide6.QtCore import QObject, Signal

from hotcards.application.background_workflow import (
    BackgroundGenerationSettings,
    BackgroundWorkflow,
    BackgroundWorkflowError,
)
from hotcards.application.commands import (
    CreateCardCommand,
    DuplicateRevisionCommand,
    EditRevisionDescriptionCommand,
    ReplaceRevisionBackgroundCommand,
    SetRevisionGenerateResolutionCommand,
    SetRevisionReferenceCommand,
    SetRevisionStyleCommand,
    UpdateStyleCommand,
)
from hotcards.application.document_controller import DocumentController, UndoToken
from hotcards.application.document_session import DocumentSession
from hotcards.application.generated_revision_change import GeneratedRevisionChange
from hotcards.domain.image_dimensions import GenerateResolution
from hotcards.domain.models import (
    Card,
    CardRevision,
    GeneratedBackground,
    HotspotSet,
    ImageReferenceSnapshot,
    ImageSourceSnapshot,
    Interaction,
    NavigateAction,
    RefineProvenance,
    RefineTransformation,
    ResolvedCardReference,
    Stack,
    UnresolvedCardReference,
)
from hotcards.generation.errors import ImageGenerationCancelled
from hotcards.generation.mflux_generator import MfluxGenerator
from hotcards.storage.stack_store import StackStore


class FakeOperation(QObject):
    succeeded = Signal(object)
    failed = Signal(object)
    cancelled = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.was_cancelled = False
        self.finished_state = False

    @property
    def is_finished(self) -> bool:
        return self.finished_state

    def cancel(self) -> None:
        self.was_cancelled = True
        self.finished_state = True
        self.cancelled.emit()


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


class FakeCallbackRegistry:
    def __init__(self) -> None:
        self.registered: list[object] = []

    def register(self, callback: object) -> None:
        self.registered.append(callback)


class FakeMfluxModel:
    def __init__(self) -> None:
        self.callbacks = FakeCallbackRegistry()
        self.calls: list[dict[str, object]] = []

    def generate_image(self, **kwargs: object) -> FakeMfluxImage:
        self.calls.append(kwargs)
        config = SimpleNamespace(num_inference_steps=kwargs["num_inference_steps"])
        for callback in self.callbacks.registered:
            callback.call_before_loop(config=config)
        for _step in range(kwargs["num_inference_steps"]):  # type: ignore[arg-type]
            for callback in self.callbacks.registered:
                callback.call_in_loop()
        for callback in self.callbacks.registered:
            callback.call_after_loop()
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
    root: Path,
) -> tuple[
    BackgroundWorkflow,
    DocumentController,
    DocumentSession,
    FakeWorkers,
    FakeMfluxModel,
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
    card = Card(name="Garden", revisions=(revision,), active_revision_id=revision.id)
    controller = DocumentController(
        Stack(name="Stack", cards=(card,), start_card_id=card.id)
    )
    session = DocumentSession(controller)
    session.create(controller.document, root / "Stack.hotcards")
    workers = FakeWorkers()
    model = FakeMfluxModel()
    workflow = BackgroundWorkflow(
        controller,
        session,
        workers,  # type: ignore[arg-type]
        _settings,
        mflux_generator=MfluxGenerator(
            model_factory=lambda *_args: model,
            edit_model_factory=lambda *_args: model,
        ),
        temporary_directory=root / "temporary",
    )
    return workflow, controller, session, workers, model, card


def _complete_generation(workers: FakeWorkers) -> None:
    operation = workers.operations[-1]
    work = workers.calls[-1]
    assert callable(work)
    operation.succeeded.emit(work())


def _create_generated_source(
    workflow: BackgroundWorkflow,
    controller: DocumentController,
    workers: FakeWorkers,
    *,
    name: str,
) -> Card:
    source_id = uuid4()
    controller.execute(CreateCardCommand(name=name, card_id=source_id))
    source = next(card for card in controller.document.cards if card.id == source_id)
    controller.execute(
        EditRevisionDescriptionCommand(
            card_id=source.id,
            revision_id=source.active_revision.id,
            value=f"{name} source image",
        )
    )
    workflow.generate(source.id)
    _complete_generation(workers)
    return next(card for card in controller.document.cards if card.id == source_id)


def test_generate_uses_description_and_preserves_result_lifecycle(
    tmp_path: Path,
) -> None:
    workflow, controller, session, workers, model, card = _bound_workflow(tmp_path)
    applied: list[GeneratedRevisionChange] = []
    progress: list[tuple[int, int]] = []
    workflow.generation_applied.connect(applied.append)
    workflow.generation_progress_changed.connect(
        lambda completed, total: progress.append((completed, total))
    )

    workflow.generate(card.id)
    _complete_generation(workers)

    assert progress == [(0, 4), (1, 4), (2, 4), (3, 4), (4, 4), (0, 0)]
    revision = controller.document.cards[0].active_revision
    assert revision.background is not None
    assert revision.description == "A garden"
    assert revision.hotspot_set is not None
    provenance = revision.provenance
    assert provenance is not None
    assert provenance.operation == "generate"
    assert provenance.inputs.description == "A garden"
    assert provenance.render_prompt == "A garden"
    assert (provenance.settings.width, provenance.settings.height) == (592, 448)
    assert model.calls[-1]["prompt"] == "A garden"
    assert session.flush()
    assert StackStore(session.state.bundle_path).load() == controller.document
    assert applied[0].message == "Image generated"
    assert applied[0].previous_revision.background is None
    assert isinstance(applied[0].token, UndoToken)

    assert controller.undo_if_current(applied[0].token)
    assert controller.document.cards[0].active_revision.background is None


def test_rejected_generation_apply_removes_the_new_unreachable_asset(
    tmp_path: Path,
) -> None:
    workflow, controller, session, workers, _model, _card = _bound_workflow(tmp_path)
    source = _create_generated_source(
        workflow,
        controller,
        workers,
        name="Source",
    )
    source_revision = source.active_revision
    source_background = source_revision.background
    assert source_background is not None

    workflow.generate(source.id)

    dependent_id = uuid4()
    controller.execute(CreateCardCommand(name="Dependent", card_id=dependent_id))
    dependent = next(
        card for card in controller.document.cards if card.id == dependent_id
    )
    derived_asset_id = uuid4()
    derived_source = tmp_path / "derived-source.png"
    Image.new("RGB", (592, 448), "green").save(derived_source)
    bundle_path = session.state.bundle_path
    assert bundle_path is not None
    store = StackStore(bundle_path)
    derived_path = store.store_image_asset(
        derived_source,
        card_id=dependent.id,
        asset_id=derived_asset_id,
    )
    generated_at = datetime.now(UTC)
    derived_background = GeneratedBackground(
        id=derived_asset_id,
        image_path=derived_path,
        provenance=RefineProvenance(
            source=ImageSourceSnapshot(
                card_id=source.id,
                revision_id=source_revision.id,
                background_id=source_background.id,
            ),
            description="A refined source",
            render_prompt="A refined source",
            resolution=GenerateResolution.RESOLUTION_512,
            transformation=RefineTransformation.BALANCED,
            strength=0.50,
            settings=source_background.provenance.settings,
        ),
        created_at=generated_at,
    )
    controller.execute(
        ReplaceRevisionBackgroundCommand(
            card_id=dependent.id,
            revision_id=dependent.active_revision.id,
            background=derived_background,
        )
    )
    document_before_completion = controller.document
    assets_before_completion = set(
        (bundle_path / "assets" / "cards").glob("*/image-*.png")
    )
    failures: list[object] = []
    workflow.failed.connect(failures.append)

    _complete_generation(workers)

    assert failures
    assert "cannot replace this source background" in str(failures[-1])
    assert controller.document == document_before_completion
    assert (
        next(card for card in controller.document.cards if card.id == source.id)
        .active_revision.background
        == source_background
    )
    assert set((bundle_path / "assets" / "cards").glob("*/image-*.png")) == (
        assets_before_completion
    )
    assert not workflow.busy
    worker_call_count = len(workers.calls)
    with pytest.raises(
        BackgroundWorkflowError,
        match="cannot replace this source background",
    ):
        workflow.generate(source.id)
    assert len(workers.calls) == worker_call_count


def test_generate_appends_and_captures_selected_style(tmp_path: Path) -> None:
    workflow, controller, _session, workers, model, card = _bound_workflow(tmp_path)
    style = controller.document.styles[0]
    controller.execute(
        SetRevisionStyleCommand(
            card_id=card.id,
            revision_id=card.active_revision.id,
            style_id=style.id,
        )
    )

    workflow.generate(card.id)
    _complete_generation(workers)

    provenance = controller.document.cards[0].active_revision.provenance
    assert provenance is not None
    assert provenance.operation == "generate"
    assert provenance.inputs.style is not None
    assert provenance.inputs.style.style_id == style.id
    assert provenance.render_prompt == f"A garden.\n\n{style.prompt_text}"
    assert model.calls[-1]["prompt"] == provenance.render_prompt


def test_generate_uses_revision_resolution_and_stack_aspect_ratio(
    tmp_path: Path,
) -> None:
    workflow, controller, _session, workers, model, card = _bound_workflow(tmp_path)
    controller.execute(
        SetRevisionGenerateResolutionCommand(
            card_id=card.id,
            revision_id=card.active_revision.id,
            resolution=GenerateResolution.RESOLUTION_1024,
        )
    )

    workflow.generate(card.id)
    _complete_generation(workers)

    provenance = controller.document.cards[0].active_revision.provenance
    assert provenance is not None
    assert provenance.operation == "generate"
    assert provenance.inputs.resolution is GenerateResolution.RESOLUTION_1024
    assert (model.calls[-1]["width"], model.calls[-1]["height"]) == (1184, 880)
    assert (provenance.settings.width, provenance.settings.height) == (1184, 880)


def test_description_and_style_changes_suppress_in_flight_generation(
    tmp_path: Path,
) -> None:
    workflow, controller, _session, workers, _model, card = _bound_workflow(tmp_path)
    failures: list[object] = []
    workflow.failed.connect(failures.append)

    workflow.generate(card.id)
    controller.execute(
        EditRevisionDescriptionCommand(
            card_id=card.id,
            revision_id=card.active_revision.id,
            value="Changed while generating",
        )
    )
    _complete_generation(workers)
    assert controller.document.cards[0].active_revision.background is None

    style = controller.document.styles[0]
    controller.execute(
        SetRevisionStyleCommand(
            card_id=card.id,
            revision_id=card.active_revision.id,
            style_id=style.id,
        )
    )
    workflow.generate(card.id)
    controller.execute(
        UpdateStyleCommand(
            style_id=style.id,
            name=style.name,
            prompt_text=style.prompt_text + " More contrast.",
        )
    )
    _complete_generation(workers)

    assert controller.document.cards[0].active_revision.background is None
    workflow.generate(card.id)
    controller.execute(
        SetRevisionGenerateResolutionCommand(
            card_id=card.id,
            revision_id=card.active_revision.id,
            resolution=GenerateResolution.RESOLUTION_768,
        )
    )
    _complete_generation(workers)

    assert controller.document.cards[0].active_revision.background is None
    assert all("changed before generation completed" in str(failure) for failure in failures)
    assert not list((tmp_path / "temporary").glob("generated-*.png"))


def test_cancelled_generation_cannot_publish_or_leave_output(tmp_path: Path) -> None:
    workflow, controller, _session, workers, _model, card = _bound_workflow(tmp_path)
    original = controller.document

    workflow.generate(card.id)
    operation = workers.operations[-1]
    work = workers.calls[-1]
    assert callable(work)

    workflow.cancel()

    assert operation.was_cancelled
    with pytest.raises(ImageGenerationCancelled, match="cancelled"):
        work()
    assert controller.document == original
    assert not list((tmp_path / "temporary").glob("generated-*.png"))


def test_generate_requires_nonempty_description(tmp_path: Path) -> None:
    workflow, controller, _session, workers, _model, card = _bound_workflow(tmp_path)
    controller.execute(
        EditRevisionDescriptionCommand(
            card_id=card.id,
            revision_id=card.active_revision.id,
            value="",
        )
    )

    with pytest.raises(BackgroundWorkflowError, match="Description"):
        workflow.generate(card.id)
    assert workers.calls == []


def test_two_references_are_sent_once_in_stable_order(tmp_path: Path) -> None:
    workflow, controller, _session, workers, model, target = _bound_workflow(tmp_path)
    sources = tuple(
        _create_generated_source(
            workflow,
            controller,
            workers,
            name=f"Reference {number}",
        )
        for number in (1, 2)
    )
    for position, source in enumerate(sources, start=1):
        controller.execute(
            SetRevisionReferenceCommand(
                card_id=target.id,
                revision_id=target.active_revision.id,
                reference=ResolvedCardReference(target_card_id=source.id),
                position=position,
            )
        )
    controller.execute(
        EditRevisionDescriptionCommand(
            card_id=target.id,
            revision_id=target.active_revision.id,
            value="Place the subject from image 1 beside the setting from image 2.",
        )
    )

    workflow.generate(target.id)
    _complete_generation(workers)

    provenance = controller.document.cards[0].active_revision.provenance
    assert provenance is not None
    assert provenance.operation == "generate"
    assert provenance.inputs.references == tuple(
        ImageReferenceSnapshot(
            card_id=source.id,
            revision_id=source.active_revision.id,
            background_id=source.active_revision.background.id,
        )
        for source in sources
        if source.active_revision.background is not None
    )
    expected_paths = [
        workflow.session.store.asset_path(source.active_revision.background.image_path)
        for source in sources
        if source.active_revision.background is not None
    ]
    assert model.calls[-1]["image_paths"] == expected_paths
    assert len(model.calls[-1]["image_paths"]) == len(set(model.calls[-1]["image_paths"]))
    assert model.calls[-1]["prompt"] == (
        "Place the subject from image 1 beside the setting from image 2."
    )


def test_reference_change_suppresses_in_flight_result(tmp_path: Path) -> None:
    workflow, controller, _session, workers, _model, target = _bound_workflow(tmp_path)
    source = _create_generated_source(
        workflow,
        controller,
        workers,
        name="Reference",
    )
    controller.execute(
        SetRevisionReferenceCommand(
            card_id=target.id,
            revision_id=target.active_revision.id,
            reference=ResolvedCardReference(target_card_id=source.id),
        )
    )
    failures: list[object] = []
    workflow.failed.connect(failures.append)

    workflow.generate(target.id)
    controller.execute(
        DuplicateRevisionCommand(
            card_id=source.id,
            source_revision_id=source.active_revision.id,
        )
    )
    _complete_generation(workers)

    assert controller.document.cards[0].active_revision.background is None
    assert "changed before generation completed" in str(failures[-1])


def test_reference_card_requires_an_active_image(tmp_path: Path) -> None:
    workflow, controller, _session, _workers, _model, target = _bound_workflow(tmp_path)
    source_id = uuid4()
    controller.execute(CreateCardCommand(name="Blank source", card_id=source_id))
    controller.execute(
        SetRevisionReferenceCommand(
            card_id=target.id,
            revision_id=target.active_revision.id,
            reference=ResolvedCardReference(target_card_id=source_id),
        )
    )

    with pytest.raises(BackgroundWorkflowError, match="has no image"):
        workflow.generate(target.id)


def test_revision_duplicate_activate_delete_round_trip(tmp_path: Path) -> None:
    workflow, controller, _session, _workers, _model, card = _bound_workflow(tmp_path)
    original = card.active_revision

    workflow.duplicate_revision(card.id, original.id)
    changed = controller.document.cards[0]
    assert len(changed.revisions) == 2
    duplicate = changed.active_revision
    assert duplicate.id != original.id
    assert duplicate.description == original.description
    assert duplicate.hotspot_set == original.hotspot_set

    workflow.activate_revision(card.id, original.id)
    workflow.delete_revision(card.id, duplicate.id)
    assert len(controller.document.cards[0].revisions) == 1


def test_unbound_stack_rejects_image_changes(tmp_path: Path) -> None:
    card = Card(name="Card", revisions=(CardRevision(description="Scene"),))
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
