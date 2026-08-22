"""Tests for card-local background drafts and durable revision application."""

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
    prepare_import_image,
)
from hypergen.application.commands import CreateCardCommand
from hypergen.application.document_controller import DocumentController
from hypergen.application.document_session import DocumentSession
from hypergen.domain.models import Card, ImageGenerationInputs, Stack
from hypergen.generation.mflux_generator import MfluxGenerator
from hypergen.generation.ollama_client import OllamaSettings
from hypergen.generation.revision_naming import (
    REVISION_NAMING_PROMPT_VERSION,
    RevisionNamingRequest,
    RevisionNamingResult,
)
from hypergen.storage.stack_store import StackStore


class FakeOperation(QObject):
    succeeded = Signal(object)
    failed = Signal(object)
    finished = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.cancelled = False

    @property
    def is_finished(self) -> bool:
        return self.cancelled

    def cancel(self) -> None:
        self.cancelled = True


class FakeWorkers:
    def __init__(self) -> None:
        self.mflux_calls: list[object] = []
        self.mflux_operations: list[FakeOperation] = []
        self.ollama_calls: list[object] = []
        self.ollama_operations: list[FakeOperation] = []

    def run_ollama(self, operation: object, *, stage: str) -> FakeOperation:
        assert stage == "naming background revision"
        self.ollama_calls.append(operation)
        handle = FakeOperation()
        self.ollama_operations.append(handle)
        return handle

    def run_mflux(self, operation: object, *, stage: str) -> FakeOperation:
        assert stage == "generating background image"
        self.mflux_calls.append(operation)
        handle = FakeOperation()
        self.mflux_operations.append(handle)
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


class FakeRevisionNamer:
    def __init__(self, proposed_name: str = "Moonlit Garden") -> None:
        self.proposed_name = proposed_name
        self.requests: list[RevisionNamingRequest] = []

    def name(self, request: RevisionNamingRequest) -> RevisionNamingResult:
        self.requests.append(request)
        return RevisionNamingResult(
            name=self.proposed_name,
            raw_response=f'{{"name":"{self.proposed_name}"}}',
            model_identifier="test-model",
            prompt_version=REVISION_NAMING_PROMPT_VERSION,
            duration_seconds=0.1,
        )


def settings() -> BackgroundGenerationSettings:
    return BackgroundGenerationSettings(
        mflux_model="flux2-klein-4b",
        step_count=4,
        quantization=None,
        random_seed=False,
        fixed_seed=42,
    )


def ollama_settings() -> OllamaSettings:
    return OllamaSettings(model="test-model")


def bound_workflow(
    tmp_path: Path,
) -> tuple[
    BackgroundWorkflow,
    DocumentController,
    DocumentSession,
    FakeWorkers,
    Card,
]:
    card = Card(
        name="Garden",
        scene_description="A garden",
        interaction_description="The gate opens",
        card_style="Pencil",
    )
    controller = DocumentController(
        Stack(
            name="Stack",
            global_style="Storybook",
            cards=(card,),
            start_card_id=card.id,
        )
    )
    session = DocumentSession(controller)
    session.create(controller.document, tmp_path / "Stack.hypergen")
    workers = FakeWorkers()
    generator = MfluxGenerator(model_factory=lambda *_args: FakeMfluxModel())
    namer = FakeRevisionNamer()
    workflow = BackgroundWorkflow(
        controller,
        session,
        workers,  # type: ignore[arg-type]
        settings,
        ollama_settings,
        mflux_generator=generator,
        revision_namer_factory=lambda _settings: namer,
        temporary_directory=tmp_path / "candidates",
    )
    return workflow, controller, session, workers, card


def test_generate_then_apply_creates_durable_revision(tmp_path: Path) -> None:
    workflow, controller, session, workers, card = bound_workflow(tmp_path)

    workflow.generate(card.id)
    generated = workers.mflux_calls[0]()  # type: ignore[operator]
    workers.mflux_operations[0].succeeded.emit(generated)
    assert workflow.draft_for(card.id) is None
    named = workers.ollama_calls[0]()  # type: ignore[operator]
    workers.ollama_operations[0].succeeded.emit(named)

    draft = workflow.draft_for(card.id)
    assert draft is not None
    assert draft.name == "Moonlit Garden"
    assert draft.image_path.is_file()
    assert generated.metadata.inputs == ImageGenerationInputs(
        scene_description="A garden",
        global_style="Storybook",
        card_style="Pencil",
    )

    workflow.apply_draft(card.id)
    assert workflow.draft_for(card.id) is None
    revision = controller.document.cards[0].image_revisions[0]
    assert revision.id == draft.revision_id
    assert revision.name == "Moonlit Garden"
    assert revision.generation_metadata is not None
    assert revision.generation_metadata.render_prompt == "A garden\n\nPencil"
    assert revision.generation_metadata.seed == 42
    assert revision.hotspot_set is None
    assert session.flush()
    bundle_path = session.state.bundle_path
    assert bundle_path is not None
    reopened = StackStore(bundle_path).load()
    assert reopened.cards[0].image_revisions[0] == revision


def test_import_crop_positions_and_discard_remain_transient(tmp_path: Path) -> None:
    source = tmp_path / "wide.png"
    image = Image.new("RGB", (200, 100), "red")
    for x in range(100, 200):
        for y in range(100):
            image.putpixel((x, y), (0, 0, 255))
    image.save(source)
    left = tmp_path / "left.png"
    right = tmp_path / "right.png"

    prepare_import_image(
        source,
        left,
        width=100,
        height=100,
        position_x=0.0,
    )
    prepare_import_image(
        source,
        right,
        width=100,
        height=100,
        position_x=1.0,
    )
    with Image.open(left) as left_image:
        assert left_image.getpixel((50, 50))[0] > 200
    with Image.open(right) as right_image:
        assert right_image.getpixel((50, 50))[2] > 200

    workflow, controller, session, _workers, card = bound_workflow(tmp_path)
    original = controller.document
    draft = workflow.import_image(card.id, source, position_x=0.25)
    assert draft.source_filename == "wide.png"
    assert controller.document == original
    bundle_path = session.state.bundle_path
    assert bundle_path is not None
    assert StackStore(bundle_path).load() == original
    with pytest.raises(BackgroundWorkflowError, match="already has"):
        workflow.import_image(card.id, source)
    workflow.discard_draft(card.id)
    assert workflow.draft_for(card.id) is None
    assert not draft.image_path.exists()
    assert controller.document == original


def test_revision_switch_delete_and_undo_restore_association(tmp_path: Path) -> None:
    workflow, controller, session, _workers, card = bound_workflow(tmp_path)
    first_source = tmp_path / "first.png"
    second_source = tmp_path / "second.png"
    Image.new("RGB", (1024, 768), "red").save(first_source)
    Image.new("RGB", (1024, 768), "blue").save(second_source)

    first_draft = workflow.import_image(card.id, first_source)
    workflow.apply_draft(card.id)
    second_draft = workflow.import_image(card.id, second_source)
    workflow.apply_draft(card.id)
    assert controller.document.cards[0].active_revision_id == second_draft.revision_id

    workflow.activate_revision(card.id, first_draft.revision_id)
    assert controller.document.cards[0].active_revision_id == first_draft.revision_id
    workflow.delete_revision(card.id, first_draft.revision_id)
    assert controller.document.cards[0].active_revision_id == second_draft.revision_id
    assert controller.undo()
    assert controller.document.cards[0].active_revision_id == first_draft.revision_id
    assert session.flush()


def test_generation_failure_preserves_document(tmp_path: Path) -> None:
    workflow, controller, _session, workers, card = bound_workflow(tmp_path)
    failures: list[object] = []
    workflow.failed.connect(failures.append)
    before = controller.document

    workflow.generate(card.id)
    error = BackgroundWorkflowError("render failed")
    workers.mflux_operations[0].failed.emit(error)

    assert failures == [error]
    assert not workflow.busy
    assert workflow.draft_for(card.id) is None
    assert controller.document == before


def test_revision_naming_failure_discards_generated_image(tmp_path: Path) -> None:
    workflow, controller, _session, workers, card = bound_workflow(tmp_path)
    failures: list[object] = []
    workflow.failed.connect(failures.append)
    before = controller.document

    workflow.generate(card.id)
    generated = workers.mflux_calls[0]()  # type: ignore[operator]
    workers.mflux_operations[0].succeeded.emit(generated)
    error = BackgroundWorkflowError("naming failed")
    workers.ollama_operations[0].failed.emit(error)

    assert failures == [error]
    assert not workflow.busy
    assert workflow.draft_for(card.id) is None
    assert not generated.output_path.exists()
    assert controller.document == before


def test_generated_revision_names_append_unique_index(tmp_path: Path) -> None:
    workflow, controller, _session, workers, card = bound_workflow(tmp_path)

    workflow.generate(card.id)
    first_generated = workers.mflux_calls[0]()  # type: ignore[operator]
    workers.mflux_operations[0].succeeded.emit(first_generated)
    first_named = workers.ollama_calls[0]()  # type: ignore[operator]
    workers.ollama_operations[0].succeeded.emit(first_named)
    workflow.apply_draft(card.id)

    workflow.generate(card.id)
    second_generated = workers.mflux_calls[1]()  # type: ignore[operator]
    workers.mflux_operations[1].succeeded.emit(second_generated)
    second_named = workers.ollama_calls[1]()  # type: ignore[operator]
    workers.ollama_operations[1].succeeded.emit(second_named)

    draft = workflow.draft_for(card.id)
    assert draft is not None
    assert draft.name == "Moonlit Garden 2"
    workflow.apply_draft(card.id)
    assert [
        revision.name
        for revision in controller.document.cards[0].image_revisions
    ] == ["Moonlit Garden", "Moonlit Garden 2"]

    second_revision_id = controller.document.cards[0].image_revisions[1].id
    workflow.delete_revision(card.id, second_revision_id)
    workflow.generate(card.id)
    third_generated = workers.mflux_calls[2]()  # type: ignore[operator]
    workers.mflux_operations[2].succeeded.emit(third_generated)
    third_named = workers.ollama_calls[2]()  # type: ignore[operator]
    workers.ollama_operations[2].succeeded.emit(third_named)
    third_draft = workflow.draft_for(card.id)
    assert third_draft is not None
    assert third_draft.name == "Moonlit Garden 2"

    assert controller.undo()
    workflow.apply_draft(card.id)
    assert [
        revision.name
        for revision in controller.document.cards[0].image_revisions
    ] == ["Moonlit Garden", "Moonlit Garden 2", "Moonlit Garden 3"]


def test_generation_result_is_discarded_after_document_identity_changes(
    tmp_path: Path,
) -> None:
    workflow, controller, _session, workers, card = bound_workflow(tmp_path)
    failures: list[object] = []
    workflow.failed.connect(failures.append)
    workflow.generate(card.id)
    generated = workers.mflux_calls[0]()  # type: ignore[operator]
    assert generated.output_path.is_file()

    controller.replace_document(Stack(name="Other", cards=(card,)))
    workers.mflux_operations[0].succeeded.emit(generated)

    assert workflow.draft_for(card.id) is None
    assert not generated.output_path.exists()
    assert failures
    assert "stack or card changed" in str(failures[0])


def test_new_candidate_requires_bound_stack_and_resolved_prior_candidate(
    tmp_path: Path,
) -> None:
    card = Card(name="Card")
    controller = DocumentController(Stack(name="Unbound", cards=(card,)))
    session = DocumentSession(controller)
    workflow = BackgroundWorkflow(
        controller,
        session,
        FakeWorkers(),  # type: ignore[arg-type]
        settings,
        ollama_settings,
        temporary_directory=tmp_path / "candidates",
    )

    with pytest.raises(BackgroundWorkflowError, match="save the stack"):
        workflow.import_image(card.id, tmp_path / "missing.png")


def test_drafts_are_card_local_and_replacement_is_atomic(tmp_path: Path) -> None:
    workflow, controller, _session, workers, first_card = bound_workflow(tmp_path)
    second_card = Card(name="Courtyard", scene_description="A courtyard")
    controller.execute(CreateCardCommand(name=second_card.name, card_id=second_card.id))
    first_source = tmp_path / "first.png"
    second_source = tmp_path / "second.png"
    Image.new("RGB", (1024, 768), "red").save(first_source)
    Image.new("RGB", (1024, 768), "blue").save(second_source)

    first_draft = workflow.import_image(first_card.id, first_source)
    second_draft = workflow.import_image(second_card.id, second_source)

    assert workflow.draft_for(first_card.id) == first_draft
    assert workflow.draft_for(second_card.id) == second_draft
    assert workflow.draft_card_ids == {first_card.id, second_card.id}

    workflow.generate(first_card.id, replace_draft=True)
    workers.mflux_operations[0].failed.emit(BackgroundWorkflowError("render failed"))
    assert workflow.draft_for(first_card.id) == first_draft
    assert first_draft.image_path.is_file()

    workflow.generate(first_card.id, replace_draft=True)
    generated = workers.mflux_calls[1]()  # type: ignore[operator]
    workers.mflux_operations[1].succeeded.emit(generated)
    named = workers.ollama_calls[0]()  # type: ignore[operator]
    workers.ollama_operations[0].succeeded.emit(named)
    replacement = workflow.draft_for(first_card.id)
    assert replacement is not None and replacement != first_draft
    assert not first_draft.image_path.exists()
    assert workflow.draft_for(second_card.id) == second_draft


def test_save_as_preserves_compatible_draft_for_acceptance(tmp_path: Path) -> None:
    workflow, controller, session, _workers, card = bound_workflow(tmp_path)
    source = tmp_path / "draft.png"
    Image.new("RGB", (1024, 768), "green").save(source)
    draft = workflow.import_image(card.id, source)

    destination = tmp_path / "Copy.hypergen"
    session.save_as(destination)

    assert workflow.draft_for(card.id) == draft
    workflow.apply_draft(card.id)
    revision = controller.document.cards[0].image_revisions[0]
    assert session.store is not None
    assert session.store.bundle_path == destination
    assert session.store.asset_path(revision.image_path).is_file()


def test_orphaned_draft_is_removed_after_card_creation_is_undone(
    tmp_path: Path,
) -> None:
    workflow, controller, _session, _workers, _card = bound_workflow(tmp_path)
    command = CreateCardCommand(name="Temporary")
    controller.execute(command)
    source = tmp_path / "temporary.png"
    Image.new("RGB", (1024, 768), "purple").save(source)
    draft = workflow.import_image(command.card_id, source)

    assert controller.undo()
    workflow.discard_orphaned_drafts(card.id for card in controller.document.cards)

    assert workflow.draft_for(command.card_id) is None
    assert not draft.image_path.exists()
