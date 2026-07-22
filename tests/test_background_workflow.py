"""Tests for transient background candidates and durable revision application."""

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
from hypergen.application.document_controller import DocumentController
from hypergen.application.document_session import DocumentSession
from hypergen.domain.models import Card, ImageGenerationInputs, Stack
from hypergen.generation.image_prompts import DerivedRenderPrompt
from hypergen.generation.mflux_generator import MfluxGenerator
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
        self.ollama_calls: list[object] = []
        self.mflux_calls: list[object] = []
        self.ollama_operations: list[FakeOperation] = []
        self.mflux_operations: list[FakeOperation] = []

    def run_ollama(self, operation: object, *, stage: str) -> FakeOperation:
        assert stage == "deriving background image prompt"
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


class FakePromptDeriver:
    def __init__(self) -> None:
        self.inputs: list[ImageGenerationInputs] = []

    def derive(self, inputs: ImageGenerationInputs) -> DerivedRenderPrompt:
        self.inputs.append(inputs)
        return DerivedRenderPrompt(
            prompt="Storybook garden with a visible gate",
            interactive_subjects=("gate",),
            raw_response='{"render_prompt":"garden"}',
            model_identifier="qwen3.5:9b",
            duration_seconds=0.1,
        )


def settings() -> BackgroundGenerationSettings:
    return BackgroundGenerationSettings(
        ollama_endpoint="http://localhost:11434",
        ollama_model="qwen3.5:9b",
        mflux_model="flux2-klein-4b",
        step_count=4,
        quantization=None,
        random_seed=False,
        fixed_seed=42,
    )


def bound_workflow(
    tmp_path: Path,
) -> tuple[
    BackgroundWorkflow,
    DocumentController,
    DocumentSession,
    FakeWorkers,
    FakePromptDeriver,
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
            art_direction="Storybook",
            cards=(card,),
            start_card_id=card.id,
        )
    )
    session = DocumentSession(controller)
    session.create(controller.document, tmp_path / "Stack.hypergen")
    workers = FakeWorkers()
    deriver = FakePromptDeriver()
    generator = MfluxGenerator(model_factory=lambda *_args: FakeMfluxModel())
    workflow = BackgroundWorkflow(
        controller,
        session,
        workers,  # type: ignore[arg-type]
        settings,
        prompt_deriver_factory=lambda _settings: deriver,
        mflux_generator=generator,
        temporary_directory=tmp_path / "candidates",
    )
    return workflow, controller, session, workers, deriver, card


def test_generate_then_apply_creates_durable_revision(tmp_path: Path) -> None:
    workflow, controller, session, workers, deriver, card = bound_workflow(tmp_path)

    workflow.generate(card.id)
    prompt_call = workers.ollama_calls[0]
    derived = prompt_call()  # type: ignore[operator]
    workers.ollama_operations[0].succeeded.emit(derived)
    generated = workers.mflux_calls[0]()  # type: ignore[operator]
    workers.mflux_operations[0].succeeded.emit(generated)

    candidate = workflow.candidate
    assert candidate is not None
    assert candidate.image_path.is_file()
    assert deriver.inputs == [
        ImageGenerationInputs(
            scene_description="A garden",
            interaction_description="The gate opens",
            stack_art_direction="Storybook",
            card_style="Pencil",
        )
    ]

    workflow.apply_candidate()
    assert workflow.candidate is None
    revision = controller.document.cards[0].image_revisions[0]
    assert revision.id == candidate.revision_id
    assert revision.generation_metadata is not None
    assert revision.generation_metadata.derived_prompt == (
        "Storybook garden with a visible gate"
    )
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

    workflow, controller, session, _workers, _deriver, card = bound_workflow(tmp_path)
    original = controller.document
    candidate = workflow.import_image(card.id, source, position_x=0.25)
    assert candidate.source_filename == "wide.png"
    assert controller.document == original
    bundle_path = session.state.bundle_path
    assert bundle_path is not None
    assert StackStore(bundle_path).load() == original
    with pytest.raises(BackgroundWorkflowError, match="apply or discard"):
        workflow.import_image(card.id, source)
    workflow.discard_candidate()
    assert workflow.candidate is None
    assert not candidate.image_path.exists()
    assert controller.document == original


def test_revision_switch_delete_and_undo_restore_association(tmp_path: Path) -> None:
    workflow, controller, session, _workers, _deriver, card = bound_workflow(tmp_path)
    first_source = tmp_path / "first.png"
    second_source = tmp_path / "second.png"
    Image.new("RGB", (1024, 768), "red").save(first_source)
    Image.new("RGB", (1024, 768), "blue").save(second_source)

    first_candidate = workflow.import_image(card.id, first_source)
    workflow.apply_candidate()
    second_candidate = workflow.import_image(card.id, second_source)
    workflow.apply_candidate()
    assert controller.document.cards[0].active_revision_id == second_candidate.revision_id

    workflow.activate_revision(card.id, first_candidate.revision_id)
    assert controller.document.cards[0].active_revision_id == first_candidate.revision_id
    workflow.delete_revision(card.id, first_candidate.revision_id)
    assert controller.document.cards[0].active_revision_id == second_candidate.revision_id
    assert controller.undo()
    assert controller.document.cards[0].active_revision_id == first_candidate.revision_id
    assert session.flush()


def test_generation_failure_preserves_document(tmp_path: Path) -> None:
    workflow, controller, _session, workers, _deriver, card = bound_workflow(tmp_path)
    failures: list[object] = []
    workflow.failed.connect(failures.append)
    before = controller.document

    workflow.generate(card.id)
    error = BackgroundWorkflowError("prompt failed")
    workers.ollama_operations[0].failed.emit(error)

    assert failures == [error]
    assert not workflow.busy
    assert workflow.candidate is None
    assert controller.document == before


def test_generation_result_is_discarded_after_document_identity_changes(
    tmp_path: Path,
) -> None:
    workflow, controller, _session, workers, _deriver, card = bound_workflow(tmp_path)
    failures: list[object] = []
    workflow.failed.connect(failures.append)
    workflow.generate(card.id)
    derived = workers.ollama_calls[0]()  # type: ignore[operator]
    workers.ollama_operations[0].succeeded.emit(derived)
    generated = workers.mflux_calls[0]()  # type: ignore[operator]
    assert generated.output_path.is_file()

    controller.replace_document(Stack(name="Other", cards=(card,)))
    workers.mflux_operations[0].succeeded.emit(generated)

    assert workflow.candidate is None
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
        temporary_directory=tmp_path / "candidates",
    )

    with pytest.raises(BackgroundWorkflowError, match="save the stack"):
        workflow.import_image(card.id, tmp_path / "missing.png")
