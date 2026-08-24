"""Tests for direct revision-local background changes."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PIL import Image
from PySide6.QtCore import QObject, Signal

from hypergen.application.background_workflow import (
    BackgroundGenerationSettings,
    BackgroundWorkflow,
    BackgroundWorkflowError,
)
from hypergen.application.commands import (
    CreateCardCommand,
    DuplicateRevisionCommand,
    EditRevisionDescriptionCommand,
    SetRevisionImagePromptCommand,
    SetRevisionReferenceCommand,
)
from hypergen.application.document_controller import DocumentController, UndoToken
from hypergen.application.document_session import DocumentSession
from hypergen.application.generated_revision_change import GeneratedRevisionChange
from hypergen.domain.models import (
    Card,
    CardRevision,
    HotspotSet,
    ImagePrompt,
    ImageReferenceSnapshot,
    Interaction,
    NavigateAction,
    ResolvedCardReference,
    Stack,
    UnresolvedCardReference,
)
from hypergen.generation.image_prompt_preparation import (
    IMAGE_PROMPT_PREPARATION_VERSION,
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
    def __init__(self) -> None:
        self.callbacks = FakeCallbackRegistry()

    def generate_image(self, **kwargs: object) -> FakeMfluxImage:
        config = SimpleNamespace(
            num_inference_steps=kwargs["num_inference_steps"]
        )
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


class FakeCallbackRegistry:
    def __init__(self) -> None:
        self.registered: list[object] = []

    def register(self, callback: object) -> None:
        self.registered.append(callback)


def _settings() -> BackgroundGenerationSettings:
    return BackgroundGenerationSettings(
        mflux_model="flux2-klein-4b",
        step_count=4,
        quantization=None,
        random_seed=False,
        fixed_seed=42,
        ollama_model="test",
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
        image_prompt=ImagePrompt(
            text="A richly detailed garden",
            source_description="A garden",
            model_identifier="test",
            prompt_version=IMAGE_PROMPT_PREPARATION_VERSION,
        ),
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
            model_factory=lambda *_args: FakeMfluxModel(),
            edit_model_factory=lambda *_args: FakeMfluxModel(),
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
    applied: list[GeneratedRevisionChange] = []
    progress: list[tuple[int, int]] = []
    workflow.generation_applied.connect(applied.append)
    workflow.generation_progress_changed.connect(
        lambda completed, total: progress.append((completed, total))
    )

    workflow.generate(card.id)
    _complete_generation(workers)

    assert progress == [
        (0, 4),
        (1, 4),
        (2, 4),
        (3, 4),
        (4, 4),
        (0, 0),
    ]
    revision = controller.document.cards[0].active_revision
    assert revision.background is not None
    assert revision.background.type == "generated"
    assert revision.description == "A garden"
    assert revision.hotspot_set is not None
    assert revision.hotspot_set.interactions[0].label == "Unresolved"
    metadata = revision.generation_metadata
    assert metadata is not None
    assert metadata.inputs.description == "A garden"
    assert metadata.inputs.image_prompt == "A richly detailed garden"
    assert metadata.inputs.reference is None
    assert session.flush()
    assert StackStore(session.state.bundle_path).load() == controller.document
    assert applied[0].message == "Image generated"
    assert applied[0].card_id == card.id
    assert applied[0].revision_id == card.active_revision.id
    assert applied[0].previous_revision.background is None
    assert isinstance(applied[0].token, UndoToken)

    assert controller.undo_if_current(applied[0].token)
    assert controller.document.cards[0].active_revision.background is None


def test_generate_uses_current_image_prompt(tmp_path: Path) -> None:
    workflow, controller, _session, workers, card = _bound_workflow(tmp_path)
    controller.execute(
        SetRevisionImagePromptCommand(
            card_id=card.id,
            revision_id=card.active_revision.id,
            value=ImagePrompt(
                text="A richly detailed garden",
                source_description="A garden",
                model_identifier="test",
                prompt_version=IMAGE_PROMPT_PREPARATION_VERSION,
            ),
        )
    )

    workflow.generate(card.id)
    _complete_generation(workers)

    revision = controller.document.cards[0].active_revision
    assert revision.description == "A garden"
    assert revision.image_prompt is not None
    assert revision.generation_metadata is not None
    assert revision.generation_metadata.inputs.description == "A garden"
    assert revision.generation_metadata.inputs.image_prompt == (
        "A richly detailed garden"
    )
    assert revision.generation_metadata.render_prompt == (
        "A richly detailed garden"
    )


def test_generate_rejects_image_prompt_after_authored_text_changes(
    tmp_path: Path,
) -> None:
    workflow, controller, _session, workers, card = _bound_workflow(tmp_path)
    revision = controller.document.cards[0].active_revision
    controller.execute(
        EditRevisionDescriptionCommand(
            card_id=card.id,
            revision_id=revision.id,
            value="",
        )
    )
    with pytest.raises(BackgroundWorkflowError, match="Description"):
        workflow.generate(card.id)
    assert workers.calls == []


def test_authored_edit_stales_image_prompt_generation(
    tmp_path: Path,
) -> None:
    workflow, controller, _session, workers, card = _bound_workflow(tmp_path)
    revision = card.active_revision
    controller.execute(
        SetRevisionImagePromptCommand(
            card_id=card.id,
            revision_id=revision.id,
            value=ImagePrompt(
                text="A richly detailed garden",
                source_description="A garden",
                model_identifier="test",
                prompt_version=IMAGE_PROMPT_PREPARATION_VERSION,
            ),
        )
    )

    controller.execute(
        EditRevisionDescriptionCommand(
            card_id=card.id,
            revision_id=revision.id,
            value="A changed authored garden",
        )
    )
    with pytest.raises(BackgroundWorkflowError, match="current Image Prompt"):
        workflow.generate(card.id)


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


def test_references_capture_exact_source_and_suppress_stale_results(
    tmp_path: Path,
) -> None:
    workflow, controller, _session, workers, target = _bound_workflow(tmp_path)
    source_id = uuid4()
    controller.execute(CreateCardCommand(name="Portrait", card_id=source_id))
    source = next(card for card in controller.document.cards if card.id == source_id)
    controller.execute(
        EditRevisionDescriptionCommand(
            card_id=source.id,
            revision_id=source.active_revision.id,
            value="A distinctive knight portrait",
        )
    )
    controller.execute(
        SetRevisionImagePromptCommand(
            card_id=source.id,
            revision_id=source.active_revision.id,
            value=ImagePrompt(
                text="A distinctive knight portrait",
                source_description="A distinctive knight portrait",
                model_identifier="test",
                prompt_version=IMAGE_PROMPT_PREPARATION_VERSION,
            ),
        )
    )
    workflow.generate(source.id)
    _complete_generation(workers)
    source = next(card for card in controller.document.cards if card.id == source_id)
    source_revision = source.active_revision
    assert source_revision.background is not None

    controller.execute(
        SetRevisionReferenceCommand(
            card_id=target.id,
            revision_id=target.active_revision.id,
            reference=ResolvedCardReference(target_card_id=source.id),
        )
    )
    snapshot = ImageReferenceSnapshot(
        card_id=source.id,
        revision_id=source_revision.id,
        background_id=source_revision.background.id,
    )
    controller.execute(
        SetRevisionImagePromptCommand(
            card_id=target.id,
            revision_id=target.active_revision.id,
            value=ImagePrompt(
                text="A richly detailed garden with the referenced visual treatment",
                source_description="A garden",
                reference=snapshot,
                model_identifier="test",
                prompt_version=IMAGE_PROMPT_PREPARATION_VERSION,
            ),
        )
    )
    workflow.generate(target.id)
    _complete_generation(workers)

    target_revision = controller.document.cards[0].active_revision
    metadata = target_revision.generation_metadata
    assert metadata is not None
    assert metadata.inputs.reference is not None
    assert metadata.inputs.reference.card_id == source.id
    assert metadata.inputs.reference.revision_id == source_revision.id
    assert metadata.inputs.reference.background_id == (
        source_revision.background.id
    )
    assert metadata.render_prompt == (
        "A richly detailed garden with the referenced visual treatment"
    )
    assert "Use image" not in metadata.render_prompt
    assert metadata.effective_settings["reference_count"] == 1

    failures: list[object] = []
    workflow.failed.connect(failures.append)
    current_background = target_revision.background
    workflow.generate(target.id)
    controller.execute(
        DuplicateRevisionCommand(
            card_id=source.id,
            source_revision_id=source_revision.id,
        )
    )
    _complete_generation(workers)

    assert controller.document.cards[0].active_revision.background == current_background
    assert "changed before generation completed" in str(failures[-1])


def test_reference_card_requires_an_active_image(tmp_path: Path) -> None:
    workflow, controller, _session, _workers, target = _bound_workflow(tmp_path)
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
