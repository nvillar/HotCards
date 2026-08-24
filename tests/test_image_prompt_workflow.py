"""Tests for direct, reviewable Image Prompt preparation."""

from __future__ import annotations

import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QObject, Signal

from hypergen.application.commands import (
    ActivateRevisionCommand,
    EditRevisionDescriptionCommand,
    SetRevisionReferenceCommand,
)
from hypergen.application.document_controller import DocumentController, UndoToken
from hypergen.application.generated_revision_change import GeneratedRevisionChange
from hypergen.application.image_prompt_workflow import (
    ImagePromptWorkflow,
    ImagePromptWorkflowError,
)
from hypergen.domain.models import (
    Card,
    CardRevision,
    GeneratedBackground,
    ImageGenerationInputs,
    ImageGenerationMetadata,
    ResolvedCardReference,
    Stack,
)
from hypergen.generation.image_prompt_preparation import (
    IMAGE_PROMPT_PREPARATION_VERSION,
    ImagePromptPreparationRequest,
    ImagePromptPreparationResult,
)
from hypergen.generation.ollama_client import OllamaSettings


class FakeOperation(QObject):
    succeeded = Signal(object)
    failed = Signal(object)

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
        self.stages: list[str] = []
        self.operations: list[FakeOperation] = []

    def run_ollama(self, operation: object, *, stage: str) -> FakeOperation:
        self.calls.append(operation)
        self.stages.append(stage)
        handle = FakeOperation()
        self.operations.append(handle)
        return handle


class RecordingPreparer:
    def __init__(self) -> None:
        self.requests: list[
            tuple[ImagePromptPreparationRequest, Path | None]
        ] = []

    def prepare(
        self,
        request: ImagePromptPreparationRequest,
        *,
        reference_image_path: Path | None = None,
    ) -> ImagePromptPreparationResult:
        self.requests.append((request, reference_image_path))
        return _result()


def _result(
    image_prompt: str = "A richer courtyard",
) -> ImagePromptPreparationResult:
    return ImagePromptPreparationResult(
        image_prompt=image_prompt,
        raw_response=f'{{"image_prompt":"{image_prompt}"}}',
        model_identifier="test",
        prompt_version=IMAGE_PROMPT_PREPARATION_VERSION,
        duration_seconds=0.1,
    )


def _background(description: str, image_path: str) -> GeneratedBackground:
    return GeneratedBackground(
        image_path=image_path,
        generation_metadata=ImageGenerationMetadata(
            inputs=ImageGenerationInputs(
                description=description,
                image_prompt=description,
            ),
            render_prompt=description,
            model_identifier="flux2-klein-4b",
            mflux_version="0.18.0",
            seed=1,
            width=1024,
            height=768,
            step_count=4,
            generated_at=datetime.now(UTC),
            duration_seconds=1,
        ),
        created_at=datetime.now(UTC),
    )


def _workflow(
    tmp_path: Path,
    card: Card,
    *other_cards: Card,
    settings_provider: Callable[[], OllamaSettings] | None = None,
) -> tuple[
    ImagePromptWorkflow,
    DocumentController,
    FakeWorkers,
    RecordingPreparer,
]:
    controller = DocumentController(
        Stack(name="Stack", cards=(card, *other_cards))
    )
    workers = FakeWorkers()
    preparer = RecordingPreparer()

    def resolve(image_path: str) -> Path:
        path = tmp_path / image_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"image")
        return path

    workflow = ImagePromptWorkflow(
        controller,
        workers,  # type: ignore[arg-type]
        settings_provider or (lambda: OllamaSettings(model="test")),
        reference_image_resolver=resolve,
        preparer_factory=lambda _settings: preparer,
    )
    return workflow, controller, workers, preparer


def _complete(workers: FakeWorkers) -> None:
    work = workers.calls[-1]
    assert callable(work)
    workers.operations[-1].succeeded.emit(work())


def test_preparation_applies_with_one_undo_boundary(tmp_path: Path) -> None:
    card = Card(
        name="Card",
        revisions=(CardRevision(description="A courtyard"),),
    )
    workflow, controller, workers, preparer = _workflow(tmp_path, card)
    applied: list[GeneratedRevisionChange] = []
    workflow.generation_applied.connect(applied.append)

    workflow.start(card.id)
    assert workers.stages == ["preparing Image Prompt"]
    _complete(workers)

    request, reference_path = preparer.requests[0]
    assert request.description == "A courtyard"
    assert not request.has_reference
    assert reference_path is None
    revision = controller.document.cards[0].active_revision
    assert revision.description == "A courtyard"
    assert revision.image_prompt is not None
    assert revision.image_prompt.text == "A richer courtyard"
    assert applied[0].message == "Image Prompt prepared"
    assert isinstance(applied[0].token, UndoToken)
    assert controller.undo_if_current(applied[0].token)
    assert controller.document.cards[0].active_revision.image_prompt is None


def test_empty_description_is_rejected_even_with_a_reference(
    tmp_path: Path,
) -> None:
    source = Card(
        name="Source",
        revisions=(CardRevision(background=_background("Source", "source.png")),),
    )
    card = Card(
        name="Card",
        revisions=(
            CardRevision(
                reference=ResolvedCardReference(target_card_id=source.id)
            ),
        ),
    )
    workflow, _controller, _workers, _preparer = _workflow(
        tmp_path,
        card,
        source,
    )

    with pytest.raises(ImagePromptWorkflowError, match="Description"):
        workflow.start(card.id)


def test_reference_image_is_attached_once_and_captured_in_provenance(
    tmp_path: Path,
) -> None:
    source = Card(
        name="Computer",
        revisions=(
            CardRevision(
                description="Source text may later change",
                background=_background("Generated prompt", "computer.png"),
            ),
        ),
    )
    target = Card(
        name="Error",
        revisions=(
            CardRevision(
                description='The screen now displays "ERROR".',
                reference=ResolvedCardReference(target_card_id=source.id),
            ),
        ),
    )
    workflow, controller, workers, preparer = _workflow(
        tmp_path,
        target,
        source,
    )

    workflow.start(target.id)
    _complete(workers)

    request, reference_path = preparer.requests[0]
    assert request.has_reference
    assert reference_path == tmp_path / "computer.png"
    prompt = controller.document.cards[0].active_revision.image_prompt
    assert prompt is not None
    assert prompt.reference is not None
    assert prompt.reference.card_id == source.id
    assert prompt.reference.revision_id == source.active_revision.id
    assert prompt.reference.background_id == source.active_revision.background.id


def test_stale_description_or_reference_result_does_not_apply(
    tmp_path: Path,
) -> None:
    first_source = Card(
        name="First",
        revisions=(CardRevision(background=_background("First", "first.png")),),
    )
    second_source = Card(
        name="Second",
        revisions=(CardRevision(background=_background("Second", "second.png")),),
    )
    target = Card(
        name="Target",
        revisions=(
            CardRevision(
                description="The screen has changed",
                reference=ResolvedCardReference(target_card_id=first_source.id),
            ),
        ),
    )
    workflow, controller, workers, _preparer = _workflow(
        tmp_path,
        target,
        first_source,
        second_source,
    )
    failures: list[object] = []
    workflow.failed.connect(failures.append)

    workflow.start(target.id)
    controller.execute(
        SetRevisionReferenceCommand(
            card_id=target.id,
            revision_id=target.active_revision.id,
            reference=ResolvedCardReference(target_card_id=second_source.id),
        )
    )
    _complete(workers)

    assert controller.document.cards[0].active_revision.image_prompt is None
    assert "changed before" in str(failures[-1])

    workflow.start(target.id)
    controller.execute(
        EditRevisionDescriptionCommand(
            card_id=target.id,
            revision_id=target.active_revision.id,
            value="A different Description",
        )
    )
    _complete(workers)
    assert controller.document.cards[0].active_revision.image_prompt is None


def test_result_does_not_apply_after_revision_switch(tmp_path: Path) -> None:
    first = CardRevision(description="First")
    second = CardRevision(description="Second")
    card = Card(
        name="Card",
        revisions=(first, second),
        active_revision_id=first.id,
    )
    workflow, controller, workers, _preparer = _workflow(tmp_path, card)
    failures: list[object] = []
    workflow.failed.connect(failures.append)

    workflow.start(card.id)
    controller.execute(
        ActivateRevisionCommand(card_id=card.id, revision_id=second.id)
    )
    workers.operations[0].succeeded.emit(_result("Late"))

    assert controller.document.cards[0].active_revision.description == "Second"
    assert "changed before" in str(failures[-1])


def test_cancelled_result_is_ignored(tmp_path: Path) -> None:
    card = Card(
        name="Card",
        revisions=(CardRevision(description="Original"),),
    )
    workflow, controller, workers, _preparer = _workflow(tmp_path, card)

    operation = workflow.start(card.id)
    workflow.cancel()
    operation.succeeded.emit(_result("Late"))

    assert operation.cancelled
    assert controller.document.cards[0].active_revision.image_prompt is None
