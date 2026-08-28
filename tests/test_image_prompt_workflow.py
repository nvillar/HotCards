"""Tests for direct, reviewable Image Prompt preparation."""

from __future__ import annotations

import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PIL import Image
from PySide6.QtCore import QObject, Signal

from hotcards.application.commands import (
    ActivateRevisionCommand,
    EditRevisionDescriptionCommand,
    SetRevisionReferenceCommand,
)
from hotcards.application.document_controller import DocumentController, UndoToken
from hotcards.application.generated_revision_change import GeneratedRevisionChange
from hotcards.application.image_prompt_workflow import (
    ImagePromptWorkflow,
    ImagePromptWorkflowError,
)
from hotcards.domain.models import (
    Card,
    CardRevision,
    GeneratedBackground,
    ImageGenerationInputs,
    ImageGenerationMetadata,
    ResolvedCardReference,
    Stack,
)
from hotcards.generation.image_prompt_preparation import (
    IMAGE_PROMPT_PREPARATION_VERSION,
    MULTI_REFERENCE_IMAGE_PROMPT_PREPARATION_VERSION,
    ImagePromptPreparationRequest,
    ImagePromptPreparationResult,
)
from hotcards.generation.ollama_client import OllamaSettings


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
            tuple[ImagePromptPreparationRequest, Path | None, Path | None]
        ] = []

    def prepare(
        self,
        request: ImagePromptPreparationRequest,
        *,
        reference_image_path: Path | None = None,
        additional_reference_image_path: Path | None = None,
    ) -> ImagePromptPreparationResult:
        self.requests.append(
            (
                request,
                reference_image_path,
                additional_reference_image_path,
            )
        )
        return _result(prompt_version=request.prompt_version)


def _result(
    image_prompt: str = "A richer courtyard",
    *,
    prompt_version: str = IMAGE_PROMPT_PREPARATION_VERSION,
) -> ImagePromptPreparationResult:
    return ImagePromptPreparationResult(
        image_prompt=image_prompt,
        raw_response=f'{{"image_prompt":"{image_prompt}"}}',
        model_identifier="test",
        prompt_version=prompt_version,
        duration_seconds=0.1,
    )


def _background(
    description: str,
    image_path: str,
    *,
    image_prompt: str | None = None,
) -> GeneratedBackground:
    return GeneratedBackground(
        image_path=image_path,
        generation_metadata=ImageGenerationMetadata(
            inputs=ImageGenerationInputs(
                description=description,
                image_prompt=image_prompt or description,
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
        if not path.exists():
            Image.new("RGB", (8, 6), "navy").save(path)
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

    request, reference_path, additional_reference_path = preparer.requests[0]
    assert request.description == "A courtyard"
    assert not request.has_reference
    assert reference_path is None
    assert additional_reference_path is None
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
                references=(ResolvedCardReference(target_card_id=source.id),)
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
                references=(ResolvedCardReference(target_card_id=source.id),),
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

    request, reference_path, additional_reference_path = preparer.requests[0]
    assert request.has_reference
    assert request.reference_description == "Generated prompt"
    assert reference_path == tmp_path / "computer.png"
    assert additional_reference_path is None
    prompt = controller.document.cards[0].active_revision.image_prompt
    assert prompt is not None
    assert prompt.references[0].card_id == source.id
    assert prompt.references[0].revision_id == source.active_revision.id
    assert prompt.references[0].background_id == source.active_revision.background.id


def test_current_reference_card_text_does_not_replace_generation_provenance(
    tmp_path: Path,
) -> None:
    source = Card(
        name="Computer",
        revisions=(
            CardRevision(
                description="Current card text",
                background=_background(
                    "Plain lab Description",
                    "computer.png",
                    image_prompt=(
                        "Black-and-white dithered graphics reminiscent of "
                        "early Mac and HyperCard."
                    ),
                ),
            ),
        ),
    )
    target = Card(
        name="Error",
        revisions=(
            CardRevision(
                description="A close-up of the computer monitor.",
                references=(ResolvedCardReference(target_card_id=source.id),),
            ),
        ),
    )
    workflow, controller, workers, preparer = _workflow(
        tmp_path,
        target,
        source,
    )

    workflow.start(target.id)
    controller.execute(
        EditRevisionDescriptionCommand(
            card_id=source.id,
            revision_id=source.active_revision.id,
            value="Edited after the image was generated",
        )
    )
    _complete(workers)

    request, _reference_path, _additional_reference_path = preparer.requests[0]
    assert request.reference_description == (
        "Black-and-white dithered graphics reminiscent of early Mac and "
        "HyperCard."
    )
    assert (
        controller.document.cards[0].active_revision.image_prompt
        is not None
    )


def test_two_reference_images_are_prepared_and_captured_in_order(
    tmp_path: Path,
) -> None:
    first = Card(
        name="Explorer",
        revisions=(
            CardRevision(
                background=_background(
                    "A compact angular exploration vehicle",
                    "explorer.png",
                )
            ),
        ),
    )
    second = Card(
        name="Hangar",
        revisions=(
            CardRevision(
                background=_background(
                    "A vast monochrome station hangar",
                    "hangar.png",
                )
            ),
        ),
    )
    target = Card(
        name="Target",
        revisions=(
            CardRevision(
                description="Place the Explorer in the Hangar.",
                references=(
                    ResolvedCardReference(target_card_id=first.id),
                    ResolvedCardReference(target_card_id=second.id),
                ),
            ),
        ),
    )
    workflow, controller, workers, preparer = _workflow(
        tmp_path,
        target,
        first,
        second,
    )

    workflow.start(target.id)
    _complete(workers)

    request, first_path, second_path = preparer.requests[0]
    assert request.reference_card_name == "Explorer"
    assert request.additional_reference_card_name == "Hangar"
    assert request.prompt_version == MULTI_REFERENCE_IMAGE_PROMPT_PREPARATION_VERSION
    assert first_path == tmp_path / "explorer.png"
    assert second_path == tmp_path / "hangar.png"
    prompt = controller.document.cards[0].active_revision.image_prompt
    assert prompt is not None
    assert prompt.prompt_version == MULTI_REFERENCE_IMAGE_PROMPT_PREPARATION_VERSION
    assert tuple(snapshot.card_id for snapshot in prompt.references) == (
        first.id,
        second.id,
    )


def test_unreadable_second_reference_is_reported_by_position(
    tmp_path: Path,
) -> None:
    first = Card(
        name="Explorer",
        revisions=(
            CardRevision(
                background=_background("An exploration vehicle", "explorer.png")
            ),
        ),
    )
    second = Card(
        name="Hangar",
        revisions=(
            CardRevision(background=_background("A hangar", "hangar.png")),
        ),
    )
    target = Card(
        name="Target",
        revisions=(
            CardRevision(
                description="Place the vehicle in the hangar.",
                references=(
                    ResolvedCardReference(target_card_id=first.id),
                    ResolvedCardReference(target_card_id=second.id),
                ),
            ),
        ),
    )
    workflow, _controller, _workers, _preparer = _workflow(
        tmp_path,
        target,
        first,
        second,
    )
    Image.new("RGB", (8, 6), "navy").save(tmp_path / "explorer.png")
    (tmp_path / "hangar.png").write_bytes(b"not an image")

    with pytest.raises(ImagePromptWorkflowError, match="Reference 2.*unreadable"):
        workflow.start(target.id)


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
                references=(
                    ResolvedCardReference(target_card_id=first_source.id),
                ),
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
