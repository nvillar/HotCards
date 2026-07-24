"""Tests for identity-bound image-to-Scene replacement."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from hypergen.application.document_controller import DocumentController
from hypergen.application.image_description_workflow import ImageDescriptionWorkflow
from hypergen.domain.models import Card, ImageOrigin, ImageRevision, Stack
from hypergen.generation.image_description import (
    ImageDescriptionRequest,
    ImageDescriptionResult,
)
from hypergen.generation.ollama_client import OllamaSettings


class FakeOperation(QObject):
    succeeded = Signal(object)
    failed = Signal(object)

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
        self.calls: list[object] = []
        self.operations: list[FakeOperation] = []

    def run_ollama(
        self,
        operation: object,
        *,
        stage: str,
    ) -> FakeOperation:
        assert stage == "describing image"
        self.calls.append(operation)
        handle = FakeOperation()
        self.operations.append(handle)
        return handle


class FakeDescriber:
    def __init__(self) -> None:
        self.requests: list[ImageDescriptionRequest] = []

    def describe(self, request: ImageDescriptionRequest) -> ImageDescriptionResult:
        self.requests.append(request)
        return ImageDescriptionResult(
            scene="A generation-ready description of the active image",
            raw_response='{"scene":"A generation-ready description"}',
            model_identifier="qwen3.5:9b",
            prompt_version=request.prompt_version,
            duration_seconds=1.0,
        )


def make_workflow(
    tmp_path: Path,
) -> tuple[
    ImageDescriptionWorkflow,
    DocumentController,
    FakeWorkers,
    FakeDescriber,
    Card,
]:
    image_path = tmp_path / "active.png"
    image_path.write_bytes(b"image")
    revision = ImageRevision(
        image_path="assets/cards/card/image.png",
        origin=ImageOrigin.IMPORTED,
        created_at=datetime.now(UTC),
    )
    card = Card(
        name="Card",
        scene_description="Original Scene",
        interaction_description="Must not be read",
        image_revisions=(revision,),
        active_revision_id=revision.id,
    )
    controller = DocumentController(Stack(name="Demo", cards=(card,)))
    workers = FakeWorkers()
    describer = FakeDescriber()
    workflow = ImageDescriptionWorkflow(
        controller,
        workers,  # type: ignore[arg-type]
        OllamaSettings,
        lambda _relative: image_path,
        describer_factory=lambda _settings: describer,
    )
    return workflow, controller, workers, describer, card


def complete(workers: FakeWorkers) -> ImageDescriptionResult:
    operation = workers.calls[-1]
    assert callable(operation)
    result = operation()
    workers.operations[-1].succeeded.emit(result)
    return result


def test_success_replaces_scene_as_one_undoable_edit(tmp_path: Path) -> None:
    workflow, controller, workers, describer, card = make_workflow(tmp_path)

    workflow.start(card.id)
    complete(workers)

    assert describer.requests[0].image_path.name == "active.png"
    assert (
        controller.document.cards[0].scene_description
        == "A generation-ready description of the active image"
    )
    assert controller.document.cards[0].interaction_description == "Must not be read"
    assert controller.undo()
    assert controller.document.cards[0].scene_description == "Original Scene"


def test_failure_cancel_and_stale_result_leave_scene_unchanged(
    tmp_path: Path,
) -> None:
    workflow, controller, workers, _describer, card = make_workflow(tmp_path)
    failures: list[object] = []
    workflow.failed.connect(failures.append)

    workflow.start(card.id)
    workers.operations[-1].failed.emit(RuntimeError("Ollama stopped"))
    assert controller.document.cards[0].scene_description == "Original Scene"
    assert failures

    workflow.start(card.id)
    workflow.cancel()
    assert workers.operations[-1].cancelled
    assert controller.document.cards[0].scene_description == "Original Scene"

    workflow.start(card.id)
    controller.replace_document(Stack(name="Replacement", cards=(Card(name="Other"),)))
    complete(workers)
    assert controller.document.cards[0].scene_description == ""
    assert failures
