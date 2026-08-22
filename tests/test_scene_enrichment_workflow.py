"""Tests for direct image-aware Description enrichment."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PIL import Image
from PySide6.QtCore import QObject, Signal

from hypergen.application.commands import (
    DuplicateRevisionCommand,
    EditRevisionDescriptionCommand,
)
from hypergen.application.document_controller import DocumentController, UndoToken
from hypergen.application.scene_enrichment_workflow import (
    SceneEnrichmentWorkflow,
    SceneEnrichmentWorkflowError,
)
from hypergen.domain.models import (
    Card,
    CardRevision,
    GenerationStyle,
    ImportedBackground,
    Stack,
)
from hypergen.generation.image_description import (
    IMAGE_DESCRIPTION_PROMPT_VERSION,
    ImageDescriptionRequest,
    ImageDescriptionResult,
)
from hypergen.generation.ollama_client import OllamaSettings
from hypergen.generation.scene_enrichment import (
    SCENE_ENRICHMENT_PROMPT_VERSION,
    SceneEnrichmentRequest,
    SceneEnrichmentResult,
)


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
        self.stages: list[str] = []
        self.operations: list[FakeOperation] = []

    def run_ollama(self, operation: object, *, stage: str) -> FakeOperation:
        self.calls.append(operation)
        self.stages.append(stage)
        handle = FakeOperation()
        self.operations.append(handle)
        return handle


class RecordingEnricher:
    def __init__(self) -> None:
        self.requests: list[SceneEnrichmentRequest] = []

    def enrich(self, request: SceneEnrichmentRequest) -> SceneEnrichmentResult:
        self.requests.append(request)
        return SceneEnrichmentResult(
            scene="A richer courtyard",
            raw_response='{"scene":"A richer courtyard"}',
            model_identifier="test",
            prompt_version=SCENE_ENRICHMENT_PROMPT_VERSION,
            duration_seconds=0.1,
        )


class RecordingDescriber:
    def __init__(self) -> None:
        self.requests: list[ImageDescriptionRequest] = []

    def describe(self, request: ImageDescriptionRequest) -> ImageDescriptionResult:
        self.requests.append(request)
        return ImageDescriptionResult(
            scene="A visible stone arch in warm sunset light",
            raw_response='{"scene":"A visible stone arch in warm sunset light"}',
            model_identifier="test",
            prompt_version=IMAGE_DESCRIPTION_PROMPT_VERSION,
            duration_seconds=0.1,
        )


def _result(scene: str = "A richer courtyard") -> SceneEnrichmentResult:
    return SceneEnrichmentResult(
        scene=scene,
        raw_response=f'{{"scene":"{scene}"}}',
        model_identifier="test",
        prompt_version=SCENE_ENRICHMENT_PROMPT_VERSION,
        duration_seconds=0.1,
    )


def _workflow(
    card: Card,
    *,
    styles: tuple[GenerationStyle, ...] = (),
    image_path: Path | None = None,
) -> tuple[
    SceneEnrichmentWorkflow,
    DocumentController,
    FakeWorkers,
    RecordingEnricher,
    RecordingDescriber,
]:
    controller = DocumentController(
        Stack(name="Stack", styles=styles, cards=(card,))
    )
    workers = FakeWorkers()
    enricher = RecordingEnricher()
    describer = RecordingDescriber()
    workflow = SceneEnrichmentWorkflow(
        controller,
        workers,  # type: ignore[arg-type]
        lambda: OllamaSettings(model="test"),
        (
            (lambda _relative: image_path)
            if image_path is not None
            else None
        ),
        enricher_factory=lambda _settings: enricher,
        describer_factory=lambda _settings: describer,
    )
    return workflow, controller, workers, enricher, describer


def test_text_only_enrichment_applies_once_and_is_undoable() -> None:
    card = Card(
        name="Card",
        revisions=(CardRevision(description="A courtyard"),),
    )
    workflow, controller, workers, enricher, _describer = _workflow(card)
    applied: list[object] = []
    workflow.change_applied.connect(lambda _message, token: applied.append(token))

    workflow.start(card.id)
    assert workers.stages == ["enriching Description"]
    work = workers.calls[0]
    assert callable(work)
    workers.operations[0].succeeded.emit(work())

    assert enricher.requests[0].scene == "A courtyard"
    assert enricher.requests[0].image_description is None
    assert controller.document.cards[0].active_revision.description == (
        "A richer courtyard"
    )
    assert isinstance(applied[0], UndoToken)
    assert controller.undo_if_current(applied[0])  # type: ignore[arg-type]
    assert controller.document.cards[0].active_revision.description == "A courtyard"


def test_image_path_runs_description_then_enrichment_with_style(
    tmp_path: Path,
) -> None:
    image = tmp_path / "image.png"
    Image.new("RGB", (32, 24), "navy").save(image)
    style = GenerationStyle(name="Ink", prompt="Fine black ink")
    asset_id = uuid4()
    card = Card(
        name="Card",
        revisions=(
            CardRevision(
                description="An authored courtyard",
                style_id=style.id,
                background=ImportedBackground(
                    id=asset_id,
                    image_path="assets/image.png",
                    source_filename="image.png",
                    created_at=datetime.now(UTC),
                ),
            ),
        ),
    )
    workflow, controller, workers, enricher, describer = _workflow(
        card,
        styles=(style,),
        image_path=image,
    )

    workflow.start(card.id)
    assert workers.stages == ["describing background"]
    describe = workers.calls[0]
    assert callable(describe)
    workers.operations[0].succeeded.emit(describe())
    assert workers.stages == ["describing background", "enriching Description"]
    enrich = workers.calls[1]
    assert callable(enrich)
    workers.operations[1].succeeded.emit(enrich())

    assert describer.requests[0].image_path == image
    assert enricher.requests[0].scene == "An authored courtyard"
    assert enricher.requests[0].effective_style == "Fine black ink"
    assert enricher.requests[0].image_description == (
        "A visible stone arch in warm sunset light"
    )
    assert controller.document.cards[0].active_revision.description == (
        "A richer courtyard"
    )


def test_image_allows_empty_authored_description(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    Image.new("RGB", (10, 10), "red").save(image)
    asset_id = uuid4()
    card = Card(
        name="Card",
        revisions=(
            CardRevision(
                background=ImportedBackground(
                    id=asset_id,
                    image_path="assets/image.png",
                    source_filename="image.png",
                    created_at=datetime.now(UTC),
                )
            ),
        ),
    )
    workflow, _controller, workers, _enricher, _describer = _workflow(
        card,
        image_path=image,
    )

    workflow.start(card.id)
    describe = workers.calls[0]
    assert callable(describe)
    workers.operations[0].succeeded.emit(describe())
    enrich = workers.calls[1]
    assert callable(enrich)
    workers.operations[1].succeeded.emit(enrich())


def test_empty_description_without_readable_image_is_rejected() -> None:
    card = Card(name="Card")
    workflow, _controller, _workers, _enricher, _describer = _workflow(card)

    with pytest.raises(SceneEnrichmentWorkflowError, match="Description"):
        workflow.start(card.id)


def test_stage_failure_does_not_change_description(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    Image.new("RGB", (10, 10), "red").save(image)
    asset_id = uuid4()
    card = Card(
        name="Card",
        revisions=(
            CardRevision(
                description="Original",
                background=ImportedBackground(
                    id=asset_id,
                    image_path="assets/image.png",
                    source_filename="image.png",
                    created_at=datetime.now(UTC),
                ),
            ),
        ),
    )
    workflow, controller, workers, _enricher, _describer = _workflow(
        card,
        image_path=image,
    )
    failures: list[object] = []
    workflow.failed.connect(failures.append)

    workflow.start(card.id)
    workers.operations[0].failed.emit(RuntimeError("vision failed"))

    assert controller.document.cards[0].active_revision.description == "Original"
    assert failures[-1].args == ("vision failed",)
    assert not workflow.busy


def test_stale_or_cancelled_results_do_not_apply() -> None:
    card = Card(
        name="Card",
        revisions=(CardRevision(description="Original"),),
    )
    workflow, controller, workers, _enricher, _describer = _workflow(card)
    failures: list[object] = []
    workflow.failed.connect(failures.append)

    workflow.start(card.id)
    revision = controller.document.cards[0].active_revision
    controller.execute(
        EditRevisionDescriptionCommand(
            card_id=card.id,
            revision_id=revision.id,
            value="Changed",
        )
    )
    workers.operations[0].succeeded.emit(_result())
    assert controller.document.cards[0].active_revision.description == "Changed"
    assert "changed before enrichment completed" in str(failures[-1])

    workflow.start(card.id)
    operation = workers.operations[-1]
    workflow.cancel()
    operation.succeeded.emit(_result("Late"))
    assert controller.document.cards[0].active_revision.description == "Changed"


def test_revision_switch_suppresses_in_flight_result() -> None:
    card = Card(
        name="Card",
        revisions=(
            CardRevision(description="First"),
            CardRevision(description="Second"),
        ),
    )
    workflow, controller, workers, _enricher, _describer = _workflow(card)

    workflow.start(card.id)
    first = controller.document.cards[0].active_revision
    controller.execute(
        DuplicateRevisionCommand(
            card_id=card.id,
            source_revision_id=first.id,
        )
    )
    workers.operations[0].succeeded.emit(_result())

    assert controller.document.cards[0].active_revision.description == "First"
    assert controller.document.cards[0].revisions[0].description == "First"
