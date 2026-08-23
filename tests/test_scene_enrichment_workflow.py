"""Tests for text-only, review-first Description enrichment."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QObject, Signal

from hypergen.application.commands import (
    ActivateRevisionCommand,
    EditRevisionDescriptionCommand,
)
from hypergen.application.document_controller import DocumentController, UndoToken
from hypergen.application.scene_enrichment_workflow import (
    SceneEnrichmentWorkflow,
    SceneEnrichmentWorkflowError,
)
from hypergen.domain.models import Card, CardRevision, Stack
from hypergen.generation.ollama_client import OllamaSettings
from hypergen.generation.scene_enrichment import (
    SCENE_ENRICHMENT_PROMPT_VERSION,
    SceneEnrichmentRequest,
    SceneEnrichmentResult,
)


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


class RecordingEnricher:
    def __init__(self) -> None:
        self.requests: list[SceneEnrichmentRequest] = []

    def enrich(self, request: SceneEnrichmentRequest) -> SceneEnrichmentResult:
        self.requests.append(request)
        return _result()


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
) -> tuple[
    SceneEnrichmentWorkflow,
    DocumentController,
    FakeWorkers,
    RecordingEnricher,
]:
    controller = DocumentController(Stack(name="Stack", cards=(card,)))
    workers = FakeWorkers()
    enricher = RecordingEnricher()
    workflow = SceneEnrichmentWorkflow(
        controller,
        workers,  # type: ignore[arg-type]
        lambda: OllamaSettings(model="test"),
        enricher_factory=lambda _settings: enricher,
    )
    return workflow, controller, workers, enricher


def test_enrichment_requires_review_before_one_undoable_apply() -> None:
    card = Card(
        name="Card",
        revisions=(CardRevision(description="A courtyard"),),
    )
    workflow, controller, workers, enricher = _workflow(card)
    proposals: list[str] = []
    applied: list[object] = []
    workflow.proposal_ready.connect(proposals.append)
    workflow.change_applied.connect(lambda _message, token: applied.append(token))

    workflow.start(card.id)
    assert workers.stages == ["enriching Description"]
    work = workers.calls[0]
    assert callable(work)
    workers.operations[0].succeeded.emit(work())

    assert enricher.requests[0].scene == "A courtyard"
    assert proposals == ["A richer courtyard"]
    assert controller.document.cards[0].active_revision.description == "A courtyard"

    workflow.apply_proposal("An edited richer courtyard")

    assert controller.document.cards[0].active_revision.description == (
        "An edited richer courtyard"
    )
    assert isinstance(applied[0], UndoToken)
    assert controller.undo_if_current(applied[0])  # type: ignore[arg-type]
    assert controller.document.cards[0].active_revision.description == "A courtyard"


def test_empty_description_is_rejected_even_with_an_image() -> None:
    card = Card(name="Card")
    workflow, _controller, _workers, _enricher = _workflow(card)

    with pytest.raises(SceneEnrichmentWorkflowError, match="Description"):
        workflow.start(card.id)


def test_discard_clears_proposal_without_changing_document() -> None:
    card = Card(
        name="Card",
        revisions=(CardRevision(description="Original"),),
    )
    workflow, controller, workers, _enricher = _workflow(card)
    cleared: list[bool] = []
    workflow.proposal_cleared.connect(lambda: cleared.append(True))

    workflow.start(card.id)
    workers.operations[0].succeeded.emit(_result())
    workflow.discard_proposal()

    assert cleared == [True]
    assert not workflow.has_proposal
    assert controller.document.cards[0].active_revision.description == "Original"


def test_stale_or_cancelled_results_do_not_create_proposals() -> None:
    card = Card(
        name="Card",
        revisions=(CardRevision(description="Original"),),
    )
    workflow, controller, workers, _enricher = _workflow(card)
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
    assert not workflow.has_proposal
    assert "changed before enrichment completed" in str(failures[-1])

    workflow.start(card.id)
    operation = workers.operations[-1]
    workflow.cancel()
    operation.succeeded.emit(_result("Late"))
    assert not workflow.has_proposal


def test_proposal_cannot_apply_after_revision_switch() -> None:
    first = CardRevision(description="First")
    second = CardRevision(description="Second")
    card = Card(
        name="Card",
        revisions=(first, second),
        active_revision_id=first.id,
    )
    workflow, controller, workers, _enricher = _workflow(card)

    workflow.start(card.id)
    workers.operations[0].succeeded.emit(_result())
    controller.execute(
        ActivateRevisionCommand(card_id=card.id, revision_id=second.id)
    )

    with pytest.raises(SceneEnrichmentWorkflowError, match="changed"):
        workflow.apply_proposal("Late")

    assert controller.document.cards[0].active_revision.description == "Second"
