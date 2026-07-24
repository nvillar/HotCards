"""Tests for transient, identity-bound Scene enrichment."""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal

from hypergen.application.document_controller import DocumentController
from hypergen.application.scene_enrichment_workflow import SceneEnrichmentWorkflow
from hypergen.domain.models import Card, Stack
from hypergen.generation.ollama_client import OllamaSettings
from hypergen.generation.scene_enrichment import (
    SceneEnrichmentRequest,
    SceneEnrichmentResult,
)


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
        assert stage == "enriching Scene"
        self.calls.append(operation)
        handle = FakeOperation()
        self.operations.append(handle)
        return handle


class FakeEnricher:
    def __init__(self) -> None:
        self.requests: list[SceneEnrichmentRequest] = []

    def enrich(self, request: SceneEnrichmentRequest) -> SceneEnrichmentResult:
        self.requests.append(request)
        return SceneEnrichmentResult(
            scene="A moonlit ancient wood veiled in silver mist",
            raw_response='{"scene":"A moonlit ancient wood veiled in silver mist"}',
            model_identifier="qwen3.5:9b",
            prompt_version=request.prompt_version,
            duration_seconds=1.0,
        )


def make_workflow() -> tuple[
    SceneEnrichmentWorkflow,
    DocumentController,
    FakeWorkers,
    FakeEnricher,
    Card,
]:
    card = Card(name="Wood", scene_description="A mysterious wood")
    controller = DocumentController(
        Stack(
            name="Demo",
            global_style="Ink and watercolor",
            cards=(card,),
        )
    )
    workers = FakeWorkers()
    enricher = FakeEnricher()
    workflow = SceneEnrichmentWorkflow(
        controller,
        workers,  # type: ignore[arg-type]
        OllamaSettings,
        enricher_factory=lambda _settings: enricher,
    )
    return workflow, controller, workers, enricher, card


def complete_generation(
    workers: FakeWorkers,
    enricher: FakeEnricher,
) -> SceneEnrichmentResult:
    operation = workers.calls[-1]
    assert callable(operation)
    result = operation()
    workers.operations[-1].succeeded.emit(result)
    return result


def test_accept_is_one_undoable_scene_edit() -> None:
    workflow, controller, workers, enricher, card = make_workflow()

    workflow.start(card.id)
    complete_generation(workers, enricher)

    assert enricher.requests[0].scene == "A mysterious wood"
    assert enricher.requests[0].effective_style == "Ink and watercolor"
    assert workflow.draft is not None
    workflow.apply(card.id, "An edited enriched wood")
    assert controller.document.cards[0].scene_description == "An edited enriched wood"
    assert workflow.draft is None
    assert controller.undo()
    assert controller.document.cards[0].scene_description == "A mysterious wood"


def test_discard_and_failure_leave_scene_unchanged() -> None:
    workflow, controller, workers, enricher, card = make_workflow()

    workflow.start(card.id)
    complete_generation(workers, enricher)
    workflow.discard()
    assert controller.document.cards[0].scene_description == "A mysterious wood"

    failures: list[object] = []
    workflow.failed.connect(failures.append)
    workflow.start(card.id)
    workers.operations[-1].failed.emit(RuntimeError("Ollama stopped"))
    assert failures
    assert workflow.draft is None
    assert controller.document.cards[0].scene_description == "A mysterious wood"


def test_stale_result_is_discarded_after_document_change() -> None:
    workflow, controller, workers, enricher, card = make_workflow()
    failures: list[object] = []
    workflow.failed.connect(failures.append)

    workflow.start(card.id)
    controller.replace_document(Stack(name="Replacement", cards=(Card(name="Other"),)))
    operation = workers.calls[-1]
    assert callable(operation)
    workers.operations[-1].succeeded.emit(operation())

    assert workflow.draft is None
    assert failures


def test_cancel_discards_running_and_reviewable_results() -> None:
    workflow, _controller, workers, enricher, card = make_workflow()

    workflow.start(card.id)
    workflow.cancel()
    assert workers.operations[-1].cancelled
    assert not workflow.busy

    workflow.start(card.id)
    complete_generation(workers, enricher)
    assert workflow.draft is not None
    workflow.cancel()
    assert workflow.draft is None
