"""Tests for transient AI hotspot candidate generation and Apply."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from hypergen.application.document_controller import DocumentController
from hypergen.application.hotspot_generation_workflow import (
    HotspotGenerationWorkflow,
)
from hypergen.domain.models import (
    Card,
    HotspotSet,
    ImageOrigin,
    ImageRevision,
    Interaction,
    NavigateAction,
    Point,
    Polygon,
    ResolvedCardReference,
    Stack,
    UnresolvedCardReference,
)
from hypergen.generation.hotspot_prompts import (
    CandidatePolygon,
    ExistingCandidateTarget,
    HotspotGenerationRequest,
    HotspotGenerationResult,
    HotspotProposal,
    HotspotReconciliationWarning,
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
        assert stage == "generating hotspots"
        self.calls.append(operation)
        handle = FakeOperation()
        self.operations.append(handle)
        return handle


class FakeGenerator:
    def __init__(self, result: HotspotGenerationResult) -> None:
        self.result = result
        self.requests: list[HotspotGenerationRequest] = []

    def generate(self, request: HotspotGenerationRequest) -> HotspotGenerationResult:
        self.requests.append(request)
        return self.result


def triangle(offset: float = 0.0) -> Polygon:
    return Polygon(
        points=(
            Point(x=0.1 + offset, y=0.1),
            Point(x=0.3 + offset, y=0.1),
            Point(x=0.2 + offset, y=0.4),
        )
    )


def generated_result() -> HotspotGenerationResult:
    return HotspotGenerationResult(
        proposals=(
            HotspotProposal(
                source_interaction_index=1,
                label="Gate",
                target=ExistingCandidateTarget(
                    card_token="C2",
                    card_name="Garden",
                ),
                polygons=(
                    CandidatePolygon(points=triangle().points),
                ),
            ),
        ),
        warnings=("model coordinates were clamped",),
        reconciliation_warnings=(
            HotspotReconciliationWarning(
                source_interaction_index=2,
                label="Fox",
                reason="no fox is visible",
            ),
        ),
        raw_response='{"interactions":[]}',
        model_identifier="qwen3.5:9b",
        prompt_version="hotspot-prompt-v3",
        schema_version="hotspot-schema-v2",
        duration_seconds=2.0,
    )


def make_workflow(
    tmp_path: Path,
) -> tuple[
    HotspotGenerationWorkflow,
    DocumentController,
    FakeWorkers,
    FakeGenerator,
    Card,
    Card,
]:
    image_path = tmp_path / "background.png"
    image_path.write_bytes(b"image")
    existing = Interaction(
        label="Old hotspot",
        action=NavigateAction(target=UnresolvedCardReference()),
        polygons=(triangle(0.4),),
    )
    revision = ImageRevision(
        image_path="assets/cards/source/image.png",
        origin=ImageOrigin.IMPORTED,
        hotspot_set=HotspotSet(interactions=(existing,)),
        created_at=datetime.now(UTC),
    )
    source = Card(
        name="Source",
        interaction_description="The gate leads to the garden. Tap the fox.",
        image_revisions=(revision,),
        active_revision_id=revision.id,
    )
    destination = Card(name="Garden", scene_description="A moonlit garden")
    controller = DocumentController(Stack(name="Demo", cards=(source, destination)))
    workers = FakeWorkers()
    generator = FakeGenerator(generated_result())
    workflow = HotspotGenerationWorkflow(
        controller,
        workers,  # type: ignore[arg-type]
        OllamaSettings,
        lambda _relative: image_path,
        generator_factory=lambda _settings: generator,
    )
    return workflow, controller, workers, generator, source, destination


def complete_generation(
    workers: FakeWorkers,
) -> HotspotGenerationResult:
    operation = workers.calls[-1]
    assert callable(operation)
    result = operation()
    workers.operations[-1].succeeded.emit(result)
    return result


def test_candidate_is_transient_and_request_uses_local_tokens(
    tmp_path: Path,
) -> None:
    workflow, controller, workers, generator, source, destination = make_workflow(
        tmp_path
    )
    before = controller.document

    workflow.start(source.id)
    complete_generation(workers)

    request = generator.requests[0]
    assert [entry.token for entry in request.card_catalogue] == ["C1", "C2"]
    assert [entry.name for entry in request.card_catalogue] == ["Source", "Garden"]
    assert str(source.id) not in request.model_dump_json()
    assert str(destination.id) not in request.model_dump_json()
    assert controller.document == before
    candidate = workflow.candidate
    assert candidate is not None
    assert candidate.hotspot_set.interactions[0].label == "Gate"
    assert isinstance(
        candidate.hotspot_set.interactions[0].action.target,
        UnresolvedCardReference,
    )
    assert candidate.reconciliation_warnings[0].source_interaction_index == 2
    assert any(
        "Edit Description and regenerate" in warning
        for warning in candidate.warnings
    )


def test_candidate_edits_apply_atomically_with_provenance_and_undo(
    tmp_path: Path,
) -> None:
    workflow, controller, workers, _generator, source, destination = make_workflow(
        tmp_path
    )
    original = controller.document.cards[0].image_revisions[0].hotspot_set
    workflow.start(source.id)
    complete_generation(workers)
    candidate = workflow.candidate
    assert candidate is not None
    interaction_id = candidate.hotspot_set.interactions[0].id

    workflow.rename_interaction(interaction_id, "Garden gate")
    workflow.replace_polygon(interaction_id, 0, triangle(0.05))
    workflow.set_destination(interaction_id, destination.id)
    workflow.apply()

    applied = controller.document.cards[0].image_revisions[0].hotspot_set
    assert applied is not None
    assert applied.interactions[0].label == "Garden gate"
    assert applied.interactions[0].action.target == ResolvedCardReference(
        target_card_id=destination.id
    )
    assert applied.generation_provenance is not None
    assert applied.generation_provenance.model_identifier == "qwen3.5:9b"
    assert workflow.candidate is None
    assert controller.undo()
    assert controller.document.cards[0].image_revisions[0].hotspot_set == original


def test_discard_failure_cancel_and_stale_result_preserve_applied_set(
    tmp_path: Path,
) -> None:
    workflow, controller, workers, _generator, source, _destination = make_workflow(
        tmp_path
    )
    original = controller.document.cards[0].image_revisions[0].hotspot_set

    workflow.start(source.id)
    complete_generation(workers)
    workflow.discard()
    assert controller.document.cards[0].image_revisions[0].hotspot_set == original

    failures: list[object] = []
    workflow.failed.connect(failures.append)
    workflow.start(source.id)
    workers.operations[-1].failed.emit(RuntimeError("Ollama stopped"))
    assert failures
    assert controller.document.cards[0].image_revisions[0].hotspot_set == original

    workflow.start(source.id)
    workflow.cancel()
    assert workers.operations[-1].cancelled
    assert controller.document.cards[0].image_revisions[0].hotspot_set == original

    workflow.start(source.id)
    replacement = Stack(name="Replacement", cards=(Card(name="Other"),))
    controller.replace_document(replacement)
    complete_generation(workers)
    assert workflow.candidate is None
    assert failures
