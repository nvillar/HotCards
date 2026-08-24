"""Tests for directly applied Description enrichment."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from uuid import uuid4

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
from hypergen.application.scene_enrichment_workflow import (
    SceneEnrichmentWorkflow,
    SceneEnrichmentWorkflowError,
)
from hypergen.domain.models import (
    Card,
    CardRevision,
    GeneratedBackground,
    ImageGenerationInputs,
    ImageGenerationMetadata,
    ReferenceRole,
    ResolvedCardReference,
    Stack,
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
    *other_cards: Card,
) -> tuple[
    SceneEnrichmentWorkflow,
    DocumentController,
    FakeWorkers,
    RecordingEnricher,
]:
    controller = DocumentController(
        Stack(name="Stack", cards=(card, *other_cards))
    )
    workers = FakeWorkers()
    enricher = RecordingEnricher()
    workflow = SceneEnrichmentWorkflow(
        controller,
        workers,  # type: ignore[arg-type]
        lambda: OllamaSettings(model="test"),
        enricher_factory=lambda _settings: enricher,
    )
    return workflow, controller, workers, enricher


def _background(description: str) -> GeneratedBackground:
    background_id = uuid4()
    return GeneratedBackground(
        id=background_id,
        image_path=f"assets/cards/source/{background_id}.png",
        generation_metadata=ImageGenerationMetadata(
            inputs=ImageGenerationInputs(description=description),
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


def test_enrichment_applies_immediately_with_one_undo_boundary() -> None:
    card = Card(
        name="Card",
        revisions=(CardRevision(description="A courtyard"),),
    )
    workflow, controller, workers, enricher = _workflow(card)
    applied: list[GeneratedRevisionChange] = []
    workflow.generation_applied.connect(applied.append)

    workflow.start(card.id)
    assert workers.stages == ["enriching Description"]
    work = workers.calls[0]
    assert callable(work)
    workers.operations[0].succeeded.emit(work())

    assert enricher.requests[0].scene == "A courtyard"
    revision = controller.document.cards[0].active_revision
    assert revision.description == "A courtyard"
    assert revision.enriched_description is not None
    assert revision.enriched_description.text == "A richer courtyard"
    assert applied[0].message == "Description enriched"
    assert applied[0].card_id == card.id
    assert applied[0].revision_id == card.active_revision.id
    assert applied[0].previous_revision.enriched_description is None
    assert isinstance(applied[0].token, UndoToken)
    assert controller.undo_if_current(applied[0].token)
    assert controller.document.cards[0].active_revision.enriched_description is None


def test_empty_description_is_rejected_even_with_an_image() -> None:
    card = Card(name="Card")
    workflow, _controller, _workers, _enricher = _workflow(card)

    with pytest.raises(SceneEnrichmentWorkflowError, match="Description"):
        workflow.start(card.id)


def test_stale_or_cancelled_results_do_not_apply() -> None:
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
    assert controller.document.cards[0].active_revision.description == "Changed"
    assert "changed before enrichment completed" in str(failures[-1])

    workflow.start(card.id)
    operation = workers.operations[-1]
    workflow.cancel()
    operation.succeeded.emit(_result("Late"))
    assert controller.document.cards[0].active_revision.description == "Changed"


def test_result_does_not_apply_after_revision_switch() -> None:
    first = CardRevision(description="First")
    second = CardRevision(description="Second")
    card = Card(
        name="Card",
        revisions=(first, second),
        active_revision_id=first.id,
    )
    workflow, controller, workers, _enricher = _workflow(card)
    failures: list[object] = []
    workflow.failed.connect(failures.append)

    workflow.start(card.id)
    controller.execute(
        ActivateRevisionCommand(card_id=card.id, revision_id=second.id)
    )
    workers.operations[0].succeeded.emit(_result("Late"))

    assert controller.document.cards[0].active_revision.description == "Second"
    assert "changed before enrichment completed" in str(failures[-1])


def test_enrichment_uses_grouped_generation_time_reference_descriptions() -> None:
    source = Card(
        name="Castle",
        revisions=(
            CardRevision(
                description="Edited after generation",
                background=_background(
                    "A black basalt castle with copper roofs, viewed from "
                    "the drawbridge"
                ),
            ),
        ),
    )
    reference = ResolvedCardReference(target_card_id=source.id)
    target = Card(
        name="Outer gate",
        revisions=(
            CardRevision(
                description="A guard approaches the same castle",
                subject=reference,
                setting=reference,
            ),
        ),
    )
    workflow, controller, workers, enricher = _workflow(target, source)

    workflow.start(target.id)
    work = workers.calls[0]
    assert callable(work)
    workers.operations[0].succeeded.emit(work())

    context = enricher.requests[0].references[0]
    assert context.roles == (
        ReferenceRole.SUBJECT,
        ReferenceRole.SETTING,
    )
    assert context.source_description.startswith("A black basalt castle")
    assert "Edited after generation" not in context.source_description
    enriched = controller.document.cards[0].active_revision.enriched_description
    assert enriched is not None
    assert enriched.text == "A richer courtyard"


def test_reference_changes_make_enrichment_result_stale() -> None:
    source = Card(
        name="Castle",
        revisions=(
            CardRevision(
                background=_background("A black basalt castle")
            ),
        ),
    )
    target = Card(
        name="Gate",
        revisions=(
            CardRevision(
                description="The outer gate",
                subject=ResolvedCardReference(target_card_id=source.id),
            ),
        ),
    )
    workflow, controller, workers, _enricher = _workflow(target, source)
    failures: list[object] = []
    workflow.failed.connect(failures.append)

    workflow.start(target.id)
    controller.execute(
        SetRevisionReferenceCommand(
            card_id=target.id,
            revision_id=target.active_revision.id,
            role=ReferenceRole.SUBJECT,
            reference=None,
        )
    )
    workers.operations[0].succeeded.emit(_result("Late"))

    assert controller.document.cards[0].active_revision.description == (
        "The outer gate"
    )
    assert "changed before enrichment completed" in str(failures[-1])


def test_editing_current_source_text_does_not_stale_reference_provenance() -> None:
    source = Card(
        name="Castle",
        revisions=(
            CardRevision(
                description="Current source text",
                background=_background("Generation-time castle text"),
            ),
        ),
    )
    target = Card(
        name="Gate",
        revisions=(
            CardRevision(
                description="The outer gate",
                setting=ResolvedCardReference(target_card_id=source.id),
            ),
        ),
    )
    workflow, controller, workers, _enricher = _workflow(target, source)

    workflow.start(target.id)
    controller.execute(
        EditRevisionDescriptionCommand(
            card_id=source.id,
            revision_id=source.active_revision.id,
            value="New source text without regeneration",
        )
    )
    workers.operations[0].succeeded.emit(_result("Enriched gate"))

    enriched = controller.document.cards[0].active_revision.enriched_description
    assert enriched is not None
    assert enriched.text == "Enriched gate"
