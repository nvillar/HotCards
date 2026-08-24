"""Tests for directly applied Description enrichment."""

from __future__ import annotations

import os
from collections.abc import Callable
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
from hypergen.generation.reference_profiles import (
    REFERENCE_PROFILE_PROMPT_VERSION,
    ReferenceProfileRequest,
    ReferenceProfileResult,
    SettingReferenceProfile,
    StyleReferenceProfile,
    SubjectReferenceProfile,
)
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


class RecordingProfiler:
    def __init__(self) -> None:
        self.requests: list[ReferenceProfileRequest] = []

    def extract(
        self,
        request: ReferenceProfileRequest,
    ) -> ReferenceProfileResult:
        self.requests.append(request)
        profiles = {
            ReferenceRole.SUBJECT: (
                SubjectReferenceProfile(identity="black basalt castle"),
                "black basalt castle",
            ),
            ReferenceRole.STYLE: (
                StyleReferenceProfile(medium="dithered graphics"),
                "dithered graphics",
            ),
            ReferenceRole.SETTING: (
                SettingReferenceProfile(environment="mountain castle"),
                "mountain castle",
            ),
        }
        profile, capsule = profiles[request.role]
        return ReferenceProfileResult(
            role=request.role,
            profile=profile,
            capsule=capsule,
            model_identifier="test",
            prompt_version=REFERENCE_PROFILE_PROMPT_VERSION,
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
    *other_cards: Card,
    settings_provider: Callable[[], OllamaSettings] | None = None,
) -> tuple[
    SceneEnrichmentWorkflow,
    DocumentController,
    FakeWorkers,
    RecordingEnricher,
    RecordingProfiler,
]:
    controller = DocumentController(
        Stack(name="Stack", cards=(card, *other_cards))
    )
    workers = FakeWorkers()
    enricher = RecordingEnricher()
    profiler = RecordingProfiler()
    workflow = SceneEnrichmentWorkflow(
        controller,
        workers,  # type: ignore[arg-type]
        settings_provider or (lambda: OllamaSettings(model="test")),
        enricher_factory=lambda _settings: enricher,
        profiler_factory=lambda _settings: profiler,
    )
    return workflow, controller, workers, enricher, profiler


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
    workflow, controller, workers, enricher, _profiler = _workflow(card)
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
    workflow, _controller, _workers, _enricher, _profiler = _workflow(card)

    with pytest.raises(SceneEnrichmentWorkflowError, match="Description"):
        workflow.start(card.id)


def test_stale_or_cancelled_results_do_not_apply() -> None:
    card = Card(
        name="Card",
        revisions=(CardRevision(description="Original"),),
    )
    workflow, controller, workers, _enricher, _profiler = _workflow(card)
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
    workflow, controller, workers, _enricher, _profiler = _workflow(card)
    failures: list[object] = []
    workflow.failed.connect(failures.append)

    workflow.start(card.id)
    controller.execute(
        ActivateRevisionCommand(card_id=card.id, revision_id=second.id)
    )
    workers.operations[0].succeeded.emit(_result("Late"))

    assert controller.document.cards[0].active_revision.description == "Second"
    assert "changed before enrichment completed" in str(failures[-1])


def test_enrichment_sends_only_role_safe_reference_capsules() -> None:
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
    workflow, controller, workers, enricher, profiler = _workflow(
        target,
        source,
    )

    workflow.start(target.id)
    work = workers.calls[0]
    assert callable(work)
    workers.operations[0].succeeded.emit(work())

    request = enricher.requests[0]
    assert request.scene == "A guard approaches the same castle"
    assert [(item.role, item.capsule) for item in request.references] == [
        (ReferenceRole.SUBJECT, "black basalt castle"),
        (ReferenceRole.SETTING, "mountain castle"),
    ]
    assert [item.role for item in profiler.requests] == [
        ReferenceRole.SUBJECT,
        ReferenceRole.SETTING,
    ]
    assert all(
        item.source_description
        == (
            "A black basalt castle with copper roofs, viewed from "
            "the drawbridge"
        )
        for item in profiler.requests
    )
    enriched = controller.document.cards[0].active_revision.enriched_description
    assert enriched is not None
    assert enriched.text == (
        "A richer courtyard "
        "The referenced subject's stable identity and appearance are "
        "black basalt castle. "
        "The stable environment is mountain castle."
    )
    assert [item.role for item in enriched.references] == [
        ReferenceRole.SUBJECT,
        ReferenceRole.SETTING,
    ]
    assert all(
        item.background_id == source.active_revision.background.id
        for item in enriched.references
    )


def test_reference_changes_invalidate_in_flight_enrichment() -> None:
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
    workflow, controller, workers, _enricher, _profiler = _workflow(
        target,
        source,
    )
    failures: list[object] = []
    workflow.failed.connect(failures.append)

    workflow.start(target.id)
    work = workers.calls[0]
    assert callable(work)
    controller.execute(
        SetRevisionReferenceCommand(
            card_id=target.id,
            revision_id=target.active_revision.id,
            role=ReferenceRole.SUBJECT,
            reference=None,
        )
    )
    workers.operations[0].succeeded.emit(work())

    enriched = controller.document.cards[0].active_revision.enriched_description
    assert enriched is None
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
    workflow, controller, workers, _enricher, profiler = _workflow(
        target,
        source,
    )

    workflow.start(target.id)
    work = workers.calls[0]
    assert callable(work)
    controller.execute(
        EditRevisionDescriptionCommand(
            card_id=source.id,
            revision_id=source.active_revision.id,
            value="New source text without regeneration",
        )
    )
    workers.operations[0].succeeded.emit(work())

    enriched = controller.document.cards[0].active_revision.enriched_description
    assert enriched is not None
    assert enriched.text.endswith("The stable environment is mountain castle.")
    assert profiler.requests[0].source_description == "Generation-time castle text"


def test_reference_profiles_are_cached_per_background_role_and_model() -> None:
    source = Card(
        name="Style source",
        revisions=(
            CardRevision(background=_background("Dithered monochrome pixels")),
        ),
    )
    target = Card(
        name="Computer",
        revisions=(
            CardRevision(
                description="A computer",
                style=ResolvedCardReference(target_card_id=source.id),
            ),
        ),
    )
    selected_model = ["test"]
    workflow, _controller, workers, _enricher, profiler = _workflow(
        target,
        source,
        settings_provider=lambda: OllamaSettings(model=selected_model[0]),
    )

    workflow.start(target.id)
    first = workers.calls[0]
    assert callable(first)
    workers.operations[0].succeeded.emit(first())
    workflow.start(target.id)
    second = workers.calls[1]
    assert callable(second)
    workers.operations[1].succeeded.emit(second())
    selected_model[0] = "other-model"
    workflow.start(target.id)
    third = workers.calls[2]
    assert callable(third)
    workers.operations[2].succeeded.emit(third())

    assert len(profiler.requests) == 2
    assert all(
        request.role is ReferenceRole.STYLE
        for request in profiler.requests
    )
