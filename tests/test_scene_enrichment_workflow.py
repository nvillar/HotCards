"""Tests for transient, identity-bound Description enrichment."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from PySide6.QtCore import QObject, Signal

from hypergen.application.commands import EditCardTextCommand
from hypergen.application.document_controller import DocumentController
from hypergen.application.scene_enrichment_workflow import (
    SceneEnrichmentWorkflow,
    SceneEnrichmentWorkflowError,
)
from hypergen.domain.models import (
    Card,
    ImageGenerationInputs,
    ImageGenerationMetadata,
    ImageOrigin,
    ImageRevision,
    Stack,
)
from hypergen.generation.image_description import (
    ImageDescriptionRequest,
    ImageDescriptionResult,
)
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
        self.stages: list[str] = []
        self.operations: list[FakeOperation] = []

    def run_ollama(
        self,
        operation: object,
        *,
        stage: str,
    ) -> FakeOperation:
        self.calls.append(operation)
        self.stages.append(stage)
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
            model_identifier="qwen3.5:9b-mlx",
            prompt_version=request.prompt_version,
            duration_seconds=1.0,
        )


class FakeDescriber:
    def __init__(self) -> None:
        self.requests: list[ImageDescriptionRequest] = []

    def describe(self, request: ImageDescriptionRequest) -> ImageDescriptionResult:
        self.requests.append(request)
        return ImageDescriptionResult(
            scene="A low-angle view of silver birches beneath a violet moon",
            raw_response='{"scene":"A low-angle view of silver birches"}',
            model_identifier="qwen3.5:9b-mlx",
            prompt_version=request.prompt_version,
            duration_seconds=1.0,
        )


def generated_metadata() -> ImageGenerationMetadata:
    generated_at = datetime.now(UTC)
    return ImageGenerationMetadata(
        inputs=ImageGenerationInputs(
            scene_description="A mysterious wood",
            global_style="Ink and watercolor",
        ),
        render_prompt="A mysterious wood. Ink and watercolor.",
        model_identifier="mflux",
        mflux_version="0.10.0",
        seed=42,
        width=1024,
        height=768,
        step_count=4,
        generated_at=generated_at,
        duration_seconds=1.0,
    )


def make_workflow(
    tmp_path: Path,
    *,
    scene: str = "A mysterious wood",
    image_origin: ImageOrigin | None = None,
) -> tuple[
    SceneEnrichmentWorkflow,
    DocumentController,
    FakeWorkers,
    FakeEnricher,
    FakeDescriber,
    Card,
    Path | None,
]:
    image_path: Path | None = None
    revisions: tuple[ImageRevision, ...] = ()
    active_revision_id = None
    if image_origin is not None:
        image_path = tmp_path / "active.png"
        image_path.write_bytes(b"image")
        metadata = (
            generated_metadata() if image_origin is ImageOrigin.GENERATED else None
        )
        revision = ImageRevision(
            image_path="assets/cards/card/image.png",
            origin=image_origin,
            source_filename=(
                "imported.png" if image_origin is ImageOrigin.IMPORTED else None
            ),
            generation_metadata=metadata,
            created_at=(
                metadata.generated_at if metadata is not None else datetime.now(UTC)
            ),
        )
        revisions = (revision,)
        active_revision_id = revision.id
    card = Card(
        name="Wood",
        scene_description=scene,
        interaction_description="The gate navigates elsewhere",
        image_revisions=revisions,
        active_revision_id=active_revision_id,
    )
    controller = DocumentController(
        Stack(
            name="Demo",
            global_style="Ink and watercolor",
            cards=(card,),
        )
    )
    workers = FakeWorkers()
    enricher = FakeEnricher()
    describer = FakeDescriber()
    workflow = SceneEnrichmentWorkflow(
        controller,
        workers,  # type: ignore[arg-type]
        OllamaSettings,
        lambda _relative: image_path,
        enricher_factory=lambda _settings: enricher,
        describer_factory=lambda _settings: describer,
    )
    return workflow, controller, workers, enricher, describer, card, image_path


def complete_latest(workers: FakeWorkers) -> object:
    operation = workers.calls[-1]
    assert callable(operation)
    result = operation()
    workers.operations[-1].succeeded.emit(result)
    return result


def test_text_only_accept_is_one_undoable_description_edit(tmp_path: Path) -> None:
    workflow, controller, workers, enricher, describer, card, _path = make_workflow(
        tmp_path
    )

    workflow.start(card.id)
    complete_latest(workers)

    assert workers.stages == ["enriching Description"]
    assert describer.requests == []
    assert enricher.requests[0].scene == "A mysterious wood"
    assert enricher.requests[0].effective_style == "Ink and watercolor"
    assert enricher.requests[0].image_description is None
    assert workflow.draft is not None
    workflow.apply(card.id, "An edited enriched wood")
    assert controller.document.cards[0].scene_description == "An edited enriched wood"
    assert controller.document.cards[0].interaction_description == (
        "The gate navigates elsewhere"
    )
    assert workflow.draft is None
    assert controller.undo()
    assert controller.document.cards[0].scene_description == "A mysterious wood"


@pytest.mark.parametrize("origin", [ImageOrigin.IMPORTED, ImageOrigin.GENERATED])
def test_active_image_runs_hidden_description_then_enrichment(
    tmp_path: Path,
    origin: ImageOrigin,
) -> None:
    workflow, _controller, workers, enricher, describer, card, image_path = (
        make_workflow(tmp_path, image_origin=origin)
    )
    progress: list[str] = []
    workflow.progress_changed.connect(progress.append)

    workflow.start(card.id)
    assert workers.stages == ["describing background"]
    complete_latest(workers)
    assert workers.stages == ["describing background", "enriching Description"]
    complete_latest(workers)

    assert describer.requests[0].image_path == image_path
    assert enricher.requests[0].scene == "A mysterious wood"
    assert enricher.requests[0].image_description == (
        "A low-angle view of silver birches beneath a violet moon"
    )
    assert progress[:2] == ["Describing background...", "Enriching Description..."]
    assert workflow.draft is not None
    assert workflow.draft.revision_id == card.active_revision_id


def test_image_allows_empty_authored_description(tmp_path: Path) -> None:
    workflow, _controller, workers, enricher, _describer, card, _path = make_workflow(
        tmp_path,
        scene="",
        image_origin=ImageOrigin.IMPORTED,
    )

    workflow.start(card.id)
    complete_latest(workers)
    complete_latest(workers)

    assert enricher.requests[0].scene == ""
    assert enricher.requests[0].image_description
    assert workflow.draft is not None


def test_empty_description_without_readable_image_is_rejected(tmp_path: Path) -> None:
    workflow, _controller, _workers, _enricher, _describer, card, _path = (
        make_workflow(tmp_path, scene="")
    )

    with pytest.raises(SceneEnrichmentWorkflowError, match="readable background"):
        workflow.start(card.id)


def test_discard_and_each_stage_failure_leave_description_unchanged(
    tmp_path: Path,
) -> None:
    workflow, controller, workers, enricher, _describer, card, _path = make_workflow(
        tmp_path,
        image_origin=ImageOrigin.IMPORTED,
    )
    failures: list[object] = []
    progress: list[str] = []
    workflow.failed.connect(failures.append)
    workflow.progress_changed.connect(progress.append)

    workflow.start(card.id)
    workers.operations[-1].failed.emit(RuntimeError("vision stopped"))
    assert progress[-1] == "Background description failed"
    assert not workflow.busy

    workflow.start(card.id)
    complete_latest(workers)
    workers.operations[-1].failed.emit(RuntimeError("enrichment stopped"))
    assert progress[-1] == "Description enrichment failed"
    assert not workflow.busy
    assert len(failures) == 2
    assert controller.document.cards[0].scene_description == "A mysterious wood"

    workflow.start(card.id)
    complete_latest(workers)
    complete_latest(workers)
    workflow.discard()
    assert controller.document.cards[0].scene_description == "A mysterious wood"
    assert enricher.requests


def test_stale_image_stage_result_is_suppressed_after_revision_change(
    tmp_path: Path,
) -> None:
    workflow, controller, workers, _enricher, _describer, card, image_path = (
        make_workflow(tmp_path, image_origin=ImageOrigin.IMPORTED)
    )
    failures: list[object] = []
    workflow.failed.connect(failures.append)
    replacement = ImageRevision(
        image_path="assets/cards/card/replacement.png",
        origin=ImageOrigin.IMPORTED,
        source_filename="replacement.png",
        created_at=datetime.now(UTC),
    )
    workflow.start(card.id)
    original = controller.document.cards[0]
    controller.replace_document(
        controller.document.model_copy(
            update={
                "cards": (
                    original.model_copy(
                        update={
                            "image_revisions": (
                                *original.image_revisions,
                                replacement,
                            ),
                            "active_revision_id": replacement.id,
                        }
                    ),
                )
            }
        )
    )
    complete_latest(workers)

    assert image_path is not None
    assert workflow.draft is None
    assert len(workers.calls) == 1
    assert failures


def test_stale_enrichment_and_apply_are_suppressed(tmp_path: Path) -> None:
    workflow, controller, workers, _enricher, _describer, card, _path = make_workflow(
        tmp_path,
        image_origin=ImageOrigin.IMPORTED,
    )
    failures: list[object] = []
    workflow.failed.connect(failures.append)

    workflow.start(card.id)
    complete_latest(workers)
    controller.execute(
        EditCardTextCommand(
            card_id=card.id,
            field="scene_description",
            value="Changed while running",
        )
    )
    complete_latest(workers)
    assert workflow.draft is None
    assert failures

    controller.undo()
    workflow.start(card.id)
    complete_latest(workers)
    complete_latest(workers)
    controller.execute(
        EditCardTextCommand(
            card_id=card.id,
            field="scene_description",
            value="Changed before Apply",
        )
    )
    with pytest.raises(SceneEnrichmentWorkflowError, match="background changed"):
        workflow.apply(card.id, "Should not apply")
    assert workflow.draft is None
    assert controller.document.cards[0].scene_description == "Changed before Apply"


def test_text_only_draft_is_stale_when_active_revision_changes(
    tmp_path: Path,
) -> None:
    workflow, controller, workers, _enricher, _describer, card, _path = make_workflow(
        tmp_path
    )
    workflow.start(card.id)
    complete_latest(workers)
    assert workflow.draft is not None

    revision = ImageRevision(
        image_path="assets/cards/card/new.png",
        origin=ImageOrigin.IMPORTED,
        source_filename="new.png",
        created_at=datetime.now(UTC),
    )
    current = controller.document.cards[0]
    controller.replace_document(
        controller.document.model_copy(
            update={
                "cards": (
                    current.model_copy(
                        update={
                            "image_revisions": (revision,),
                            "active_revision_id": revision.id,
                        }
                    ),
                )
            }
        )
    )

    with pytest.raises(SceneEnrichmentWorkflowError, match="background changed"):
        workflow.apply(card.id, "Should not apply")
    assert workflow.draft is None
    assert controller.document.cards[0].scene_description == "A mysterious wood"


def test_cancel_suppresses_both_stages_and_review_draft(tmp_path: Path) -> None:
    workflow, _controller, workers, _enricher, _describer, card, _path = make_workflow(
        tmp_path,
        image_origin=ImageOrigin.IMPORTED,
    )

    workflow.start(card.id)
    workflow.cancel()
    assert workers.operations[-1].cancelled
    assert not workflow.busy

    workflow.start(card.id)
    complete_latest(workers)
    workflow.cancel()
    assert workers.operations[-1].cancelled
    assert not workflow.busy

    workflow.start(card.id)
    complete_latest(workers)
    complete_latest(workers)
    assert workflow.draft is not None
    workflow.cancel()
    assert workflow.draft is None
