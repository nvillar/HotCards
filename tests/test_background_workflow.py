"""Tests for direct revision-local background changes."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace
from uuid import uuid4

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PIL import Image
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import QApplication

import hotcards.storage.stack_store as stack_store_module
from hotcards.application.background_workflow import (
    BackgroundGenerationSettings,
    BackgroundWorkflow,
    BackgroundWorkflowError,
)
from hotcards.application.commands import (
    CommandError,
    CreateCardCommand,
    DeleteRevisionCommand,
    DuplicateRevisionCommand,
    EditRevisionDescriptionCommand,
    RenameCardCommand,
    ReplaceHotspotSetCommand,
    ReplaceRevisionBackgroundCommand,
    SetRevisionGenerateResolutionCommand,
    SetRevisionReferenceCommand,
    SetRevisionStyleCommand,
    UpdateStyleCommand,
)
from hotcards.application.document_controller import DocumentController, UndoToken
from hotcards.application.document_session import DocumentSession
from hotcards.application.generated_revision_change import GeneratedRevisionChange
from hotcards.domain.image_dimensions import (
    AspectRatio,
    GenerateResolution,
    output_dimensions,
)
from hotcards.domain.models import (
    Card,
    CardRevision,
    DuplicateProvenance,
    GeneratedBackground,
    HotspotSet,
    ImageReferenceSnapshot,
    ImageSourceSnapshot,
    Interaction,
    NavigateAction,
    RefineProvenance,
    RefineTransformation,
    ResolvedCardReference,
    Stack,
    UnresolvedCardReference,
)
from hotcards.generation.errors import ImageGenerationCancelled
from hotcards.generation.mflux_generator import MfluxGenerator
from hotcards.storage.stack_store import (
    StackStore,
    StackStoreError,
    StackStoreTransactionError,
)
from hotcards.ui.inspector import Inspector


@pytest.fixture(scope="module")
def application() -> QApplication:
    return QApplication.instance() or QApplication([])


class FakeOperation(QObject):
    succeeded = Signal(object)
    failed = Signal(object)
    cancelled = Signal()

    def __init__(self, request_cancel: object = None) -> None:
        super().__init__()
        self.was_cancelled = False
        self.finished_state = False
        self.request_cancel = request_cancel

    @property
    def is_finished(self) -> bool:
        return self.finished_state

    def cancel(self) -> None:
        self.was_cancelled = True
        self.finished_state = True
        if callable(self.request_cancel):
            self.request_cancel()
        self.cancelled.emit()


class FakeWorkers:
    def __init__(self) -> None:
        self.calls: list[object] = []
        self.operations: list[FakeOperation] = []
        self.disposers: list[object] = []

    def run_mflux(
        self,
        operation: object,
        *,
        stage: str,
        request_cancel: object = None,
        dispose_result: object = None,
        invocation_started: object = None,
        invocation_finished: object = None,
    ) -> FakeOperation:
        assert stage in {
            "generating background image",
            "refining background image",
        }

        def wrapped_operation() -> object:
            if callable(invocation_started):
                invocation_started()
            try:
                assert callable(operation)
                return operation()
            finally:
                if callable(invocation_finished):
                    invocation_finished()

        self.calls.append(wrapped_operation)
        self.disposers.append(dispose_result)
        handle = FakeOperation(request_cancel)
        self.operations.append(handle)
        return handle


class FakeMfluxImage:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height

    def save(self, path: Path, *, overwrite: bool) -> None:
        assert not overwrite
        Image.new("RGB", (self.width, self.height), "navy").save(path)


class FakeCallbackRegistry:
    def __init__(self) -> None:
        self.registered: list[object] = []

    def register(self, callback: object) -> None:
        self.registered.append(callback)


class FakeMfluxModel:
    def __init__(self) -> None:
        self.callbacks = FakeCallbackRegistry()
        self.calls: list[dict[str, object]] = []
        self.consumed_source_pixels: list[object] = []

    def generate_image(self, **kwargs: object) -> FakeMfluxImage:
        self.calls.append(kwargs)
        image_path = kwargs.get("image_path")
        if isinstance(image_path, Path):
            with Image.open(image_path) as image:
                self.consumed_source_pixels.append(image.getpixel((0, 0)))
        config = SimpleNamespace(num_inference_steps=kwargs["num_inference_steps"])
        for callback in self.callbacks.registered:
            callback.call_before_loop(config=config)
        for _step in range(kwargs["num_inference_steps"]):  # type: ignore[arg-type]
            for callback in self.callbacks.registered:
                callback.call_in_loop()
        for callback in self.callbacks.registered:
            callback.call_after_loop()
        return FakeMfluxImage(
            width=kwargs["width"],  # type: ignore[arg-type]
            height=kwargs["height"],  # type: ignore[arg-type]
        )


def _settings() -> BackgroundGenerationSettings:
    return BackgroundGenerationSettings(
        mflux_model="flux2-klein-4b",
        step_count=4,
        quantization=None,
        random_seed=False,
        fixed_seed=42,
    )


def _bound_workflow(
    root: Path,
    *,
    aspect_ratio: AspectRatio = AspectRatio.LANDSCAPE,
    resolution: GenerateResolution = GenerateResolution.RESOLUTION_512,
) -> tuple[
    BackgroundWorkflow,
    DocumentController,
    DocumentSession,
    FakeWorkers,
    FakeMfluxModel,
    Card,
]:
    hotspot = Interaction(
        label="Gate",
        action=NavigateAction(target=UnresolvedCardReference()),
    )
    revision = CardRevision(
        description="A garden",
        hotspot_set=HotspotSet(interactions=(hotspot,)),
        generate_resolution=resolution,
    )
    card = Card(name="Garden", revisions=(revision,), active_revision_id=revision.id)
    controller = DocumentController(
        Stack(
            name="Stack",
            aspect_ratio=aspect_ratio,
            cards=(card,),
            start_card_id=card.id,
        )
    )
    session = DocumentSession(controller)
    session.create(controller.document, root / "Stack.hotcards")
    workers = FakeWorkers()
    model = FakeMfluxModel()
    workflow = BackgroundWorkflow(
        controller,
        session,
        workers,  # type: ignore[arg-type]
        _settings,
        mflux_generator=MfluxGenerator(
            model_factory=lambda *_args: model,
            edit_model_factory=lambda *_args: model,
        ),
        temporary_directory=root / "temporary",
    )
    return workflow, controller, session, workers, model, card


def _complete_generation(workers: FakeWorkers) -> None:
    operation = workers.operations[-1]
    work = workers.calls[-1]
    assert callable(work)
    assert callable(workers.disposers[-1])
    operation.succeeded.emit(work())


def _create_generated_source(
    workflow: BackgroundWorkflow,
    controller: DocumentController,
    workers: FakeWorkers,
    *,
    name: str,
) -> Card:
    source_id = uuid4()
    controller.execute(CreateCardCommand(name=name, card_id=source_id))
    source = next(card for card in controller.document.cards if card.id == source_id)
    controller.execute(
        EditRevisionDescriptionCommand(
            card_id=source.id,
            revision_id=source.active_revision.id,
            value=f"{name} source image",
        )
    )
    workflow.generate(source.id)
    _complete_generation(workers)
    return next(card for card in controller.document.cards if card.id == source_id)


def test_generate_uses_description_and_preserves_result_lifecycle(
    tmp_path: Path,
) -> None:
    workflow, controller, session, workers, model, card = _bound_workflow(tmp_path)
    applied: list[GeneratedRevisionChange] = []
    progress: list[tuple[int, int]] = []
    workflow.generation_applied.connect(applied.append)
    workflow.generation_progress_changed.connect(
        lambda completed, total: progress.append((completed, total))
    )

    workflow.generate(card.id)
    _complete_generation(workers)

    assert progress == [(0, 4), (1, 4), (2, 4), (3, 4), (4, 4), (0, 0)]
    revision = controller.document.cards[0].active_revision
    assert revision.background is not None
    assert revision.description == "A garden"
    assert revision.hotspot_set is not None
    provenance = revision.provenance
    assert provenance is not None
    assert provenance.operation == "generate"
    assert provenance.inputs.description == "A garden"
    assert provenance.render_prompt == "A garden"
    assert (provenance.settings.width, provenance.settings.height) == (592, 448)
    assert model.calls[-1]["prompt"] == "A garden"
    assert session.flush()
    assert StackStore(session.state.bundle_path).load() == controller.document
    generated_path = session.store.asset_path(revision.background.image_path)
    assert applied[0].message == "Image generated"
    assert applied[0].previous_revision.background is None
    assert isinstance(applied[0].token, UndoToken)

    assert controller.undo_if_current(applied[0].token)
    assert session.flush()
    assert controller.document.cards[0].active_revision.background is None
    assert generated_path.is_file()
    assert controller.redo()
    assert session.flush()
    assert generated_path.is_file()
    assert controller.undo()
    assert session.flush()
    controller.execute(RenameCardCommand(card_id=card.id, name="Garden renamed"))
    assert session.flush()
    assert not generated_path.exists()


def test_refine_uses_current_image_seed_and_creates_complete_version(
    tmp_path: Path,
) -> None:
    workflow, controller, session, workers, model, card = _bound_workflow(tmp_path)
    workflow.generate(card.id)
    _complete_generation(workers)
    source_card = controller.document.cards[0]
    source_revision = source_card.active_revision
    source_background = source_revision.background
    assert source_background is not None
    source_path = session.store.asset_path(source_background.image_path)
    style = controller.document.styles[0]
    controller.execute(
        SetRevisionStyleCommand(
            card_id=card.id,
            revision_id=source_revision.id,
            style_id=style.id,
        )
    )
    reference = _create_generated_source(
        workflow,
        controller,
        workers,
        name="Reference",
    )
    controller.execute(
        SetRevisionReferenceCommand(
            card_id=card.id,
            revision_id=source_revision.id,
            reference=ResolvedCardReference(target_card_id=reference.id),
        )
    )
    source_revision = controller.document.cards[0].active_revision
    applied: list[tuple[str, UndoToken]] = []
    workflow.change_applied.connect(lambda message, token: applied.append((message, token)))

    workflow.refine(
        card.id,
        transformation=RefineTransformation.BALANCED,
        resolution=GenerateResolution.RESOLUTION_768,
    )
    _complete_generation(workers)

    changed_card = controller.document.cards[0]
    refined = changed_card.active_revision
    assert len(changed_card.revisions) == 2
    assert changed_card.revisions[0] == source_revision
    assert refined.id != source_revision.id
    assert refined.description == source_revision.description
    assert refined.style_id == source_revision.style_id
    assert refined.references == source_revision.references
    assert refined.hotspot_set == source_revision.hotspot_set
    assert refined.generate_resolution == source_revision.generate_resolution
    assert refined.background is not None
    provenance = refined.background.provenance
    assert isinstance(provenance, RefineProvenance)
    assert provenance.source == ImageSourceSnapshot(
        card_id=card.id,
        revision_id=source_revision.id,
        background_id=source_background.id,
    )
    assert provenance.description == source_revision.description
    assert provenance.style is not None
    assert provenance.style.style_id == style.id
    assert provenance.transformation is RefineTransformation.BALANCED
    assert provenance.strength == 0.50
    assert provenance.settings.seed == source_background.provenance.settings.seed
    assert model.calls[-1]["seed"] == source_background.provenance.settings.seed
    refine_source_path = model.calls[-1]["image_path"]
    assert isinstance(refine_source_path, Path)
    assert refine_source_path.parent == tmp_path / "temporary"
    assert refine_source_path != source_path
    assert not refine_source_path.exists()
    assert model.calls[-1]["image_strength"] == 0.50
    assert "image_paths" not in model.calls[-1]
    assert (model.calls[-1]["width"], model.calls[-1]["height"]) == (880, 672)
    assert applied and applied[-1][0] == "Image refined"
    token = applied[-1][1]
    refined_path = session.store.asset_path(refined.background.image_path)
    assert refined_path.is_file()

    assert controller.undo_if_current(token)
    assert session.flush()
    assert controller.document.cards[0].active_revision == source_revision
    assert refined_path.is_file()
    assert controller.redo()
    assert session.flush()
    assert controller.document.cards[0].active_revision == refined
    assert refined_path.is_file()
    with pytest.raises(CommandError, match="cannot delete this source revision"):
        controller.execute(
            DeleteRevisionCommand(
                card_id=card.id,
                revision_id=source_revision.id,
            )
        )

    assert controller.undo()
    assert session.flush()
    controller.execute(RenameCardCommand(card_id=card.id, name="Garden renamed"))
    assert session.flush()
    assert not refined_path.exists()


@pytest.mark.parametrize("hotspot_set", (None, HotspotSet()))
def test_refine_preserves_none_vs_empty_hotspot_set(
    tmp_path: Path,
    hotspot_set: HotspotSet | None,
) -> None:
    workflow, controller, _session, workers, _model, card = _bound_workflow(tmp_path)
    controller.execute(
        ReplaceHotspotSetCommand(
            card_id=card.id,
            revision_id=card.active_revision.id,
            hotspot_set=hotspot_set,
        )
    )
    workflow.generate(card.id)
    _complete_generation(workers)

    workflow.refine(
        card.id,
        transformation=RefineTransformation.PRESERVE,
        resolution=GenerateResolution.RESOLUTION_768,
    )
    _complete_generation(workers)

    revisions = controller.document.cards[0].revisions
    assert revisions[0].hotspot_set == hotspot_set
    assert revisions[1].hotspot_set == hotspot_set


def test_refine_filters_legacy_pixel_size_and_disables_at_maximum(
    tmp_path: Path,
) -> None:
    workflow, controller, session, workers, _model, card = _bound_workflow(
        tmp_path,
        resolution=GenerateResolution.RESOLUTION_1024,
    )
    workflow.generate(card.id)
    _complete_generation(workers)
    revision = controller.document.cards[0].active_revision
    assert revision.background is not None

    assert workflow.available_refine_resolutions(card.id) == ()
    with pytest.raises(BackgroundWorkflowError, match="maximum"):
        workflow.refine(
            card.id,
            transformation=RefineTransformation.BALANCED,
            resolution=GenerateResolution.RESOLUTION_1024,
        )

    source_path = session.store.asset_path(revision.background.image_path)
    Image.new("RGB", (1024, 768), "navy").save(source_path)
    assert workflow.available_refine_resolutions(card.id) == (GenerateResolution.RESOLUTION_1024,)

    workflow.refine(
        card.id,
        transformation=RefineTransformation.REIMAGINE,
        resolution=GenerateResolution.RESOLUTION_1024,
    )
    _complete_generation(workers)
    provenance = controller.document.cards[0].active_revision.provenance
    assert isinstance(provenance, RefineProvenance)
    assert provenance.transformation is RefineTransformation.REIMAGINE
    assert provenance.strength == 0.25


def test_refine_flattens_duplicate_source_settings(tmp_path: Path) -> None:
    workflow, controller, _session, workers, model, card = _bound_workflow(tmp_path)
    workflow.generate(card.id)
    _complete_generation(workers)
    revision = controller.document.cards[0].active_revision
    background = revision.background
    assert background is not None
    duplicate_background = background.model_copy(
        update={
            "provenance": DuplicateProvenance(
                source=ImageSourceSnapshot(
                    card_id=uuid4(),
                    revision_id=uuid4(),
                    background_id=uuid4(),
                ),
                original_provenance=background.provenance,
            )
        }
    )
    controller.execute(
        ReplaceRevisionBackgroundCommand(
            card_id=card.id,
            revision_id=revision.id,
            background=duplicate_background,
        )
    )

    workflow.refine(
        card.id,
        transformation=RefineTransformation.PRESERVE,
        resolution=GenerateResolution.RESOLUTION_768,
    )
    _complete_generation(workers)

    refined = controller.document.cards[0].active_revision
    provenance = refined.provenance
    assert isinstance(provenance, RefineProvenance)
    assert provenance.source.card_id == card.id
    assert provenance.source.revision_id == revision.id
    assert provenance.source.background_id == duplicate_background.id
    assert provenance.settings.seed == background.provenance.settings.seed
    assert model.calls[-1]["seed"] == background.provenance.settings.seed


def test_refine_stale_revision_or_model_change_creates_no_version(
    tmp_path: Path,
) -> None:
    workflow, controller, _session, workers, _model, card = _bound_workflow(tmp_path)
    workflow.generate(card.id)
    _complete_generation(workers)
    source = controller.document.cards[0].active_revision
    failures: list[object] = []
    workflow.failed.connect(failures.append)

    workflow.refine(
        card.id,
        transformation=RefineTransformation.BALANCED,
        resolution=GenerateResolution.RESOLUTION_768,
    )
    controller.execute(
        ReplaceHotspotSetCommand(
            card_id=card.id,
            revision_id=source.id,
            hotspot_set=HotspotSet(),
        )
    )
    _complete_generation(workers)

    assert len(controller.document.cards[0].revisions) == 1
    assert "source revision changed" in str(failures[-1])

    current = controller.document.cards[0].active_revision
    workflow.refine(
        card.id,
        transformation=RefineTransformation.BALANCED,
        resolution=GenerateResolution.RESOLUTION_768,
    )
    workflow._settings_provider = lambda: BackgroundGenerationSettings(
        mflux_model="flux2-klein-9b",
        step_count=4,
        quantization=8,
        random_seed=False,
        fixed_seed=42,
    )
    _complete_generation(workers)

    assert controller.document.cards[0].active_revision == current
    assert len(controller.document.cards[0].revisions) == 1
    assert "source revision changed" in str(failures[-1])


def test_refine_uses_immutable_snapshot_and_rejects_replaced_source(
    tmp_path: Path,
) -> None:
    workflow, controller, session, workers, model, card = _bound_workflow(tmp_path)
    workflow.generate(card.id)
    _complete_generation(workers)
    source_revision = controller.document.cards[0].active_revision
    assert source_revision.background is not None
    source_path = session.store.asset_path(source_revision.background.image_path)
    failures: list[object] = []
    workflow.failed.connect(failures.append)

    workflow.refine(
        card.id,
        transformation=RefineTransformation.BALANCED,
        resolution=GenerateResolution.RESOLUTION_768,
    )
    snapshot_paths = list((tmp_path / "temporary").glob(".refine-source-*.png"))
    assert len(snapshot_paths) == 1
    snapshot_path = snapshot_paths[0]
    assert snapshot_path != source_path
    Image.new("RGB", (592, 448), "gold").save(source_path, format="PNG")
    _complete_generation(workers)

    assert model.consumed_source_pixels[-1] == (0, 0, 128)
    assert model.calls[-1]["image_path"] == snapshot_path
    assert not snapshot_path.exists()
    assert controller.document.cards[0].revisions == (source_revision,)
    assert "changed while Refine was running" in str(failures[-1])


def test_refine_rejects_symlink_source_without_starting_model(
    tmp_path: Path,
) -> None:
    workflow, controller, session, workers, _model, card = _bound_workflow(tmp_path)
    workflow.generate(card.id)
    _complete_generation(workers)
    revision = controller.document.cards[0].active_revision
    assert revision.background is not None
    source_path = session.store.asset_path(revision.background.image_path)
    outside = tmp_path / "outside.png"
    Image.new("RGB", (592, 448), "gold").save(outside, format="PNG")
    source_path.unlink()
    source_path.symlink_to(outside)
    call_count = len(workers.calls)

    with pytest.raises(BackgroundWorkflowError, match="unavailable or unreadable"):
        workflow.available_refine_resolutions(card.id)
    with pytest.raises(BackgroundWorkflowError, match="outside the stack bundle"):
        workflow.refine(
            card.id,
            transformation=RefineTransformation.BALANCED,
            resolution=GenerateResolution.RESOLUTION_768,
        )

    assert len(workers.calls) == call_count
    assert outside.is_file()
    assert not list((tmp_path / "temporary").glob(".refine-source-*.png"))


def test_refine_snapshot_cleanup_waits_for_native_invocation(
    tmp_path: Path,
) -> None:
    workflow, _controller, _session, _workers, _model, card = _bound_workflow(tmp_path)
    workflow.generate(card.id)
    _complete_generation(_workers)
    workflow.refine(
        card.id,
        transformation=RefineTransformation.BALANCED,
        resolution=GenerateResolution.RESOLUTION_768,
    )
    snapshot_path = next((tmp_path / "temporary").glob(".refine-source-*.png"))

    workflow._invocation_started()
    workflow.cancel()

    assert snapshot_path.is_file()
    assert workflow.invocation_active
    with pytest.raises(BackgroundWorkflowError, match="already running"):
        workflow.refine(
            card.id,
            transformation=RefineTransformation.PRESERVE,
            resolution=GenerateResolution.RESOLUTION_768,
        )
    assert list((tmp_path / "temporary").glob(".refine-source-*.png")) == [
        snapshot_path
    ]
    assert workflow._request_id is None
    workflow._invocation_finished()
    assert not workflow.invocation_active
    assert not snapshot_path.exists()


def test_refine_cleanup_never_exposes_idle_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow, _controller, _session, workers, _model, card = _bound_workflow(
        tmp_path
    )
    workflow.generate(card.id)
    _complete_generation(workers)
    workflow.refine(
        card.id,
        transformation=RefineTransformation.BALANCED,
        resolution=GenerateResolution.RESOLUTION_768,
    )
    snapshot = workflow._source_snapshot
    assert snapshot is not None
    cleanup_started = Event()
    allow_cleanup = Event()
    real_dispose = type(snapshot).dispose

    def blocking_dispose(source_snapshot: object) -> bool:
        cleanup_started.set()
        assert allow_cleanup.wait(2)
        return real_dispose(source_snapshot)  # type: ignore[arg-type]

    monkeypatch.setattr(type(snapshot), "dispose", blocking_dispose)
    workflow._invocation_started()
    workflow.cancel()
    cleanup_thread = Thread(target=workflow._invocation_finished)
    cleanup_thread.start()
    assert cleanup_started.wait(2)

    assert workflow.invocation_active
    with pytest.raises(BackgroundWorkflowError, match="could not be cleaned up"):
        workflow.refine(
            card.id,
            transformation=RefineTransformation.PRESERVE,
            resolution=GenerateResolution.RESOLUTION_768,
        )

    allow_cleanup.set()
    cleanup_thread.join(timeout=2)
    assert not cleanup_thread.is_alive()
    assert not workflow.invocation_active
    assert not snapshot.snapshot_path.exists()


def test_refine_cleanup_mismatch_preserves_foreign_entry_and_blocks_work(
    tmp_path: Path,
) -> None:
    workflow, _controller, _session, workers, _model, card = _bound_workflow(
        tmp_path
    )
    workflow.generate(card.id)
    _complete_generation(workers)
    workflow.refine(
        card.id,
        transformation=RefineTransformation.BALANCED,
        resolution=GenerateResolution.RESOLUTION_768,
    )
    snapshot = workflow._source_snapshot
    assert snapshot is not None
    owned_backup = snapshot.snapshot_path.with_name("owned-backup.png")
    snapshot.snapshot_path.rename(owned_backup)
    foreign = tmp_path / "foreign"
    foreign.write_bytes(b"foreign")
    snapshot.snapshot_path.symlink_to(foreign)

    workflow.cancel()

    assert workflow.invocation_active
    assert snapshot.snapshot_path.is_symlink()
    assert snapshot.snapshot_path.resolve() == foreign
    with pytest.raises(BackgroundWorkflowError, match="could not be cleaned up"):
        workflow.generate(card.id)
    workflow.close()
    assert snapshot.snapshot_path.is_symlink()
    assert owned_backup.is_file()


def test_refine_indeterminate_observed_after_renders_and_promotes_one_history_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow, controller, session, workers, _model, card = _bound_workflow(tmp_path)
    workflow.generate(card.id)
    _complete_generation(workers)
    source = controller.document
    source_revision = source.cards[0].active_revision
    previous_token = controller.current_undo_token
    changed_documents: list[Stack] = []
    applied: list[tuple[str, UndoToken]] = []
    failures: list[object] = []
    workflow.document_changed.connect(changed_documents.append)
    workflow.change_applied.connect(lambda message, token: applied.append((message, token)))
    workflow.failed.connect(failures.append)
    assert session.store is not None
    real_store = session.store.store_image_asset_and_save

    def fail_indeterminate(
        source_file_path: Path,
        *,
        destination_card_id: object,
        destination_asset_id: object,
        previous_stack: Stack,
        changed_stack: Stack,
        **_kwargs: object,
    ) -> object:
        relative_path = session.store.store_image_asset(
            source_file_path,
            card_id=destination_card_id,  # type: ignore[arg-type]
            asset_id=destination_asset_id,  # type: ignore[arg-type]
        )
        stored = session.store.stored_image_asset(
            relative_path,
            card_id=destination_card_id,  # type: ignore[arg-type]
            asset_id=destination_asset_id,  # type: ignore[arg-type]
        )
        raise StackStoreTransactionError(
            RuntimeError("manifest directory fsync failed"),
            observed_stack=changed_stack,
            durability_indeterminate=True,
            owned_asset=stored,
        )

    monkeypatch.setattr(
        session.store,
        "store_image_asset_and_save",
        fail_indeterminate,
    )
    workflow.refine(
        card.id,
        transformation=RefineTransformation.BALANCED,
        resolution=GenerateResolution.RESOLUTION_768,
    )
    _complete_generation(workers)

    pending = controller.document
    assert len(pending.cards[0].revisions) == 2
    assert pending.cards[0].active_revision != source_revision
    assert changed_documents[-1] == pending
    assert controller.mutation_blocked
    assert session.state.dirty
    assert applied == []
    assert "durability remains indeterminate" in str(failures[-1])

    monkeypatch.setattr(
        session.store,
        "store_image_asset_and_save",
        real_store,
    )
    assert session.flush()
    assert not controller.mutation_blocked
    assert controller.current_undo_token != previous_token
    controller.execute(RenameCardCommand(card_id=card.id, name="Edited later"))
    assert session.flush()
    assert controller.undo()
    assert session.flush()
    assert controller.document == pending
    assert controller.undo()
    assert session.flush()
    assert controller.document == source


def test_refine_storage_failure_creates_no_revision_or_asset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow, controller, session, workers, _model, card = _bound_workflow(tmp_path)
    workflow.generate(card.id)
    _complete_generation(workers)
    source = controller.document
    assert session.store is not None
    assets_before = set((session.store.bundle_path / "assets" / "cards").glob("*/image-*.png"))
    failures: list[object] = []
    workflow.failed.connect(failures.append)

    def fail_store(*_args: object, **_kwargs: object) -> object:
        raise StackStoreError("injected Refine transaction failure")

    monkeypatch.setattr(
        session.store,
        "store_image_asset_and_save",
        fail_store,
    )
    workflow.refine(
        card.id,
        transformation=RefineTransformation.BALANCED,
        resolution=GenerateResolution.RESOLUTION_768,
    )
    _complete_generation(workers)

    assert controller.document == source
    assert (
        set((session.store.bundle_path / "assets" / "cards").glob("*/image-*.png")) == assets_before
    )
    assert "injected Refine transaction failure" in str(failures[-1])
    assert not list((tmp_path / "temporary").glob("refined-*.png"))


def test_refine_cancel_or_model_failure_creates_no_version(
    tmp_path: Path,
) -> None:
    workflow, controller, _session, workers, _model, card = _bound_workflow(tmp_path)
    workflow.generate(card.id)
    _complete_generation(workers)
    source = controller.document

    workflow.refine(
        card.id,
        transformation=RefineTransformation.BALANCED,
        resolution=GenerateResolution.RESOLUTION_768,
    )
    operation = workers.operations[-1]
    work = workers.calls[-1]
    workflow.cancel()

    assert operation.was_cancelled
    assert callable(work)
    with pytest.raises(ImageGenerationCancelled, match="cancelled"):
        work()
    assert controller.document == source
    assert not list((tmp_path / "temporary").glob("refined-*.png"))

    failures: list[object] = []
    workflow.failed.connect(failures.append)
    workflow.refine(
        card.id,
        transformation=RefineTransformation.BALANCED,
        resolution=GenerateResolution.RESOLUTION_768,
    )
    workers.operations[-1].failed.emit(RuntimeError("injected model failure"))

    assert controller.document == source
    assert "injected model failure" in str(failures[-1])
    assert not list((tmp_path / "temporary").glob("refined-*.png"))


@pytest.mark.parametrize(
    "replacement_checkpoint",
    ("manifest-file-fsynced", "manifest-directory-fsynced"),
)
def test_refine_source_replacement_during_commit_rolls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement_checkpoint: str,
) -> None:
    workflow, controller, session, workers, _model, card = _bound_workflow(
        tmp_path
    )
    workflow.generate(card.id)
    _complete_generation(workers)
    source = controller.document
    source_revision = source.cards[0].active_revision
    assert source_revision.background is not None
    source_path = session.store.asset_path(source_revision.background.image_path)
    replacement = tmp_path / f"{replacement_checkpoint}.png"
    Image.new("RGB", (592, 448), "gold").save(replacement, format="PNG")
    replaced = False
    failures: list[object] = []
    workflow.failed.connect(failures.append)

    def replace_source_at_commit(checkpoint: str) -> None:
        nonlocal replaced
        if checkpoint == replacement_checkpoint and not replaced:
            replaced = True
            os.replace(replacement, source_path)

    monkeypatch.setattr(
        stack_store_module,
        "_io_checkpoint",
        replace_source_at_commit,
    )
    assets_before = set(
        (session.store.bundle_path / "assets" / "cards").glob("*/image-*.png")
    )
    workflow.refine(
        card.id,
        transformation=RefineTransformation.BALANCED,
        resolution=GenerateResolution.RESOLUTION_768,
    )
    _complete_generation(workers)

    assert replaced
    assert controller.document == source
    assert session.store.load() == source
    assert set(
        (session.store.bundle_path / "assets" / "cards").glob("*/image-*.png")
    ) == assets_before
    assert "changed while Refine was running" in str(failures[-1])


def test_refine_fifo_replacement_after_manifest_fsync_rolls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow, controller, session, workers, _model, card = _bound_workflow(
        tmp_path
    )
    workflow.generate(card.id)
    _complete_generation(workers)
    source = controller.document
    source_revision = source.cards[0].active_revision
    assert source_revision.background is not None
    source_path = session.store.asset_path(source_revision.background.image_path)
    replaced = False
    failures: list[object] = []
    workflow.failed.connect(failures.append)

    def replace_source_with_fifo(checkpoint: str) -> None:
        nonlocal replaced
        if checkpoint == "manifest-directory-fsynced" and not replaced:
            replaced = True
            source_path.unlink()
            os.mkfifo(source_path)

    monkeypatch.setattr(
        stack_store_module,
        "_io_checkpoint",
        replace_source_with_fifo,
    )
    assets_before = set(
        (session.store.bundle_path / "assets" / "cards").glob("*/image-*.png")
    )
    workflow.refine(
        card.id,
        transformation=RefineTransformation.BALANCED,
        resolution=GenerateResolution.RESOLUTION_768,
    )
    _complete_generation(workers)

    assert replaced
    assert controller.document == source
    assert Stack.model_validate_json(session.store.stack_path.read_text()) == source
    assert set(
        (session.store.bundle_path / "assets" / "cards").glob("*/image-*.png")
    ) == assets_before
    assert "no longer a regular file" in str(failures[-1])


def test_rejected_generation_apply_removes_the_new_unreachable_asset(
    tmp_path: Path,
) -> None:
    workflow, controller, session, workers, _model, _card = _bound_workflow(tmp_path)
    source = _create_generated_source(
        workflow,
        controller,
        workers,
        name="Source",
    )
    source_revision = source.active_revision
    source_background = source_revision.background
    assert source_background is not None

    workflow.generate(source.id)

    dependent_id = uuid4()
    controller.execute(CreateCardCommand(name="Dependent", card_id=dependent_id))
    dependent = next(card for card in controller.document.cards if card.id == dependent_id)
    derived_asset_id = uuid4()
    derived_source = tmp_path / "derived-source.png"
    Image.new("RGB", (592, 448), "green").save(derived_source)
    bundle_path = session.state.bundle_path
    assert bundle_path is not None
    store = StackStore(bundle_path)
    derived_path = store.store_image_asset(
        derived_source,
        card_id=dependent.id,
        asset_id=derived_asset_id,
    )
    generated_at = datetime.now(UTC)
    derived_background = GeneratedBackground(
        id=derived_asset_id,
        image_path=derived_path,
        provenance=RefineProvenance(
            source=ImageSourceSnapshot(
                card_id=source.id,
                revision_id=source_revision.id,
                background_id=source_background.id,
            ),
            description="A refined source",
            render_prompt="A refined source",
            resolution=GenerateResolution.RESOLUTION_512,
            transformation=RefineTransformation.BALANCED,
            strength=0.50,
            settings=source_background.provenance.settings,
        ),
        created_at=generated_at,
    )
    controller.execute(
        ReplaceRevisionBackgroundCommand(
            card_id=dependent.id,
            revision_id=dependent.active_revision.id,
            background=derived_background,
        )
    )
    document_before_completion = controller.document
    assets_before_completion = set((bundle_path / "assets" / "cards").glob("*/image-*.png"))
    failures: list[object] = []
    workflow.failed.connect(failures.append)

    _complete_generation(workers)

    assert failures
    assert "cannot replace this source background" in str(failures[-1])
    assert controller.document == document_before_completion
    assert (
        next(
            card for card in controller.document.cards if card.id == source.id
        ).active_revision.background
        == source_background
    )
    assert set((bundle_path / "assets" / "cards").glob("*/image-*.png")) == (
        assets_before_completion
    )
    assert not workflow.busy
    worker_call_count = len(workers.calls)
    with pytest.raises(
        BackgroundWorkflowError,
        match="cannot replace this source background",
    ):
        workflow.generate(source.id)
    assert len(workers.calls) == worker_call_count


def test_generate_appends_and_captures_selected_style(tmp_path: Path) -> None:
    workflow, controller, _session, workers, model, card = _bound_workflow(tmp_path)
    style = controller.document.styles[0]
    controller.execute(
        SetRevisionStyleCommand(
            card_id=card.id,
            revision_id=card.active_revision.id,
            style_id=style.id,
        )
    )

    workflow.generate(card.id)
    _complete_generation(workers)

    provenance = controller.document.cards[0].active_revision.provenance
    assert provenance is not None
    assert provenance.operation == "generate"
    assert provenance.inputs.style is not None
    assert provenance.inputs.style.style_id == style.id
    assert provenance.render_prompt == f"A garden.\n\n{style.prompt_text}"
    assert model.calls[-1]["prompt"] == provenance.render_prompt


def test_generate_uses_revision_resolution_and_stack_aspect_ratio(
    tmp_path: Path,
) -> None:
    workflow, controller, _session, workers, model, card = _bound_workflow(tmp_path)
    controller.execute(
        SetRevisionGenerateResolutionCommand(
            card_id=card.id,
            revision_id=card.active_revision.id,
            resolution=GenerateResolution.RESOLUTION_1024,
        )
    )

    workflow.generate(card.id)
    _complete_generation(workers)

    provenance = controller.document.cards[0].active_revision.provenance
    assert provenance is not None
    assert provenance.operation == "generate"
    assert provenance.inputs.resolution is GenerateResolution.RESOLUTION_1024
    assert (model.calls[-1]["width"], model.calls[-1]["height"]) == (1184, 880)
    assert (provenance.settings.width, provenance.settings.height) == (1184, 880)


@pytest.mark.parametrize("aspect_ratio", tuple(AspectRatio))
@pytest.mark.parametrize("resolution", tuple(GenerateResolution))
def test_generate_uses_every_supported_ratio_and_resolution(
    tmp_path: Path,
    aspect_ratio: AspectRatio,
    resolution: GenerateResolution,
) -> None:
    root = tmp_path / aspect_ratio.name.lower() / str(resolution.value)
    workflow, controller, _session, workers, model, card = _bound_workflow(
        root,
        aspect_ratio=aspect_ratio,
        resolution=resolution,
    )

    workflow.generate(card.id)
    _complete_generation(workers)

    expected_dimensions = output_dimensions(resolution, aspect_ratio)
    provenance = controller.document.cards[0].active_revision.provenance
    assert provenance is not None
    assert provenance.operation == "generate"
    assert provenance.inputs.resolution is resolution
    assert (
        model.calls[-1]["width"],
        model.calls[-1]["height"],
    ) == expected_dimensions
    assert (
        provenance.settings.width,
        provenance.settings.height,
    ) == expected_dimensions


def test_description_and_style_changes_suppress_in_flight_generation(
    tmp_path: Path,
) -> None:
    workflow, controller, _session, workers, _model, card = _bound_workflow(tmp_path)
    failures: list[object] = []
    workflow.failed.connect(failures.append)

    workflow.generate(card.id)
    controller.execute(
        EditRevisionDescriptionCommand(
            card_id=card.id,
            revision_id=card.active_revision.id,
            value="Changed while generating",
        )
    )
    _complete_generation(workers)
    assert controller.document.cards[0].active_revision.background is None

    style = controller.document.styles[0]
    controller.execute(
        SetRevisionStyleCommand(
            card_id=card.id,
            revision_id=card.active_revision.id,
            style_id=style.id,
        )
    )
    workflow.generate(card.id)
    controller.execute(
        UpdateStyleCommand(
            style_id=style.id,
            name=style.name,
            prompt_text=style.prompt_text + " More contrast.",
        )
    )
    _complete_generation(workers)

    assert controller.document.cards[0].active_revision.background is None
    workflow.generate(card.id)
    controller.execute(
        SetRevisionGenerateResolutionCommand(
            card_id=card.id,
            revision_id=card.active_revision.id,
            resolution=GenerateResolution.RESOLUTION_768,
        )
    )
    _complete_generation(workers)

    assert controller.document.cards[0].active_revision.background is None
    assert all("changed before generation completed" in str(failure) for failure in failures)
    assert not list((tmp_path / "temporary").glob("generated-*.png"))


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("name", "Updated Style"),
        ("prompt", "Updated visual treatment."),
    ),
)
def test_live_style_draft_cancels_generation_before_commit(
    application: QApplication,
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    workflow, controller, _session, workers, _model, card = _bound_workflow(tmp_path)
    style = controller.document.styles[0]
    controller.execute(
        SetRevisionStyleCommand(
            card_id=card.id,
            revision_id=card.active_revision.id,
            style_id=style.id,
        )
    )
    inspector = Inspector(controller)
    inspector.render(controller.document, card.id)
    inspector.inspector_tabs.setCurrentIndex(inspector._styles_tab_index)
    inspector.render_inputs_changed.connect(workflow.cancel)
    inspector.show()
    editor = inspector.style_name_edit if field == "name" else inspector.style_prompt_edit
    editor.setFocus()
    application.processEvents()

    workflow.generate(card.id)
    operation = workers.operations[-1]
    work = workers.calls[-1]
    if field == "name":
        inspector.style_name_edit.setText(value)
    else:
        inspector.style_prompt_edit.setPlainText(value)

    assert operation.was_cancelled
    assert controller.document.style_by_id(style.id) == style
    assert controller.document.cards[0].active_revision.background is None
    assert callable(work)
    with pytest.raises(ImageGenerationCancelled, match="cancelled"):
        work()
    assert controller.document.cards[0].active_revision.background is None

    inspector._style_editing_finished(
        None,
        Qt.FocusReason.OtherFocusReason,
    )
    changed_style = controller.document.style_by_id(style.id)
    assert changed_style is not None
    expected_name = value if field == "name" else style.name
    expected_prompt = value if field == "prompt" else style.prompt_text
    assert changed_style.name == expected_name
    assert changed_style.prompt_text == expected_prompt

    workflow.generate(card.id)
    _complete_generation(workers)

    provenance = controller.document.cards[0].active_revision.provenance
    assert provenance is not None
    assert provenance.operation == "generate"
    assert provenance.inputs.style is not None
    assert provenance.inputs.style.name == expected_name
    assert provenance.inputs.style.prompt_text == expected_prompt
    inspector.close()


def test_cancelled_generation_cannot_publish_or_leave_output(tmp_path: Path) -> None:
    workflow, controller, _session, workers, _model, card = _bound_workflow(tmp_path)
    original = controller.document

    workflow.generate(card.id)
    operation = workers.operations[-1]
    work = workers.calls[-1]
    assert callable(work)

    workflow.cancel()

    assert operation.was_cancelled
    with pytest.raises(ImageGenerationCancelled, match="cancelled"):
        work()
    assert controller.document == original
    assert not list((tmp_path / "temporary").glob("generated-*.png"))


def test_generate_requires_nonempty_description(tmp_path: Path) -> None:
    workflow, controller, _session, workers, _model, card = _bound_workflow(tmp_path)
    controller.execute(
        EditRevisionDescriptionCommand(
            card_id=card.id,
            revision_id=card.active_revision.id,
            value="",
        )
    )

    with pytest.raises(BackgroundWorkflowError, match="Description"):
        workflow.generate(card.id)
    assert workers.calls == []


def test_two_references_are_sent_once_in_stable_order(tmp_path: Path) -> None:
    workflow, controller, _session, workers, model, target = _bound_workflow(tmp_path)
    sources = tuple(
        _create_generated_source(
            workflow,
            controller,
            workers,
            name=f"Reference {number}",
        )
        for number in (1, 2)
    )
    for position, source in enumerate(sources, start=1):
        controller.execute(
            SetRevisionReferenceCommand(
                card_id=target.id,
                revision_id=target.active_revision.id,
                reference=ResolvedCardReference(target_card_id=source.id),
                position=position,
            )
        )
    controller.execute(
        EditRevisionDescriptionCommand(
            card_id=target.id,
            revision_id=target.active_revision.id,
            value="Place the subject from image 1 beside the setting from image 2.",
        )
    )

    workflow.generate(target.id)
    _complete_generation(workers)

    provenance = controller.document.cards[0].active_revision.provenance
    assert provenance is not None
    assert provenance.operation == "generate"
    assert provenance.inputs.references == tuple(
        ImageReferenceSnapshot(
            card_id=source.id,
            revision_id=source.active_revision.id,
            background_id=source.active_revision.background.id,
        )
        for source in sources
        if source.active_revision.background is not None
    )
    expected_paths = [
        workflow.session.store.asset_path(source.active_revision.background.image_path)
        for source in sources
        if source.active_revision.background is not None
    ]
    assert model.calls[-1]["image_paths"] == expected_paths
    assert len(model.calls[-1]["image_paths"]) == len(set(model.calls[-1]["image_paths"]))
    assert model.calls[-1]["prompt"] == (
        "Place the subject from image 1 beside the setting from image 2."
    )


def test_reference_change_suppresses_in_flight_result(tmp_path: Path) -> None:
    workflow, controller, _session, workers, _model, target = _bound_workflow(tmp_path)
    source = _create_generated_source(
        workflow,
        controller,
        workers,
        name="Reference",
    )
    controller.execute(
        SetRevisionReferenceCommand(
            card_id=target.id,
            revision_id=target.active_revision.id,
            reference=ResolvedCardReference(target_card_id=source.id),
        )
    )
    failures: list[object] = []
    workflow.failed.connect(failures.append)

    workflow.generate(target.id)
    controller.execute(
        DuplicateRevisionCommand(
            card_id=source.id,
            source_revision_id=source.active_revision.id,
        )
    )
    _complete_generation(workers)

    assert controller.document.cards[0].active_revision.background is None
    assert "changed before generation completed" in str(failures[-1])


def test_reference_card_requires_an_active_image(tmp_path: Path) -> None:
    workflow, controller, _session, _workers, _model, target = _bound_workflow(tmp_path)
    source_id = uuid4()
    controller.execute(CreateCardCommand(name="Blank source", card_id=source_id))
    controller.execute(
        SetRevisionReferenceCommand(
            card_id=target.id,
            revision_id=target.active_revision.id,
            reference=ResolvedCardReference(target_card_id=source_id),
        )
    )

    with pytest.raises(BackgroundWorkflowError, match="has no image"):
        workflow.generate(target.id)


def test_revision_duplicate_activate_delete_round_trip(tmp_path: Path) -> None:
    workflow, controller, _session, _workers, _model, card = _bound_workflow(tmp_path)
    original = card.active_revision

    workflow.duplicate_revision(card.id, original.id)
    changed = controller.document.cards[0]
    assert len(changed.revisions) == 2
    duplicate = changed.active_revision
    assert duplicate.id != original.id
    assert duplicate.description == original.description
    assert duplicate.hotspot_set == original.hotspot_set

    workflow.activate_revision(card.id, original.id)
    workflow.delete_revision(card.id, duplicate.id)
    assert len(controller.document.cards[0].revisions) == 1


def test_unbound_stack_rejects_image_changes(tmp_path: Path) -> None:
    card = Card(name="Card", revisions=(CardRevision(description="Scene"),))
    controller = DocumentController(Stack(name="Stack", cards=(card,)))
    workflow = BackgroundWorkflow(
        controller,
        DocumentSession(controller),
        FakeWorkers(),  # type: ignore[arg-type]
        _settings,
        temporary_directory=tmp_path / "temporary",
    )

    with pytest.raises(BackgroundWorkflowError, match="save the stack"):
        workflow.generate(card.id)
