"""Tests for direct revision-local background changes."""

from __future__ import annotations

import gc
import json
import os
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from threading import Event
from time import monotonic
from types import SimpleNamespace
from uuid import UUID, uuid4
from weakref import ref

import pytest
from PIL import Image
from pydantic import ValidationError
from PySide6.QtCore import QCoreApplication, QEvent, QObject, QSettings, Qt, QTimer, Signal
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QWidget
from shiboken6 import isValid

import hotcards.storage.stack_store as stack_store_module
from hotcards.application.applied_image_change import (
    AppliedImageChange,
    EditImageOperation,
)
from hotcards.application.background_workflow import (
    BackgroundGenerationSettings,
    BackgroundWorkflow,
    BackgroundWorkflowError,
)
from hotcards.application.commands import (
    CreateCardCommand,
    CreateImageRevisionCommand,
    DeleteRevisionCommand,
    DuplicateRevisionCommand,
    EditRevisionDescriptionCommand,
    RenameCardCommand,
    ReplaceHotspotSetCommand,
    ReplaceRevisionBackgroundCommand,
    SetRevisionGenerateOutputSizeCommand,
    SetRevisionReferenceCommand,
    SetRevisionStyleCommand,
    UpdateStyleCommand,
)
from hotcards.application.document_controller import DocumentController, UndoToken
from hotcards.application.document_session import DocumentSession
from hotcards.application.workers import AdapterKind, AdapterWorkers
from hotcards.domain.image_dimensions import (
    AspectRatio,
    ResolutionTier,
    higher_output_tiers,
    output_dimensions,
)
from hotcards.domain.models import (
    CURRENT_SCHEMA_VERSION,
    AcceptedEdit,
    Card,
    CardRevision,
    CurrentSourceSize,
    DerivedImageSourceSnapshot,
    DuplicateOperation,
    EditDraft,
    EditOperation,
    ExactOutputSize,
    GeneratedBackground,
    GenerateOperation,
    HotspotSet,
    ImageOriginFacts,
    ImageProvenance,
    ImageReferenceSnapshot,
    ImageSourceSnapshot,
    Interaction,
    NavigateAction,
    Point,
    Polygon,
    PresetOutputSize,
    ResolvedCardReference,
    Stack,
    UnresolvedCardReference,
    image_edit_lineage,
)
from hotcards.generation.mflux_generator import MfluxGenerator, run_model_invocation
from hotcards.storage.stack_store import (
    StackStore,
    StackStoreError,
    StackStoreTransactionError,
)
from hotcards.ui.inspector import Inspector
from hotcards.ui.main_window import MainWindow
from hotcards.ui.utility_windows import StyleManagerWindow


@pytest.fixture(autouse=True)
def qt_resources(qt_application: QApplication, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    resources: list[QObject] = []

    def retain_instances(resource_type: type[QObject]) -> None:
        initialize = resource_type.__init__

        def initialize_owned(resource: QObject, *args: object, **kwargs: object) -> None:
            initialize(resource, *args, **kwargs)
            resources.append(resource)

        monkeypatch.setattr(resource_type, "__init__", initialize_owned)

    for resource_type in (
        DocumentSession,
        BackgroundWorkflow,
        AdapterWorkers,
        MainWindow,
        Inspector,
    ):
        retain_instances(resource_type)
    try:
        yield
    finally:
        for resource in reversed(resources):
            if not isValid(resource):
                continue
            if isinstance(resource, QWidget):
                if resource.parentWidget() is not None:
                    continue
                resource.close()
            elif isinstance(resource, BackgroundWorkflow):
                resource.close()
            elif isinstance(resource, AdapterWorkers):
                resource.shutdown(wait_milliseconds=1_000)
            elif isinstance(resource, DocumentSession):
                for timer in resource.findChildren(QTimer):
                    timer.stop()
                resource.controller.set_autosave_hook(None)
                resource.controller.set_owned_asset_release_hook(None)
            resource.deleteLater()
            QCoreApplication.sendPostedEvents(resource, QEvent.Type.DeferredDelete)


class FakeOperation(QObject):
    succeeded = Signal(object)
    failed = Signal(object)
    cancelled = Signal()
    finished = Signal()

    def __init__(self, request_cancel: object = None) -> None:
        super().__init__()
        self.was_cancelled = False
        self.finished_state = False
        self.request_cancel = request_cancel

    @property
    def is_finished(self) -> bool:
        return self.finished_state

    def cancel(self) -> None:
        if self.finished_state:
            return
        self.was_cancelled = True
        self.finished_state = True
        if callable(self.request_cancel):
            self.request_cancel()
        self.cancelled.emit()
        self.finished.emit()

    def succeed(self, result: object) -> bool:
        if self.finished_state:
            return False
        self.finished_state = True
        self.succeeded.emit(result)
        self.finished.emit()
        return True

    def fail(self, error: object) -> None:
        if self.finished_state:
            return
        self.finished_state = True
        self.failed.emit(error)
        self.finished.emit()


class FakeWorkers:
    def __init__(self) -> None:
        self.calls: list[object] = []
        self.operations: list[FakeOperation] = []
        self.disposers: list[object] = []

    def complete(self) -> None:
        handle = self.operations[-1]
        if handle.is_finished:
            return
        operation = self.calls[-1]
        disposer = self.disposers[-1]
        assert callable(operation)
        try:
            result = operation()
        except Exception as error:
            handle.fail(error)
            return
        if not handle.succeed(result):
            assert callable(disposer)
            disposer(result)

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
            "editing background image",
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
        self.tokenizers = {
            "qwen3": SimpleNamespace(
                tokenizer=lambda _prompt, **_kwargs: {"input_ids": list(range(24))},
                template=None,
                use_chat_template=False,
                add_special_tokens=True,
            )
        }

    def generate_image(self, **kwargs: object) -> FakeMfluxImage:
        self.calls.append(kwargs)
        for image_path in kwargs.get("image_paths", []):
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
    resolution: ResolutionTier = ResolutionTier.SMALL,
    owned_workspace: bool = False,
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
        polygons=(
            Polygon(
                points=(
                    Point(x=0.1, y=0.1),
                    Point(x=0.4, y=0.1),
                    Point(x=0.2, y=0.4),
                )
            ),
        ),
    )
    revision = CardRevision(
        description="A garden",
        hotspot_set=HotspotSet(interactions=(hotspot,)),
        generate_output_size=PresetOutputSize(tier=resolution),
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
        temporary_directory=None if owned_workspace else root / "temporary",
    )
    return workflow, controller, session, workers, model, card


def _complete_generation(workers: FakeWorkers) -> None:
    workers.complete()


def _start_image_operation(workflow: BackgroundWorkflow, card: Card, operation: str) -> None:
    if operation == "generate":
        workflow.generate(card.id)
    else:
        workflow.edit(
            card.id,
            draft=_edit_draft(workflow, card.id, "Open the gate."),
            output_size=workflow.available_edit_output_sizes(card.id)[0],
        )


def _edit_draft(
    workflow: BackgroundWorkflow,
    card_id: UUID,
    instruction: str,
) -> EditDraft:
    card = next(card for card in workflow.controller.document.cards if card.id == card_id)
    return workflow.controller.replace_edit_draft(
        card.id,
        card.active_revision.id,
        instruction,
    )


def _assert_same_revision_except_draft(
    actual: CardRevision,
    expected: CardRevision,
) -> None:
    assert actual.model_copy(update={"edit_draft": expected.edit_draft}) == expected


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
    controller.execute(
        SetRevisionGenerateOutputSizeCommand(
            card_id=source.id,
            revision_id=source.active_revision.id,
            output_size=PresetOutputSize(tier=ResolutionTier.SMALL),
        )
    )
    workflow.generate(source.id)
    _complete_generation(workers)
    return next(card for card in controller.document.cards if card.id == source_id)


def _assign_two_references(workflow: BackgroundWorkflow, workers: FakeWorkers) -> tuple[Card, ...]:
    controller = workflow.controller
    target = controller.document.cards[0]
    sources = tuple(
        _create_generated_source(workflow, controller, workers, name=f"Reference {position}")
        for position in (1, 2)
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
    return sources


def _assert_current_image_controls(window: MainWindow, tier: ResolutionTier) -> None:
    width, height = output_dimensions(tier, window.controller.document.aspect_ratio)
    inspector = window.inspector
    assert inspector.resolution_combo.currentData() == PresetOutputSize(tier=tier)
    assert inspector.edit_resolution_combo.currentData() == CurrentSourceSize(
        width=width,
        height=height,
    )
    assert f"{width} × {height}" in inspector.edit_resolution_combo.toolTip()


def test_explicit_image_journey_reopens_without_sources_and_preserves_run_navigation(
    qt_application: QApplication,
    tmp_path: Path,
) -> None:
    workflow, controller, session, workers, model, card = _bound_workflow(
        tmp_path, resolution=ResolutionTier.SMALL
    )
    references = tuple(
        _create_generated_source(workflow, controller, workers, name=name)
        for name in ("Pattern", "Palette")
    )
    for position, reference in enumerate(references, start=1):
        controller.execute(
            SetRevisionReferenceCommand(
                card_id=card.id,
                revision_id=card.active_revision.id,
                reference=ResolvedCardReference(target_card_id=reference.id),
                position=position,
            )
        )
    style = controller.document.styles[0]
    controller.execute(
        SetRevisionStyleCommand(
            card_id=card.id,
            revision_id=card.active_revision.id,
            style_id=style.id,
        )
    )
    hotspot = card.active_revision.hotspot_set.interactions[0].model_copy(
        update={
            "action": NavigateAction(target=ResolvedCardReference(target_card_id=references[0].id))
        }
    )
    controller.execute(
        ReplaceHotspotSetCommand(
            card_id=card.id,
            revision_id=card.active_revision.id,
            hotspot_set=HotspotSet(interactions=(hotspot,)),
        )
    )
    assert session.flush()
    window_workers = AdapterWorkers()
    window = MainWindow(
        controller,
        window_workers,
        QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat),
        document_session=session,
        background_workflow=workflow,
        project_directory=tmp_path / "projects",
        start_diagnostics=False,
    )
    applied: list[AppliedImageChange] = []
    workflow.image_applied.connect(applied.append)
    window._availability[AdapterKind.MFLUX] = True
    window._update_generation_actions()
    inspector = window.inspector
    try:
        inspector.description_edit.setPlainText(
            "A garden gate with the pattern from image 1 and the palette from image 2."
        )
        inspector.generate_background_button.click()
        before = controller.document.cards[0].active_revision
        assert workflow.busy
        _complete_generation(workers)
        generated = controller.document.cards[0].active_revision
        assert generated.model_copy(update={"background": None}) == before
        assert len(controller.document.cards[0].revisions) == 1
        assert isinstance(generated.provenance.authoring, GenerateOperation)
        assert image_edit_lineage(generated.provenance) == ()
        _assert_current_image_controls(window, ResolutionTier.SMALL)
        generated_path = session.store.asset_path(generated.image_path)
        unrelated_path = generated_path.with_name("unowned.png")
        unrelated_path.write_bytes(generated_path.read_bytes())
        token = controller.current_undo_token
        assert window.notification_bar.dismiss_button.text() == "Keep"
        window.notification_bar.dismiss_button.click()
        assert controller.current_undo_token == token
        assert controller.document.cards[0].active_revision == generated
        assert window._applied_image_change is None

        instruction = "Open the garden gate.\nKeep the hand-painted stars exactly as they are."
        inspector.inspector_tabs.setCurrentIndex(inspector._edit_tab_index)
        inspector.edit_instruction_edit.setPlainText(f"  {instruction}  ")
        inspector.edit_resolution_combo.setCurrentIndex(
            inspector._combo_index_for_data(
                inspector.edit_resolution_combo,
                PresetOutputSize(tier=ResolutionTier.MEDIUM),
            )
        )
        inspector.edit_background_button.click()
        assert workflow.busy
        submitted = controller.document.cards[0].active_revision
        assert submitted.model_copy(update={"edit_draft": generated.edit_draft}) == generated
        _complete_generation(workers)
        edited = controller.document.cards[0].active_revision
        assert (
            edited.model_copy(
                update={
                    "background": generated.background,
                    "edit_draft": generated.edit_draft,
                }
            )
            == generated
        )
        assert len(controller.document.cards[0].revisions) == 1
        assert isinstance(edited.provenance.authoring, EditOperation)
        assert edited.provenance.authoring.source == DerivedImageSourceSnapshot(
            card_id=card.id,
            revision_id=before.id,
            background_id=generated.background.id,
            width=256,
            height=192,
            seed=generated.provenance.settings.seed,
        )
        assert edited.provenance.origin.edit_lineage[-1].instruction == instruction
        assert edited.provenance.origin.render_prompt.endswith(style.prompt_text)
        assert inspector.edit_instruction_edit.toPlainText() == ""
        assert inspector.edit_history_list.item(0).text() == instruction
        _assert_current_image_controls(window, ResolutionTier.MEDIUM)
        edited_path = session.store.asset_path(edited.image_path)

        assert [change.operation.kind for change in applied] == ["generate", "edit"]
        assert [change.previous_revision for change in applied] == [before, submitted]
        assert all(change.revision_id == before.id for change in applied)
        assert session.store.load() == controller.document

        invocation_count = len(model.calls)
        assert window.notification_bar.primary_button.text() == "Create New Version"
        window.notification_bar.primary_button.click()
        checkpoint = controller.document.cards[0].active_revision
        assert checkpoint.id != before.id
        assert (
            checkpoint.model_copy(update={"id": before.id, "edit_draft": edited.edit_draft})
            == edited
        )
        assert checkpoint.edit_draft.instruction == ""
        assert checkpoint.edit_draft.generation_id != edited.edit_draft.generation_id
        restored_original = controller.document.cards[0].revisions[0]
        _assert_same_revision_except_draft(restored_original, generated)
        assert restored_original.edit_draft == edited.edit_draft
        assert controller.document.cards[0].revisions == (restored_original, checkpoint)
        assert controller.current_undo_token != applied[-1].token
        assert checkpoint.background == edited.background
        assert len(model.calls) == invocation_count
        window._delete_revision(before.id)
        assert controller.document.cards[0].revisions == (checkpoint,)
        assert window.revision_combo.currentText() == "1"
        assert session.flush()
        assert session.store.load() == controller.document
        assert all(path.is_file() for path in (generated_path, edited_path))
        assert session.close_history()
        assert not generated_path.exists()
        assert edited_path.is_file()
        assert unrelated_path.is_file()
        assert all(
            session.store.asset_path(reference.active_revision.image_path).is_file()
            for reference in references
        )

        saved = controller.document
        manifest = json.loads((session.store.bundle_path / "stack.json").read_text())
        assert manifest["schema_version"] == CURRENT_SCHEMA_VERSION
        persisted_provenance = manifest["cards"][0]["revisions"][0]["background"]["provenance"]
        assert "edit_lineage" not in persisted_provenance
        assert "edit_lineage" not in persisted_provenance["authoring"]["source"]
        assert persisted_provenance["authoring"]["source"] == (
            edited.provenance.authoring.source.model_dump(mode="json")
        )
        assert persisted_provenance["origin"]["edit_lineage"] == [
            entry.model_dump(mode="json") for entry in image_edit_lineage(edited.provenance)
        ]
        assert session.open(session.store.bundle_path) == saved
        assert controller.document.cards[0].active_revision == checkpoint
        assert not controller.can_undo
        _assert_current_image_controls(window, ResolutionTier.MEDIUM)
        inspector.inspector_tabs.setCurrentIndex(inspector._edit_tab_index)
        window.show()
        qt_application.processEvents()
        history = inspector.edit_history_list
        assert history.count() == 1
        draft = controller.edit_draft(card.id, checkpoint.id)
        QTest.mouseClick(
            history.viewport(),
            Qt.MouseButton.LeftButton,
            pos=history.visualItemRect(history.item(0)).center(),
        )
        recalled = controller.edit_draft(card.id, checkpoint.id)
        assert recalled.instruction == instruction
        assert recalled.generation_id != draft.generation_id
        assert controller.document != saved
        recalled_document = controller.document
        assert not controller.can_undo
        assert len(model.calls) == invocation_count

        window.select_card(references[0].id)
        assert inspector.edit_history_list.count() == 0
        window.mode_button.click()
        assert window._run_session.state.current_card_id == references[0].id
        assert window.revision_combo.isHidden()
        assert window._applied_image_change is None
        window.restart_button.click()
        assert window._run_session.state.current_card_id == card.id
        window.card_canvas.interaction_activated.emit(hotspot.id)
        assert window._run_session.state.current_card_id == references[0].id
        window.back_button.click()
        assert window._run_session.state.current_card_id == card.id
        assert controller.document == recalled_document
        assert len(model.calls) == invocation_count
        assert not workflow.busy
    finally:
        window.close()
        window_workers.shutdown()


def test_restored_edit_branch_survives_duplication_source_deletion_and_further_edits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow, controller, session, workers, model, card = _bound_workflow(tmp_path)
    seeds = iter((101, 202, 303, 404))
    monkeypatch.setattr(
        "hotcards.application.background_workflow.secrets.randbelow",
        lambda _limit: next(seeds),
    )
    window_workers = AdapterWorkers()
    window = MainWindow(
        controller,
        window_workers,
        QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat),
        document_session=session,
        background_workflow=workflow,
        project_directory=tmp_path / "projects",
        start_diagnostics=False,
    )
    window._availability[AdapterKind.MFLUX] = True
    window._update_generation_actions()
    inspector = window.inspector
    try:
        inspector.generate_background_button.click()
        _complete_generation(workers)
        window.notification_bar.dismiss_button.click()
        inspector.inspector_tabs.setCurrentIndex(inspector._edit_tab_index)
        instruction = "Paint a small star on the gate.\nKeep its uneven brush strokes."
        edits: list[CardRevision] = []
        for tier in (ResolutionTier.MEDIUM, ResolutionTier.LARGE):
            inspector.edit_instruction_edit.setPlainText(instruction)
            inspector.edit_resolution_combo.setCurrentIndex(
                inspector._combo_index_for_data(
                    inspector.edit_resolution_combo, PresetOutputSize(tier=tier)
                )
            )
            inspector.edit_background_button.click()
            assert workflow.busy
            _complete_generation(workers)
            edits.append(controller.document.cards[0].active_revision)
            window.notification_bar.dismiss_button.click()
        first, abandoned = edits
        assert first.id == abandoned.id == card.active_revision.id
        assert first.provenance.settings.seed == 101
        assert abandoned.provenance.settings.seed == 202
        abandoned_path = session.store.asset_path(abandoned.image_path)
        abandoned_token = controller.current_undo_token
        assert inspector.edit_history_list.count() == 2

        window.undo()
        restored_first = controller.document.cards[0].active_revision
        assert restored_first.model_copy(update={"edit_draft": first.edit_draft}) == first
        assert inspector.edit_history_list.count() == 1
        assert inspector.edit_instruction_edit.toPlainText() == instruction
        _assert_current_image_controls(window, ResolutionTier.MEDIUM)
        window.redo()
        assert controller.document.cards[0].active_revision == abandoned
        assert inspector.edit_history_list.count() == 2
        assert inspector.edit_instruction_edit.toPlainText() == ""
        _assert_current_image_controls(window, ResolutionTier.LARGE)
        window.undo()
        restored_first = controller.document.cards[0].active_revision
        assert restored_first.model_copy(update={"edit_draft": first.edit_draft}) == first
        assert inspector.edit_instruction_edit.toPlainText() == instruction
        assert abandoned_path.is_file()
        assert session.flush()

        history = inspector.edit_history_list
        history.setCurrentRow(0)
        previous_draft = controller.edit_draft(card.id, first.id)
        QTest.keyClick(history, Qt.Key.Key_Return)
        recalled = controller.edit_draft(card.id, first.id)
        assert recalled.instruction == instruction
        assert recalled.generation_id != previous_draft.generation_id
        window.redo()
        assert controller.edit_draft(card.id, first.id) == recalled
        window.undo()
        assert controller.edit_draft(card.id, first.id) == recalled
        restored_first = controller.document.cards[0].active_revision
        assert restored_first.model_copy(update={"edit_draft": first.edit_draft}) == first
        _assert_current_image_controls(window, ResolutionTier.MEDIUM)
        inspector.edit_background_button.click()
        assert workflow.busy
        _complete_generation(workers)
        branch = controller.document.cards[0].active_revision
        assert branch.id == first.id
        assert branch.provenance.authoring.source.background_id == first.background.id
        assert image_edit_lineage(branch.provenance)[:-1] == image_edit_lineage(first.provenance)
        assert branch.provenance.settings.seed == 303
        assert tuple(edit.instruction for edit in image_edit_lineage(branch.provenance)) == (
            instruction,
            instruction,
        )
        assert [history.item(index).text() for index in range(history.count())] == [
            instruction,
            instruction,
        ]
        assert not controller.can_redo
        assert abandoned_token not in controller.retained_history_tokens
        assert not abandoned_path.exists()
        assert inspector.edit_instruction_edit.toPlainText() == ""
        assert session.store.load() == controller.document
        window.notification_bar.dismiss_button.click()

        branch_path = session.store.asset_path(branch.image_path)
        window._duplicate_card()
        duplicate = controller.document.cards[1]
        duplicate_revision = duplicate.active_revision
        duplicate_path = session.store.asset_path(duplicate_revision.image_path)
        assert window._selected_card_id == duplicate.id
        assert duplicate.id != card.id
        assert duplicate_path != branch_path
        assert duplicate_path.read_bytes() == branch_path.read_bytes()
        assert isinstance(duplicate_revision.provenance.authoring, DuplicateOperation)
        assert duplicate_revision.provenance.origin == branch.provenance.origin
        assert (
            duplicate_revision.provenance.authoring.original_authoring
            == branch.provenance.authoring
        )
        assert duplicate_revision.provenance.authoring.source == ImageSourceSnapshot(
            card_id=card.id,
            revision_id=branch.id,
            background_id=branch.background.id,
        )
        assert (
            duplicate_revision.model_copy(
                update={
                    "id": branch.id,
                    "background": branch.background,
                    "hotspot_set": branch.hotspot_set,
                    "edit_draft": branch.edit_draft,
                }
            )
            == branch
        )
        assert image_edit_lineage(duplicate_revision.provenance) == image_edit_lineage(
            branch.provenance
        )
        window._delete_card(card.id)
        assert controller.document.cards == (duplicate,)
        assert session.flush()
        assert branch_path.is_file()
        assert session.close_history()
        assert not branch_path.exists()
        assert not tuple((session.store.bundle_path / "assets" / "cards" / str(card.id)).glob("*"))
        assert duplicate_path.is_file()
        assert session.open(session.store.bundle_path).cards == (duplicate,)
        assert inspector.edit_history_list.count() == 2
        _assert_current_image_controls(window, ResolutionTier.MEDIUM)

        inspector.inspector_tabs.setCurrentIndex(inspector._edit_tab_index)
        final_instruction = "Add a blue ribbon beside the stars."
        inspector.edit_instruction_edit.setPlainText(final_instruction)
        inspector.edit_background_button.click()
        assert workflow.busy
        _complete_generation(workers)
        result = controller.document.cards[0].active_revision
        assert len(controller.document.cards[0].revisions) == 1
        assert (
            result.model_copy(
                update={
                    "background": duplicate_revision.background,
                    "edit_draft": duplicate_revision.edit_draft,
                }
            )
            == duplicate_revision
        )
        assert result.provenance.authoring.source.background_id == duplicate_revision.background.id
        assert result.provenance.settings.seed == 404
        lineage = image_edit_lineage(result.provenance)
        assert lineage == (
            *image_edit_lineage(branch.provenance),
            result.provenance.origin.edit_lineage[-1],
        )
        assert tuple(edit.instruction for edit in lineage) == (
            instruction,
            instruction,
            final_instruction,
        )
        assert [history.item(index).text() for index in range(history.count())] == [
            edit.instruction for edit in lineage
        ]
        assert session.store.load() == controller.document
        result_path = session.store.asset_path(result.image_path)
        assert session.close_history()
        assert not duplicate_path.exists()
        assert result_path.is_file()
        saved = controller.document
        assert session.open(session.store.bundle_path) == saved
        assert (
            image_edit_lineage(controller.document.cards[0].active_revision.provenance) == lineage
        )
        assert inspector.edit_history_list.count() == 3
    finally:
        window.close()
        window_workers.shutdown()


@pytest.mark.parametrize("outcome", ("success", "committed-error", "observed-after"))
def test_edit_completion_retains_its_token_across_reentrant_session_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
) -> None:
    workflow, controller, session, workers, _model, card = _bound_workflow(tmp_path)
    workflow.generate(card.id)
    _complete_generation(workers)
    source = controller.document.cards[0].active_revision
    renamed = False

    def rename_on_durable_edit(_state: object) -> None:
        nonlocal renamed
        if (
            not renamed
            and not controller.mutation_blocked
            and isinstance(
                controller.document.cards[0].active_revision.provenance.authoring,
                EditOperation,
            )
        ):
            renamed = True
            controller.execute(RenameCardCommand(card_id=card.id, name="After Edit"))

    session.state_changed.disconnect(workflow._session_state_changed)
    session.state_changed.connect(rename_on_durable_edit)
    session.state_changed.connect(workflow._session_state_changed)
    window_workers = AdapterWorkers()
    window = MainWindow(
        controller,
        window_workers,
        QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat),
        document_session=session,
        background_workflow=workflow,
        project_directory=tmp_path / "projects",
        start_diagnostics=False,
    )
    window._availability[AdapterKind.MFLUX] = True
    window._update_generation_actions()
    applied: list[AppliedImageChange] = []
    workflow.image_applied.connect(applied.append)
    real_store = session.store.store_image_asset_and_save

    def persist_with_outcome(path: Path, **kwargs: object) -> object:
        stored = real_store(path, **kwargs)
        after = kwargs["changed_stack"]
        if outcome != "success":
            raise StackStoreTransactionError(
                RuntimeError("injected post-write outcome"),
                persisted_stack=after if outcome == "committed-error" else None,
                observed_stack=after,
                durability_indeterminate=outcome == "observed-after",
                owned_asset=stored,
            )
        return stored

    monkeypatch.setattr(session.store, "store_image_asset_and_save", persist_with_outcome)
    instruction = "Open the gate."
    try:
        window.inspector.inspector_tabs.setCurrentIndex(window.inspector._edit_tab_index)
        window.inspector.set_edit_instruction(instruction)
        window._update_generation_actions()
        window.inspector.edit_background_button.click()
        _complete_generation(workers)
        if outcome == "observed-after":
            assert controller.mutation_blocked
            assert applied == []
            session.flush()
        assert renamed
        assert len(applied) == 1
        change = applied[0]
        assert change.token != controller.current_undo_token
        assert change.token in controller.retained_history_tokens
        assert window._applied_image_change is None
        assert window.inspector.edit_instruction_edit.toPlainText() == ""
        assert window.canvas_card_name.text() == "After Edit"
        assert session.flush()
        assert session.store.load() == controller.document
        assert len(applied) == 1

        window.undo()
        assert controller.current_undo_token == change.token
        assert window.inspector.edit_instruction_edit.toPlainText() == ""
        window.undo()
        restored = controller.document.cards[0].active_revision
        assert restored.model_copy(update={"edit_draft": source.edit_draft}) == source
        assert window.inspector.edit_instruction_edit.toPlainText() == instruction
    finally:
        window.close()
        window_workers.shutdown()
        assert session.close_history()


def test_generate_preserves_autosave_scheduled_by_redo_cleanup_notification(
    tmp_path: Path,
) -> None:
    workflow, controller, session, workers, _model, card = _bound_workflow(tmp_path)
    workflow.generate(card.id)
    _complete_generation(workers)
    original_background = controller.document.cards[0].active_revision.background
    assert original_background is not None
    assert controller.undo()
    renamed = False

    def rename_on_replacement(_state: object) -> None:
        nonlocal renamed
        background = controller.document.cards[0].active_revision.background
        if not renamed and background is not None and background.id != original_background.id:
            renamed = True
            controller.execute(RenameCardCommand(card_id=card.id, name="After generation"))

    session.state_changed.connect(rename_on_replacement)
    workflow.generate(card.id)
    _complete_generation(workers)

    assert renamed
    assert controller.document.cards[0].name == "After generation"
    assert session.state.dirty
    assert session.flush()
    assert session.store.load() == controller.document
    assert not workflow.busy


def test_generate_reports_asset_directory_failure_and_disposes_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow, controller, session, workers, _model, card = _bound_workflow(tmp_path)
    before = controller.document
    failures: list[object] = []
    workflow.failed.connect(failures.append)
    real_mkdir = os.mkdir

    def deny_card_directory(
        path: str | os.PathLike[str],
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> None:
        if str(path) == str(card.id) and dir_fd is not None:
            raise PermissionError("image directory is not writable")
        real_mkdir(path, mode, dir_fd=dir_fd)

    workflow.generate(card.id)
    monkeypatch.setattr(os, "mkdir", deny_card_directory)
    _complete_generation(workers)

    assert len(failures) == 1
    assert "image directory is not writable" in str(failures[0])
    assert not workflow.busy
    assert workflow._pending_result is None
    assert not tuple(workflow._temporary_directory.glob("*.png"))
    assert controller.document == before
    assert session.store.load() == before


def test_generate_uses_description_and_preserves_result_lifecycle(
    qt_application: QApplication,
    tmp_path: Path,
) -> None:
    workflow, controller, session, workers, model, card = _bound_workflow(tmp_path)
    applied: list[AppliedImageChange] = []
    progress: list[tuple[int, int]] = []
    workflow.image_applied.connect(applied.append)
    workflow.generation_progress_changed.connect(
        lambda completed, total: progress.append((completed, total))
    )

    workflow.generate(card.id)
    _complete_generation(workers)
    qt_application.processEvents()

    assert progress == [(0, 4), (1, 4), (2, 4), (3, 4), (4, 4), (0, 0)]
    revision = controller.document.cards[0].active_revision
    assert revision.background is not None
    assert revision.description == "A garden"
    assert revision.hotspot_set is not None
    provenance = revision.provenance
    assert provenance is not None
    assert isinstance(provenance.authoring, GenerateOperation)
    assert provenance.authoring.inputs.description == "A garden"
    assert provenance.origin.render_prompt == "A garden"
    assert (provenance.settings.width, provenance.settings.height) == (256, 192)
    assert model.calls[-1]["prompt"] == "A garden"
    assert not session.state.dirty
    assert session.store.load() == controller.document
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


@pytest.mark.parametrize("hotspot_set", (None, HotspotSet()))
def test_edit_uses_only_secure_current_image_and_replaces_complete_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hotspot_set: HotspotSet | None,
) -> None:
    workflow, controller, session, workers, model, card = _bound_workflow(tmp_path)
    controller.execute(
        ReplaceHotspotSetCommand(
            card_id=card.id,
            revision_id=card.active_revision.id,
            hotspot_set=hotspot_set,
        )
    )
    workflow.generate(card.id)
    _complete_generation(workers)
    source = controller.document.cards[0].active_revision
    assert source.background is not None
    source_path = session.store.asset_path(source.background.image_path)
    reference = _create_generated_source(
        workflow,
        controller,
        workers,
        name="Reference",
    )
    controller.execute(
        SetRevisionReferenceCommand(
            card_id=card.id,
            revision_id=source.id,
            reference=ResolvedCardReference(target_card_id=reference.id),
        )
    )
    source = controller.document.cards[0].active_revision
    style = controller.document.styles[0]
    controller.execute(
        SetRevisionStyleCommand(
            card_id=card.id,
            revision_id=source.id,
            style_id=style.id,
        )
    )
    source = controller.document.cards[0].active_revision
    monkeypatch.setattr(
        "hotcards.application.background_workflow.secrets.randbelow",
        lambda _limit: 8675309,
    )
    applied: list[AppliedImageChange] = []
    workflow.image_applied.connect(applied.append)
    options = workflow.available_edit_output_sizes(card.id)
    assert options[0] == CurrentSourceSize(width=256, height=192)

    draft = _edit_draft(workflow, card.id, "  Open the garden gate.  ")
    submitted = controller.document.cards[0].active_revision
    workflow.edit(card.id, draft=draft, output_size=options[0])
    snapshot_path = next((tmp_path / "temporary").glob(".image-source-*.png"))
    _complete_generation(workers)

    changed_card = controller.document.cards[0]
    edited = changed_card.active_revision
    assert len(changed_card.revisions) == 1
    assert edited.id == source.id
    assert edited.description == source.description
    assert edited.style_id == source.style_id
    assert edited.references == source.references
    assert edited.hotspot_set == hotspot_set
    assert edited.generate_output_size == source.generate_output_size
    assert edited.background is not None
    provenance = edited.background.provenance
    assert isinstance(provenance.authoring, EditOperation)
    assert provenance.authoring.source == DerivedImageSourceSnapshot(
        card_id=card.id,
        revision_id=source.id,
        background_id=source.background.id,
        width=256,
        height=192,
        seed=source.background.provenance.settings.seed,
    )
    assert provenance.origin.edit_lineage[-1].instruction == "Open the garden gate."
    assert provenance.authoring.output_size == CurrentSourceSize(
        width=256,
        height=192,
    )
    assert provenance.origin.render_prompt == f"Open the garden gate.\n\n{style.prompt_text}"
    assert provenance.settings.seed == 8675309
    assert provenance.authoring.prompt_token_count == 24
    assert model.calls[-1]["image_paths"] == [snapshot_path]
    assert model.calls[-1]["prompt"] == provenance.origin.render_prompt
    assert "image_path" not in model.calls[-1]
    assert "image_strength" not in model.calls[-1]
    assert "description" not in model.calls[-1]
    assert "style" not in model.calls[-1]
    assert "references" not in model.calls[-1]
    assert snapshot_path != source_path
    assert not snapshot_path.exists()
    assert applied[-1].message == "Image edited"
    assert applied[-1].previous_revision == submitted
    assert len(applied) == 1
    assert applied[0].card_id == card.id
    assert applied[0].revision_id == edited.id
    assert applied[0].operation == EditImageOperation(instruction="Open the garden gate.")

    assert session.store.load() == controller.document
    assert not session.state.dirty
    token = applied[-1].token
    edited_path = session.store.asset_path(edited.background.image_path)
    assert controller.undo_if_current(token)
    assert session.flush()
    restored = controller.document.cards[0].active_revision
    _assert_same_revision_except_draft(restored, source)
    assert restored.edit_draft.instruction == "  Open the garden gate.  "
    assert edited_path.is_file()
    assert controller.redo()
    assert session.flush()
    assert controller.document.cards[0].active_revision == edited
    assert controller.document.cards[0].revisions == (edited,)

    assert controller.undo()
    assert session.flush()
    controller.execute(RenameCardCommand(card_id=card.id, name="Garden renamed"))
    assert session.flush()
    assert not edited_path.exists()


def test_sequential_edits_append_lineage_and_use_fresh_seeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow, controller, _session, workers, _model, card = _bound_workflow(tmp_path)
    workflow.generate(card.id)
    _complete_generation(workers)
    seeds = iter((101, 202))
    monkeypatch.setattr(
        "hotcards.application.background_workflow.secrets.randbelow",
        lambda _limit: next(seeds),
    )

    accepted_seeds: list[int] = []
    for instruction in ("Open the gate.", "Add ivy."):
        current_size = workflow.available_edit_output_sizes(card.id)[0]
        workflow.edit(
            card.id,
            draft=_edit_draft(workflow, card.id, instruction),
            output_size=current_size,
        )
        _complete_generation(workers)
        accepted_seeds.append(
            controller.document.cards[0].active_revision.background.provenance.settings.seed
        )

    provenance = controller.document.cards[0].active_revision.provenance
    assert isinstance(provenance.authoring, EditOperation)
    assert tuple(edit.instruction for edit in image_edit_lineage(provenance)) == (
        "Open the gate.",
        "Add ivy.",
    )
    assert accepted_seeds == [101, 202]
    assert len(controller.document.cards[0].revisions) == 1


@pytest.mark.parametrize("source_kind", ("duplicate", "origin-only"))
def test_edit_accepts_duplicate_and_origin_only_sources(
    tmp_path: Path,
    source_kind: str,
) -> None:
    workflow, controller, _session, workers, _model, card = _bound_workflow(tmp_path)
    workflow.generate(card.id)
    _complete_generation(workers)
    source = controller.document.cards[0].active_revision
    assert source.background is not None
    provenance = (
        ImageProvenance(
            origin=source.background.provenance.origin,
            authoring=DuplicateOperation(
                source=ImageSourceSnapshot(
                    card_id=uuid4(),
                    revision_id=uuid4(),
                    background_id=uuid4(),
                ),
                original_authoring=source.background.provenance.authoring,
            ),
        )
        if source_kind == "duplicate"
        else ImageProvenance(
            origin=source.background.provenance.origin.model_copy(
                update={
                    "edit_lineage": (
                        AcceptedEdit(
                            instruction="An earlier instruction.\nKeep its exact text.",
                            expanded_prompt="An earlier instruction with its original Style.",
                        ),
                    )
                }
            )
        )
    )
    source_background = source.background.model_copy(
        update={
            "provenance": provenance,
        }
    )
    controller.execute(
        ReplaceRevisionBackgroundCommand(
            card_id=card.id,
            revision_id=source.id,
            background=source_background,
        )
    )

    workflow.edit(
        card.id,
        draft=_edit_draft(workflow, card.id, "Open the gate."),
        output_size=workflow.available_edit_output_sizes(card.id)[0],
    )
    _complete_generation(workers)
    edited = controller.document.cards[0].active_revision
    assert isinstance(edited.provenance.authoring, EditOperation)
    lineage = image_edit_lineage(edited.provenance)
    assert lineage[:-1] == image_edit_lineage(provenance)
    assert lineage[-1].instruction == "Open the gate."
    assert "edit_lineage" not in edited.provenance.authoring.source.model_dump()


@pytest.mark.parametrize(
    ("aspect_ratio", "resolution"),
    (
        (AspectRatio.LANDSCAPE, ResolutionTier.SMALL),
        (AspectRatio.PORTRAIT, ResolutionTier.MEDIUM),
        (AspectRatio.SQUARE, ResolutionTier.FULL),
    ),
)
def test_edit_output_sizes_include_exact_current_then_only_higher_presets(
    tmp_path: Path,
    aspect_ratio: AspectRatio,
    resolution: ResolutionTier,
) -> None:
    workflow, _controller, _session, workers, _model, card = _bound_workflow(
        tmp_path / aspect_ratio.name / str(resolution.value),
        aspect_ratio=aspect_ratio,
        resolution=resolution,
    )
    workflow.generate(card.id)
    _complete_generation(workers)
    width, height = output_dimensions(resolution, aspect_ratio)

    assert workflow.available_edit_output_sizes(card.id) == (
        CurrentSourceSize(width=width, height=height),
        *(
            PresetOutputSize(tier=candidate)
            for candidate in higher_output_tiers(
                width,
                height,
                aspect_ratio,
            )
        ),
    )


def test_edit_rejects_replaced_source_and_failed_model_without_new_version(
    tmp_path: Path,
) -> None:
    workflow, controller, session, workers, model, card = _bound_workflow(tmp_path)
    workflow.generate(card.id)
    _complete_generation(workers)
    source = controller.document.cards[0].active_revision
    assert source.background is not None
    source_path = session.store.asset_path(source.background.image_path)
    failures: list[object] = []
    workflow.failed.connect(failures.append)

    workflow.edit(
        card.id,
        draft=_edit_draft(workflow, card.id, "Open the gate."),
        output_size=workflow.available_edit_output_sizes(card.id)[0],
    )
    snapshot_path = next((tmp_path / "temporary").glob(".image-source-*.png"))
    Image.new("RGB", (256, 192), "gold").save(source_path, format="PNG")
    _complete_generation(workers)

    assert model.consumed_source_pixels[-1] == (0, 0, 128)
    assert model.calls[-1]["image_paths"] == [snapshot_path]
    failed_revision = controller.document.cards[0].active_revision
    _assert_same_revision_except_draft(failed_revision, source)
    assert failed_revision.edit_draft.instruction == "Open the gate."
    assert "changed while Edit was running" in str(failures[-1])
    assert not snapshot_path.exists()

    workflow.edit(
        card.id,
        draft=_edit_draft(workflow, card.id, "Add ivy."),
        output_size=workflow.available_edit_output_sizes(card.id)[0],
    )
    workers.operations[-1].fail(RuntimeError("model failed"))
    failed_revision = controller.document.cards[0].active_revision
    _assert_same_revision_except_draft(failed_revision, source)
    assert failed_revision.edit_draft.instruction == "Add ivy."
    assert "model failed" in str(failures[-1])
    assert not workflow.busy


def test_edit_allows_empty_description_but_suppresses_complete_revision_changes(
    tmp_path: Path,
) -> None:
    workflow, controller, _session, workers, _model, card = _bound_workflow(tmp_path)
    workflow.generate(card.id)
    _complete_generation(workers)
    source = controller.document.cards[0].active_revision
    controller.execute(
        EditRevisionDescriptionCommand(
            card_id=card.id,
            revision_id=source.id,
            value="",
        )
    )
    empty_description_source = controller.document.cards[0].active_revision
    workflow.edit(
        card.id,
        draft=_edit_draft(workflow, card.id, "Open the gate."),
        output_size=workflow.available_edit_output_sizes(card.id)[0],
    )
    _complete_generation(workers)
    assert controller.document.cards[0].active_revision.description == ""

    edited = controller.document.cards[0].active_revision
    style = controller.document.styles[0]
    controller.execute(
        SetRevisionStyleCommand(
            card_id=card.id,
            revision_id=edited.id,
            style_id=style.id,
        )
    )
    styled = controller.document.cards[0].active_revision
    failures: list[object] = []
    workflow.failed.connect(failures.append)
    workflow.edit(
        card.id,
        draft=_edit_draft(workflow, card.id, "Add ivy."),
        output_size=workflow.available_edit_output_sizes(card.id)[0],
    )
    controller.execute(
        UpdateStyleCommand(
            style_id=style.id,
            name=style.name,
            prompt_text=style.prompt_text + " More contrast.",
        )
    )
    _complete_generation(workers)

    failed_revision = controller.document.cards[0].active_revision
    _assert_same_revision_except_draft(failed_revision, styled)
    assert failed_revision.edit_draft.instruction == "Add ivy."
    assert len(controller.document.cards[0].revisions) == 1
    assert "changed before Edit completed" in str(failures[-1])
    assert empty_description_source.description == ""


@pytest.mark.parametrize(
    ("source_size", "edit_tiers"),
    [
        ((641, 480), (ResolutionTier.LARGE, ResolutionTier.FULL)),
        ((1008, 784), ()),
    ],
)
def test_invalid_current_size_keeps_named_workflow_outputs_available(
    tmp_path: Path,
    source_size: tuple[int, int],
    edit_tiers: tuple[ResolutionTier, ...],
) -> None:
    workflow, controller, session, workers, _model, card = _bound_workflow(tmp_path)
    workflow.generate(card.id)
    _complete_generation(workers)
    revision = controller.document.cards[0].active_revision
    assert revision.background is not None
    Image.new("RGB", source_size, "navy").save(
        session.store.asset_path(revision.background.image_path)
    )

    assert workflow.available_edit_output_sizes(card.id) == tuple(
        PresetOutputSize(tier=tier) for tier in edit_tiers
    )

    if not edit_tiers:
        with pytest.raises(BackgroundWorkflowError, match="more pixels"):
            workflow.edit(
                card.id,
                draft=_edit_draft(workflow, card.id, "Open the gate."),
                output_size=PresetOutputSize(tier=ResolutionTier.FULL),
            )
        assert not workflow.busy
        assert not list((tmp_path / "temporary").glob(".image-source-*.png"))
        return

    selected_tier = edit_tiers[0]
    workflow.edit(
        card.id,
        draft=_edit_draft(workflow, card.id, "Open the gate."),
        output_size=PresetOutputSize(tier=selected_tier),
    )
    _complete_generation(workers)

    assert not workflow.busy
    assert not list((tmp_path / "temporary").glob(".image-source-*.png"))
    provenance = controller.document.cards[0].active_revision.provenance
    assert isinstance(provenance.authoring, EditOperation)
    assert provenance.authoring.output_size == PresetOutputSize(tier=selected_tier)
    assert (provenance.authoring.source.width, provenance.authoring.source.height) == source_size
    assert provenance.authoring.source.seed == revision.background.provenance.settings.seed


def test_derived_image_reopens_after_replaced_source_asset_is_reclaimed(
    tmp_path: Path,
) -> None:
    workflow, controller, session, workers, _model, card = _bound_workflow(tmp_path)
    workflow.generate(card.id)
    _complete_generation(workers)
    source = controller.document.cards[0].active_revision
    source_path = session.store.asset_path(source.background.image_path)
    workflow.edit(
        card.id,
        draft=_edit_draft(workflow, card.id, "Open the gate."),
        output_size=CurrentSourceSize(width=256, height=192),
    )
    _complete_generation(workers)
    result = controller.document.cards[0].active_revision
    result_path = session.store.asset_path(result.background.image_path)

    assert source_path.is_file()
    assert controller.undo()
    assert session.flush()
    restored = controller.document.cards[0].active_revision
    _assert_same_revision_except_draft(restored, source)
    assert restored.edit_draft.instruction == "Open the gate."
    assert controller.redo()
    assert session.flush()
    controller.clear_history()
    assert session.flush()

    assert not source_path.exists()
    assert result_path.is_file()
    assert session.store.load().cards[0].revisions == (result,)


def test_edit_indeterminate_observed_after_keeps_instruction_and_promotes_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow, controller, session, workers, _model, card = _bound_workflow(tmp_path)
    workflow.generate(card.id)
    _complete_generation(workers)
    source = controller.document
    previous_token = controller.current_undo_token
    changed_documents: list[Stack] = []
    applied: list[AppliedImageChange] = []
    failures: list[object] = []
    workflow.document_changed.connect(changed_documents.append)
    workflow.image_applied.connect(applied.append)
    workflow.failed.connect(failures.append)
    assert session.store is not None
    real_store = session.store.store_image_asset_and_save

    def fail_indeterminate(
        source_file_path: Path,
        *,
        destination_card_id: object,
        destination_asset_id: object,
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
    workflow.edit(
        card.id,
        draft=_edit_draft(workflow, card.id, "Open the gate."),
        output_size=workflow.available_edit_output_sizes(card.id)[0],
    )
    submitted = controller.document
    _complete_generation(workers)

    pending = controller.document
    assert len(pending.cards[0].revisions) == 1
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
    assert len(applied) == 1
    assert applied[0].message == "Image edited"
    assert applied[0].token == controller.current_undo_token
    assert applied[0].previous_revision == submitted.cards[0].active_revision
    assert applied[0].operation == EditImageOperation(instruction="Open the gate.")
    assert applied[0].card_id == card.id
    assert applied[0].revision_id == pending.cards[0].active_revision.id
    assert changed_documents[-1] == pending
    assert controller.undo()
    assert session.flush()
    restored = controller.document.cards[0].active_revision
    _assert_same_revision_except_draft(restored, source.cards[0].active_revision)
    assert restored.edit_draft.instruction == "Open the gate."


@pytest.mark.parametrize("operation", ("generate", "edit"))
@pytest.mark.parametrize(
    "checkpoint",
    ("destination-created", "asset-file-fsynced", "manifest-file-fsynced", "manifest-replaced"),
)
def test_image_transaction_failure_retains_exact_before_and_no_result_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    checkpoint: str,
) -> None:
    workflow, controller, session, workers, _model, card = _bound_workflow(tmp_path)
    workflow.generate(card.id)
    _complete_generation(workers)
    before = controller.document
    token = controller.current_undo_token
    paths = set(session.store.bundle_path.rglob("image-*.png"))
    applied: list[AppliedImageChange] = []
    failures: list[object] = []
    workflow.image_applied.connect(applied.append)
    workflow.failed.connect(failures.append)
    _start_image_operation(workflow, card, operation)
    before = controller.document

    def fail_checkpoint(name: str) -> None:
        if name == checkpoint:
            raise OSError("injected image transaction failure")

    monkeypatch.setattr(stack_store_module, "_io_checkpoint", fail_checkpoint)
    _complete_generation(workers)

    assert controller.document == before
    assert session.store.load() == before
    assert controller.current_undo_token == token
    assert set(session.store.bundle_path.rglob("image-*.png")) == paths
    assert applied == []
    assert "injected image transaction failure" in str(failures[-1])
    assert not workflow.busy
    assert not list((tmp_path / "temporary").iterdir())


@pytest.mark.parametrize("operation", ("generate", "edit"))
@pytest.mark.parametrize(
    "outcome",
    ("observed-after", "observed-before", "committed-error", "rolled-back-error"),
)
def test_image_partial_transaction_ownership_completion_and_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    outcome: str,
) -> None:
    workflow, controller, session, workers, _model, card = _bound_workflow(tmp_path)
    workflow.generate(card.id)
    _complete_generation(workers)
    controller.clear_history()
    before = controller.document
    applied: list[AppliedImageChange] = []
    failures: list[object] = []
    workflow.image_applied.connect(applied.append)
    workflow.failed.connect(failures.append)
    store = session.store
    real_store = store.store_image_asset_and_save
    real_save = store.save
    owned_paths: list[Path] = []

    def partial_transaction(path: Path, **kwargs: object) -> object:
        stored = real_store(path, **kwargs)
        after = kwargs["changed_stack"]
        owned_paths.append(store.asset_path(stored.relative_path))
        if outcome in {"observed-before", "rolled-back-error"}:
            real_save(before)
        raise StackStoreTransactionError(
            RuntimeError("injected transaction cleanup error"),
            persisted_stack=(
                after
                if outcome == "committed-error"
                else before
                if outcome == "rolled-back-error"
                else None
            ),
            observed_stack=before if outcome in {"observed-before", "rolled-back-error"} else after,
            durability_indeterminate=outcome in {"observed-after", "observed-before"},
            owned_asset=stored,
        )

    _start_image_operation(workflow, card, operation)
    before = controller.document
    monkeypatch.setattr(store, "store_image_asset_and_save", partial_transaction)
    _complete_generation(workers)
    assert failures
    assert owned_paths[0].is_file() == (outcome != "rolled-back-error")
    assert not list((tmp_path / "temporary").iterdir())
    if outcome != "committed-error":
        assert applied == []
        assert session.state.dirty == (outcome != "rolled-back-error")
    if outcome == "observed-after":
        assert controller.mutation_blocked
        assert not controller.can_undo

        def fail_save(_snapshot: Stack) -> None:
            raise StackStoreError("retry save failed")

        monkeypatch.setattr(store, "save", fail_save)
        assert not session.flush()
        assert applied == []
        assert controller.mutation_blocked
        monkeypatch.setattr(store, "save", real_save)
        real_cleanup = session._cleanup_released_assets

        def cleanup_with_error() -> None:
            real_cleanup()
            session._error = "injected retry cleanup error"

        monkeypatch.setattr(session, "_cleanup_released_assets", cleanup_with_error)
    elif outcome in {"observed-before", "rolled-back-error"}:
        assert controller.document == before
        assert not controller.mutation_blocked
        controller.execute(RenameCardCommand(card_id=card.id, name="Newer authoring"))

    assert session.flush()
    assert session.flush()
    assert not controller.mutation_blocked
    if outcome == "observed-after":
        assert session.state.error == "injected retry cleanup error"
    if outcome in {"observed-before", "rolled-back-error"}:
        assert applied == []
        assert not owned_paths[0].exists()
        assert controller.document.cards[0].name == "Newer authoring"
    else:
        assert len(applied) == 1
        assert applied[0].previous_revision == before.cards[0].active_revision
        assert applied[0].token == controller.current_undo_token
        assert applied[0].operation.kind == operation
        if operation == "edit":
            assert applied[0].operation == EditImageOperation(instruction="Open the gate.")
        assert controller.undo_if_current(applied[0].token)
        assert controller.document == before
        assert not controller.can_undo
        assert session.flush()
        assert owned_paths[0].is_file()
        controller.clear_history()
        assert session.flush()
        assert not owned_paths[0].exists()


@pytest.mark.parametrize("operation", ("generate", "edit"))
def test_explicit_versions_share_assets_until_their_last_reachable_revision(
    tmp_path: Path, operation: str
) -> None:
    workflow, controller, session, workers, _model, card = _bound_workflow(tmp_path)
    workflow.generate(card.id)
    _complete_generation(workers)
    source = controller.document.cards[0].active_revision
    source_path = session.store.asset_path(source.background.image_path)
    controller.execute(DuplicateRevisionCommand(card_id=card.id, source_revision_id=source.id))
    before = controller.document
    applied: list[AppliedImageChange] = []
    workflow.image_applied.connect(applied.append)
    _start_image_operation(workflow, card, operation)
    before = controller.document
    _complete_generation(workers)
    after = controller.document
    assert len(after.cards[0].revisions) == 2
    assert after.cards[0].revisions[0] == source
    change = applied[0]
    assert change.previous_revision == before.cards[0].active_revision
    result_path = session.store.asset_path(after.cards[0].active_revision.background.image_path)
    controller.execute(
        CreateImageRevisionCommand(
            card_id=card.id,
            revision_id=change.revision_id,
            previous_revision=change.previous_revision,
        )
    )
    explicit = controller.document
    assert len(explicit.cards[0].revisions) == 3
    assert explicit.cards[0].revisions[0] == before.cards[0].revisions[0]
    _assert_same_revision_except_draft(
        explicit.cards[0].revisions[1],
        before.cards[0].revisions[1],
    )
    assert controller.current_undo_token != change.token
    assert session.flush()
    assert controller.undo()
    assert controller.document == after
    assert controller.undo_if_current(change.token)
    assert controller.document == before
    assert controller.redo()
    assert controller.redo()
    assert session.flush()
    controller.clear_history()
    assert source_path.is_file()
    assert result_path.is_file()
    for revision in before.cards[0].revisions:
        controller.execute(DeleteRevisionCommand(card_id=card.id, revision_id=revision.id))
    assert session.flush()
    assert source_path.is_file()
    assert session.close_history()
    assert not source_path.exists()
    assert result_path.is_file()
    reopened = DocumentSession(DocumentController(Stack(name="Welcome")))
    assert reopened.open(session.store.bundle_path) == controller.document
    assert reopened.close_history()


@pytest.mark.parametrize("operation", ("generate", "edit"))
def test_image_replacement_save_as_and_history_close_keep_only_reachable_owned_bytes(
    tmp_path: Path, operation: str
) -> None:
    workflow, controller, session, workers, _model, card = _bound_workflow(tmp_path)
    workflow.generate(card.id)
    _complete_generation(workers)
    store = session.store
    source = controller.document.cards[0].active_revision.background
    source_path = store.asset_path(source.image_path)
    unknown = source_path.with_name("unknown.png")
    unknown.write_bytes(source_path.read_bytes())
    _start_image_operation(workflow, card, operation)
    _complete_generation(workers)
    result = controller.document.cards[0].active_revision.background
    result_path = store.asset_path(result.image_path)
    assert source_path.is_file()
    session.save_as(tmp_path / "Copy.hotcards")
    assert not source_path.exists()
    assert unknown.is_file()
    assert result_path.is_file()
    copied_path = session.store.asset_path(result.image_path)
    assert copied_path.is_file()
    assert not controller.can_undo
    _start_image_operation(workflow, card, operation)
    _complete_generation(workers)
    newest = controller.document.cards[0].active_revision.background
    newest_path = session.store.asset_path(newest.image_path)
    assert session.close_history()
    assert not copied_path.exists()
    assert newest_path.is_file()
    assert result_path.is_file()
    assert unknown.is_file()


@pytest.mark.parametrize("outcome", ("observed-after", "committed-error"))
@pytest.mark.parametrize("draft_change", ("unchanged", "new-text", "recalled", "context"))
def test_durable_edit_completion_updates_actions_and_drafts_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
    draft_change: str,
) -> None:
    workflow, controller, session, workers, model, card = _bound_workflow(tmp_path)
    workflow.generate(card.id)
    _complete_generation(workers)
    other = CreateCardCommand(name="Other")
    controller.execute(other)
    assert session.flush()
    controller.clear_history()
    before = controller.document
    window_workers = AdapterWorkers()
    window = MainWindow(
        controller,
        window_workers,
        QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat),
        document_session=session,
        background_workflow=workflow,
        start_diagnostics=False,
    )
    applied: list[AppliedImageChange] = []
    workflow.image_applied.connect(applied.append)
    store = session.store
    real_store = store.store_image_asset_and_save

    def partial_transaction(path: Path, **kwargs: object) -> object:
        stored = real_store(path, **kwargs)
        after = kwargs["changed_stack"]
        raise StackStoreTransactionError(
            RuntimeError("injected image cleanup error"),
            persisted_stack=after if outcome == "committed-error" else None,
            observed_stack=after,
            durability_indeterminate=outcome == "observed-after",
            owned_asset=stored,
        )

    monkeypatch.setattr(store, "store_image_asset_and_save", partial_transaction)
    instruction = "Open the gate.\nLeave the tree exactly as it is."
    window.inspector.set_edit_instruction(f"  {instruction}  ")
    window._edit_background(instruction, workflow.available_edit_output_sizes(card.id)[0])
    _complete_generation(workers)
    assert window.inspector.edit_instruction_edit.toPlainText() == ""
    assert len(applied) == (0 if outcome == "observed-after" else 1)
    if outcome == "observed-after":
        assert session.flush()
    if draft_change == "new-text":
        window.inspector.edit_instruction_edit.setPlainText("Paint a blue gate instead.")
    elif draft_change == "recalled":
        window.inspector.set_edit_instruction(f"  {instruction}  ")
    elif draft_change == "context":
        window._selected_card_id = other.card_id
        window.render_document()
        window.inspector.set_edit_instruction(f"  {instruction}  ")
    expected_text = (
        "" if draft_change == "unchanged" else window.inspector.edit_instruction_edit.toPlainText()
    )
    try:
        assert session.flush()
        assert session.flush()
        assert len(applied) == 1
        change = applied[0]
        assert change.operation == EditImageOperation(instruction)
        assert window._applied_image_change is change
        assert change.token in controller.retained_history_tokens
        assert window.inspector.edit_instruction_edit.toPlainText() == expected_text
        after = controller.document
        invocation_count = len(model.calls)
        window._notification_action_requested("create-image-revision")
        assert session.flush()
        assert len(model.calls) == invocation_count
        assert len(applied) == 1
        assert len(controller.document.cards[0].revisions) == 2
        assert controller.current_undo_token != change.token
        window.undo()
        assert controller.document == after
        assert window.inspector.edit_instruction_edit.toPlainText() == expected_text
        window.undo()
        assert controller.document.cards[0].active_revision.background == (
            before.cards[0].active_revision.background
        )
        assert window.inspector.edit_instruction_edit.toPlainText() == (
            f"  {instruction}  " if draft_change == "unchanged" else expected_text
        )
        assert session.flush()
        assert store.load() == controller.document
    finally:
        window.close()
        window_workers.shutdown()


def test_generate_flushes_complete_before_state_without_overwriting_intervening_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workflow, controller, session, workers, _model, card = _bound_workflow(tmp_path)
    workflow.generate(card.id)
    controller.execute(RenameCardCommand(card_id=card.id, name="Before flush"))
    real_save = session.store.save
    injected = False

    def save_then_edit(snapshot: Stack) -> None:
        nonlocal injected
        real_save(snapshot)
        if not injected:
            injected = True
            controller.execute(RenameCardCommand(card_id=card.id, name="During flush"))

    monkeypatch.setattr(session.store, "save", save_then_edit)
    failures: list[object] = []
    applied: list[AppliedImageChange] = []
    workflow.failed.connect(failures.append)
    workflow.image_applied.connect(applied.append)
    _complete_generation(workers)
    assert controller.document.cards[0].name == "During flush"
    assert controller.document.cards[0].active_revision.background is None
    assert session.state.dirty
    assert failures
    assert applied == []
    assert session.flush()
    assert session.store.load() == controller.document
    workflow.generate(card.id)
    _complete_generation(workers)
    assert len(applied) == 1
    assert session.store.load() == controller.document


def test_generate_rejects_settings_changes_without_model_cancellation(
    tmp_path: Path,
) -> None:
    workflow, controller, _session, workers, _model, card = _bound_workflow(tmp_path)
    before = controller.document
    workflow.generate(card.id)
    workflow._settings_provider = lambda: BackgroundGenerationSettings(
        mflux_model="flux2-klein-9b-kv",
        step_count=4,
        quantization=None,
        random_seed=False,
        fixed_seed=42,
    )
    failures: list[object] = []
    workflow.failed.connect(failures.append)
    _complete_generation(workers)
    assert controller.document == before
    assert failures
    assert not list((tmp_path / "temporary").iterdir())


@pytest.mark.parametrize("replacement", ("file", "symlink"))
def test_discarded_generated_asset_never_reclaims_a_replaced_file_identity(
    tmp_path: Path, replacement: str
) -> None:
    workflow, controller, session, workers, _model, card = _bound_workflow(tmp_path)
    workflow.generate(card.id)
    _complete_generation(workers)
    background = controller.document.cards[0].active_revision.background
    path = session.store.asset_path(background.image_path)
    assert controller.undo()
    assert session.flush()
    held = path.with_name("held-owned.png")
    path.rename(held)
    foreign = tmp_path / "foreign.png"
    foreign.write_bytes(b"foreign bytes")
    if replacement == "symlink":
        path.symlink_to(foreign)
    else:
        path.write_bytes(foreign.read_bytes())
    controller.execute(RenameCardCommand(card_id=card.id, name="Truncate redo"))
    assert session.flush()
    assert path.read_bytes() == b"foreign bytes"
    assert foreign.read_bytes() == b"foreign bytes"
    assert held.is_file()
    assert not session.close_history()
    expected_error = (
        "could not securely open owned image" if replacement == "symlink" else "identity changed"
    )
    assert expected_error in session.state.error
    assert path.read_bytes() == b"foreign bytes"


@pytest.mark.parametrize(
    "replacement_checkpoint",
    ("manifest-file-fsynced", "manifest-directory-fsynced"),
)
def test_derived_source_replacement_during_commit_rolls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement_checkpoint: str,
) -> None:
    workflow, controller, session, workers, _model, card = _bound_workflow(tmp_path)
    workflow.generate(card.id)
    _complete_generation(workers)
    source = controller.document
    source_revision = source.cards[0].active_revision
    assert source_revision.background is not None
    source_path = session.store.asset_path(source_revision.background.image_path)
    replacement = tmp_path / f"{replacement_checkpoint}.png"
    Image.new("RGB", (256, 192), "gold").save(replacement, format="PNG")
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
    assets_before = set((session.store.bundle_path / "assets" / "cards").glob("*/image-*.png"))
    _start_image_operation(workflow, card, "edit")
    source = controller.document
    _complete_generation(workers)

    assert replaced
    assert controller.document == source
    assert session.store.load() == source
    assert (
        set((session.store.bundle_path / "assets" / "cards").glob("*/image-*.png")) == assets_before
    )
    assert "changed while Edit was running" in str(failures[-1])


def test_generate_preserves_independent_normalized_origin_backgrounds(
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
    Image.new("RGB", (256, 192), "green").save(derived_source)
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
        provenance=ImageProvenance(
            origin=ImageOriginFacts(
                render_prompt="A refined source",
                settings=source_background.provenance.settings,
            ),
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

    assert not failures
    assert controller.document != document_before_completion
    assert (
        next(
            card for card in controller.document.cards if card.id == source.id
        ).active_revision.background
        != source_background
    )
    assert assets_before_completion < set((bundle_path / "assets" / "cards").glob("*/image-*.png"))
    assert controller.document.cards[-1].active_revision.background == derived_background
    assert not workflow.busy
    worker_call_count = len(workers.calls)
    workflow.generate(source.id)
    _complete_generation(workers)
    assert len(workers.calls) == worker_call_count + 1
    assert not failures


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
    assert isinstance(provenance.authoring, GenerateOperation)
    assert provenance.authoring.inputs.style is not None
    assert provenance.authoring.inputs.style.style_id == style.id
    assert provenance.origin.render_prompt == f"A garden.\n\n{style.prompt_text}"
    assert model.calls[-1]["prompt"] == provenance.origin.render_prompt


def test_generate_uses_revision_output_size_and_stack_aspect_ratio(
    tmp_path: Path,
) -> None:
    workflow, controller, _session, workers, model, card = _bound_workflow(tmp_path)
    controller.execute(
        SetRevisionGenerateOutputSizeCommand(
            card_id=card.id,
            revision_id=card.active_revision.id,
            output_size=PresetOutputSize(tier=ResolutionTier.FULL),
        )
    )

    workflow.generate(card.id)
    _complete_generation(workers)

    provenance = controller.document.cards[0].active_revision.provenance
    assert provenance is not None
    assert isinstance(provenance.authoring, GenerateOperation)
    assert provenance.authoring.inputs.output_size == PresetOutputSize(tier=ResolutionTier.FULL)
    assert (model.calls[-1]["width"], model.calls[-1]["height"]) == (1024, 768)
    assert (provenance.settings.width, provenance.settings.height) == (1024, 768)


def test_generate_can_reuse_an_exact_nonstandard_current_image_size(
    tmp_path: Path,
) -> None:
    workflow, controller, session, workers, model, card = _bound_workflow(tmp_path)
    workflow.generate(card.id)
    _complete_generation(workers)
    revision = controller.document.cards[0].active_revision
    assert revision.background is not None
    Image.new("RGB", (64, 48), "navy").save(
        session.store.asset_path(revision.background.image_path)
    )
    controller.execute(
        SetRevisionGenerateOutputSizeCommand(
            card_id=card.id,
            revision_id=revision.id,
            output_size=ExactOutputSize(width=64, height=48),
        )
    )

    workflow.generate(card.id)
    _complete_generation(workers)

    generated = controller.document.cards[0].active_revision
    assert isinstance(generated.provenance.authoring, GenerateOperation)
    assert generated.provenance.authoring.inputs.output_size == ExactOutputSize(
        width=64,
        height=48,
    )
    assert (model.calls[-1]["width"], model.calls[-1]["height"]) == (
        64,
        48,
    )


@pytest.mark.parametrize(
    ("aspect_ratio", "resolution"),
    (
        (AspectRatio.SQUARE, ResolutionTier.SMALL),
        (AspectRatio.PORTRAIT, ResolutionTier.MEDIUM),
        (AspectRatio.WIDESCREEN, ResolutionTier.LARGE),
    ),
)
def test_generate_forwards_selected_ratio_and_tier(
    tmp_path: Path,
    aspect_ratio: AspectRatio,
    resolution: ResolutionTier,
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
    assert isinstance(provenance.authoring, GenerateOperation)
    assert provenance.authoring.inputs.output_size == PresetOutputSize(tier=resolution)
    assert (
        model.calls[-1]["width"],
        model.calls[-1]["height"],
    ) == expected_dimensions
    assert (
        provenance.settings.width,
        provenance.settings.height,
    ) == expected_dimensions


@pytest.mark.parametrize("change", ("description", "style", "resolution"))
def test_context_changes_suppress_in_flight_generation(
    tmp_path: Path,
    change: str,
) -> None:
    workflow, controller, _session, workers, _model, card = _bound_workflow(
        tmp_path, resolution=ResolutionTier.SMALL
    )
    style = controller.document.styles[0]
    controller.execute(
        SetRevisionStyleCommand(
            card_id=card.id,
            revision_id=card.active_revision.id,
            style_id=style.id,
        )
    )
    failures: list[object] = []
    applied: list[AppliedImageChange] = []
    changes: list[object] = []
    busy: list[bool] = []
    workflow.failed.connect(failures.append)
    workflow.image_applied.connect(applied.append)
    workflow.document_changed.connect(changes.append)
    workflow.busy_changed.connect(busy.append)

    workflow.generate(card.id)
    commands = {
        "description": EditRevisionDescriptionCommand(
            card_id=card.id,
            revision_id=card.active_revision.id,
            value="Changed while generating",
        ),
        "style": UpdateStyleCommand(
            style_id=style.id,
            name=style.name,
            prompt_text=style.prompt_text + " More contrast.",
        ),
        "resolution": SetRevisionGenerateOutputSizeCommand(
            card_id=card.id,
            revision_id=card.active_revision.id,
            output_size=PresetOutputSize(tier=ResolutionTier.LARGE),
        ),
    }
    controller.execute(commands[change])
    before = controller.document
    token = controller.current_undo_token
    _complete_generation(workers)

    assert controller.document == before
    assert controller.current_undo_token == token
    assert len(failures) == 1
    assert "changed before generation completed" in str(failures[0])
    assert applied == changes == []
    assert busy == [True, False]
    assert not list((tmp_path / "temporary").iterdir())


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("name", "Updated Style"),
        ("prompt", "Updated visual treatment."),
    ),
)
def test_live_style_draft_cancels_generation_before_commit(
    qt_application: QApplication,
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
    manager = StyleManagerWindow(controller, inspector)
    manager.inputs_changed.connect(workflow.cancel)
    manager.show()
    editor = manager.name_edit if field == "name" else manager.prompt_edit
    editor.setFocus()
    qt_application.processEvents()

    workflow.generate(card.id)
    operation = workers.operations[-1]
    if field == "name":
        manager.name_edit.setText(value)
    else:
        manager.prompt_edit.setPlainText(value)

    assert operation.was_cancelled
    assert controller.document.style_by_id(style.id) == style
    assert controller.document.cards[0].active_revision.background is None
    _complete_generation(workers)
    assert controller.document.cards[0].active_revision.background is None

    manager._editing_finished(
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
    assert isinstance(provenance.authoring, GenerateOperation)
    assert provenance.authoring.inputs.style is not None
    assert provenance.authoring.inputs.style.name == expected_name
    assert provenance.authoring.inputs.style.prompt_text == expected_prompt
    manager.close()


@pytest.mark.parametrize("with_references", (False, True))
def test_cancelled_queued_generation_never_invokes_the_model(
    tmp_path: Path,
    with_references: bool,
) -> None:
    workflow, controller, _session, workers, _model, card = _bound_workflow(tmp_path)
    if with_references:
        _assign_two_references(workflow, workers)
    original = controller.document
    model_calls = len(_model.calls)

    workflow.generate(card.id)
    operation = workers.operations[-1]
    finished: list[None] = []
    operation.finished.connect(lambda: finished.append(None))

    workflow.cancel()

    assert operation.was_cancelled
    _complete_generation(workers)
    assert controller.document == original
    assert finished == [None]
    assert len(_model.calls) == model_calls
    assert not list((tmp_path / "temporary").iterdir())


def test_cancellation_after_native_success_disposes_undelivered_output(tmp_path: Path) -> None:
    workflow, controller, _session, workers, _model, card = _bound_workflow(tmp_path)
    original = controller.document
    applied: list[AppliedImageChange] = []
    failures: list[object] = []
    workflow.image_applied.connect(applied.append)
    workflow.failed.connect(failures.append)
    operation = workflow.generate(card.id)
    native_work = workers.calls[-1]
    assert callable(native_work)
    output_paths: list[Path] = []

    def cancel_before_delivery() -> object:
        result = native_work()
        output_paths.append(result.output_path)
        assert result.output_path.is_file()
        workflow.cancel()
        return result

    workers.calls[-1] = cancel_before_delivery
    _complete_generation(workers)

    assert operation.is_finished
    assert operation.was_cancelled
    assert controller.document == original
    assert applied == failures == []
    assert len(output_paths) == 1
    assert not output_paths[0].exists()
    assert not list(workflow._temporary_directory.iterdir())


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


def test_two_references_are_snapshotted_once_in_stable_order(tmp_path: Path) -> None:
    workflow, controller, _session, workers, model, target = _bound_workflow(tmp_path)
    sources = _assign_two_references(workflow, workers)
    for position, source in enumerate(sources, start=1):
        path = workflow.session.store.asset_path(source.active_revision.background.image_path)
        Image.new("RGB", (32, 32), ("red", "green")[position - 1]).save(path)
    controller.execute(
        EditRevisionDescriptionCommand(
            card_id=target.id,
            revision_id=target.active_revision.id,
            value="Place the subject from image 1 beside the setting from image 2.",
        )
    )

    workflow.generate(target.id)
    snapshots = workflow._source_snapshots
    assert len(snapshots) == 2
    assert all(snapshot.snapshot_path.is_file() for snapshot in snapshots)
    _complete_generation(workers)

    provenance = controller.document.cards[0].active_revision.provenance
    assert provenance is not None
    assert isinstance(provenance.authoring, GenerateOperation)
    assert provenance.authoring.inputs.references == tuple(
        ImageReferenceSnapshot(
            card_id=source.id,
            revision_id=source.active_revision.id,
            background_id=source.active_revision.background.id,
        )
        for source in sources
        if source.active_revision.background is not None
    )
    live_paths = [
        workflow.session.store.asset_path(source.active_revision.background.image_path)
        for source in sources
        if source.active_revision.background is not None
    ]
    expected_paths = [snapshot.snapshot_path for snapshot in snapshots]
    assert model.calls[-1]["image_paths"] == expected_paths
    assert set(expected_paths).isdisjoint(live_paths)
    assert model.consumed_source_pixels[-2:] == [(255, 0, 0), (0, 128, 0)]
    assert all(not path.exists() for path in expected_paths)
    assert len(model.calls[-1]["image_paths"]) == len(set(model.calls[-1]["image_paths"]))
    assert model.calls[-1]["prompt"] == (
        "Place the subject from image 1 beside the setting from image 2."
    )


@pytest.mark.parametrize(
    ("checkpoint", "replacement", "source_index"),
    (
        ("before-invocation", "file", 0),
        ("before-invocation", "in-place", 1),
        ("before-invocation", "symlink", 0),
        ("manifest-file-fsynced", "file", 1),
        ("manifest-directory-fsynced", "file", 0),
    ),
)
def test_generate_rejects_reference_asset_replacement_through_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    checkpoint: str,
    replacement: str,
    source_index: int,
) -> None:
    workflow, controller, session, workers, model, target = _bound_workflow(tmp_path)
    references = _assign_two_references(workflow, workers)
    source = references[source_index]
    source_path = session.store.asset_path(source.active_revision.image_path)
    foreign = tmp_path / "foreign.png"
    Image.new("RGB", (256, 192), "gold").save(foreign)
    failures: list[object] = []
    applied: list[AppliedImageChange] = []
    workflow.failed.connect(failures.append)
    workflow.image_applied.connect(applied.append)
    assert session.flush()
    manifest = session.store.stack_path.read_bytes()
    before = controller.document
    token = controller.current_undo_token
    replaced = False

    def replace_reference(name: str) -> None:
        nonlocal replaced
        if name != checkpoint or replaced:
            return
        replaced = True
        if replacement == "in-place":
            source_path.write_bytes(foreign.read_bytes())
        else:
            source_path.rename(source_path.with_name("held-source.png"))
            if replacement == "symlink":
                source_path.symlink_to(foreign)
            else:
                source_path.write_bytes(foreign.read_bytes())

    workflow.generate(target.id)
    snapshots = workflow._source_snapshots
    replace_reference("before-invocation")
    monkeypatch.setattr(stack_store_module, "_io_checkpoint", replace_reference)
    _complete_generation(workers)

    assert replaced
    assert model.consumed_source_pixels[-2:] == [(0, 0, 128), (0, 0, 128)]
    assert len(failures) == 1
    assert applied == []
    assert controller.document == before
    assert controller.current_undo_token == token
    assert session.store.stack_path.read_bytes() == manifest
    assert not list((session.store.bundle_path / "assets" / "cards" / str(target.id)).glob("*"))
    assert not workflow.busy
    assert not workflow.invocation_active
    assert all(not snapshot.snapshot_path.exists() for snapshot in snapshots)
    assert not list(workflow._temporary_directory.iterdir())
    assert source_path.read_bytes() == foreign.read_bytes()
    assert source_path.is_symlink() == (replacement == "symlink")


@pytest.mark.parametrize("invalid_source", ("missing", "unreadable", "symlink"))
def test_generate_cleans_prior_snapshots_when_later_reference_is_invalid(
    tmp_path: Path,
    invalid_source: str,
) -> None:
    workflow, controller, session, workers, _model, target = _bound_workflow(tmp_path)
    sources = _assign_two_references(workflow, workers)
    path = session.store.asset_path(sources[1].active_revision.image_path)
    held = path.with_name("held-source.png")
    path.rename(held)
    if invalid_source == "unreadable":
        path.write_bytes(b"not an image")
    elif invalid_source == "symlink":
        path.symlink_to(held)
    before = controller.document
    call_count = len(workers.calls)

    with pytest.raises(BackgroundWorkflowError, match="Reference 2.*unavailable or unreadable"):
        workflow.generate(target.id)

    assert controller.document == before
    assert len(workers.calls) == call_count
    assert not workflow.busy
    assert not workflow.invocation_active
    assert not list(workflow._temporary_directory.iterdir())
    assert held.is_file()
    assert path.is_symlink() == (invalid_source == "symlink")


def test_generate_submission_failure_releases_all_reference_snapshots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow, _controller, _session, workers, _model, target = _bound_workflow(tmp_path)
    _assign_two_references(workflow, workers)

    def fail_submission(*_args: object, **_kwargs: object) -> None:
        assert len(workflow._source_snapshots) == 2
        raise RuntimeError("worker unavailable")

    monkeypatch.setattr(workers, "run_mflux", fail_submission)
    with pytest.raises(RuntimeError, match="worker unavailable"):
        workflow.generate(target.id)

    assert not workflow.busy
    assert not workflow.invocation_active
    assert workflow._request_id is None
    assert workflow._source_snapshots == ()
    assert not list(workflow._temporary_directory.iterdir())


@pytest.mark.parametrize("close", (False, True))
def test_reference_snapshots_live_until_cancelled_invocation_unwinds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    close: bool,
) -> None:
    workflow, controller, _session, workers, model, target = _bound_workflow(tmp_path)
    _assign_two_references(workflow, workers)
    original_generate = model.generate_image
    before = controller.document
    applied: list[AppliedImageChange] = []
    failed: list[object] = []
    active: list[bool] = []
    workflow.image_applied.connect(applied.append)
    workflow.failed.connect(failed.append)
    workflow.invocation_active_changed.connect(active.append)

    def cancel_during_inference(**kwargs: object) -> FakeMfluxImage:
        assert workflow.invocation_active
        paths = kwargs["image_paths"]
        assert len(paths) == 2
        (workflow.close if close else workflow.cancel)()
        assert workflow.invocation_active
        assert all(path.is_file() for path in paths)
        return original_generate(**kwargs)

    monkeypatch.setattr(model, "generate_image", cancel_during_inference)
    workflow.generate(target.id)
    _complete_generation(workers)

    assert controller.document == before
    assert applied == failed == []
    assert active == [True, False]
    assert workers.operations[-1].was_cancelled
    assert not workflow.busy
    assert not workflow.invocation_active
    assert workflow._source_snapshots == ()
    assert not list(workflow._temporary_directory.iterdir())


@pytest.mark.parametrize("replacement", ("file", "symlink"))
def test_reference_snapshot_cleanup_never_removes_replacement_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement: str,
) -> None:
    monkeypatch.setattr("hotcards.application.background_workflow.tempfile.tempdir", str(tmp_path))
    workflow, _controller, _session, workers, _model, target = _bound_workflow(
        tmp_path, owned_workspace=True
    )
    _assign_two_references(workflow, workers)
    workflow.generate(target.id)
    snapshots = workflow._source_snapshots
    changed = snapshots[0].snapshot_path
    held = tmp_path / "held-snapshot.png"
    changed.rename(held)
    foreign = tmp_path / "foreign.txt"
    foreign.write_bytes(b"not owned by the workflow")
    if replacement == "symlink":
        changed.symlink_to(foreign)
    else:
        changed.write_bytes(foreign.read_bytes())

    workflow.close()
    _complete_generation(workers)

    assert changed.read_bytes() == foreign.read_bytes()
    assert held.is_file()
    assert not snapshots[1].snapshot_path.exists()
    assert workflow._source_snapshots == (snapshots[0],)
    assert workflow.invocation_active
    with pytest.raises(BackgroundWorkflowError, match="could not be cleaned up"):
        workflow.generate(target.id)


def test_real_worker_keeps_references_until_native_unwind_after_close(
    qt_application: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("hotcards.application.background_workflow.tempfile.tempdir", str(tmp_path))
    workflow, controller, _session, workers, model, target = _bound_workflow(
        tmp_path, owned_workspace=True
    )
    _assign_two_references(workflow, workers)
    real_workers = AdapterWorkers()
    workflow.workers = real_workers
    entered = Event()
    resume = Event()
    original_generate = model.generate_image
    paths: list[Path] = []
    before = controller.document
    applied: list[AppliedImageChange] = []
    failures: list[object] = []
    workflow.image_applied.connect(applied.append)
    workflow.failed.connect(failures.append)

    def blocked_generate(**kwargs: object) -> FakeMfluxImage:
        paths.extend(kwargs["image_paths"])
        entered.set()
        assert resume.wait(5)
        assert all(path.is_file() for path in paths)
        return original_generate(**kwargs)

    def wait_until(predicate: Callable[[], bool]) -> None:
        deadline = monotonic() + 5
        while not predicate() and monotonic() < deadline:
            qt_application.processEvents()
            QTest.qWait(1)
        assert predicate()

    monkeypatch.setattr(model, "generate_image", blocked_generate)
    try:
        operation = workflow.generate(target.id)
        wait_until(entered.is_set)
        workflow.close()
        assert operation.is_finished
        assert workflow.invocation_active
        assert len(paths) == 2
        assert all(path.is_file() for path in paths)
        resume.set()
        wait_until(lambda: not workflow.invocation_active)
        assert controller.document == before
        assert applied == failures == []
        assert all(not path.exists() for path in paths)
        wait_until(lambda: not workflow._temporary_directory.exists())
    finally:
        resume.set()
        workflow.close()
        real_workers.shutdown()


def test_timer_resources_remain_owned_during_native_thread_collection(tmp_path: Path) -> None:
    workflow, controller, session, _workers, _model, card = _bound_workflow(tmp_path)
    native_workers = AdapterWorkers()
    native_workers._deadline_timer.start()
    controller.execute(RenameCardCommand(card_id=card.id, name="Autosave pending"))
    assert session._timer.isActive()
    assert native_workers._deadline_timer.isActive()
    retained = tuple(ref(resource) for resource in (workflow, session, native_workers))
    del workflow, controller, session, native_workers

    run_model_invocation(gc.collect)

    assert all(resource() is not None and isValid(resource()) for resource in retained)


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


@pytest.mark.parametrize("operation", ["generate", "edit"])
@pytest.mark.parametrize("corruption", ["receipt", "nested_dimensions"])
def test_acceptance_revalidates_complete_result_before_persistence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str, corruption: str
) -> None:
    workflow, controller, session, workers, _model, card = _bound_workflow(tmp_path)
    workflow.generate(card.id)
    workers.complete()
    failures: list[object] = []
    applied: list[object] = []
    outputs: list[Path] = []
    workflow.failed.connect(failures.append)
    workflow.image_applied.connect(applied.append)
    generate = getattr(workflow._mflux_generator, operation)

    def invalid_result(*args, **kwargs):
        result = generate(*args, **kwargs)
        outputs.append(result.output_path)
        provenance = result.provenance
        if corruption == "receipt":
            provenance = provenance.model_copy(update={"authoring": None})
        else:
            settings = provenance.settings.model_copy(update={"width": 0})
            origin = provenance.origin.model_copy(update={"settings": settings})
            provenance = provenance.model_copy(update={"origin": origin})
        return result.model_copy(update={"provenance": provenance})

    monkeypatch.setattr(workflow._mflux_generator, operation, invalid_result)
    try:
        _start_image_operation(workflow, card, operation)
        assert session.flush()
        before = controller.document
        undo_token = controller.current_undo_token
        workers.complete()

        assert len(failures) == 1 and isinstance(failures[0], ValidationError)
        assert not applied and not workflow.busy
        assert controller.document == before
        assert controller.current_undo_token == undo_token
        assert session.store.load() == before
        assert len(outputs) == 1 and not outputs[0].exists()
    finally:
        workflow.close()
        session.close_history()
        workflow.deleteLater()
        session.deleteLater()
