"""Tests for direct revision-local background changes."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PIL import Image
from PySide6.QtCore import QObject, QSettings, Qt, Signal
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

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
    Card,
    CardRevision,
    CurrentSourceSize,
    DerivedImageSourceSnapshot,
    DirectGenerateProvenance,
    DuplicateProvenance,
    EditProvenance,
    ExactOutputSize,
    GeneratedBackground,
    HotspotSet,
    ImageReferenceSnapshot,
    ImageSourceSnapshot,
    Interaction,
    NavigateAction,
    Point,
    Polygon,
    PresetOutputSize,
    RefineProvenance,
    RefineTransformation,
    ResolvedCardReference,
    Stack,
    UnresolvedCardReference,
    image_edit_lineage,
)
from hotcards.generation.errors import ImageGenerationCancelled
from hotcards.generation.mflux_generator import MfluxGenerator
from hotcards.storage.stack_store import (
    StackStore,
    StackStoreError,
    StackStoreTransactionError,
)
from hotcards.ui.inspector import Inspector
from hotcards.ui.main_window import MainWindow
from hotcards.ui.utility_windows import StyleManagerWindow


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
        image_path = kwargs.get("image_path")
        image_paths = kwargs.get("image_paths")
        if image_path is None and isinstance(image_paths, list) and len(image_paths) == 1:
            image_path = image_paths[0]
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
    resolution: ResolutionTier = ResolutionTier.MEDIUM,
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
        temporary_directory=root / "temporary",
    )
    return workflow, controller, session, workers, model, card


def _complete_generation(workers: FakeWorkers) -> None:
    operation = workers.operations[-1]
    work = workers.calls[-1]
    assert callable(work)
    assert callable(workers.disposers[-1])
    operation.succeeded.emit(work())


def _start_image_operation(workflow: BackgroundWorkflow, card: Card, operation: str) -> None:
    if operation == "generate":
        workflow.generate(card.id)
    else:
        workflow.edit(
            card.id,
            instruction="Open the gate.",
            output_size=workflow.available_edit_output_sizes(card.id)[0],
        )


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
    application: QApplication,
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
        assert isinstance(generated.provenance, DirectGenerateProvenance)
        assert generated.provenance.inputs.description == before.description
        assert generated.provenance.inputs.style.prompt_text == style.prompt_text
        assert tuple(
            snapshot.card_id for snapshot in generated.provenance.inputs.references
        ) == tuple(reference.id for reference in references)
        assert model.calls[-1]["image_paths"] == [
            session.store.asset_path(reference.active_revision.image_path)
            for reference in references
        ]
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
        assert controller.document.cards[0].active_revision == generated
        _complete_generation(workers)
        edited = controller.document.cards[0].active_revision
        assert edited.model_copy(update={"background": generated.background}) == generated
        assert len(controller.document.cards[0].revisions) == 1
        assert isinstance(edited.provenance, EditProvenance)
        assert edited.provenance.source == DerivedImageSourceSnapshot(
            card_id=card.id,
            revision_id=before.id,
            background_id=generated.background.id,
            width=256,
            height=192,
            seed=generated.provenance.settings.seed,
            edit_lineage=(),
        )
        assert edited.provenance.instruction == instruction
        assert edited.provenance.expanded_prompt.endswith(style.prompt_text)
        assert model.calls[-1]["prompt"] == edited.provenance.expanded_prompt
        assert len(model.calls[-1]["image_paths"]) == 1
        assert not model.calls[-1]["image_paths"][0].exists()
        assert inspector.edit_instruction_edit.toPlainText() == ""
        assert inspector.edit_history_list.item(0).text() == instruction
        _assert_current_image_controls(window, ResolutionTier.MEDIUM)
        edited_path = session.store.asset_path(edited.image_path)

        assert [change.operation.kind for change in applied] == ["generate", "edit"]
        assert [change.previous_revision for change in applied] == [before, generated]
        assert all(change.revision_id == before.id for change in applied)
        assert session.store.load() == controller.document

        invocation_count = len(model.calls)
        assert window.notification_bar.primary_button.text() == "Create New Version"
        window.notification_bar.primary_button.click()
        checkpoint = controller.document.cards[0].active_revision
        assert checkpoint.id != before.id
        assert checkpoint.model_copy(
            update={"id": before.id, "edit_draft": edited.edit_draft}
        ) == edited
        assert checkpoint.edit_draft.instruction == ""
        assert checkpoint.edit_draft.generation_id != edited.edit_draft.generation_id
        assert controller.document.cards[0].revisions == (generated, checkpoint)
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
        assert persisted_provenance["source"] == edited.provenance.source.model_dump(mode="json")
        assert session.open(session.store.bundle_path) == saved
        assert controller.document.cards[0].active_revision == checkpoint
        assert not controller.can_undo
        _assert_current_image_controls(window, ResolutionTier.MEDIUM)
        assert window._edit_undo_changes == {}
        inspector.inspector_tabs.setCurrentIndex(inspector._edit_tab_index)
        window.show()
        application.processEvents()
        history = inspector.edit_history_list
        assert history.count() == 1
        draft = inspector.edit_instruction_draft
        QTest.mouseClick(
            history.viewport(),
            Qt.MouseButton.LeftButton,
            pos=history.visualItemRect(history.item(0)).center(),
        )
        assert inspector.edit_instruction_draft.text == instruction
        assert inspector.edit_instruction_draft.sequence == draft.sequence + 1
        assert controller.document == saved
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
        assert controller.document == saved
        assert len(model.calls) == invocation_count
        assert not workflow.busy
    finally:
        window.close()
        window_workers.shutdown()


def test_restored_edit_branch_survives_duplication_source_deletion_and_further_edits(
    application: QApplication,
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
        for tier in (ResolutionTier.LARGE, ResolutionTier.FULL):
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
        assert controller.document.cards[0].active_revision == first
        assert inspector.edit_history_list.count() == 1
        assert inspector.edit_instruction_edit.toPlainText() == instruction
        _assert_current_image_controls(window, ResolutionTier.LARGE)
        window.redo()
        assert controller.document.cards[0].active_revision == abandoned
        assert inspector.edit_history_list.count() == 2
        assert inspector.edit_instruction_edit.toPlainText() == ""
        _assert_current_image_controls(window, ResolutionTier.FULL)
        window.undo()
        assert controller.document.cards[0].active_revision == first
        assert inspector.edit_instruction_edit.toPlainText() == instruction
        assert abandoned_path.is_file()
        assert session.flush()

        history = inspector.edit_history_list
        history.setCurrentRow(0)
        previous_draft = inspector.edit_instruction_draft
        QTest.keyClick(history, Qt.Key.Key_Return)
        recalled = inspector.edit_instruction_draft
        assert recalled.text == instruction
        assert recalled.sequence == previous_draft.sequence + 1
        window.redo()
        assert inspector.edit_instruction_draft == recalled
        window.undo()
        assert inspector.edit_instruction_draft == recalled
        assert controller.document.cards[0].active_revision == first
        _assert_current_image_controls(window, ResolutionTier.LARGE)
        inspector.edit_background_button.click()
        assert workflow.busy
        _complete_generation(workers)
        branch = controller.document.cards[0].active_revision
        assert branch.id == first.id
        assert branch.provenance.source.background_id == first.background.id
        assert branch.provenance.source.edit_lineage == image_edit_lineage(first.provenance)
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
        assert abandoned_token not in window._edit_undo_changes
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
        assert duplicate_revision.id != branch.id
        assert len(duplicate.revisions) == 1
        assert duplicate_revision.background.id != branch.background.id
        assert duplicate_path != branch_path
        assert duplicate_path.read_bytes() == branch_path.read_bytes()
        assert isinstance(duplicate_revision.provenance, DuplicateProvenance)
        assert duplicate_revision.provenance.original_provenance == branch.provenance
        assert duplicate_revision.provenance.source == ImageSourceSnapshot(
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
        assert duplicate_revision.edit_draft.instruction == ""
        assert (
            duplicate_revision.edit_draft.generation_id
            != branch.edit_draft.generation_id
        )
        original_hotspot = branch.hotspot_set.interactions[0]
        copied_hotspot = duplicate_revision.hotspot_set.interactions[0]
        assert copied_hotspot.id != original_hotspot.id
        assert copied_hotspot.model_copy(update={"id": original_hotspot.id}) == original_hotspot
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
        _assert_current_image_controls(window, ResolutionTier.LARGE)

        inspector.inspector_tabs.setCurrentIndex(inspector._edit_tab_index)
        final_instruction = "Add a blue ribbon beside the stars."
        inspector.edit_instruction_edit.setPlainText(final_instruction)
        inspector.edit_background_button.click()
        assert workflow.busy
        _complete_generation(workers)
        result = controller.document.cards[0].active_revision
        assert len(controller.document.cards[0].revisions) == 1
        assert result.model_copy(update={"background": duplicate_revision.background}) == (
            duplicate_revision
        )
        assert result.provenance.source.background_id == duplicate_revision.background.id
        assert result.provenance.settings.seed == 404
        lineage = image_edit_lineage(result.provenance)
        assert lineage == (
            *image_edit_lineage(branch.provenance),
            result.provenance.accepted_edit,
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
    application: QApplication,
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
            and isinstance(controller.document.cards[0].active_revision.provenance, EditProvenance)
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
        assert change.token in window._edit_undo_changes
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
        assert controller.document.cards[0].active_revision == source
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
    assert (provenance.settings.width, provenance.settings.height) == (512, 384)
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
    assert options[0] == CurrentSourceSize(width=512, height=384)

    workflow.edit(
        card.id,
        instruction="  Open the garden gate.  ",
        output_size=options[0],
    )
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
    assert isinstance(provenance, EditProvenance)
    assert provenance.source == DerivedImageSourceSnapshot(
        card_id=card.id,
        revision_id=source.id,
        background_id=source.background.id,
        width=512,
        height=384,
        seed=source.background.provenance.settings.seed,
        edit_lineage=(),
    )
    assert provenance.instruction == "Open the garden gate."
    assert provenance.output_size == CurrentSourceSize(
        width=512,
        height=384,
    )
    assert provenance.expanded_prompt == (
        "Open the garden gate.\n\n"
        "Unless the Edit Instruction explicitly changes the visual treatment, "
        "keep the result consistent with this selected Style:\n\n"
        f"{style.prompt_text}"
    )
    assert provenance.settings.seed == 8675309
    assert provenance.prompt_token_count == 24
    assert model.calls[-1]["image_paths"] == [snapshot_path]
    assert model.calls[-1]["prompt"] == provenance.expanded_prompt
    assert "image_path" not in model.calls[-1]
    assert "image_strength" not in model.calls[-1]
    assert "description" not in model.calls[-1]
    assert "style" not in model.calls[-1]
    assert "references" not in model.calls[-1]
    assert snapshot_path != source_path
    assert not snapshot_path.exists()
    assert applied[-1].message == "Image edited"
    assert applied[-1].previous_revision == source
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
    assert controller.document.cards[0].active_revision == source
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
            instruction=instruction,
            output_size=current_size,
        )
        _complete_generation(workers)
        accepted_seeds.append(
            controller.document.cards[0].active_revision.background.provenance.settings.seed
        )

    provenance = controller.document.cards[0].active_revision.provenance
    assert isinstance(provenance, EditProvenance)
    assert tuple(edit.instruction for edit in provenance.edit_lineage) == (
        "Open the gate.",
        "Add ivy.",
    )
    assert accepted_seeds == [101, 202]
    assert len(controller.document.cards[0].revisions) == 1


def test_edit_flattens_duplicate_source_provenance(
    tmp_path: Path,
) -> None:
    workflow, controller, _session, workers, _model, card = _bound_workflow(tmp_path)
    workflow.generate(card.id)
    _complete_generation(workers)
    source = controller.document.cards[0].active_revision
    assert source.background is not None
    duplicate_background = source.background.model_copy(
        update={
            "provenance": DuplicateProvenance(
                source=ImageSourceSnapshot(
                    card_id=uuid4(),
                    revision_id=uuid4(),
                    background_id=uuid4(),
                ),
                original_provenance=source.background.provenance,
            )
        }
    )
    controller.execute(
        ReplaceRevisionBackgroundCommand(
            card_id=card.id,
            revision_id=source.id,
            background=duplicate_background,
        )
    )

    workflow.edit(
        card.id,
        instruction="Open the gate.",
        output_size=workflow.available_edit_output_sizes(card.id)[0],
    )
    _complete_generation(workers)
    edited = controller.document.cards[0].active_revision
    assert isinstance(edited.provenance, EditProvenance)
    assert tuple(edit.instruction for edit in edited.provenance.edit_lineage) == ("Open the gate.",)

@pytest.mark.parametrize("aspect_ratio", tuple(AspectRatio))
@pytest.mark.parametrize("resolution", tuple(ResolutionTier))
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
        instruction="Open the gate.",
        output_size=workflow.available_edit_output_sizes(card.id)[0],
    )
    snapshot_path = next((tmp_path / "temporary").glob(".image-source-*.png"))
    Image.new("RGB", (512, 384), "gold").save(source_path, format="PNG")
    _complete_generation(workers)

    assert model.consumed_source_pixels[-1] == (0, 0, 128)
    assert model.calls[-1]["image_paths"] == [snapshot_path]
    assert controller.document.cards[0].revisions == (source,)
    assert "changed while Edit was running" in str(failures[-1])
    assert not snapshot_path.exists()

    workflow.edit(
        card.id,
        instruction="Add ivy.",
        output_size=workflow.available_edit_output_sizes(card.id)[0],
    )
    workers.operations[-1].failed.emit(RuntimeError("model failed"))
    assert controller.document.cards[0].revisions == (source,)
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
        instruction="Open the gate.",
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
        instruction="Add ivy.",
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

    assert controller.document.cards[0].active_revision == styled
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
                instruction="Open the gate.",
                output_size=PresetOutputSize(tier=ResolutionTier.FULL),
            )
        assert not workflow.busy
        assert not list((tmp_path / "temporary").glob(".edit-source-*.png"))
        return

    selected_tier = edit_tiers[0]
    workflow.edit(
        card.id,
        instruction="Open the gate.",
        output_size=PresetOutputSize(tier=selected_tier),
    )
    _complete_generation(workers)

    assert not workflow.busy
    assert not list((tmp_path / "temporary").glob(".edit-source-*.png"))
    provenance = controller.document.cards[0].active_revision.provenance
    assert isinstance(provenance, EditProvenance)
    assert provenance.output_size == PresetOutputSize(tier=selected_tier)
    assert (provenance.source.width, provenance.source.height) == source_size
    assert provenance.source.seed == revision.background.provenance.settings.seed


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
        instruction="Open the gate.",
        output_size=CurrentSourceSize(width=512, height=384),
    )
    _complete_generation(workers)
    result = controller.document.cards[0].active_revision
    result_path = session.store.asset_path(result.background.image_path)

    assert source_path.is_file()
    assert controller.undo()
    assert session.flush()
    assert controller.document.cards[0].revisions == (source,)
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
        instruction="Open the gate.",
        output_size=workflow.available_edit_output_sizes(card.id)[0],
    )
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
    assert applied[0].previous_revision == source.cards[0].active_revision
    assert applied[0].operation == EditImageOperation(instruction="Open the gate.")
    assert applied[0].card_id == card.id
    assert applied[0].revision_id == pending.cards[0].active_revision.id
    assert changed_documents[-1] == pending
    assert controller.undo()
    assert session.flush()
    assert controller.document == source


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
    assert controller.document == before

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
    assert explicit.cards[0].revisions[:2] == before.cards[0].revisions
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
    application: QApplication,
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
    assert window.inspector.edit_instruction_edit.toPlainText() == (
        f"  {instruction}  " if outcome == "observed-after" else ""
    )
    assert len(applied) == (0 if outcome == "observed-after" else 1)
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
        assert window._edit_undo_changes == {change.token: change}
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
        assert controller.document == before
        assert window.inspector.edit_instruction_edit.toPlainText() == (
            instruction if draft_change == "unchanged" else expected_text
        )
        assert session.flush()
        assert store.load() == before
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
    Image.new("RGB", (512, 384), "gold").save(replacement, format="PNG")
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
    _complete_generation(workers)

    assert replaced
    assert controller.document == source
    assert session.store.load() == source
    assert (
        set((session.store.bundle_path / "assets" / "cards").glob("*/image-*.png")) == assets_before
    )
    assert "changed while Edit was running" in str(failures[-1])


def test_generate_can_replace_a_historical_source_with_retained_derived_backgrounds(
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
    Image.new("RGB", (512, 384), "green").save(derived_source)
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
            source=DerivedImageSourceSnapshot(
                card_id=source.id,
                revision_id=source_revision.id,
                background_id=source_background.id,
                width=512,
                height=384,
                seed=source_background.provenance.settings.seed,
                edit_lineage=(),
            ),
            description="A refined source",
            render_prompt="A refined source",
            output_size=CurrentSourceSize(width=512, height=384),
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
    assert provenance.operation == "generate"
    assert provenance.inputs.style is not None
    assert provenance.inputs.style.style_id == style.id
    assert provenance.render_prompt == f"A garden.\n\n{style.prompt_text}"
    assert model.calls[-1]["prompt"] == provenance.render_prompt


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
    assert provenance.operation == "generate"
    assert provenance.inputs.output_size == PresetOutputSize(tier=ResolutionTier.FULL)
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
    Image.new("RGB", (640, 480), "navy").save(
        session.store.asset_path(revision.background.image_path)
    )
    controller.execute(
        SetRevisionGenerateOutputSizeCommand(
            card_id=card.id,
            revision_id=revision.id,
            output_size=ExactOutputSize(width=640, height=480),
        )
    )

    workflow.generate(card.id)
    _complete_generation(workers)

    generated = controller.document.cards[0].active_revision
    assert isinstance(generated.provenance, DirectGenerateProvenance)
    assert generated.provenance.inputs.output_size == ExactOutputSize(
        width=640,
        height=480,
    )
    assert (model.calls[-1]["width"], model.calls[-1]["height"]) == (
        640,
        480,
    )


@pytest.mark.parametrize("aspect_ratio", tuple(AspectRatio))
@pytest.mark.parametrize("resolution", tuple(ResolutionTier))
def test_generate_uses_every_supported_ratio_and_tier(
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
    assert provenance.operation == "generate"
    assert provenance.inputs.output_size == PresetOutputSize(tier=resolution)
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
        SetRevisionGenerateOutputSizeCommand(
            card_id=card.id,
            revision_id=card.active_revision.id,
            output_size=PresetOutputSize(tier=ResolutionTier.LARGE),
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
    manager = StyleManagerWindow(controller, inspector)
    manager.inputs_changed.connect(workflow.cancel)
    manager.show()
    editor = manager.name_edit if field == "name" else manager.prompt_edit
    editor.setFocus()
    application.processEvents()

    workflow.generate(card.id)
    operation = workers.operations[-1]
    work = workers.calls[-1]
    if field == "name":
        manager.name_edit.setText(value)
    else:
        manager.prompt_edit.setPlainText(value)

    assert operation.was_cancelled
    assert controller.document.style_by_id(style.id) == style
    assert controller.document.cards[0].active_revision.background is None
    assert callable(work)
    with pytest.raises(ImageGenerationCancelled, match="cancelled"):
        work()
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
    assert provenance.operation == "generate"
    assert provenance.inputs.style is not None
    assert provenance.inputs.style.name == expected_name
    assert provenance.inputs.style.prompt_text == expected_prompt
    manager.close()


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
