"""Offscreen tests for the issue #11 application shell."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PIL import Image
from PySide6.QtCore import QModelIndex, QObject, Qt, Signal
from PySide6.QtGui import QAction, QCloseEvent
from PySide6.QtWidgets import QApplication, QDialog, QLabel

import hypergen.ui.inspector as inspector_module
import hypergen.ui.main_window as main_window_module
from hypergen.application.background_workflow import BackgroundCandidate
from hypergen.application.commands import CreateCardCommand
from hypergen.application.document_controller import DocumentController
from hypergen.application.document_session import DocumentSession
from hypergen.application.workers import (
    AdapterKind,
    WorkerFailure,
    WorkerFailureKind,
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
from hypergen.generation.errors import ModelUnavailableError
from hypergen.main import build_availability_checks, build_main_window
from hypergen.storage.stack_store import StackStore, StackStoreError
from hypergen.ui.main_window import MainWindow
from hypergen.ui.new_stack_dialog import NewStackDialog
from hypergen.ui.settings_dialog import (
    MachineSettings,
    SettingsDialog,
    load_machine_settings,
)


class FakeSettings:
    def __init__(self) -> None:
        self.values: dict[str, Any] = {}
        self.synced = False

    def value(self, key: str, default_value: Any = None) -> Any:
        return self.values.get(key, default_value)

    def setValue(self, key: str, value: Any) -> None:
        self.values[key] = value

    def sync(self) -> None:
        self.synced = True


class FakeOperation(QObject):
    succeeded = Signal(object)
    failed = Signal(object)
    finished = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True
        self.finished.emit()


class FakeWorkers(QObject):
    availability_changed = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.ollama_checks: list[object] = []
        self.mflux_checks: list[object] = []
        self.ollama_operations: list[FakeOperation] = []
        self.mflux_operations: list[FakeOperation] = []
        self.shutdown_calls = 0

    def check_ollama(
        self,
        check: object,
        *,
        emit_diagnostic: bool = True,
    ) -> FakeOperation:
        assert not emit_diagnostic
        self.ollama_checks.append(check)
        operation = FakeOperation()
        self.ollama_operations.append(operation)
        return operation

    def check_mflux(
        self,
        check: object,
        *,
        emit_diagnostic: bool = True,
    ) -> FakeOperation:
        assert not emit_diagnostic
        self.mflux_checks.append(check)
        operation = FakeOperation()
        self.mflux_operations.append(operation)
        return operation

    def shutdown(self, *, wait_milliseconds: int = 0) -> None:
        self.shutdown_calls += 1


class FakeBackgroundWorkflow(QObject):
    candidate_changed = Signal(object)
    busy_changed = Signal(bool)
    progress_changed = Signal(str)
    failed = Signal(object)
    document_changed = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.candidate = None
        self.busy = False
        self.closed = False
        self.generate_calls: list[object] = []

    def close(self) -> None:
        self.closed = True

    def discard_candidate(self) -> None:
        self.candidate = None
        self.candidate_changed.emit(None)

    def is_generating_for(self, _card_id: object) -> bool:
        return False

    def cancel(self) -> None:
        self.busy = False
        self.busy_changed.emit(False)

    def generate(self, card_id: object) -> None:
        self.generate_calls.append(card_id)


@pytest.fixture(scope="module")
def application() -> QApplication:
    return QApplication.instance() or QApplication([])


def loaded_stack() -> Stack:
    hotspot = Interaction(
        label="Door",
        action=NavigateAction(target=UnresolvedCardReference(target_name="Hall")),
        polygons=(
            Polygon(
                points=(
                    Point(x=0.1, y=0.1),
                    Point(x=0.3, y=0.1),
                    Point(x=0.2, y=0.4),
                )
            ),
        ),
    )
    revision = ImageRevision(
        image_path="assets/cards/foyer/background.png",
        origin=ImageOrigin.IMPORTED,
        hotspot_set=HotspotSet(interactions=(hotspot,)),
        created_at=datetime.now(UTC),
    )
    foyer = Card(
        name="Foyer",
        scene_description="A quiet entrance",
        image_revisions=(revision,),
        active_revision_id=revision.id,
    )
    hall = Card(name="Hall")
    return Stack(name="Demo", cards=(foyer, hall), start_card_id=foyer.id)


def make_window(
    stack: Stack | None = None,
    *,
    start_diagnostics: bool = False,
) -> tuple[MainWindow, DocumentController, FakeWorkers, FakeSettings]:
    controller = DocumentController(stack or loaded_stack())
    workers = FakeWorkers()
    settings = FakeSettings()
    background_workflow = FakeBackgroundWorkflow()
    window = MainWindow(
        controller,
        workers,  # type: ignore[arg-type]
        settings,
        availability_checks={
            AdapterKind.OLLAMA: lambda: None,
            AdapterKind.MFLUX: lambda: None,
        },
        background_workflow=background_workflow,  # type: ignore[arg-type]
        start_diagnostics=start_diagnostics,
    )
    return window, controller, workers, settings


def test_three_panes_render_loaded_stack_in_sidebar_and_inspector(
    application: QApplication,
) -> None:
    window, _controller, _workers, _settings = make_window()

    assert window.pane_splitter.count() == 3
    assert window.card_sidebar.card_list.count() == 2
    assert window.card_sidebar.card_list.item(0).text() == "★  Foyer"
    assert window.inspector.card_name_edit.text() == "Foyer"
    assert window.inspector.background_value.text() == "Imported background"
    assert window.inspector.hotspot_list.item(0).text() == "Door (1 area)"
    assert window.card_sidebar.findChild(QObject, "addCardButton") is not None
    window.close()


def test_empty_document_presents_first_card_path(
    application: QApplication,
) -> None:
    window, controller, workers, _settings = make_window(
        Stack(name="Empty"),
        start_diagnostics=True,
    )
    window.show()
    application.processEvents()

    assert window.canvas_pages.currentIndex() == 0
    assert window.inspector.pages.currentIndex() == 0
    assert window.card_sidebar.empty_label.isVisible()
    workers.ollama_operations[0].succeeded.emit(None)
    workers.mflux_operations[0].succeeded.emit(None)
    assert not window.inspector.generate_background_button.isEnabled()
    assert not window.inspector.import_background_button.isEnabled()
    window.create_first_card_button.click()

    assert [card.name for card in controller.document.cards] == ["Card 1"]
    assert window.card_sidebar.selected_card_id == controller.document.cards[0].id
    assert window.inspector.selected_card_id == controller.document.cards[0].id
    assert window.inspector.pages.currentIndex() == 1
    assert window.canvas_pages.currentIndex() == 1
    assert not window.inspector.generate_background_button.isEnabled()
    assert "Scene or Style" in window.inspector.generate_background_button.toolTip()
    assert window.inspector.import_background_button.isEnabled()
    window.close()


def test_sidebar_actions_fit_at_minimum_width(application: QApplication) -> None:
    window, _controller, _workers, _settings = make_window()
    window.show()
    window.pane_splitter.setSizes([180, 760, 240])
    application.processEvents()

    sidebar_right = window.card_sidebar.contentsRect().right()
    for button in (
        window.card_sidebar.add_button,
        window.card_sidebar.start_button,
        window.card_sidebar.move_up_button,
        window.card_sidebar.move_down_button,
        window.card_sidebar.delete_button,
    ):
        assert button.width() > 0
        assert button.geometry().right() <= sidebar_right
    window.close()


def test_new_stack_dialog_builds_initial_saved_shape(application: QApplication) -> None:
    dialog = NewStackDialog()
    dialog.name_edit.setText("Garden")
    dialog.global_style_edit.setPlainText("Pencil sketch")
    dialog.width_spin.setValue(1280)
    dialog.height_spin.setValue(720)

    stack = dialog.stack()

    assert stack.name == "Garden"
    assert stack.global_style == "Pencil sketch"
    assert (stack.canvas.width, stack.canvas.height) == (1280, 720)
    assert [card.name for card in stack.cards] == ["Card 1"]
    assert stack.start_card_id == stack.cards[0].id
    dialog.close()


def test_bound_new_stack_flow_enables_editing_and_persists(
    application: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = DocumentController(Stack(name="Welcome"))
    session = DocumentSession(controller)
    workers = FakeWorkers()
    window = MainWindow(
        controller,
        workers,  # type: ignore[arg-type]
        FakeSettings(),
        document_session=session,
        start_diagnostics=False,
    )
    created = Stack(name="Garden", global_style="Pencil sketch", cards=(Card(name="Card 1"),))

    class AcceptedNewStackDialog:
        def __init__(self, _parent: object) -> None:
            pass

        def exec(self) -> QDialog.DialogCode:
            return QDialog.DialogCode.Accepted

        def stack(self) -> Stack:
            return created

    bundle = tmp_path / "Garden.hypergen"
    monkeypatch.setattr(main_window_module, "NewStackDialog", AcceptedNewStackDialog)
    monkeypatch.setattr(
        main_window_module.QFileDialog,
        "getSaveFileName",
        lambda *_args, **_kwargs: (str(bundle), "HyperGen Stack (*.hypergen)"),
    )

    assert not window.card_sidebar.add_button.isEnabled()
    assert window.create_first_card_button.text() == "Create New Stack"
    window.new_stack()

    assert session.state.bundle_path == bundle
    assert session.store is not None
    assert session.store.load() == created
    assert window.card_sidebar.add_button.isEnabled()
    assert window.card_sidebar.card_list.count() == 1
    window.close()


def test_close_is_cancelled_when_pending_autosave_fails(
    application: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = DocumentController(Stack(name="Welcome"))
    session = DocumentSession(controller)
    session.create(Stack(name="Saved"), tmp_path / "Saved.hypergen")
    workers = FakeWorkers()
    window = MainWindow(
        controller,
        workers,  # type: ignore[arg-type]
        FakeSettings(),
        document_session=session,
        start_diagnostics=False,
    )
    controller.execute(CreateCardCommand(name="Pending"))
    assert session.store is not None
    real_save = session.store.save

    def fail_save(_stack: Stack) -> None:
        raise StackStoreError("disk full")

    monkeypatch.setattr(session.store, "save", fail_save)
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "warning",
        lambda *_args, **_kwargs: main_window_module.QMessageBox.StandardButton.Cancel,
    )
    event = QCloseEvent()
    window.closeEvent(event)

    assert not event.isAccepted()
    assert session.state.dirty
    assert session.state.error == "disk full"

    monkeypatch.setattr(session.store, "save", real_save)
    assert session.flush()
    window.close()


def test_close_commits_focused_inspector_edits_before_flush(
    application: QApplication,
    tmp_path: Path,
) -> None:
    card = Card(name="Card")
    controller = DocumentController(Stack(name="Saved", cards=(card,)))
    session = DocumentSession(controller)
    bundle = tmp_path / "Saved.hypergen"
    session.create(controller.document, bundle)
    workers = FakeWorkers()
    window = MainWindow(
        controller,
        workers,  # type: ignore[arg-type]
        FakeSettings(),
        document_session=session,
        start_diagnostics=False,
    )
    window.inspector.scene_edit.setPlainText("Committed during close")

    window.close()

    assert StackStore(bundle).load().cards[0].scene_description == "Committed during close"


def test_undo_redo_actions_and_confirmed_delete_use_controller(
    application: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window, controller, _workers, _settings = make_window()
    created_id = window.card_sidebar.add_card("Library")
    assert window.undo_action.isEnabled()

    window.undo_action.trigger()
    assert [card.name for card in controller.document.cards] == ["Foyer", "Hall"]
    assert window.redo_action.isEnabled()
    window.redo_action.trigger()
    assert [card.name for card in controller.document.cards] == ["Foyer", "Hall", "Library"]

    monkeypatch.setattr(window, "_ask_delete_card", lambda _message: True)
    window._confirm_delete_card(created_id)
    assert [card.name for card in controller.document.cards] == ["Foyer", "Hall"]
    window.close()


def test_add_rename_and_start_card_mutations_use_controller(
    application: QApplication,
) -> None:
    window, controller, _workers, _settings = make_window()

    created_id = window.card_sidebar.add_card("Library")
    assert [card.name for card in controller.document.cards] == ["Foyer", "Hall", "Library"]
    assert window.card_sidebar.selected_card_id == created_id
    assert window.inspector.selected_card_id == created_id
    window.inspector.card_name_edit.setText("Archive")
    window.inspector.commit_card_metadata()
    assert controller.document.cards[-1].name == "Archive"
    window.inspector.start_card_check.setChecked(True)
    assert controller.document.start_card_id == created_id
    assert window.card_sidebar.card_list.item(2).text() == "★  Archive"
    window.close()


def test_drag_reorder_is_one_undoable_controller_command(
    application: QApplication,
) -> None:
    base = loaded_stack()
    stack = base.model_copy(update={"cards": (*base.cards, Card(name="Tower"))})
    window, controller, _workers, _settings = make_window(stack)

    assert window.card_sidebar.card_list.model().moveRow(
        QModelIndex(),
        0,
        QModelIndex(),
        3,
    )
    assert [card.name for card in controller.document.cards] == ["Hall", "Tower", "Foyer"]
    assert controller.undo()
    assert [card.name for card in controller.document.cards] == ["Foyer", "Hall", "Tower"]
    window.close()


def test_invalid_card_name_is_rejected_and_inspector_is_restored(
    application: QApplication,
) -> None:
    window, controller, _workers, _settings = make_window()

    window.inspector.card_name_edit.clear()
    window.inspector.commit_card_metadata()

    assert controller.document.cards[0].name == "Foyer"
    assert window.inspector.card_name_edit.text() == "Foyer"
    assert not window.inspector.validation_error.isHidden()
    window.close()


def test_inspector_uses_pipeline_tabs_and_groups_fields_by_stage(
    application: QApplication,
) -> None:
    window, _controller, _workers, _settings = make_window()

    tabs = window.inspector.inspector_tabs
    assert tabs.count() == 3
    assert [tabs.tabText(index) for index in range(3)] == [
        "Card",
        "Background",
        "Interactivity (1)",
    ]
    assert tabs.indexOf(window.inspector.scene_edit.parentWidget()) == 0
    assert tabs.indexOf(window.inspector.style_edit.parentWidget()) == 1
    assert tabs.indexOf(window.inspector.interactions_edit.parentWidget()) == 2
    assert not window.inspector.enrich_scene_button.isEnabled()
    assert not window.inspector.hotspot_help_button.toolTip() == ""
    window.close()


def test_background_style_switches_between_global_and_card_override(
    application: QApplication,
) -> None:
    card = Card(name="Garden", scene_description="A garden")
    window, controller, _workers, _settings = make_window(
        Stack(name="Demo", global_style="Watercolor", cards=(card,))
    )

    assert window.inspector.global_style_radio.isChecked()
    assert window.inspector.style_edit.toPlainText() == "Watercolor"
    assert controller.document.cards[0].card_style is None

    window.inspector.card_style_radio.click()
    assert controller.document.cards[0].card_style == "Watercolor"
    window.inspector.style_edit.setPlainText("Photographic night scene")
    window.inspector.style_edit.editing_finished.emit()
    assert controller.document.cards[0].card_style == "Photographic night scene"
    assert controller.document.global_style == "Watercolor"

    window.inspector.global_style_radio.click()
    assert controller.document.cards[0].card_style is None
    assert window.inspector.style_edit.toPlainText() == "Watercolor"
    window.inspector.style_edit.setPlainText("Ink wash")
    window.inspector.style_edit.editing_finished.emit()
    assert controller.document.global_style == "Ink wash"
    assert controller.undo()
    assert controller.document.global_style == "Watercolor"
    window.close()


def test_global_style_enables_generation_when_scene_is_only_whitespace(
    application: QApplication,
) -> None:
    card = Card(name="Garden", scene_description="   ")
    window, _controller, _workers, _settings = make_window(
        Stack(name="Demo", global_style="Watercolor", cards=(card,))
    )
    window._availability[AdapterKind.MFLUX] = True
    window._update_generation_actions()

    assert window.inspector.generate_background_button.isEnabled()
    window.close()


def test_pending_style_text_enables_generation_before_focus_changes(
    application: QApplication,
) -> None:
    card = Card(name="Garden")
    window, controller, _workers, _settings = make_window(
        Stack(name="Demo", cards=(card,))
    )
    window._availability[AdapterKind.MFLUX] = True
    window._update_generation_actions()
    assert not window.inspector.generate_background_button.isEnabled()

    window.inspector.style_edit.setFocus()
    window.inspector.style_edit.setPlainText("Watercolor")
    application.processEvents()

    assert window.inspector.generate_background_button.isEnabled()
    assert controller.document.global_style == ""
    window.inspector.commit_card_metadata()
    assert controller.document.global_style == "Watercolor"
    window.close()


def test_generation_aborts_when_pending_metadata_is_invalid(
    application: QApplication,
) -> None:
    card = Card(name="Garden")
    window, controller, _workers, _settings = make_window(
        Stack(name="Demo", cards=(card,))
    )
    workflow = window.background_workflow
    assert isinstance(workflow, FakeBackgroundWorkflow)
    window._availability[AdapterKind.MFLUX] = True
    window.inspector.scene_edit.setPlainText("New scene")
    window.inspector.card_name_edit.clear()
    application.processEvents()

    assert window.inspector.generate_background_button.isEnabled()
    window.inspector.generate_background_button.click()

    assert workflow.generate_calls == []
    assert controller.document.cards[0].scene_description == ""
    assert not window.inspector.validation_error.isHidden()
    window.close()


def test_background_candidate_is_contextual_and_previews_on_canvas(
    application: QApplication,
    tmp_path: Path,
) -> None:
    window, controller, _workers, _settings = make_window()
    workflow = window.background_workflow
    assert isinstance(workflow, FakeBackgroundWorkflow)
    image_path = tmp_path / "candidate.png"
    Image.new("RGB", (1024, 768), "navy").save(image_path)
    candidate = BackgroundCandidate(
        card_id=controller.document.cards[0].id,
        revision_id=uuid4(),
        image_path=image_path,
        origin=ImageOrigin.IMPORTED,
        source_filename="candidate.png",
        created_at=datetime.now(UTC),
    )

    workflow.candidate = candidate
    workflow.candidate_changed.emit(candidate)

    assert not window.inspector.candidate_widget.isHidden()
    assert window.inspector.inspector_tabs.tabText(1) == "Background ●"
    assert not window.inspector.generate_background_button.isEnabled()
    assert not window.inspector.import_background_button.isEnabled()
    assert window.card_canvas._border_item is not None
    assert window.card_canvas._border_item.pen().style() == Qt.PenStyle.DashLine

    workflow.candidate = None
    workflow.candidate_changed.emit(None)
    assert window.inspector.candidate_widget.isHidden()
    window.close()


def test_import_and_apply_background_through_contextual_inspector(
    application: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    card = Card(name="Card")
    controller = DocumentController(
        Stack(name="Saved", cards=(card,), start_card_id=card.id)
    )
    session = DocumentSession(controller)
    bundle = tmp_path / "Saved.hypergen"
    session.create(controller.document, bundle)
    source = tmp_path / "source.png"
    Image.new("RGB", (1024, 768), "green").save(source)
    workers = FakeWorkers()
    window = MainWindow(
        controller,
        workers,  # type: ignore[arg-type]
        FakeSettings(),
        document_session=session,
        start_diagnostics=False,
    )
    monkeypatch.setattr(
        main_window_module.QFileDialog,
        "getOpenFileName",
        lambda *_args, **_kwargs: (str(source), "Images (*.png)"),
    )

    window.inspector.import_background_button.click()
    assert window.background_workflow is not None
    assert window.background_workflow.candidate is not None
    assert not window.inspector.candidate_widget.isHidden()

    window.inspector.apply_background_button.click()
    assert window.background_workflow.candidate is None
    revision = controller.document.cards[0].image_revisions[0]
    assert revision.source_filename == "source.png"
    assert window.inspector.hotspots_placeholder.text() == "No hotspots yet."
    assert not window.inspector.hotspots_placeholder.isHidden()
    assert session.flush()
    assert session.store is not None
    assert session.store.asset_path(revision.image_path).is_file()
    window.close()


def test_manual_hotspot_canvas_commands_are_undoable(
    application: QApplication,
) -> None:
    window, controller, _workers, _settings = make_window()
    source = controller.document.cards[0]
    revision = source.image_revisions[0]
    assert revision.hotspot_set is not None
    original = revision.hotspot_set.interactions[0]
    added_polygon = Polygon(
        points=(
            Point(x=0.55, y=0.2),
            Point(x=0.85, y=0.2),
            Point(x=0.7, y=0.55),
        )
    )

    assert window.inspector.add_hotspot_button.isEnabled()
    window.card_canvas.polygon_created.emit(None, added_polygon)
    hotspot_set = controller.document.cards[0].image_revisions[0].hotspot_set
    assert hotspot_set is not None
    assert len(hotspot_set.interactions) == 2
    created = hotspot_set.interactions[-1]
    assert created.label == "Hotspot 2"
    assert window.inspector.selected_interaction_id == created.id
    window.inspector.move_hotspot_up_button.click()
    hotspot_set = controller.document.cards[0].image_revisions[0].hotspot_set
    assert hotspot_set is not None
    assert [interaction.id for interaction in hotspot_set.interactions] == [
        created.id,
        original.id,
    ]

    window.card_canvas.polygon_created.emit(created.id, original.polygons[0])
    hotspot_set = controller.document.cards[0].image_revisions[0].hotspot_set
    assert hotspot_set is not None
    edited = next(
        interaction
        for interaction in hotspot_set.interactions
        if interaction.id == created.id
    )
    assert len(edited.polygons) == 2

    window.card_canvas.polygon_deletion_requested.emit(created.id, 1)
    hotspot_set = controller.document.cards[0].image_revisions[0].hotspot_set
    assert hotspot_set is not None
    edited = next(
        interaction
        for interaction in hotspot_set.interactions
        if interaction.id == created.id
    )
    assert len(edited.polygons) == 1
    assert controller.undo()
    window.render_document()
    hotspot_set = controller.document.cards[0].image_revisions[0].hotspot_set
    assert hotspot_set is not None
    edited = next(
        interaction
        for interaction in hotspot_set.interactions
        if interaction.id == created.id
    )
    assert len(edited.polygons) == 2
    window.close()


def test_manual_hotspot_properties_and_destination_actions(
    application: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window, controller, _workers, _settings = make_window()
    source = controller.document.cards[0]
    destination = controller.document.cards[1]
    interaction_id = window.inspector.selected_interaction_id
    assert interaction_id is not None

    window.inspector.hotspot_label_edit.setText("Archway")
    window.inspector.hotspot_label_edit.editingFinished.emit()
    hotspot_set = controller.document.cards[0].image_revisions[0].hotspot_set
    assert hotspot_set is not None
    assert hotspot_set.interactions[0].label == "Archway"

    destination_index = window.inspector.hotspot_destination_combo.findData(
        destination.id
    )
    window.inspector.hotspot_destination_combo.setCurrentIndex(destination_index)
    window.inspector.hotspot_destination_combo.activated.emit(destination_index)
    application.processEvents()
    hotspot_set = controller.document.cards[0].image_revisions[0].hotspot_set
    assert hotspot_set is not None
    assert hotspot_set.interactions[0].action.target == ResolvedCardReference(
        target_card_id=destination.id
    )

    window.inspector.hotspot_destination_combo.setCurrentIndex(0)
    window.inspector.hotspot_destination_combo.activated.emit(0)
    application.processEvents()
    hotspot_set = controller.document.cards[0].image_revisions[0].hotspot_set
    assert hotspot_set is not None
    assert hotspot_set.interactions[0].action.target == UnresolvedCardReference()

    monkeypatch.setattr(
        inspector_module.QInputDialog,
        "getText",
        lambda *_args, **_kwargs: ("Ignored", False),
    )
    create_index = window.inspector.hotspot_destination_combo.findData("create")
    window.inspector.hotspot_destination_combo.setCurrentIndex(create_index)
    window.inspector.hotspot_destination_combo.activated.emit(create_index)
    application.processEvents()
    assert len(controller.document.cards) == 2

    monkeypatch.setattr(
        inspector_module.QInputDialog,
        "getText",
        lambda *_args, **_kwargs: ("New Room", True),
    )
    create_index = window.inspector.hotspot_destination_combo.findData("create")
    window.inspector.hotspot_destination_combo.setCurrentIndex(create_index)
    window.inspector.hotspot_destination_combo.activated.emit(create_index)
    application.processEvents()

    assert [card.name for card in controller.document.cards] == [
        source.name,
        destination.name,
        "New Room",
    ]
    hotspot_set = controller.document.cards[0].image_revisions[0].hotspot_set
    assert hotspot_set is not None
    target = hotspot_set.interactions[0].action.target
    assert isinstance(target, ResolvedCardReference)
    assert target.target_card_id == controller.document.cards[-1].id
    assert controller.undo()
    assert len(controller.document.cards) == 2
    window.close()


def test_existing_destination_survives_card_and_hotspot_reselection(
    application: QApplication,
) -> None:
    window, controller, _workers, _settings = make_window()
    source = controller.document.cards[0]
    destination = controller.document.cards[1]
    destination_index = window.inspector.hotspot_destination_combo.findData(
        destination.id
    )

    window.inspector.hotspot_destination_combo.showPopup()
    window.inspector.hotspot_destination_combo.setCurrentIndex(destination_index)
    window.inspector.hotspot_destination_combo.activated.emit(destination_index)
    window.inspector.hotspot_destination_combo.hidePopup()
    application.processEvents()
    window.select_card(destination.id)
    window.select_card(source.id)

    assert window._selected_card_id == source.id
    assert window.inspector.hotspot_list.count() == 1
    hotspot_set = controller.document.cards[0].image_revisions[0].hotspot_set
    assert hotspot_set is not None
    assert hotspot_set.interactions[0].action.target == ResolvedCardReference(
        target_card_id=destination.id
    )
    assert (
        window.inspector.hotspot_destination_combo.currentData()
        == destination.id
    )
    window.close()


def test_existing_destination_survives_bundle_flush_and_reopen(
    application: QApplication,
    tmp_path: Path,
) -> None:
    source_image = tmp_path / "source.png"
    Image.new("RGB", (1024, 768), "green").save(source_image)
    source = Card(name="Source")
    destination = Card(name="Destination")
    revision_id = uuid4()
    bundle = tmp_path / "Destination.hypergen"
    store = StackStore(bundle)
    image_path = store.import_image(
        source_image,
        card_id=source.id,
        revision_id=revision_id,
    )
    hotspot = Interaction(
        label="Door",
        action=NavigateAction(target=UnresolvedCardReference()),
        polygons=(
            Polygon(
                points=(
                    Point(x=0.1, y=0.1),
                    Point(x=0.4, y=0.1),
                    Point(x=0.2, y=0.5),
                )
            ),
        ),
    )
    revision = ImageRevision(
        id=revision_id,
        image_path=image_path,
        origin=ImageOrigin.IMPORTED,
        hotspot_set=HotspotSet(interactions=(hotspot,)),
        created_at=datetime.now(UTC),
    )
    source = source.model_copy(
        update={
            "image_revisions": (revision,),
            "active_revision_id": revision.id,
        }
    )
    stack = Stack(name="Saved", cards=(source, destination))
    store.save(stack)
    controller = DocumentController(Stack(name="Bootstrap"))
    session = DocumentSession(controller)
    session.open(bundle)
    window = MainWindow(
        controller,
        FakeWorkers(),  # type: ignore[arg-type]
        FakeSettings(),
        document_session=session,
        background_workflow=FakeBackgroundWorkflow(),  # type: ignore[arg-type]
        start_diagnostics=False,
    )
    destination_index = next(
        index
        for index in range(window.inspector.hotspot_destination_combo.count())
        if window.inspector.hotspot_destination_combo.itemData(index)
        == destination.id
    )

    window.inspector.hotspot_destination_combo.setCurrentIndex(destination_index)
    window.inspector.hotspot_destination_combo.activated.emit(destination_index)
    application.processEvents()
    assert session.state.dirty
    assert session.flush()
    session.open(bundle)
    window.select_card(source.id)

    saved = StackStore(bundle).load()
    saved_hotspots = saved.cards[0].image_revisions[0].hotspot_set
    assert saved_hotspots is not None
    assert saved_hotspots.interactions[0].action.target == ResolvedCardReference(
        target_card_id=destination.id
    )
    assert (
        window.inspector.hotspot_destination_combo.currentData()
        == destination.id
    )
    window.close()


def test_canvas_deselection_keeps_inspector_and_canvas_in_sync(
    application: QApplication,
) -> None:
    window, _controller, _workers, _settings = make_window()
    assert window.inspector.selected_interaction_id is not None

    window.card_canvas.interaction_selected.emit(None)

    assert window.inspector.selected_interaction_id is None
    assert window.card_canvas._selected_interaction_id is None
    assert not window.inspector.hotspot_label_edit.isEnabled()
    window.close()


def test_settings_round_trip_excludes_credentials_and_stack(
    application: QApplication,
) -> None:
    settings = FakeSettings()
    assert load_machine_settings(settings) == MachineSettings()
    stack = loaded_stack()
    before = stack.model_dump_json()
    dialog = SettingsDialog(settings)
    dialog.ollama_endpoint_edit.setText("http://example.test:11434")
    dialog.ollama_model_edit.setText("local-model:latest")
    dialog.mflux_model_edit.setText("flux2-klein-9b")
    dialog.step_count_spin.setValue(12)
    dialog.quantization_combo.setCurrentIndex(dialog.quantization_combo.findData(8))
    dialog.random_seed_check.setChecked(False)
    dialog.fixed_seed_spin.setValue(8675309)
    dialog.save()

    values = load_machine_settings(settings)
    assert values.ollama_endpoint == "http://example.test:11434"
    assert values.ollama_model == "local-model:latest"
    assert values.mflux_model == "flux2-klein-9b"
    assert values.step_count == 12
    assert values.quantization == 8
    assert not values.random_seed
    assert values.fixed_seed == 8675309
    assert settings.synced
    assert all(
        forbidden not in key.casefold()
        for key in settings.values
        for forbidden in ("credential", "token", "password", "hugging")
    )
    assert stack.model_dump_json() == before
    assert "never asks for or stores" in dialog.findChild(
        QObject, "externalAuthenticationNote"
    ).property("text")
    dialog.ollama_endpoint_edit.setText("http://user:secret@example.test:11434")
    with pytest.raises(ValueError, match="without credentials"):
        dialog.save()
    assert settings.values["services/ollama_endpoint"] == "http://example.test:11434"
    dialog.close()


def test_toolbar_switches_between_authoring_and_run_controls(
    application: QApplication,
) -> None:
    window, controller, _workers, _settings = make_window()

    assert [window.mode_selector.itemText(index) for index in range(2)] == [
        "Author",
        "Run",
    ]
    assert window.service_status_label.text() == "AI services not checked"
    assert not window.check_services_button.isHidden()
    before = controller.document
    window.mode_selector.setCurrentText("Run")
    assert controller.document == before
    assert [action.text() for action in window.player_navigation_actions] == [
        "Back",
        "Restart",
    ]
    assert not window.back_action.isEnabled()
    assert window.restart_action.isEnabled()
    assert window.back_action.isVisible()
    assert window.restart_action.isVisible()
    assert window.card_sidebar.isHidden()
    assert window.inspector.isHidden()
    assert not window.overlay_selector.isHidden()
    assert window.check_services_button.isHidden()
    assert "unavailable" in window.run_status_label.text()
    window.overlay_selector.setCurrentText("Visible")
    assert controller.document.run_overlay_mode.value == "visible"
    window.mode_selector.setCurrentText("Author")
    assert not window.card_sidebar.isHidden()
    assert not window.inspector.isHidden()
    assert window.overlay_selector.isHidden()
    window.close()


def test_run_mode_blocks_empty_document_mutation_and_cancels_ai_work(
    application: QApplication,
) -> None:
    window, controller, workers, _settings = make_window(
        Stack(name="Empty"),
        start_diagnostics=True,
    )
    workflow = window.background_workflow
    assert isinstance(workflow, FakeBackgroundWorkflow)
    workflow.busy = True

    window.mode_selector.setCurrentText("Run")
    window._primary_empty_action()

    assert controller.document.cards == ()
    assert window.create_first_card_button.isHidden()
    assert window.empty_canvas_title.text() == "No cards to run"
    assert all(
        operation.cancelled
        for operation in (
            *workers.ollama_operations,
            *workers.mflux_operations,
        )
    )
    assert not workflow.busy
    assert window.run_status_label.text() == "This stack has no cards to run."
    window.close()


def test_service_diagnostics_toggle_only_generation_actions(
    application: QApplication,
) -> None:
    window, _controller, workers, _settings = make_window(start_diagnostics=True)
    assert len(workers.ollama_checks) == 1
    assert len(workers.mflux_checks) == 1
    assert not window.inspector.generate_background_button.isEnabled()
    assert window.inspector.import_background_button.isEnabled()
    assert window.card_sidebar.add_button.isEnabled()

    unavailable = WorkerFailure(
        adapter=AdapterKind.MFLUX,
        stage="checking MFLUX model availability",
        kind=WorkerFailureKind.MODEL_UNAVAILABLE,
        message="MFLUX model is unavailable",
    )
    workers.mflux_operations[0].failed.emit(unavailable)
    workers.ollama_operations[0].succeeded.emit(None)
    assert not window.inspector.generate_background_button.isEnabled()
    assert window.inspector.import_background_button.isEnabled()
    assert window.card_sidebar.add_button.isEnabled()

    workers.mflux_operations[0].succeeded.emit(None)
    assert window.inspector.generate_background_button.isEnabled()
    window.close()


def test_service_rechecks_use_latest_completion_and_reenable_actions(
    application: QApplication,
) -> None:
    window, _controller, workers, _settings = make_window(start_diagnostics=True)
    old_ollama = workers.ollama_operations[0]
    old_mflux = workers.mflux_operations[0]
    old_ollama.succeeded.emit(None)
    old_mflux.succeeded.emit(None)
    assert window.inspector.generate_background_button.isEnabled()

    window.run_availability_checks()
    assert not window.inspector.generate_background_button.isEnabled()
    assert window.service_status_label.text() == "Checking local AI services…"
    assert "is available" not in window.service_status_label.text()
    workers.ollama_operations[1].succeeded.emit(None)
    workers.mflux_operations[1].succeeded.emit(None)
    assert window.inspector.generate_background_button.isEnabled()

    stale_failure = WorkerFailure(
        adapter=AdapterKind.MFLUX,
        stage="old check",
        kind=WorkerFailureKind.MODEL_UNAVAILABLE,
        message="stale failure",
    )
    old_mflux.failed.emit(stale_failure)
    assert window.inspector.generate_background_button.isEnabled()
    window.close()


def test_status_summary_is_bounded_and_preserves_full_diagnostic(
    application: QApplication,
) -> None:
    window, _controller, workers, _settings = make_window(start_diagnostics=True)
    detail = "MFLUX model is unavailable because " + ("a runtime artifact is missing; " * 20)
    failure = WorkerFailure(
        adapter=AdapterKind.MFLUX,
        stage="checking MFLUX model availability",
        kind=WorkerFailureKind.MODEL_UNAVAILABLE,
        message=detail,
    )
    workers.mflux_operations[0].failed.emit(failure)
    workers.ollama_operations[0].succeeded.emit(None)

    assert window.service_status_label.text() == "MFLUX unavailable"
    assert len(window.service_status_label.text()) < 40
    assert detail in window.service_status_label.toolTip()
    assert not window.review_settings_button.isHidden()
    window.close()


def test_shell_copy_does_not_expose_issue_roadmap(application: QApplication) -> None:
    window, _controller, _workers, _settings = make_window()

    visible_copy = " ".join(label.text() for label in window.findChildren(QLabel))
    action_tooltips = " ".join(
        action.toolTip()
        for action in window.player_navigation_actions
    )
    assert "issue #" not in f"{visible_copy} {action_tooltips}".casefold()
    toolbar_action_names = {
        action.objectName() for action in window.findChildren(QAction)
    }
    assert "generateBackgroundAction" not in toolbar_action_names
    assert "generateHotspotsAction" not in toolbar_action_names
    window.close()


def test_availability_checks_capture_settings_before_worker_execution() -> None:
    settings = FakeSettings()
    settings.values["services/ollama_endpoint"] = "http://localhost:11434"
    checks = build_availability_checks(settings)
    settings.values["services/ollama_endpoint"] = "http://changed.invalid"

    assert checks[AdapterKind.OLLAMA].__closure__ is not None
    captured = [
        cell.cell_contents
        for cell in checks[AdapterKind.OLLAMA].__closure__
        if isinstance(cell.cell_contents, MachineSettings)
    ]
    assert captured[0].ollama_endpoint == "http://localhost:11434"


def test_mflux_check_uses_runtime_artifact_patterns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import huggingface_hub

    calls: list[dict[str, Any]] = []

    def incomplete_but_usable_snapshot(**kwargs: Any) -> str:
        calls.append(kwargs)
        if "allow_patterns" not in kwargs:
            raise AssertionError("A full repository snapshot would reject optional missing files")
        return "/cached/flux2-klein"

    monkeypatch.setattr(huggingface_hub, "snapshot_download", incomplete_but_usable_snapshot)
    checks = build_availability_checks(FakeSettings())
    checks[AdapterKind.MFLUX]()

    assert calls[0]["local_files_only"] is True
    assert set(calls[0]["allow_patterns"]) >= {
        "vae/*.safetensors",
        "transformer/*.safetensors",
        "text_encoder/*.safetensors",
        "tokenizer/**",
    }


def test_mflux_check_reports_missing_runtime_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import huggingface_hub
    from huggingface_hub.errors import LocalEntryNotFoundError

    def missing_snapshot(**_kwargs: Any) -> str:
        raise LocalEntryNotFoundError("required model files are missing")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", missing_snapshot)
    checks = build_availability_checks(FakeSettings())

    with pytest.raises(ModelUnavailableError, match="not available in the local"):
        checks[AdapterKind.MFLUX]()


def test_bootstrap_construction_uses_injected_services_without_live_clients(
    application: QApplication,
) -> None:
    controller = DocumentController(Stack(name="Injected"))
    workers = FakeWorkers()
    window = build_main_window(
        controller=controller,
        workers=workers,  # type: ignore[arg-type]
        settings=FakeSettings(),
    )

    assert window.controller is controller
    assert window.workers is workers
    assert not workers.ollama_checks
    assert not workers.mflux_checks
    assert window.windowTitle() == "HyperGen — Injected"
    window.close()


def test_default_bootstrap_requires_bound_stack_before_editing(
    application: QApplication,
) -> None:
    workers = FakeWorkers()
    window = build_main_window(
        workers=workers,  # type: ignore[arg-type]
        settings=FakeSettings(),
    )

    assert window.document_session is not None
    assert window.document_session.store is None
    assert not window.card_sidebar.add_button.isEnabled()
    assert window.create_first_card_button.text() == "Create New Stack"
    assert window.new_stack_action.isEnabled()
    assert window.open_stack_action.isEnabled()
    assert not window.save_action.isEnabled()
    assert not window.save_as_action.isEnabled()
    window.close()
