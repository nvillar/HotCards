"""Offscreen integration tests for revision-centric application chrome."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication, QMessageBox

import hypergen.ui.main_window as main_window_module
from hypergen.application.commands import (
    ActivateRevisionCommand,
    DeleteRevisionCommand,
    DuplicateRevisionCommand,
    RenameCardCommand,
    ReplaceRevisionBackgroundCommand,
)
from hypergen.application.document_controller import DocumentController
from hypergen.application.workers import AdapterKind
from hypergen.domain.models import (
    Card,
    CardRevision,
    GenerationStyle,
    ImportedBackground,
    Point,
    Stack,
)
from hypergen.ui.main_window import MainWindow
from hypergen.ui.new_stack_dialog import NewStackDialog


class FakeSettings:
    def __init__(self) -> None:
        self.values: dict[str, Any] = {}

    def value(self, key: str, default_value: Any = None) -> Any:
        return self.values.get(key, default_value)

    def setValue(self, key: str, value: Any) -> None:
        self.values[key] = value

    def sync(self) -> None:
        pass


class FakeOperation(QObject):
    succeeded = Signal(object)
    failed = Signal(object)
    finished = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.cancelled = False

    @property
    def is_finished(self) -> bool:
        return self.cancelled

    def cancel(self) -> None:
        self.cancelled = True
        self.finished.emit()


class FakeWorkers(QObject):
    availability_changed = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.ollama_operations: list[FakeOperation] = []
        self.mflux_operations: list[FakeOperation] = []
        self.shutdown_calls = 0

    def check_ollama(
        self,
        _check: object,
        *,
        emit_diagnostic: bool = True,
    ) -> FakeOperation:
        assert not emit_diagnostic
        operation = FakeOperation()
        self.ollama_operations.append(operation)
        return operation

    def check_mflux(
        self,
        _check: object,
        *,
        emit_diagnostic: bool = True,
    ) -> FakeOperation:
        assert not emit_diagnostic
        operation = FakeOperation()
        self.mflux_operations.append(operation)
        return operation

    def run_ollama(self, _work: object, *, stage: str) -> FakeOperation:
        assert stage in {
            "describing background",
            "enriching Description",
            "remapping hotspots",
        }
        return FakeOperation()

    def shutdown(self, *, wait_milliseconds: int = 0) -> None:
        self.shutdown_calls += 1


class FakeBackgroundWorkflow(QObject):
    busy_changed = Signal(bool)
    progress_changed = Signal(str)
    failed = Signal(object)
    document_changed = Signal(object)
    change_applied = Signal(str, object)

    def __init__(self, controller: DocumentController) -> None:
        super().__init__()
        self.controller = controller
        self.busy = False
        self.generate_calls: list[object] = []
        self.import_calls: list[tuple[object, Path]] = []
        self.clear_calls: list[object] = []
        self.closed = False

    def generate(self, card_id: object) -> None:
        self.generate_calls.append(card_id)

    def import_image(self, card_id: object, source_path: Path, **_kwargs: object) -> None:
        self.import_calls.append((card_id, source_path))

    def clear_background(self, card_id: object) -> None:
        self.clear_calls.append(card_id)

    def activate_revision(self, card_id: object, revision_id: object) -> None:
        changed = self.controller.execute(
            ActivateRevisionCommand(card_id=card_id, revision_id=revision_id)  # type: ignore[arg-type]
        )
        self.document_changed.emit(changed)

    def duplicate_revision(self, card_id: object, revision_id: object) -> None:
        changed = self.controller.execute(
            DuplicateRevisionCommand(  # type: ignore[arg-type]
                card_id=card_id,
                source_revision_id=revision_id,
            )
        )
        self.document_changed.emit(changed)

    def delete_revision(self, card_id: object, revision_id: object) -> None:
        changed = self.controller.execute(
            DeleteRevisionCommand(  # type: ignore[arg-type]
                card_id=card_id,
                revision_id=revision_id,
            )
        )
        self.document_changed.emit(changed)

    def is_generating_for(self, _card_id: object) -> bool:
        return False

    def cancel(self) -> None:
        self.busy = False

    def close(self) -> None:
        self.closed = True


@pytest.fixture(scope="module")
def application() -> QApplication:
    return QApplication.instance() or QApplication([])


def _stack() -> Stack:
    style = GenerationStyle(name="Ink", prompt="Detailed ink illustration")
    first = CardRevision(description="First", style_id=style.id)
    second = CardRevision(description="Second")
    card = Card(
        name="Foyer",
        revisions=(first, second),
        active_revision_id=first.id,
    )
    return Stack(name="Demo", styles=(style,), cards=(card,), start_card_id=card.id)


def _window(
    stack: Stack | None = None,
) -> tuple[MainWindow, DocumentController, FakeWorkers, FakeBackgroundWorkflow]:
    controller = DocumentController(stack or _stack())
    workers = FakeWorkers()
    background = FakeBackgroundWorkflow(controller)
    window = MainWindow(
        controller,
        workers,  # type: ignore[arg-type]
        FakeSettings(),
        availability_checks={
            AdapterKind.OLLAMA: lambda: None,
            AdapterKind.MFLUX: lambda: None,
        },
        background_workflow=background,  # type: ignore[arg-type]
        start_diagnostics=False,
    )
    return window, controller, workers, background


def test_card_header_and_toolbar_match_revision_hierarchy(
    application: QApplication,
) -> None:
    window, _controller, _workers, _background = _window()

    assert window.canvas_card_name.text() == "Foyer"
    assert [window.revision_combo.itemText(index) for index in range(2)] == [
        "1",
        "2",
    ]
    assert window.revision_combo.currentText() == "1"
    assert window.add_revision_button.text() == "+"
    assert window.delete_revision_button.text() == "−"
    assert window.overlay_label.text() == "Hotspots"
    assert window.styles_button.text() == "Styles"
    assert window.inspector.inspector_tabs.tabText(0) == "Background"


def test_card_header_displays_serialized_active_revision(
    application: QApplication,
) -> None:
    stack = _stack()
    card = stack.cards[0]
    stack = stack.model_copy(
        update={
            "cards": (
                card.model_copy(
                    update={"active_revision_id": card.revisions[1].id}
                ),
            )
        }
    )
    serialized_stack = Stack.model_validate_json(stack.model_dump_json())

    window, _controller, _workers, _background = _window(serialized_stack)

    assert window.revision_combo.currentIndex() == 1
    assert window.revision_combo.currentText() == "2"


def test_card_name_edit_uses_controller_and_restores_invalid_value(
    application: QApplication,
) -> None:
    window, controller, _workers, _background = _window()
    card_id = controller.document.cards[0].id

    window.canvas_card_name.setText("Renamed")
    window._commit_canvas_card_name()
    assert controller.document.cards[0].name == "Renamed"
    assert "Renamed" in window.card_sidebar.card_list.item(0).text()
    assert controller.undo()
    window.render_document()
    assert window.canvas_card_name.text() == "Foyer"

    window.canvas_card_name.setText("   ")
    window._commit_canvas_card_name()
    assert controller.document.cards[0].name == "Foyer"
    assert window.canvas_card_name.text() == "Foyer"
    assert card_id == controller.document.cards[0].id


def test_render_preserves_focused_card_name_draft(
    application: QApplication,
) -> None:
    window, controller, _workers, _background = _window()
    card = controller.document.cards[0]
    window.show()
    window.canvas_card_name.setFocus()
    window.canvas_card_name.setText("Uncommitted name")
    application.processEvents()

    changed = controller.execute(
        DuplicateRevisionCommand(
            card_id=card.id,
            source_revision_id=card.active_revision.id,
        )
    )
    window.render_document(changed)

    assert window.canvas_card_name.text() == "Uncommitted name"
    window.close()


def test_revision_selection_duplicate_delete_and_undo(
    application: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window, controller, _workers, _background = _window()
    card = controller.document.cards[0]

    window.revision_combo.setCurrentIndex(1)
    assert controller.document.cards[0].active_revision.description == "Second"

    window.add_revision_button.click()
    changed = controller.document.cards[0]
    assert len(changed.revisions) == 3
    assert changed.active_revision.description == "Second"
    assert window.revision_combo.currentIndex() == 2

    class ConfirmDelete:
        Icon = QMessageBox.Icon
        ButtonRole = QMessageBox.ButtonRole
        StandardButton = QMessageBox.StandardButton

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.delete_button = object()

        def addButton(self, value: object, *_args: object) -> object:
            return self.delete_button if isinstance(value, str) else object()

        def exec(self) -> None:
            pass

        def clickedButton(self) -> object:
            return self.delete_button

    monkeypatch.setattr(main_window_module, "QMessageBox", ConfirmDelete)
    window.delete_revision_button.click()
    assert len(controller.document.cards[0].revisions) == 2
    assert controller.undo()
    window.render_document()
    assert len(controller.document.cards[0].revisions) == 3
    assert card.id == controller.document.cards[0].id


def test_final_revision_cannot_be_deleted(
    application: QApplication,
) -> None:
    card = Card(name="Only")
    window, _controller, _workers, _background = _window(
        Stack(name="Demo", cards=(card,))
    )
    assert not window.delete_revision_button.isEnabled()


def test_author_and_run_modes_apply_consistent_read_only_chrome(
    application: QApplication,
) -> None:
    window, _controller, _workers, _background = _window()

    window.mode_selector.setCurrentText("Run")
    assert window.canvas_card_name.isReadOnly()
    assert not window.revision_combo.isEnabled()
    assert window.add_revision_button.isHidden()
    assert window.delete_revision_button.isHidden()
    assert window.card_sidebar.isHidden()
    assert window.inspector.isHidden()
    assert not window.styles_button.isEnabled()
    assert not window.overlay_selector.isHidden()

    window.mode_selector.setCurrentText("Author")
    assert not window.canvas_card_name.isReadOnly()
    assert window.revision_combo.isEnabled()
    assert not window.add_revision_button.isHidden()
    assert window.styles_button.isEnabled()


def test_generate_replacement_confirms_before_direct_action(
    application: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window, controller, _workers, background = _window()
    window._availability[AdapterKind.MFLUX] = True
    card_id = controller.document.cards[0].id

    window._generate_background()
    assert background.generate_calls == [card_id]

    revision = controller.document.cards[0].active_revision
    asset_id = uuid4()
    controller.execute(
        ReplaceRevisionBackgroundCommand(
            card_id=card_id,
            revision_id=revision.id,
            background=ImportedBackground(
                id=asset_id,
                image_path=f"assets/cards/{card_id}/image-{asset_id}.png",
                source_filename="image.png",
                created_at=datetime.now(UTC),
            ),
        )
    )
    window.render_document()
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *args, **kwargs: QMessageBox.StandardButton.Cancel,
    )
    window._generate_background()
    assert background.generate_calls == [card_id]


def test_notification_undo_expires_after_another_command(
    application: QApplication,
) -> None:
    window, controller, _workers, _background = _window()
    card = controller.document.cards[0]
    controller.execute(RenameCardCommand(card_id=card.id, name="First"))
    token = controller.current_undo_token
    assert token is not None
    window._show_undo_notification("Renamed", token)
    assert not window.inspector.undo_message.isHidden()

    controller.execute(RenameCardCommand(card_id=card.id, name="Second"))
    window.render_document()
    assert window.inspector.undo_message.isHidden()
    window._undo_notification()
    assert controller.document.cards[0].name == "Second"


def test_empty_canvas_request_creates_and_selects_blank_hotspot(
    application: QApplication,
) -> None:
    card = Card(name="Card")
    window, controller, _workers, _background = _window(
        Stack(name="Demo", cards=(card,))
    )

    window._begin_implicit_hotspot_area(Point(x=0.2, y=0.3))

    hotspot_set = controller.document.cards[0].active_revision.hotspot_set
    assert hotspot_set is not None
    assert len(hotspot_set.interactions) == 1
    assert hotspot_set.interactions[0].polygons == ()
    assert window.inspector.selected_interaction_id == hotspot_set.interactions[0].id


def test_new_stack_dialog_has_no_legacy_global_style(
    application: QApplication,
) -> None:
    dialog = NewStackDialog()
    assert not hasattr(dialog, "global_style_edit")
    stack = dialog.stack()
    assert stack.styles == ()
    assert len(stack.cards) == 1
    assert len(stack.cards[0].revisions) == 1


def test_styles_button_opens_stack_manager(
    application: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window, _controller, _workers, _background = _window()
    opened: list[bool] = []

    class FakeStylesDialog:
        def __init__(self, *_args: object) -> None:
            self.document_changed = SignalProxy()

        def exec(self) -> None:
            opened.append(True)

    class SignalProxy:
        def connect(self, _slot: object) -> None:
            pass

    monkeypatch.setattr(main_window_module, "StylesDialog", FakeStylesDialog)
    window.styles_button.click()
    assert opened == [True]


def test_empty_stack_has_clear_first_card_path(
    application: QApplication,
) -> None:
    window, _controller, _workers, _background = _window(Stack(name="Empty"))
    assert window.canvas_pages.currentIndex() == 0
    assert window.empty_canvas_title.text() == "Create your first card"
