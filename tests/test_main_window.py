"""Offscreen integration tests for revision-centric application chrome."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QObject, QSize, Qt, Signal
from PySide6.QtGui import QCloseEvent, QColor, QPixmap
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLabel

import hotcards.ui.main_window as main_window_module
from hotcards.application.commands import (
    ActivateRevisionCommand,
    DeleteRevisionCommand,
    DuplicateRevisionCommand,
    RenameCardCommand,
    ReplaceRevisionBackgroundCommand,
    SetRevisionImagePromptCommand,
)
from hotcards.application.document_controller import DocumentController
from hotcards.application.document_session import DocumentSession, DocumentSessionState
from hotcards.application.generated_revision_change import GeneratedRevisionChange
from hotcards.application.workers import AdapterKind, AvailabilityDiagnostic
from hotcards.domain.models import (
    HYPERCARD_STYLE_ID,
    Card,
    CardRevision,
    GeneratedBackground,
    HotspotConditions,
    HotspotKeyChanges,
    HotspotSet,
    ImageGenerationInputs,
    ImageGenerationMetadata,
    ImagePrompt,
    Interaction,
    KeyDefinition,
    NavigateAction,
    Point,
    Polygon,
    Stack,
    UnresolvedCardReference,
)
from hotcards.ui.card_sidebar import CardSidebar
from hotcards.ui.main_window import MainWindow
from hotcards.ui.new_stack_dialog import NewStackDialog


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
        assert stage == "preparing Image Prompt"
        return FakeOperation()

    def shutdown(self, *, wait_milliseconds: int = 0) -> None:
        self.shutdown_calls += 1


class FakeBackgroundWorkflow(QObject):
    busy_changed = Signal(bool)
    progress_changed = Signal(str)
    generation_progress_changed = Signal(int, int)
    failed = Signal(object)
    document_changed = Signal(object)
    change_applied = Signal(str, object)
    generation_applied = Signal(object)

    def __init__(self, controller: DocumentController) -> None:
        super().__init__()
        self.controller = controller
        self.busy = False
        self.generate_calls: list[object] = []
        self.clear_calls: list[object] = []
        self.cancel_calls = 0
        self.closed = False

    def generate(self, card_id: object) -> None:
        self.generate_calls.append(card_id)

    def clear_background(self, card_id: object) -> None:
        self.clear_calls.append(card_id)

    def activate_revision(self, card_id: object, revision_id: object) -> None:
        changed = self.controller.execute(
            ActivateRevisionCommand(card_id=card_id, revision_id=revision_id)  # type: ignore[arg-type]
        )
        self.document_changed.emit(changed)

    def duplicate_revision(self, card_id: object, revision_id: object) -> None:
        previous_token = self.controller.current_undo_token
        changed = self.controller.execute(
            DuplicateRevisionCommand(  # type: ignore[arg-type]
                card_id=card_id,
                source_revision_id=revision_id,
            )
        )
        self.document_changed.emit(changed)
        token = self.controller.current_undo_token
        if token is not None and token != previous_token:
            self.change_applied.emit("Revision duplicated", token)

    def delete_revision(self, card_id: object, revision_id: object) -> None:
        previous_token = self.controller.current_undo_token
        changed = self.controller.execute(
            DeleteRevisionCommand(  # type: ignore[arg-type]
                card_id=card_id,
                revision_id=revision_id,
            )
        )
        self.document_changed.emit(changed)
        token = self.controller.current_undo_token
        if token is not None and token != previous_token:
            self.change_applied.emit("Revision deleted", token)

    def is_generating_for(self, _card_id: object) -> bool:
        return self.busy

    def cancel(self) -> None:
        self.cancel_calls += 1
        self.busy = False

    def close(self) -> None:
        self.closed = True


@pytest.fixture(scope="module")
def application() -> QApplication:
    return QApplication.instance() or QApplication([])


def _stack() -> Stack:
    first = CardRevision(
        description="First",
        image_prompt=ImagePrompt(text="First prompt", source_description="First"),
    )
    second = CardRevision(
        description="Second",
        image_prompt=ImagePrompt(text="Second prompt", source_description="Second"),
    )
    card = Card(
        name="Foyer",
        revisions=(first, second),
        active_revision_id=first.id,
    )
    return Stack(name="Demo", cards=(card,), start_card_id=card.id)


def _generated_background(
    *,
    asset_id: UUID,
    image_path: str,
    description: str = "Test image",
) -> GeneratedBackground:
    generated_at = datetime.now(UTC)
    return GeneratedBackground(
        id=asset_id,
        image_path=image_path,
        generation_metadata=ImageGenerationMetadata(
            inputs=ImageGenerationInputs(
                description=description,
                image_prompt=description,
            ),
            render_prompt=description,
            model_identifier="test",
            mflux_version="test",
            seed=1,
            width=1024,
            height=768,
            step_count=4,
            generated_at=generated_at,
            duration_seconds=1,
        ),
        created_at=generated_at,
    )


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
    assert window.revision_label.text() == "Version:"
    assert window.revision_combo.width() < 100


def test_hotspot_rule_editor_scrolls_without_growing_the_window(
    application: QApplication,
) -> None:
    keys = tuple(KeyDefinition(name=f"Key {number}") for number in range(6))
    interaction = Interaction(
        conditions=HotspotConditions(
            requires=tuple(key.id for key in keys[:3]),
        ),
        key_changes=HotspotKeyChanges(
            grant=tuple(key.id for key in keys[3:]),
        ),
    )
    card = Card(
        name="Card",
        revisions=(
            CardRevision(hotspot_set=HotspotSet(interactions=(interaction,))),
        ),
    )
    window, _controller, _workers, _background = _window(
        Stack(name="Demo", keys=keys, cards=(card,))
    )
    window.inspector.inspector_tabs.setCurrentIndex(
        window.inspector._hotspots_tab_index
    )
    window.resize(1180, 760)
    window.show()
    application.processEvents()
    try:
        assert window.minimumSizeHint().height() <= 760
        assert window.height() == 760
        assert window.inspector.hotspot_rule_scroll.verticalScrollBar().maximum() > 0
    finally:
        window.close()
        application.processEvents()
    assert window.revision_combo.width() >= (
        window.revision_combo.fontMetrics().horizontalAdvance("0000")
    )
    assert window.add_revision_button.text() == "+"
    assert window.delete_revision_button.text() == "−"
    assert window.card_header.indexOf(window.delete_revision_button) < (
        window.card_header.indexOf(window.add_revision_button)
    )
    assert window.overlay_label.text() == "Hotspots"
    assert window.toolbar_leading_spacer.width() == 8
    assert window.mode_button.text() == "Run"
    assert window.mode_button.toolTip() == "Switch to Run mode"
    toolbar_actions = window.authoring_toolbar.actions()
    assert toolbar_actions.index(window.back_button_action) < toolbar_actions.index(
        window.restart_button_action
    )
    assert toolbar_actions.index(window.restart_button_action) < toolbar_actions.index(
        window.run_overlay_separator
    )
    assert toolbar_actions.index(window.run_overlay_separator) < toolbar_actions.index(
        window.overlay_label_action
    )
    assert toolbar_actions.index(window.overlay_label_action) < (
        toolbar_actions.index(window.overlay_selector_action)
    )
    assert window.back_button.font().pointSizeF() == (
        window.mode_button.font().pointSizeF()
    )
    assert window.restart_button.font().pointSizeF() == (
        window.mode_button.font().pointSizeF()
    )
    assert window.back_button.sizeHint().height() >= (
        window.mode_button.sizeHint().height()
    )
    assert window.restart_button.sizeHint().height() >= (
        window.mode_button.sizeHint().height()
    )
    assert not window.run_controls_separator.isVisible()
    assert not window.run_overlay_separator.isVisible()
    assert not window.overlay_label_action.isVisible()
    assert not window.overlay_selector_action.isVisible()
    assert not hasattr(window, "styles_button")
    assert not hasattr(window, "document_status_label")
    assert window.generation_progress_container.isHidden()
    margins = window.generation_progress_layout.contentsMargins()
    assert margins.left() == 8
    assert margins.right() == 8
    central_layout = window.centralWidget().layout()
    assert central_layout is not None
    assert central_layout.indexOf(window.pane_splitter) < (
        central_layout.indexOf(window.notification_bar)
    )
    assert window.inspector.inspector_tabs.tabText(0) == "Image"
    assert window.fit_canvas_button.size() == window.clear_background_button.size()


def test_card_browser_uses_thumbnails_and_compact_action_row(
    application: QApplication,
    tmp_path: Path,
) -> None:
    thumbnail_path = tmp_path / "thumbnail.png"
    thumbnail = QPixmap(144, 96)
    thumbnail.fill(QColor("red"))
    assert thumbnail.save(str(thumbnail_path))
    asset_id = uuid4()
    revision = CardRevision(
        background=_generated_background(
            asset_id=asset_id,
            image_path=f"assets/cards/card/image-{asset_id}.png",
        )
    )
    card = Card(name="Preview", revisions=(revision,))
    controller = DocumentController(
        Stack(name="Demo", cards=(card,), start_card_id=card.id)
    )
    sidebar = CardSidebar(
        controller,
        image_path_resolver=lambda _path: thumbnail_path,
    )

    assert not any(
        label.text() == "Cards" for label in sidebar.findChildren(QLabel)
    )
    item = sidebar.card_list.item(0)
    icon = item.icon().pixmap(QSize(72, 48))
    assert icon.size() == QSize(72, 48)
    assert icon.toImage().pixelColor(10, 10) == QColor("red")
    assert not sidebar.start_button.icon().isNull()
    assert sidebar.start_button.toolTip() == (
        "Make the selected card the start card"
    )
    assert sidebar.add_button.toolTip() == "Add a new card"
    assert sidebar.delete_button.toolTip() == "Delete the selected card"
    assert sidebar.card_actions.indexOf(sidebar.move_up_button) == 0
    assert sidebar.card_actions.indexOf(sidebar.move_down_button) == 1
    assert sidebar.card_actions.indexOf(sidebar.start_button) == 3
    assert sidebar.card_actions.indexOf(sidebar.delete_button) == 4
    assert sidebar.card_actions.indexOf(sidebar.add_button) == 5
    control_sizes = {
        button.size()
        for button in (
            sidebar.move_up_button,
            sidebar.move_down_button,
            sidebar.start_button,
            sidebar.add_button,
            sidebar.delete_button,
        )
    }
    assert len(control_sizes) == 1
    assert sidebar.add_button.font().pointSizeF() > (
        sidebar.move_up_button.font().pointSizeF()
    )


def test_selecting_scrolled_card_survives_focus_out_render(
    application: QApplication,
) -> None:
    cards = tuple(Card(name=f"Card {number}") for number in range(20))
    window, _controller, _workers, _background = _window(
        Stack(name="Demo", cards=cards)
    )
    window.resize(900, 500)
    window.show()
    application.processEvents()
    try:
        window.canvas_card_name.setFocus()
        window.canvas_card_name.setText("Renamed Card")
        card_list = window.card_sidebar.card_list
        scroll_bar = card_list.verticalScrollBar()
        scroll_bar.setValue(scroll_bar.maximum())
        application.processEvents()
        target = cards[-2]
        target_item = card_list.item(len(cards) - 2)
        target_position = card_list.visualItemRect(target_item).center()
        assert card_list.viewport().rect().contains(target_position)
        scroll_position = scroll_bar.value()
        assert scroll_position > scroll_bar.minimum()

        QTest.mouseClick(
            card_list.viewport(),
            Qt.MouseButton.LeftButton,
            pos=target_position,
        )
        application.processEvents()

        assert window.controller.document.cards[0].name == "Renamed Card"
        assert window.card_sidebar.selected_card_id == target.id
        assert window._selected_card_id == target.id
        assert scroll_bar.value() == scroll_position
    finally:
        window.close()
        application.processEvents()


def test_card_header_displays_serialized_active_revision(
    application: QApplication,
) -> None:
    stack = _stack()
    card = stack.cards[0]
    stack = stack.model_copy(
        update={"cards": (card.model_copy(update={"active_revision_id": card.revisions[1].id}),)}
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
    assert not window.canvas_card_name_error.isHidden()
    assert not hasattr(window.inspector, "validation_error")
    assert card_id == controller.document.cards[0].id


def test_card_name_validation_error_does_not_follow_card_selection(
    application: QApplication,
) -> None:
    first = Card(name="First")
    second = Card(name="Second")
    window, _controller, _workers, _background = _window(Stack(name="Demo", cards=(first, second)))
    window.canvas_card_name.setText("   ")
    window._commit_canvas_card_name()
    assert not window.canvas_card_name_error.isHidden()

    window.select_card(second.id)

    assert window.canvas_card_name.text() == "Second"
    assert window.canvas_card_name_error.isHidden()


def test_focus_triggered_name_failure_aborts_run_transition(
    application: QApplication,
) -> None:
    window, _controller, _workers, _background = _window()
    window.show()
    window.canvas_card_name.setFocus()
    window.canvas_card_name.setText("   ")
    application.processEvents()

    window.mode_button.setFocus()
    application.processEvents()
    assert not window.canvas_card_name_error.isHidden()
    window.mode_button.click()

    assert window.mode_button.text() == "Run"
    assert not window._is_running
    window.close()


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
) -> None:
    window, controller, _workers, background = _window()
    card = controller.document.cards[0]

    window.revision_combo.setCurrentIndex(1)
    assert controller.document.cards[0].active_revision.description == "Second"

    window.add_revision_button.click()
    changed = controller.document.cards[0]
    assert len(changed.revisions) == 3
    assert changed.active_revision.description == "Second"
    assert window.revision_combo.currentIndex() == 2

    background.busy = True
    window.delete_revision_button.click()
    assert len(controller.document.cards[0].revisions) == 2
    assert background.cancel_calls == 1
    assert window.notification_bar.message_label.text() == "Revision deleted"
    window.notification_bar.primary_button.click()
    assert len(controller.document.cards[0].revisions) == 3
    assert card.id == controller.document.cards[0].id


def test_final_revision_cannot_be_deleted(
    application: QApplication,
) -> None:
    card = Card(name="Only")
    window, _controller, _workers, _background = _window(Stack(name="Demo", cards=(card,)))
    assert not window.delete_revision_button.isEnabled()


def test_card_delete_applies_immediately_and_offers_targeted_undo(
    application: QApplication,
) -> None:
    window, controller, _workers, _background = _window()

    window.card_sidebar.delete_button.click()
    assert controller.document.cards == ()
    assert window.notification_bar.message_label.text() == (
        "Card deleted; the start card was cleared"
    )

    window.notification_bar.primary_button.click()
    assert len(controller.document.cards) == 1
    assert controller.document.start_card_id == controller.document.cards[0].id


def test_card_delete_ignores_destinationless_hotspots(
    application: QApplication,
) -> None:
    destination = Card(name="Destination")
    source = Card(
        name="Source",
        revisions=(
            CardRevision(
                hotspot_set=HotspotSet(interactions=(Interaction(),))
            ),
        ),
    )
    window, controller, _workers, _background = _window(
        Stack(
            name="Demo",
            cards=(source, destination),
            start_card_id=source.id,
        )
    )

    window._delete_card(destination.id)

    assert [card.id for card in controller.document.cards] == [source.id]


def test_context_change_cancels_background_generation_without_prompt(
    application: QApplication,
) -> None:
    window, _controller, _workers, background = _window()
    background.busy = True

    window._cancel_background_generation()
    assert background.cancel_calls == 1
    assert not background.busy


def test_successful_save_as_cancels_generation_and_expires_undo(
    application: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    controller = DocumentController(_stack())
    session = DocumentSession(controller)
    session.create(controller.document, tmp_path / "Original.hotcards")
    workers = FakeWorkers()
    background = FakeBackgroundWorkflow(controller)
    background.busy = True
    window = MainWindow(
        controller,
        workers,  # type: ignore[arg-type]
        FakeSettings(),
        document_session=session,
        background_workflow=background,  # type: ignore[arg-type]
        start_diagnostics=False,
    )
    card = controller.document.cards[0]
    changed = controller.execute(RenameCardCommand(card_id=card.id, name="Renamed"))
    window.render_document(changed)
    token = controller.current_undo_token
    assert token is not None
    window._show_undo_notification("Renamed", token)
    monkeypatch.setattr(
        main_window_module.QFileDialog,
        "getSaveFileName",
        lambda *_args, **_kwargs: (
            str(tmp_path / "Copy.hotcards"),
            "HotCards Stack (*.hotcards)",
        ),
    )
    prompt_cancellations: list[bool] = []
    monkeypatch.setattr(
        window.image_prompt_workflow,
        "cancel",
        lambda: prompt_cancellations.append(True),
    )

    window.save_as()

    assert background.cancel_calls == 1
    assert prompt_cancellations == [True]
    assert window.notification_bar.current_key != "undo"
    assert not controller.can_undo


def test_failed_authoring_commit_blocks_save_and_close(
    application: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window, _controller, _workers, background = _window()
    flush_calls: list[bool] = []
    window.document_session = SimpleNamespace(
        store=object(),
        flush=lambda: flush_calls.append(True) or True,
    )
    monkeypatch.setattr(window, "_commit_authoring_metadata", lambda: False)

    assert not window.save_document()
    assert flush_calls == []

    event = QCloseEvent()
    window.closeEvent(event)
    assert not event.isAccepted()
    assert not background.closed


def test_failed_style_focus_commit_blocks_save(
    application: QApplication,
) -> None:
    window, controller, _workers, _background = _window()
    flush_calls: list[bool] = []
    window.document_session = SimpleNamespace(
        store=object(),
        flush=lambda: flush_calls.append(True) or True,
    )
    style = controller.document.styles[0]
    window.inspector.inspector_tabs.setCurrentIndex(
        window.inspector._styles_tab_index
    )
    window.inspector.style_name_edit.setFocus()
    window.inspector.style_name_edit.setText("")

    window.inspector._style_editing_finished(
        window,
        Qt.FocusReason.MouseFocusReason,
    )

    assert window.inspector.style_name_edit.text() == ""
    assert not window.save_document()
    assert flush_calls == []
    assert controller.document.style_by_id(style.id) == style


def test_author_and_run_modes_apply_consistent_read_only_chrome(
    application: QApplication,
) -> None:
    window, _controller, _workers, _background = _window()
    window.show()
    application.processEvents()

    window.mode_button.click()
    application.processEvents()
    assert window.mode_button.text() == "Author"
    assert window.mode_button.toolTip() == "Switch to Author mode"
    assert window.canvas_card_name.isReadOnly()
    assert window.canvas_card_name.isHidden()
    assert window.canvas_card_name_error.isHidden()
    assert window.revision_label.isHidden()
    assert window.revision_combo.isHidden()
    assert not window.revision_combo.isEnabled()
    assert window.add_revision_button.isHidden()
    assert window.delete_revision_button.isHidden()
    assert window.card_sidebar.isHidden()
    assert window.inspector.isHidden()
    assert window.run_controls_separator.isVisible()
    assert window.run_overlay_separator.isVisible()
    assert window.back_action.isVisible()
    assert window.restart_action.isVisible()
    assert window.back_button_action.isVisible()
    assert window.restart_button_action.isVisible()
    assert window.back_button.isVisible()
    assert window.restart_button.isVisible()
    assert window.overlay_label_action.isVisible()
    assert window.overlay_selector_action.isVisible()
    assert not window.overlay_selector.isHidden()
    assert window.llm_model_label.isHidden()
    assert window.llm_model_combo.isHidden()
    assert window.image_model_label.isHidden()
    assert window.image_model_combo.isHidden()

    window.mode_button.click()
    application.processEvents()
    assert window.mode_button.text() == "Run"
    assert window.mode_button.toolTip() == "Switch to Run mode"
    assert not window.canvas_card_name.isReadOnly()
    assert not window.canvas_card_name.isHidden()
    assert not window.revision_label.isHidden()
    assert not window.revision_combo.isHidden()
    assert window.revision_combo.isEnabled()
    assert not window.add_revision_button.isHidden()
    assert not window.llm_model_label.isHidden()
    assert not window.llm_model_combo.isHidden()
    assert not window.image_model_label.isHidden()
    assert not window.image_model_combo.isHidden()
    assert not window.run_controls_separator.isVisible()
    assert not window.run_overlay_separator.isVisible()
    assert not window.back_action.isVisible()
    assert not window.restart_action.isVisible()
    assert not window.back_button_action.isVisible()
    assert not window.restart_button_action.isVisible()
    assert window.back_button.isHidden()
    assert window.restart_button.isHidden()
    assert not window.overlay_label_action.isVisible()
    assert not window.overlay_selector_action.isVisible()
    window.hide()
    application.processEvents()


def test_status_bar_is_passive_and_ai_recovery_uses_notification_bar(
    application: QApplication,
) -> None:
    window, controller, _workers, _background = _window()
    window.apply_availability_diagnostic(
        AvailabilityDiagnostic(
            adapter=AdapterKind.OLLAMA,
            available=False,
            message="Ollama is unavailable",
        )
    )
    window.apply_availability_diagnostic(
        AvailabilityDiagnostic(
            adapter=AdapterKind.MFLUX,
            available=True,
            message="MFLUX is available",
        )
    )

    assert not hasattr(window, "check_services_button")
    assert not hasattr(window, "review_settings_button")
    assert not hasattr(window, "service_status_label")
    assert "Ollama is unavailable" in window.llm_model_combo.toolTip()
    assert window.notification_bar.current_key == "ai-services"
    assert window.notification_bar.primary_button.text() == "Settings"
    assert window.notification_bar.secondary_button.text() == "Check Again"

    window.notification_bar.dismiss_current()
    window.apply_availability_diagnostic(
        AvailabilityDiagnostic(
            adapter=AdapterKind.OLLAMA,
            available=True,
            message="Ollama is available",
        )
    )
    window.apply_availability_diagnostic(
        AvailabilityDiagnostic(
            adapter=AdapterKind.OLLAMA,
            available=False,
            message="Ollama is unavailable again",
        )
    )
    assert window.notification_bar.current_key == "ai-services"

    card = controller.document.cards[0]
    changed = controller.execute(RenameCardCommand(card_id=card.id, name="Renamed"))
    window.render_document(changed)
    token = controller.current_undo_token
    assert token is not None
    window._show_undo_notification("Renamed", token)
    assert window.notification_bar.current_key == "undo"


def test_bottom_model_selectors_persist_and_follow_operation_state(
    application: QApplication,
) -> None:
    window, _controller, _workers, background = _window()
    settings = window.settings
    assert isinstance(settings, FakeSettings)

    window._availability_check_succeeded(
        AdapterKind.OLLAMA,
        window._diagnostic_generation,
        ("llama3.2:latest", "qwen3.5:9b-mlx"),
    )
    assert [
        window.llm_model_combo.itemData(index)
        for index in range(window.llm_model_combo.count())
    ] == ["llama3.2:latest", "qwen3.5:9b-mlx"]
    window.llm_model_combo.setCurrentIndex(
        window.llm_model_combo.findData("llama3.2:latest")
    )
    assert settings.values["services/ollama_model"] == "llama3.2:latest"

    window.image_model_combo.setCurrentIndex(
        window.image_model_combo.findData("flux2-klein-9b-kv")
    )
    assert settings.values["generation/mflux_model"] == "flux2-klein-9b-kv"
    assert window.image_model_combo.currentText() == "FLUX.2 Klein 9B KV"
    assert window.image_model_combo.toolTip() == window.llm_model_combo.toolTip()

    background.busy = True
    window.image_prompt_workflow._busy = False
    window._update_generation_actions()
    assert not window.llm_model_combo.isEnabled()
    assert not window.image_model_combo.isEnabled()

    window._availability_check_succeeded(
        AdapterKind.OLLAMA,
        window._diagnostic_generation,
        ("qwen3.5:9b-mlx",),
    )
    assert background.cancel_calls == 1
    assert settings.values["services/ollama_model"] == "qwen3.5:9b-mlx"

    window._availability_check_succeeded(
        AdapterKind.OLLAMA,
        window._diagnostic_generation,
        ("llama3.2:latest", "qwen3.5:9b-mlx"),
    )
    background.busy = True
    window._llm_model_changed(
        window.llm_model_combo.findData("llama3.2:latest")
    )
    assert background.cancel_calls == 2
    assert settings.values["services/ollama_model"] == "llama3.2:latest"

    background.busy = False
    window.image_prompt_workflow._busy = False
    window.mode_button.click()
    assert not window.llm_model_combo.isEnabled()
    assert not window.image_model_combo.isEnabled()


def test_generation_failure_keeps_current_image_prompt_reusable(
    application: QApplication,
) -> None:
    revision = CardRevision(
        description="A courtyard",
        image_prompt=ImagePrompt(
            text="A prepared courtyard",
            source_description="A courtyard",
            model_identifier="qwen3.5:9b-mlx",
            prompt_version=main_window_module.IMAGE_PROMPT_PREPARATION_VERSION,
        ),
    )
    card = Card(name="Card", revisions=(revision,))
    window, _controller, _workers, background = _window(
        Stack(name="Demo", cards=(card,))
    )
    window._availability[AdapterKind.OLLAMA] = True
    window._availability[AdapterKind.MFLUX] = True
    window._update_generation_actions()
    assert window.inspector.has_current_image_prompt()
    assert window.inspector.generate_background_button.isEnabled()

    background.busy = True
    window._update_generation_actions()
    background.busy = False
    background.failed.emit(RuntimeError("transient MFLUX failure"))

    assert window.inspector.has_current_image_prompt()
    assert window.inspector.enrich_button.text() == "Image Prompt Current"
    assert window.inspector.generate_background_button.isEnabled()


def test_manual_image_prompt_edit_reenables_generation_after_description_change(
    application: QApplication,
) -> None:
    revision = CardRevision(
        description="A courtyard",
        image_prompt=ImagePrompt(
            text="A prepared courtyard",
            source_description="A courtyard",
            model_identifier="qwen3.5:9b-mlx",
            prompt_version=main_window_module.IMAGE_PROMPT_PREPARATION_VERSION,
        ),
    )
    card = Card(name="Card", revisions=(revision,))
    window, controller, _workers, background = _window(Stack(name="Demo", cards=(card,)))
    window._availability[AdapterKind.OLLAMA] = True
    window._availability[AdapterKind.MFLUX] = True
    window._update_generation_actions()

    window.inspector.description_edit.setPlainText("A moonlit courtyard")
    assert window.inspector.commit_revision_metadata()
    assert not window.inspector.generate_background_button.isEnabled()

    window.inspector.image_prompt_button.click()
    window.inspector.description_edit.setPlainText("A manually revised moonlit courtyard")

    assert window.inspector.generate_background_button.isEnabled()
    window._generate_background()

    assert background.generate_calls == [card.id]
    image_prompt = controller.document.cards[0].active_revision.image_prompt
    assert image_prompt is not None
    assert image_prompt.text == "A manually revised moonlit courtyard"
    assert image_prompt.source_description == "A moonlit courtyard"
    assert image_prompt.model_identifier == "qwen3.5:9b-mlx"
    assert image_prompt.prompt_version == (main_window_module.IMAGE_PROMPT_PREPARATION_VERSION)
    window.close()


def test_ollama_selector_disables_when_no_vision_model_is_installed(
    application: QApplication,
) -> None:
    window, _controller, _workers, _background = _window()

    window._availability_check_succeeded(
        AdapterKind.OLLAMA,
        window._diagnostic_generation,
        (),
    )

    assert window.llm_model_combo.count() == 1
    assert window.llm_model_combo.currentData() is None
    assert window.llm_model_combo.currentText() == (
        "No vision-capable models installed"
    )
    assert not window.llm_model_combo.isEnabled()
    assert not window.inspector.enrich_button.isEnabled()


def test_changing_llm_model_re_enables_image_prompt_preparation(
    application: QApplication,
) -> None:
    revision = CardRevision(
        description="A courtyard",
        image_prompt=ImagePrompt(
            text="A richly detailed courtyard",
            source_description="A courtyard",
            model_identifier="qwen3.5:9b-mlx",
            prompt_version=main_window_module.IMAGE_PROMPT_PREPARATION_VERSION,
        ),
    )
    card = Card(name="Card", revisions=(revision,))
    window, _controller, _workers, _background = _window(
        Stack(name="Demo", cards=(card,))
    )
    models = ("llama3.2:latest", "qwen3.5:9b-mlx")
    window._availability_check_succeeded(
        AdapterKind.OLLAMA,
        window._diagnostic_generation,
        models,
    )
    assert window.inspector.enrich_button.text() == (
        "Image Prompt Current"
    )
    assert not window.inspector.enrich_button.isEnabled()

    window.llm_model_combo.setCurrentIndex(
        window.llm_model_combo.findData("llama3.2:latest")
    )
    assert window.inspector.enrich_button.text() == (
        "Update Image Prompt"
    )
    assert not window.inspector.enrich_button.isEnabled()

    window._availability_check_succeeded(
        AdapterKind.OLLAMA,
        window._diagnostic_generation,
        models,
    )

    assert window.inspector.enrich_button.text() == (
        "Update Image Prompt"
    )
    assert window.inspector.enrich_button.isEnabled()


def test_generate_replacement_starts_without_a_second_confirmation(
    application: QApplication,
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
            background=_generated_background(
                asset_id=asset_id,
                image_path=f"assets/cards/{card_id}/image-{asset_id}.png",
            ),
        )
    )
    window.render_document()
    assert window.inspector.generate_background_button.text() == (
        "Re-generate Image"
    )
    assert window.clear_background_button.isEnabled()
    assert window.clear_background_button.text() == ""
    assert not window.clear_background_button.icon().isNull()
    assert window.canvas_fit_controls.indexOf(window.clear_background_button) == (
        window.canvas_fit_controls.indexOf(window.fit_canvas_button) + 1
    )
    window.clear_background_button.click()
    assert background.clear_calls == [card_id]

    window._generate_background()
    assert background.generate_calls == [card_id, card_id]


def test_generate_is_disabled_without_description_even_with_image_prompt(
    application: QApplication,
) -> None:
    revision = CardRevision(
        image_prompt=ImagePrompt(
            text="A richly detailed courtyard",
            source_description="",
        ),
    )
    card = Card(name="Card", revisions=(revision,))
    window, _controller, _workers, background = _window(
        Stack(name="Demo", cards=(card,))
    )
    window._availability[AdapterKind.MFLUX] = True
    window._update_generation_actions()

    assert not window.inspector.generate_background_button.isEnabled()


def test_notification_undo_expires_after_another_command(
    application: QApplication,
) -> None:
    window, controller, _workers, _background = _window()
    card = controller.document.cards[0]
    controller.execute(RenameCardCommand(card_id=card.id, name="First"))
    token = controller.current_undo_token
    assert token is not None
    window._show_undo_notification("Renamed", token)
    assert not window.notification_bar.isHidden()
    assert window.notification_bar.message_label.text() == "Renamed"

    controller.execute(RenameCardCommand(card_id=card.id, name="Second"))
    window.render_document()
    assert window.notification_bar.isHidden()
    window._undo_notification()
    assert controller.document.cards[0].name == "Second"


def test_re_enrich_click_is_not_consumed_by_description_commit(
    application: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    revision = CardRevision(
        description="A courtyard",
        image_prompt=ImagePrompt(
            text="A richly detailed courtyard",
            source_description="A courtyard",
        ),
    )
    card = Card(name="Card", revisions=(revision,))
    window, _controller, _workers, _background = _window(
        Stack(name="Demo", cards=(card,))
    )
    window.show()
    window._availability[AdapterKind.OLLAMA] = True
    window._update_generation_actions()
    window.inspector.description_button.click()
    window.inspector.description_edit.setPlainText("A changed courtyard")
    window.inspector.description_edit.setFocus()
    application.processEvents()
    starts: list[object] = []
    monkeypatch.setattr(
        window.image_prompt_workflow,
        "start",
        starts.append,
    )

    QTest.mousePress(
        window.inspector.enrich_button,
        Qt.MouseButton.LeftButton,
    )
    application.processEvents()
    assert starts == []
    assert _controller.document.cards[0].active_revision.description == (
        "A changed courtyard"
    )
    QTest.mouseRelease(
        window.inspector.enrich_button,
        Qt.MouseButton.LeftButton,
    )

    assert starts == [card.id]
    window.close()


@pytest.mark.parametrize("mode", ["description", "image_prompt"])
def test_mouse_focus_commit_does_not_render_before_button_release(
    application: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    revision = CardRevision(
        description="A courtyard",
        image_prompt=ImagePrompt(
            text="A detailed courtyard",
            source_description="A courtyard",
        ),
    )
    card = Card(name="Card", revisions=(revision,))
    window, controller, _workers, _background = _window(
        Stack(name="Demo", cards=(card,))
    )
    window.show()
    if mode == "image_prompt":
        window.inspector.image_prompt_button.click()
    draft = f"A changed {mode}"
    window.inspector.description_edit.setPlainText(draft)
    window.inspector.description_edit.setFocus()
    application.processEvents()
    renders: list[object] = []
    monkeypatch.setattr(
        window.inspector,
        "render",
        lambda *_args: renders.append(object()),
    )

    window.inspector.description_edit.editing_finished.emit(
        None,
        Qt.FocusReason.MouseFocusReason,
    )

    changed_revision = controller.document.cards[0].active_revision
    if mode == "description":
        assert changed_revision.description == draft
    else:
        assert changed_revision.image_prompt is not None
        assert changed_revision.image_prompt.text == draft
    assert renders == []
    window.close()


def test_generated_result_can_move_to_a_new_complete_version(
    application: QApplication,
) -> None:
    hotspot_set = HotspotSet(
        interactions=(
            Interaction(
                label="Door",
                action=NavigateAction(target=UnresolvedCardReference()),
            ),
        )
    )
    original = CardRevision(
        description="A courtyard",
        hotspot_set=hotspot_set,
    )
    card = Card(name="Card", revisions=(original,))
    window, controller, _workers, _background = _window(
        Stack(name="Demo", cards=(card,))
    )
    changed = controller.execute(
        SetRevisionImagePromptCommand(
            card_id=card.id,
            revision_id=original.id,
            value=ImagePrompt(
                text="A richly detailed courtyard",
                source_description=original.description,
            ),
        )
    )
    window.render_document(changed)
    token = controller.current_undo_token
    assert token is not None
    window._show_generated_revision_notification(
        GeneratedRevisionChange(
            message="Image Prompt prepared",
            token=token,
            card_id=card.id,
            revision_id=original.id,
            previous_revision=original,
        )
    )

    assert window.notification_bar.message_label.text() == (
        "Image Prompt prepared on the current version"
    )
    assert window.notification_bar.primary_button.text() == "Create New Version"
    assert window.notification_bar.secondary_button.text() == "Undo"
    assert window.notification_bar.dismiss_button.text() == "Keep"
    window.notification_bar.primary_button.click()

    changed_card = controller.document.cards[0]
    assert len(changed_card.revisions) == 2
    assert changed_card.revisions[0].id == original.id
    assert changed_card.revisions[0].image_prompt is None
    assert changed_card.revisions[0].hotspot_set == original.hotspot_set
    assert changed_card.active_revision.id != original.id
    assert changed_card.active_revision.description == original.description
    assert changed_card.active_revision.hotspot_set == original.hotspot_set
    assert changed_card.active_revision.image_prompt is not None
    assert changed_card.active_revision.image_prompt.text == (
        "A richly detailed courtyard"
    )
    assert window.notification_bar.message_label.text() == "New version created"
    assert window.notification_bar.primary_button.text() == "Undo"
    assert window.notification_bar.secondary_button.isHidden()

    window.notification_bar.primary_button.click()
    restored_card = controller.document.cards[0]
    assert len(restored_card.revisions) == 1
    assert restored_card.active_revision.id == original.id
    assert restored_card.active_revision.image_prompt is not None
    assert restored_card.active_revision.image_prompt.text == (
        "A richly detailed courtyard"
    )


def test_preparation_completion_selects_image_prompt(
    application: QApplication,
) -> None:
    previous = CardRevision(
        description="A changed courtyard",
        image_prompt=ImagePrompt(
            text="An older Image Prompt",
            source_description="A courtyard",
        ),
    )
    card = Card(name="Card", revisions=(previous,))
    window, controller, _workers, _background = _window(
        Stack(name="Demo", cards=(card,))
    )
    window.inspector.description_button.click()
    assert window.inspector.description_button.isChecked()
    changed = controller.execute(
        SetRevisionImagePromptCommand(
            card_id=card.id,
            revision_id=previous.id,
            value=ImagePrompt(
                text="A newly prepared courtyard",
                source_description=previous.description,
            ),
        )
    )
    window.render_document(changed)
    assert window.inspector.description_button.isChecked()
    window.inspector.description_edit.setFocus()
    application.processEvents()
    token = controller.current_undo_token
    assert token is not None

    window.image_prompt_workflow.generation_applied.emit(
        GeneratedRevisionChange(
            message="Image Prompt prepared",
            token=token,
            card_id=card.id,
            revision_id=previous.id,
            previous_revision=previous,
        )
    )

    assert window.inspector.image_prompt_button.isChecked()
    assert window.inspector.description_edit.toPlainText() == (
        "A newly prepared courtyard"
    )
    assert window.notification_bar.message_label.text() == (
        "Image Prompt prepared on the current version"
    )
    window.close()


def test_dismissing_generated_result_keeps_it_on_current_version(
    application: QApplication,
) -> None:
    original = CardRevision(description="A courtyard")
    card = Card(name="Card", revisions=(original,))
    window, controller, _workers, _background = _window(
        Stack(name="Demo", cards=(card,))
    )
    changed = controller.execute(
        SetRevisionImagePromptCommand(
            card_id=card.id,
            revision_id=original.id,
            value=ImagePrompt(
                text="A richly detailed courtyard",
                source_description=original.description,
            ),
        )
    )
    window.render_document(changed)
    token = controller.current_undo_token
    assert token is not None
    window._show_generated_revision_notification(
        GeneratedRevisionChange(
            message="Image Prompt prepared",
            token=token,
            card_id=card.id,
            revision_id=original.id,
            previous_revision=original,
        )
    )

    window.notification_bar.dismiss_button.click()

    revision = controller.document.cards[0].active_revision
    assert revision.id == original.id
    assert revision.image_prompt is not None
    assert revision.image_prompt.text == "A richly detailed courtyard"
    assert controller.current_undo_token == token
    assert window._generated_revision_change is None


def test_description_edits_and_notification_undo_cancel_preparation(
    application: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window, controller, _workers, _background = _window()
    cancellations: list[bool] = []
    monkeypatch.setattr(
        window.image_prompt_workflow,
        "cancel",
        lambda: cancellations.append(True),
    )

    window.inspector.description_edit.setPlainText("Edited draft")
    assert cancellations == [True]

    card = controller.document.cards[0]
    changed = controller.execute(RenameCardCommand(card_id=card.id, name="Renamed"))
    window.render_document(changed)
    token = controller.current_undo_token
    assert token is not None
    window._show_undo_notification("Renamed", token)
    window._undo_notification()

    assert cancellations == [True, True]
    assert controller.document.cards[0].name == "Foyer"


def test_notification_undo_cannot_mutate_document_in_run_mode(
    application: QApplication,
) -> None:
    window, controller, _workers, _background = _window()
    card = controller.document.cards[0]
    changed = controller.execute(RenameCardCommand(card_id=card.id, name="Renamed"))
    window.render_document(changed)
    token = controller.current_undo_token
    assert token is not None
    window._show_undo_notification("Renamed", token)

    window.mode_button.click()
    assert window.notification_bar.current_key != "undo"
    window._notification_action_requested("undo")
    assert controller.document.cards[0].name == "Renamed"


def test_new_workflow_progress_clears_stale_failure_notification(
    application: QApplication,
) -> None:
    window, controller, _workers, _background = _window()
    window._show_error(
        "image-prompt-error",
        "Image Prompt preparation failed",
        detail="Old failure",
    )
    window._image_prompt_progress_changed("Preparing Image Prompt…")

    card = controller.document.cards[0]
    controller.execute(RenameCardCommand(card_id=card.id, name="Renamed"))
    token = controller.current_undo_token
    assert token is not None
    window._show_undo_notification("Renamed", token)
    assert window.notification_bar.current_key == "undo"


def test_generation_progress_bar_is_indeterminate_for_text_and_uses_image_steps(
    application: QApplication,
) -> None:
    window, _controller, _workers, background = _window()
    progress_container = window.generation_progress_container
    progress = window.generation_progress_bar

    window.image_prompt_workflow._busy = True
    window.image_prompt_workflow.progress_changed.emit(
        "Preparing Image Prompt..."
    )
    assert not progress_container.isHidden()
    assert progress.minimum() == 0
    assert progress.maximum() == 0
    assert not progress.isTextVisible()
    assert window.generation_step_label.text() == (
        "Preparing Image Prompt..."
    )

    window.image_prompt_workflow._busy = False
    window.image_prompt_workflow.progress_changed.emit("Image Prompt prepared")
    assert progress_container.isHidden()

    background.busy = True
    background.progress_changed.emit("Generating image...")
    assert not progress_container.isHidden()
    assert window.generation_step_label.text() == "Generating image..."
    assert progress.minimum() == 0
    assert progress.maximum() == 0

    background.generation_progress_changed.emit(1, 4)
    assert progress.minimum() == 0
    assert progress.maximum() == 4
    assert progress.value() == 1

    window.image_prompt_workflow._busy = True
    window.image_prompt_workflow.progress_changed.emit(
        "Preparing Image Prompt..."
    )
    assert window.generation_step_label.text() == "Generating image..."
    assert progress.maximum() == 4
    assert progress.value() == 1

    background.generation_progress_changed.emit(3, 4)
    assert progress.value() == 3

    background.generation_progress_changed.emit(4, 4)
    assert progress.value() == 4

    background.generation_progress_changed.emit(0, 0)
    assert progress.minimum() == 0
    assert progress.maximum() == 0

    background.busy = False
    background.progress_changed.emit("Image generated")
    assert not progress_container.isHidden()
    assert progress.minimum() == 0
    assert progress.maximum() == 0

    window.image_prompt_workflow._busy = False
    window.image_prompt_workflow.progress_changed.emit("Image Prompt prepared")
    assert progress_container.isHidden()
    assert window.generation_step_label.text() == ""


def test_background_success_clears_previous_cancellation_notice(
    application: QApplication,
) -> None:
    window, _controller, _workers, _background = _window()
    window._background_progress_changed("Generation cancelled")
    assert window.notification_bar.current_key == "background-cancelled"

    window._background_progress_changed("Image generated")
    assert window.notification_bar.isHidden()


def test_document_replacement_clears_document_specific_notifications(
    application: QApplication,
) -> None:
    window, _controller, _workers, background = _window()
    background.busy = True
    window._show_error(
        "image-prompt-error",
        "Image Prompt preparation failed",
    )
    window._set_canvas_card_name_error("Invalid card name")
    window.inspector.set_hotspot_error("Invalid hotspot")

    window._document_replaced(object())
    assert window.notification_bar.isHidden()
    assert background.cancel_calls == 1
    assert window.canvas_card_name_error.isHidden()
    assert window.inspector.hotspot_error.isHidden()


def test_clean_session_state_clears_previous_document_error(
    application: QApplication,
    tmp_path: Path,
) -> None:
    window, _controller, _workers, _background = _window()
    window._show_document_error("Could Not Save Stack", "Disk is full")

    window._session_state_changed(
        DocumentSessionState(
            bundle_path=tmp_path / "Demo.hotcards",
            dirty=False,
            error=None,
        )
    )
    assert window.notification_bar.isHidden()


def test_hotspot_geometry_change_shows_targeted_undo(
    application: QApplication,
) -> None:
    interaction = Interaction(
        label="Door",
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
    revision = CardRevision(hotspot_set=HotspotSet(interactions=(interaction,)))
    card = Card(name="Card", revisions=(revision,))
    window, controller, _workers, _background = _window(Stack(name="Demo", cards=(card,)))

    window._delete_hotspot_polygon(interaction.id, 0)

    changed = controller.document.cards[0].active_revision.hotspot_set
    assert changed is not None
    assert changed.interactions[0].polygons == ()
    assert window.notification_bar.message_label.text() == "Hotspot area deleted"
    assert not window.notification_bar.isHidden()

    window._undo_notification()

    restored = controller.document.cards[0].active_revision.hotspot_set
    assert restored is not None
    assert restored.interactions[0].polygons == interaction.polygons


def test_empty_canvas_request_creates_and_selects_blank_hotspot(
    application: QApplication,
) -> None:
    card = Card(name="Card")
    window, controller, _workers, _background = _window(Stack(name="Demo", cards=(card,)))

    window._begin_implicit_hotspot_area(Point(x=0.2, y=0.3))

    hotspot_set = controller.document.cards[0].active_revision.hotspot_set
    assert hotspot_set is not None
    assert len(hotspot_set.interactions) == 1
    assert hotspot_set.interactions[0].polygons == ()
    assert window.inspector.selected_interaction_id == hotspot_set.interactions[0].id


def test_hotspot_editing_is_scoped_to_hotspots_tab(
    application: QApplication,
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "background.png"
    image = QPixmap(1024, 768)
    image.fill(QColor("navy"))
    assert image.save(str(image_path))
    interaction = Interaction(
        label="Door",
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
    asset_id = uuid4()
    revision = CardRevision(
        background=_generated_background(
            asset_id=asset_id,
            image_path=f"assets/cards/card/image-{asset_id}.png",
        ),
        hotspot_set=HotspotSet(interactions=(interaction,)),
    )
    card = Card(name="Card", revisions=(revision,))
    window, _controller, _workers, _background = _window(
        Stack(name="Demo", cards=(card,))
    )
    window.document_session = SimpleNamespace(
        store=SimpleNamespace(asset_path=lambda _path: image_path),
        state=DocumentSessionState(bundle_path=None, dirty=False, error=None),
    )
    window.render_document()

    assert not window.inspector.hotspots_active
    assert not window.card_canvas._editable
    assert window.card_canvas._overlay_items == []

    window.inspector.inspector_tabs.setCurrentIndex(
        window.inspector._hotspots_tab_index
    )
    assert window.inspector.hotspots_active
    assert window.card_canvas._editable
    assert window.card_canvas._overlay_items
    window.card_canvas.begin_polygon(
        interaction.id,
        initial_point=Point(x=0.6, y=0.6),
    )
    assert window.card_canvas.drawing

    window.inspector.inspector_tabs.setCurrentIndex(0)
    assert not window.card_canvas.drawing
    assert not window.card_canvas._editable
    assert window.card_canvas._overlay_items == []


def test_new_stack_dialog_creates_one_blank_revision(
    application: QApplication,
) -> None:
    dialog = NewStackDialog()
    assert not hasattr(dialog, "global_style_edit")
    stack = dialog.stack()
    assert len(stack.cards) == 1
    assert len(stack.cards[0].revisions) == 1
    assert stack.cards[0].active_revision.style_id == HYPERCARD_STYLE_ID
    assert stack.new_card_style_id == HYPERCARD_STYLE_ID


def test_empty_stack_has_clear_first_card_path(
    application: QApplication,
) -> None:
    window, _controller, _workers, _background = _window(Stack(name="Empty"))
    assert window.canvas_pages.currentIndex() == 0
    assert window.empty_canvas_title.text() == "Create your first card"
