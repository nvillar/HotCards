"""Tests for startup project selection."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication, QDialog, QLabel, QMessageBox

import hotcards.ui.welcome_dialog as welcome_dialog_module
from hotcards.application.document_controller import DocumentController
from hotcards.application.document_session import DocumentSession
from hotcards.domain.models import Card, Stack
from hotcards.main import _apply_welcome_selection
from hotcards.ui.project_paths import bundle_path, default_project_directory
from hotcards.ui.welcome_dialog import WelcomeDialog, WelcomeSelection


@pytest.fixture(scope="module")
def application() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_default_project_directory_is_under_documents(tmp_path: Path) -> None:
    assert default_project_directory(tmp_path) == tmp_path / "HotCards"
    assert bundle_path(tmp_path / "Garden") == tmp_path / "Garden.hotcards"
    assert bundle_path(tmp_path / "Garden.hotcards") == (tmp_path / "Garden.hotcards")


def test_welcome_lists_only_direct_bundle_directories(
    application: QApplication,
    tmp_path: Path,
) -> None:
    projects = tmp_path / "HotCards"
    (projects / "Beta.hotcards").mkdir(parents=True)
    (projects / "alpha.HOTCARDS").mkdir()
    (projects / "Ordinary").mkdir()
    (projects / "linked.hotcards").symlink_to(
        projects / "Ordinary",
        target_is_directory=True,
    )
    (projects / "file.hotcards").write_text("not a bundle", encoding="utf-8")
    (projects / "Ordinary" / "Nested.hotcards").mkdir()

    dialog = WelcomeDialog(projects)

    assert dialog.stacks_label.text() == "Stacks"
    assert dialog.findChild(QLabel, "welcomeSubtitle") is None
    assert dialog.new_button.text() == "Create New Stack"
    assert dialog.delete_button.text() == "Delete Stack"
    assert dialog.open_button.text() == "Open Stack"
    assert [dialog.project_list.item(row).text() for row in range(dialog.project_list.count())] == [
        "alpha",
        "Beta",
    ]
    assert dialog.open_button.isEnabled()
    assert dialog.delete_button.isEnabled()
    assert dialog.empty_label.isHidden()
    dialog.open_button.click()
    assert dialog.result() == QDialog.DialogCode.Accepted
    assert dialog.selection == WelcomeSelection(bundle_path=projects / "alpha.HOTCARDS")


def test_welcome_empty_state_creates_default_directory(
    application: QApplication,
    tmp_path: Path,
) -> None:
    projects = tmp_path / "Documents" / "HotCards"

    dialog = WelcomeDialog(projects)

    assert dialog.banner.objectName() == "welcomeBanner"
    assert dialog.banner.accessibleName() == "Spaceship"
    banner = dialog.banner.pixmap()
    assert not banner.isNull()
    assert banner.deviceIndependentSize().width() == 128
    assert banner.deviceIndependentSize().height() == 128
    assert banner.toImage().pixelColor(0, 0).alpha() == 0
    assert not dialog.windowIcon().isNull()
    dialog.show()
    application.processEvents()
    assert dialog.banner.geometry().top() == dialog.title.geometry().top()
    dialog.hide()
    assert projects.is_dir()
    assert dialog.project_list.count() == 0
    assert not dialog.open_button.isEnabled()
    assert "No stacks yet" in dialog.empty_label.text()
    assert not dialog.delete_button.isEnabled()
    dialog.close()


def test_welcome_directory_error_does_not_block_project_creation(
    application: QApplication,
    tmp_path: Path,
) -> None:
    projects = tmp_path / "HotCards"
    projects.write_text("not a directory", encoding="utf-8")

    dialog = WelcomeDialog(projects)

    assert dialog.project_list.count() == 0
    assert not dialog.open_button.isEnabled()
    assert dialog.new_button.isEnabled()
    assert "could not read" in dialog.empty_label.text()
    dialog.close()


def test_welcome_new_project_defaults_save_location(
    application: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projects = tmp_path / "HotCards"
    stack = Stack(name="Garden", cards=(Card(name="Card 1"),))

    class AcceptedNewStackDialog:
        def __init__(self, _parent: object) -> None:
            pass

        def exec(self) -> QDialog.DialogCode:
            return QDialog.DialogCode.Accepted

        def stack(self) -> Stack:
            return stack

    requested_paths: list[str] = []

    def select_path(
        _parent: object,
        _title: str,
        suggested_path: str,
        _filter: str,
    ) -> tuple[str, str]:
        requested_paths.append(suggested_path)
        assert _title == "Create HotCards Stack"
        return str(projects / "Garden"), "HotCards Stack (*.hotcards)"

    monkeypatch.setattr(
        welcome_dialog_module,
        "NewStackDialog",
        AcceptedNewStackDialog,
    )
    monkeypatch.setattr(
        welcome_dialog_module.QFileDialog,
        "getSaveFileName",
        select_path,
    )
    dialog = WelcomeDialog(projects)

    dialog.new_button.click()

    assert requested_paths == [str(projects / "Garden.hotcards")]
    assert dialog.selection == WelcomeSelection(
        bundle_path=projects / "Garden.hotcards",
        stack=stack,
    )
    assert dialog.result() == QDialog.DialogCode.Accepted


def test_existing_project_destination_is_reported_inside_welcome_dialog(
    application: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projects = tmp_path / "HotCards"
    existing = projects / "Garden.hotcards"
    existing.mkdir(parents=True)
    stack = Stack(name="Garden", cards=(Card(name="Card 1"),))

    class AcceptedNewStackDialog:
        def __init__(self, _parent: object) -> None:
            pass

        def exec(self) -> QDialog.DialogCode:
            return QDialog.DialogCode.Accepted

        def stack(self) -> Stack:
            return stack

    monkeypatch.setattr(
        welcome_dialog_module,
        "NewStackDialog",
        AcceptedNewStackDialog,
    )
    monkeypatch.setattr(
        welcome_dialog_module.QFileDialog,
        "getSaveFileName",
        lambda *_args, **_kwargs: (
            str(existing),
            "HotCards Stack (*.hotcards)",
        ),
    )
    dialog = WelcomeDialog(projects)

    dialog.new_button.click()

    assert dialog.selection is None
    assert not dialog.error_label.isHidden()
    assert "already exists" in dialog.error_label.text()
    assert str(existing) in dialog.error_label.text()


def test_welcome_deletes_only_the_selected_stack_after_confirmation(
    application: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projects = tmp_path / "HotCards"
    first = projects / "First.hotcards"
    second = projects / "Second.hotcards"
    first.mkdir(parents=True)
    second.mkdir()
    (first / "stack.json").write_text("{}", encoding="utf-8")
    dialog = WelcomeDialog(projects)
    dialog.project_list.setCurrentRow(1)
    confirmations: list[Path] = []
    monkeypatch.setattr(
        dialog,
        "_confirm_stack_deletion",
        lambda path: confirmations.append(path) or True,
    )

    dialog.delete_button.click()

    assert confirmations == [second]
    assert first.is_dir()
    assert not second.exists()
    assert [dialog.project_list.item(row).text() for row in range(dialog.project_list.count())] == [
        "First"
    ]


def test_welcome_preserves_stack_when_deletion_is_cancelled(
    application: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stack = tmp_path / "HotCards" / "Garden.hotcards"
    stack.mkdir(parents=True)
    dialog = WelcomeDialog(stack.parent)
    monkeypatch.setattr(
        dialog,
        "_confirm_stack_deletion",
        lambda _path: False,
    )

    dialog.delete_button.click()

    assert stack.is_dir()
    assert dialog.project_list.count() == 1


def test_delete_confirmation_centers_on_welcome_window(
    application: QApplication,
    tmp_path: Path,
) -> None:
    dialog = WelcomeDialog(tmp_path / "HotCards")
    dialog.move(120, 80)
    dialog.show()
    confirmation = QMessageBox(dialog)
    confirmation.setText("Delete this stack?")
    confirmation.show()
    application.processEvents()

    dialog._center_child_dialog(confirmation)

    assert confirmation.frameGeometry().center() == dialog.frameGeometry().center()
    confirmation.close()
    dialog.close()


def test_startup_selection_creates_and_reopens_project(tmp_path: Path) -> None:
    bundle = tmp_path / "Garden.hotcards"
    stack = Stack(name="Garden", cards=(Card(name="Card 1"),))
    creator_controller = DocumentController(Stack(name="Welcome"))
    creator_session = DocumentSession(creator_controller)

    _apply_welcome_selection(
        creator_session,
        WelcomeSelection(bundle_path=bundle, stack=stack),
    )

    assert creator_session.state.bundle_path == bundle
    assert creator_controller.document == stack

    opener_controller = DocumentController(Stack(name="Welcome"))
    opener_session = DocumentSession(opener_controller)
    _apply_welcome_selection(
        opener_session,
        WelcomeSelection(bundle_path=bundle),
    )
    assert opener_session.state.bundle_path == bundle
    assert opener_controller.document == stack
