"""Startup project chooser shown before the authoring shell."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from hotcards.domain.models import Stack
from hotcards.ui.branding import application_icon, application_icon_pixmap
from hotcards.ui.new_stack_dialog import NewStackDialog
from hotcards.ui.project_paths import bundle_path


@dataclass(frozen=True, slots=True)
class WelcomeSelection:
    """One validated user intent returned to application startup."""

    bundle_path: Path
    stack: Stack | None = None


class WelcomeDialog(QDialog):
    """Choose an existing project or define a new one before opening the editor."""

    def __init__(
        self,
        project_directory: Path,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.project_directory = project_directory
        self.selection: WelcomeSelection | None = None
        self.setWindowTitle("HotCards")
        icon = application_icon()
        self.setWindowIcon(icon)
        self.setMinimumSize(560, 500)

        self.banner = QLabel()
        self.banner.setObjectName("welcomeBanner")
        self.banner.setAccessibleName("Spaceship")
        self.banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.banner.setFixedSize(128, 128)
        self.banner.setPixmap(
            application_icon_pixmap(
                128,
                device_pixel_ratio=self.devicePixelRatioF(),
            )
        )

        self.title = QLabel("HotCards")
        self.title.setObjectName("welcomeTitle")
        self.title.setStyleSheet("font-size: 28px; font-weight: 600;")
        heading = QVBoxLayout()
        heading.setContentsMargins(0, 0, 0, 0)
        heading.setSpacing(6)
        heading.addWidget(self.title)
        heading.addStretch()
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(20)
        header.addLayout(heading, 1)
        header.addWidget(
            self.banner,
            0,
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight,
        )

        self.stacks_label = QLabel("Stacks")
        self.stacks_label.setObjectName("welcomeStacksLabel")
        self.stacks_label.setStyleSheet("font-weight: 600;")
        self.project_list = QListWidget()
        self.project_list.setObjectName("welcomeProjectList")
        self.project_list.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.project_list.itemSelectionChanged.connect(self._update_open_button)
        self.project_list.itemDoubleClicked.connect(self._open_project)

        self.empty_label = QLabel()
        self.empty_label.setObjectName("welcomeEmptyLabel")
        self.empty_label.setWordWrap(True)
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        location_label = QLabel(str(project_directory))
        location_label.setObjectName("welcomeProjectDirectory")
        location_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        location_label.setStyleSheet("color: palette(mid);")

        self.error_label = QLabel()
        self.error_label.setObjectName("welcomeErrorLabel")
        self.error_label.setWordWrap(True)
        self.error_label.setVisible(False)

        self.new_button = QPushButton("Create New Stack")
        self.new_button.setObjectName("welcomeNewProjectButton")
        self.new_button.clicked.connect(self._create_project)
        self.delete_button = QPushButton("Delete Stack")
        self.delete_button.setObjectName("welcomeDeleteStackButton")
        self.delete_button.setEnabled(False)
        self.delete_button.clicked.connect(self._delete_stack)
        self.open_button = QPushButton("Open Stack")
        self.open_button.setObjectName("welcomeOpenProjectButton")
        self.open_button.setEnabled(False)
        self.open_button.clicked.connect(self._open_project)
        self.open_button.setDefault(True)

        buttons = QHBoxLayout()
        buttons.addWidget(self.new_button)
        buttons.addWidget(self.delete_button)
        buttons.addStretch()
        buttons.addWidget(self.open_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(12)
        layout.addLayout(header)
        layout.addSpacing(12)
        layout.addWidget(self.stacks_label)
        layout.addWidget(self.project_list, 1)
        layout.addWidget(self.empty_label)
        layout.addWidget(location_label)
        layout.addWidget(self.error_label)
        layout.addSpacing(8)
        layout.addLayout(buttons)

        self.refresh_projects()

    def refresh_projects(self) -> None:
        """Refresh direct child bundles without opening untrusted stack data."""
        self._set_error("")
        self.project_list.clear()
        self.new_button.setEnabled(True)
        try:
            self.project_directory.mkdir(parents=True, exist_ok=True)
            entries = tuple(self.project_directory.iterdir())
        except OSError as error:
            self.empty_label.setText(
                f"HotCards could not read the project directory:\n{error}"
            )
            self.empty_label.setVisible(True)
            self._update_open_button()
            return
        projects: list[Path] = []
        unreadable_entries = 0
        for path in entries:
            try:
                if (
                    not path.is_symlink()
                    and path.is_dir()
                    and path.suffix.casefold() == ".hotcards"
                ):
                    projects.append(path)
            except OSError:
                unreadable_entries += 1
        projects.sort(key=lambda path: path.stem.casefold())
        for path in projects:
            item = QListWidgetItem(path.stem)
            item.setData(Qt.ItemDataRole.UserRole, path)
            item.setToolTip(str(path))
            self.project_list.addItem(item)
        if unreadable_entries:
            self.empty_label.setText(
                f"{unreadable_entries} project directory "
                f"{'entry was' if unreadable_entries == 1 else 'entries were'} "
                "not readable."
            )
        elif not projects:
            self.empty_label.setText("No stacks yet. Create one to get started.")
        else:
            self.empty_label.clear()
        self.empty_label.setVisible(not projects or unreadable_entries > 0)
        if projects:
            self.project_list.setCurrentRow(0)
        self._update_open_button()

    def _update_open_button(self) -> None:
        has_selection = self.project_list.currentItem() is not None
        self.open_button.setEnabled(has_selection)
        self.delete_button.setEnabled(has_selection)

    def _open_project(self, *_args: object) -> None:
        item = self.project_list.currentItem()
        if item is None:
            return
        path = item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(path, Path):
            return
        self.selection = WelcomeSelection(bundle_path=path)
        self.accept()

    def _create_project(self) -> None:
        self._set_error("")
        dialog = NewStackDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        stack = dialog.stack()
        suggested_path = self.project_directory / f"{stack.name}.hotcards"
        selected_path, _filter = QFileDialog.getSaveFileName(
            self,
            "Create HotCards Stack",
            str(suggested_path),
            "HotCards Stack (*.hotcards)",
        )
        if not selected_path:
            return
        path = bundle_path(selected_path)
        if path.exists():
            self._set_error(
                f"A stack already exists at:\n{path}"
            )
            return
        self.selection = WelcomeSelection(bundle_path=path, stack=stack)
        self.accept()

    def _delete_stack(self) -> None:
        item = self.project_list.currentItem()
        if item is None:
            return
        path = item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(path, Path):
            self._set_error("Could not identify the selected stack.")
            return
        try:
            stack_path = self._validated_stack_path(path)
        except (OSError, ValueError) as error:
            self._set_error(f"Could not delete stack:\n{error}")
            return
        if not self._confirm_stack_deletion(stack_path):
            return
        try:
            shutil.rmtree(stack_path)
        except OSError as error:
            self._set_error(f"Could not delete stack:\n{error}")
            return
        self.refresh_projects()

    def _validated_stack_path(self, path: Path) -> Path:
        project_directory = self.project_directory.resolve(strict=True)
        if path.is_symlink():
            raise ValueError("symbolic-link stacks cannot be deleted")
        stack_path = path.resolve(strict=True)
        if (
            stack_path.parent != project_directory
            or stack_path.suffix.casefold() != ".hotcards"
            or not stack_path.is_dir()
        ):
            raise ValueError("the selected stack is outside the stack directory")
        return stack_path

    def _confirm_stack_deletion(self, path: Path) -> bool:
        confirmation = QMessageBox(self)
        confirmation.setIcon(QMessageBox.Icon.Warning)
        confirmation.setWindowTitle("Delete Stack")
        confirmation.setText(f'Delete “{path.stem}”?')
        confirmation.setInformativeText(
            "This permanently deletes the stack and all of its generated images. "
            "This action cannot be undone."
        )
        delete_button = confirmation.addButton(
            "Delete Stack",
            QMessageBox.ButtonRole.DestructiveRole,
        )
        cancel_button = confirmation.addButton(QMessageBox.StandardButton.Cancel)
        confirmation.setDefaultButton(cancel_button)
        QTimer.singleShot(
            0,
            lambda: self._center_child_dialog(confirmation),
        )
        confirmation.exec()
        return confirmation.clickedButton() is delete_button

    def _center_child_dialog(self, dialog: QDialog) -> None:
        frame = dialog.frameGeometry()
        frame.moveCenter(self.frameGeometry().center())
        dialog.move(frame.topLeft())

    def _set_error(self, message: str) -> None:
        self.error_label.setText(message)
        self.error_label.setVisible(bool(message))


__all__ = ["WelcomeDialog", "WelcomeSelection"]
