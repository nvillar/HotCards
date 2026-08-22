"""Stack-level named generation style manager."""

from __future__ import annotations

from uuid import UUID

from pydantic import ValidationError
from PySide6.QtCore import QSignalBlocker, Qt, Signal
from PySide6.QtGui import QFocusEvent
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from hypergen.application.commands import (
    AddStyleCommand,
    CommandError,
    DeleteStyleCommand,
    EditStyleCommand,
)
from hypergen.application.document_controller import DocumentController
from hypergen.domain.models import GenerationStyle


class _CommitPlainTextEdit(QPlainTextEdit):
    editing_finished = Signal()

    def focusOutEvent(self, event: QFocusEvent) -> None:
        super().focusOutEvent(event)
        self.editing_finished.emit()


class StylesDialog(QDialog):
    """Edit the current stack's reusable image-generation styles."""

    document_changed = Signal(object)

    def __init__(
        self,
        controller: DocumentController,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.controller = controller
        self._rendering = False
        self.setObjectName("stylesDialog")
        self.setWindowTitle("Styles")
        self.resize(520, 360)

        layout = QVBoxLayout(self)
        self.style_list = QListWidget()
        self.style_list.setObjectName("styleList")
        layout.addWidget(self.style_list, 1)

        controls = QHBoxLayout()
        self.add_button = QPushButton("+")
        self.add_button.setObjectName("addStyleButton")
        self.add_button.setAccessibleName("Add style")
        self.delete_button = QPushButton("−")
        self.delete_button.setObjectName("deleteStyleButton")
        self.delete_button.setAccessibleName("Delete style")
        controls.addWidget(self.add_button)
        controls.addWidget(self.delete_button)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.name_edit = QLineEdit()
        self.name_edit.setObjectName("styleNameEdit")
        self.name_edit.setPlaceholderText("Style name")
        self.name_edit.setAccessibleName("Style name")
        layout.addWidget(self.name_edit)
        self.prompt_edit = _CommitPlainTextEdit()
        self.prompt_edit.setObjectName("stylePromptEdit")
        self.prompt_edit.setPlaceholderText("Image-generation style prompt")
        self.prompt_edit.setAccessibleName("Style prompt")
        layout.addWidget(self.prompt_edit)

        self.error_label = QLabel()
        self.error_label.setObjectName("styleValidationError")
        self.error_label.setWordWrap(True)
        self.error_label.setVisible(False)
        layout.addWidget(self.error_label)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.style_list.currentItemChanged.connect(self._selection_changed)
        self.add_button.clicked.connect(self._add_style)
        self.delete_button.clicked.connect(self._delete_style)
        self.name_edit.editingFinished.connect(self._commit_selected)
        self.prompt_edit.editing_finished.connect(self._commit_selected)
        self._render()

    def reject(self) -> None:
        self._commit_selected()
        super().reject()

    def _render(self, selected_id: UUID | None = None) -> None:
        if selected_id is None:
            selected_id = self._selected_style_id()
        self._rendering = True
        try:
            with QSignalBlocker(self.style_list):
                self.style_list.clear()
                for style in self.controller.document.styles:
                    item = QListWidgetItem(style.name)
                    item.setData(Qt.ItemDataRole.UserRole, style.id)
                    self.style_list.addItem(item)
                row = next(
                    (
                        index
                        for index in range(self.style_list.count())
                        if self.style_list.item(index).data(
                            Qt.ItemDataRole.UserRole
                        )
                        == selected_id
                    ),
                    0 if self.style_list.count() else -1,
                )
                self.style_list.setCurrentRow(row)
            self._show_selected()
        finally:
            self._rendering = False

    def _show_selected(self) -> None:
        style = self._selected_style()
        with QSignalBlocker(self.name_edit):
            self.name_edit.setText(style.name if style is not None else "")
        with QSignalBlocker(self.prompt_edit):
            self.prompt_edit.setPlainText(style.prompt if style is not None else "")
        enabled = style is not None
        self.name_edit.setEnabled(enabled)
        self.prompt_edit.setEnabled(enabled)
        self.delete_button.setEnabled(enabled)

    def _selection_changed(self) -> None:
        if not self._rendering:
            self._show_selected()

    def _add_style(self) -> None:
        used = {style.name.casefold() for style in self.controller.document.styles}
        number = 1
        while f"Style {number}".casefold() in used:
            number += 1
        style = GenerationStyle(name=f"Style {number}")
        self._execute(AddStyleCommand(style=style), selected_id=style.id)
        self.name_edit.setFocus()
        self.name_edit.selectAll()

    def _delete_style(self) -> None:
        style_id = self._selected_style_id()
        if style_id is not None:
            self._execute(DeleteStyleCommand(style_id=style_id))

    def _commit_selected(self) -> None:
        if self._rendering:
            return
        style = self._selected_style()
        if style is None:
            return
        name = self.name_edit.text()
        prompt = self.prompt_edit.toPlainText()
        if name == style.name and prompt == style.prompt:
            return
        self._execute(
            EditStyleCommand(
                style_id=style.id,
                name=name,
                prompt=prompt,
            ),
            selected_id=style.id,
        )

    def _execute(
        self,
        command: AddStyleCommand | EditStyleCommand | DeleteStyleCommand,
        *,
        selected_id: UUID | None = None,
    ) -> None:
        try:
            changed = self.controller.execute(command)
        except (CommandError, ValidationError) as error:
            self.error_label.setText(str(error))
            self.error_label.setVisible(True)
            self._render(selected_id)
            return
        self.error_label.clear()
        self.error_label.setVisible(False)
        self._render(selected_id)
        self.document_changed.emit(changed)

    def _selected_style_id(self) -> UUID | None:
        item = self.style_list.currentItem()
        if item is None:
            return None
        value = item.data(Qt.ItemDataRole.UserRole)
        return value if isinstance(value, UUID) else None

    def _selected_style(self) -> GenerationStyle | None:
        style_id = self._selected_style_id()
        return next(
            (
                style
                for style in self.controller.document.styles
                if style.id == style_id
            ),
            None,
        )


__all__ = ["StylesDialog"]
