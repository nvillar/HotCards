"""Modeless stack-global Style, Sound, and Key managers."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from uuid import UUID, uuid4

from pydantic import ValidationError
from PySide6.QtCore import QSignalBlocker, Qt, QTimer, Signal
from PySide6.QtGui import QCloseEvent, QFocusEvent
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from hotcards.application.commands import (
    AddKeyCommand,
    AddSoundCommand,
    AddStyleCommand,
    CommandError,
    DeleteKeyCommand,
    DeleteSoundCommand,
    DeleteStyleCommand,
    DocumentCommand,
    RenameKeyCommand,
    UpdateSoundCommand,
    UpdateStyleCommand,
)
from hotcards.application.document_controller import (
    DocumentController,
    DocumentMutationBlockedError,
)
from hotcards.application.sound_player import SoundPlayer
from hotcards.application.sound_workflow import SoundWorkflow, SoundWorkflowError
from hotcards.domain.models import (
    Card,
    Interaction,
    KeyDefinition,
    SoundDefinition,
    Stack,
    StyleDefinition,
)

UTILITY_WINDOW_WIDTH = 420
UTILITY_WINDOW_HEIGHT = 600
UTILITY_USAGE_LIST_HEIGHT = 70


class _CommitPlainTextEdit(QPlainTextEdit):
    editing_finished = Signal(object, object)

    def focusOutEvent(self, event: QFocusEvent) -> None:
        super().focusOutEvent(event)
        self.editing_finished.emit(QApplication.focusWidget(), event.reason())


class _CommitLineEdit(QLineEdit):
    editing_finished = Signal(object, object)

    def focusOutEvent(self, event: QFocusEvent) -> None:
        super().focusOutEvent(event)
        self.editing_finished.emit(QApplication.focusWidget(), event.reason())


def _compact_button(
    text: str,
    *,
    object_name: str,
    accessible_name: str,
    tooltip: str,
) -> QToolButton:
    button = QToolButton()
    button.setObjectName(object_name)
    button.setText(text)
    button.setAccessibleName(accessible_name)
    button.setToolTip(tooltip)
    font = button.font()
    font.setPointSizeF(max(font.pointSizeF() + 4.0, 16.0))
    font.setBold(True)
    button.setFont(font)
    return button


class _ControllerUtilityWindow(QWidget):
    document_changed = Signal(object)
    change_applied = Signal(str, object)
    closing = Signal(object)

    def __init__(
        self,
        controller: DocumentController,
        *,
        title: str,
        object_name: str,
        parent: QWidget,
    ) -> None:
        super().__init__(
            parent,
            Qt.WindowType.Tool | Qt.WindowType.WindowStaysOnTopHint,
        )
        self.controller = controller
        self._rendering = False
        self._mutation_allowed = True
        self._discard_pending_on_close = False
        self.setWindowTitle(title)
        self.setObjectName(object_name)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.resize(UTILITY_WINDOW_WIDTH, UTILITY_WINDOW_HEIGHT)

    def set_mutation_allowed(self, allowed: bool) -> None:
        self._mutation_allowed = allowed
        self._update_enabled_state()

    def _execute(
        self,
        command: DocumentCommand,
        *,
        error_label: QLabel,
        undo_message: str,
        render_change: bool = True,
    ) -> bool:
        previous_token = self.controller.current_undo_token
        try:
            changed = self.controller.execute(command)
        except (
            CommandError,
            DocumentMutationBlockedError,
            ValidationError,
        ) as error:
            self._set_error(error_label, str(error))
            if render_change:
                self.render(self.controller.document)
            return False
        self._set_error(error_label, "")
        if render_change:
            self.render(changed)
        self.document_changed.emit(changed)
        token = self.controller.current_undo_token
        if token is not None and token != previous_token:
            self.change_applied.emit(undo_message, token)
        return True

    @staticmethod
    def _set_error(label: QLabel, message: str) -> None:
        label.setText(message)
        label.setVisible(bool(message))

    def closeEvent(self, event: QCloseEvent) -> None:
        if not self._discard_pending_on_close and not self.commit_pending_edits(
            render_change=False
        ):
            event.ignore()
            return
        self.closing.emit(self.saveGeometry())
        super().closeEvent(event)

    def close_without_committing(self) -> bool:
        """Close after project replacement without applying stale editor state."""
        self._discard_pending_on_close = True
        return self.close()

    def _add_done_button(self, layout: QVBoxLayout) -> None:
        button_row = QHBoxLayout()
        button_row.addStretch(1)
        self.done_button = QPushButton("Done")
        self.done_button.setObjectName("doneButton")
        self.done_button.setAccessibleName("Done")
        self.done_button.clicked.connect(self.close)
        button_row.addWidget(self.done_button)
        layout.addLayout(button_row)

    def commit_pending_edits(self, *, render_change: bool) -> bool:
        raise NotImplementedError

    def render(self, document: Stack) -> None:
        raise NotImplementedError

    def _update_enabled_state(self) -> None:
        raise NotImplementedError


class StyleManagerWindow(_ControllerUtilityWindow):
    """Edit the authoritative stack Style library."""

    inputs_changed = Signal()

    def __init__(
        self,
        controller: DocumentController,
        parent: QWidget,
    ) -> None:
        super().__init__(
            controller,
            title="Styles",
            object_name="styleManagerWindow",
            parent=parent,
        )
        self._selected_style_id: UUID | None = None
        self._rendered_style_id: UUID | None = None
        self._name_dirty = False
        self._prompt_dirty = False

        layout = QVBoxLayout(self)
        self.placeholder = QLabel("No Styles yet.")
        self.placeholder.setWordWrap(True)
        layout.addWidget(self.placeholder)
        self.style_list = QListWidget()
        self.style_list.setObjectName("styleList")
        self.style_list.setAccessibleName("Stack Styles")
        layout.addWidget(self.style_list, 1)

        controls = QHBoxLayout()
        controls.addStretch(1)
        self.delete_button = _compact_button(
            "−",
            object_name="deleteStyleButton",
            accessible_name="Delete Style",
            tooltip="Delete Style",
        )
        self.add_button = _compact_button(
            "+",
            object_name="addStyleButton",
            accessible_name="Add Style",
            tooltip="Add Style",
        )
        extent = max(
            self.delete_button.sizeHint().width(),
            self.delete_button.sizeHint().height(),
            self.add_button.sizeHint().width(),
            self.add_button.sizeHint().height(),
        )
        self.delete_button.setFixedSize(extent, extent)
        self.add_button.setFixedSize(extent, extent)
        controls.addWidget(self.delete_button)
        controls.addWidget(self.add_button)
        layout.addLayout(controls)

        self.name_label = QLabel("Name")
        layout.addWidget(self.name_label)
        self.name_edit = _CommitLineEdit()
        self.name_edit.setObjectName("styleNameEdit")
        self.name_edit.setAccessibleName("Style name")
        layout.addWidget(self.name_edit)
        self.prompt_label = QLabel("Style Text")
        layout.addWidget(self.prompt_label)
        self.prompt_edit = _CommitPlainTextEdit()
        self.prompt_edit.setObjectName("stylePromptEdit")
        self.prompt_edit.setAccessibleName("Style text")
        self.prompt_edit.setPlaceholderText(
            "Describe the visual treatment appended during generation"
        )
        self.prompt_edit.setMinimumHeight(self.prompt_edit.fontMetrics().lineSpacing() * 7 + 20)
        layout.addWidget(self.prompt_edit)
        self.error_label = QLabel()
        self.error_label.setObjectName("styleValidationError")
        self.error_label.setWordWrap(True)
        self.error_label.setVisible(False)
        layout.addWidget(self.error_label)
        self._add_done_button(layout)

        self.style_list.currentItemChanged.connect(self._selection_changed)
        self.add_button.clicked.connect(self._add_style)
        self.delete_button.clicked.connect(self._delete_style)
        self.name_edit.editing_finished.connect(self._editing_finished)
        self.prompt_edit.editing_finished.connect(self._editing_finished)
        self.name_edit.textChanged.connect(self._name_draft_changed)
        self.prompt_edit.textChanged.connect(self._prompt_draft_changed)
        self.render(controller.document)

    @property
    def selected_style_id(self) -> UUID | None:
        return self._selected_style_id

    def render(self, document: Stack) -> None:
        name_draft = self.name_edit.text()
        prompt_draft = self.prompt_edit.toPlainText()
        previous_id = self._rendered_style_id
        desired_id = self._selected_style_id
        known_ids = {style.id for style in document.styles}
        if desired_id not in known_ids:
            desired_id = document.styles[0].id if document.styles else None
        if previous_id != desired_id:
            self._name_dirty = False
            self._prompt_dirty = False
        self._rendering = True
        try:
            with QSignalBlocker(self.style_list):
                self.style_list.clear()
                for style in document.styles:
                    item = QListWidgetItem(style.name)
                    item.setData(Qt.ItemDataRole.UserRole, style.id)
                    self.style_list.addItem(item)
                selected_row = next(
                    (
                        row
                        for row in range(self.style_list.count())
                        if self.style_list.item(row).data(Qt.ItemDataRole.UserRole) == desired_id
                    ),
                    -1,
                )
                self.style_list.setCurrentRow(selected_row)
            self._selected_style_id = desired_id
            self._rendered_style_id = desired_id
            self.placeholder.setVisible(not document.styles)
            style = document.style_by_id(desired_id)
            self._render_properties(
                style,
                name_draft=(name_draft if self._name_dirty and previous_id == desired_id else None),
                prompt_draft=(
                    prompt_draft if self._prompt_dirty and previous_id == desired_id else None
                ),
            )
        finally:
            self._rendering = False
        self._update_enabled_state()

    def commit_pending_edits(self, *, render_change: bool) -> bool:
        if self._discard_pending_on_close or self._rendering or self._selected_style_id is None:
            return True
        if self._selected_style_id not in {style.id for style in self.controller.document.styles}:
            return True
        style = self.controller.document.style_by_id(self._selected_style_id)
        name = self.name_edit.text() if self._name_dirty else style.name
        prompt_text = self.prompt_edit.toPlainText() if self._prompt_dirty else style.prompt_text
        if name == style.name and prompt_text.strip() == style.prompt_text:
            self._name_dirty = False
            self._prompt_dirty = False
            return True
        self.inputs_changed.emit()
        committed = self._execute(
            UpdateStyleCommand(
                style_id=style.id,
                name=name,
                prompt_text=prompt_text,
            ),
            error_label=self.error_label,
            undo_message="Style updated",
            render_change=render_change,
        )
        if committed:
            self._name_dirty = False
            self._prompt_dirty = False
        return committed

    def _render_properties(
        self,
        style: StyleDefinition | None,
        *,
        name_draft: str | None = None,
        prompt_draft: str | None = None,
    ) -> None:
        with QSignalBlocker(self.name_edit):
            self.name_edit.setText(
                name_draft if name_draft is not None else style.name if style is not None else ""
            )
        with QSignalBlocker(self.prompt_edit):
            self.prompt_edit.setPlainText(
                prompt_draft
                if prompt_draft is not None
                else style.prompt_text
                if style is not None
                else ""
            )

    def _selection_changed(
        self,
        current: QListWidgetItem | None,
        previous: QListWidgetItem | None,
    ) -> None:
        if self._rendering:
            return
        previous_id = previous.data(Qt.ItemDataRole.UserRole) if previous is not None else None
        if (
            isinstance(previous_id, UUID)
            and previous_id == self._selected_style_id
            and not self.commit_pending_edits(render_change=False)
        ):
            with QSignalBlocker(self.style_list):
                self.style_list.setCurrentItem(previous)
            return
        value = current.data(Qt.ItemDataRole.UserRole) if current is not None else None
        self._selected_style_id = value if isinstance(value, UUID) else None
        self._rendered_style_id = self._selected_style_id
        self._set_error(self.error_label, "")
        style = (
            self.controller.document.style_by_id(self._selected_style_id)
            if self._selected_style_id is not None
            else None
        )
        self._render_properties(style)
        self._update_enabled_state()

    def _editing_finished(
        self,
        _next_focus: object,
        reason: object,
    ) -> None:
        mouse_focus = reason == Qt.FocusReason.MouseFocusReason
        if not self.commit_pending_edits(render_change=not mouse_focus):
            return
        if mouse_focus:
            style = self.controller.document.style_by_id(self._selected_style_id)
            item = self.style_list.currentItem()
            if style is not None and item is not None:
                item.setText(style.name)
                item.setToolTip(style.name)

    def _name_draft_changed(self) -> None:
        if self._rendering:
            return
        self._name_dirty = True
        self.inputs_changed.emit()

    def _prompt_draft_changed(self) -> None:
        if self._rendering:
            return
        self._prompt_dirty = True
        self.inputs_changed.emit()

    def _add_style(self) -> None:
        if not self.commit_pending_edits(render_change=False):
            return
        document = self.controller.document
        names = {style.name.casefold() for style in document.styles}
        number = 1
        name = "New Style"
        while name.casefold() in names:
            number += 1
            name = f"New Style {number}"
        command = AddStyleCommand(name=name)
        self._selected_style_id = command.style_id
        if self._execute(
            command,
            error_label=self.error_label,
            undo_message="Style added",
        ):
            self.name_edit.setFocus()
            self.name_edit.selectAll()

    def _delete_style(self) -> None:
        if self._selected_style_id is None or not self.commit_pending_edits(render_change=False):
            return
        style_id = self._selected_style_id
        self._selected_style_id = None
        self.inputs_changed.emit()
        self._execute(
            DeleteStyleCommand(style_id=style_id),
            error_label=self.error_label,
            undo_message="Style deleted",
        )

    def _update_enabled_state(self) -> None:
        selected = self._selected_style_id is not None
        enabled = self._mutation_allowed and selected
        self.style_list.setEnabled(self._mutation_allowed)
        self.add_button.setEnabled(self._mutation_allowed)
        self.delete_button.setEnabled(enabled)
        self.name_edit.setEnabled(enabled)
        self.prompt_edit.setEnabled(enabled)


class KeyManagerWindow(_ControllerUtilityWindow):
    """Edit the authoritative stack Key catalog and inspect all usages."""

    hotspot_usage_requested = Signal(object, object, object)

    def __init__(
        self,
        controller: DocumentController,
        parent: QWidget,
    ) -> None:
        super().__init__(
            controller,
            title="Keys",
            object_name="keyManagerWindow",
            parent=parent,
        )
        self._selected_key_id: UUID | None = None
        self._rendered_key_id: UUID | None = None
        self._draft_dirty = False

        layout = QVBoxLayout(self)
        self.placeholder = QLabel("No Keys yet.")
        self.placeholder.setWordWrap(True)
        layout.addWidget(self.placeholder)
        self.key_list = QListWidget()
        self.key_list.setObjectName("keyList")
        self.key_list.setAccessibleName("Stack Keys")
        layout.addWidget(self.key_list, 1)

        controls = QHBoxLayout()
        controls.addStretch(1)
        self.delete_button = _compact_button(
            "−",
            object_name="deleteKeyButton",
            accessible_name="Delete Key",
            tooltip="Delete unused Key",
        )
        self.add_button = _compact_button(
            "+",
            object_name="addKeyButton",
            accessible_name="Add Key",
            tooltip="Add Key",
        )
        extent = max(
            self.delete_button.sizeHint().width(),
            self.delete_button.sizeHint().height(),
            self.add_button.sizeHint().width(),
            self.add_button.sizeHint().height(),
        )
        self.delete_button.setFixedSize(extent, extent)
        self.add_button.setFixedSize(extent, extent)
        controls.addWidget(self.delete_button)
        controls.addWidget(self.add_button)
        layout.addLayout(controls)

        self.name_label = QLabel("Name")
        layout.addWidget(self.name_label)
        self.name_edit = _CommitLineEdit()
        self.name_edit.setObjectName("keyNameEdit")
        self.name_edit.setAccessibleName("Key name")
        layout.addWidget(self.name_edit)
        self.usage_label = QLabel("Used By")
        layout.addWidget(self.usage_label)
        self.usage_list = QListWidget()
        self.usage_list.setObjectName("keyUsageList")
        self.usage_list.setAccessibleName("Key usages")
        self.usage_list.setToolTip("Double-click a usage to show its hotspot")
        self.usage_list.setFixedHeight(UTILITY_USAGE_LIST_HEIGHT)
        layout.addWidget(self.usage_list)
        self.error_label = QLabel()
        self.error_label.setObjectName("keyValidationError")
        self.error_label.setWordWrap(True)
        self.error_label.setVisible(False)
        layout.addWidget(self.error_label)
        self._add_done_button(layout)

        self.key_list.currentItemChanged.connect(self._selection_changed)
        self.add_button.clicked.connect(self._add_key)
        self.delete_button.clicked.connect(self._delete_key)
        self.name_edit.editing_finished.connect(self._editing_finished)
        self.name_edit.returnPressed.connect(lambda: self.commit_pending_edits(render_change=True))
        self.name_edit.textChanged.connect(self._draft_changed)
        self.usage_list.itemDoubleClicked.connect(lambda _item: self._show_usage())
        self.render(controller.document)

    @property
    def selected_key_id(self) -> UUID | None:
        return self._selected_key_id

    @staticmethod
    def key_usages(
        document: Stack,
        key_id: UUID,
    ) -> list[tuple[Card, int, Interaction, tuple[str, ...]]]:
        usages: list[tuple[Card, int, Interaction, tuple[str, ...]]] = []
        for card in document.cards:
            for revision_number, revision in enumerate(
                card.revisions,
                start=1,
            ):
                if revision.hotspot_set is None:
                    continue
                for interaction in revision.hotspot_set.interactions:
                    roles = tuple(
                        role
                        for role, key_ids in (
                            ("Requires", interaction.conditions.requires),
                            ("Forbids", interaction.conditions.forbids),
                            ("Removes", interaction.key_changes.remove),
                            ("Grants", interaction.key_changes.grant),
                        )
                        if key_id in key_ids
                    )
                    if roles:
                        usages.append((card, revision_number, interaction, roles))
        return usages

    def render(self, document: Stack) -> None:
        name_draft = self.name_edit.text()
        previous_id = self._rendered_key_id
        desired_id = self._selected_key_id
        known_ids = {key.id for key in document.keys}
        if desired_id not in known_ids:
            desired_id = document.keys[0].id if document.keys else None
        if previous_id != desired_id:
            self._draft_dirty = False
        self._rendering = True
        try:
            with QSignalBlocker(self.key_list):
                self.key_list.clear()
                for key in document.keys:
                    count = len(self.key_usages(document, key.id))
                    suffix = "unused" if count == 0 else "1 use" if count == 1 else f"{count} uses"
                    item = QListWidgetItem(f"{key.name} — {suffix}")
                    item.setData(Qt.ItemDataRole.UserRole, key.id)
                    item.setToolTip(key.name)
                    self.key_list.addItem(item)
                selected_row = next(
                    (
                        row
                        for row in range(self.key_list.count())
                        if self.key_list.item(row).data(Qt.ItemDataRole.UserRole) == desired_id
                    ),
                    -1,
                )
                self.key_list.setCurrentRow(selected_row)
            self._selected_key_id = desired_id
            self._rendered_key_id = desired_id
            self.placeholder.setVisible(not document.keys)
            key = document.key_by_id(desired_id) if desired_id is not None else None
            self._render_properties(
                document,
                key,
                name_draft=(
                    name_draft if self._draft_dirty and previous_id == desired_id else None
                ),
            )
        finally:
            self._rendering = False
        self._update_enabled_state()

    def commit_pending_edits(self, *, render_change: bool) -> bool:
        if self._discard_pending_on_close or self._rendering or self._selected_key_id is None:
            return True
        if self._selected_key_id not in {key.id for key in self.controller.document.keys}:
            return True
        key = self.controller.document.key_by_id(self._selected_key_id)
        name = self.name_edit.text()
        if name.strip() == key.name:
            self._draft_dirty = False
            return True
        committed = self._execute(
            RenameKeyCommand(key_id=key.id, name=name),
            error_label=self.error_label,
            undo_message="Key renamed",
            render_change=render_change,
        )
        if committed:
            self._draft_dirty = False
        return committed

    def _render_properties(
        self,
        document: Stack,
        key: KeyDefinition | None,
        *,
        name_draft: str | None = None,
    ) -> None:
        usages = self.key_usages(document, key.id) if key is not None else []
        with QSignalBlocker(self.name_edit):
            self.name_edit.setText(
                name_draft if name_draft is not None else key.name if key is not None else ""
            )
        with QSignalBlocker(self.usage_list):
            self.usage_list.clear()
            for card, revision_number, interaction, _roles in usages:
                text = f"{card.name} V{revision_number}: {interaction.label}"
                item = QListWidgetItem(text)
                item.setData(
                    Qt.ItemDataRole.UserRole,
                    (
                        card.id,
                        card.revisions[revision_number - 1].id,
                        interaction.id,
                    ),
                )
                item.setToolTip(text)
                self.usage_list.addItem(item)
        self.delete_button.setToolTip(
            "Remove hotspot references before deleting this Key" if usages else "Delete unused Key"
        )

    def _selection_changed(
        self,
        current: QListWidgetItem | None,
        previous: QListWidgetItem | None,
    ) -> None:
        if self._rendering:
            return
        previous_id = previous.data(Qt.ItemDataRole.UserRole) if previous is not None else None
        if (
            isinstance(previous_id, UUID)
            and previous_id == self._selected_key_id
            and not self.commit_pending_edits(render_change=False)
        ):
            with QSignalBlocker(self.key_list):
                self.key_list.setCurrentItem(previous)
            return
        value = current.data(Qt.ItemDataRole.UserRole) if current is not None else None
        self._selected_key_id = value if isinstance(value, UUID) else None
        self._rendered_key_id = self._selected_key_id
        self._set_error(self.error_label, "")
        key = (
            self.controller.document.key_by_id(self._selected_key_id)
            if self._selected_key_id is not None
            else None
        )
        self._render_properties(self.controller.document, key)
        self._update_enabled_state()

    def _editing_finished(
        self,
        _next_focus: object,
        reason: object,
    ) -> None:
        mouse_focus = reason == Qt.FocusReason.MouseFocusReason
        if (
            not self.commit_pending_edits(render_change=not mouse_focus)
            or not mouse_focus
            or self._selected_key_id is None
        ):
            return
        document = self.controller.document
        key = document.key_by_id(self._selected_key_id)
        item = self.key_list.currentItem()
        if key is None or item is None:
            return
        usages = self.key_usages(document, key.id)
        suffix = "unused" if not usages else "1 use" if len(usages) == 1 else f"{len(usages)} uses"
        item.setText(f"{key.name} — {suffix}")
        item.setToolTip(key.name)
        for row, (card, revision_number, interaction, _roles) in enumerate(usages):
            usage_item = self.usage_list.item(row)
            if usage_item is None:
                continue
            text = f"{card.name} V{revision_number}: {interaction.label}"
            usage_item.setText(text)
            usage_item.setToolTip(text)

    def _draft_changed(self) -> None:
        if not self._rendering:
            self._draft_dirty = True

    def _add_key(self) -> None:
        if not self.commit_pending_edits(render_change=False):
            return
        document = self.controller.document
        names = {key.name.casefold() for key in document.keys}
        number = 1
        name = "New Key"
        while name.casefold() in names:
            number += 1
            name = f"New Key {number}"
        command = AddKeyCommand(name=name)
        self._selected_key_id = command.key_id
        if self._execute(
            command,
            error_label=self.error_label,
            undo_message="Key added",
        ):
            self.name_edit.setFocus()
            self.name_edit.selectAll()

    def _delete_key(self) -> None:
        if self._selected_key_id is None or not self.commit_pending_edits(render_change=False):
            return
        key_id = self._selected_key_id
        self._selected_key_id = None
        self._execute(
            DeleteKeyCommand(key_id=key_id),
            error_label=self.error_label,
            undo_message="Key deleted",
        )

    def _show_usage(self) -> None:
        if not self._mutation_allowed:
            return
        item = self.usage_list.currentItem()
        value = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
        if (
            isinstance(value, tuple)
            and len(value) == 3
            and all(isinstance(identifier, UUID) for identifier in value)
        ):
            self.hotspot_usage_requested.emit(*value)

    def _update_enabled_state(self) -> None:
        selected = self._selected_key_id is not None
        usages = (
            self.key_usages(
                self.controller.document,
                self._selected_key_id,
            )
            if self._selected_key_id is not None
            else []
        )
        self.key_list.setEnabled(self._mutation_allowed)
        self.add_button.setEnabled(self._mutation_allowed)
        self.name_edit.setEnabled(self._mutation_allowed and selected)
        self.delete_button.setEnabled(self._mutation_allowed and selected and not usages)


class SoundManagerWindow(_ControllerUtilityWindow):
    """Generate and manage the authoritative stack Sound catalog."""

    hotspot_usage_requested = Signal(object, object, object)

    def __init__(
        self,
        controller: DocumentController,
        workflow: SoundWorkflow,
        player: SoundPlayer,
        asset_path_resolver: Callable[[str], Path],
        parent: QWidget,
    ) -> None:
        super().__init__(
            controller,
            title="Sounds",
            object_name="soundManagerWindow",
            parent=parent,
        )
        self.workflow = workflow
        self.player = player
        self._asset_path_resolver = asset_path_resolver
        self._selected_sound_id: UUID | None = None
        self._rendered_sound_id: UUID | None = None
        self._draft_dirty = False
        self._generating_sound_id: UUID | None = None
        self._preview_token: UUID | None = None

        layout = QVBoxLayout(self)
        self.placeholder = QLabel("No Sounds yet.")
        layout.addWidget(self.placeholder)
        self.sound_list = QListWidget()
        self.sound_list.setObjectName("soundList")
        self.sound_list.setAccessibleName("Stack Sounds")
        layout.addWidget(self.sound_list, 1)

        controls = QHBoxLayout()
        controls.addStretch(1)
        self.delete_button = _compact_button(
            "−",
            object_name="deleteSoundButton",
            accessible_name="Delete Sound",
            tooltip="Delete unused Sound",
        )
        self.add_button = _compact_button(
            "+",
            object_name="addSoundButton",
            accessible_name="Add Sound",
            tooltip="Add Sound",
        )
        extent = max(
            self.delete_button.sizeHint().width(),
            self.delete_button.sizeHint().height(),
            self.add_button.sizeHint().width(),
            self.add_button.sizeHint().height(),
        )
        self.delete_button.setFixedSize(extent, extent)
        self.add_button.setFixedSize(extent, extent)
        controls.addWidget(self.delete_button)
        controls.addWidget(self.add_button)
        layout.addLayout(controls)

        layout.addWidget(QLabel("Name"))
        self.name_edit = _CommitLineEdit()
        self.name_edit.setObjectName("soundNameEdit")
        self.name_edit.setAccessibleName("Sound name")
        layout.addWidget(self.name_edit)
        layout.addWidget(QLabel("Prompt"))
        self.prompt_edit = _CommitPlainTextEdit()
        self.prompt_edit.setObjectName("soundPromptEdit")
        self.prompt_edit.setAccessibleName("Sound prompt")
        self.prompt_edit.setPlaceholderText("Describe the sound effect to generate")
        self.prompt_edit.setFixedHeight(self.prompt_edit.fontMetrics().lineSpacing() * 2 + 20)
        layout.addWidget(self.prompt_edit)

        duration_row = QHBoxLayout()
        duration_row.addWidget(QLabel("Duration"))
        self.duration_spin = QSpinBox()
        self.duration_spin.setObjectName("soundDurationSpin")
        self.duration_spin.setAccessibleName("Sound duration in seconds")
        self.duration_spin.setRange(1, 30)
        self.duration_spin.setSuffix(" s")
        duration_row.addWidget(self.duration_spin)
        duration_row.addStretch(1)
        layout.addLayout(duration_row)

        action_row = QHBoxLayout()
        self.generate_button = QPushButton("Generate")
        self.generate_button.setObjectName("generateSoundButton")
        self.preview_button = QPushButton("Play")
        self.preview_button.setObjectName("previewSoundButton")
        action_row.addWidget(self.generate_button)
        action_row.addWidget(self.preview_button)
        layout.addLayout(action_row)
        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("soundGenerationProgress")
        self.progress_bar.setRange(0, 8)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setObjectName("cancelSoundGenerationButton")
        self.cancel_button.setVisible(False)
        layout.addWidget(self.cancel_button)

        layout.addWidget(QLabel("Used By"))
        self.usage_list = QListWidget()
        self.usage_list.setObjectName("soundUsageList")
        self.usage_list.setAccessibleName("Sound usages")
        self.usage_list.setToolTip("Double-click a usage to show its hotspot")
        self.usage_list.setFixedHeight(UTILITY_USAGE_LIST_HEIGHT)
        layout.addWidget(self.usage_list)
        self.error_label = QLabel()
        self.error_label.setObjectName("soundValidationError")
        self.error_label.setWordWrap(True)
        self.error_label.setVisible(False)
        layout.addWidget(self.error_label)
        self._add_done_button(layout)

        self.sound_list.currentItemChanged.connect(self._selection_changed)
        self.add_button.clicked.connect(self._add_sound)
        self.delete_button.clicked.connect(self._delete_sound)
        self.name_edit.textChanged.connect(self._draft_changed)
        self.prompt_edit.textChanged.connect(self._draft_changed)
        self.duration_spin.valueChanged.connect(self._draft_changed)
        self.name_edit.editing_finished.connect(self._editing_finished)
        self.prompt_edit.editing_finished.connect(self._editing_finished)
        self.generate_button.clicked.connect(self._generate)
        self.preview_button.clicked.connect(self._toggle_preview)
        self.cancel_button.clicked.connect(self.workflow.cancel)
        self.usage_list.itemDoubleClicked.connect(lambda _item: self._show_usage())
        self.workflow.generation_started.connect(self._generation_started)
        self.workflow.sampling_progress.connect(self._sampling_progress)
        self.workflow.document_changed.connect(self._workflow_document_changed)
        self.workflow.change_applied.connect(self.change_applied)
        self.workflow.failed.connect(lambda message: self._set_error(self.error_label, message))
        self.workflow.finished.connect(self._generation_finished)
        self.render(controller.document)

    @property
    def selected_sound_id(self) -> UUID | None:
        return self._selected_sound_id

    @staticmethod
    def sound_usages(
        document: Stack,
        sound_id: UUID,
    ) -> list[tuple[Card, int, Interaction]]:
        usages: list[tuple[Card, int, Interaction]] = []
        for card in document.cards:
            for revision_number, revision in enumerate(card.revisions, start=1):
                if revision.hotspot_set is None:
                    continue
                usages.extend(
                    (card, revision_number, interaction)
                    for interaction in revision.hotspot_set.interactions
                    if interaction.sound_id == sound_id
                )
        return usages

    def render(self, document: Stack) -> None:
        previous_id = self._rendered_sound_id
        desired_id = self._selected_sound_id
        known_ids = {sound.id for sound in document.sounds}
        if desired_id not in known_ids:
            desired_id = document.sounds[0].id if document.sounds else None
        if previous_id != desired_id:
            self._draft_dirty = False
        draft = (
            (self.name_edit.text(), self.prompt_edit.toPlainText(), self.duration_spin.value())
            if self._draft_dirty and previous_id == desired_id
            else None
        )
        self._rendering = True
        try:
            with QSignalBlocker(self.sound_list):
                self.sound_list.clear()
                for sound in document.sounds:
                    count = len(self.sound_usages(document, sound.id))
                    suffix = "unused" if count == 0 else "1 use" if count == 1 else f"{count} uses"
                    item = QListWidgetItem(f"{sound.name} — {suffix}")
                    item.setData(Qt.ItemDataRole.UserRole, sound.id)
                    item.setToolTip(sound.name)
                    self.sound_list.addItem(item)
                row = next(
                    (
                        index
                        for index in range(self.sound_list.count())
                        if self.sound_list.item(index).data(Qt.ItemDataRole.UserRole) == desired_id
                    ),
                    -1,
                )
                self.sound_list.setCurrentRow(row)
            self._selected_sound_id = desired_id
            self._rendered_sound_id = desired_id
            self.placeholder.setVisible(not document.sounds)
            sound = document.sound_by_id(desired_id) if desired_id is not None else None
            self._render_properties(document, sound, draft=draft)
        finally:
            self._rendering = False
        self._update_enabled_state()

    def commit_pending_edits(self, *, render_change: bool) -> bool:
        if (
            self._discard_pending_on_close
            or self._rendering
            or not self._draft_dirty
            or self._selected_sound_id is None
        ):
            return True
        if self._selected_sound_id not in {sound.id for sound in self.controller.document.sounds}:
            return True
        sound = self.controller.document.sound_by_id(self._selected_sound_id)
        values = (
            self.name_edit.text(),
            self.prompt_edit.toPlainText(),
            self.duration_spin.value(),
        )
        if values == (sound.name, sound.prompt, sound.duration_seconds):
            self._draft_dirty = False
            return True
        committed = self._execute(
            UpdateSoundCommand(
                sound_id=sound.id,
                name=values[0],
                prompt=values[1],
                duration_seconds=values[2],
            ),
            error_label=self.error_label,
            undo_message="Sound updated",
            render_change=render_change,
        )
        if committed:
            self._draft_dirty = False
        return committed

    def _render_properties(
        self,
        document: Stack,
        sound: SoundDefinition | None,
        *,
        draft: tuple[str, str, int] | None,
    ) -> None:
        name, prompt, duration = (
            draft
            if draft is not None
            else (
                sound.name if sound is not None else "",
                sound.prompt if sound is not None else "",
                sound.duration_seconds if sound is not None else 2,
            )
        )
        with QSignalBlocker(self.name_edit):
            self.name_edit.setText(name)
        with QSignalBlocker(self.prompt_edit):
            self.prompt_edit.setPlainText(prompt)
        with QSignalBlocker(self.duration_spin):
            self.duration_spin.setValue(duration)
        usages = self.sound_usages(document, sound.id) if sound is not None else []
        with QSignalBlocker(self.usage_list):
            self.usage_list.clear()
            for card, revision_number, interaction in usages:
                text = f"{card.name} V{revision_number}: {interaction.label}"
                item = QListWidgetItem(text)
                item.setData(
                    Qt.ItemDataRole.UserRole,
                    (card.id, card.revisions[revision_number - 1].id, interaction.id),
                )
                item.setToolTip(text)
                self.usage_list.addItem(item)
        self.delete_button.setToolTip(
            "Remove hotspot references before deleting this Sound"
            if usages
            else "Delete unused Sound"
        )
        self.generate_button.setText(
            "Re-generate" if sound is not None and sound.generated is not None else "Generate"
        )

    def _selection_changed(
        self,
        current: QListWidgetItem | None,
        previous: QListWidgetItem | None,
    ) -> None:
        if self._rendering:
            return
        previous_id = previous.data(Qt.ItemDataRole.UserRole) if previous is not None else None
        if (
            isinstance(previous_id, UUID)
            and previous_id == self._selected_sound_id
            and not self.commit_pending_edits(render_change=False)
        ):
            with QSignalBlocker(self.sound_list):
                self.sound_list.setCurrentItem(previous)
            return
        self._stop_preview()
        value = current.data(Qt.ItemDataRole.UserRole) if current is not None else None
        self._selected_sound_id = value if isinstance(value, UUID) else None
        self._rendered_sound_id = self._selected_sound_id
        self._draft_dirty = False
        self._set_error(self.error_label, "")
        sound = (
            self.controller.document.sound_by_id(self._selected_sound_id)
            if self._selected_sound_id is not None
            else None
        )
        self._render_properties(self.controller.document, sound, draft=None)
        self._update_enabled_state()

    def _editing_finished(self, _next_focus: object, reason: object) -> None:
        mouse_focus = reason == Qt.FocusReason.MouseFocusReason
        if not self.commit_pending_edits(render_change=not mouse_focus) or not mouse_focus:
            return
        sound_id = self._selected_sound_id
        item = self.sound_list.currentItem()
        if sound_id is None or item is None:
            return
        sound = self.controller.document.sound_by_id(sound_id)
        count = len(self.sound_usages(self.controller.document, sound.id))
        suffix = "unused" if count == 0 else "1 use" if count == 1 else f"{count} uses"
        item.setText(f"{sound.name} — {suffix}")
        item.setToolTip(sound.name)

    def _draft_changed(self) -> None:
        if not self._rendering:
            self._draft_dirty = True
            self._update_enabled_state()

    def _add_sound(self) -> None:
        if not self.commit_pending_edits(render_change=False):
            return
        names = {sound.name.casefold() for sound in self.controller.document.sounds}
        number = 1
        name = "New Sound"
        while name.casefold() in names:
            number += 1
            name = f"New Sound {number}"
        command = AddSoundCommand(name=name)
        self._selected_sound_id = command.sound_id
        if self._execute(
            command,
            error_label=self.error_label,
            undo_message="Sound added",
        ):
            self.name_edit.setFocus()
            self.name_edit.selectAll()

    def _delete_sound(self) -> None:
        if self._selected_sound_id is None or not self.commit_pending_edits(render_change=False):
            return
        sound_id = self._selected_sound_id
        self._selected_sound_id = None
        self._stop_preview()
        self._execute(
            DeleteSoundCommand(sound_id=sound_id),
            error_label=self.error_label,
            undo_message="Sound deleted",
        )

    def _generate(self) -> None:
        if self._selected_sound_id is None or not self.commit_pending_edits(render_change=True):
            return
        try:
            self.workflow.generate(self._selected_sound_id)
        except SoundWorkflowError as error:
            self._set_error(self.error_label, str(error))

    def _generation_started(self, sound_id: object) -> None:
        self._generating_sound_id = sound_id if isinstance(sound_id, UUID) else None
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.cancel_button.setVisible(True)
        self._update_enabled_state()

    def _sampling_progress(self, step: int, total: int) -> None:
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(step)

    def _generation_finished(self) -> None:
        self._generating_sound_id = None
        self.progress_bar.setVisible(False)
        self.cancel_button.setVisible(False)
        self.render(self.controller.document)

    def _workflow_document_changed(self, document: object) -> None:
        if isinstance(document, Stack):
            self.document_changed.emit(document)
            self.render(document)

    def _toggle_preview(self) -> None:
        if self._preview_token is not None:
            self._stop_preview()
            return
        if self._selected_sound_id is None:
            return
        sound = self.controller.document.sound_by_id(self._selected_sound_id)
        if sound.generated is None:
            return
        try:
            path = self._asset_path_resolver(sound.generated.audio_path)
        except (OSError, ValueError) as error:
            self._set_error(self.error_label, str(error))
            return
        token = uuid4()
        self._preview_token = token
        self.player.play(path)
        self.preview_button.setText("Stop")
        QTimer.singleShot(
            sound.generated.provenance.duration_seconds * 1_000,
            lambda: self._preview_finished(token),
        )

    def _preview_finished(self, token: UUID) -> None:
        if token == self._preview_token:
            self._stop_preview()

    def _stop_preview(self) -> None:
        if self._preview_token is not None:
            self.player.stop()
        self._preview_token = None
        self.preview_button.setText("Play")

    def _show_usage(self) -> None:
        if not self._mutation_allowed:
            return
        item = self.usage_list.currentItem()
        value = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
        if (
            isinstance(value, tuple)
            and len(value) == 3
            and all(isinstance(identifier, UUID) for identifier in value)
        ):
            self.hotspot_usage_requested.emit(*value)

    def _update_enabled_state(self) -> None:
        selected = self._selected_sound_id is not None
        generating = self.workflow.is_active
        sound = self.controller.document.sound_by_id(self._selected_sound_id) if selected else None
        usages = self.sound_usages(self.controller.document, sound.id) if sound else []
        can_edit = self._mutation_allowed and selected and not generating
        self.sound_list.setEnabled(self._mutation_allowed and not generating)
        self.add_button.setEnabled(self._mutation_allowed and not generating)
        self.delete_button.setEnabled(can_edit and not usages)
        self.name_edit.setEnabled(can_edit)
        self.prompt_edit.setEnabled(can_edit)
        self.duration_spin.setEnabled(can_edit)
        self.generate_button.setEnabled(can_edit and bool(self.prompt_edit.toPlainText().strip()))
        self.preview_button.setEnabled(sound is not None and sound.generated is not None)
        self.cancel_button.setEnabled(generating)

    def closeEvent(self, event: QCloseEvent) -> None:
        if not self._discard_pending_on_close and not self.commit_pending_edits(
            render_change=False
        ):
            event.ignore()
            return
        self._stop_preview()
        self.workflow.cancel()
        super().closeEvent(event)


__all__ = ["KeyManagerWindow", "SoundManagerWindow", "StyleManagerWindow"]
