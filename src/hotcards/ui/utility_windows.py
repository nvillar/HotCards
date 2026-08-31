"""Modeless stack-global Style and Key managers."""

from __future__ import annotations

from uuid import UUID

from pydantic import ValidationError
from PySide6.QtCore import QSignalBlocker, Qt, Signal
from PySide6.QtGui import QCloseEvent, QFocusEvent
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from hotcards.application.commands import (
    AddKeyCommand,
    AddStyleCommand,
    CommandError,
    DeleteKeyCommand,
    DeleteStyleCommand,
    DocumentCommand,
    RenameKeyCommand,
    UpdateStyleCommand,
)
from hotcards.application.document_controller import (
    DocumentController,
    DocumentMutationBlockedError,
)
from hotcards.domain.models import (
    Card,
    Interaction,
    KeyDefinition,
    Stack,
    StyleDefinition,
)


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
        super().__init__(parent, Qt.WindowType.Window)
        self.controller = controller
        self._rendering = False
        self._mutation_allowed = True
        self._discard_pending_on_close = False
        self.setWindowTitle(title)
        self.setObjectName(object_name)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.resize(420, 560)

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
        layout.addWidget(self.usage_list, 1)
        self.show_usage_button = QPushButton("Show Hotspot")
        self.show_usage_button.setObjectName("showHotspotUsageButton")
        self.show_usage_button.setToolTip("Show the selected Key reference in Hotspots")
        layout.addWidget(self.show_usage_button)
        self.error_label = QLabel()
        self.error_label.setObjectName("keyValidationError")
        self.error_label.setWordWrap(True)
        self.error_label.setVisible(False)
        layout.addWidget(self.error_label)

        self.key_list.currentItemChanged.connect(self._selection_changed)
        self.add_button.clicked.connect(self._add_key)
        self.delete_button.clicked.connect(self._delete_key)
        self.name_edit.editing_finished.connect(self._editing_finished)
        self.name_edit.returnPressed.connect(lambda: self.commit_pending_edits(render_change=True))
        self.name_edit.textChanged.connect(self._draft_changed)
        self.usage_list.currentItemChanged.connect(
            lambda current, _previous: self.show_usage_button.setEnabled(
                current is not None and self._mutation_allowed
            )
        )
        self.show_usage_button.clicked.connect(self._show_usage)
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
            for card, revision_number, interaction, roles in usages:
                text = (
                    f"{card.name} · Version {revision_number}\n"
                    f"{interaction.label} · {', '.join(roles)}"
                )
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
        self.show_usage_button.setEnabled(False)
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
        for row, (card, revision_number, interaction, roles) in enumerate(usages):
            usage_item = self.usage_list.item(row)
            if usage_item is None:
                continue
            text = (
                f"{card.name} · Version {revision_number}\n{interaction.label} · {', '.join(roles)}"
            )
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
        self.show_usage_button.setEnabled(
            self._mutation_allowed and self.usage_list.currentItem() is not None
        )


__all__ = ["KeyManagerWindow", "StyleManagerWindow"]
