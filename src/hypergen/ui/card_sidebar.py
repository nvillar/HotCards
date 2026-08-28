"""Controller-backed ordered card sidebar."""

from __future__ import annotations

from collections.abc import Callable, Collection
from pathlib import Path
from uuid import UUID

from PySide6.QtCore import QSignalBlocker, QSize, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from hypergen.application.commands import (
    CreateCardCommand,
    DeleteCardCommand,
    RenameCardCommand,
    ReorderCardCommand,
    SetStartCardCommand,
)
from hypergen.application.document_controller import DocumentController
from hypergen.domain.models import Card, Stack

ImagePathResolver = Callable[[str], Path | None]


class CardSidebar(QWidget):
    """Render card order and route all document mutations through the controller."""

    card_selected = Signal(object)
    delete_requested = Signal(object)
    document_changed = Signal(object)

    def __init__(
        self,
        controller: DocumentController,
        parent: QWidget | None = None,
        *,
        image_path_resolver: ImagePathResolver | None = None,
    ) -> None:
        super().__init__(parent)
        self.controller = controller
        self._image_path_resolver = image_path_resolver
        self.setObjectName("cardSidebar")
        self.setMinimumWidth(180)

        self.card_list = QListWidget()
        self.card_list.setObjectName("cardList")
        self.card_list.setIconSize(QSize(72, 48))
        self.card_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.card_list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.card_list.currentItemChanged.connect(self._selection_changed)
        self.card_list.model().rowsMoved.connect(self._rows_moved)

        self.add_button = QToolButton()
        self.add_button.setObjectName("addCardButton")
        self.add_button.setText("+")
        self.add_button.setAccessibleName("Add card")
        self.add_button.setToolTip("Add a new card")
        self.add_button.clicked.connect(self.add_card)
        self.start_button = QToolButton()
        self.start_button.setObjectName("setStartCardButton")
        self.start_button.setIcon(self._star_icon())
        self.start_button.setAccessibleName("Make start card")
        self.start_button.setToolTip("Make the selected card the start card")
        self.start_button.clicked.connect(self.set_selected_as_start)
        self.move_up_button = QToolButton()
        self.move_up_button.setObjectName("moveCardUpButton")
        self.move_up_button.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowUp)
        )
        self.move_up_button.setAccessibleName("Move card up")
        self.move_up_button.setToolTip("Move card up")
        self.move_up_button.clicked.connect(lambda: self._move_selected(-1))
        self.move_down_button = QToolButton()
        self.move_down_button.setObjectName("moveCardDownButton")
        self.move_down_button.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowDown)
        )
        self.move_down_button.setAccessibleName("Move card down")
        self.move_down_button.setToolTip("Move card down")
        self.move_down_button.clicked.connect(lambda: self._move_selected(1))
        self.delete_button = QToolButton()
        self.delete_button.setObjectName("deleteCardButton")
        self.delete_button.setText("−")
        self.delete_button.setAccessibleName("Delete card")
        self.delete_button.setToolTip("Delete the selected card")
        self.delete_button.clicked.connect(self._request_delete)
        for button in (self.add_button, self.delete_button):
            font = button.font()
            font.setPointSizeF(max(font.pointSizeF() + 4.0, 16.0))
            font.setBold(True)
            button.setFont(font)
        control_extent = max(
            max(button.sizeHint().width(), button.sizeHint().height())
            for button in (
                self.move_up_button,
                self.move_down_button,
                self.start_button,
                self.add_button,
                self.delete_button,
            )
        )
        for button in (
            self.move_up_button,
            self.move_down_button,
            self.start_button,
            self.add_button,
            self.delete_button,
        ):
            button.setFixedSize(control_extent, control_extent)
        self._document_editable = True

        self.empty_label = QLabel("No cards yet")
        self.empty_label.setObjectName("emptyCardListLabel")
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.card_actions = QHBoxLayout()
        self.card_actions.addWidget(self.move_up_button)
        self.card_actions.addWidget(self.move_down_button)
        self.card_actions.addStretch(1)
        self.card_actions.addWidget(self.start_button)
        self.card_actions.addWidget(self.delete_button)
        self.card_actions.addWidget(self.add_button)

        layout = QVBoxLayout(self)
        layout.addWidget(self.card_list, 1)
        layout.addWidget(self.empty_label)
        layout.addLayout(self.card_actions)
        self.render(controller.document)

    @property
    def selected_card_id(self) -> UUID | None:
        item = self.card_list.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item is not None else None

    def render(
        self,
        document: Stack,
        selected_card_id: UUID | None = None,
        *,
        draft_card_ids: Collection[UUID] = (),
    ) -> None:
        """Render a controller snapshot while preserving valid selection."""
        current_card_id = self.selected_card_id
        desired = selected_card_id if selected_card_id is not None else current_card_id
        scroll_bar = self.card_list.verticalScrollBar()
        scroll_position = scroll_bar.value()
        draft_ids = frozenset(draft_card_ids)
        with QSignalBlocker(self.card_list):
            self.card_list.clear()
            for card in document.cards:
                is_start = card.id == document.start_card_id
                label = f"★  {card.name}" if is_start else card.name
                has_draft = card.id in draft_ids
                if has_draft:
                    label = f"{label}  · Draft"
                item = QListWidgetItem(self._card_icon(card), label)
                item.setData(Qt.ItemDataRole.UserRole, card.id)
                states = [
                    state
                    for state, active in (
                        ("Start card", is_start),
                        ("Background draft pending", has_draft),
                    )
                    if active
                ]
                item.setToolTip(" · ".join(states) if states else card.name)
                self.card_list.addItem(item)
            selected_row = self._row_for(desired)
            preserve_scroll = selected_row >= 0 and desired == current_card_id
            if selected_row < 0 and self.card_list.count():
                selected_row = 0
            self.card_list.setCurrentRow(selected_row)
            if preserve_scroll:
                scroll_bar.setValue(scroll_position)
        self.empty_label.setVisible(not document.cards)
        self._update_buttons()

    def select_card(self, card_id: UUID | None) -> None:
        row = self._row_for(card_id)
        self.card_list.setCurrentRow(row)

    def set_document_editable(self, editable: bool) -> None:
        """Enable mutations only after the application has a durable document."""
        self._document_editable = editable
        self.card_list.setDragEnabled(editable)
        self.add_button.setEnabled(editable)
        self._update_buttons()

    def add_card(self, name: str | None = None) -> UUID:
        """Create and select a uniquely named blank card."""
        document = self.controller.document
        card_name = name or self._next_card_name(document)
        command = CreateCardCommand(name=card_name)
        changed = self.controller.execute(command)
        self.render(changed, command.card_id)
        self.card_selected.emit(command.card_id)
        self.document_changed.emit(changed)
        return command.card_id

    def rename_selected(self, name: str) -> None:
        card_id = self.selected_card_id
        if card_id is None:
            return
        changed = self.controller.execute(RenameCardCommand(card_id=card_id, name=name))
        self.render(changed, card_id)
        self.document_changed.emit(changed)

    def set_selected_as_start(self) -> None:
        card_id = self.selected_card_id
        if card_id is None:
            return
        changed = self.controller.execute(SetStartCardCommand(card_id=card_id))
        self.render(changed, card_id)
        self.document_changed.emit(changed)

    def move_card(self, card_id: UUID, new_index: int) -> None:
        changed = self.controller.execute(ReorderCardCommand(card_id=card_id, new_index=new_index))
        self.render(changed, card_id)
        self.document_changed.emit(changed)

    def delete_card(self, card_id: UUID) -> None:
        """Delete one card through a typed, undoable controller command."""
        changed = self.controller.execute(DeleteCardCommand(card_id=card_id))
        self.render(changed)
        self.document_changed.emit(changed)

    def _selection_changed(
        self,
        current: QListWidgetItem | None,
        _previous: QListWidgetItem | None,
    ) -> None:
        self._update_buttons()
        self.card_selected.emit(
            current.data(Qt.ItemDataRole.UserRole) if current is not None else None
        )

    def _move_selected(self, offset: int) -> None:
        card_id = self.selected_card_id
        row = self.card_list.currentRow()
        destination = row + offset
        if card_id is not None and 0 <= destination < self.card_list.count():
            self.move_card(card_id, destination)

    def _request_delete(self) -> None:
        card_id = self.selected_card_id
        if card_id is not None:
            self.delete_requested.emit(card_id)

    def _rows_moved(
        self,
        _source_parent: object,
        source_start: int,
        source_end: int,
        _destination_parent: object,
        destination_row: int,
    ) -> None:
        moved_count = source_end - source_start + 1
        final_row = (
            destination_row if destination_row < source_start else destination_row - moved_count
        )
        item = self.card_list.item(final_row)
        if item is None:
            return
        card_id = item.data(Qt.ItemDataRole.UserRole)
        document = self.controller.execute(ReorderCardCommand(card_id=card_id, new_index=final_row))
        selected = self.selected_card_id
        self.render(document, selected)
        self.document_changed.emit(document)

    def _update_buttons(self) -> None:
        row = self.card_list.currentRow()
        has_selection = self._document_editable and row >= 0
        self.add_button.setEnabled(self._document_editable)
        self.start_button.setEnabled(has_selection)
        self.move_up_button.setEnabled(has_selection and row > 0)
        self.move_down_button.setEnabled(has_selection and row < self.card_list.count() - 1)
        self.delete_button.setEnabled(has_selection)

    def _row_for(self, card_id: UUID | None) -> int:
        for row in range(self.card_list.count()):
            if self.card_list.item(row).data(Qt.ItemDataRole.UserRole) == card_id:
                return row
        return -1

    @staticmethod
    def _next_card_name(document: Stack) -> str:
        names = {card.name.casefold() for card in document.cards}
        number = len(document.cards) + 1
        while f"Card {number}".casefold() in names:
            number += 1
        return f"Card {number}"

    @staticmethod
    def _placeholder_icon() -> QIcon:
        pixmap = QPixmap(72, 48)
        pixmap.fill(QColor("#d7d9dc"))
        return QIcon(pixmap)

    def _card_icon(self, card: Card) -> QIcon:
        background = card.active_revision.background
        if background is None or self._image_path_resolver is None:
            return self._placeholder_icon()
        path = self._image_path_resolver(background.image_path)
        if path is None:
            return self._placeholder_icon()
        source = QPixmap(str(path))
        if source.isNull():
            return self._placeholder_icon()
        scaled = source.scaled(
            72,
            48,
            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.SmoothTransformation,
        )
        x = max(0, (scaled.width() - 72) // 2)
        y = max(0, (scaled.height() - 48) // 2)
        return QIcon(scaled.copy(x, y, 72, 48))

    @staticmethod
    def _star_icon() -> QIcon:
        pixmap = QPixmap(20, 20)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        font = painter.font()
        font.setPointSize(15)
        painter.setFont(font)
        painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "★")
        painter.end()
        return QIcon(pixmap)


__all__ = ["CardSidebar"]
