"""Controller-backed ordered card sidebar."""

from __future__ import annotations

from uuid import UUID

from PySide6.QtCore import QSignalBlocker, QSize, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QGridLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from hypergen.application.commands import (
    CreateCardCommand,
    RenameCardCommand,
    ReorderCardCommand,
    SetStartCardCommand,
)
from hypergen.application.document_controller import DocumentController
from hypergen.domain.models import Stack


class CardSidebar(QWidget):
    """Render card order and route all document mutations through the controller."""

    card_selected = Signal(object)
    document_changed = Signal(object)

    def __init__(self, controller: DocumentController, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.controller = controller
        self.setObjectName("cardSidebar")
        self.setMinimumWidth(180)

        heading = QLabel("Cards")
        heading.setObjectName("cardSidebarHeading")
        self.card_list = QListWidget()
        self.card_list.setObjectName("cardList")
        self.card_list.setIconSize(QSize(72, 48))
        self.card_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.card_list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.card_list.currentItemChanged.connect(self._selection_changed)
        self.card_list.model().rowsMoved.connect(self._rows_moved)

        self.add_button = QPushButton("Add Card")
        self.add_button.setObjectName("addCardButton")
        self.add_button.clicked.connect(self.add_card)
        self.start_button = QPushButton("Make Start")
        self.start_button.setObjectName("setStartCardButton")
        self.start_button.clicked.connect(self.set_selected_as_start)
        self.move_up_button = QToolButton()
        self.move_up_button.setObjectName("moveCardUpButton")
        self.move_up_button.setText("↑")
        self.move_up_button.setToolTip("Move card up")
        self.move_up_button.clicked.connect(lambda: self._move_selected(-1))
        self.move_down_button = QToolButton()
        self.move_down_button.setObjectName("moveCardDownButton")
        self.move_down_button.setText("↓")
        self.move_down_button.setToolTip("Move card down")
        self.move_down_button.clicked.connect(lambda: self._move_selected(1))

        self.empty_label = QLabel("No cards yet")
        self.empty_label.setObjectName("emptyCardListLabel")
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        buttons = QGridLayout()
        buttons.addWidget(self.add_button, 0, 0)
        buttons.addWidget(self.start_button, 0, 1)
        buttons.addWidget(self.move_up_button, 1, 0)
        buttons.addWidget(self.move_down_button, 1, 1)

        layout = QVBoxLayout(self)
        layout.addWidget(heading)
        layout.addWidget(self.card_list, 1)
        layout.addWidget(self.empty_label)
        layout.addLayout(buttons)
        self.render(controller.document)

    @property
    def selected_card_id(self) -> UUID | None:
        item = self.card_list.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item is not None else None

    def render(self, document: Stack, selected_card_id: UUID | None = None) -> None:
        """Render a controller snapshot while preserving valid selection."""
        desired = selected_card_id if selected_card_id is not None else self.selected_card_id
        with QSignalBlocker(self.card_list):
            self.card_list.clear()
            for card in document.cards:
                is_start = card.id == document.start_card_id
                label = f"★  {card.name}" if is_start else card.name
                item = QListWidgetItem(self._placeholder_icon(), label)
                item.setData(Qt.ItemDataRole.UserRole, card.id)
                item.setToolTip("Start card" if is_start else card.name)
                self.card_list.addItem(item)
            selected_row = self._row_for(desired)
            if selected_row < 0 and self.card_list.count():
                selected_row = 0
            self.card_list.setCurrentRow(selected_row)
        self.empty_label.setVisible(not document.cards)
        self._update_buttons()

    def select_card(self, card_id: UUID | None) -> None:
        row = self._row_for(card_id)
        self.card_list.setCurrentRow(row)

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
        has_selection = row >= 0
        self.start_button.setEnabled(has_selection)
        self.move_up_button.setEnabled(has_selection and row > 0)
        self.move_down_button.setEnabled(has_selection and row < self.card_list.count() - 1)

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


__all__ = ["CardSidebar"]
