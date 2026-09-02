"""Searchable modeless card picker window."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from uuid import UUID

from PySide6.QtCore import QPoint, QSize, Qt, Signal
from PySide6.QtGui import QHideEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListView,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from hotcards.domain.models import Card
from hotcards.ui.card_thumbnails import ImagePathResolver, card_thumbnail_icon

THUMBNAIL_SIZE = QSize(144, 96)
GRID_SIZE = QSize(172, 136)
DEFAULT_COLUMN_COUNT = 3
POPOVER_SIZE = QSize(GRID_SIZE.width() * DEFAULT_COLUMN_COUNT + 44, 480)


class CardPickerWindow(QWidget):
    """Choose one card from a searchable adaptive thumbnail grid."""

    card_selected = Signal(object)
    dismissed = Signal()

    def __init__(
        self,
        parent: QWidget,
        *,
        image_path_resolver: ImagePathResolver | None = None,
    ) -> None:
        super().__init__(
            parent,
            Qt.WindowType.Tool | Qt.WindowType.WindowStaysOnTopHint,
        )
        self._image_path_resolver = image_path_resolver
        self._positioned = False
        self.setObjectName("cardPickerWindow")
        self.resize(POPOVER_SIZE)

        layout = QVBoxLayout(self)
        self.search_edit = QLineEdit()
        self.search_edit.setObjectName("cardPickerSearch")
        self.search_edit.setAccessibleName("Search cards")
        self.search_edit.setPlaceholderText("Search cards")
        layout.addWidget(self.search_edit)

        self.card_list = QListWidget()
        self.card_list.setObjectName("cardPickerList")
        self.card_list.setAccessibleName("Cards")
        self.card_list.setViewMode(QListView.ViewMode.IconMode)
        self.card_list.setResizeMode(QListView.ResizeMode.Adjust)
        self.card_list.setMovement(QListView.Movement.Static)
        self.card_list.setWrapping(True)
        self.card_list.setWordWrap(True)
        self.card_list.setUniformItemSizes(True)
        self.card_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.card_list.setIconSize(THUMBNAIL_SIZE)
        self.card_list.setGridSize(GRID_SIZE)
        layout.addWidget(self.card_list, 1)

        self.empty_label = QLabel("No matching cards")
        self.empty_label.setObjectName("cardPickerEmptyLabel")
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_label.setVisible(False)
        layout.addWidget(self.empty_label)

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setObjectName("cardPickerCancelButton")
        self.cancel_button.setAccessibleName("Cancel")
        self.cancel_button.clicked.connect(self.close)
        button_row.addWidget(self.cancel_button)
        layout.addLayout(button_row)

        self.search_edit.textChanged.connect(self._filter_cards)
        self.search_edit.returnPressed.connect(self._choose_current)
        self.card_list.itemClicked.connect(self._choose_item)
        self.card_list.itemActivated.connect(self._choose_item)

    def open_for(
        self,
        anchor: QWidget,
        cards: Sequence[Card],
        *,
        title: str,
        enabled_card_ids: Collection[UUID],
        selected_card_id: UUID | None,
    ) -> None:
        """Populate and show the picker beside its invoking control."""
        enabled_ids = frozenset(enabled_card_ids)
        self.setWindowTitle(title)
        self.search_edit.clear()
        self.card_list.clear()
        selected_item: QListWidgetItem | None = None
        first_enabled_item: QListWidgetItem | None = None
        for card in cards:
            item = QListWidgetItem(
                card_thumbnail_icon(
                    card,
                    size=THUMBNAIL_SIZE,
                    image_path_resolver=self._image_path_resolver,
                ),
                card.name,
            )
            item.setData(Qt.ItemDataRole.UserRole, card.id)
            item.setTextAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
            if card.id not in enabled_ids:
                item.setFlags(
                    item.flags() & ~Qt.ItemFlag.ItemIsEnabled & ~Qt.ItemFlag.ItemIsSelectable
                )
            elif first_enabled_item is None:
                first_enabled_item = item
            self.card_list.addItem(item)
            if card.id == selected_card_id:
                selected_item = item
        current_item = first_enabled_item
        if selected_item is not None and selected_item.flags() & Qt.ItemFlag.ItemIsEnabled:
            current_item = selected_item
        if current_item is not None:
            self.card_list.setCurrentItem(current_item)
            self.card_list.scrollToItem(
                current_item,
                QAbstractItemView.ScrollHint.PositionAtCenter,
            )
        self._filter_cards("")
        self._show_at(anchor)
        self.search_edit.setFocus()

    def hideEvent(self, event: QHideEvent) -> None:
        super().hideEvent(event)
        self.dismissed.emit()

    def _filter_cards(self, text: str) -> None:
        query = text.strip().casefold()
        visible_count = 0
        first_enabled_item: QListWidgetItem | None = None
        for index in range(self.card_list.count()):
            item = self.card_list.item(index)
            visible = not query or query in item.text().casefold()
            item.setHidden(not visible)
            if visible:
                visible_count += 1
                if first_enabled_item is None and item.flags() & Qt.ItemFlag.ItemIsEnabled:
                    first_enabled_item = item
        current = self.card_list.currentItem()
        if current is None or current.isHidden() or not current.flags() & Qt.ItemFlag.ItemIsEnabled:
            if first_enabled_item is not None:
                self.card_list.setCurrentItem(first_enabled_item)
            else:
                self.card_list.setCurrentRow(-1)
        self.empty_label.setVisible(visible_count == 0)
        self.card_list.setVisible(visible_count > 0)

    def _choose_current(self) -> None:
        self._choose_item(self.card_list.currentItem())

    def _choose_item(self, item: QListWidgetItem | None) -> None:
        if item is None or item.isHidden() or not item.flags() & Qt.ItemFlag.ItemIsEnabled:
            return
        card_id = item.data(Qt.ItemDataRole.UserRole)
        if isinstance(card_id, UUID):
            self.card_selected.emit(card_id)
            self.hide()

    def _show_at(self, anchor: QWidget) -> None:
        if not self._positioned:
            available = anchor.screen().availableGeometry()
            width = min(POPOVER_SIZE.width(), available.width() - 24)
            height = min(POPOVER_SIZE.height(), available.height() - 24)
            self.resize(width, height)
            below = anchor.mapToGlobal(QPoint(0, anchor.height()))
            x = min(max(below.x(), available.left()), available.right() - width + 1)
            y = below.y()
            if y + height > available.bottom() + 1:
                y = anchor.mapToGlobal(QPoint(0, 0)).y() - height
            y = min(max(y, available.top()), available.bottom() - height + 1)
            self.move(x, y)
            self._positioned = True
        self.show()
        self.raise_()
        self.activateWindow()


__all__ = [
    "CardPickerWindow",
    "DEFAULT_COLUMN_COUNT",
    "GRID_SIZE",
    "POPOVER_SIZE",
    "THUMBNAIL_SIZE",
]
