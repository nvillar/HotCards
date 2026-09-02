"""Searchable modeless Sound picker window."""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from PySide6.QtCore import QPoint, QSize, Qt, Signal
from PySide6.QtGui import QHideEvent, QMouseEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListView,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from hotcards.domain.models import SoundDefinition

SOUND_GRID_SIZE = QSize(160, 112)
DEFAULT_COLUMN_COUNT = 3
WINDOW_SIZE = QSize(SOUND_GRID_SIZE.width() * DEFAULT_COLUMN_COUNT + 44, 420)
SEARCH_ROLE = Qt.ItemDataRole.UserRole + 1


class _ClickableLabel(QLabel):
    clicked = Signal()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self.rect().contains(
            event.position().toPoint()
        ):
            self.clicked.emit()
            event.accept()
            return
        super().mouseReleaseEvent(event)


class _SoundTile(QWidget):
    selected = Signal(object)
    preview_requested = Signal(object)

    def __init__(
        self,
        sound: SoundDefinition,
        *,
        parent: QWidget,
    ) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(10)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.play_button = QToolButton()
        self.play_button.setObjectName("soundPickerPlayButton")
        self.play_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.play_button.setIconSize(QSize(28, 28))
        self.play_button.setFixedSize(42, 42)
        self.play_button.setAccessibleName(f"Play {sound.name}")
        self.play_button.setToolTip(f"Play {sound.name}")
        self.play_button.setEnabled(sound.generated is not None)
        self.play_button.clicked.connect(lambda: self.preview_requested.emit(sound.id))
        layout.addWidget(
            self.play_button,
            alignment=Qt.AlignmentFlag.AlignHCenter,
        )

        self.name_label = _ClickableLabel(sound.name)
        self.name_label.setObjectName("soundPickerNameLabel")
        self.name_label.setAccessibleName(f"Select {sound.name}")
        self.name_label.setToolTip(sound.name)
        self.name_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.name_label.setWordWrap(True)
        self.name_label.setCursor(Qt.CursorShape.PointingHandCursor)
        self.name_label.clicked.connect(lambda: self.selected.emit(sound.id))
        layout.addWidget(self.name_label)


class SoundPickerWindow(QWidget):
    """Choose or preview one Sound from a searchable adaptive grid."""

    sound_selected = Signal(object)
    preview_requested = Signal(object)
    dismissed = Signal()

    def __init__(self, parent: QWidget) -> None:
        super().__init__(
            parent,
            Qt.WindowType.Tool | Qt.WindowType.WindowStaysOnTopHint,
        )
        self._positioned = False
        self._tiles: dict[UUID, _SoundTile] = {}
        self.setObjectName("soundPickerWindow")
        self.setWindowTitle("Select Sound")
        self.resize(WINDOW_SIZE)

        layout = QVBoxLayout(self)
        self.search_edit = QLineEdit()
        self.search_edit.setObjectName("soundPickerSearch")
        self.search_edit.setAccessibleName("Search Sounds")
        self.search_edit.setPlaceholderText("Search Sounds")
        layout.addWidget(self.search_edit)

        self.sound_list = QListWidget()
        self.sound_list.setObjectName("soundPickerList")
        self.sound_list.setAccessibleName("Sounds")
        self.sound_list.setViewMode(QListView.ViewMode.IconMode)
        self.sound_list.setResizeMode(QListView.ResizeMode.Adjust)
        self.sound_list.setMovement(QListView.Movement.Static)
        self.sound_list.setWrapping(True)
        self.sound_list.setUniformItemSizes(True)
        self.sound_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.sound_list.setGridSize(SOUND_GRID_SIZE)
        layout.addWidget(self.sound_list, 1)

        self.empty_label = QLabel("No Sounds")
        self.empty_label.setObjectName("soundPickerEmptyLabel")
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_label.setVisible(False)
        layout.addWidget(self.empty_label)

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setObjectName("soundPickerCancelButton")
        self.cancel_button.setAccessibleName("Cancel")
        self.cancel_button.clicked.connect(self.close)
        button_row.addWidget(self.cancel_button)
        layout.addLayout(button_row)

        self.search_edit.textChanged.connect(self._filter_sounds)
        self.search_edit.returnPressed.connect(self._choose_current)
        self.sound_list.itemActivated.connect(self._choose_item)

    def open_for(
        self,
        anchor: QWidget,
        sounds: Sequence[SoundDefinition],
        *,
        selected_sound_id: UUID | None,
    ) -> None:
        """Populate and show the picker beside its invoking control."""
        self.search_edit.clear()
        self.sound_list.clear()
        self._tiles.clear()
        selected_item: QListWidgetItem | None = None
        first_item: QListWidgetItem | None = None
        for sound in sounds:
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, sound.id)
            item.setData(SEARCH_ROLE, sound.name)
            item.setSizeHint(SOUND_GRID_SIZE)
            self.sound_list.addItem(item)
            tile = _SoundTile(
                sound,
                parent=self.sound_list,
            )
            tile.selected.connect(self._select_sound)
            tile.preview_requested.connect(self.preview_requested)
            self.sound_list.setItemWidget(item, tile)
            self._tiles[sound.id] = tile
            if first_item is None:
                first_item = item
            if sound.id == selected_sound_id:
                selected_item = item
        current_item = selected_item or first_item
        if current_item is not None:
            self.sound_list.setCurrentItem(current_item)
            self.sound_list.scrollToItem(
                current_item,
                QAbstractItemView.ScrollHint.PositionAtCenter,
            )
        self._filter_sounds("")
        self._show_at(anchor)
        self.search_edit.setFocus()

    def tile(self, sound_id: UUID) -> _SoundTile | None:
        """Return the visible tile for focused UI coordination and tests."""
        return self._tiles.get(sound_id)

    def hideEvent(self, event: QHideEvent) -> None:
        super().hideEvent(event)
        self.dismissed.emit()

    def _filter_sounds(self, text: str) -> None:
        query = text.strip().casefold()
        visible_count = 0
        first_item: QListWidgetItem | None = None
        for index in range(self.sound_list.count()):
            item = self.sound_list.item(index)
            name = item.data(SEARCH_ROLE)
            visible = isinstance(name, str) and (not query or query in name.casefold())
            item.setHidden(not visible)
            if visible:
                visible_count += 1
                if first_item is None:
                    first_item = item
        current = self.sound_list.currentItem()
        if current is None or current.isHidden():
            if first_item is not None:
                self.sound_list.setCurrentItem(first_item)
            else:
                self.sound_list.setCurrentRow(-1)
        self.empty_label.setText(
            "No Sounds" if not query and not visible_count else "No matching Sounds"
        )
        self.empty_label.setVisible(visible_count == 0)
        self.sound_list.setVisible(visible_count > 0)

    def _choose_current(self) -> None:
        self._choose_item(self.sound_list.currentItem())

    def _choose_item(self, item: QListWidgetItem | None) -> None:
        if item is None or item.isHidden():
            return
        self._select_sound(item.data(Qt.ItemDataRole.UserRole))

    def _select_sound(self, sound_id: object) -> None:
        if isinstance(sound_id, UUID):
            self.sound_selected.emit(sound_id)
            self.hide()

    def _show_at(self, anchor: QWidget) -> None:
        if not self._positioned:
            available = anchor.screen().availableGeometry()
            width = min(WINDOW_SIZE.width(), available.width() - 24)
            height = min(WINDOW_SIZE.height(), available.height() - 24)
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
    "DEFAULT_COLUMN_COUNT",
    "SOUND_GRID_SIZE",
    "SoundPickerWindow",
    "WINDOW_SIZE",
]
