"""One predictable surface for transient application feedback."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStyle,
    QToolButton,
)


class NotificationKind(StrEnum):
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class NotificationAction:
    action_id: str
    label: str


@dataclass(frozen=True, slots=True)
class Notification:
    message: str
    kind: NotificationKind = NotificationKind.INFO
    detail: str = ""
    primary_action: NotificationAction | None = None
    secondary_action: NotificationAction | None = None
    dismissible: bool = True


class NotificationBar(QFrame):
    """Display the highest-priority queued notification in one fixed location."""

    action_requested = Signal(str)
    notification_dismissed = Signal(str)

    _PRIORITY = {
        NotificationKind.INFO: 0,
        NotificationKind.SUCCESS: 1,
        NotificationKind.WARNING: 2,
        NotificationKind.ERROR: 3,
    }
    _ICONS = {
        NotificationKind.INFO: QStyle.StandardPixmap.SP_MessageBoxInformation,
        NotificationKind.SUCCESS: QStyle.StandardPixmap.SP_DialogApplyButton,
        NotificationKind.WARNING: QStyle.StandardPixmap.SP_MessageBoxWarning,
        NotificationKind.ERROR: QStyle.StandardPixmap.SP_MessageBoxCritical,
    }

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("notificationBar")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setAccessibleName("Application notification")
        self.setStyleSheet(
            """
            QFrame#notificationBar {
                background-color: palette(alternate-base);
                border: 1px solid palette(mid);
                border-radius: 4px;
            }
            """
        )
        self._entries: dict[str, tuple[int, Notification]] = {}
        self._sequence = 0
        self._current_key: str | None = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 6, 6)
        self.icon_label = QLabel()
        self.icon_label.setObjectName("notificationIcon")
        layout.addWidget(self.icon_label, 0, Qt.AlignmentFlag.AlignTop)
        self.message_label = QLabel()
        self.message_label.setObjectName("notificationMessage")
        self.message_label.setWordWrap(True)
        layout.addWidget(self.message_label, 1)
        self.primary_button = QPushButton()
        self.primary_button.setObjectName("notificationPrimaryAction")
        self.primary_button.clicked.connect(
            lambda: self._request_action(primary=True)
        )
        layout.addWidget(self.primary_button)
        self.secondary_button = QPushButton()
        self.secondary_button.setObjectName("notificationSecondaryAction")
        self.secondary_button.clicked.connect(
            lambda: self._request_action(primary=False)
        )
        layout.addWidget(self.secondary_button)
        self.dismiss_button = QToolButton()
        self.dismiss_button.setObjectName("dismissNotificationButton")
        self.dismiss_button.setIcon(
            self.style().standardIcon(
                QStyle.StandardPixmap.SP_DialogCloseButton
            )
        )
        self.dismiss_button.setAccessibleName("Dismiss notification")
        self.dismiss_button.setToolTip("Dismiss notification")
        self.dismiss_button.setAutoRaise(True)
        self.dismiss_button.clicked.connect(self.dismiss_current)
        layout.addWidget(
            self.dismiss_button,
            0,
            Qt.AlignmentFlag.AlignTop,
        )
        self.setVisible(False)

    @property
    def current_key(self) -> str | None:
        return self._current_key

    @property
    def current_notification(self) -> Notification | None:
        if self._current_key is None:
            return None
        entry = self._entries.get(self._current_key)
        return entry[1] if entry is not None else None

    def show_notification(self, key: str, notification: Notification) -> None:
        self._sequence += 1
        self._entries[key] = (self._sequence, notification)
        self._render_current()

    def clear_notification(self, key: str) -> None:
        self._entries.pop(key, None)
        self._render_current()

    def clear_all(self) -> None:
        self._entries.clear()
        self._render_current()

    def dismiss_current(self) -> None:
        key = self._current_key
        if key is None:
            return
        self._entries.pop(key, None)
        self.notification_dismissed.emit(key)
        self._render_current()

    def _request_action(self, *, primary: bool) -> None:
        notification = self.current_notification
        if notification is None:
            return
        action = (
            notification.primary_action
            if primary
            else notification.secondary_action
        )
        if action is not None:
            self.action_requested.emit(action.action_id)

    def _render_current(self) -> None:
        if not self._entries:
            self._current_key = None
            self.message_label.clear()
            self.setToolTip("")
            self.setVisible(False)
            return
        key, (_sequence, notification) = max(
            self._entries.items(),
            key=lambda entry: (
                self._PRIORITY[entry[1][1].kind],
                entry[1][0],
            ),
        )
        self._current_key = key
        icon = self.style().standardIcon(self._ICONS[notification.kind])
        self.icon_label.setPixmap(icon.pixmap(16, 16))
        self.message_label.setText(notification.message)
        self.message_label.setToolTip(notification.detail)
        self.setToolTip(notification.detail)
        self._render_action(self.primary_button, notification.primary_action)
        self._render_action(
            self.secondary_button,
            notification.secondary_action,
        )
        self.dismiss_button.setVisible(notification.dismissible)
        self.setVisible(True)

    @staticmethod
    def _render_action(
        button: QPushButton,
        action: NotificationAction | None,
    ) -> None:
        button.setText(action.label if action is not None else "")
        button.setVisible(action is not None)


__all__ = [
    "Notification",
    "NotificationAction",
    "NotificationBar",
    "NotificationKind",
]
