"""Focused tests for the application-wide notification surface."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication
from shiboken6 import isValid

from hotcards.ui.notification_bar import (
    Notification,
    NotificationAction,
    NotificationBar,
    NotificationKind,
)


@pytest.fixture
def application(
    qt_application: QApplication, monkeypatch: pytest.MonkeyPatch
) -> Iterator[QApplication]:
    bars: list[NotificationBar] = []
    initialize = NotificationBar.__init__

    def initialize_owned(bar: NotificationBar, *args: object, **kwargs: object) -> None:
        initialize(bar, *args, **kwargs)
        bars.append(bar)

    monkeypatch.setattr(NotificationBar, "__init__", initialize_owned)
    try:
        yield qt_application
    finally:
        for bar in reversed(bars):
            if isValid(bar):
                bar.close()
                bar.deleteLater()
                QCoreApplication.sendPostedEvents(bar, QEvent.Type.DeferredDelete)


def test_notifications_are_prioritized_and_restored_after_dismissal(
    application: QApplication,
) -> None:
    bar = NotificationBar()
    layout = bar.layout()
    assert layout is not None
    margins = layout.contentsMargins()
    assert margins.left() == margins.right() == 8
    bar.show_notification(
        "progress",
        Notification("Generating image"),
    )
    bar.show_notification(
        "failure",
        Notification(
            "Image generation failed",
            kind=NotificationKind.ERROR,
        ),
    )

    assert bar.current_key == "failure"
    assert bar.message_label.text() == "Image generation failed"
    bar.dismiss_current()
    assert bar.current_key == "progress"
    assert bar.message_label.text() == "Generating image"


def test_notification_actions_and_dismissal_report_stable_ids(
    application: QApplication,
) -> None:
    bar = NotificationBar()
    actions: list[str] = []
    dismissed: list[str] = []
    bar.action_requested.connect(actions.append)
    bar.notification_dismissed.connect(dismissed.append)
    bar.show_notification(
        "undo",
        Notification(
            "Image cleared",
            kind=NotificationKind.SUCCESS,
            primary_action=NotificationAction("undo", "Undo"),
        ),
    )

    bar.primary_button.click()
    assert actions == ["undo"]
    assert bar.dismiss_button.text() == "Dismiss"
    assert type(bar.dismiss_button) is type(bar.primary_button)
    assert not hasattr(bar, "icon_label")
    bar.dismiss_button.click()
    assert dismissed == ["undo"]
    assert bar.isHidden()


def test_explicit_priority_keeps_time_sensitive_action_visible(
    application: QApplication,
) -> None:
    bar = NotificationBar()
    bar.show_notification(
        "service",
        Notification(
            "MFLUX unavailable",
            kind=NotificationKind.WARNING,
        ),
    )
    bar.show_notification(
        "undo",
        Notification(
            "Card deleted",
            kind=NotificationKind.SUCCESS,
            primary_action=NotificationAction("undo", "Undo"),
            priority=3,
        ),
    )

    assert bar.current_key == "undo"
    bar.show_notification(
        "error",
        Notification(
            "Stack could not be saved",
            kind=NotificationKind.ERROR,
        ),
    )
    assert bar.current_key == "error"


def test_notification_can_label_its_dismiss_action(
    application: QApplication,
) -> None:
    bar = NotificationBar()
    dismissed: list[str] = []
    bar.notification_dismissed.connect(dismissed.append)
    bar.show_notification(
        "generation",
        Notification(
            "Image generated on the current version",
            dismiss_label="Keep",
        ),
    )

    assert bar.dismiss_button.text() == "Keep"
    assert bar.dismiss_button.icon().isNull()
    assert bar.dismiss_button.accessibleName() == "Keep"
    assert type(bar.dismiss_button) is type(bar.primary_button)
    bar.dismiss_button.click()
    assert dismissed == ["generation"]
