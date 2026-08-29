"""Bootstrap tests for package entry points."""

from PySide6.QtWidgets import QApplication

from hotcards.evaluation.cli import build_parser
from hotcards.main import configure_application
from hotcards.ui.branding import (
    APPLICATION_ICON_SIZES,
    application_icon_pixmap,
)


def test_evaluation_parser_uses_expected_program_name() -> None:
    assert build_parser().prog == "hotcards-eval"


def test_application_uses_hotcards_identity_and_packaged_icon() -> None:
    application = QApplication.instance() or QApplication([])

    configure_application(application)

    assert application.organizationName() == "HotCards"
    assert application.applicationName() == "HotCards"
    assert not application.windowIcon().isNull()
    assert tuple(
        size.width() for size in application.windowIcon().availableSizes()
    ) == APPLICATION_ICON_SIZES
    icon = application.windowIcon().pixmap(1024, 1024).toImage()
    assert icon.pixelColor(0, 0).alpha() == 0
    assert icon.pixelColor(512, 512).alpha() == 255


def test_application_icon_pixmap_renders_at_device_resolution() -> None:
    pixmap = application_icon_pixmap(128, device_pixel_ratio=2.0)

    assert pixmap.width() == 256
    assert pixmap.height() == 256
    assert pixmap.devicePixelRatio() == 2.0
    assert pixmap.deviceIndependentSize().width() == 128
    assert pixmap.deviceIndependentSize().height() == 128
