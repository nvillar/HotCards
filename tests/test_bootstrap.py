"""Bootstrap tests for package entry points."""

from PySide6.QtWidgets import QApplication

from hotcards.evaluation.cli import build_parser
from hotcards.main import configure_application
from hotcards.ui.branding import APPLICATION_ICON_SIZES


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
