"""Offscreen tests for the shared image viewport."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PIL import Image
from PySide6.QtCore import QPointF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from hotcards.domain.models import CanvasSize
from hotcards.ui.card_canvas import CardCanvas


@pytest.fixture(scope="module")
def application() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_canvas_renders_image_and_round_trips_document_coordinates(
    application: QApplication,
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "background.png"
    Image.new("RGB", (1024, 768), "navy").save(image_path)
    canvas = CardCanvas()
    canvas.resize(800, 600)
    canvas.show()
    canvas.set_canvas_size(CanvasSize(width=1024, height=768))
    canvas.show_image(image_path)
    application.processEvents()

    assert canvas.scene().sceneRect().size().toSize().width() == 1024
    assert canvas._border_item is not None
    assert canvas._border_item.pen().style() == Qt.PenStyle.SolidLine
    viewport_point = canvas.viewport_point_for(QPointF(0.25, 0.75))
    restored = canvas.document_point_at(viewport_point)
    assert restored.x() == pytest.approx(0.25, abs=0.01)
    assert restored.y() == pytest.approx(0.75, abs=0.01)
    canvas.close()


def test_clicking_empty_image_requests_an_implicit_hotspot_area(
    application: QApplication,
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "background.png"
    Image.new("RGB", (1024, 768), "navy").save(image_path)
    canvas = CardCanvas()
    canvas.resize(800, 600)
    canvas.show()
    canvas.show_image(image_path)
    canvas.set_hotspots(None, None, editable=True)
    requested: list[object] = []
    canvas.empty_area_requested.connect(requested.append)
    application.processEvents()

    QTest.mouseClick(
        canvas.viewport(),
        Qt.MouseButton.LeftButton,
        pos=canvas.viewport_point_for(QPointF(0.5, 0.5)),
    )

    assert len(requested) == 1
    point = requested[0]
    assert point.x == pytest.approx(0.5, abs=0.01)  # type: ignore[union-attr]
    assert point.y == pytest.approx(0.5, abs=0.01)  # type: ignore[union-attr]
    canvas.close()


def test_canvas_reports_missing_image(application: QApplication, tmp_path: Path) -> None:
    canvas = CardCanvas()
    canvas.show_image(tmp_path / "missing.png")

    assert canvas._message_item is not None
    assert "Image unavailable" in canvas._message_item.toPlainText()
    canvas.close()
