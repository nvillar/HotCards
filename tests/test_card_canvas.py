"""Offscreen tests for the shared image viewport and import crop preview."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PIL import Image
from PySide6.QtCore import QPointF, Qt
from PySide6.QtWidgets import QApplication

from hypergen.domain.models import CanvasSize
from hypergen.ui.card_canvas import CardCanvas
from hypergen.ui.crop_dialog import CropDialog


@pytest.fixture(scope="module")
def application() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_canvas_renders_candidate_and_round_trips_document_coordinates(
    application: QApplication,
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "background.png"
    Image.new("RGB", (1024, 768), "navy").save(image_path)
    canvas = CardCanvas()
    canvas.resize(800, 600)
    canvas.show()
    canvas.set_canvas_size(CanvasSize(width=1024, height=768))
    canvas.show_image(image_path, candidate=True)
    application.processEvents()

    assert canvas.scene().sceneRect().size().toSize().width() == 1024
    assert canvas._border_item is not None
    assert canvas._border_item.pen().style() == Qt.PenStyle.DashLine
    viewport_point = canvas.viewport_point_for(QPointF(0.25, 0.75))
    restored = canvas.document_point_at(viewport_point)
    assert restored.x() == pytest.approx(0.25, abs=0.01)
    assert restored.y() == pytest.approx(0.75, abs=0.01)
    canvas.close()


def test_canvas_reports_missing_image(application: QApplication, tmp_path: Path) -> None:
    canvas = CardCanvas()
    canvas.show_image(tmp_path / "missing.png")

    assert canvas._message_item is not None
    assert "Image unavailable" in canvas._message_item.toPlainText()
    canvas.close()


def test_crop_dialog_enables_only_relevant_axis(
    application: QApplication,
    tmp_path: Path,
) -> None:
    wide = tmp_path / "wide.png"
    tall = tmp_path / "tall.png"
    Image.new("RGB", (200, 100), "red").save(wide)
    Image.new("RGB", (100, 200), "blue").save(tall)
    size = CanvasSize(width=100, height=100)

    wide_dialog = CropDialog(wide, size)
    assert wide_dialog.horizontal_slider.isEnabled()
    assert not wide_dialog.vertical_slider.isEnabled()
    wide_dialog.horizontal_slider.setValue(250)
    assert wide_dialog.position_x == 0.25
    assert CropDialog.requires_crop(wide, size)
    wide_dialog.close()

    tall_dialog = CropDialog(tall, size)
    assert not tall_dialog.horizontal_slider.isEnabled()
    assert tall_dialog.vertical_slider.isEnabled()
    tall_dialog.close()
