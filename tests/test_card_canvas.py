"""Offscreen tests for the shared image viewport."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PIL import Image, ImageDraw
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from hotcards.domain.image_dimensions import (
    AspectRatio,
    GenerateResolution,
    output_dimensions,
)
from hotcards.domain.models import (
    CanvasSize,
    Card,
    HotspotSet,
    Interaction,
    Point,
    Polygon,
    Stack,
)
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


def test_canvas_crops_to_logical_format_without_stretching(
    application: QApplication,
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "wide.png"
    image = Image.new("RGB", (400, 200), "navy")
    draw = ImageDraw.Draw(image)
    draw.rectangle((160, 60, 239, 139), fill="red")
    image.save(image_path)
    canvas = CardCanvas()
    canvas.set_canvas_size(CanvasSize(width=200, height=200))

    canvas.show_image(image_path)

    assert canvas._image_item is not None
    rendered = canvas._image_item.pixmap().toImage()
    red_points = [
        (x, y)
        for y in range(rendered.height())
        for x in range(rendered.width())
        if rendered.pixelColor(x, y) == QColor("red")
    ]
    assert red_points
    red_width = max(x for x, _y in red_points) - min(x for x, _y in red_points) + 1
    red_height = max(y for _x, y in red_points) - min(y for _x, y in red_points) + 1
    assert red_width == pytest.approx(red_height, abs=1)


@pytest.mark.parametrize("aspect_ratio", tuple(AspectRatio))
def test_mixed_resolution_backgrounds_keep_hotspots_aligned(
    application: QApplication,
    tmp_path: Path,
    aspect_ratio: AspectRatio,
) -> None:
    logical_size = Stack(name="Stack", aspect_ratio=aspect_ratio).canvas
    paths = []
    for resolution in (
        GenerateResolution.RESOLUTION_256,
        GenerateResolution.RESOLUTION_1024,
    ):
        path = tmp_path / (
            f"{aspect_ratio.name.lower()}-{resolution.value}.png"
        )
        Image.new(
            "RGB",
            output_dimensions(resolution, aspect_ratio),
            "navy",
        ).save(path)
        paths.append(path)
    polygon = Polygon(
        points=(
            Point(x=0.2, y=0.2),
            Point(x=0.4, y=0.2),
            Point(x=0.4, y=0.4),
            Point(x=0.2, y=0.4),
        )
    )
    interaction = Interaction(polygons=(polygon,))
    hotspot_set = HotspotSet(interactions=(interaction,))
    canvas = CardCanvas()
    canvas.resize(900, 700)
    canvas.show()
    canvas.set_canvas_size(logical_size)

    for path in paths:
        canvas.show_image(path)
        canvas.set_hotspots(
            hotspot_set,
            interaction.id,
            editable=True,
            context_id=Card(name="Card").active_revision.id,
        )
        canvas.select_polygon(interaction.id, 0)
        application.processEvents()

        assert canvas._image_item is not None
        assert canvas._image_item.pixmap().width() == logical_size.width
        assert canvas._image_item.pixmap().height() == logical_size.height
        viewport_point = canvas.viewport_point_for(QPointF(0.2, 0.2))
        assert canvas._vertex_at(viewport_point) == (interaction.id, 0, 0)
        restored = canvas.document_point_at(
            canvas.viewport_point_for(QPointF(0.3, 0.3))
        )
        assert restored.x() == pytest.approx(0.3, abs=0.01)
        assert restored.y() == pytest.approx(0.3, abs=0.01)

        canvas.set_run_hotspots(hotspot_set, canvas._overlay_mode)
        assert canvas._run_interaction_at(QPointF(0.3, 0.3)) == interaction
        assert canvas._run_interaction_at(QPointF(0.8, 0.8)) is None

    canvas.close()
