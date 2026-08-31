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
    ResolutionTier,
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
    assert restored is not None
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


def test_canvas_fits_complete_image_without_cropping_or_stretching(
    application: QApplication,
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "wide.png"
    image = Image.new("RGB", (400, 200), "navy")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 19, 19), fill="red")
    draw.rectangle((380, 0, 399, 19), fill="green")
    draw.rectangle((0, 180, 19, 199), fill="blue")
    draw.rectangle((380, 180, 399, 199), fill="yellow")
    draw.rectangle((160, 60, 239, 139), fill="red")
    image.save(image_path)
    canvas = CardCanvas()
    canvas.set_canvas_size(CanvasSize(width=200, height=200))

    canvas.show_image(image_path)

    assert canvas._image_item is not None
    assert canvas._image_scene_rect is not None
    rendered = canvas._image_item.pixmap().toImage()
    assert rendered.width() / rendered.height() == pytest.approx(2.0)
    assert rendered.pixelColor(0, 0) == QColor("red")
    assert rendered.pixelColor(rendered.width() - 1, 0) == QColor("green")
    assert rendered.pixelColor(0, rendered.height() - 1) == QColor("blue")
    assert rendered.pixelColor(
        rendered.width() - 1,
        rendered.height() - 1,
    ) == QColor("yellow")
    assert canvas._image_scene_rect.top() > 0
    assert canvas._image_scene_rect.bottom() < canvas.scene().sceneRect().bottom()
    canvas.close()


@pytest.mark.parametrize("aspect_ratio", tuple(AspectRatio))
@pytest.mark.parametrize("resolution", tuple(ResolutionTier))
def test_mixed_resolution_backgrounds_keep_hotspots_aligned(
    application: QApplication,
    tmp_path: Path,
    aspect_ratio: AspectRatio,
    resolution: ResolutionTier,
) -> None:
    logical_size = Stack(name="Stack", aspect_ratio=aspect_ratio).canvas
    path = tmp_path / f"{aspect_ratio.name.lower()}-{resolution.value}.png"
    Image.new(
        "RGB",
        output_dimensions(resolution, aspect_ratio),
        "navy",
    ).save(path)
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
    assert canvas._image_scene_rect is not None
    image_rect = canvas._image_scene_rect
    assert image_rect.center() == canvas.scene().sceneRect().center()
    assert canvas._image_item.sceneBoundingRect() == image_rect
    viewport_point = canvas.viewport_point_for(QPointF(0.2, 0.2))
    assert canvas._vertex_at(viewport_point) == (interaction.id, 0, 0)
    restored = canvas.document_point_at(canvas.viewport_point_for(QPointF(0.3, 0.3)))
    assert restored is not None
    assert restored.x() == pytest.approx(0.3, abs=0.01)
    assert restored.y() == pytest.approx(0.3, abs=0.01)
    for corner in (QPointF(0.0, 0.0), QPointF(1.0, 1.0)):
        restored_corner = canvas.document_point_at(canvas.viewport_point_for(corner))
        assert restored_corner is not None
        assert restored_corner.x() == pytest.approx(corner.x(), abs=0.01)
        assert restored_corner.y() == pytest.approx(corner.y(), abs=0.01)

    QTest.mouseClick(
        canvas.viewport(),
        Qt.MouseButton.LeftButton,
        pos=canvas.viewport_point_for(QPointF(0.6, 0.6)),
    )
    assert canvas.drawing
    assert canvas._draft_points is not None
    assert len(canvas._draft_points) == 1
    assert canvas._draft_points[0].x() == pytest.approx(0.6, abs=0.01)
    assert canvas._draft_points[0].y() == pytest.approx(0.6, abs=0.01)

    activated: list[object] = []
    canvas.interaction_activated.connect(activated.append)
    canvas.cancel_drawing()
    canvas.set_run_hotspots(hotspot_set, canvas._overlay_mode)
    QTest.mouseClick(
        canvas.viewport(),
        Qt.MouseButton.LeftButton,
        pos=canvas.viewport_point_for(QPointF(0.3, 0.3)),
    )
    assert activated == [interaction.id]
    assert canvas._run_interaction_at(QPointF(0.8, 0.8)) is None

    bar_scene_point = None
    scene_rect = canvas.scene().sceneRect()
    if image_rect.left() > scene_rect.left():
        bar_scene_point = QPointF(
            (scene_rect.left() + image_rect.left()) / 2,
            image_rect.center().y(),
        )
    elif image_rect.top() > scene_rect.top():
        bar_scene_point = QPointF(
            image_rect.center().x(),
            (scene_rect.top() + image_rect.top()) / 2,
        )
    if bar_scene_point is not None:
        bar_viewport_point = canvas.mapFromScene(bar_scene_point)
        assert canvas.document_point_at(bar_viewport_point) is None
        QTest.mouseClick(
            canvas.viewport(),
            Qt.MouseButton.LeftButton,
            pos=bar_viewport_point,
        )
        assert activated == [interaction.id]

        canvas.set_hotspots(
            hotspot_set,
            interaction.id,
            editable=True,
            context_id=Card(name="Card").active_revision.id,
        )
        canvas.begin_polygon(interaction.id, initial_point=Point(x=0.5, y=0.5))
        assert canvas._draft_points is not None
        QTest.mouseClick(
            canvas.viewport(),
            Qt.MouseButton.LeftButton,
            pos=bar_viewport_point,
        )
        assert canvas._draft_points == [QPointF(0.5, 0.5)]

    canvas.close()
