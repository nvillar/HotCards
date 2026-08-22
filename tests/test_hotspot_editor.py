"""Offscreen tests for direct manual hotspot authoring."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PIL import Image
from PySide6.QtCore import QPointF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from hypergen.domain.models import (
    CanvasSize,
    HotspotSet,
    Interaction,
    NavigateAction,
    Point,
    Polygon,
    UnresolvedCardReference,
)
from hypergen.ui.card_canvas import CardCanvas


@pytest.fixture(scope="module")
def application() -> QApplication:
    return QApplication.instance() or QApplication([])


def triangle(offset: float = 0.0) -> Polygon:
    return Polygon(
        points=(
            Point(x=0.1 + offset, y=0.1),
            Point(x=0.4 + offset, y=0.1),
            Point(x=0.2 + offset, y=0.4),
        )
    )


def hotspot() -> Interaction:
    return Interaction(
        label="Door",
        action=NavigateAction(target=UnresolvedCardReference()),
        polygons=(triangle(),),
    )


def configured_canvas(
    application: QApplication,
    tmp_path: Path,
) -> tuple[CardCanvas, Interaction]:
    image_path = tmp_path / "background.png"
    Image.new("RGB", (1024, 768), "navy").save(image_path)
    interaction = hotspot()
    canvas = CardCanvas()
    canvas.resize(800, 600)
    canvas.show()
    canvas.set_canvas_size(CanvasSize())
    canvas.show_image(image_path)
    canvas.set_hotspots(
        HotspotSet(interactions=(interaction,)),
        interaction.id,
        editable=True,
    )
    application.processEvents()
    return canvas, interaction


def test_canvas_renders_selected_hotspot_and_draws_new_polygon(
    application: QApplication,
    tmp_path: Path,
) -> None:
    canvas, interaction = configured_canvas(application, tmp_path)
    created: list[tuple[object, Polygon]] = []
    canvas.polygon_created.connect(
        lambda interaction_id, polygon: created.append(
            (interaction_id, polygon)
        )
    )

    assert len(canvas._overlay_items) == 4
    canvas.begin_polygon()
    for point in (
        QPointF(0.55, 0.2),
        QPointF(0.85, 0.2),
        QPointF(0.7, 0.55),
    ):
        QTest.mouseClick(
            canvas.viewport(),
            Qt.MouseButton.LeftButton,
            pos=canvas.viewport_point_for(point),
        )
    QTest.keyClick(canvas, Qt.Key.Key_Return)

    assert len(created) == 1
    assert created[0][0] is None
    assert created[0][1].points[0].x == pytest.approx(0.55, abs=0.01)
    assert not canvas.drawing

    canvas.begin_polygon(interaction.id)
    for point in (QPointF(0.5, 0.6), QPointF(0.8, 0.6)):
        QTest.mouseClick(
            canvas.viewport(),
            Qt.MouseButton.LeftButton,
            pos=canvas.viewport_point_for(point),
        )
    QTest.mouseDClick(
        canvas.viewport(),
        Qt.MouseButton.LeftButton,
        pos=canvas.viewport_point_for(QPointF(0.65, 0.85)),
    )
    assert len(created) == 2
    assert created[-1][0] == interaction.id
    assert not canvas.drawing

    canvas.begin_polygon(interaction.id)
    QTest.keyClick(canvas, Qt.Key.Key_Escape)
    assert not canvas.drawing
    canvas.close()


def test_canvas_vertex_move_and_edge_insertion_emit_valid_polygons(
    application: QApplication,
    tmp_path: Path,
) -> None:
    canvas, interaction = configured_canvas(application, tmp_path)
    changed: list[Polygon] = []
    canvas.polygon_changed.connect(
        lambda _interaction_id, _polygon_index, polygon: changed.append(polygon)
    )

    start = canvas.viewport_point_for(QPointF(0.1, 0.1))
    end = canvas.viewport_point_for(QPointF(0.12, 0.14))
    QTest.mousePress(canvas.viewport(), Qt.MouseButton.LeftButton, pos=start)
    QTest.mouseMove(canvas.viewport(), end)
    QTest.mouseRelease(canvas.viewport(), Qt.MouseButton.LeftButton, pos=end)

    assert changed
    assert changed[-1].points[0].x == pytest.approx(0.12, abs=0.02)

    canvas.set_hotspots(
        HotspotSet(
            interactions=(
                interaction.model_copy(
                    update={"polygons": (changed[-1],)}
                ),
            )
        ),
        interaction.id,
        editable=True,
    )
    inside = canvas.viewport_point_for(QPointF(0.23, 0.2))
    QTest.mouseClick(canvas.viewport(), Qt.MouseButton.LeftButton, pos=inside)
    edge = canvas.viewport_point_for(QPointF(0.26, 0.12))
    QTest.mouseDClick(canvas.viewport(), Qt.MouseButton.LeftButton, pos=edge)

    assert len(changed[-1].points) == 4
    canvas.close()


def test_drawing_and_minimum_vertex_deletion_are_non_destructive(
    application: QApplication,
    tmp_path: Path,
) -> None:
    canvas, interaction = configured_canvas(application, tmp_path)
    deletions: list[object] = []
    errors: list[str] = []
    canvas.interaction_deletion_requested.connect(deletions.append)
    canvas.editing_error.connect(errors.append)

    vertex = canvas.viewport_point_for(QPointF(0.1, 0.1))
    QTest.mouseClick(canvas.viewport(), Qt.MouseButton.LeftButton, pos=vertex)
    QTest.keyClick(canvas, Qt.Key.Key_Delete)
    assert not deletions
    assert "at least three vertices" in errors[-1]

    polygon_deletions: list[tuple[object, int]] = []
    canvas.polygon_deletion_requested.connect(
        lambda interaction_id, polygon_index: polygon_deletions.append(
            (interaction_id, polygon_index)
        )
    )
    canvas._selected_vertex_index = None
    canvas._selected_polygon_index = 0
    QTest.keyClick(canvas, Qt.Key.Key_Delete)
    assert polygon_deletions == [(interaction.id, 0)]
    assert not deletions

    canvas.begin_polygon(interaction.id)
    draft_point = canvas.viewport_point_for(QPointF(0.7, 0.7))
    QTest.mouseClick(
        canvas.viewport(),
        Qt.MouseButton.LeftButton,
        pos=draft_point,
    )
    QTest.keyClick(canvas, Qt.Key.Key_Delete)
    assert canvas.drawing
    assert canvas._draft_points == []
    assert not deletions

    second_component = triangle(0.4)
    canvas.cancel_drawing()
    canvas.set_hotspots(
        HotspotSet(
            interactions=(
                interaction.model_copy(
                    update={
                        "polygons": (
                            interaction.polygons[0],
                            second_component,
                        )
                    }
                ),
            )
        ),
        interaction.id,
        editable=True,
    )
    canvas._selected_polygon_index = 1
    canvas.set_hotspots(
        HotspotSet(interactions=(interaction,)),
        interaction.id,
        editable=True,
    )
    assert canvas._selected_polygon_index is None
    QTest.keyClick(canvas, Qt.Key.Key_Delete)
    assert not deletions
    canvas.close()
