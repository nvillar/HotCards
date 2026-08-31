"""Offscreen tests for direct manual hotspot authoring."""

from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PIL import Image
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QContextMenuEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

import hotcards.ui.card_canvas as card_canvas_module
from hotcards.domain.models import (
    CanvasSize,
    HotspotSet,
    Interaction,
    NavigateAction,
    Point,
    Polygon,
    UnresolvedCardReference,
)
from hotcards.ui.card_canvas import CardCanvas


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
        context_id=interaction.id,
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

    assert len(canvas._overlay_items) == 1
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


def test_drawing_ignores_overlapping_vertices_and_first_vertex_closes(
    application: QApplication,
    tmp_path: Path,
) -> None:
    canvas, interaction = configured_canvas(application, tmp_path)
    created: list[Polygon] = []
    errors: list[str] = []
    canvas.polygon_created.connect(
        lambda _interaction_id, polygon: created.append(polygon)
    )
    canvas.editing_error.connect(errors.append)
    points = (
        QPointF(0.55, 0.2),
        QPointF(0.85, 0.2),
        QPointF(0.7, 0.55),
    )

    canvas.begin_polygon(interaction.id)
    for point in points[:2]:
        QTest.mouseClick(
            canvas.viewport(),
            Qt.MouseButton.LeftButton,
            pos=canvas.viewport_point_for(point),
        )
    second_vertex = canvas.viewport_point_for(points[1])
    QTest.mouseClick(
        canvas.viewport(),
        Qt.MouseButton.LeftButton,
        pos=second_vertex + QPoint(4, 3),
    )

    assert canvas.drawing
    assert canvas._draft_points is not None
    assert len(canvas._draft_points) == 2
    assert errors == []

    QTest.mouseClick(
        canvas.viewport(),
        Qt.MouseButton.LeftButton,
        pos=canvas.viewport_point_for(points[2]),
    )
    first_vertex = canvas.viewport_point_for(points[0])
    QTest.mouseClick(
        canvas.viewport(),
        Qt.MouseButton.LeftButton,
        pos=first_vertex + QPoint(3, 4),
    )

    assert len(created) == 1
    assert len(created[0].points) == 3
    assert not canvas.drawing
    assert errors == []
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

    inside = canvas.viewport_point_for(QPointF(0.23, 0.2))
    QTest.mouseClick(canvas.viewport(), Qt.MouseButton.LeftButton, pos=inside)
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
    QTest.mouseClick(canvas.viewport(), Qt.MouseButton.LeftButton, pos=inside)
    edge = canvas.viewport_point_for(QPointF(0.26, 0.12))
    QTest.mouseMove(canvas.viewport(), edge)
    assert canvas._hovered_edge_insertion is not None
    QTest.mouseClick(canvas.viewport(), Qt.MouseButton.LeftButton, pos=edge)

    assert len(changed[-1].points) == 4
    canvas.close()


def test_first_drag_selects_another_hotspot_area_atomically(
    application: QApplication,
    tmp_path: Path,
) -> None:
    canvas, first = configured_canvas(application, tmp_path)
    second = Interaction(
        label="Chest",
        action=NavigateAction(target=UnresolvedCardReference()),
        polygons=(triangle(0.45),),
    )
    context_id = uuid4()
    canvas.set_hotspots(
        HotspotSet(interactions=(first, second)),
        first.id,
        editable=True,
        context_id=context_id,
    )
    canvas.interaction_selected.connect(canvas.select_interaction)
    changed: list[tuple[object, int, Polygon]] = []
    canvas.polygon_changed.connect(
        lambda interaction_id, polygon_index, polygon: changed.append(
            (interaction_id, polygon_index, polygon)
        )
    )

    start = canvas.viewport_point_for(QPointF(0.68, 0.2))
    end = canvas.viewport_point_for(QPointF(0.72, 0.24))
    QTest.mousePress(canvas.viewport(), Qt.MouseButton.LeftButton, pos=start)

    assert canvas.hasFocus()
    assert canvas._selected_interaction_id == second.id
    assert canvas._selected_polygon_index == 0

    QTest.mouseMove(canvas.viewport(), end)
    QTest.mouseRelease(canvas.viewport(), Qt.MouseButton.LeftButton, pos=end)

    assert changed[-1][0:2] == (second.id, 0)
    assert changed[-1][2].points[0].x > second.polygons[0].points[0].x
    canvas.close()


def test_geometry_selection_survives_updates_and_escape_steps_outward(
    application: QApplication,
    tmp_path: Path,
) -> None:
    canvas, interaction = configured_canvas(application, tmp_path)
    context_id = uuid4()
    hotspot_set = HotspotSet(interactions=(interaction,))
    canvas.set_hotspots(
        hotspot_set,
        interaction.id,
        editable=True,
        context_id=context_id,
    )

    inside = canvas.viewport_point_for(QPointF(0.23, 0.2))
    QTest.mouseClick(canvas.viewport(), Qt.MouseButton.LeftButton, pos=inside)
    vertex = canvas.viewport_point_for(QPointF(0.1, 0.1))
    QTest.mouseClick(canvas.viewport(), Qt.MouseButton.LeftButton, pos=vertex)
    canvas.set_hotspots(
        hotspot_set,
        interaction.id,
        editable=True,
        context_id=context_id,
    )

    assert canvas._selected_polygon_index == 0
    assert canvas._selected_vertex_index == 0

    QTest.keyClick(canvas, Qt.Key.Key_Escape)
    assert canvas._selected_polygon_index == 0
    assert canvas._selected_vertex_index is None

    QTest.keyClick(canvas, Qt.Key.Key_Escape)
    assert canvas._selected_interaction_id == interaction.id
    assert canvas._selected_polygon_index is None
    canvas.close()


def test_revision_context_change_cancels_polygon_drawing(
    application: QApplication,
    tmp_path: Path,
) -> None:
    canvas, interaction = configured_canvas(application, tmp_path)
    canvas.begin_polygon(
        interaction.id,
        initial_point=Point(x=0.6, y=0.6),
    )
    assert canvas.drawing

    canvas.set_hotspots(
        HotspotSet(interactions=(interaction,)),
        interaction.id,
        editable=True,
        context_id=uuid4(),
    )

    assert not canvas.drawing
    assert canvas._drawing_interaction_id is None
    canvas.close()


def test_disabling_editing_cancels_drawing_and_hides_authoring_overlays(
    application: QApplication,
    tmp_path: Path,
) -> None:
    canvas, interaction = configured_canvas(application, tmp_path)
    context_id = uuid4()
    hotspot_set = HotspotSet(interactions=(interaction,))
    canvas.set_hotspots(
        hotspot_set,
        interaction.id,
        editable=True,
        context_id=context_id,
    )
    canvas.begin_polygon(
        interaction.id,
        initial_point=Point(x=0.6, y=0.6),
    )

    canvas.set_hotspots(
        hotspot_set,
        interaction.id,
        editable=False,
        context_id=context_id,
    )

    assert not canvas.drawing
    assert not canvas._editable
    assert canvas._overlay_items == []
    canvas.close()


def test_selecting_another_hotspot_cancels_polygon_drawing(
    application: QApplication,
    tmp_path: Path,
) -> None:
    canvas, first = configured_canvas(application, tmp_path)
    second = Interaction(
        label="Chest",
        action=NavigateAction(target=UnresolvedCardReference()),
        polygons=(triangle(0.45),),
    )
    canvas.set_hotspots(
        HotspotSet(interactions=(first, second)),
        first.id,
        editable=True,
        context_id=uuid4(),
    )
    canvas.begin_polygon(
        first.id,
        initial_point=Point(x=0.6, y=0.6),
    )

    canvas.select_interaction(second.id)

    assert not canvas.drawing
    assert canvas._drawing_interaction_id is None
    assert canvas._selected_interaction_id == second.id
    canvas.close()


def test_double_clicking_selected_edge_adds_one_vertex(
    application: QApplication,
    tmp_path: Path,
) -> None:
    canvas, interaction = configured_canvas(application, tmp_path)
    context_id = uuid4()
    canvas.set_hotspots(
        HotspotSet(interactions=(interaction,)),
        interaction.id,
        editable=True,
        context_id=context_id,
    )
    changes: list[Polygon] = []

    def replace_polygon(
        _interaction_id: object,
        _polygon_index: int,
        changed: Polygon,
    ) -> None:
        nonlocal interaction
        changes.append(changed)
        interaction = interaction.model_copy(update={"polygons": (changed,)})
        canvas.set_hotspots(
            HotspotSet(interactions=(interaction,)),
            interaction.id,
            editable=True,
            context_id=context_id,
        )

    canvas.polygon_changed.connect(replace_polygon)
    inside = canvas.viewport_point_for(QPointF(0.23, 0.2))
    QTest.mouseClick(canvas.viewport(), Qt.MouseButton.LeftButton, pos=inside)
    edge = canvas.viewport_point_for(QPointF(0.26, 0.1))
    QTest.mouseDClick(canvas.viewport(), Qt.MouseButton.LeftButton, pos=edge)

    assert len(changes) == 1
    assert len(interaction.polygons[0].points) == 4
    assert canvas._selected_polygon_index == 0
    assert canvas._selected_vertex_index is not None
    canvas.close()


def test_vertex_and_area_deletion_leave_parent_selection(
    application: QApplication,
    tmp_path: Path,
) -> None:
    canvas, interaction = configured_canvas(application, tmp_path)
    polygon = Polygon(
        points=(
            Point(x=0.1, y=0.1),
            Point(x=0.4, y=0.1),
            Point(x=0.4, y=0.4),
            Point(x=0.1, y=0.4),
        )
    )
    interaction = interaction.model_copy(update={"polygons": (polygon,)})
    context_id = uuid4()
    current = HotspotSet(interactions=(interaction,))
    canvas.set_hotspots(
        current,
        interaction.id,
        editable=True,
        context_id=context_id,
    )

    def replace_polygon(
        _interaction_id: object,
        _polygon_index: int,
        changed: Polygon,
    ) -> None:
        nonlocal current, interaction
        interaction = interaction.model_copy(update={"polygons": (changed,)})
        current = HotspotSet(interactions=(interaction,))
        canvas.set_hotspots(
            current,
            interaction.id,
            editable=True,
            context_id=context_id,
        )

    def delete_polygon(_interaction_id: object, _polygon_index: int) -> None:
        nonlocal current, interaction
        interaction = interaction.model_copy(update={"polygons": ()})
        current = HotspotSet(interactions=(interaction,))
        canvas.set_hotspots(
            current,
            interaction.id,
            editable=True,
            context_id=context_id,
        )

    canvas.polygon_changed.connect(replace_polygon)
    canvas.polygon_deletion_requested.connect(delete_polygon)
    inside = canvas.viewport_point_for(QPointF(0.25, 0.25))
    QTest.mouseClick(canvas.viewport(), Qt.MouseButton.LeftButton, pos=inside)
    vertex = canvas.viewport_point_for(QPointF(0.1, 0.1))
    QTest.mouseClick(canvas.viewport(), Qt.MouseButton.LeftButton, pos=vertex)

    QTest.keyClick(canvas, Qt.Key.Key_Delete)

    assert len(interaction.polygons[0].points) == 3
    assert canvas._selected_polygon_index == 0
    assert canvas._selected_vertex_index is None

    QTest.keyClick(canvas, Qt.Key.Key_Delete)

    assert interaction.polygons == ()
    assert canvas._selected_interaction_id == interaction.id
    assert canvas._selected_polygon_index is None
    canvas.close()


def test_context_menu_dismissal_does_not_delete_selected_area(
    application: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canvas, interaction = configured_canvas(application, tmp_path)
    inside = canvas.viewport_point_for(QPointF(0.23, 0.2))
    QTest.mouseClick(canvas.viewport(), Qt.MouseButton.LeftButton, pos=inside)
    deletions: list[tuple[object, int]] = []
    canvas.polygon_deletion_requested.connect(
        lambda interaction_id, polygon_index: deletions.append(
            (interaction_id, polygon_index)
        )
    )

    class FakeAction:
        def setEnabled(self, _enabled: bool) -> None:
            pass

    class DismissedMenu:
        def __init__(self, _parent: object) -> None:
            pass

        def addAction(self, _label: str) -> FakeAction:
            return FakeAction()

        def exec(self, _position: object) -> None:
            return None

    monkeypatch.setattr(card_canvas_module, "QMenu", DismissedMenu)
    event = QContextMenuEvent(
        QContextMenuEvent.Reason.Mouse,
        inside,
        canvas.viewport().mapToGlobal(inside),
    )

    canvas.contextMenuEvent(event)

    assert deletions == []
    assert canvas._selected_interaction_id == interaction.id
    assert canvas._selected_polygon_index == 0
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

    inside = canvas.viewport_point_for(QPointF(0.23, 0.2))
    QTest.mouseClick(canvas.viewport(), Qt.MouseButton.LeftButton, pos=inside)
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
