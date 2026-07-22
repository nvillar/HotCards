"""Shared fixed-aspect card image viewport."""

from __future__ import annotations

from math import hypot
from pathlib import Path
from uuid import UUID

from pydantic import ValidationError
from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QResizeEvent,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
    QWidget,
)

from hypergen.domain.models import CanvasSize, HotspotSet, Interaction, Point, Polygon


class CardCanvas(QGraphicsView):
    """Display one active image or candidate using stable document coordinates."""

    interaction_selected = Signal(object)
    polygon_created = Signal(object, object)
    polygon_changed = Signal(object, int, object)
    polygon_deletion_requested = Signal(object, int)
    interaction_deletion_requested = Signal(object)
    editing_error = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setObjectName("cardCanvas")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setBackgroundBrush(QColor("#34373b"))
        self._canvas_size = CanvasSize()
        self._fit_mode = True
        self._image_item: QGraphicsPixmapItem | None = None
        self._border_item: QGraphicsRectItem | None = None
        self._message_item: QGraphicsTextItem | None = None
        self._current_image: tuple[Path, bool] | None = None
        self._current_message: str | None = None
        self._hotspot_set: HotspotSet | None = None
        self._selected_interaction_id: UUID | None = None
        self._selected_polygon_index: int | None = None
        self._selected_vertex_index: int | None = None
        self._editable = False
        self._overlay_items: list[QGraphicsItem] = []
        self._drawing_interaction_id: UUID | None = None
        self._draft_points: list[QPointF] | None = None
        self._drag_kind: str | None = None
        self._drag_origin: QPointF | None = None
        self._drag_original: tuple[Point, ...] | None = None
        self._preview_polygon: tuple[Point, ...] | None = None
        self._panning = False
        self._pan_origin: QPoint | None = None
        self.set_canvas_size(self._canvas_size)
        self.show_message("Select a card")

    def set_canvas_size(self, size: CanvasSize) -> None:
        if size == self._canvas_size and self.scene().sceneRect().width() > 0:
            return
        self._canvas_size = size
        self._current_image = None
        self._current_message = None
        self.scene().setSceneRect(QRectF(0, 0, size.width, size.height))

    def show_image(self, path: Path, *, candidate: bool = False) -> None:
        """Render one decoded image across the logical canvas."""
        image_key = (path.resolve(), candidate)
        if image_key == self._current_image and self._image_item is not None:
            return
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            self.show_message(f"Image unavailable\n{path.name}")
            return
        self._clear_scene_items()
        scaled = pixmap.scaled(
            self._canvas_size.width,
            self._canvas_size.height,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._image_item = self.scene().addPixmap(scaled)
        self._image_item.setZValue(0)
        self._current_image = image_key
        self._current_message = None
        border_color = QColor("#4da3ff") if candidate else QColor("#8a8f98")
        border = QGraphicsRectItem(self.scene().sceneRect())
        pen = QPen(border_color, 3 if candidate else 1)
        pen.setCosmetic(True)
        if candidate:
            pen.setStyle(Qt.PenStyle.DashLine)
        border.setPen(pen)
        self.scene().addItem(border)
        border.setZValue(1)
        self._border_item = border
        self.fit_to_window()

    def set_hotspots(
        self,
        hotspot_set: HotspotSet | None,
        selected_interaction_id: UUID | None,
        *,
        editable: bool,
    ) -> None:
        """Render applied hotspots and enable author gestures when appropriate."""
        self._hotspot_set = hotspot_set
        interaction_ids = {
            interaction.id
            for interaction in hotspot_set.interactions
        } if hotspot_set is not None else set()
        self._selected_interaction_id = (
            selected_interaction_id
            if selected_interaction_id in interaction_ids
            else None
        )
        self._selected_polygon_index = None
        self._selected_vertex_index = None
        self._drag_kind = None
        self._drag_origin = None
        self._drag_original = None
        self._preview_polygon = None
        self._editable = editable
        self._render_hotspots()

    def select_interaction(self, interaction_id: UUID | None) -> None:
        self._selected_interaction_id = interaction_id
        self._selected_polygon_index = None
        self._selected_vertex_index = None
        self._render_hotspots()

    def begin_polygon(self, interaction_id: UUID | None = None) -> None:
        """Begin a new interaction polygon or another component."""
        if not self._editable or self._image_item is None:
            self.editing_error.emit("Apply a background before drawing hotspots.")
            return
        if interaction_id is not None and self._interaction(interaction_id) is None:
            self.editing_error.emit("The selected hotspot no longer exists.")
            return
        self._drawing_interaction_id = interaction_id
        self._draft_points = []
        self._selected_polygon_index = None
        self._selected_vertex_index = None
        self._drag_kind = None
        self.viewport().setCursor(Qt.CursorShape.CrossCursor)
        self.setFocus()
        self._render_hotspots()

    def cancel_drawing(self) -> None:
        if self._draft_points is None:
            return
        self._drawing_interaction_id = None
        self._draft_points = None
        self.viewport().unsetCursor()
        self._render_hotspots()

    @property
    def drawing(self) -> bool:
        return self._draft_points is not None

    def show_message(self, message: str) -> None:
        if message == self._current_message and self._message_item is not None:
            return
        self._clear_scene_items()
        self._current_image = None
        self._current_message = message
        self._message_item = self.scene().addText(message)
        self._message_item.setDefaultTextColor(QColor("#d7d9dc"))
        bounds = self._message_item.boundingRect()
        self._message_item.setPos(
            (self._canvas_size.width - bounds.width()) / 2,
            (self._canvas_size.height - bounds.height()) / 2,
        )
        self.fit_to_window()

    def fit_to_window(self) -> None:
        self._fit_mode = True
        self.fitInView(
            self.scene().sceneRect(),
            Qt.AspectRatioMode.KeepAspectRatio,
        )
        self._render_hotspots()

    def document_point_at(self, viewport_point: QPoint) -> QPointF:
        """Map a viewport point into normalized document coordinates."""
        scene_point = self.mapToScene(viewport_point)
        return QPointF(
            scene_point.x() / self._canvas_size.width,
            scene_point.y() / self._canvas_size.height,
        )

    def viewport_point_for(self, document_point: QPointF) -> QPoint:
        """Map a normalized document point into viewport coordinates."""
        return self.mapFromScene(
            QPointF(
                document_point.x() * self._canvas_size.width,
                document_point.y() * self._canvas_size.height,
            )
        )

    def wheelEvent(self, event: QWheelEvent) -> None:
        vertical_delta = event.angleDelta().y()
        if vertical_delta == 0:
            super().wheelEvent(event)
            return
        factor = 1.2 if vertical_delta > 0 else 1 / 1.2
        current_scale = self.transform().m11()
        next_scale = current_scale * factor
        if 0.05 <= next_scale <= 20:
            self._fit_mode = False
            self.scale(factor, factor)
            self._render_hotspots()
        event.accept()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._pan_origin = event.position().toPoint()
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        if event.button() != Qt.MouseButton.LeftButton or not self._editable:
            super().mousePressEvent(event)
            return
        if self._draft_points is not None:
            self._append_draft_point(event.position().toPoint())
            event.accept()
            return
        normalized = self._clamped_document_point(event.position().toPoint())
        vertex = self._vertex_at(event.position().toPoint())
        if vertex is not None:
            interaction_id, polygon_index, vertex_index = vertex
            if interaction_id != self._selected_interaction_id:
                self.interaction_selected.emit(interaction_id)
            self._selected_interaction_id = interaction_id
            self._selected_polygon_index = polygon_index
            self._selected_vertex_index = vertex_index
            interaction = self._interaction(interaction_id)
            assert interaction is not None
            self._drag_kind = "vertex"
            self._drag_origin = normalized
            self._drag_original = interaction.polygons[polygon_index].points
            self._render_hotspots()
            event.accept()
            return
        hit = self._polygon_at(normalized)
        if hit is None:
            self.interaction_selected.emit(None)
            self.select_interaction(None)
            event.accept()
            return
        interaction_id, polygon_index = hit
        already_selected = interaction_id == self._selected_interaction_id
        self._selected_interaction_id = interaction_id
        self._selected_polygon_index = polygon_index
        self._selected_vertex_index = None
        self.interaction_selected.emit(interaction_id)
        if already_selected:
            interaction = self._interaction(interaction_id)
            assert interaction is not None
            self._drag_kind = "polygon"
            self._drag_origin = normalized
            self._drag_original = interaction.polygons[polygon_index].points
        self._render_hotspots()
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._panning and self._pan_origin is not None:
            current = event.position().toPoint()
            delta = current - self._pan_origin
            self._pan_origin = current
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )
            event.accept()
            return
        if (
            self._drag_kind is None
            or self._drag_origin is None
            or self._drag_original is None
        ):
            super().mouseMoveEvent(event)
            return
        current = self._clamped_document_point(event.position().toPoint())
        if self._drag_kind == "vertex" and self._selected_vertex_index is not None:
            points = list(self._drag_original)
            points[self._selected_vertex_index] = Point(
                x=current.x(),
                y=current.y(),
            )
            self._preview_polygon = tuple(points)
        else:
            dx = current.x() - self._drag_origin.x()
            dy = current.y() - self._drag_origin.y()
            minimum_x = min(point.x for point in self._drag_original)
            maximum_x = max(point.x for point in self._drag_original)
            minimum_y = min(point.y for point in self._drag_original)
            maximum_y = max(point.y for point in self._drag_original)
            dx = min(max(dx, -minimum_x), 1 - maximum_x)
            dy = min(max(dy, -minimum_y), 1 - maximum_y)
            self._preview_polygon = tuple(
                Point(x=point.x + dx, y=point.y + dy)
                for point in self._drag_original
            )
        self._render_hotspots()
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.MiddleButton and self._panning:
            self._panning = False
            self._pan_origin = None
            if self._draft_points is not None:
                self.viewport().setCursor(Qt.CursorShape.CrossCursor)
            else:
                self.viewport().unsetCursor()
            event.accept()
            return
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self._drag_kind is not None
        ):
            if (
                self._preview_polygon is not None
                and self._selected_interaction_id is not None
                and self._selected_polygon_index is not None
            ):
                try:
                    polygon = Polygon(points=self._preview_polygon)
                except ValidationError as error:
                    self.editing_error.emit(self._geometry_error(error))
                else:
                    self.polygon_changed.emit(
                        self._selected_interaction_id,
                        self._selected_polygon_index,
                        polygon,
                    )
            self._drag_kind = None
            self._drag_origin = None
            self._drag_original = None
            self._preview_polygon = None
            self._render_hotspots()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton or not self._editable:
            super().mouseDoubleClickEvent(event)
            return
        if self._draft_points is not None:
            point = self._clamped_document_point(event.position().toPoint())
            if (
                not self._draft_points
                or self._point_distance(self._draft_points[-1], point) > 1e-6
            ):
                self._draft_points.append(point)
            self._finish_drawing()
            event.accept()
            return
        if (
            self._selected_interaction_id is not None
            and self._selected_polygon_index is not None
        ):
            interaction = self._interaction(self._selected_interaction_id)
            if interaction is not None:
                polygon = interaction.polygons[self._selected_polygon_index]
                insertion = self._nearest_edge_insertion(
                    polygon,
                    event.position().toPoint(),
                )
                if insertion is not None:
                    edge_index, point = insertion
                    points = list(polygon.points)
                    points.insert(edge_index + 1, point)
                    try:
                        changed = Polygon(points=tuple(points))
                    except ValidationError as error:
                        self.editing_error.emit(self._geometry_error(error))
                    else:
                        self.polygon_changed.emit(
                            interaction.id,
                            self._selected_polygon_index,
                            changed,
                        )
                    event.accept()
                    return
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if self._draft_points is not None:
            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self._finish_drawing()
                event.accept()
                return
            if event.key() == Qt.Key.Key_Escape:
                self.cancel_drawing()
                event.accept()
                return
            if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
                if self._draft_points:
                    self._draft_points.pop()
                    self._render_hotspots()
                event.accept()
                return
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self._delete_selected_geometry()
            event.accept()
            return
        super().keyPressEvent(event)

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        if self._fit_mode:
            self.fit_to_window()

    def _clear_scene_items(self) -> None:
        self.scene().clear()
        self._overlay_items.clear()
        self._image_item = None
        self._border_item = None
        self._message_item = None
        self._hotspot_set = None
        self._editable = False
        self.cancel_drawing()

    def _render_hotspots(self) -> None:
        for item in self._overlay_items:
            if item.scene() is self.scene():
                self.scene().removeItem(item)
        self._overlay_items.clear()
        if self._hotspot_set is not None:
            for interaction_index, interaction in enumerate(
                self._hotspot_set.interactions
            ):
                selected = interaction.id == self._selected_interaction_id
                for polygon_index, polygon in enumerate(interaction.polygons):
                    points = (
                        self._preview_polygon
                        if selected
                        and polygon_index == self._selected_polygon_index
                        and self._preview_polygon is not None
                        else polygon.points
                    )
                    path = self._polygon_path(points)
                    color = QColor("#ffb347") if selected else QColor("#4da3ff")
                    fill = QColor(color)
                    fill.setAlpha(70 if selected else 42)
                    pen = QPen(color, 3 if selected else 2)
                    pen.setCosmetic(True)
                    item = self.scene().addPath(
                        path,
                        pen,
                        QBrush(fill),
                    )
                    item.setZValue(10 + interaction_index)
                    self._overlay_items.append(item)
                    if selected:
                        self._add_vertex_handles(
                            interaction.id,
                            polygon_index,
                            points,
                        )
        if self._draft_points is not None:
            self._add_draft_items()

    def _add_vertex_handles(
        self,
        interaction_id: UUID,
        polygon_index: int,
        points: tuple[Point, ...],
    ) -> None:
        radius = 6 / max(self.transform().m11(), 0.01)
        for vertex_index, point in enumerate(points):
            scene_point = self._scene_point(point)
            handle = QGraphicsEllipseItem(
                scene_point.x() - radius,
                scene_point.y() - radius,
                radius * 2,
                radius * 2,
            )
            handle.setPen(QPen(QColor("#202225"), 1))
            handle.setBrush(QBrush(QColor("#fff3d6")))
            handle.setZValue(100)
            handle.setData(0, interaction_id)
            handle.setData(1, polygon_index)
            handle.setData(2, vertex_index)
            self.scene().addItem(handle)
            self._overlay_items.append(handle)

    def _add_draft_items(self) -> None:
        assert self._draft_points is not None
        if self._draft_points:
            path = QPainterPath(self._scene_point(self._draft_points[0]))
            for point in self._draft_points[1:]:
                path.lineTo(self._scene_point(point))
            pen = QPen(QColor("#ffb347"), 3)
            pen.setCosmetic(True)
            item = self.scene().addPath(path, pen)
            item.setZValue(110)
            self._overlay_items.append(item)
        radius = 5 / max(self.transform().m11(), 0.01)
        for point in self._draft_points:
            scene_point = self._scene_point(point)
            handle = self.scene().addEllipse(
                scene_point.x() - radius,
                scene_point.y() - radius,
                radius * 2,
                radius * 2,
                QPen(QColor("#202225"), 1),
                QBrush(QColor("#ffb347")),
            )
            handle.setZValue(111)
            self._overlay_items.append(handle)

    def _append_draft_point(self, viewport_point: QPoint) -> None:
        assert self._draft_points is not None
        self._draft_points.append(self._clamped_document_point(viewport_point))
        self._render_hotspots()

    def _finish_drawing(self) -> None:
        assert self._draft_points is not None
        if len(self._draft_points) < 3:
            self.editing_error.emit("A hotspot needs at least three vertices.")
            return
        try:
            polygon = Polygon(
                points=tuple(
                    Point(x=point.x(), y=point.y())
                    for point in self._draft_points
                )
            )
        except ValidationError as error:
            self.editing_error.emit(self._geometry_error(error))
            return
        interaction_id = self._drawing_interaction_id
        self.cancel_drawing()
        self.polygon_created.emit(interaction_id, polygon)

    def _delete_selected_geometry(self) -> None:
        interaction_id = self._selected_interaction_id
        polygon_index = self._selected_polygon_index
        if interaction_id is None or polygon_index is None:
            return
        interaction = self._interaction(interaction_id)
        if interaction is None:
            return
        polygon = interaction.polygons[polygon_index]
        if self._selected_vertex_index is not None:
            if len(polygon.points) == 3:
                self.editing_error.emit(
                    "A polygon must retain at least three vertices. "
                    "Delete the area or hotspot instead."
                )
                return
            points = list(polygon.points)
            points.pop(self._selected_vertex_index)
            try:
                changed = Polygon(points=tuple(points))
            except ValidationError as error:
                self.editing_error.emit(self._geometry_error(error))
            else:
                self._selected_vertex_index = None
                self.polygon_changed.emit(interaction_id, polygon_index, changed)
            return
        if len(interaction.polygons) == 1:
            self.interaction_deletion_requested.emit(interaction_id)
        else:
            self.polygon_deletion_requested.emit(interaction_id, polygon_index)

    def _vertex_at(self, viewport_point: QPoint) -> tuple[UUID, int, int] | None:
        interaction = self._interaction(self._selected_interaction_id)
        if interaction is None:
            return None
        for polygon_index, polygon in enumerate(interaction.polygons):
            for vertex_index, point in enumerate(polygon.points):
                candidate = self.viewport_point_for(QPointF(point.x, point.y))
                if hypot(
                    candidate.x() - viewport_point.x(),
                    candidate.y() - viewport_point.y(),
                ) <= 9:
                    return interaction.id, polygon_index, vertex_index
        return None

    def _polygon_at(self, point: QPointF) -> tuple[UUID, int] | None:
        if self._hotspot_set is None:
            return None
        scene_point = QPointF(
            point.x() * self._canvas_size.width,
            point.y() * self._canvas_size.height,
        )
        for interaction in reversed(self._hotspot_set.interactions):
            for polygon_index in reversed(range(len(interaction.polygons))):
                if self._polygon_path(
                    interaction.polygons[polygon_index].points
                ).contains(scene_point):
                    return interaction.id, polygon_index
        return None

    def _nearest_edge_insertion(
        self,
        polygon: Polygon,
        viewport_point: QPoint,
    ) -> tuple[int, Point] | None:
        best: tuple[float, int, QPointF] | None = None
        points = [
            self.viewport_point_for(QPointF(point.x, point.y))
            for point in polygon.points
        ]
        target = QPointF(viewport_point)
        for index, (start, end) in enumerate(
            zip(points, points[1:] + points[:1], strict=True)
        ):
            start_f = QPointF(start)
            end_f = QPointF(end)
            edge = end_f - start_f
            length_squared = edge.x() ** 2 + edge.y() ** 2
            if length_squared == 0:
                continue
            offset = target - start_f
            fraction = min(
                max(
                    (offset.x() * edge.x() + offset.y() * edge.y())
                    / length_squared,
                    0.0,
                ),
                1.0,
            )
            projection = start_f + edge * fraction
            distance = self._point_distance(target, projection)
            if best is None or distance < best[0]:
                best = (distance, index, projection)
        if best is None or best[0] > 9:
            return None
        normalized = self._clamped_document_point(best[2].toPoint())
        return best[1], Point(x=normalized.x(), y=normalized.y())

    def _interaction(self, interaction_id: UUID | None) -> Interaction | None:
        if interaction_id is None or self._hotspot_set is None:
            return None
        return next(
            (
                interaction
                for interaction in self._hotspot_set.interactions
                if interaction.id == interaction_id
            ),
            None,
        )

    def _clamped_document_point(self, viewport_point: QPoint) -> QPointF:
        point = self.document_point_at(viewport_point)
        return QPointF(
            min(max(point.x(), 0.0), 1.0),
            min(max(point.y(), 0.0), 1.0),
        )

    def _scene_point(self, point: Point | QPointF) -> QPointF:
        return QPointF(
            point.x * self._canvas_size.width
            if isinstance(point, Point)
            else point.x() * self._canvas_size.width,
            point.y * self._canvas_size.height
            if isinstance(point, Point)
            else point.y() * self._canvas_size.height,
        )

    def _polygon_path(self, points: tuple[Point, ...]) -> QPainterPath:
        path = QPainterPath(self._scene_point(points[0]))
        for point in points[1:]:
            path.lineTo(self._scene_point(point))
        path.closeSubpath()
        return path

    @staticmethod
    def _point_distance(first: QPointF, second: QPointF) -> float:
        return hypot(first.x() - second.x(), first.y() - second.y())

    @staticmethod
    def _geometry_error(error: ValidationError) -> str:
        first = error.errors(include_url=False)[0]
        return str(first["msg"]).removeprefix("Value error, ")


__all__ = ["CardCanvas"]
