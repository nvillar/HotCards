"""Shared fixed-aspect card image viewport."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QPoint, QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap, QResizeEvent, QWheelEvent
from PySide6.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
    QWidget,
)

from hypergen.domain.models import CanvasSize


class CardCanvas(QGraphicsView):
    """Display one active image or candidate using stable document coordinates."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setObjectName("cardCanvas")
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
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
        self._border_item = border
        self.fit_to_window()

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
        factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        current_scale = self.transform().m11()
        next_scale = current_scale * factor
        if 0.05 <= next_scale <= 20:
            self._fit_mode = False
            self.scale(factor, factor)
        event.accept()

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        if self._fit_mode:
            self.fit_to_window()

    def _clear_scene_items(self) -> None:
        self.scene().clear()
        self._image_item = None
        self._border_item = None
        self._message_item = None


__all__ = ["CardCanvas"]
