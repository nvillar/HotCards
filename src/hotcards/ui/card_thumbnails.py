"""Shared card thumbnail rendering for card-selection surfaces."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QColor, QIcon, QPixmap, QPixmapCache

from hotcards.domain.models import Card

ImagePathResolver = Callable[[str], Path | None]


def card_thumbnail_icon(
    card: Card,
    *,
    size: QSize,
    image_path_resolver: ImagePathResolver | None,
) -> QIcon:
    """Render the active revision background or a neutral placeholder."""
    background = card.active_revision.background
    if background is None or image_path_resolver is None:
        return placeholder_card_icon(size)
    path = image_path_resolver(background.image_path)
    if path is None:
        return placeholder_card_icon(size)
    cache_key = f"hotcards-card-thumbnail:{path}:{size.width()}x{size.height()}"
    cached = QPixmapCache.find(cache_key)
    if cached is not None and not cached.isNull():
        return QIcon(cached)
    source = QPixmap(str(path))
    if source.isNull():
        return placeholder_card_icon(size)
    scaled = source.scaled(
        size,
        Qt.AspectRatioMode.KeepAspectRatioByExpanding,
        Qt.TransformationMode.SmoothTransformation,
    )
    x = max(0, (scaled.width() - size.width()) // 2)
    y = max(0, (scaled.height() - size.height()) // 2)
    thumbnail = scaled.copy(x, y, size.width(), size.height())
    QPixmapCache.insert(cache_key, thumbnail)
    return QIcon(thumbnail)


def placeholder_card_icon(size: QSize) -> QIcon:
    pixmap = QPixmap(size)
    pixmap.fill(QColor("#d7d9dc"))
    return QIcon(pixmap)


__all__ = ["ImagePathResolver", "card_thumbnail_icon", "placeholder_card_icon"]
