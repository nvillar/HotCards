"""Packaged visual identity shared by HotCards windows."""

from pathlib import Path

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QIcon, QPainter, QPainterPath, QPixmap

APPLICATION_ARTWORK_PATH = (
    Path(__file__).resolve().parent.parent / "assets" / "donkeyfigs.png"
)
APPLICATION_ICON_SIZES = (16, 32, 64, 128, 256, 512, 1024)


def application_artwork() -> QPixmap:
    """Load the packaged application artwork or fail startup explicitly."""
    artwork = QPixmap(str(APPLICATION_ARTWORK_PATH))
    if artwork.isNull():
        raise RuntimeError(
            f"could not load HotCards artwork: {APPLICATION_ARTWORK_PATH}"
        )
    return artwork


def application_icon() -> QIcon:
    """Return rounded artwork at standard application icon sizes."""
    artwork = application_artwork()
    icon = QIcon()
    for size in APPLICATION_ICON_SIZES:
        icon.addPixmap(application_icon_pixmap(size, artwork=artwork))
    return icon


def application_icon_pixmap(
    size: int,
    *,
    artwork: QPixmap | None = None,
) -> QPixmap:
    """Render one padded, rounded icon tile at the requested square size."""
    if size <= 0:
        raise ValueError("application icon size must be positive")
    source = artwork if artwork is not None else application_artwork()
    icon = QPixmap(size, size)
    icon.fill(Qt.GlobalColor.transparent)
    inset = size * 0.06
    bounds = QRectF(inset, inset, size - (2 * inset), size - (2 * inset))
    corner_radius = bounds.width() * 0.18
    clip = QPainterPath()
    clip.addRoundedRect(bounds, corner_radius, corner_radius)
    painter = QPainter(icon)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
    painter.setClipPath(clip)
    painter.drawPixmap(bounds, source, QRectF(source.rect()))
    painter.end()
    return icon


__all__ = [
    "APPLICATION_ARTWORK_PATH",
    "APPLICATION_ICON_SIZES",
    "application_artwork",
    "application_icon",
    "application_icon_pixmap",
]
