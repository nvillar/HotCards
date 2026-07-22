"""Simple crop-to-fill preview for imported background images."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QImageReader, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from hypergen.domain.models import CanvasSize


class CropDialog(QDialog):
    """Preview and reposition the one crop needed to fill the stack canvas."""

    def __init__(
        self,
        source_path: Path,
        canvas_size: CanvasSize,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Crop Background to Fill")
        self.setModal(True)
        self._source = self._load_pixmap(source_path)
        self._canvas_size = canvas_size

        self.preview = QLabel()
        self.preview.setObjectName("cropPreview")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(480, 320)

        self.horizontal_slider = self._slider("cropHorizontalPosition")
        self.vertical_slider = self._slider("cropVerticalPosition")
        self.horizontal_slider.valueChanged.connect(self._update_preview)
        self.vertical_slider.valueChanged.connect(self._update_preview)

        source_ratio = self._source.width() / self._source.height()
        target_ratio = canvas_size.width / canvas_size.height
        self.horizontal_slider.setEnabled(source_ratio > target_ratio)
        self.vertical_slider.setEnabled(source_ratio < target_ratio)

        form = QFormLayout()
        form.addRow("Horizontal position", self.horizontal_slider)
        form.addRow("Vertical position", self.vertical_slider)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.Ok
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(self.preview)
        layout.addLayout(form)
        layout.addWidget(buttons)
        self._update_preview()

    @property
    def position_x(self) -> float:
        return self.horizontal_slider.value() / 1000

    @property
    def position_y(self) -> float:
        return self.vertical_slider.value() / 1000

    @staticmethod
    def requires_crop(source_path: Path, canvas_size: CanvasSize) -> bool:
        pixmap = CropDialog._load_pixmap(source_path)
        source_ratio = pixmap.width() / pixmap.height()
        target_ratio = canvas_size.width / canvas_size.height
        return abs(source_ratio - target_ratio) > 1e-6

    @staticmethod
    def _load_pixmap(source_path: Path) -> QPixmap:
        reader = QImageReader(str(source_path))
        reader.setAutoTransform(True)
        image = reader.read()
        if image.isNull():
            raise ValueError(f"Could not preview image {source_path.name!r}")
        return QPixmap.fromImage(image)

    @staticmethod
    def _slider(object_name: str) -> QSlider:
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setObjectName(object_name)
        slider.setRange(0, 1000)
        slider.setValue(500)
        return slider

    def _update_preview(self) -> None:
        source_width = self._source.width()
        source_height = self._source.height()
        target_ratio = self._canvas_size.width / self._canvas_size.height
        source_ratio = source_width / source_height
        if source_ratio > target_ratio:
            crop_width = round(source_height * target_ratio)
            left = round((source_width - crop_width) * self.position_x)
            crop = QRect(left, 0, crop_width, source_height)
        else:
            crop_height = round(source_width / target_ratio)
            top = round((source_height - crop_height) * self.position_y)
            crop = QRect(0, top, source_width, crop_height)
        preview = self._source.copy(crop).scaled(
            self.preview.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview.setPixmap(preview)


__all__ = ["CropDialog"]
