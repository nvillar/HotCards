"""Small metadata dialog for creating a bound HotCards stack."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from hotcards.domain.image_dimensions import AspectRatio
from hotcards.domain.models import (
    HYPERCARD_STYLE_ID,
    Card,
    CardRevision,
    Stack,
)


class NewStackDialog(QDialog):
    """Collect portable stack metadata without turning creation into a wizard."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("New Stack")
        self.setModal(True)

        self.name_edit = QLineEdit("Untitled Stack")
        self.name_edit.setObjectName("newStackNameEdit")

        form = QFormLayout()
        form.addRow("Name", self.name_edit)
        self.format_combo = QComboBox()
        self.format_combo.setObjectName("newStackFormatCombo")
        self.format_combo.setAccessibleName("Stack format")
        for label, aspect_ratio in (
            ("Square 1:1", AspectRatio.SQUARE),
            ("Landscape 4:3", AspectRatio.LANDSCAPE),
            ("Portrait 3:4", AspectRatio.PORTRAIT),
            ("Widescreen 16:9", AspectRatio.WIDESCREEN),
        ):
            self.format_combo.addItem(label, aspect_ratio)
        self.format_combo.setCurrentIndex(self.format_combo.findData(AspectRatio.LANDSCAPE))
        form.addRow("Format", self.format_combo)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok
        )
        buttons.accepted.connect(self._accept_if_valid)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def stack(self) -> Stack:
        """Build the initial saved document with one selected start card."""
        aspect_ratio = AspectRatio(self.format_combo.currentData())
        first_revision = CardRevision(style_id=HYPERCARD_STYLE_ID)
        first_card = Card(
            name="Card 1",
            revisions=(first_revision,),
            active_revision_id=first_revision.id,
        )
        return Stack(
            name=self.name_edit.text(),
            aspect_ratio=aspect_ratio,
            cards=(first_card,),
            start_card_id=first_card.id,
        )

    def _accept_if_valid(self) -> None:
        try:
            self.stack()
        except (TypeError, ValueError):
            if not self.name_edit.text().strip():
                self.name_edit.setFocus()
            else:
                self.format_combo.setFocus()
            return
        self.accept()


__all__ = ["NewStackDialog"]
