"""Small metadata dialog for creating a bound HyperGen stack."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from hypergen.domain.models import (
    HYPERCARD_STYLE_ID,
    CanvasSize,
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
        self.width_spin = QSpinBox()
        self.width_spin.setObjectName("newStackWidthSpin")
        self.width_spin.setRange(64, 8192)
        self.width_spin.setValue(1024)
        self.height_spin = QSpinBox()
        self.height_spin.setObjectName("newStackHeightSpin")
        self.height_spin.setRange(64, 8192)
        self.height_spin.setValue(768)

        form = QFormLayout()
        form.addRow("Name", self.name_edit)
        form.addRow("Canvas width", self.width_spin)
        form.addRow("Canvas height", self.height_spin)

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
        first_revision = CardRevision(style_id=HYPERCARD_STYLE_ID)
        first_card = Card(
            name="Card 1",
            revisions=(first_revision,),
            active_revision_id=first_revision.id,
        )
        return Stack(
            name=self.name_edit.text(),
            canvas=CanvasSize(
                width=self.width_spin.value(),
                height=self.height_spin.value(),
            ),
            cards=(first_card,),
            start_card_id=first_card.id,
        )

    def _accept_if_valid(self) -> None:
        try:
            self.stack()
        except ValueError:
            self.name_edit.setFocus()
            return
        self.accept()


__all__ = ["NewStackDialog"]
