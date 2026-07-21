"""Progressively disclosed selected-card inspector."""

from __future__ import annotations

from uuid import UUID

from pydantic import ValidationError
from PySide6.QtCore import Signal
from PySide6.QtGui import QFocusEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QListWidget,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

from hypergen.application.commands import (
    CommandError,
    EditCardTextCommand,
    RenameCardCommand,
    SetStartCardCommand,
)
from hypergen.application.document_controller import DocumentController
from hypergen.domain.models import Card, ImageRevision, Stack


class _CommitPlainTextEdit(QPlainTextEdit):
    editing_finished = Signal()

    def focusOutEvent(self, event: QFocusEvent) -> None:
        super().focusOutEvent(event)
        self.editing_finished.emit()


class Inspector(QWidget):
    """Render selected-card snapshots and issue typed metadata commands."""

    document_changed = Signal(object)

    def __init__(self, controller: DocumentController, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.controller = controller
        self.selected_card_id: UUID | None = None
        self._rendering = False
        self.setObjectName("inspector")
        self.setMinimumWidth(240)

        heading = QLabel("Inspector")
        heading.setObjectName("inspectorHeading")

        self.card_section, card_layout = self._section("Card", expanded=True)
        self.card_section.setObjectName("cardInspectorSection")
        self.card_name_edit = QLineEdit()
        self.card_name_edit.setObjectName("cardNameEdit")
        self.scene_edit = _CommitPlainTextEdit()
        self.scene_edit.setObjectName("sceneDescriptionEdit")
        self.scene_edit.setMaximumHeight(90)
        self.interactions_edit = _CommitPlainTextEdit()
        self.interactions_edit.setObjectName("interactionDescriptionEdit")
        self.interactions_edit.setMaximumHeight(90)
        self.card_style_edit = QLineEdit()
        self.card_style_edit.setObjectName("cardStyleEdit")
        self.start_card_check = QCheckBox("Use as start card")
        self.start_card_check.setObjectName("startCardCheck")
        self.validation_error = QLabel()
        self.validation_error.setObjectName("inspectorValidationError")
        self.validation_error.setWordWrap(True)
        self.validation_error.setVisible(False)
        card_form = QFormLayout()
        card_form.addRow("Name", self.card_name_edit)
        card_form.addRow("Scene", self.scene_edit)
        card_form.addRow("Interactions", self.interactions_edit)
        card_form.addRow("Style", self.card_style_edit)
        card_layout.addLayout(card_form)
        card_layout.addWidget(self.start_card_check)
        card_layout.addWidget(self.validation_error)

        self.background_section, background_layout = self._section("Background", expanded=False)
        self.background_section.setObjectName("backgroundInspectorSection")
        self.background_value = QLabel("No background revision")
        self.background_value.setObjectName("backgroundRevisionValue")
        self.background_value.setWordWrap(True)
        background_layout.addWidget(self.background_value)

        self.hotspots_section, hotspots_layout = self._section("Hotspots", expanded=False)
        self.hotspots_section.setObjectName("hotspotsInspectorSection")
        self.hotspot_list = QListWidget()
        self.hotspot_list.setObjectName("hotspotList")
        hotspots_layout.addWidget(self.hotspot_list)

        layout = QVBoxLayout(self)
        layout.addWidget(heading)
        layout.addWidget(self.card_section)
        layout.addWidget(self.background_section)
        layout.addWidget(self.hotspots_section)
        layout.addStretch(1)

        self.card_name_edit.editingFinished.connect(self.commit_card_metadata)
        self.scene_edit.editing_finished.connect(self.commit_card_metadata)
        self.interactions_edit.editing_finished.connect(self.commit_card_metadata)
        self.card_style_edit.editingFinished.connect(self.commit_card_metadata)
        self.start_card_check.toggled.connect(self._set_start_card)
        self.render(controller.document, None)

    @staticmethod
    def _section(title: str, *, expanded: bool) -> tuple[QGroupBox, QVBoxLayout]:
        section = QGroupBox(title)
        section.setCheckable(True)
        section.setChecked(expanded)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        layout = QVBoxLayout(section)
        layout.addWidget(content)
        section.toggled.connect(content.setVisible)
        content.setVisible(expanded)
        return section, content_layout

    def render(self, document: Stack, selected_card_id: UUID | None) -> None:
        """Render only data from the supplied authoritative snapshot."""
        self._rendering = True
        try:
            card = next(
                (candidate for candidate in document.cards if candidate.id == selected_card_id),
                None,
            )
            self.selected_card_id = card.id if card is not None else None
            self.setEnabled(card is not None)
            if card is None:
                self.card_name_edit.clear()
                self.scene_edit.clear()
                self.interactions_edit.clear()
                self.card_style_edit.clear()
                self.start_card_check.setChecked(False)
                self.background_value.setText("No card selected")
                self.hotspot_list.clear()
                return
            self.card_name_edit.setText(card.name)
            self.scene_edit.setPlainText(card.scene_description)
            self.interactions_edit.setPlainText(card.interaction_description)
            self.card_style_edit.setText(card.card_style or "")
            self.start_card_check.setChecked(card.id == document.start_card_id)
            self._render_revision(self._active_revision(card))
        finally:
            self._rendering = False

    def commit_card_metadata(self) -> None:
        """Commit displayed basic metadata through typed controller commands."""
        if self._rendering or self.selected_card_id is None:
            return
        card_id = self.selected_card_id
        card = next(card for card in self.controller.document.cards if card.id == card_id)
        changed = self.controller.document
        try:
            name = self.card_name_edit.text()
            if name != card.name:
                changed = self.controller.execute(RenameCardCommand(card_id=card_id, name=name))
                card = next(card for card in changed.cards if card.id == card_id)
            field_values = (
                ("scene_description", self.scene_edit.toPlainText()),
                ("interaction_description", self.interactions_edit.toPlainText()),
                ("card_style", self.card_style_edit.text() or None),
            )
            for field, value in field_values:
                if getattr(card, field) != value:
                    changed = self.controller.execute(
                        EditCardTextCommand(
                            card_id=card_id,
                            field=field,  # type: ignore[arg-type]
                            value=value,
                        )
                    )
                    card = next(card for card in changed.cards if card.id == card_id)
        except (CommandError, ValidationError) as error:
            self.validation_error.setText(str(error))
            self.validation_error.setVisible(True)
            self.render(self.controller.document, card_id)
            return
        self.validation_error.clear()
        self.validation_error.setVisible(False)
        self.render(changed, card_id)
        self.document_changed.emit(changed)

    def _set_start_card(self, checked: bool) -> None:
        if self._rendering or self.selected_card_id is None:
            return
        changed = self.controller.execute(
            SetStartCardCommand(card_id=self.selected_card_id if checked else None)
        )
        self.render(changed, self.selected_card_id)
        self.document_changed.emit(changed)

    def _render_revision(self, revision: ImageRevision | None) -> None:
        self.hotspot_list.clear()
        if revision is None:
            self.background_value.setText("No background revision")
            return
        self.background_value.setText(revision.image_path)
        if revision.hotspot_set is None:
            return
        for interaction in revision.hotspot_set.interactions:
            self.hotspot_list.addItem(interaction.label)

    @staticmethod
    def _active_revision(card: Card) -> ImageRevision | None:
        return next(
            (
                revision
                for revision in card.image_revisions
                if revision.id == card.active_revision_id
            ),
            None,
        )


__all__ = ["Inspector"]
