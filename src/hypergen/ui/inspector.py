"""Scrollable selected-card inspector."""

from __future__ import annotations

from uuid import UUID

from pydantic import ValidationError
from PySide6.QtCore import QSignalBlocker, Qt, QTimer, Signal
from PySide6.QtGui import QFocusEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from hypergen.application.background_workflow import BackgroundCandidate
from hypergen.application.commands import (
    ChangeHotspotDestinationCommand,
    CommandError,
    CreateCardAndResolveCommand,
    DeleteInteractionCommand,
    DocumentCommand,
    EditCardTextCommand,
    RenameCardCommand,
    RenameInteractionCommand,
    ReorderHotspotCommand,
    SetStartCardCommand,
)
from hypergen.application.document_controller import DocumentController
from hypergen.domain.models import (
    Card,
    ImageRevision,
    Interaction,
    ResolvedCardReference,
    Stack,
    UnresolvedCardReference,
)


class _CommitPlainTextEdit(QPlainTextEdit):
    editing_finished = Signal()

    def focusOutEvent(self, event: QFocusEvent) -> None:
        super().focusOutEvent(event)
        self.editing_finished.emit()


class InspectorSection(QWidget):
    """An always-visible titled inspector section."""

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("font-weight: 600; margin-top: 6px;")
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        self.content = QWidget()
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(8, 4, 0, 10)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self.title_label)
        layout.addWidget(separator)
        layout.addWidget(self.content)


class Inspector(QWidget):
    """Render selected-card snapshots and issue typed metadata commands."""

    document_changed = Signal(object)
    generate_background_requested = Signal()
    import_background_requested = Signal()
    apply_background_requested = Signal()
    discard_background_requested = Signal()
    revision_activation_requested = Signal(object)
    revision_deletion_requested = Signal(object)
    hotspot_selected = Signal(object)
    add_hotspot_requested = Signal()
    add_hotspot_component_requested = Signal(object)

    def __init__(self, controller: DocumentController, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.controller = controller
        self.selected_card_id: UUID | None = None
        self._rendering = False
        self.setObjectName("inspector")
        self.setMinimumWidth(240)

        heading = QLabel("Inspector")
        heading.setObjectName("inspectorHeading")

        self.card_section, card_layout = self._section("Card")
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

        self.background_section, background_layout = self._section("Background")
        self.background_section.setObjectName("backgroundInspectorSection")
        self.background_value = QLabel("No background revision")
        self.background_value.setObjectName("backgroundRevisionValue")
        self.background_value.setWordWrap(True)
        background_layout.addWidget(self.background_value)
        background_actions = QHBoxLayout()
        self.generate_background_button = QPushButton("Generate")
        self.generate_background_button.setObjectName("generateBackgroundButton")
        self.import_background_button = QPushButton("Import...")
        self.import_background_button.setObjectName("importBackgroundButton")
        background_actions.addWidget(self.generate_background_button)
        background_actions.addWidget(self.import_background_button)
        background_layout.addLayout(background_actions)
        self.background_status = QLabel()
        self.background_status.setObjectName("backgroundStatus")
        self.background_status.setWordWrap(True)
        background_layout.addWidget(self.background_status)

        self.candidate_widget = QWidget()
        candidate_layout = QVBoxLayout(self.candidate_widget)
        candidate_layout.setContentsMargins(0, 4, 0, 4)
        self.candidate_value = QLabel()
        self.candidate_value.setObjectName("backgroundCandidateValue")
        self.candidate_value.setWordWrap(True)
        candidate_layout.addWidget(self.candidate_value)
        candidate_actions = QHBoxLayout()
        self.apply_background_button = QPushButton("Apply")
        self.apply_background_button.setObjectName("applyBackgroundButton")
        self.discard_background_button = QPushButton("Discard")
        self.discard_background_button.setObjectName("discardBackgroundButton")
        candidate_actions.addWidget(self.apply_background_button)
        candidate_actions.addWidget(self.discard_background_button)
        candidate_layout.addLayout(candidate_actions)
        background_layout.addWidget(self.candidate_widget)
        self.candidate_widget.setVisible(False)

        revision_form = QFormLayout()
        self.revision_combo = QComboBox()
        self.revision_combo.setObjectName("backgroundRevisionCombo")
        revision_form.addRow("Revision", self.revision_combo)
        background_layout.addLayout(revision_form)
        self.revision_metadata = QLabel()
        self.revision_metadata.setObjectName("backgroundRevisionMetadata")
        self.revision_metadata.setWordWrap(True)
        background_layout.addWidget(self.revision_metadata)
        self.delete_revision_button = QPushButton("Delete Revision...")
        self.delete_revision_button.setObjectName("deleteRevisionButton")
        background_layout.addWidget(self.delete_revision_button)

        self.hotspots_section, hotspots_layout = self._section("Hotspots")
        self.hotspots_section.setObjectName("hotspotsInspectorSection")
        self.hotspots_placeholder = QLabel(
            "Apply a background before adding hotspots."
        )
        self.hotspots_placeholder.setWordWrap(True)
        hotspots_layout.addWidget(self.hotspots_placeholder)
        self.hotspot_list = QListWidget()
        self.hotspot_list.setObjectName("hotspotList")
        hotspots_layout.addWidget(self.hotspot_list)
        hotspot_actions = QHBoxLayout()
        self.add_hotspot_button = QPushButton("Draw Hotspot")
        self.add_hotspot_button.setObjectName("addHotspotButton")
        self.add_component_button = QPushButton("Add Area")
        self.add_component_button.setObjectName("addHotspotComponentButton")
        hotspot_actions.addWidget(self.add_hotspot_button)
        hotspot_actions.addWidget(self.add_component_button)
        hotspots_layout.addLayout(hotspot_actions)
        self.hotspot_label_edit = QLineEdit()
        self.hotspot_label_edit.setObjectName("hotspotLabelEdit")
        self.hotspot_destination_combo = QComboBox()
        self.hotspot_destination_combo.setObjectName("hotspotDestinationCombo")
        hotspot_form = QFormLayout()
        hotspot_form.addRow("Label", self.hotspot_label_edit)
        hotspot_form.addRow("Destination", self.hotspot_destination_combo)
        hotspots_layout.addLayout(hotspot_form)
        order_actions = QHBoxLayout()
        self.move_hotspot_up_button = QPushButton("Move Up")
        self.move_hotspot_up_button.setObjectName("moveHotspotUpButton")
        self.move_hotspot_down_button = QPushButton("Move Down")
        self.move_hotspot_down_button.setObjectName("moveHotspotDownButton")
        order_actions.addWidget(self.move_hotspot_up_button)
        order_actions.addWidget(self.move_hotspot_down_button)
        hotspots_layout.addLayout(order_actions)
        self.delete_hotspot_button = QPushButton("Delete Hotspot...")
        self.delete_hotspot_button.setObjectName("deleteHotspotButton")
        hotspots_layout.addWidget(self.delete_hotspot_button)
        self.hotspot_help = QLabel(
            "Click to add vertices; double-click or Return to close. "
            "Drag vertices or a selected area. Double-click an edge to insert a vertex. "
            "Delete removes the selected vertex or area; Escape cancels drawing. "
            "Use the middle mouse button to pan."
        )
        self.hotspot_help.setWordWrap(True)
        hotspots_layout.addWidget(self.hotspot_help)
        self.hotspot_error = QLabel()
        self.hotspot_error.setObjectName("hotspotValidationError")
        self.hotspot_error.setWordWrap(True)
        self.hotspot_error.setVisible(False)
        hotspots_layout.addWidget(self.hotspot_error)

        self.pages = QStackedWidget()
        empty_page = QWidget()
        empty_layout = QVBoxLayout(empty_page)
        empty_layout.addStretch(1)
        self.empty_state_label = QLabel("Select or create a card to inspect its details")
        self.empty_state_label.setObjectName("emptyInspectorLabel")
        self.empty_state_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_state_label.setWordWrap(True)
        empty_layout.addWidget(self.empty_state_label)
        empty_layout.addStretch(1)
        self.pages.addWidget(empty_page)

        form_page = QWidget()
        form_layout = QVBoxLayout(form_page)
        form_layout.addWidget(heading)
        form_layout.addWidget(self.card_section)
        form_layout.addWidget(self.background_section)
        form_layout.addWidget(self.hotspots_section)
        form_layout.addStretch(1)
        form_scroll = QScrollArea()
        form_scroll.setObjectName("inspectorScrollArea")
        form_scroll.setWidgetResizable(True)
        form_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        form_scroll.setWidget(form_page)
        self.pages.addWidget(form_scroll)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.pages)

        self.card_name_edit.editingFinished.connect(self.commit_card_metadata)
        self.scene_edit.editing_finished.connect(self.commit_card_metadata)
        self.interactions_edit.editing_finished.connect(self.commit_card_metadata)
        self.card_style_edit.editingFinished.connect(self.commit_card_metadata)
        self.start_card_check.toggled.connect(self._set_start_card)
        self.generate_background_button.clicked.connect(
            self.generate_background_requested
        )
        self.import_background_button.clicked.connect(
            self.import_background_requested
        )
        self.apply_background_button.clicked.connect(
            self.apply_background_requested
        )
        self.discard_background_button.clicked.connect(
            self.discard_background_requested
        )
        self.revision_combo.currentIndexChanged.connect(self._revision_selected)
        self.delete_revision_button.clicked.connect(self._delete_selected_revision)
        self.hotspot_list.currentItemChanged.connect(
            self._hotspot_selection_changed
        )
        self.add_hotspot_button.clicked.connect(self.add_hotspot_requested)
        self.add_component_button.clicked.connect(self._add_hotspot_component)
        self.hotspot_label_edit.editingFinished.connect(
            self._commit_hotspot_label
        )
        self.hotspot_destination_combo.activated.connect(
            self._destination_changed
        )
        self.move_hotspot_up_button.clicked.connect(
            lambda: self._move_hotspot(-1)
        )
        self.move_hotspot_down_button.clicked.connect(
            lambda: self._move_hotspot(1)
        )
        self.delete_hotspot_button.clicked.connect(self._delete_hotspot)
        self.render(controller.document, None)

    @staticmethod
    def _section(
        title: str,
    ) -> tuple[InspectorSection, QVBoxLayout]:
        section = InspectorSection(title)
        return section, section.content_layout

    def render(self, document: Stack, selected_card_id: UUID | None) -> None:
        """Render only data from the supplied authoritative snapshot."""
        self._rendering = True
        try:
            card = next(
                (candidate for candidate in document.cards if candidate.id == selected_card_id),
                None,
            )
            self.selected_card_id = card.id if card is not None else None
            if card is None:
                self.pages.setCurrentIndex(0)
                self.card_name_edit.clear()
                self.scene_edit.clear()
                self.interactions_edit.clear()
                self.card_style_edit.clear()
                self.start_card_check.setChecked(False)
                self.background_value.setText("No card selected")
                self.revision_combo.clear()
                self.revision_metadata.clear()
                self.hotspots_placeholder.setText("Select a card to view hotspots.")
                self.hotspots_placeholder.setVisible(True)
                self.hotspot_list.clear()
                self._render_hotspot_properties(document, None, None)
                return
            self.pages.setCurrentIndex(1)
            self.card_name_edit.setText(card.name)
            self.scene_edit.setPlainText(card.scene_description)
            self.interactions_edit.setPlainText(card.interaction_description)
            self.card_style_edit.setText(card.card_style or "")
            self.start_card_check.setChecked(card.id == document.start_card_id)
            self._render_revisions(document, card)
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

    def set_background_capabilities(
        self,
        *,
        can_generate: bool,
        generate_reason: str,
        can_import: bool,
        import_reason: str,
        busy: bool,
    ) -> None:
        self.generate_background_button.setEnabled(can_generate and not busy)
        self.generate_background_button.setToolTip(generate_reason)
        self.import_background_button.setEnabled(can_import and not busy)
        self.import_background_button.setToolTip(import_reason)

    def set_background_status(self, message: str, *, detail: str = "") -> None:
        self.background_status.setText(message)
        self.background_status.setToolTip(detail)

    @property
    def selected_interaction_id(self) -> UUID | None:
        item = self.hotspot_list.currentItem()
        if item is None:
            return None
        interaction_id = item.data(Qt.ItemDataRole.UserRole)
        return interaction_id if isinstance(interaction_id, UUID) else None

    def select_interaction(self, interaction_id: UUID | None) -> None:
        for row in range(self.hotspot_list.count()):
            item = self.hotspot_list.item(row)
            if item.data(Qt.ItemDataRole.UserRole) == interaction_id:
                self.hotspot_list.setCurrentRow(row)
                return
        self.hotspot_list.setCurrentRow(-1)

    def set_hotspot_error(self, message: str) -> None:
        self.hotspot_error.setText(message)
        self.hotspot_error.setVisible(bool(message))

    def show_background_candidate(
        self,
        candidate: BackgroundCandidate | None,
    ) -> None:
        self.candidate_widget.setVisible(candidate is not None)
        if candidate is None:
            self.candidate_value.clear()
            return
        origin = "Generated" if candidate.origin.value == "generated" else "Imported"
        detail = (
            candidate.generation_metadata.derived_prompt
            if candidate.generation_metadata is not None
            else candidate.source_filename or ""
        )
        self.candidate_value.setText(
            f"{origin} candidate ready for review"
            + (f"\nPrompt: {detail}" if detail and candidate.generation_metadata else "")
        )
        self.candidate_value.setToolTip(detail)

    def _render_revisions(self, document: Stack, card: Card) -> None:
        active_revision = self._active_revision(card)
        with QSignalBlocker(self.revision_combo):
            self.revision_combo.clear()
            for index, revision in enumerate(card.image_revisions, start=1):
                origin = "Generated" if revision.origin.value == "generated" else "Imported"
                self.revision_combo.addItem(f"{index}. {origin}", revision.id)
            active_index = self._combo_index_for_data(
                self.revision_combo,
                card.active_revision_id,
            )
            self.revision_combo.setCurrentIndex(active_index)
        self.revision_combo.setEnabled(bool(card.image_revisions))
        self.delete_revision_button.setEnabled(active_revision is not None)
        self._render_revision(document, active_revision)

    def _render_revision(
        self,
        document: Stack,
        revision: ImageRevision | None,
    ) -> None:
        desired_interaction_id = self.selected_interaction_id
        with QSignalBlocker(self.hotspot_list):
            self.hotspot_list.clear()
        if revision is None:
            self.background_value.setText("No background revision")
            self.revision_metadata.clear()
            self.hotspots_placeholder.setText(
                "Apply a background before adding hotspots."
            )
            self.hotspots_placeholder.setVisible(True)
            self._render_hotspot_properties(document, None, None)
            return
        self.hotspots_placeholder.setText("No hotspots yet.")
        self.hotspots_placeholder.setVisible(
            revision.hotspot_set is None
            or not revision.hotspot_set.interactions
        )
        self.background_value.setText(
            "Generated background"
            if revision.origin.value == "generated"
            else revision.source_filename or "Imported background"
        )
        if revision.generation_metadata is not None:
            metadata = revision.generation_metadata
            self.revision_metadata.setText(
                f"{metadata.model_identifier} | seed {metadata.seed} | "
                f"{metadata.width}x{metadata.height} | {metadata.step_count} steps"
                f"\nPrompt: {metadata.derived_prompt}"
            )
            self.revision_metadata.setToolTip(metadata.derived_prompt)
        else:
            self.revision_metadata.setText(
                f"Imported from {revision.source_filename or 'image'}"
            )
            self.revision_metadata.setToolTip("")
        interactions = (
            revision.hotspot_set.interactions
            if revision.hotspot_set is not None
            else ()
        )
        with QSignalBlocker(self.hotspot_list):
            for interaction in interactions:
                item = QListWidgetItem(
                    f"{interaction.label} ({len(interaction.polygons)} "
                    f"{'area' if len(interaction.polygons) == 1 else 'areas'})"
                )
                item.setData(Qt.ItemDataRole.UserRole, interaction.id)
                self.hotspot_list.addItem(item)
            selected_row = next(
                (
                    row
                    for row in range(self.hotspot_list.count())
                    if self.hotspot_list.item(row).data(Qt.ItemDataRole.UserRole)
                    == desired_interaction_id
                ),
                0 if interactions else -1,
            )
            self.hotspot_list.setCurrentRow(selected_row)
        selected = interactions[selected_row] if selected_row >= 0 else None
        self._render_hotspot_properties(document, revision, selected)

    def _revision_selected(self, index: int) -> None:
        if self._rendering or index < 0:
            return
        revision_id = self.revision_combo.itemData(index)
        if isinstance(revision_id, UUID):
            self.revision_activation_requested.emit(revision_id)

    def _delete_selected_revision(self) -> None:
        revision_id = self.revision_combo.currentData()
        if isinstance(revision_id, UUID):
            self.revision_deletion_requested.emit(revision_id)

    def _hotspot_selection_changed(
        self,
        current: QListWidgetItem | None,
        _previous: QListWidgetItem | None,
    ) -> None:
        if self._rendering:
            return
        interaction_id = (
            current.data(Qt.ItemDataRole.UserRole)
            if current is not None
            else None
        )
        revision = self._active_revision_for_selected_card()
        interaction = self._selected_interaction()
        self._render_hotspot_properties(
            self.controller.document,
            revision,
            interaction,
        )
        self.hotspot_selected.emit(interaction_id)

    def _render_hotspot_properties(
        self,
        document: Stack,
        revision: ImageRevision | None,
        interaction: Interaction | None,
    ) -> None:
        has_revision = revision is not None
        has_interaction = interaction is not None
        self.add_hotspot_button.setEnabled(has_revision)
        self.add_component_button.setEnabled(has_interaction)
        self.hotspot_label_edit.setEnabled(has_interaction)
        self.hotspot_destination_combo.setEnabled(has_interaction)
        self.move_hotspot_up_button.setEnabled(
            has_interaction and self.hotspot_list.currentRow() > 0
        )
        self.move_hotspot_down_button.setEnabled(
            has_interaction
            and self.hotspot_list.currentRow() < self.hotspot_list.count() - 1
        )
        self.delete_hotspot_button.setEnabled(has_interaction)
        self.hotspot_help.setVisible(has_revision)
        with QSignalBlocker(self.hotspot_label_edit):
            self.hotspot_label_edit.setText(
                interaction.label if interaction is not None else ""
            )
        with QSignalBlocker(self.hotspot_destination_combo):
            self.hotspot_destination_combo.clear()
            self.hotspot_destination_combo.addItem("Unresolved", "unresolved")
            for card in document.cards:
                self.hotspot_destination_combo.addItem(card.name, card.id)
            self.hotspot_destination_combo.addItem(
                "Create New Card...",
                "create",
            )
            if interaction is None:
                self.hotspot_destination_combo.setCurrentIndex(-1)
            elif isinstance(
                interaction.action.target,
                ResolvedCardReference,
            ):
                self.hotspot_destination_combo.setCurrentIndex(
                    self._combo_index_for_data(
                        self.hotspot_destination_combo,
                        interaction.action.target.target_card_id
                    )
                )
            else:
                unresolved = interaction.action.target
                label = (
                    f"Unresolved ({unresolved.target_name})"
                    if unresolved.target_name
                    else "Unresolved"
                )
                self.hotspot_destination_combo.setItemText(0, label)
                self.hotspot_destination_combo.setCurrentIndex(0)

    def _add_hotspot_component(self) -> None:
        interaction_id = self.selected_interaction_id
        if interaction_id is not None:
            self.add_hotspot_component_requested.emit(interaction_id)

    def _commit_hotspot_label(self) -> None:
        interaction = self._selected_interaction()
        revision_id = self._active_revision_id()
        if (
            self._rendering
            or interaction is None
            or revision_id is None
            or self.selected_card_id is None
            or self.hotspot_label_edit.text() == interaction.label
        ):
            return
        self._execute_hotspot_command(
            RenameInteractionCommand(
                card_id=self.selected_card_id,
                revision_id=revision_id,
                interaction_id=interaction.id,
                label=self.hotspot_label_edit.text(),
            )
        )

    def _destination_changed(self, index: int) -> None:
        if self._rendering or index < 0:
            return
        interaction = self._selected_interaction()
        revision_id = self._active_revision_id()
        card_id = self.selected_card_id
        if interaction is None or revision_id is None or card_id is None:
            return
        destination = self.hotspot_destination_combo.itemData(index)
        QTimer.singleShot(
            0,
            lambda: self._apply_destination_change(
                card_id,
                revision_id,
                interaction.id,
                destination,
            ),
        )

    def _apply_destination_change(
        self,
        card_id: UUID,
        revision_id: UUID,
        interaction_id: UUID,
        destination: object,
    ) -> None:
        interaction = self._interaction_by_id(
            card_id,
            revision_id,
            interaction_id,
        )
        if interaction is None:
            self.set_hotspot_error("The selected hotspot no longer exists.")
            self.render(self.controller.document, self.selected_card_id)
            return
        if destination == "create":
            name, accepted = QInputDialog.getText(
                self,
                "Create Destination Card",
                "Card name",
            )
            if not accepted:
                self.render(self.controller.document, self.selected_card_id)
                return
            command = CreateCardAndResolveCommand(
                source_card_id=card_id,
                revision_id=revision_id,
                interaction_id=interaction_id,
                card_name=name,
            )
        else:
            target = (
                ResolvedCardReference(target_card_id=destination)
                if isinstance(destination, UUID)
                else UnresolvedCardReference()
            )
            if interaction.action.target == target:
                return
            command = ChangeHotspotDestinationCommand(
                card_id=card_id,
                revision_id=revision_id,
                interaction_id=interaction_id,
                destination=target,
            )
        self._execute_hotspot_command(command)

    def _move_hotspot(self, offset: int) -> None:
        interaction_id = self.selected_interaction_id
        revision_id = self._active_revision_id()
        card_id = self.selected_card_id
        destination = self.hotspot_list.currentRow() + offset
        if (
            interaction_id is None
            or revision_id is None
            or card_id is None
            or not 0 <= destination < self.hotspot_list.count()
        ):
            return
        self._execute_hotspot_command(
            ReorderHotspotCommand(
                card_id=card_id,
                revision_id=revision_id,
                interaction_id=interaction_id,
                new_index=destination,
            )
        )

    def _delete_hotspot(self) -> None:
        interaction_id = self.selected_interaction_id
        revision_id = self._active_revision_id()
        card_id = self.selected_card_id
        if interaction_id is None or revision_id is None or card_id is None:
            return
        dialog = QMessageBox(
            QMessageBox.Icon.Warning,
            "Delete Hotspot?",
            "Delete this hotspot and all of its polygon areas?",
            parent=self,
        )
        delete_button = dialog.addButton(
            "Delete Hotspot",
            QMessageBox.ButtonRole.DestructiveRole,
        )
        dialog.addButton(QMessageBox.StandardButton.Cancel)
        dialog.exec()
        if dialog.clickedButton() is not delete_button:
            return
        self._execute_hotspot_command(
            DeleteInteractionCommand(
                card_id=card_id,
                revision_id=revision_id,
                interaction_id=interaction_id,
            )
        )

    def _execute_hotspot_command(self, command: DocumentCommand) -> None:
        card_id = self.selected_card_id
        if card_id is None:
            return
        try:
            changed = self.controller.execute(command)
        except (CommandError, ValidationError) as error:
            self.set_hotspot_error(str(error))
            self.render(self.controller.document, card_id)
            return
        self.set_hotspot_error("")
        self.render(changed, card_id)
        self.document_changed.emit(changed)

    def _selected_interaction(self) -> Interaction | None:
        interaction_id = self.selected_interaction_id
        revision = self._active_revision_for_selected_card()
        if interaction_id is None or revision is None or revision.hotspot_set is None:
            return None
        return next(
            (
                interaction
                for interaction in revision.hotspot_set.interactions
                if interaction.id == interaction_id
            ),
            None,
        )

    def _interaction_by_id(
        self,
        card_id: UUID,
        revision_id: UUID,
        interaction_id: UUID,
    ) -> Interaction | None:
        card = next(
            (
                card
                for card in self.controller.document.cards
                if card.id == card_id
            ),
            None,
        )
        if card is None:
            return None
        revision = next(
            (
                revision
                for revision in card.image_revisions
                if revision.id == revision_id
            ),
            None,
        )
        if revision is None or revision.hotspot_set is None:
            return None
        return next(
            (
                interaction
                for interaction in revision.hotspot_set.interactions
                if interaction.id == interaction_id
            ),
            None,
        )

    def _active_revision_id(self) -> UUID | None:
        revision = self._active_revision_for_selected_card()
        return revision.id if revision is not None else None

    def _active_revision_for_selected_card(self) -> ImageRevision | None:
        card = next(
            (
                card
                for card in self.controller.document.cards
                if card.id == self.selected_card_id
            ),
            None,
        )
        return self._active_revision(card) if card is not None else None

    @staticmethod
    def _combo_index_for_data(combo: QComboBox, value: object) -> int:
        return next(
            (
                index
                for index in range(combo.count())
                if combo.itemData(index) == value
            ),
            -1,
        )

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
