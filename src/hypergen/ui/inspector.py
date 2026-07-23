"""Scrollable selected-card inspector."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from uuid import UUID

from pydantic import ValidationError
from PySide6.QtCore import QSignalBlocker, Qt, QTimer, Signal
from PySide6.QtGui import QFocusEvent, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QStackedWidget,
    QTabWidget,
    QToolButton,
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
    EditGlobalStyleCommand,
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
    render_inputs_changed = Signal()

    def __init__(
        self,
        controller: DocumentController,
        parent: QWidget | None = None,
        *,
        image_path_resolver: Callable[[str], Path | None] | None = None,
    ) -> None:
        super().__init__(parent)
        self.controller = controller
        self._image_path_resolver = image_path_resolver
        self.selected_card_id: UUID | None = None
        self._rendering = False
        self._has_background_candidate = False
        self.setObjectName("inspector")
        self.setMinimumWidth(300)

        heading = QLabel("Inspector")
        heading.setObjectName("inspectorHeading")

        self.inspector_tabs = QTabWidget()
        self.inspector_tabs.setObjectName("inspectorTabs")

        card_page = QWidget()
        card_page.setObjectName("cardInspectorTab")
        card_layout = QVBoxLayout(card_page)
        self.card_name_edit = QLineEdit()
        self.card_name_edit.setObjectName("cardNameEdit")
        name_form = QFormLayout()
        name_form.addRow("Name", self.card_name_edit)
        card_layout.addLayout(name_form)
        scene_heading = QHBoxLayout()
        scene_heading.addWidget(QLabel("Scene"))
        scene_heading.addStretch(1)
        self.enrich_scene_button = QPushButton("Enrich")
        self.enrich_scene_button.setObjectName("enrichSceneButton")
        self.enrich_scene_button.setEnabled(False)
        self.enrich_scene_button.setToolTip("Scene enrichment is planned in issue #25")
        scene_heading.addWidget(self.enrich_scene_button)
        card_layout.addLayout(scene_heading)
        self.scene_edit = _CommitPlainTextEdit()
        self.scene_edit.setObjectName("sceneDescriptionEdit")
        self.scene_edit.setPlaceholderText("Describe the image to generate")
        self.scene_edit.setMaximumHeight(150)
        card_layout.addWidget(self.scene_edit)
        self.start_card_check = QCheckBox("Use as start card")
        self.start_card_check.setObjectName("startCardCheck")
        card_layout.addWidget(self.start_card_check)
        self.validation_error = QLabel()
        self.validation_error.setObjectName("inspectorValidationError")
        self.validation_error.setWordWrap(True)
        self.validation_error.setVisible(False)
        card_layout.addWidget(self.validation_error)
        card_layout.addStretch(1)
        self.inspector_tabs.addTab(card_page, "Card")

        background_page = QWidget()
        background_page.setObjectName("backgroundInspectorTab")
        background_layout = QVBoxLayout(background_page)
        revision_header = QHBoxLayout()
        self.revision_thumbnail = QLabel("No image")
        self.revision_thumbnail.setObjectName("backgroundRevisionThumbnail")
        self.revision_thumbnail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.revision_thumbnail.setFixedSize(120, 90)
        self.revision_thumbnail.setStyleSheet(
            "border: 1px solid palette(mid); color: palette(mid);"
        )
        revision_header.addWidget(self.revision_thumbnail)
        revision_summary = QVBoxLayout()
        self.background_value = QLabel("No background revision")
        self.background_value.setObjectName("backgroundRevisionValue")
        self.background_value.setWordWrap(True)
        revision_summary.addWidget(self.background_value)
        self.revision_combo = QComboBox()
        self.revision_combo.setObjectName("backgroundRevisionCombo")
        revision_summary.addWidget(self.revision_combo)
        revision_header.addLayout(revision_summary, 1)
        background_layout.addLayout(revision_header)
        self.revision_metadata = QLabel()
        self.revision_metadata.setObjectName("backgroundRevisionMetadata")
        background_layout.addWidget(self.revision_metadata)
        self.revision_details_button = QToolButton()
        self.revision_details_button.setObjectName("backgroundRevisionDetailsButton")
        self.revision_details_button.setText("Details ▸")
        self.revision_details_button.setCheckable(True)
        self.revision_details_button.setVisible(False)
        background_layout.addWidget(self.revision_details_button)
        self.revision_details = QLabel()
        self.revision_details.setObjectName("backgroundRevisionDetails")
        self.revision_details.setWordWrap(True)
        self.revision_details.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.revision_details.setVisible(False)
        background_layout.addWidget(self.revision_details)

        style_heading = QLabel("Style")
        style_heading.setStyleSheet("font-weight: 600; margin-top: 8px;")
        background_layout.addWidget(style_heading)
        style_modes = QHBoxLayout()
        self.global_style_radio = QRadioButton("Global")
        self.global_style_radio.setObjectName("globalStyleRadio")
        self.card_style_radio = QRadioButton("This card only")
        self.card_style_radio.setObjectName("cardStyleRadio")
        self.style_mode_group = QButtonGroup(self)
        self.style_mode_group.addButton(self.global_style_radio)
        self.style_mode_group.addButton(self.card_style_radio)
        style_modes.addWidget(self.global_style_radio)
        style_modes.addWidget(self.card_style_radio)
        style_modes.addStretch(1)
        background_layout.addLayout(style_modes)
        self.style_edit = _CommitPlainTextEdit()
        self.style_edit.setObjectName("styleEdit")
        self.style_edit.setMaximumHeight(100)
        self.style_edit.setPlaceholderText("Visual style for generated backgrounds")
        background_layout.addWidget(self.style_edit)
        self.style_scope_caption = QLabel()
        self.style_scope_caption.setObjectName("styleScopeCaption")
        self.style_scope_caption.setWordWrap(True)
        background_layout.addWidget(self.style_scope_caption)

        background_actions = QHBoxLayout()
        self.generate_background_button = QPushButton("Generate")
        self.generate_background_button.setObjectName("generateBackgroundButton")
        self.import_background_button = QPushButton("Import...")
        self.import_background_button.setObjectName("importBackgroundButton")
        self.delete_revision_button = QPushButton("Delete Revision...")
        self.delete_revision_button.setObjectName("deleteRevisionButton")
        background_actions.addWidget(self.generate_background_button)
        background_actions.addWidget(self.import_background_button)
        background_actions.addWidget(self.delete_revision_button)
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
        background_layout.addStretch(1)
        self.inspector_tabs.addTab(background_page, "Background")

        interactivity_page = QWidget()
        interactivity_page.setObjectName("interactivityInspectorTab")
        hotspots_layout = QVBoxLayout(interactivity_page)
        hotspots_layout.addWidget(QLabel("Intent"))
        self.interactions_edit = _CommitPlainTextEdit()
        self.interactions_edit.setObjectName("interactionDescriptionEdit")
        self.interactions_edit.setPlaceholderText(
            "Describe what people can interact with and where it leads"
        )
        self.interactions_edit.setMaximumHeight(110)
        hotspots_layout.addWidget(self.interactions_edit)
        hotspots_heading = QHBoxLayout()
        hotspots_title = QLabel("Hotspots")
        hotspots_title.setStyleSheet("font-weight: 600; margin-top: 8px;")
        hotspots_heading.addWidget(hotspots_title)
        hotspots_heading.addStretch(1)
        self.hotspot_help_button = QToolButton()
        self.hotspot_help_button.setObjectName("hotspotHelpButton")
        self.hotspot_help_button.setText("ⓘ")
        self.hotspot_help_button.setToolTip(
            "Click to add vertices; double-click or Return to close. "
            "Drag vertices or a selected area. Double-click an edge to insert a vertex. "
            "Delete removes the selected vertex or area; Escape cancels drawing. "
            "Use the middle mouse button to pan."
        )
        hotspots_heading.addWidget(self.hotspot_help_button)
        hotspots_layout.addLayout(hotspots_heading)
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
        self.move_hotspot_up_button = QPushButton("↑")
        self.move_hotspot_up_button.setObjectName("moveHotspotUpButton")
        self.move_hotspot_up_button.setToolTip("Move hotspot up")
        self.move_hotspot_down_button = QPushButton("↓")
        self.move_hotspot_down_button.setObjectName("moveHotspotDownButton")
        self.move_hotspot_down_button.setToolTip("Move hotspot down")
        self.delete_hotspot_button = QPushButton("🗑")
        self.delete_hotspot_button.setObjectName("deleteHotspotButton")
        self.delete_hotspot_button.setToolTip("Delete hotspot")
        order_actions.addWidget(self.move_hotspot_up_button)
        order_actions.addWidget(self.move_hotspot_down_button)
        order_actions.addWidget(self.delete_hotspot_button)
        order_actions.addStretch(1)
        hotspots_layout.addLayout(order_actions)
        self.hotspot_error = QLabel()
        self.hotspot_error.setObjectName("hotspotValidationError")
        self.hotspot_error.setWordWrap(True)
        self.hotspot_error.setVisible(False)
        hotspots_layout.addWidget(self.hotspot_error)
        self.inspector_tabs.addTab(interactivity_page, "Interactivity")

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
        form_layout.addWidget(self.inspector_tabs, 1)
        self.pages.addWidget(form_page)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.pages)

        self.card_name_edit.editingFinished.connect(self.commit_card_metadata)
        self.scene_edit.editing_finished.connect(self.commit_card_metadata)
        self.interactions_edit.editing_finished.connect(self.commit_card_metadata)
        self.style_edit.editing_finished.connect(self._commit_style_text)
        self.scene_edit.textChanged.connect(self._render_input_edited)
        self.style_edit.textChanged.connect(self._render_input_edited)
        self.global_style_radio.toggled.connect(
            lambda checked: checked and self._set_style_mode(use_override=False)
        )
        self.card_style_radio.toggled.connect(
            lambda checked: checked and self._set_style_mode(use_override=True)
        )
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
        self.revision_details_button.toggled.connect(
            self._toggle_revision_details
        )
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

    def _render_input_edited(self) -> None:
        if not self._rendering:
            self.render_inputs_changed.emit()

    def has_render_prompt_input(self) -> bool:
        """Return whether the visible Scene or effective Style can render."""
        return bool(
            self.selected_card_id is not None
            and (
                self.scene_edit.toPlainText().strip()
                or self.style_edit.toPlainText().strip()
            )
        )

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
                self.style_edit.clear()
                self.start_card_check.setChecked(False)
                self.background_value.setText("No card selected")
                self.revision_combo.clear()
                self.revision_metadata.clear()
                self.revision_details.clear()
                self._set_revision_thumbnail(None)
                self.hotspots_placeholder.setText("Select a card to view hotspots.")
                self.hotspots_placeholder.setVisible(True)
                self.hotspot_list.clear()
                self._render_hotspot_properties(document, None, None)
                self._update_tab_labels(0)
                return
            self.pages.setCurrentIndex(1)
            self.card_name_edit.setText(card.name)
            self.scene_edit.setPlainText(card.scene_description)
            self.interactions_edit.setPlainText(card.interaction_description)
            use_override = card.card_style is not None
            with QSignalBlocker(self.global_style_radio):
                self.global_style_radio.setChecked(not use_override)
            with QSignalBlocker(self.card_style_radio):
                self.card_style_radio.setChecked(use_override)
            self.style_edit.setPlainText(
                card.card_style if use_override else document.global_style
            )
            self.style_scope_caption.setText(
                "This card only — overrides the global style"
                if use_override
                else "Global — edits apply to all cards"
            )
            self.start_card_check.setChecked(card.id == document.start_card_id)
            self._render_revisions(document, card)
        finally:
            self._rendering = False

    def commit_card_metadata(self) -> bool:
        """Commit displayed metadata and report whether validation succeeded."""
        if self._rendering or self.selected_card_id is None:
            return False
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
            style_value = self.style_edit.toPlainText()
            if card.card_style is None and style_value != changed.global_style:
                changed = self.controller.execute(
                    EditGlobalStyleCommand(value=style_value)
                )
            elif card.card_style is not None and style_value != card.card_style:
                changed = self.controller.execute(
                    EditCardTextCommand(
                        card_id=card_id,
                        field="card_style",
                        value=style_value,
                    )
                )
        except (CommandError, ValidationError) as error:
            self.validation_error.setText(str(error))
            self.validation_error.setVisible(True)
            self.render(self.controller.document, card_id)
            return False
        self.validation_error.clear()
        self.validation_error.setVisible(False)
        self.render(changed, card_id)
        self.document_changed.emit(changed)
        return True

    def _set_style_mode(self, *, use_override: bool) -> None:
        if self._rendering or self.selected_card_id is None:
            return
        card = next(
            card
            for card in self.controller.document.cards
            if card.id == self.selected_card_id
        )
        if use_override and card.card_style is None:
            value: str | None = self.controller.document.global_style
        elif not use_override and card.card_style is not None:
            value = None
        else:
            return
        changed = self.controller.execute(
            EditCardTextCommand(
                card_id=card.id,
                field="card_style",
                value=value,
            )
        )
        self.render(changed, card.id)
        self.document_changed.emit(changed)

    def _commit_style_text(self) -> None:
        if self._rendering or self.selected_card_id is None:
            return
        document = self.controller.document
        card = next(card for card in document.cards if card.id == self.selected_card_id)
        value = self.style_edit.toPlainText()
        if card.card_style is None:
            if value == document.global_style:
                return
            changed = self.controller.execute(EditGlobalStyleCommand(value=value))
        else:
            if value == card.card_style:
                return
            changed = self.controller.execute(
                EditCardTextCommand(
                    card_id=card.id,
                    field="card_style",
                    value=value,
                )
            )
        self.render(changed, card.id)
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
        self._has_background_candidate = candidate is not None
        self.candidate_widget.setVisible(candidate is not None)
        self._update_background_tab_label()
        if candidate is None:
            self.candidate_value.clear()
            return
        origin = "Generated" if candidate.origin.value == "generated" else "Imported"
        detail = (
            candidate.generation_metadata.render_prompt
            if candidate.generation_metadata is not None
            else candidate.source_filename or ""
        )
        self.candidate_value.setText(f"{origin} candidate ready for review")
        self.candidate_value.setToolTip(detail)
        self._set_revision_thumbnail(candidate.image_path)

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
            self.revision_details.clear()
            self.revision_details_button.setVisible(False)
            self.revision_details_button.setChecked(False)
            self._set_revision_thumbnail(None)
            self.hotspots_placeholder.setText(
                "Apply a background before adding hotspots."
            )
            self.hotspots_placeholder.setVisible(True)
            self._render_hotspot_properties(document, None, None)
            self._update_tab_labels(0)
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
                f"{metadata.model_identifier} · seed {metadata.seed} · "
                f"{metadata.width}×{metadata.height} · {metadata.step_count} steps"
            )
            self.revision_metadata.setToolTip(metadata.render_prompt)
            self.revision_details.setText(
                json.dumps(
                    metadata.model_dump(mode="json"),
                    indent=2,
                    ensure_ascii=False,
                )
            )
            self.revision_details_button.setVisible(True)
        else:
            self.revision_metadata.setText(
                f"Imported from {revision.source_filename or 'image'}"
            )
            self.revision_metadata.setToolTip("")
            self.revision_details.clear()
            self.revision_details_button.setChecked(False)
            self.revision_details_button.setVisible(False)
        image_path = (
            self._image_path_resolver(revision.image_path)
            if self._image_path_resolver is not None
            else None
        )
        self._set_revision_thumbnail(image_path)
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
        self._update_tab_labels(len(interactions))

    def _toggle_revision_details(self, visible: bool) -> None:
        self.revision_details.setVisible(visible)
        self.revision_details_button.setText(
            "Details ▾" if visible else "Details ▸"
        )

    def _set_revision_thumbnail(self, image_path: Path | None) -> None:
        if image_path is None:
            self.revision_thumbnail.setPixmap(QPixmap())
            self.revision_thumbnail.setText("No image")
            return
        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            self.revision_thumbnail.setPixmap(QPixmap())
            self.revision_thumbnail.setText("Unavailable")
            return
        self.revision_thumbnail.setText("")
        self.revision_thumbnail.setPixmap(
            pixmap.scaled(
                self.revision_thumbnail.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _update_tab_labels(self, hotspot_count: int) -> None:
        self.inspector_tabs.setTabText(0, "Card")
        self._update_background_tab_label()
        self.inspector_tabs.setTabText(2, f"Interactivity ({hotspot_count})")

    def _update_background_tab_label(self) -> None:
        self.inspector_tabs.setTabText(
            1,
            "Background ●" if self._has_background_candidate else "Background",
        )

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
        self.hotspot_help_button.setEnabled(has_revision)
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
