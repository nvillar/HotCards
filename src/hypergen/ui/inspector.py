"""Minimal revision-local Background and Hotspots inspector."""

from __future__ import annotations

from uuid import UUID

from pydantic import ValidationError
from PySide6.QtCore import QSignalBlocker, Qt, QTimer, Signal
from PySide6.QtGui import QFocusEvent
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QStackedWidget,
    QStyle,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from hypergen.application.commands import (
    AddInteractionCommand,
    ChangeHotspotDestinationCommand,
    CommandError,
    CreateCardAndResolveCommand,
    DeleteInteractionCommand,
    DocumentCommand,
    EditRevisionDescriptionCommand,
    ReorderHotspotCommand,
    SetRevisionEnrichedDescriptionCommand,
    SetRevisionReferenceCommand,
)
from hypergen.application.document_controller import DocumentController
from hypergen.application.scene_enrichment_workflow import (
    enrichment_reference_snapshots,
)
from hypergen.domain.models import (
    Card,
    CardRevision,
    Interaction,
    NavigateAction,
    ReferenceRole,
    ResolvedCardReference,
    Stack,
    UnresolvedCardReference,
)


class _CommitPlainTextEdit(QPlainTextEdit):
    editing_finished = Signal()

    def focusOutEvent(self, event: QFocusEvent) -> None:
        super().focusOutEvent(event)
        self.editing_finished.emit()


def _compact_icon_button(
    style: QStyle,
    *,
    object_name: str,
    icon: QStyle.StandardPixmap,
    accessible_name: str,
    tooltip: str,
) -> QToolButton:
    button = QToolButton()
    button.setObjectName(object_name)
    button.setIcon(style.standardIcon(icon))
    button.setAccessibleName(accessible_name)
    button.setToolTip(tooltip)
    return button


def _compact_text_button(
    text: str,
    *,
    object_name: str,
    accessible_name: str,
    tooltip: str,
) -> QToolButton:
    button = QToolButton()
    button.setObjectName(object_name)
    button.setText(text)
    button.setAccessibleName(accessible_name)
    button.setToolTip(tooltip)
    font = button.font()
    font.setPointSizeF(max(font.pointSizeF() + 4.0, 16.0))
    font.setBold(True)
    button.setFont(font)
    return button


class Inspector(QWidget):
    """Render the selected active revision and issue typed commands."""

    document_changed = Signal(object)
    generate_background_requested = Signal()
    enrich_scene_requested = Signal()
    hotspot_selected = Signal(object)
    change_applied = Signal(str, object)
    render_inputs_changed = Signal()

    def __init__(
        self,
        controller: DocumentController,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.controller = controller
        self.selected_card_id: UUID | None = None
        self._rendered_revision_id: UUID | None = None
        self._description_mode = "original"
        self._had_enrichment = False
        self._enrich_using_text = ""
        self._generate_using_text = ""
        self._enrich_reason = ""
        self._generate_reason = ""
        self._can_enrich = False
        self._enrichment_busy = False
        self._enrichment_current: bool | None = None
        self._rendering = False
        self.setObjectName("inspector")
        self.setMinimumWidth(300)

        root = QVBoxLayout(self)
        horizontal_margin = self.style().pixelMetric(
            QStyle.PixelMetric.PM_LayoutLeftMargin
        )
        root.setContentsMargins(horizontal_margin, 16, horizontal_margin, 0)

        self.pages = QStackedWidget()
        empty_page = QWidget()
        empty_layout = QVBoxLayout(empty_page)
        empty_layout.addStretch(1)
        self.empty_state_label = QLabel("Select or create a card")
        self.empty_state_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_layout.addWidget(self.empty_state_label)
        empty_layout.addStretch(1)
        self.pages.addWidget(empty_page)

        self.inspector_tabs = QTabWidget()
        self.inspector_tabs.setObjectName("inspectorTabs")
        self.pages.addWidget(self.inspector_tabs)
        root.addWidget(self.pages, 1)

        self._build_background_tab()
        self._build_hotspots_tab()
        self._connect_signals()

    def _build_background_tab(self) -> None:
        scroll = QScrollArea()
        scroll.setObjectName("backgroundInspectorTab")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        page = QWidget()
        page.setObjectName("backgroundInspectorContent")
        layout = QVBoxLayout(page)

        layout.addWidget(QLabel("Description"))
        self.description_edit = _CommitPlainTextEdit()
        self.description_edit.setObjectName("descriptionEdit")
        self.description_edit.setPlaceholderText("Description")
        self.description_edit.setAccessibleName("Description")
        editor_height = round(
            (self.description_edit.fontMetrics().lineSpacing() * 10 + 20) * 1.25
        )
        self.description_edit.setFixedHeight(editor_height)
        layout.addWidget(self.description_edit)
        self.description_error = QLabel()
        self.description_error.setObjectName("descriptionValidationError")
        self.description_error.setWordWrap(True)
        self.description_error.setVisible(False)
        layout.addWidget(self.description_error)

        self.description_toggle = QWidget()
        self.description_toggle.setObjectName("descriptionToggle")
        toggle_layout = QHBoxLayout(self.description_toggle)
        toggle_layout.setContentsMargins(0, 0, 0, 0)
        self.original_description_button = QRadioButton("Original")
        self.original_description_button.setObjectName("originalDescriptionButton")
        self.original_description_button.setToolTip(
            "Display and edit the authored Description"
        )
        self.enriched_description_button = QRadioButton("Enriched")
        self.enriched_description_button.setObjectName("enrichedDescriptionButton")
        self.enriched_description_button.setToolTip(
            "Display and edit the Enriched Description"
        )
        self.description_button_group = QButtonGroup(self)
        self.description_button_group.setExclusive(True)
        self.description_button_group.addButton(self.original_description_button)
        self.description_button_group.addButton(self.enriched_description_button)
        toggle_layout.addWidget(self.original_description_button)
        toggle_layout.addWidget(self.enriched_description_button)
        toggle_layout.addStretch(1)
        layout.addWidget(self.description_toggle)

        layout.addSpacing(8)
        self.references_label = QLabel("References")
        self.references_label.setObjectName("referencesLabel")
        layout.addWidget(self.references_label)
        self.references_panel = QFrame()
        self.references_panel.setObjectName("referencesPanel")
        self.references_panel.setFrameShape(QFrame.Shape.StyledPanel)
        references_layout = QVBoxLayout(self.references_panel)
        self.subject_reference_combo = QComboBox()
        self.style_reference_combo = QComboBox()
        self.setting_reference_combo = QComboBox()
        self.reference_combos = {
            ReferenceRole.SUBJECT: self.subject_reference_combo,
            ReferenceRole.STYLE: self.style_reference_combo,
            ReferenceRole.SETTING: self.setting_reference_combo,
        }
        reference_details = (
            (
                ReferenceRole.SUBJECT,
                "Subject",
                "Preserve the recognizable appearance of a subject, object, or place",
            ),
            (
                ReferenceRole.STYLE,
                "Style",
                "Transfer medium, linework, texture, palette, and lighting treatment",
            ),
            (
                ReferenceRole.SETTING,
                "Setting",
                "Preserve environment, architecture, materials, and location vocabulary",
            ),
        )
        for role, label_text, tooltip in reference_details:
            row = QHBoxLayout()
            label = QLabel(label_text)
            label.setMinimumWidth(58)
            label.setToolTip(tooltip)
            row.addWidget(label)
            combo = self.reference_combos[role]
            combo.setObjectName(f"{role.value}ReferenceCombo")
            combo.setAccessibleName(f"{label_text} reference card")
            combo.setToolTip(tooltip)
            row.addWidget(combo, 1)
            references_layout.addLayout(row)
        self.reference_error = QLabel()
        self.reference_error.setObjectName("referenceValidationError")
        self.reference_error.setWordWrap(True)
        self.reference_error.setVisible(False)
        references_layout.addWidget(self.reference_error)
        layout.addWidget(self.references_panel)

        layout.addSpacing(8)
        self.enrich_scene_button = QPushButton("Enrich Description")
        self.enrich_scene_button.setObjectName("enrichSceneButton")
        layout.addWidget(self.enrich_scene_button)

        layout.addSpacing(8)
        self.generate_background_button = QPushButton("Generate Image")
        self.generate_background_button.setObjectName("generateBackgroundButton")
        layout.addWidget(self.generate_background_button)

        layout.addStretch(1)
        scroll.setWidget(page)
        self.inspector_tabs.addTab(scroll, "Background")

    def _build_hotspots_tab(self) -> None:
        page = QWidget()
        page.setObjectName("hotspotsInspectorTab")
        layout = QVBoxLayout(page)

        self.hotspots_placeholder = QLabel("No hotspots yet.")
        self.hotspots_placeholder.setWordWrap(True)
        layout.addWidget(self.hotspots_placeholder)
        self.hotspot_list = QListWidget()
        self.hotspot_list.setObjectName("hotspotList")
        layout.addWidget(self.hotspot_list)

        controls = QHBoxLayout()
        self.move_hotspot_up_button = _compact_icon_button(
            self.style(),
            object_name="moveHotspotUpButton",
            icon=QStyle.StandardPixmap.SP_ArrowUp,
            accessible_name="Move hotspot up",
            tooltip="Move hotspot up",
        )
        self.move_hotspot_down_button = _compact_icon_button(
            self.style(),
            object_name="moveHotspotDownButton",
            icon=QStyle.StandardPixmap.SP_ArrowDown,
            accessible_name="Move hotspot down",
            tooltip="Move hotspot down",
        )
        self.add_hotspot_button = _compact_text_button(
            "+",
            object_name="addHotspotButton",
            accessible_name="Add hotspot",
            tooltip="Add hotspot",
        )
        self.delete_hotspot_button = _compact_text_button(
            "−",
            object_name="deleteHotspotButton",
            accessible_name="Delete hotspot",
            tooltip="Delete hotspot",
        )
        control_extent = max(
            button.sizeHint().width()
            for button in (
                self.move_hotspot_up_button,
                self.move_hotspot_down_button,
                self.add_hotspot_button,
                self.delete_hotspot_button,
            )
        )
        control_extent = max(
            control_extent,
            *(
                button.sizeHint().height()
                for button in (
                    self.move_hotspot_up_button,
                    self.move_hotspot_down_button,
                    self.add_hotspot_button,
                    self.delete_hotspot_button,
                )
            ),
        )
        for button in (
            self.move_hotspot_up_button,
            self.move_hotspot_down_button,
            self.add_hotspot_button,
            self.delete_hotspot_button,
        ):
            button.setFixedSize(control_extent, control_extent)
        controls.addWidget(self.move_hotspot_up_button)
        controls.addWidget(self.move_hotspot_down_button)
        controls.addStretch(1)
        controls.addWidget(self.add_hotspot_button)
        controls.addWidget(self.delete_hotspot_button)
        layout.addLayout(controls)

        self.hotspot_destination_combo = QComboBox()
        self.hotspot_destination_combo.setObjectName("hotspotDestinationCombo")
        self.hotspot_destination_combo.setAccessibleName("Hotspot destination")
        layout.addWidget(self.hotspot_destination_combo)

        self.hotspot_error = QLabel()
        self.hotspot_error.setObjectName("hotspotValidationError")
        self.hotspot_error.setWordWrap(True)
        self.hotspot_error.setVisible(False)
        layout.addWidget(self.hotspot_error)
        self._hotspots_tab_index = self.inspector_tabs.addTab(page, "Hotspots")

    @property
    def hotspots_active(self) -> bool:
        return self.inspector_tabs.currentIndex() == self._hotspots_tab_index

    def _connect_signals(self) -> None:
        self.description_edit.editing_finished.connect(
            self.commit_revision_metadata
        )
        self.description_edit.textChanged.connect(self._render_inputs_changed)
        self.original_description_button.clicked.connect(
            lambda: self._switch_description_mode("original")
        )
        self.enriched_description_button.clicked.connect(
            lambda: self._switch_description_mode("enriched")
        )
        self.enrich_scene_button.clicked.connect(self.enrich_scene_requested)
        self.generate_background_button.clicked.connect(self.generate_background_requested)
        for role, combo in self.reference_combos.items():
            combo.currentIndexChanged.connect(
                lambda index, selected_role=role: self._reference_changed(
                    selected_role,
                    index,
                )
            )
        self.hotspot_list.currentItemChanged.connect(self._hotspot_selection_changed)
        self.move_hotspot_up_button.clicked.connect(lambda: self._move_hotspot(-1))
        self.move_hotspot_down_button.clicked.connect(lambda: self._move_hotspot(1))
        self.add_hotspot_button.clicked.connect(self._add_hotspot)
        self.delete_hotspot_button.clicked.connect(self._delete_hotspot)
        self.hotspot_destination_combo.currentIndexChanged.connect(self._destination_changed)

    def _render_inputs_changed(self) -> None:
        if not self._rendering:
            self._update_enrichment_freshness()
            self.render_inputs_changed.emit()

    def render(self, document: Stack, selected_card_id: UUID | None) -> None:
        previous_card_id = self.selected_card_id
        previous_revision_id = self._rendered_revision_id
        previous_had_enrichment = self._had_enrichment
        preserve_description = self.description_edit.hasFocus()
        description_draft = self.description_edit.toPlainText()
        previous_mode = self._description_mode
        self._rendering = True
        try:
            card = next(
                (candidate for candidate in document.cards if candidate.id == selected_card_id),
                None,
            )
            self.selected_card_id = card.id if card is not None else None
            if card is None:
                self._rendered_revision_id = None
                self._had_enrichment = False
                self.pages.setCurrentIndex(0)
                self._set_error(self.description_error, "")
                self._set_error(self.reference_error, "")
                self.set_hotspot_error("")
                self.description_edit.clear()
                self.description_toggle.hide()
                self._enrichment_current = None
                self._update_enrich_button()
                self.hotspot_list.clear()
                self._render_hotspot_properties(document, None)
                return
            self.pages.setCurrentIndex(1)
            revision = card.active_revision
            same_revision = previous_card_id == card.id and previous_revision_id == revision.id
            if not same_revision:
                self._set_error(self.description_error, "")
                self._set_error(self.reference_error, "")
                self.set_hotspot_error("")
            self._rendered_revision_id = revision.id
            if revision.enriched_description is None:
                self._description_mode = "original"
            elif not same_revision or not previous_had_enrichment:
                self._description_mode = "enriched"
            else:
                self._description_mode = previous_mode
            self._had_enrichment = revision.enriched_description is not None
            displayed_value = (
                revision.enriched_description.text
                if (
                    self._description_mode == "enriched"
                    and revision.enriched_description is not None
                )
                else revision.description
            )
            self.description_edit.setPlainText(
                description_draft
                if preserve_description and same_revision
                else displayed_value
            )
            self._render_references(document, card, revision)
            self._render_description_workflow(document, card, revision)
            self._render_hotspots(document, revision)
        finally:
            self._rendering = False

    def commit_revision_metadata(self) -> bool:
        if self._rendering or self.selected_card_id is None:
            return False
        card = self._selected_card()
        if card is None:
            return False
        value = self.description_edit.toPlainText()
        if self._description_mode == "original":
            if value == card.active_revision.description:
                return True
            return self._execute(
                EditRevisionDescriptionCommand(
                    card_id=card.id,
                    revision_id=card.active_revision.id,
                    value=value,
                ),
                error_label=self.description_error,
            )
        existing = card.active_revision.enriched_description
        if existing is None or value.strip() == existing.text:
            return True
        if not value.strip():
            return self._execute(
                SetRevisionEnrichedDescriptionCommand(
                    card_id=card.id,
                    revision_id=card.active_revision.id,
                    value=None,
                ),
                error_label=self.description_error,
                undo_message="Enrichment removed",
            )
        return self._execute(
            SetRevisionEnrichedDescriptionCommand(
                card_id=card.id,
                revision_id=card.active_revision.id,
                value=existing.model_copy(update={"text": value.strip()}),
            ),
            error_label=self.description_error,
        )

    def commit_card_metadata(self) -> bool:
        """Compatibility alias while MainWindow is migrated."""
        return self.commit_revision_metadata()

    def has_render_prompt_input(self) -> bool:
        card = self._selected_card()
        if card is None:
            return False
        visible_value = self.description_edit.toPlainText().strip()
        if self._description_mode == "original":
            original = visible_value
            enriched = (
                card.active_revision.enriched_description.text
                if card.active_revision.enriched_description is not None
                else ""
            )
        else:
            original = card.active_revision.description
            enriched = visible_value
        return bool(original.strip() or enriched.strip())

    def has_description_input(self) -> bool:
        card = self._selected_card()
        if card is None:
            return False
        if self._description_mode == "original":
            return bool(self.description_edit.toPlainText().strip())
        return bool(card.active_revision.description.strip())

    def _switch_description_mode(self, mode: str) -> None:
        if self._rendering or mode == self._description_mode:
            return
        card = self._selected_card()
        if (
            card is None
            or (
                mode == "enriched"
                and card.active_revision.enriched_description is None
            )
            or not self.commit_revision_metadata()
        ):
            self.render(self.controller.document, self.selected_card_id)
            return
        self._description_mode = mode
        self.render(self.controller.document, self.selected_card_id)

    def _render_description_workflow(
        self,
        document: Stack,
        card: Card,
        revision: CardRevision,
    ) -> None:
        roles = [
            role.value.title()
            for role in ReferenceRole
            if getattr(revision, role.value) is not None
        ]
        reference_suffix = (
            f" + References ({', '.join(roles)})"
            if roles
            else ""
        )
        self.references_label.setText(
            f"References ({len(roles)})" if roles else "References"
        )
        self._enrich_using_text = f"Using: Description{reference_suffix}"
        enrichment = revision.enriched_description
        if enrichment is None:
            generation_sources = "Description"
            self._enrichment_current = None
        else:
            authored_description = (
                self.description_edit.toPlainText()
                if self._description_mode == "original"
                else revision.description
            )
            self._enrichment_current = enrichment.is_current(
                source_description=authored_description,
                references=enrichment_reference_snapshots(document, card),
            )
            generation_sources = "Enriched Description"
        self.description_toggle.setVisible(enrichment is not None)
        self.original_description_button.setChecked(
            self._description_mode == "original"
        )
        self.enriched_description_button.setChecked(
            self._description_mode == "enriched"
        )
        self.description_edit.setAccessibleName(
            "Enriched Description"
            if self._description_mode == "enriched"
            else "Description"
        )
        self.description_edit.setPlaceholderText(
            "Enriched Description"
            if self._description_mode == "enriched"
            else "Description"
        )
        self._generate_using_text = (
            f"Using: {generation_sources}{reference_suffix}"
        )
        self._update_enrich_button()
        self._refresh_generation_tooltips()

    def set_background_capabilities(
        self,
        *,
        can_generate: bool,
        generate_reason: str,
        has_image: bool,
        busy: bool,
        generating: bool,
    ) -> None:
        self._generate_reason = generate_reason
        self.generate_background_button.setEnabled(can_generate and not busy)
        self.generate_background_button.setText(
            "Generating…"
            if generating
            else ("Re-generate Image" if has_image else "Generate Image")
        )
        self._refresh_generation_tooltips()

    def set_scene_enrichment_capabilities(
        self,
        *,
        can_enrich: bool,
        reason: str,
        busy: bool,
    ) -> None:
        self._enrich_reason = reason
        self._can_enrich = can_enrich
        self._enrichment_busy = busy
        self._update_enrich_button()
        self._refresh_generation_tooltips()

    def _update_enrichment_freshness(self) -> None:
        card = self._selected_card()
        if card is None or card.active_revision.enriched_description is None:
            self._enrichment_current = None
        else:
            source_description = (
                self.description_edit.toPlainText()
                if self._description_mode == "original"
                else card.active_revision.description
            )
            self._enrichment_current = (
                card.active_revision.enriched_description.is_current(
                    source_description=source_description,
                    references=enrichment_reference_snapshots(
                        self.controller.document,
                        card,
                    ),
                )
            )
        self._update_enrich_button()
        self._refresh_generation_tooltips()

    def _update_enrich_button(self) -> None:
        card = self._selected_card()
        has_enrichment = (
            card is not None
            and card.active_revision.enriched_description is not None
        )
        if self._enrichment_busy:
            text = "Enriching…"
            enabled = False
        elif self._enrichment_current is True:
            text = "Description Enriched ✓"
            enabled = False
        else:
            text = (
                "Re-enrich Description"
                if has_enrichment
                else "Enrich Description"
            )
            enabled = self._can_enrich
        self.enrich_scene_button.setText(text)
        self.enrich_scene_button.setEnabled(enabled)
        self.enrich_scene_button.setAccessibleDescription(
            self._enrich_state_reason()
        )

    def _refresh_generation_tooltips(self) -> None:
        self.enrich_scene_button.setToolTip(
            self._tooltip_with_using(
                self._enrich_state_reason(),
                self._enrich_using_text,
            )
        )
        self.generate_background_button.setToolTip(
            self._tooltip_with_using(
                self._generate_reason,
                self._generate_using_text,
            )
        )

    def _enrich_state_reason(self) -> str:
        if self._enrichment_busy:
            return "Description enrichment is running"
        if self._enrichment_current is True:
            return "Current"
        if self._enrichment_current is False:
            return self._tooltip_with_using(
                "Out of date",
                self._enrich_reason,
            )
        return self._enrich_reason

    @staticmethod
    def _tooltip_with_using(reason: str, using: str) -> str:
        return "\n".join(part for part in (reason, using) if part)

    @property
    def selected_interaction_id(self) -> UUID | None:
        item = self.hotspot_list.currentItem()
        if item is None:
            return None
        value = item.data(Qt.ItemDataRole.UserRole)
        return value if isinstance(value, UUID) else None

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

    def reset_context(self) -> None:
        self.selected_card_id = None
        self._rendered_revision_id = None
        self._set_error(self.description_error, "")
        self._set_error(self.reference_error, "")
        self.set_hotspot_error("")

    def _render_references(
        self,
        document: Stack,
        card: Card,
        revision: CardRevision,
    ) -> None:
        for role, combo in self.reference_combos.items():
            reference = getattr(revision, role.value)
            with QSignalBlocker(combo):
                combo.clear()
                combo.addItem("No reference", None)
                for candidate in document.cards:
                    if candidate.id == card.id:
                        continue
                    combo.addItem(candidate.name, candidate.id)
                if isinstance(reference, ResolvedCardReference):
                    combo.setCurrentIndex(
                        self._combo_index_for_data(
                            combo,
                            reference.target_card_id,
                        )
                    )
                elif isinstance(reference, UnresolvedCardReference):
                    name = reference.target_name or "Unknown card"
                    combo.addItem(f"Missing: {name}", reference)
                    combo.setCurrentIndex(combo.count() - 1)
                else:
                    combo.setCurrentIndex(0)

    def _reference_changed(self, role: ReferenceRole, index: int) -> None:
        if self._rendering or index < 0:
            return
        card = self._selected_card()
        if card is None:
            return
        value = self.reference_combos[role].itemData(index)
        if isinstance(value, UUID):
            reference = ResolvedCardReference(target_card_id=value)
        elif isinstance(value, UnresolvedCardReference):
            reference = value
        else:
            reference = None
        if reference == getattr(card.active_revision, role.value):
            return
        self._execute(
            SetRevisionReferenceCommand(
                card_id=card.id,
                revision_id=card.active_revision.id,
                role=role,
                reference=reference,
            ),
            error_label=self.reference_error,
            undo_message=f"{role.value.replace('_', ' ').title()} reference changed",
        )

    def _render_hotspots(
        self,
        document: Stack,
        revision: CardRevision,
    ) -> None:
        desired_id = self.selected_interaction_id
        interactions = revision.hotspot_set.interactions if revision.hotspot_set is not None else ()
        with QSignalBlocker(self.hotspot_list):
            self.hotspot_list.clear()
            for interaction in interactions:
                item = QListWidgetItem(interaction.label)
                item.setData(Qt.ItemDataRole.UserRole, interaction.id)
                self.hotspot_list.addItem(item)
            selected_row = next(
                (
                    row
                    for row in range(self.hotspot_list.count())
                    if self.hotspot_list.item(row).data(Qt.ItemDataRole.UserRole) == desired_id
                ),
                0 if interactions else -1,
            )
            self.hotspot_list.setCurrentRow(selected_row)
        self.hotspots_placeholder.setVisible(not interactions)
        selected = interactions[selected_row] if selected_row >= 0 else None
        self._render_hotspot_properties(document, selected)
        self.inspector_tabs.setTabText(1, "Hotspots")

    def _render_hotspot_properties(
        self,
        document: Stack,
        interaction: Interaction | None,
    ) -> None:
        has_interaction = interaction is not None
        self.hotspot_destination_combo.setEnabled(has_interaction)
        self.delete_hotspot_button.setEnabled(has_interaction)
        current_row = self.hotspot_list.currentRow()
        self.move_hotspot_up_button.setEnabled(has_interaction and current_row > 0)
        self.move_hotspot_down_button.setEnabled(
            has_interaction and current_row < self.hotspot_list.count() - 1
        )
        with QSignalBlocker(self.hotspot_destination_combo):
            self.hotspot_destination_combo.clear()
            self.hotspot_destination_combo.addItem("Unresolved", None)
            for card in document.cards:
                self.hotspot_destination_combo.addItem(card.name, card.id)
            self.hotspot_destination_combo.addItem("Create New Card...", "create")
            if interaction is None:
                self.hotspot_destination_combo.setCurrentIndex(-1)
            elif isinstance(interaction.action.target, ResolvedCardReference):
                self.hotspot_destination_combo.setCurrentIndex(
                    self._combo_index_for_data(
                        self.hotspot_destination_combo,
                        interaction.action.target.target_card_id,
                    )
                )
            else:
                self.hotspot_destination_combo.setCurrentIndex(0)

    def _hotspot_selection_changed(
        self,
        current: QListWidgetItem | None,
        _previous: QListWidgetItem | None,
    ) -> None:
        if self._rendering:
            return
        interaction_id = current.data(Qt.ItemDataRole.UserRole) if current is not None else None
        self._render_hotspot_properties(
            self.controller.document,
            self._selected_interaction(),
        )
        self.hotspot_selected.emit(interaction_id)

    def _add_hotspot(self) -> None:
        card = self._selected_card()
        if card is None:
            return
        interaction = Interaction(
            action=NavigateAction(target=UnresolvedCardReference()),
        )
        if self._execute(
            AddInteractionCommand(
                card_id=card.id,
                revision_id=card.active_revision.id,
                interaction=interaction,
            )
        ):
            self.select_interaction(interaction.id)
            self.hotspot_selected.emit(interaction.id)

    def _destination_changed(self, index: int) -> None:
        if self._rendering or index < 0:
            return
        card = self._selected_card()
        interaction = self._selected_interaction()
        if card is None or interaction is None:
            return
        destination = self.hotspot_destination_combo.itemData(index)
        QTimer.singleShot(
            0,
            lambda: self._apply_destination_change(
                card.id,
                card.active_revision.id,
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
        if destination == "create":
            name, accepted = QInputDialog.getText(
                self,
                "Create Destination Card",
                "Card name",
            )
            if not accepted:
                self.render(self.controller.document, self.selected_card_id)
                return
            command: DocumentCommand = CreateCardAndResolveCommand(
                source_card_id=card_id,
                revision_id=revision_id,
                interaction_id=interaction_id,
                card_name=name,
            )
        else:
            command = ChangeHotspotDestinationCommand(
                card_id=card_id,
                revision_id=revision_id,
                interaction_id=interaction_id,
                destination=(
                    ResolvedCardReference(target_card_id=destination)
                    if isinstance(destination, UUID)
                    else UnresolvedCardReference()
                ),
            )
        self._execute(command)

    def _move_hotspot(self, offset: int) -> None:
        card = self._selected_card()
        interaction_id = self.selected_interaction_id
        destination = self.hotspot_list.currentRow() + offset
        if (
            card is None
            or interaction_id is None
            or not 0 <= destination < self.hotspot_list.count()
        ):
            return
        self._execute(
            ReorderHotspotCommand(
                card_id=card.id,
                revision_id=card.active_revision.id,
                interaction_id=interaction_id,
                new_index=destination,
            )
        )

    def _delete_hotspot(self) -> None:
        card = self._selected_card()
        interaction_id = self.selected_interaction_id
        if card is None or interaction_id is None:
            return
        self._execute(
            DeleteInteractionCommand(
                card_id=card.id,
                revision_id=card.active_revision.id,
                interaction_id=interaction_id,
            ),
            undo_message="Hotspot deleted",
        )

    def _execute(
        self,
        command: DocumentCommand,
        *,
        error_label: QLabel | None = None,
        undo_message: str | None = None,
    ) -> bool:
        target_error = error_label if error_label is not None else self.hotspot_error
        previous_token = self.controller.current_undo_token
        try:
            changed = self.controller.execute(command)
        except (CommandError, ValidationError) as error:
            self._set_error(target_error, str(error))
            self.render(self.controller.document, self.selected_card_id)
            return False
        self._set_error(target_error, "")
        self.render(changed, self.selected_card_id)
        self.document_changed.emit(changed)
        token = self.controller.current_undo_token
        if undo_message is not None and token is not None and token != previous_token:
            self.change_applied.emit(undo_message, token)
        return True

    @staticmethod
    def _set_error(label: QLabel, message: str) -> None:
        label.setText(message)
        label.setVisible(bool(message))

    def _selected_card(self) -> Card | None:
        return next(
            (card for card in self.controller.document.cards if card.id == self.selected_card_id),
            None,
        )

    def _selected_interaction(self) -> Interaction | None:
        card = self._selected_card()
        interaction_id = self.selected_interaction_id
        if card is None or interaction_id is None or card.active_revision.hotspot_set is None:
            return None
        return next(
            (
                interaction
                for interaction in card.active_revision.hotspot_set.interactions
                if interaction.id == interaction_id
            ),
            None,
        )

    @staticmethod
    def _combo_index_for_data(combo: QComboBox, value: object) -> int:
        return next(
            (index for index in range(combo.count()) if combo.itemData(index) == value),
            -1,
        )


__all__ = ["Inspector"]
