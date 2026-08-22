"""Scrollable selected-card inspector."""

from __future__ import annotations

import json
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
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
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

from hypergen.application.background_workflow import BackgroundDraft
from hypergen.application.commands import (
    ChangeHotspotDestinationCommand,
    CommandError,
    CreateCardAndResolveCommand,
    DeleteInteractionCommand,
    DocumentCommand,
    EditCardTextCommand,
    EditGlobalStyleCommand,
    RenameInteractionCommand,
    ReorderHotspotCommand,
)
from hypergen.application.document_controller import DocumentController
from hypergen.application.hotspot_generation_workflow import HotspotGenerationDraft
from hypergen.application.scene_enrichment_workflow import SceneEnrichmentDraft
from hypergen.domain.models import (
    Card,
    ImageRevision,
    Interaction,
    ResolvedCardReference,
    Stack,
    UnresolvedCardReference,
)
from hypergen.generation.hotspot_prompts import ExistingCandidateTarget


class _CommitPlainTextEdit(QPlainTextEdit):
    editing_finished = Signal()

    def focusOutEvent(self, event: QFocusEvent) -> None:
        super().focusOutEvent(event)
        self.editing_finished.emit()


def _section_heading(text: str) -> QLabel:
    label = QLabel(text)
    font = label.font()
    font.setBold(True)
    label.setFont(font)
    return label


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


class _DismissibleMessage(QFrame):
    def __init__(
        self,
        *,
        object_name: str,
        label_object_name: str,
        dismiss_object_name: str,
    ) -> None:
        super().__init__()
        self.setObjectName(object_name)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            f"""
            QFrame#{object_name} {{
                background-color: palette(alternate-base);
                border: 1px solid palette(highlight);
                border-radius: 4px;
            }}
            """
        )
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 6, 6)
        self.label = QLabel()
        self.label.setObjectName(label_object_name)
        self.label.setWordWrap(True)
        layout.addWidget(self.label, 1)
        self.dismiss_button = _compact_icon_button(
            self.style(),
            object_name=dismiss_object_name,
            icon=QStyle.StandardPixmap.SP_DialogCloseButton,
            accessible_name="Dismiss message",
            tooltip="Dismiss message",
        )
        self.dismiss_button.setAutoRaise(True)
        layout.addWidget(
            self.dismiss_button,
            0,
            Qt.AlignmentFlag.AlignTop,
        )
        self.dismiss_button.clicked.connect(self.dismiss)
        self.setVisible(False)

    def set_message(self, message: str, *, detail: str = "") -> None:
        self.label.setText(message)
        self.label.setToolTip(detail)
        self.setToolTip(detail)
        self.setVisible(bool(message))

    def dismiss(self) -> None:
        self.set_message("")


def _review_frame(object_name: str) -> QFrame:
    frame = QFrame()
    frame.setObjectName(object_name)
    frame.setAccessibleName("Review required")
    frame.setFrameShape(QFrame.Shape.StyledPanel)
    frame.setStyleSheet(
        f"""
        QFrame#{object_name} {{
            background-color: palette(alternate-base);
            border: 2px solid palette(highlight);
            border-radius: 4px;
        }}
        """
    )
    return frame


class Inspector(QWidget):
    """Render selected-card snapshots and issue typed metadata commands."""

    document_changed = Signal(object)
    generate_background_requested = Signal()
    import_background_requested = Signal()
    accept_background_draft_requested = Signal()
    discard_background_draft_requested = Signal()
    revision_activation_requested = Signal(object)
    revision_deletion_requested = Signal(object)
    hotspot_selected = Signal(object)
    add_hotspot_requested = Signal()
    add_hotspot_component_requested = Signal(object)
    render_inputs_changed = Signal()
    enrich_scene_requested = Signal()
    accept_scene_enrichment_requested = Signal()
    discard_scene_enrichment_requested = Signal()
    generate_hotspots_requested = Signal()
    apply_hotspot_candidate_requested = Signal()
    discard_hotspot_candidate_requested = Signal()
    candidate_label_changed = Signal(object, str)
    candidate_destination_changed = Signal(object, object)
    candidate_create_destination_requested = Signal(object, str)
    candidate_reorder_requested = Signal(object, int)
    candidate_delete_requested = Signal(object)
    summarize_hotspots_requested = Signal()

    def __init__(
        self,
        controller: DocumentController,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.controller = controller
        self.selected_card_id: UUID | None = None
        self._rendering = False
        self._has_background_draft = False
        self._scene_enrichment_identity: tuple[UUID, UUID, str, str] | None = None
        self._hotspot_candidate: HotspotGenerationDraft | None = None
        self._candidate_selected_interaction_id: UUID | None = None
        self._dismissed_warning_identity: (
            tuple[UUID, UUID, str, tuple[str, ...]] | None
        ) = None
        self.setObjectName("inspector")
        self.setMinimumWidth(300)

        self.inspector_tabs = QTabWidget()
        self.inspector_tabs.setObjectName("inspectorTabs")

        card_scroll = QScrollArea()
        card_scroll.setObjectName("cardInspectorTab")
        card_scroll.setWidgetResizable(True)
        card_scroll.setFrameShape(QFrame.Shape.NoFrame)
        card_page = QWidget()
        card_page.setObjectName("cardInspectorContent")
        card_layout = QVBoxLayout(card_page)
        self.card_layout = card_layout
        scene_heading = QHBoxLayout()
        self.scene_heading = _section_heading("Description")
        scene_heading.addWidget(self.scene_heading)
        scene_heading.addStretch(1)
        card_layout.addLayout(scene_heading)
        self.scene_edit = _CommitPlainTextEdit()
        self.scene_edit.setObjectName("sceneDescriptionEdit")
        self.scene_edit.setPlaceholderText("Describe the image to generate")
        self.scene_edit.setMaximumHeight(150)
        card_layout.addWidget(self.scene_edit)
        self.scene_actions = QHBoxLayout()
        self.enrich_scene_button = QPushButton("Enrich")
        self.enrich_scene_button.setObjectName("enrichSceneButton")
        self.enrich_scene_button.setEnabled(False)
        self.scene_actions.addWidget(self.enrich_scene_button)
        card_layout.addLayout(self.scene_actions)
        self.scene_enrichment_widget = QWidget()
        self.scene_enrichment_widget.setObjectName("sceneEnrichmentReview")
        enrichment_layout = QVBoxLayout(self.scene_enrichment_widget)
        enrichment_layout.setContentsMargins(0, 4, 0, 4)
        enrichment_layout.addWidget(_section_heading("Enriched Description"))
        self.enriched_scene_edit = QPlainTextEdit()
        self.enriched_scene_edit.setObjectName("enrichedSceneEdit")
        self.enriched_scene_edit.setMaximumHeight(150)
        enrichment_layout.addWidget(self.enriched_scene_edit)
        enrichment_actions = QHBoxLayout()
        self.accept_scene_enrichment_button = QPushButton("Accept")
        self.accept_scene_enrichment_button.setObjectName(
            "acceptSceneEnrichmentButton"
        )
        self.discard_scene_enrichment_button = QPushButton("Discard")
        self.discard_scene_enrichment_button.setObjectName(
            "discardSceneEnrichmentButton"
        )
        enrichment_actions.addWidget(self.accept_scene_enrichment_button)
        enrichment_actions.addWidget(self.discard_scene_enrichment_button)
        enrichment_layout.addLayout(enrichment_actions)
        card_layout.addWidget(self.scene_enrichment_widget)
        self.scene_enrichment_widget.setVisible(False)
        self.scene_enrichment_status_message = _DismissibleMessage(
            object_name="sceneEnrichmentStatusMessage",
            label_object_name="sceneEnrichmentStatus",
            dismiss_object_name="dismissSceneEnrichmentStatusButton",
        )
        self.scene_enrichment_status = self.scene_enrichment_status_message.label
        card_layout.addWidget(self.scene_enrichment_status_message)
        self.validation_error = QLabel()
        self.validation_error.setObjectName("inspectorValidationError")
        self.validation_error.setWordWrap(True)
        self.validation_error.setVisible(False)
        card_layout.addWidget(self.validation_error)

        card_layout.addSpacing(8)
        self.background_heading = _section_heading("Background")
        card_layout.addWidget(self.background_heading)
        self.revision_selector = QHBoxLayout()
        self.revision_selector.addWidget(QLabel("Revision"))
        self.revision_combo = QComboBox()
        self.revision_combo.setObjectName("backgroundRevisionCombo")
        self.revision_selector.addWidget(self.revision_combo, 1)
        self.delete_revision_button = _compact_icon_button(
            self.style(),
            object_name="deleteRevisionButton",
            icon=QStyle.StandardPixmap.SP_TrashIcon,
            accessible_name="Delete revision",
            tooltip="Delete revision",
        )
        self.revision_selector.addWidget(self.delete_revision_button)
        card_layout.addLayout(self.revision_selector)
        self.revision_details_button = QToolButton()
        self.revision_details_button.setObjectName("backgroundRevisionDetailsButton")
        self.revision_details_button.setText("Details ▸")
        self.revision_details_button.setCheckable(True)
        self.revision_details_button.setVisible(False)
        card_layout.addWidget(self.revision_details_button)
        self.revision_details = QLabel()
        self.revision_details.setObjectName("backgroundRevisionDetails")
        self.revision_details.setWordWrap(True)
        self.revision_details.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.revision_details.setVisible(False)
        card_layout.addWidget(self.revision_details)

        self.style_details_button = QToolButton()
        self.style_details_button.setObjectName("styleDetailsButton")
        self.style_details_button.setText("Style ▸")
        self.style_details_button.setCheckable(True)
        self.style_details = QWidget()
        self.style_details.setObjectName("styleDetails")
        style_layout = QVBoxLayout(self.style_details)
        style_layout.setContentsMargins(0, 0, 0, 0)
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
        style_layout.addLayout(style_modes)
        self.style_edit = _CommitPlainTextEdit()
        self.style_edit.setObjectName("styleEdit")
        self.style_edit.setMaximumHeight(100)
        self.style_edit.setPlaceholderText("Visual style for generated backgrounds")
        style_layout.addWidget(self.style_edit)
        self.style_scope_caption = QLabel()
        self.style_scope_caption.setObjectName("styleScopeCaption")
        self.style_scope_caption.setWordWrap(True)
        style_layout.addWidget(self.style_scope_caption)
        self.style_details.setVisible(False)

        self.generate_background_button = QPushButton("Generate Background")
        self.generate_background_button.setObjectName("generateBackgroundButton")
        card_layout.addWidget(self.generate_background_button)
        card_layout.addWidget(self.style_details_button)
        card_layout.addWidget(self.style_details)
        self.background_status_message = _DismissibleMessage(
            object_name="backgroundStatusMessage",
            label_object_name="backgroundStatus",
            dismiss_object_name="dismissBackgroundStatusButton",
        )
        self.background_status = self.background_status_message.label
        card_layout.addWidget(self.background_status_message)

        self.draft_widget = _review_frame("backgroundDraftReview")
        draft_layout = QVBoxLayout(self.draft_widget)
        draft_layout.setContentsMargins(10, 8, 10, 8)
        self.draft_badge = _section_heading("Review required")
        self.draft_badge.setObjectName("backgroundDraftBadge")
        draft_layout.addWidget(self.draft_badge)
        self.draft_value = QLabel()
        self.draft_value.setObjectName("backgroundDraftValue")
        self.draft_value.setWordWrap(True)
        draft_layout.addWidget(self.draft_value)
        draft_actions = QHBoxLayout()
        self.accept_draft_button = QPushButton("Accept as Revision")
        self.accept_draft_button.setObjectName("acceptBackgroundDraftButton")
        self.discard_draft_button = QPushButton("Discard")
        self.discard_draft_button.setObjectName("discardBackgroundDraftButton")
        draft_actions.addWidget(self.accept_draft_button)
        draft_actions.addWidget(self.discard_draft_button)
        draft_layout.addLayout(draft_actions)
        card_layout.addWidget(self.draft_widget)
        self.draft_widget.setVisible(False)

        self.import_background_button = QPushButton("Import Background...")
        self.import_background_button.setObjectName("importBackgroundButton")
        card_layout.addWidget(self.import_background_button)
        card_layout.addStretch(1)
        card_scroll.setWidget(card_page)
        self.inspector_tabs.addTab(card_scroll, "Background")

        interactivity_page = QWidget()
        interactivity_page.setObjectName("interactivityInspectorTab")
        hotspots_layout = QVBoxLayout(interactivity_page)
        self.intent_heading = _section_heading("Intent")
        hotspots_layout.addWidget(self.intent_heading)
        self.interactions_edit = _CommitPlainTextEdit()
        self.interactions_edit.setObjectName("interactionDescriptionEdit")
        self.interactions_edit.setPlaceholderText(
            "Describe what people can interact with and where it leads"
        )
        self.interactions_edit.setMaximumHeight(110)
        hotspots_layout.addWidget(self.interactions_edit)
        self.intent_actions = QHBoxLayout()
        self.summarize_hotspots_button = QPushButton("Summarize Hotspots")
        self.summarize_hotspots_button.setObjectName("summarizeHotspotsButton")
        self.summarize_hotspots_button.setEnabled(False)
        self.intent_actions.addWidget(self.summarize_hotspots_button)
        hotspots_layout.addLayout(self.intent_actions)
        self.intent_status_message = _DismissibleMessage(
            object_name="intentStatusMessage",
            label_object_name="intentStatus",
            dismiss_object_name="dismissIntentStatusButton",
        )
        self.intent_status = self.intent_status_message.label
        hotspots_layout.addWidget(self.intent_status_message)
        hotspots_layout.addSpacing(8)
        hotspots_heading = QHBoxLayout()
        self.hotspots_heading = _section_heading("Hotspots")
        hotspots_heading.addWidget(self.hotspots_heading)
        hotspots_heading.addStretch(1)
        self.generate_hotspots_button = QPushButton("Generate Hotspots")
        self.generate_hotspots_button.setObjectName("generateHotspotsButton")
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
        self.hotspot_generation_status_message = _DismissibleMessage(
            object_name="hotspotGenerationStatusMessage",
            label_object_name="hotspotGenerationStatus",
            dismiss_object_name="dismissHotspotGenerationStatusButton",
        )
        self.hotspot_generation_status = self.hotspot_generation_status_message.label
        self.hotspot_candidate_widget = _review_frame("hotspotCandidateReview")
        candidate_layout = QVBoxLayout(self.hotspot_candidate_widget)
        candidate_layout.setContentsMargins(10, 8, 10, 8)
        self.hotspot_candidate_heading = _section_heading("Review required")
        self.hotspot_candidate_heading.setObjectName("hotspotCandidateHeading")
        candidate_layout.addWidget(self.hotspot_candidate_heading)
        self.hotspot_candidate_label = QLabel()
        self.hotspot_candidate_label.setObjectName("hotspotCandidateLabel")
        self.hotspot_candidate_label.setWordWrap(True)
        candidate_layout.addWidget(self.hotspot_candidate_label)
        self.hotspot_candidate_warnings = QLabel()
        self.hotspot_candidate_warnings.setObjectName("hotspotCandidateWarnings")
        self.hotspot_candidate_warnings.setWordWrap(True)
        candidate_layout.addWidget(self.hotspot_candidate_warnings)
        self.dismiss_candidate_warnings_button = QPushButton("Dismiss Warnings")
        self.dismiss_candidate_warnings_button.setObjectName(
            "dismissHotspotCandidateWarningsButton"
        )
        candidate_layout.addWidget(self.dismiss_candidate_warnings_button)
        candidate_actions = QHBoxLayout()
        self.apply_hotspot_candidate_button = QPushButton("Apply Hotspots")
        self.apply_hotspot_candidate_button.setObjectName(
            "applyHotspotCandidateButton"
        )
        self.discard_hotspot_candidate_button = QPushButton("Discard")
        self.discard_hotspot_candidate_button.setObjectName(
            "discardHotspotCandidateButton"
        )
        candidate_actions.addWidget(self.apply_hotspot_candidate_button)
        candidate_actions.addWidget(self.discard_hotspot_candidate_button)
        candidate_layout.addLayout(candidate_actions)
        self.hotspot_candidate_widget.setVisible(False)
        self.hotspots_placeholder = QLabel(
            "Apply a background before adding hotspots."
        )
        self.hotspots_placeholder.setWordWrap(True)
        hotspots_layout.addWidget(self.hotspots_placeholder)
        self.hotspot_list = QListWidget()
        self.hotspot_list.setObjectName("hotspotList")
        hotspots_layout.addWidget(self.hotspot_list)
        self.hotspot_order_actions = QHBoxLayout()
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
        self.delete_hotspot_button = _compact_icon_button(
            self.style(),
            object_name="deleteHotspotButton",
            icon=QStyle.StandardPixmap.SP_TrashIcon,
            accessible_name="Delete hotspot",
            tooltip="Delete hotspot",
        )
        self.hotspot_order_actions.addWidget(self.move_hotspot_up_button)
        self.hotspot_order_actions.addWidget(self.move_hotspot_down_button)
        self.hotspot_order_actions.addStretch(1)
        self.hotspot_order_actions.addWidget(self.delete_hotspot_button)
        hotspots_layout.addLayout(self.hotspot_order_actions)
        self.hotspot_label_edit = QLineEdit()
        self.hotspot_label_edit.setObjectName("hotspotLabelEdit")
        self.hotspot_destination_combo = QComboBox()
        self.hotspot_destination_combo.setObjectName("hotspotDestinationCombo")
        self.hotspot_properties_frame = QFrame()
        self.hotspot_properties_frame.setObjectName("hotspotProperties")
        self.hotspot_properties_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.hotspot_properties_frame.setStyleSheet(
            """
            QFrame#hotspotProperties {
                border: 1px solid palette(mid);
                border-radius: 4px;
            }
            """
        )
        self.hotspot_properties_layout = QVBoxLayout(
            self.hotspot_properties_frame
        )
        self.hotspot_properties_layout.setContentsMargins(8, 8, 8, 8)
        self.hotspot_label = QLabel("Label")
        self.hotspot_label.setObjectName("hotspotLabel")
        self.hotspot_properties_layout.addWidget(self.hotspot_label)
        self.hotspot_properties_layout.addWidget(self.hotspot_label_edit)
        self.hotspot_destination_label = QLabel("Destination")
        self.hotspot_destination_label.setObjectName("hotspotDestinationLabel")
        self.hotspot_properties_layout.addWidget(self.hotspot_destination_label)
        self.hotspot_properties_layout.addWidget(self.hotspot_destination_combo)
        self.hotspot_areas_label = QLabel("Areas")
        self.hotspot_areas_label.setObjectName("hotspotAreasLabel")
        self.hotspot_properties_layout.addWidget(self.hotspot_areas_label)
        self.hotspot_area_controls = QWidget()
        area_controls_layout = QHBoxLayout(self.hotspot_area_controls)
        area_controls_layout.setContentsMargins(0, 0, 0, 0)
        self.hotspot_area_count = QLabel("0")
        self.hotspot_area_count.setObjectName("hotspotAreaCount")
        area_controls_layout.addWidget(self.hotspot_area_count)
        area_controls_layout.addStretch(1)
        self.add_component_button = QPushButton("Add Area")
        self.add_component_button.setObjectName("addHotspotComponentButton")
        area_controls_layout.addWidget(self.add_component_button)
        self.hotspot_properties_layout.addWidget(self.hotspot_area_controls)
        hotspots_layout.addWidget(self.hotspot_properties_frame)
        self.hotspot_error = QLabel()
        self.hotspot_error.setObjectName("hotspotValidationError")
        self.hotspot_error.setWordWrap(True)
        self.hotspot_error.setVisible(False)
        hotspots_layout.addWidget(self.hotspot_error)
        hotspots_layout.addWidget(self.generate_hotspots_button)
        hotspots_layout.addWidget(self.hotspot_generation_status_message)
        hotspots_layout.addWidget(self.hotspot_candidate_widget)
        self.add_hotspot_button = QPushButton("Draw Hotspot")
        self.add_hotspot_button.setObjectName("addHotspotButton")
        hotspots_layout.addWidget(self.add_hotspot_button)
        self.inspector_tabs.addTab(interactivity_page, "Hotspots")

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
        form_layout.addWidget(self.inspector_tabs, 1)
        self.pages.addWidget(form_page)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.pages)

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
        self.generate_background_button.clicked.connect(
            self.generate_background_requested
        )
        self.import_background_button.clicked.connect(
            self.import_background_requested
        )
        self.accept_draft_button.clicked.connect(
            self.accept_background_draft_requested
        )
        self.discard_draft_button.clicked.connect(
            self.discard_background_draft_requested
        )
        self.revision_combo.currentIndexChanged.connect(self._revision_selected)
        self.revision_details_button.toggled.connect(
            self._toggle_revision_details
        )
        self.style_details_button.toggled.connect(self._toggle_style_details)
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
        self.generate_hotspots_button.clicked.connect(
            self.generate_hotspots_requested
        )
        self.apply_hotspot_candidate_button.clicked.connect(
            self.apply_hotspot_candidate_requested
        )
        self.discard_hotspot_candidate_button.clicked.connect(
            self.discard_hotspot_candidate_requested
        )
        self.dismiss_candidate_warnings_button.clicked.connect(
            self._dismiss_candidate_warnings
        )
        self.enrich_scene_button.clicked.connect(self.enrich_scene_requested)
        self.summarize_hotspots_button.clicked.connect(
            self.summarize_hotspots_requested
        )
        self.accept_scene_enrichment_button.clicked.connect(
            self.accept_scene_enrichment_requested
        )
        self.discard_scene_enrichment_button.clicked.connect(
            self.discard_scene_enrichment_requested
        )
        self.render(controller.document, None)

    def _render_input_edited(self) -> None:
        if not self._rendering:
            self.render_inputs_changed.emit()

    def has_render_prompt_input(self) -> bool:
        """Return whether the visible Description or effective Style can render."""
        return bool(
            self.selected_card_id is not None
            and (
                self.scene_edit.toPlainText().strip()
                or self.style_edit.toPlainText().strip()
            )
        )

    def has_description_input(self) -> bool:
        """Return whether the visible Description can be enriched."""
        return bool(
            self.selected_card_id is not None
            and self.scene_edit.toPlainText().strip()
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
                self.scene_edit.clear()
                self.interactions_edit.clear()
                self.style_edit.clear()
                self.revision_combo.clear()
                self.revision_details.clear()
                self.hotspots_placeholder.setText("Select a card to view hotspots.")
                self.hotspots_placeholder.setVisible(True)
                self.hotspot_list.clear()
                self._render_hotspot_properties(document, None, None)
                self._update_tab_labels(0)
                return
            self.pages.setCurrentIndex(1)
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
        self.background_status_message.set_message(message, detail=detail)

    def set_validation_error(self, message: str) -> None:
        self.validation_error.setText(message)
        self.validation_error.setVisible(bool(message))

    def set_scene_enrichment_capabilities(
        self,
        *,
        can_enrich: bool,
        reason: str,
    ) -> None:
        self.enrich_scene_button.setEnabled(can_enrich)
        self.enrich_scene_button.setToolTip(reason)

    def set_hotspot_summary_capabilities(
        self,
        *,
        can_summarize: bool,
        reason: str,
    ) -> None:
        self.summarize_hotspots_button.setEnabled(can_summarize)
        self.summarize_hotspots_button.setToolTip(reason)

    def set_intent_status(self, message: str, *, detail: str = "") -> None:
        self.intent_status_message.set_message(message, detail=detail)

    def show_scene_enrichment(
        self,
        draft: SceneEnrichmentDraft | None,
    ) -> None:
        self.scene_enrichment_widget.setVisible(draft is not None)
        if draft is None:
            self._scene_enrichment_identity = None
            self.enriched_scene_edit.clear()
            return
        self.scene_enrichment_status_message.dismiss()
        identity = (
            draft.stack_id,
            draft.card_id,
            draft.enriched_scene,
            draft.prompt_version,
        )
        if identity != self._scene_enrichment_identity:
            self.enriched_scene_edit.setPlainText(draft.enriched_scene)
            self._scene_enrichment_identity = identity
        self.enriched_scene_edit.setToolTip(
            f"Generated by {draft.model_identifier} ({draft.prompt_version})"
        )

    def set_scene_enrichment_status(
        self,
        message: str,
        *,
        detail: str = "",
    ) -> None:
        self.scene_enrichment_status_message.set_message(message, detail=detail)

    def set_hotspot_generation_capabilities(
        self,
        *,
        can_generate: bool,
        reason: str,
    ) -> None:
        self.generate_hotspots_button.setEnabled(can_generate)
        self.generate_hotspots_button.setToolTip(reason)

    def set_hotspot_generation_status(
        self,
        message: str,
        *,
        detail: str = "",
    ) -> None:
        self.hotspot_generation_status_message.set_message(message, detail=detail)

    def show_hotspot_candidate(
        self,
        candidate: HotspotGenerationDraft | None,
    ) -> None:
        self._hotspot_candidate = candidate
        self.hotspot_candidate_widget.setVisible(candidate is not None)
        if candidate is None:
            self._candidate_selected_interaction_id = None
            self.hotspot_candidate_warnings.clear()
            self._update_tab_labels(self.hotspot_list.count())
            return
        self.hotspot_generation_status_message.dismiss()
        revision = self._active_revision_for_selected_card()
        has_applied_set = revision is not None and revision.hotspot_set is not None
        self.apply_hotspot_candidate_button.setText(
            "Replace Hotspots" if has_applied_set else "Apply Hotspots"
        )
        warning_identity = (
            candidate.card_id,
            candidate.revision_id,
            candidate.raw_response,
            candidate.warnings,
        )
        warnings_dismissed = warning_identity == self._dismissed_warning_identity
        self.hotspot_candidate_warnings.setText(
            "" if warnings_dismissed else "\n\n".join(candidate.warnings)
        )
        self.hotspot_candidate_warnings.setVisible(
            bool(candidate.warnings) and not warnings_dismissed
        )
        self.dismiss_candidate_warnings_button.setVisible(
            bool(candidate.warnings) and not warnings_dismissed
        )
        interactions = candidate.hotspot_set.interactions
        self.hotspot_candidate_label.setText(
            f"{len(interactions)} generated "
            f"{'hotspot is' if len(interactions) == 1 else 'hotspots are'} "
            "ready to review."
        )
        desired_id = self._candidate_selected_interaction_id
        with QSignalBlocker(self.hotspot_list):
            self.hotspot_list.clear()
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
                    == desired_id
                ),
                0 if interactions else -1,
            )
            self.hotspot_list.setCurrentRow(selected_row)
        selected = interactions[selected_row] if selected_row >= 0 else None
        self._candidate_selected_interaction_id = (
            selected.id if selected is not None else None
        )
        self.hotspots_placeholder.setText("Candidate contains no hotspots.")
        self.hotspots_placeholder.setVisible(not interactions)
        self._render_hotspot_properties(
            self.controller.document,
            revision,
            selected,
        )
        self._update_tab_labels(len(interactions))

    def _dismiss_candidate_warnings(self) -> None:
        candidate = self._hotspot_candidate
        if candidate is None:
            return
        self._dismissed_warning_identity = (
            candidate.card_id,
            candidate.revision_id,
            candidate.raw_response,
            candidate.warnings,
        )
        self.hotspot_candidate_warnings.setVisible(False)
        self.dismiss_candidate_warnings_button.setVisible(False)

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

    def show_background_draft(
        self,
        draft: BackgroundDraft | None,
    ) -> None:
        self._has_background_draft = draft is not None
        self.draft_widget.setVisible(draft is not None)
        self._update_card_tab_label()
        if draft is None:
            self.draft_value.clear()
            return
        self.background_status_message.dismiss()
        origin = "Generated" if draft.origin.value == "generated" else "Imported"
        draft_name = draft.name or origin
        detail = (
            draft.generation_metadata.render_prompt
            if draft.generation_metadata is not None
            else draft.source_filename or ""
        )
        self.draft_value.setText(f"{draft_name} background is ready to review.")
        self.draft_value.setToolTip(detail)

    def _render_revisions(self, document: Stack, card: Card) -> None:
        active_revision = self._active_revision(card)
        with QSignalBlocker(self.revision_combo):
            self.revision_combo.clear()
            for index, revision in enumerate(card.image_revisions, start=1):
                origin = "Generated" if revision.origin.value == "generated" else "Imported"
                self.revision_combo.addItem(
                    f"{index}. {revision.name or origin}",
                    revision.id,
                )
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
            self.revision_details.clear()
            self.revision_details_button.setVisible(False)
            self.revision_details_button.setChecked(False)
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
        if revision.generation_metadata is not None:
            metadata = revision.generation_metadata
            self.revision_details.setText(
                json.dumps(
                    metadata.model_dump(mode="json"),
                    indent=2,
                    ensure_ascii=False,
                )
            )
            self.revision_details_button.setVisible(True)
        else:
            self.revision_details.clear()
            self.revision_details_button.setChecked(False)
            self.revision_details_button.setVisible(False)
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

    def _toggle_style_details(self, visible: bool) -> None:
        self.style_details.setVisible(visible)
        self.style_details_button.setText("Style ▾" if visible else "Style ▸")

    def _update_tab_labels(self, hotspot_count: int) -> None:
        self._update_card_tab_label()
        review_suffix = " • Review" if self._hotspot_candidate is not None else ""
        self.inspector_tabs.setTabText(
            1,
            f"Hotspots ({hotspot_count}){review_suffix}",
        )

    def _update_card_tab_label(self) -> None:
        self.inspector_tabs.setTabText(
            0,
            "Background • Review" if self._has_background_draft else "Background",
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
        if self._hotspot_candidate is not None:
            self._candidate_selected_interaction_id = (
                interaction_id if isinstance(interaction_id, UUID) else None
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
        self.hotspot_area_count.setText(
            str(len(interaction.polygons)) if interaction is not None else "0"
        )
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
            elif self._hotspot_candidate is not None:
                target = self._hotspot_candidate.targets_by_interaction_id.get(
                    interaction.id
                )
                if isinstance(target, ExistingCandidateTarget):
                    target_card_id = self._hotspot_candidate.card_ids_by_token.get(
                        target.card_token
                    )
                    self.hotspot_destination_combo.setCurrentIndex(
                        self._combo_index_for_data(
                            self.hotspot_destination_combo,
                            target_card_id,
                        )
                    )
                else:
                    self.hotspot_destination_combo.setCurrentIndex(0)
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
        if self._hotspot_candidate is not None:
            self.candidate_label_changed.emit(
                interaction.id,
                self.hotspot_label_edit.text(),
            )
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
                if self._hotspot_candidate is not None:
                    self.show_hotspot_candidate(self._hotspot_candidate)
                else:
                    self.render(self.controller.document, self.selected_card_id)
                return
            if self._hotspot_candidate is not None:
                self.candidate_create_destination_requested.emit(
                    interaction_id,
                    name,
                )
                return
            command = CreateCardAndResolveCommand(
                source_card_id=card_id,
                revision_id=revision_id,
                interaction_id=interaction_id,
                card_name=name,
            )
        else:
            if self._hotspot_candidate is not None:
                self.candidate_destination_changed.emit(
                    interaction_id,
                    destination if isinstance(destination, UUID) else None,
                )
                return
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
        if self._hotspot_candidate is not None:
            self.candidate_reorder_requested.emit(interaction_id, destination)
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
        if self._hotspot_candidate is not None:
            self.candidate_delete_requested.emit(interaction_id)
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
        if self._hotspot_candidate is not None:
            return next(
                (
                    interaction
                    for interaction in self._hotspot_candidate.hotspot_set.interactions
                    if interaction.id == interaction_id
                ),
                None,
            )
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
        if (
            self._hotspot_candidate is not None
            and self._hotspot_candidate.card_id == card_id
            and self._hotspot_candidate.revision_id == revision_id
        ):
            return next(
                (
                    interaction
                    for interaction in self._hotspot_candidate.hotspot_set.interactions
                    if interaction.id == interaction_id
                ),
                None,
            )
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
