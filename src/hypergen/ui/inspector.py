"""Minimal revision-local Background and Hotspots inspector."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import ValidationError
from PySide6.QtCore import QSignalBlocker, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QFocusEvent
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QStackedWidget,
    QStyle,
    QTableWidget,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from hypergen.application.commands import (
    AddInteractionCommand,
    AddKeyCommand,
    AddStyleCommand,
    ChangeHotspotDestinationCommand,
    CommandError,
    CreateCardAndResolveCommand,
    CreateKeyAndAddHotspotReferenceCommand,
    DeleteInteractionCommand,
    DeleteKeyCommand,
    DeleteStyleCommand,
    DocumentCommand,
    EditRevisionDescriptionCommand,
    RenameKeyCommand,
    ReorderHotspotCommand,
    SetHotspotConditionsCommand,
    SetHotspotKeyChangesCommand,
    SetHotspotNameCommand,
    SetRevisionImagePromptCommand,
    SetRevisionReferenceCommand,
    SetRevisionStyleCommand,
    UpdateStyleCommand,
)
from hypergen.application.document_controller import DocumentController
from hypergen.application.image_prompt_workflow import (
    image_prompt_reference_snapshot,
)
from hypergen.domain.models import (
    Card,
    CardRevision,
    HotspotConditions,
    HotspotKeyChanges,
    Interaction,
    KeyDefinition,
    ResolvedCardReference,
    Stack,
    StyleDefinition,
    UnresolvedCardReference,
)


class _CommitPlainTextEdit(QPlainTextEdit):
    editing_finished = Signal(object, object)

    def focusOutEvent(self, event: QFocusEvent) -> None:
        super().focusOutEvent(event)
        self.editing_finished.emit(QApplication.focusWidget(), event.reason())


class _CommitLineEdit(QLineEdit):
    editing_finished = Signal(object, object)

    def focusOutEvent(self, event: QFocusEvent) -> None:
        super().focusOutEvent(event)
        self.editing_finished.emit(QApplication.focusWidget(), event.reason())


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
    prepare_image_prompt_requested = Signal()
    hotspot_selected = Signal(object)
    hotspot_usage_requested = Signal(object, object, object)
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
        self._selected_style_id: UUID | None = None
        self._rendered_style_id: UUID | None = None
        self._selected_key_id: UUID | None = None
        self._rendered_key_id: UUID | None = None
        self._description_mode = "description"
        self._enrich_using_text = ""
        self._generate_using_text = ""
        self._enrich_reason = ""
        self._generate_reason = ""
        self._can_enrich = False
        self._image_prompt_busy = False
        self._image_prompt_current: bool | None = None
        self._image_prompt_model_identifier: str | None = None
        self._image_prompt_prompt_version: str | None = None
        self._rendering = False
        self.setObjectName("inspector")
        self.setMinimumWidth(300)

        root = QVBoxLayout(self)
        horizontal_margin = self.style().pixelMetric(QStyle.PixelMetric.PM_LayoutLeftMargin)
        root.setContentsMargins(horizontal_margin, 16, horizontal_margin, 16)

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
        self._build_styles_tab()
        self._build_hotspots_tab()
        self._build_keys_tab()
        self._connect_signals()

    def _build_background_tab(self) -> None:
        scroll = QScrollArea()
        scroll.setObjectName("backgroundInspectorTab")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        page = QWidget()
        page.setObjectName("backgroundInspectorContent")
        layout = QVBoxLayout(page)

        self.description_label = QLabel("Description")
        self.description_label.setObjectName("descriptionLabel")
        layout.addWidget(self.description_label)
        self.description_edit = _CommitPlainTextEdit()
        self.description_edit.setObjectName("descriptionEdit")
        self.description_edit.setPlaceholderText("Description")
        self.description_edit.setAccessibleName("Description")
        editor_height = round((self.description_edit.fontMetrics().lineSpacing() * 10 + 20) * 1.25)
        self.description_edit.setMinimumHeight(editor_height)
        layout.addWidget(self.description_edit, 1)
        self.description_error = QLabel()
        self.description_error.setObjectName("descriptionValidationError")
        self.description_error.setWordWrap(True)
        self.description_error.setVisible(False)
        layout.addWidget(self.description_error)

        self.description_toggle = QWidget()
        self.description_toggle.setObjectName("descriptionToggle")
        toggle_layout = QHBoxLayout(self.description_toggle)
        toggle_layout.setContentsMargins(0, 0, 0, 0)
        self.description_button = QRadioButton("Description")
        self.description_button.setObjectName("descriptionButton")
        self.description_button.setToolTip("Display and edit the authored Description")
        self.image_prompt_button = QRadioButton("Image Prompt")
        self.image_prompt_button.setObjectName("imagePromptButton")
        self.image_prompt_button.setToolTip("Display and edit the prepared Image Prompt")
        self.description_button_group = QButtonGroup(self)
        self.description_button_group.setExclusive(True)
        self.description_button_group.addButton(self.description_button)
        self.description_button_group.addButton(self.image_prompt_button)
        toggle_layout.addWidget(self.description_button)
        toggle_layout.addWidget(self.image_prompt_button)
        toggle_layout.addStretch(1)
        layout.addWidget(self.description_toggle)

        layout.addSpacing(8)
        self.style_label = QLabel("Style")
        self.style_label.setObjectName("styleLabel")
        layout.addWidget(self.style_label)
        self.style_combo = QComboBox()
        self.style_combo.setObjectName("styleCombo")
        self.style_combo.setAccessibleName("Style")
        self.style_combo.setToolTip(
            "Rendering treatment appended to the Image Prompt during generation"
        )
        layout.addWidget(self.style_combo)

        layout.addSpacing(8)
        self.reference_label = QLabel("Reference")
        self.reference_label.setObjectName("referenceLabel")
        layout.addWidget(self.reference_label)
        self.reference_panel = QWidget()
        self.reference_panel.setObjectName("referencePanel")
        reference_layout = QVBoxLayout(self.reference_panel)
        reference_layout.setContentsMargins(0, 0, 0, 0)
        self.reference_combo = QComboBox()
        self.reference_combo.setObjectName("referenceCombo")
        self.reference_combo.setAccessibleName("Reference card")
        self.reference_combo.setToolTip(
            "Optional image whose relevant visible characteristics can inform the Image Prompt"
        )
        reference_layout.addWidget(self.reference_combo)
        self.reference_error = QLabel()
        self.reference_error.setObjectName("referenceValidationError")
        self.reference_error.setWordWrap(True)
        self.reference_error.setVisible(False)
        reference_layout.addWidget(self.reference_error)
        layout.addWidget(self.reference_panel)

        layout.addSpacing(8)
        self.enrich_button = QPushButton("Prepare Image Prompt")
        self.enrich_button.setObjectName("enrichButton")
        layout.addWidget(self.enrich_button)

        layout.addSpacing(8)
        self.generate_background_button = QPushButton("Generate Image")
        self.generate_background_button.setObjectName("generateBackgroundButton")
        layout.addWidget(self.generate_background_button)
        self._focus_commit_targets = {
            self.description_button,
            self.image_prompt_button,
            self.enrich_button,
            self.generate_background_button,
        }

        scroll.setWidget(page)
        self.inspector_tabs.addTab(scroll, "Image")

    def _build_styles_tab(self) -> None:
        page = QWidget()
        page.setObjectName("stylesInspectorTab")
        layout = QVBoxLayout(page)

        self.styles_placeholder = QLabel("No Styles yet.")
        self.styles_placeholder.setWordWrap(True)
        layout.addWidget(self.styles_placeholder)
        self.style_list = QListWidget()
        self.style_list.setObjectName("styleList")
        self.style_list.setAccessibleName("Stack Styles")
        layout.addWidget(self.style_list, 1)

        controls = QHBoxLayout()
        controls.addStretch(1)
        self.delete_style_button = _compact_text_button(
            "−",
            object_name="deleteStyleButton",
            accessible_name="Delete Style",
            tooltip="Delete Style",
        )
        self.add_style_button = _compact_text_button(
            "+",
            object_name="addStyleButton",
            accessible_name="Add Style",
            tooltip="Add Style",
        )
        style_control_extent = max(
            self.delete_style_button.sizeHint().width(),
            self.delete_style_button.sizeHint().height(),
            self.add_style_button.sizeHint().width(),
            self.add_style_button.sizeHint().height(),
        )
        self.delete_style_button.setFixedSize(
            style_control_extent,
            style_control_extent,
        )
        self.add_style_button.setFixedSize(
            style_control_extent,
            style_control_extent,
        )
        controls.addWidget(self.delete_style_button)
        controls.addWidget(self.add_style_button)
        layout.addLayout(controls)

        self.style_name_label = QLabel("Name")
        self.style_name_label.setObjectName("styleNameLabel")
        layout.addWidget(self.style_name_label)
        self.style_name_edit = _CommitLineEdit()
        self.style_name_edit.setObjectName("styleNameEdit")
        self.style_name_edit.setAccessibleName("Style name")
        layout.addWidget(self.style_name_edit)

        self.style_prompt_label = QLabel("Style Text")
        self.style_prompt_label.setObjectName("stylePromptLabel")
        layout.addWidget(self.style_prompt_label)
        self.style_prompt_edit = _CommitPlainTextEdit()
        self.style_prompt_edit.setObjectName("stylePromptEdit")
        self.style_prompt_edit.setAccessibleName("Style text")
        self.style_prompt_edit.setPlaceholderText(
            "Describe the visual treatment appended during generation"
        )
        self.style_prompt_edit.setMinimumHeight(
            self.style_prompt_edit.fontMetrics().lineSpacing() * 7 + 20
        )
        layout.addWidget(self.style_prompt_edit)

        self.style_error = QLabel()
        self.style_error.setObjectName("styleValidationError")
        self.style_error.setWordWrap(True)
        self.style_error.setVisible(False)
        layout.addWidget(self.style_error)
        self._styles_tab_index = self.inspector_tabs.addTab(page, "Styles")

    def _build_hotspots_tab(self) -> None:
        page = QWidget()
        page.setObjectName("hotspotsInspectorTab")
        layout = QVBoxLayout(page)

        self.hotspots_placeholder = QLabel("No hotspots yet.")
        self.hotspots_placeholder.setWordWrap(True)
        layout.addWidget(self.hotspots_placeholder)
        self.hotspot_list = QListWidget()
        self.hotspot_list.setObjectName("hotspotList")
        self.hotspot_list.setWordWrap(True)
        layout.addWidget(self.hotspot_list, 1)

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
        controls.addWidget(self.delete_hotspot_button)
        controls.addWidget(self.add_hotspot_button)
        layout.addLayout(controls)

        self.hotspot_name_label = QLabel("Name")
        self.hotspot_name_label.setObjectName("hotspotNameLabel")
        layout.addWidget(self.hotspot_name_label)
        self.hotspot_name_edit = _CommitLineEdit()
        self.hotspot_name_edit.setObjectName("hotspotNameEdit")
        self.hotspot_name_edit.setAccessibleName("Hotspot name")
        self.hotspot_name_edit.setPlaceholderText("Leave blank to name automatically")
        layout.addWidget(self.hotspot_name_edit)

        self.hotspot_when_label = QLabel("WHEN")
        self.hotspot_when_label.setObjectName("hotspotWhenLabel")
        layout.addWidget(self.hotspot_when_label)
        self.condition_table = QTableWidget(0, 3)
        self.condition_table.setObjectName("hotspotConditionTable")
        self.condition_table.setHorizontalHeaderLabels(("Key", "State", ""))
        self.condition_table.verticalHeader().setVisible(False)
        self.condition_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self.condition_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        self.condition_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.ResizeToContents
        )
        self.condition_table.setMinimumHeight(76)
        self.condition_table.setMaximumHeight(132)
        layout.addWidget(self.condition_table)
        self.add_condition_button = QPushButton("Add Condition")
        self.add_condition_button.setObjectName("addConditionButton")
        layout.addWidget(self.add_condition_button)

        self.hotspot_then_label = QLabel("THEN")
        self.hotspot_then_label.setObjectName("hotspotThenLabel")
        layout.addWidget(self.hotspot_then_label)
        self.key_change_table = QTableWidget(0, 3)
        self.key_change_table.setObjectName("hotspotKeyChangeTable")
        self.key_change_table.setHorizontalHeaderLabels(("Change", "Key", ""))
        self.key_change_table.verticalHeader().setVisible(False)
        self.key_change_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self.key_change_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self.key_change_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.ResizeToContents
        )
        self.key_change_table.setMinimumHeight(76)
        self.key_change_table.setMaximumHeight(132)
        layout.addWidget(self.key_change_table)
        self.add_key_change_button = QPushButton("Add Key Change")
        self.add_key_change_button.setObjectName("addKeyChangeButton")
        layout.addWidget(self.add_key_change_button)
        self.clear_all_keys_checkbox = QCheckBox("Clear all keys")
        self.clear_all_keys_checkbox.setObjectName("clearAllKeysCheckbox")
        layout.addWidget(self.clear_all_keys_checkbox)

        self.hotspot_target_label = QLabel("Go to")
        self.hotspot_target_label.setObjectName("hotspotTargetLabel")
        layout.addWidget(self.hotspot_target_label)
        self.hotspot_destination_combo = QComboBox()
        self.hotspot_destination_combo.setObjectName("hotspotDestinationCombo")
        self.hotspot_destination_combo.setAccessibleName("Hotspot destination")
        layout.addWidget(self.hotspot_destination_combo)

        self.hotspot_summary = QLabel()
        self.hotspot_summary.setObjectName("hotspotSummary")
        self.hotspot_summary.setWordWrap(True)
        layout.addWidget(self.hotspot_summary)

        self.hotspot_error = QLabel()
        self.hotspot_error.setObjectName("hotspotValidationError")
        self.hotspot_error.setWordWrap(True)
        self.hotspot_error.setVisible(False)
        layout.addWidget(self.hotspot_error)
        self._hotspots_tab_index = self.inspector_tabs.addTab(page, "Hotspots")

    def _build_keys_tab(self) -> None:
        page = QWidget()
        page.setObjectName("keysInspectorTab")
        layout = QVBoxLayout(page)

        description = QLabel("Keys are global to this stack.")
        description.setObjectName("keysDescription")
        description.setWordWrap(True)
        layout.addWidget(description)

        self.keys_placeholder = QLabel("No Keys yet.")
        self.keys_placeholder.setWordWrap(True)
        layout.addWidget(self.keys_placeholder)
        self.key_list = QListWidget()
        self.key_list.setObjectName("keyList")
        self.key_list.setAccessibleName("Stack Keys")
        layout.addWidget(self.key_list, 1)

        controls = QHBoxLayout()
        controls.addStretch(1)
        self.delete_key_button = _compact_text_button(
            "−",
            object_name="deleteKeyButton",
            accessible_name="Delete Key",
            tooltip="Delete unused Key",
        )
        self.add_key_button = _compact_text_button(
            "+",
            object_name="addKeyButton",
            accessible_name="Add Key",
            tooltip="Add Key",
        )
        control_extent = max(
            self.delete_key_button.sizeHint().width(),
            self.delete_key_button.sizeHint().height(),
            self.add_key_button.sizeHint().width(),
            self.add_key_button.sizeHint().height(),
        )
        self.delete_key_button.setFixedSize(control_extent, control_extent)
        self.add_key_button.setFixedSize(control_extent, control_extent)
        controls.addWidget(self.delete_key_button)
        controls.addWidget(self.add_key_button)
        layout.addLayout(controls)

        self.key_name_label = QLabel("Name")
        self.key_name_label.setObjectName("keyNameLabel")
        layout.addWidget(self.key_name_label)
        self.key_name_edit = _CommitLineEdit()
        self.key_name_edit.setObjectName("keyNameEdit")
        self.key_name_edit.setAccessibleName("Key name")
        layout.addWidget(self.key_name_edit)

        self.key_usage_label = QLabel("Used By")
        self.key_usage_label.setObjectName("keyUsageLabel")
        layout.addWidget(self.key_usage_label)
        self.key_usage_list = QListWidget()
        self.key_usage_list.setObjectName("keyUsageList")
        self.key_usage_list.setAccessibleName("Key usages")
        layout.addWidget(self.key_usage_list, 1)
        self.show_hotspot_usage_button = QPushButton("Show Hotspot")
        self.show_hotspot_usage_button.setObjectName("showHotspotUsageButton")
        layout.addWidget(self.show_hotspot_usage_button)

        self.key_error = QLabel()
        self.key_error.setObjectName("keyValidationError")
        self.key_error.setWordWrap(True)
        self.key_error.setVisible(False)
        layout.addWidget(self.key_error)
        self._keys_tab_index = self.inspector_tabs.addTab(page, "Keys")

    @property
    def hotspots_active(self) -> bool:
        return self.inspector_tabs.currentIndex() == self._hotspots_tab_index

    def _connect_signals(self) -> None:
        self.description_edit.editing_finished.connect(self._description_editing_finished)
        self.description_edit.textChanged.connect(self._render_inputs_changed)
        self.description_button.clicked.connect(
            lambda: self._switch_description_mode("description")
        )
        self.image_prompt_button.clicked.connect(
            lambda: self._switch_description_mode("image_prompt")
        )
        self.enrich_button.clicked.connect(self.prepare_image_prompt_requested)
        self.generate_background_button.clicked.connect(self.generate_background_requested)
        self.style_combo.currentIndexChanged.connect(self._revision_style_changed)
        self.reference_combo.currentIndexChanged.connect(self._reference_changed)
        self.style_list.currentItemChanged.connect(self._style_selection_changed)
        self.add_style_button.clicked.connect(self._add_style)
        self.delete_style_button.clicked.connect(self._delete_style)
        self.style_name_edit.editing_finished.connect(self._style_editing_finished)
        self.style_prompt_edit.editing_finished.connect(self._style_editing_finished)
        self.hotspot_list.currentItemChanged.connect(self._hotspot_selection_changed)
        self.move_hotspot_up_button.clicked.connect(lambda: self._move_hotspot(-1))
        self.move_hotspot_down_button.clicked.connect(lambda: self._move_hotspot(1))
        self.add_hotspot_button.clicked.connect(self._add_hotspot)
        self.delete_hotspot_button.clicked.connect(self._delete_hotspot)
        self.hotspot_name_edit.editing_finished.connect(
            self._hotspot_name_editing_finished
        )
        self.hotspot_name_edit.returnPressed.connect(self._commit_hotspot_name)
        self.add_condition_button.clicked.connect(self._add_condition)
        self.add_key_change_button.clicked.connect(self._add_key_change)
        self.clear_all_keys_checkbox.toggled.connect(
            self._clear_all_keys_toggled
        )
        self.hotspot_destination_combo.currentIndexChanged.connect(self._destination_changed)
        self.key_list.currentItemChanged.connect(self._key_selection_changed)
        self.add_key_button.clicked.connect(self._add_key)
        self.delete_key_button.clicked.connect(self._delete_key)
        self.key_name_edit.editing_finished.connect(
            self._key_name_editing_finished
        )
        self.key_name_edit.returnPressed.connect(self._commit_key)
        self.key_usage_list.currentItemChanged.connect(
            lambda current, _previous: self.show_hotspot_usage_button.setEnabled(
                current is not None
            )
        )
        self.show_hotspot_usage_button.clicked.connect(self._show_hotspot_usage)

    def _render_inputs_changed(self) -> None:
        if not self._rendering:
            self._update_image_prompt_freshness()
            self.render_inputs_changed.emit()

    def _description_editing_finished(
        self,
        next_focus: object,
        reason: object,
    ) -> None:
        mouse_focus = reason == Qt.FocusReason.MouseFocusReason
        self.commit_revision_metadata(
            render_change=not mouse_focus and next_focus not in self._focus_commit_targets
        )

    def render(self, document: Stack, selected_card_id: UUID | None) -> None:
        previous_card_id = self.selected_card_id
        previous_revision_id = self._rendered_revision_id
        preserve_description = self.description_edit.hasFocus()
        description_draft = self.description_edit.toPlainText()
        preserve_style_name = self.style_name_edit.hasFocus()
        style_name_draft = self.style_name_edit.text()
        preserve_style_prompt = self.style_prompt_edit.hasFocus()
        style_prompt_draft = self.style_prompt_edit.toPlainText()
        previous_style_id = self._rendered_style_id
        preserve_key_name = self.key_name_edit.hasFocus()
        key_name_draft = self.key_name_edit.text()
        previous_key_id = self._rendered_key_id
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
                self.pages.setCurrentIndex(0)
                self._set_error(self.description_error, "")
                self._set_error(self.reference_error, "")
                self._set_error(self.style_error, "")
                self._set_error(self.key_error, "")
                self.set_hotspot_error("")
                self.description_edit.clear()
                self.description_toggle.hide()
                self._image_prompt_current = None
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
            if revision.image_prompt is None or not same_revision:
                self._description_mode = "description"
            else:
                self._description_mode = previous_mode
            displayed_value = (
                revision.image_prompt.text
                if (self._description_mode == "image_prompt" and revision.image_prompt is not None)
                else revision.description
            )
            self.description_edit.setPlainText(
                description_draft if preserve_description and same_revision else displayed_value
            )
            self._render_style_selector(document, revision)
            self._render_styles(
                document,
                revision,
                style_name_draft=(
                    style_name_draft
                    if preserve_style_name and previous_style_id == self._selected_style_id
                    else None
                ),
                style_prompt_draft=(
                    style_prompt_draft
                    if preserve_style_prompt and previous_style_id == self._selected_style_id
                    else None
                ),
            )
            self._render_reference(document, card, revision)
            self._render_description_workflow(document, card, revision)
            self._render_hotspots(document, revision)
            self._render_keys(
                document,
                key_name_draft=(
                    key_name_draft
                    if preserve_key_name and previous_key_id == self._selected_key_id
                    else None
                ),
            )
        finally:
            self._rendering = False

    def commit_revision_metadata(self, *, render_change: bool = True) -> bool:
        if self._rendering or self.selected_card_id is None:
            return False
        card = self._selected_card()
        if card is None:
            return False
        value = self.description_edit.toPlainText()
        if self._description_mode == "description":
            if value == card.active_revision.description:
                return True
            return self._execute(
                EditRevisionDescriptionCommand(
                    card_id=card.id,
                    revision_id=card.active_revision.id,
                    value=value,
                ),
                error_label=self.description_error,
                render_change=render_change,
            )
        existing = card.active_revision.image_prompt
        if existing is None or value.strip() == existing.text:
            return True
        if not value.strip():
            return self._execute(
                SetRevisionImagePromptCommand(
                    card_id=card.id,
                    revision_id=card.active_revision.id,
                    value=None,
                ),
                error_label=self.description_error,
                undo_message="Image Prompt removed",
                render_change=render_change,
            )
        return self._execute(
            SetRevisionImagePromptCommand(
                card_id=card.id,
                revision_id=card.active_revision.id,
                value=existing.model_copy(update={"text": value.strip()}),
            ),
            error_label=self.description_error,
            render_change=render_change,
        )

    def commit_card_metadata(self) -> bool:
        """Commit every visible authoring draft before a context change or save."""
        if self._selected_interaction() is not None and not self._commit_hotspot_name():
            return False
        if self._selected_style() is not None and not self._commit_style():
            return False
        if self._selected_key_id is not None and not self._commit_key():
            return False
        return self.commit_revision_metadata()

    def has_current_image_prompt(self) -> bool:
        card = self._selected_card()
        if card is None:
            return False
        visible_value = self.description_edit.toPlainText().strip()
        image_prompt = (
            visible_value
            if self._description_mode == "image_prompt"
            else (
                card.active_revision.image_prompt.text
                if card.active_revision.image_prompt is not None
                else ""
            )
        )
        return (
            bool(card.active_revision.description.strip())
            and bool(image_prompt)
            and self._image_prompt_current is True
        )

    def has_description_input(self) -> bool:
        card = self._selected_card()
        if card is None:
            return False
        if self._description_mode == "description":
            return bool(self.description_edit.toPlainText().strip())
        return bool(card.active_revision.description.strip())

    def show_image_prompt(
        self,
        card_id: UUID,
        revision_id: UUID,
    ) -> None:
        """Display a newly prepared prompt when its revision is still active."""
        card = self._selected_card()
        if (
            card is None
            or card.id != card_id
            or card.active_revision.id != revision_id
            or card.active_revision.image_prompt is None
        ):
            return
        self._description_mode = "image_prompt"
        self.render(self.controller.document, self.selected_card_id)
        image_prompt = card.active_revision.image_prompt
        assert image_prompt is not None
        with QSignalBlocker(self.description_edit):
            self.description_edit.setPlainText(image_prompt.text)

    def _switch_description_mode(self, mode: str) -> None:
        if self._rendering or mode == self._description_mode:
            return
        card = self._selected_card()
        if (
            card is None
            or (mode == "image_prompt" and card.active_revision.image_prompt is None)
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
        reference_suffix = " + Reference" if revision.reference is not None else ""
        style_suffix = " + Style" if revision.style_id is not None else ""
        self._enrich_using_text = f"Using: Description{reference_suffix}"
        image_prompt = revision.image_prompt
        if image_prompt is None:
            self._image_prompt_current = None
        else:
            authored_description = (
                self.description_edit.toPlainText()
                if self._description_mode == "description"
                else revision.description
            )
            reference_snapshot = image_prompt_reference_snapshot(
                document,
                card,
            )
            reference_is_usable = revision.reference is None or reference_snapshot is not None
            self._image_prompt_current = reference_is_usable and image_prompt.is_current(
                source_description=authored_description,
                reference=reference_snapshot,
                model_identifier=self._image_prompt_model_identifier,
                prompt_version=self._image_prompt_prompt_version,
            )
        self.description_toggle.setVisible(image_prompt is not None)
        self.description_button.setChecked(self._description_mode == "description")
        self.image_prompt_button.setChecked(self._description_mode == "image_prompt")
        label = "Image Prompt" if self._description_mode == "image_prompt" else "Description"
        self.description_label.setText(label)
        self.description_edit.setAccessibleName(label)
        self.description_edit.setPlaceholderText(label)
        self._generate_using_text = f"Using: Image Prompt{style_suffix}{reference_suffix}"
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

    def set_image_prompt_capabilities(
        self,
        *,
        can_enrich: bool,
        reason: str,
        busy: bool,
        model_identifier: str | None = None,
        prompt_version: str | None = None,
    ) -> None:
        self._enrich_reason = reason
        self._can_enrich = can_enrich
        self._image_prompt_busy = busy
        self._image_prompt_model_identifier = model_identifier
        self._image_prompt_prompt_version = prompt_version
        self._update_image_prompt_freshness()

    def _update_image_prompt_freshness(self) -> None:
        card = self._selected_card()
        if card is None or card.active_revision.image_prompt is None:
            self._image_prompt_current = None
        else:
            source_description = (
                self.description_edit.toPlainText()
                if self._description_mode == "description"
                else card.active_revision.description
            )
            reference_snapshot = image_prompt_reference_snapshot(
                self.controller.document,
                card,
            )
            reference_is_usable = (
                card.active_revision.reference is None or reference_snapshot is not None
            )
            self._image_prompt_current = (
                reference_is_usable
                and card.active_revision.image_prompt.is_current(
                    source_description=source_description,
                    reference=reference_snapshot,
                    model_identifier=self._image_prompt_model_identifier,
                    prompt_version=self._image_prompt_prompt_version,
                )
            )
        self._update_enrich_button()
        self._refresh_generation_tooltips()

    def _update_enrich_button(self) -> None:
        card = self._selected_card()
        has_image_prompt = card is not None and card.active_revision.image_prompt is not None
        if self._image_prompt_busy:
            text = "Preparing Image Prompt…"
            enabled = False
        elif self._image_prompt_current is True:
            text = "Image Prompt Current"
            enabled = False
        else:
            text = "Update Image Prompt" if has_image_prompt else "Prepare Image Prompt"
            enabled = self._can_enrich
        self.enrich_button.setText(text)
        self.enrich_button.setEnabled(enabled)
        self.enrich_button.setAccessibleDescription(self._enrich_state_reason())

    def _refresh_generation_tooltips(self) -> None:
        self.enrich_button.setToolTip(
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
        if self._image_prompt_busy:
            return "Image Prompt preparation is running"
        if self._image_prompt_current is True:
            return "Current"
        if self._image_prompt_current is False:
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

    def show_interaction(self, interaction_id: UUID) -> None:
        self.inspector_tabs.setCurrentIndex(self._hotspots_tab_index)
        self.select_interaction(interaction_id)

    def set_hotspot_error(self, message: str) -> None:
        self.hotspot_error.setText(message)
        self.hotspot_error.setVisible(bool(message))

    def reset_context(self) -> None:
        self.selected_card_id = None
        self._rendered_revision_id = None
        self._set_error(self.description_error, "")
        self._set_error(self.reference_error, "")
        self._set_error(self.style_error, "")
        self._set_error(self.key_error, "")
        self.set_hotspot_error("")

    def _render_style_selector(
        self,
        document: Stack,
        revision: CardRevision,
    ) -> None:
        with QSignalBlocker(self.style_combo):
            self.style_combo.clear()
            self.style_combo.addItem("No Style", None)
            for style in document.styles:
                self.style_combo.addItem(style.name, style.id)
            self.style_combo.setCurrentIndex(
                self._combo_index_for_data(
                    self.style_combo,
                    revision.style_id,
                )
            )

    def _revision_style_changed(self, index: int) -> None:
        if self._rendering or index < 0:
            return
        card = self._selected_card()
        if card is None:
            return
        style_id = self.style_combo.itemData(index)
        if not isinstance(style_id, UUID):
            style_id = None
        if style_id == card.active_revision.style_id:
            return
        self._execute(
            SetRevisionStyleCommand(
                card_id=card.id,
                revision_id=card.active_revision.id,
                style_id=style_id,
            ),
            error_label=self.style_error,
            undo_message="Style changed",
        )

    @property
    def selected_style_id(self) -> UUID | None:
        item = self.style_list.currentItem()
        if item is None:
            return None
        value = item.data(Qt.ItemDataRole.UserRole)
        return value if isinstance(value, UUID) else None

    def _render_styles(
        self,
        document: Stack,
        revision: CardRevision,
        *,
        style_name_draft: str | None,
        style_prompt_draft: str | None,
    ) -> None:
        desired_id = self._selected_style_id
        known_ids = {style.id for style in document.styles}
        if desired_id not in known_ids:
            desired_id = (
                revision.style_id
                if revision.style_id in known_ids
                else document.styles[0].id
                if document.styles
                else None
            )
        with QSignalBlocker(self.style_list):
            self.style_list.clear()
            for style in document.styles:
                item = QListWidgetItem(style.name)
                item.setData(Qt.ItemDataRole.UserRole, style.id)
                self.style_list.addItem(item)
            selected_row = next(
                (
                    row
                    for row in range(self.style_list.count())
                    if self.style_list.item(row).data(Qt.ItemDataRole.UserRole) == desired_id
                ),
                -1,
            )
            self.style_list.setCurrentRow(selected_row)
        self._selected_style_id = desired_id
        self._rendered_style_id = desired_id
        self.styles_placeholder.setVisible(not document.styles)
        style = document.style_by_id(desired_id)
        self._render_style_properties(
            style,
            style_name_draft=style_name_draft,
            style_prompt_draft=style_prompt_draft,
        )

    def _render_style_properties(
        self,
        style: StyleDefinition | None,
        *,
        style_name_draft: str | None = None,
        style_prompt_draft: str | None = None,
    ) -> None:
        enabled = style is not None
        self.delete_style_button.setEnabled(enabled)
        self.style_name_edit.setEnabled(enabled)
        self.style_prompt_edit.setEnabled(enabled)
        with QSignalBlocker(self.style_name_edit):
            self.style_name_edit.setText(
                style_name_draft
                if style_name_draft is not None
                else style.name
                if style is not None
                else ""
            )
        with QSignalBlocker(self.style_prompt_edit):
            self.style_prompt_edit.setPlainText(
                style_prompt_draft
                if style_prompt_draft is not None
                else style.prompt_text
                if style is not None
                else ""
            )

    def _style_selection_changed(
        self,
        current: QListWidgetItem | None,
        _previous: QListWidgetItem | None,
    ) -> None:
        if self._rendering:
            return
        value = current.data(Qt.ItemDataRole.UserRole) if current is not None else None
        self._selected_style_id = value if isinstance(value, UUID) else None
        self._rendered_style_id = self._selected_style_id
        self._set_error(self.style_error, "")
        self._render_style_properties(self._selected_style())

    def _style_editing_finished(
        self,
        _next_focus: object,
        reason: object,
    ) -> None:
        self._commit_style(
            render_change=reason != Qt.FocusReason.MouseFocusReason
        )

    def _add_style(self) -> None:
        document = self.controller.document
        existing_names = {style.name.casefold() for style in document.styles}
        number = 1
        name = "New Style"
        while name.casefold() in existing_names:
            number += 1
            name = f"New Style {number}"
        command = AddStyleCommand(name=name)
        self._selected_style_id = command.style_id
        if self._execute(
            command,
            error_label=self.style_error,
            undo_message="Style added",
        ):
            self.inspector_tabs.setCurrentIndex(self._styles_tab_index)
            self.style_name_edit.setFocus()
            self.style_name_edit.selectAll()

    def _delete_style(self) -> None:
        style_id = self.selected_style_id
        if style_id is None:
            return
        self._selected_style_id = None
        self._execute(
            DeleteStyleCommand(style_id=style_id),
            error_label=self.style_error,
            undo_message="Style deleted",
        )

    def _commit_style(self, *, render_change: bool = True) -> bool:
        if self._rendering:
            return False
        style = self._selected_style()
        if style is None:
            return False
        name = self.style_name_edit.text()
        prompt_text = self.style_prompt_edit.toPlainText()
        if name == style.name and prompt_text.strip() == style.prompt_text:
            return True
        return self._execute(
            UpdateStyleCommand(
                style_id=style.id,
                name=name,
                prompt_text=prompt_text,
            ),
            error_label=self.style_error,
            undo_message="Style updated",
            render_change=render_change,
        )

    def _selected_style(self) -> StyleDefinition | None:
        return self.controller.document.style_by_id(self._selected_style_id)

    @property
    def selected_key_id(self) -> UUID | None:
        item = self.key_list.currentItem()
        if item is None:
            return None
        value = item.data(Qt.ItemDataRole.UserRole)
        return value if isinstance(value, UUID) else None

    @staticmethod
    def _key_usages(
        document: Stack,
        key_id: UUID,
    ) -> list[tuple[Card, int, Interaction, tuple[str, ...]]]:
        usages: list[tuple[Card, int, Interaction, tuple[str, ...]]] = []
        for card in document.cards:
            for revision_number, revision in enumerate(card.revisions, start=1):
                if revision.hotspot_set is None:
                    continue
                for interaction in revision.hotspot_set.interactions:
                    roles = tuple(
                        role
                        for role, key_ids in (
                            ("Requires", interaction.conditions.requires),
                            ("Forbids", interaction.conditions.forbids),
                            ("Removes", interaction.key_changes.remove),
                            ("Grants", interaction.key_changes.grant),
                        )
                        if key_id in key_ids
                    )
                    if roles:
                        usages.append(
                            (card, revision_number, interaction, roles)
                        )
        return usages

    def _render_keys(
        self,
        document: Stack,
        *,
        key_name_draft: str | None,
    ) -> None:
        desired_id = self._selected_key_id
        known_ids = {key.id for key in document.keys}
        if desired_id not in known_ids:
            desired_id = document.keys[0].id if document.keys else None
        with QSignalBlocker(self.key_list):
            self.key_list.clear()
            for key in document.keys:
                use_count = len(self._key_usages(document, key.id))
                suffix = (
                    "unused"
                    if use_count == 0
                    else f"{use_count} use"
                    if use_count == 1
                    else f"{use_count} uses"
                )
                item = QListWidgetItem(f"{key.name} — {suffix}")
                item.setData(Qt.ItemDataRole.UserRole, key.id)
                item.setToolTip(key.name)
                self.key_list.addItem(item)
            selected_row = next(
                (
                    row
                    for row in range(self.key_list.count())
                    if self.key_list.item(row).data(Qt.ItemDataRole.UserRole)
                    == desired_id
                ),
                -1,
            )
            self.key_list.setCurrentRow(selected_row)
        self._selected_key_id = desired_id
        self._rendered_key_id = desired_id
        self.keys_placeholder.setVisible(not document.keys)
        key = (
            document.key_by_id(desired_id)
            if desired_id is not None
            else None
        )
        self._render_key_properties(
            document,
            key,
            key_name_draft=key_name_draft,
        )

    def _render_key_properties(
        self,
        document: Stack,
        key: KeyDefinition | None,
        *,
        key_name_draft: str | None = None,
    ) -> None:
        enabled = key is not None
        usages = self._key_usages(document, key.id) if key is not None else []
        self.key_name_edit.setEnabled(enabled)
        self.delete_key_button.setEnabled(enabled and not usages)
        self.delete_key_button.setToolTip(
            "Remove hotspot references before deleting this Key"
            if usages
            else "Delete unused Key"
        )
        with QSignalBlocker(self.key_name_edit):
            self.key_name_edit.setText(
                key_name_draft
                if key_name_draft is not None
                else key.name
                if key is not None
                else ""
            )
        with QSignalBlocker(self.key_usage_list):
            self.key_usage_list.clear()
            for card, revision_number, interaction, roles in usages:
                text = self._key_usage_text(
                    card,
                    revision_number,
                    interaction,
                    roles,
                )
                item = QListWidgetItem(text)
                item.setData(
                    Qt.ItemDataRole.UserRole,
                    (card.id, card.revisions[revision_number - 1].id, interaction.id),
                )
                item.setToolTip(text)
                self.key_usage_list.addItem(item)
        self.show_hotspot_usage_button.setEnabled(False)

    @staticmethod
    def _key_usage_text(
        card: Card,
        revision_number: int,
        interaction: Interaction,
        roles: tuple[str, ...],
    ) -> str:
        return (
            f"{card.name} · Version {revision_number}\n"
            f"{interaction.label}\n"
            f"{' · '.join(roles)}"
        )

    def _key_selection_changed(
        self,
        current: QListWidgetItem | None,
        _previous: QListWidgetItem | None,
    ) -> None:
        if self._rendering:
            return
        value = current.data(Qt.ItemDataRole.UserRole) if current is not None else None
        self._selected_key_id = value if isinstance(value, UUID) else None
        self._rendered_key_id = self._selected_key_id
        self._set_error(self.key_error, "")
        key = (
            self.controller.document.key_by_id(self._selected_key_id)
            if self._selected_key_id is not None
            else None
        )
        self._render_key_properties(self.controller.document, key)

    def _add_key(self) -> None:
        document = self.controller.document
        existing_names = {key.name.casefold() for key in document.keys}
        number = 1
        name = "New Key"
        while name.casefold() in existing_names:
            number += 1
            name = f"New Key {number}"
        command = AddKeyCommand(name=name)
        self._selected_key_id = command.key_id
        if self._execute(
            command,
            error_label=self.key_error,
            undo_message="Key added",
        ):
            self.inspector_tabs.setCurrentIndex(self._keys_tab_index)
            self.key_name_edit.setFocus()
            self.key_name_edit.selectAll()

    def _delete_key(self) -> None:
        key_id = self.selected_key_id
        if key_id is None:
            return
        self._selected_key_id = None
        self._execute(
            DeleteKeyCommand(key_id=key_id),
            error_label=self.key_error,
            undo_message="Key deleted",
        )

    def _key_name_editing_finished(
        self,
        _next_focus: object,
        reason: object,
    ) -> None:
        mouse_focus = reason == Qt.FocusReason.MouseFocusReason
        if not self._commit_key(render_change=not mouse_focus) or not mouse_focus:
            return
        key_id = self._selected_key_id
        item = self.key_list.currentItem()
        if key_id is None or item is None:
            return
        document = self.controller.document
        key = document.key_by_id(key_id)
        usages = self._key_usages(document, key_id)
        suffix = (
            "unused"
            if not usages
            else "1 use"
            if len(usages) == 1
            else f"{len(usages)} uses"
        )
        item.setText(f"{key.name} — {suffix}")
        item.setToolTip(key.name)
        for row, usage in enumerate(usages):
            usage_item = self.key_usage_list.item(row)
            if usage_item is None:
                continue
            card, revision_number, interaction, roles = usage
            text = self._key_usage_text(
                card,
                revision_number,
                interaction,
                roles,
            )
            usage_item.setText(text)
            usage_item.setToolTip(text)

    def _commit_key(self, *, render_change: bool = True) -> bool:
        if self._rendering or self._selected_key_id is None:
            return False
        key = self.controller.document.key_by_id(self._selected_key_id)
        name = self.key_name_edit.text()
        if name.strip() == key.name:
            return True
        return self._execute(
            RenameKeyCommand(key_id=key.id, name=name),
            error_label=self.key_error,
            undo_message="Key renamed",
            render_change=render_change,
        )

    def _show_hotspot_usage(self) -> None:
        item = self.key_usage_list.currentItem()
        value = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
        if (
            isinstance(value, tuple)
            and len(value) == 3
            and all(isinstance(identifier, UUID) for identifier in value)
        ):
            self.hotspot_usage_requested.emit(*value)

    def _render_reference(
        self,
        document: Stack,
        card: Card,
        revision: CardRevision,
    ) -> None:
        reference = revision.reference
        with QSignalBlocker(self.reference_combo):
            self.reference_combo.clear()
            self.reference_combo.addItem("No reference", None)
            for candidate in document.cards:
                if candidate.id == card.id:
                    continue
                self.reference_combo.addItem(candidate.name, candidate.id)
            if isinstance(reference, ResolvedCardReference):
                self.reference_combo.setCurrentIndex(
                    self._combo_index_for_data(
                        self.reference_combo,
                        reference.target_card_id,
                    )
                )
            elif isinstance(reference, UnresolvedCardReference):
                name = reference.target_name or "Unknown card"
                self.reference_combo.addItem(f"Missing: {name}", reference)
                self.reference_combo.setCurrentIndex(self.reference_combo.count() - 1)
            else:
                self.reference_combo.setCurrentIndex(0)

    def _reference_changed(self, index: int) -> None:
        if self._rendering or index < 0:
            return
        card = self._selected_card()
        if card is None:
            return
        value = self.reference_combo.itemData(index)
        if isinstance(value, UUID):
            reference = ResolvedCardReference(target_card_id=value)
        elif isinstance(value, UnresolvedCardReference):
            reference = value
        else:
            reference = None
        if reference == card.active_revision.reference:
            return
        self._execute(
            SetRevisionReferenceCommand(
                card_id=card.id,
                revision_id=card.active_revision.id,
                reference=reference,
            ),
            error_label=self.reference_error,
            undo_message="Reference changed",
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
                label = self._hotspot_list_label(interaction)
                item = QListWidgetItem(label)
                item.setData(Qt.ItemDataRole.UserRole, interaction.id)
                item.setToolTip(interaction.label)
                self._size_hotspot_item(item)
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
        self.inspector_tabs.setTabText(self._hotspots_tab_index, "Hotspots")

    @staticmethod
    def _hotspot_list_label(interaction: Interaction) -> str:
        label = interaction.label
        if " → " in label:
            return label.replace(" → ", "\n→ ", 1)
        if len(label) > 32 and " · " in label:
            return label.replace(" · ", "\n· ", 1)
        return label

    def _size_hotspot_item(self, item: QListWidgetItem) -> None:
        line_count = 2 if "\n" in item.text() else 1
        item.setSizeHint(
            QSize(
                0,
                self.hotspot_list.fontMetrics().lineSpacing() * line_count + 8,
            )
        )

    def _render_hotspot_properties(
        self,
        document: Stack,
        interaction: Interaction | None,
    ) -> None:
        has_interaction = interaction is not None
        self.hotspot_name_edit.setEnabled(has_interaction)
        self.condition_table.setEnabled(has_interaction)
        self.add_condition_button.setEnabled(has_interaction)
        self.key_change_table.setEnabled(has_interaction)
        self.add_key_change_button.setEnabled(
            has_interaction
            and interaction is not None
            and not interaction.key_changes.clear_all
        )
        self.clear_all_keys_checkbox.setEnabled(has_interaction)
        self.hotspot_destination_combo.setEnabled(has_interaction)
        self.delete_hotspot_button.setEnabled(has_interaction)
        current_row = self.hotspot_list.currentRow()
        self.move_hotspot_up_button.setEnabled(has_interaction and current_row > 0)
        self.move_hotspot_down_button.setEnabled(
            has_interaction and current_row < self.hotspot_list.count() - 1
        )
        with QSignalBlocker(self.hotspot_name_edit):
            self.hotspot_name_edit.setText(
                interaction.name if interaction is not None and interaction.name else ""
            )
        self._render_condition_rows(document, interaction)
        self._render_key_change_rows(document, interaction)
        with QSignalBlocker(self.clear_all_keys_checkbox):
            self.clear_all_keys_checkbox.setChecked(
                interaction is not None and interaction.key_changes.clear_all
            )
        with QSignalBlocker(self.hotspot_destination_combo):
            self.hotspot_destination_combo.clear()
            self.hotspot_destination_combo.addItem("No destination", None)
            for card in document.cards:
                self.hotspot_destination_combo.addItem(card.name, card.id)
            if (
                interaction is not None
                and interaction.action is not None
                and isinstance(interaction.action.target, UnresolvedCardReference)
            ):
                target = interaction.action.target
                self.hotspot_destination_combo.addItem(
                    (
                        f"Missing: {target.target_name}"
                        if target.target_name
                        else "Unresolved destination"
                    ),
                    target,
                )
            self.hotspot_destination_combo.addItem("Create New Card...", "create")
            if interaction is None:
                self.hotspot_destination_combo.setCurrentIndex(-1)
            elif (
                interaction.action is not None
                and isinstance(interaction.action.target, ResolvedCardReference)
            ):
                self.hotspot_destination_combo.setCurrentIndex(
                    self._combo_index_for_data(
                        self.hotspot_destination_combo,
                        interaction.action.target.target_card_id,
                    )
                )
            elif (
                interaction.action is not None
                and isinstance(interaction.action.target, UnresolvedCardReference)
            ):
                self.hotspot_destination_combo.setCurrentIndex(
                    self._combo_index_for_data(
                        self.hotspot_destination_combo,
                        interaction.action.target,
                    )
                )
            else:
                self.hotspot_destination_combo.setCurrentIndex(0)
        self.hotspot_summary.setText(
            self._hotspot_summary_text(document, interaction)
        )

    def _render_condition_rows(
        self,
        document: Stack,
        interaction: Interaction | None,
    ) -> None:
        self.condition_table.setRowCount(0)
        if interaction is None:
            return
        rows = (
            *(("Present", key_id) for key_id in interaction.conditions.requires),
            *(("Absent", key_id) for key_id in interaction.conditions.forbids),
        )
        for row, (state, key_id) in enumerate(rows):
            self.condition_table.insertRow(row)
            key_combo = self._key_combo(document, key_id)
            state_combo = QComboBox()
            state_combo.addItem("Present", "requires")
            state_combo.addItem("Absent", "forbids")
            state_combo.setCurrentIndex(0 if state == "Present" else 1)
            remove_button = _compact_text_button(
                "−",
                object_name=f"removeConditionButton{row}",
                accessible_name="Remove condition",
                tooltip="Remove condition",
            )
            self.condition_table.setCellWidget(row, 0, key_combo)
            self.condition_table.setCellWidget(row, 1, state_combo)
            self.condition_table.setCellWidget(row, 2, remove_button)
            role = "requires" if state == "Present" else "forbids"
            key_combo.currentIndexChanged.connect(
                lambda _index, old_key_id=key_id, old_role=role, combo=key_combo: (
                    self._change_condition(
                        old_key_id,
                        old_role,
                        combo.currentData(),
                        old_role,
                    )
                )
            )
            state_combo.currentIndexChanged.connect(
                lambda _index, old_key_id=key_id, old_role=role, combo=state_combo: (
                    self._change_condition(
                        old_key_id,
                        old_role,
                        old_key_id,
                        combo.currentData(),
                    )
                )
            )
            remove_button.clicked.connect(
                lambda _checked=False, existing_key_id=key_id, existing_role=role: (
                    self._remove_condition(existing_key_id, existing_role)
                )
            )

    def _render_key_change_rows(
        self,
        document: Stack,
        interaction: Interaction | None,
    ) -> None:
        self.key_change_table.setRowCount(0)
        if interaction is None or interaction.key_changes.clear_all:
            return
        rows = (
            *(("Remove", key_id) for key_id in interaction.key_changes.remove),
            *(("Grant", key_id) for key_id in interaction.key_changes.grant),
        )
        for row, (change, key_id) in enumerate(rows):
            self.key_change_table.insertRow(row)
            change_combo = QComboBox()
            change_combo.addItem("Remove", "remove")
            change_combo.addItem("Grant", "grant")
            change_combo.setCurrentIndex(0 if change == "Remove" else 1)
            key_combo = self._key_combo(document, key_id)
            remove_button = _compact_text_button(
                "−",
                object_name=f"removeKeyChangeButton{row}",
                accessible_name="Remove key change",
                tooltip="Remove key change",
            )
            self.key_change_table.setCellWidget(row, 0, change_combo)
            self.key_change_table.setCellWidget(row, 1, key_combo)
            self.key_change_table.setCellWidget(row, 2, remove_button)
            role = "remove" if change == "Remove" else "grant"
            change_combo.currentIndexChanged.connect(
                lambda _index, old_key_id=key_id, old_role=role, combo=change_combo: (
                    self._change_key_change(
                        old_key_id,
                        old_role,
                        old_key_id,
                        combo.currentData(),
                    )
                )
            )
            key_combo.currentIndexChanged.connect(
                lambda _index, old_key_id=key_id, old_role=role, combo=key_combo: (
                    self._change_key_change(
                        old_key_id,
                        old_role,
                        combo.currentData(),
                        old_role,
                    )
                )
            )
            remove_button.clicked.connect(
                lambda _checked=False, existing_key_id=key_id, existing_role=role: (
                    self._remove_key_change(existing_key_id, existing_role)
                )
            )

    @staticmethod
    def _key_combo(document: Stack, selected_key_id: UUID) -> QComboBox:
        combo = QComboBox()
        for key in document.keys:
            combo.addItem(key.name, key.id)
        combo.setCurrentIndex(Inspector._combo_index_for_data(combo, selected_key_id))
        return combo

    @staticmethod
    def _hotspot_summary_text(
        document: Stack,
        interaction: Interaction | None,
    ) -> str:
        if interaction is None:
            return ""
        condition_parts = [
            *(
                f"{document.key_by_id(key_id).name} is present"
                for key_id in interaction.conditions.requires
            ),
            *(
                f"{document.key_by_id(key_id).name} is absent"
                for key_id in interaction.conditions.forbids
            ),
        ]
        action_parts: list[str] = []
        if interaction.key_changes.clear_all:
            action_parts.append("clear all keys")
        else:
            action_parts.extend(
                f"remove {document.key_by_id(key_id).name}"
                for key_id in interaction.key_changes.remove
            )
            action_parts.extend(
                f"grant {document.key_by_id(key_id).name}"
                for key_id in interaction.key_changes.grant
            )
        if interaction.action is not None:
            target = interaction.action.target
            destination = (
                next(
                    card.name
                    for card in document.cards
                    if isinstance(target, ResolvedCardReference)
                    and card.id == target.target_card_id
                )
                if isinstance(target, ResolvedCardReference)
                else target.target_name or "an unresolved destination"
            )
            action_parts.append(f"go to {destination}")
        condition_text = (
            f"When {' and '.join(condition_parts)}"
            if condition_parts
            else "Always"
        )
        return (
            f"{condition_text}, {', then '.join(action_parts)}."
            if action_parts
            else f"{condition_text}; no actions."
        )

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
        interaction = Interaction()
        if self._execute(
            AddInteractionCommand(
                card_id=card.id,
                revision_id=card.active_revision.id,
                interaction=interaction,
            )
        ):
            self.select_interaction(interaction.id)
            self.hotspot_selected.emit(interaction.id)

    def _hotspot_name_editing_finished(
        self,
        _next_focus: object,
        reason: object,
    ) -> None:
        mouse_focus = reason == Qt.FocusReason.MouseFocusReason
        if self._commit_hotspot_name(render_change=not mouse_focus) and mouse_focus:
            interaction = self._selected_interaction()
            item = self.hotspot_list.currentItem()
            if interaction is not None and item is not None:
                item.setText(self._hotspot_list_label(interaction))
                item.setToolTip(interaction.label)
                self._size_hotspot_item(item)

    def _commit_hotspot_name(self, *, render_change: bool = True) -> bool:
        if self._rendering:
            return False
        card = self._selected_card()
        interaction = self._selected_interaction()
        if card is None or interaction is None:
            return False
        name = self.hotspot_name_edit.text().strip() or None
        if name == interaction.name:
            return True
        return self._execute(
            SetHotspotNameCommand(
                card_id=card.id,
                revision_id=card.active_revision.id,
                interaction_id=interaction.id,
                name=name,
            ),
            undo_message="Hotspot name changed",
            render_change=render_change,
        )

    def _add_condition(self) -> None:
        interaction = self._selected_interaction()
        if interaction is None:
            return
        used = set(interaction.conditions.requires) | set(
            interaction.conditions.forbids
        )
        self._choose_key_reference(
            title="Add Condition",
            role="requires",
            excluded=used,
        )

    def _add_key_change(self) -> None:
        interaction = self._selected_interaction()
        if interaction is None or interaction.key_changes.clear_all:
            return
        used = set(interaction.key_changes.remove) | set(
            interaction.key_changes.grant
        )
        self._choose_key_reference(
            title="Add Key Change",
            role="grant",
            excluded=used,
        )

    def _choose_key_reference(
        self,
        *,
        title: str,
        role: Literal["requires", "forbids", "remove", "grant"],
        excluded: set[UUID],
    ) -> None:
        available = [
            key for key in self.controller.document.keys if key.id not in excluded
        ]
        create_label = "Create New Key..."
        labels = [key.name for key in available]
        while create_label.casefold() in {
            label.casefold() for label in labels
        }:
            create_label += "."
        selected, accepted = QInputDialog.getItem(
            self,
            title,
            "Key",
            (*labels, create_label),
            editable=False,
        )
        if not accepted:
            return
        if selected == create_label:
            self._create_key_reference(title=title, role=role)
            return
        key = next((key for key in available if key.name == selected), None)
        if key is None:
            return
        if role in {"requires", "forbids"}:
            self._append_condition(key.id, role)
        else:
            self._append_key_change(key.id, role)

    def _create_key_reference(
        self,
        *,
        title: str,
        role: Literal["requires", "forbids", "remove", "grant"],
    ) -> None:
        card = self._selected_card()
        interaction = self._selected_interaction()
        if card is None or interaction is None:
            return
        document = self.controller.document
        existing_names = {key.name.casefold() for key in document.keys}
        number = 1
        default_name = "New Key"
        while default_name.casefold() in existing_names:
            number += 1
            default_name = f"New Key {number}"
        name, accepted = QInputDialog.getText(
            self,
            title,
            "Key name",
            QLineEdit.EchoMode.Normal,
            default_name,
        )
        if not accepted:
            return
        command = CreateKeyAndAddHotspotReferenceCommand(
            card_id=card.id,
            revision_id=card.active_revision.id,
            interaction_id=interaction.id,
            name=name,
            role=role,
        )
        self._selected_key_id = command.key_id
        self._execute(
            command,
            undo_message="Key and hotspot behavior added",
        )

    def _append_condition(self, key_id: UUID, role: str) -> None:
        card = self._selected_card()
        interaction = self._selected_interaction()
        if card is None or interaction is None:
            return
        values = getattr(interaction.conditions, role)
        self._execute(
            SetHotspotConditionsCommand(
                card_id=card.id,
                revision_id=card.active_revision.id,
                interaction_id=interaction.id,
                conditions=interaction.conditions.model_copy(
                    update={role: (*values, key_id)}
                ),
            ),
            undo_message="Hotspot condition added",
        )

    def _change_condition(
        self,
        old_key_id: UUID,
        old_role: str,
        new_key_id: object,
        new_role: object,
    ) -> None:
        card = self._selected_card()
        interaction = self._selected_interaction()
        if (
            self._rendering
            or card is None
            or interaction is None
            or not isinstance(new_key_id, UUID)
            or new_role not in {"requires", "forbids"}
        ):
            return
        requires = [
            key_id
            for key_id in interaction.conditions.requires
            if not (old_role == "requires" and key_id == old_key_id)
        ]
        forbids = [
            key_id
            for key_id in interaction.conditions.forbids
            if not (old_role == "forbids" and key_id == old_key_id)
        ]
        target = requires if new_role == "requires" else forbids
        target.append(new_key_id)
        try:
            conditions = HotspotConditions(
                requires=tuple(requires),
                forbids=tuple(forbids),
            )
        except ValidationError as error:
            self.set_hotspot_error(str(error))
            self.render(self.controller.document, self.selected_card_id)
            return
        self._execute(
            SetHotspotConditionsCommand(
                card_id=card.id,
                revision_id=card.active_revision.id,
                interaction_id=interaction.id,
                conditions=conditions,
            ),
            undo_message="Hotspot condition changed",
        )

    def _remove_condition(self, key_id: UUID, role: str) -> None:
        card = self._selected_card()
        interaction = self._selected_interaction()
        if card is None or interaction is None:
            return
        values = tuple(
            candidate
            for candidate in getattr(interaction.conditions, role)
            if candidate != key_id
        )
        self._execute(
            SetHotspotConditionsCommand(
                card_id=card.id,
                revision_id=card.active_revision.id,
                interaction_id=interaction.id,
                conditions=interaction.conditions.model_copy(
                    update={role: values}
                ),
            ),
            undo_message="Hotspot condition removed",
        )

    def _append_key_change(self, key_id: UUID, role: str) -> None:
        card = self._selected_card()
        interaction = self._selected_interaction()
        if card is None or interaction is None:
            return
        values = getattr(interaction.key_changes, role)
        self._execute(
            SetHotspotKeyChangesCommand(
                card_id=card.id,
                revision_id=card.active_revision.id,
                interaction_id=interaction.id,
                key_changes=interaction.key_changes.model_copy(
                    update={role: (*values, key_id)}
                ),
            ),
            undo_message="Hotspot key change added",
        )

    def _change_key_change(
        self,
        old_key_id: UUID,
        old_role: str,
        new_key_id: object,
        new_role: object,
    ) -> None:
        card = self._selected_card()
        interaction = self._selected_interaction()
        if (
            self._rendering
            or card is None
            or interaction is None
            or not isinstance(new_key_id, UUID)
            or new_role not in {"remove", "grant"}
        ):
            return
        remove = [
            key_id
            for key_id in interaction.key_changes.remove
            if not (old_role == "remove" and key_id == old_key_id)
        ]
        grant = [
            key_id
            for key_id in interaction.key_changes.grant
            if not (old_role == "grant" and key_id == old_key_id)
        ]
        target = remove if new_role == "remove" else grant
        target.append(new_key_id)
        try:
            key_changes = HotspotKeyChanges(
                remove=tuple(remove),
                grant=tuple(grant),
            )
        except ValidationError as error:
            self.set_hotspot_error(str(error))
            self.render(self.controller.document, self.selected_card_id)
            return
        self._execute(
            SetHotspotKeyChangesCommand(
                card_id=card.id,
                revision_id=card.active_revision.id,
                interaction_id=interaction.id,
                key_changes=key_changes,
            ),
            undo_message="Hotspot key change updated",
        )

    def _remove_key_change(self, key_id: UUID, role: str) -> None:
        card = self._selected_card()
        interaction = self._selected_interaction()
        if card is None or interaction is None:
            return
        values = tuple(
            candidate
            for candidate in getattr(interaction.key_changes, role)
            if candidate != key_id
        )
        self._execute(
            SetHotspotKeyChangesCommand(
                card_id=card.id,
                revision_id=card.active_revision.id,
                interaction_id=interaction.id,
                key_changes=interaction.key_changes.model_copy(
                    update={role: values}
                ),
            ),
            undo_message="Hotspot key change removed",
        )

    def _clear_all_keys_toggled(self, checked: bool) -> None:
        if self._rendering:
            return
        card = self._selected_card()
        interaction = self._selected_interaction()
        if card is None or interaction is None:
            return
        self._execute(
            SetHotspotKeyChangesCommand(
                card_id=card.id,
                revision_id=card.active_revision.id,
                interaction_id=interaction.id,
                key_changes=(
                    HotspotKeyChanges(clear_all=True)
                    if checked
                    else HotspotKeyChanges()
                ),
            ),
            undo_message="Hotspot key changes updated",
        )

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
                    else destination
                    if isinstance(destination, UnresolvedCardReference)
                    else None
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
        render_change: bool = True,
    ) -> bool:
        target_error = error_label if error_label is not None else self.hotspot_error
        previous_token = self.controller.current_undo_token
        try:
            changed = self.controller.execute(command)
        except (CommandError, ValidationError) as error:
            self._set_error(target_error, str(error))
            if render_change:
                self.render(self.controller.document, self.selected_card_id)
            return False
        self._set_error(target_error, "")
        if render_change:
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
