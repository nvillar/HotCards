"""Minimal revision-local Background and Hotspots inspector."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import ValidationError
from PySide6.QtCore import QSignalBlocker, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QFocusEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
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
    QScrollArea,
    QStackedWidget,
    QStyle,
    QTableWidget,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from hotcards.application.commands import (
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
    SetRevisionGenerateResolutionCommand,
    SetRevisionReferenceCommand,
    SetRevisionStyleCommand,
    UpdateStyleCommand,
)
from hotcards.application.document_controller import (
    DocumentController,
    DocumentMutationBlockedError,
)
from hotcards.domain.image_dimensions import (
    GenerateResolution,
    higher_output_resolutions,
    output_dimensions,
)
from hotcards.domain.models import (
    Card,
    CardRevision,
    HotspotConditions,
    HotspotKeyChanges,
    Interaction,
    KeyDefinition,
    RefineTransformation,
    ResolvedCardReference,
    Stack,
    StyleDefinition,
    UnresolvedCardReference,
    image_edit_lineage,
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


def _configure_rule_table(
    table: QTableWidget,
    headers: tuple[str, str, str],
    *,
    stretch_column: int,
) -> None:
    table.setHorizontalHeaderLabels(headers)
    table.verticalHeader().setVisible(False)
    table.horizontalHeader().setSectionResizeMode(
        stretch_column, QHeaderView.ResizeMode.Stretch
    )
    for column in range(2):
        if column != stretch_column:
            table.horizontalHeader().setSectionResizeMode(
                column, QHeaderView.ResizeMode.ResizeToContents
            )
    table.horizontalHeader().setSectionResizeMode(
        2, QHeaderView.ResizeMode.Fixed
    )
    table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    table.setFrameShape(QFrame.Shape.NoFrame)


def _rule_panel(object_name: str) -> tuple[QFrame, QVBoxLayout]:
    panel = QFrame()
    panel.setObjectName(object_name)
    panel.setFrameShape(QFrame.Shape.StyledPanel)
    panel.setStyleSheet(
        f"""
        QFrame#{object_name} {{
            background-color: palette(base);
            border: 1px solid palette(mid);
            border-radius: 4px;
        }}
        """
    )
    layout = QVBoxLayout(panel)
    layout.setContentsMargins(8, 8, 8, 8)
    layout.setSpacing(8)
    return panel, layout


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
    prominent: bool = True,
) -> QToolButton:
    button = QToolButton()
    button.setObjectName(object_name)
    button.setText(text)
    button.setAccessibleName(accessible_name)
    button.setToolTip(tooltip)
    font = button.font()
    if prominent:
        font.setPointSizeF(max(font.pointSizeF() + 4.0, 16.0))
    font.setBold(True)
    button.setFont(font)
    return button


def _centered_cell_widget(widget: QWidget) -> QWidget:
    container = QWidget()
    layout = QHBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    layout.addWidget(widget, 0, Qt.AlignmentFlag.AlignCenter)
    return container


class Inspector(QWidget):
    """Render the selected active revision and issue typed commands."""

    document_changed = Signal(object)
    generate_background_requested = Signal()
    refine_background_requested = Signal(object, object)
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
        self._generate_using_text = ""
        self._generate_reason = ""
        self._refine_using_text = ""
        self._refine_reason = ""
        self._refine_source_size: tuple[int, int] | None = None
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
        self._build_refine_tab()
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
        self.description_edit.setPlaceholderText(
            "Describe the image to generate. Refer to selected References as "
            "image 1 and image 2."
        )
        self.description_edit.setAccessibleName("Description")
        editor_height = round((self.description_edit.fontMetrics().lineSpacing() * 10 + 20) * 1.25)
        self.description_edit.setMinimumHeight(editor_height)
        layout.addWidget(self.description_edit, 1)
        self.description_error = QLabel()
        self.description_error.setObjectName("descriptionValidationError")
        self.description_error.setWordWrap(True)
        self.description_error.setVisible(False)
        layout.addWidget(self.description_error)

        layout.addSpacing(8)
        self.style_label = QLabel("Style")
        self.style_label.setObjectName("styleLabel")
        layout.addWidget(self.style_label)
        self.style_combo = QComboBox()
        self.style_combo.setObjectName("styleCombo")
        self.style_combo.setAccessibleName("Style")
        self.style_combo.setToolTip(
            "Rendering treatment appended to the Description during generation"
        )
        layout.addWidget(self.style_combo)

        layout.addSpacing(8)
        self.reference_label = QLabel("References")
        self.reference_label.setObjectName("referenceLabel")
        layout.addWidget(self.reference_label)
        self.reference_panel = QWidget()
        self.reference_panel.setObjectName("referencePanel")
        reference_layout = QVBoxLayout(self.reference_panel)
        reference_layout.setContentsMargins(0, 0, 0, 0)
        reference_layout.setSpacing(4)
        first_reference_row = QHBoxLayout()
        first_reference_row.addWidget(QLabel("1."))
        self.reference_combo = QComboBox()
        self.reference_combo.setObjectName("referenceCombo")
        self.reference_combo.setAccessibleName("Reference card 1")
        self.reference_combo.setToolTip(
            "Primary Reference; its image is sent to the model as image 1"
        )
        first_reference_row.addWidget(self.reference_combo, 1)
        reference_layout.addLayout(first_reference_row)
        second_reference_row = QHBoxLayout()
        second_reference_row.addWidget(QLabel("2."))
        self.additional_reference_combo = QComboBox()
        self.additional_reference_combo.setObjectName("additionalReferenceCombo")
        self.additional_reference_combo.setAccessibleName("Reference card 2")
        self.additional_reference_combo.setToolTip(
            "Optional additional Reference; its image is sent to the model as image 2"
        )
        second_reference_row.addWidget(self.additional_reference_combo, 1)
        reference_layout.addLayout(second_reference_row)
        self.reference_error = QLabel()
        self.reference_error.setObjectName("referenceValidationError")
        self.reference_error.setWordWrap(True)
        self.reference_error.setVisible(False)
        reference_layout.addWidget(self.reference_error)
        layout.addWidget(self.reference_panel)

        layout.addSpacing(8)
        self.resolution_label = QLabel("Resolution")
        self.resolution_label.setObjectName("resolutionLabel")
        layout.addWidget(self.resolution_label)
        self.resolution_combo = QComboBox()
        self.resolution_combo.setObjectName("resolutionCombo")
        self.resolution_combo.setAccessibleName("Generate resolution")
        self.resolution_combo.setToolTip(
            "Square-equivalent Generate resolution and actual output dimensions"
        )
        layout.addWidget(self.resolution_combo)

        layout.addSpacing(8)
        self.generate_background_button = QPushButton("Generate Image")
        self.generate_background_button.setObjectName("generateBackgroundButton")
        layout.addWidget(self.generate_background_button)
        self._focus_commit_targets = {
            self.generate_background_button,
        }

        scroll.setWidget(page)
        self.inspector_tabs.addTab(scroll, "Generate")

    def _build_refine_tab(self) -> None:
        scroll = QScrollArea()
        scroll.setObjectName("refineInspectorTab")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        page = QWidget()
        page.setObjectName("refineInspectorContent")
        layout = QVBoxLayout(page)

        self.refine_transformation_label = QLabel("Transformation")
        layout.addWidget(self.refine_transformation_label)
        self.refine_transformation_combo = QComboBox()
        self.refine_transformation_combo.setObjectName(
            "refineTransformationCombo"
        )
        self.refine_transformation_combo.setAccessibleName(
            "Refine transformation"
        )
        for transformation in RefineTransformation:
            self.refine_transformation_combo.addItem(
                f"{transformation.value.title()} "
                f"({transformation.strength:.2f})",
                transformation,
            )
        self.refine_transformation_combo.setCurrentIndex(
            self._combo_index_for_data(
                self.refine_transformation_combo,
                RefineTransformation.BALANCED,
            )
        )
        layout.addWidget(self.refine_transformation_combo)

        layout.addSpacing(8)
        self.refine_resolution_label = QLabel("Output Resolution")
        layout.addWidget(self.refine_resolution_label)
        self.refine_resolution_combo = QComboBox()
        self.refine_resolution_combo.setObjectName("refineResolutionCombo")
        self.refine_resolution_combo.setAccessibleName(
            "Refine output resolution"
        )
        layout.addWidget(self.refine_resolution_combo)
        self.refine_error = QLabel()
        self.refine_error.setObjectName("refineValidationError")
        self.refine_error.setWordWrap(True)
        self.refine_error.setVisible(False)
        layout.addWidget(self.refine_error)

        layout.addSpacing(8)
        self.refine_background_button = QPushButton("Refine")
        self.refine_background_button.setObjectName("refineBackgroundButton")
        layout.addWidget(self.refine_background_button)
        self._focus_commit_targets.add(self.refine_background_button)
        layout.addStretch(1)

        scroll.setWidget(page)
        self._refine_tab_index = self.inspector_tabs.addTab(scroll, "Refine")

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

        self.hotspot_rule_scroll = QScrollArea()
        self.hotspot_rule_scroll.setObjectName("hotspotRuleScroll")
        self.hotspot_rule_scroll.setWidgetResizable(True)
        self.hotspot_rule_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.hotspot_rule_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        rule_content = QWidget()
        rule_content.setObjectName("hotspotRuleContent")
        rule_layout = QVBoxLayout(rule_content)
        rule_layout.setContentsMargins(0, 0, 0, 0)

        self.hotspot_when_label = QLabel("When")
        self.hotspot_when_label.setObjectName("hotspotWhenLabel")
        rule_layout.addWidget(self.hotspot_when_label)
        self.hotspot_when_panel, when_layout = _rule_panel("hotspotWhenPanel")
        self.condition_table = QTableWidget(0, 3)
        self.condition_table.setObjectName("hotspotConditionTable")
        _configure_rule_table(
            self.condition_table,
            ("State", "Key", ""),
            stretch_column=1,
        )
        when_layout.addWidget(self.condition_table)
        self.no_conditions_label = QLabel("No conditions (always activates)")
        self.no_conditions_label.setObjectName("noHotspotConditionsLabel")
        when_layout.addWidget(self.no_conditions_label)
        self.add_condition_button = QPushButton("+ Add condition")
        self.add_condition_button.setObjectName("addConditionButton")
        self.add_condition_button.setFlat(True)
        when_layout.addWidget(
            self.add_condition_button,
            0,
            Qt.AlignmentFlag.AlignLeft,
        )
        rule_layout.addWidget(self.hotspot_when_panel)

        self.hotspot_then_label = QLabel("Then")
        self.hotspot_then_label.setObjectName("hotspotThenLabel")
        rule_layout.addWidget(self.hotspot_then_label)
        self.hotspot_then_panel, then_layout = _rule_panel("hotspotThenPanel")
        self.key_change_table = QTableWidget(0, 3)
        self.key_change_table.setObjectName("hotspotKeyChangeTable")
        _configure_rule_table(
            self.key_change_table,
            ("Change", "Key", ""),
            stretch_column=1,
        )
        then_layout.addWidget(self.key_change_table)
        self.no_key_changes_label = QLabel("No key changes")
        self.no_key_changes_label.setObjectName("noHotspotKeyChangesLabel")
        then_layout.addWidget(self.no_key_changes_label)
        self.add_key_change_button = QPushButton("+ Add key change")
        self.add_key_change_button.setObjectName("addKeyChangeButton")
        self.add_key_change_button.setFlat(True)
        then_layout.addWidget(
            self.add_key_change_button,
            0,
            Qt.AlignmentFlag.AlignLeft,
        )

        self.hotspot_target_label = QLabel("Go to")
        self.hotspot_target_label.setObjectName("hotspotTargetLabel")
        then_layout.addWidget(self.hotspot_target_label)
        self.hotspot_destination_combo = QComboBox()
        self.hotspot_destination_combo.setObjectName("hotspotDestinationCombo")
        self.hotspot_destination_combo.setAccessibleName("Hotspot destination")
        then_layout.addWidget(self.hotspot_destination_combo)
        rule_layout.addWidget(self.hotspot_then_panel)

        self.hotspot_summary = QLabel()
        self.hotspot_summary.setObjectName("hotspotSummary")
        self.hotspot_summary.setWordWrap(True)
        rule_layout.addWidget(self.hotspot_summary)

        self.hotspot_error = QLabel()
        self.hotspot_error.setObjectName("hotspotValidationError")
        self.hotspot_error.setWordWrap(True)
        self.hotspot_error.setVisible(False)
        rule_layout.addWidget(self.hotspot_error)
        rule_layout.addStretch(1)
        self.hotspot_rule_scroll.setWidget(rule_content)
        layout.addWidget(self.hotspot_rule_scroll, 2)
        self._hotspots_tab_index = self.inspector_tabs.addTab(page, "Hotspots")

    def _build_keys_tab(self) -> None:
        page = QWidget()
        page.setObjectName("keysInspectorTab")
        layout = QVBoxLayout(page)

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
        self.generate_background_button.clicked.connect(self.generate_background_requested)
        self.refine_background_button.clicked.connect(
            self._request_refine_background
        )
        self.refine_transformation_combo.currentIndexChanged.connect(
            lambda _index: self._render_inputs_changed()
        )
        self.refine_resolution_combo.currentIndexChanged.connect(
            lambda _index: self._render_inputs_changed()
        )
        self.style_combo.currentIndexChanged.connect(self._revision_style_changed)
        self.reference_combo.currentIndexChanged.connect(
            lambda index: self._reference_changed(1, index)
        )
        self.additional_reference_combo.currentIndexChanged.connect(
            lambda index: self._reference_changed(2, index)
        )
        self.resolution_combo.currentIndexChanged.connect(
            self._revision_resolution_changed
        )
        self.style_list.currentItemChanged.connect(self._style_selection_changed)
        self.add_style_button.clicked.connect(self._add_style)
        self.delete_style_button.clicked.connect(self._delete_style)
        self.style_name_edit.editing_finished.connect(self._style_editing_finished)
        self.style_prompt_edit.editing_finished.connect(self._style_editing_finished)
        self.style_name_edit.textChanged.connect(self._style_draft_changed)
        self.style_prompt_edit.textChanged.connect(self._style_draft_changed)
        self.hotspot_list.currentItemChanged.connect(self._hotspot_selection_changed)
        self.move_hotspot_up_button.clicked.connect(lambda: self._move_hotspot(-1))
        self.move_hotspot_down_button.clicked.connect(lambda: self._move_hotspot(1))
        self.add_hotspot_button.clicked.connect(self._add_hotspot)
        self.delete_hotspot_button.clicked.connect(self._delete_hotspot)
        self.add_condition_button.clicked.connect(self._add_condition)
        self.add_key_change_button.clicked.connect(self._add_key_change)
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

    def render(
        self,
        document: Stack,
        selected_card_id: UUID | None,
        *,
        refine_source_size: tuple[int, int] | None = None,
    ) -> None:
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
                self._set_error(self.refine_error, "")
                self._set_error(self.style_error, "")
                self._set_error(self.key_error, "")
                self.set_hotspot_error("")
                self.description_edit.clear()
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
            self.description_edit.setPlainText(
                description_draft
                if preserve_description and same_revision
                else revision.description
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
            self._render_resolution(document, revision)
            self._render_description_workflow(document, card, revision)
            self._render_refine(
                document,
                revision,
                source_size=refine_source_size,
            )
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

    def commit_card_metadata(self, *, render_change: bool = True) -> bool:
        """Commit every visible authoring draft before a context change or save."""
        if (
            self._selected_style() is not None
            and not self._commit_style(render_change=render_change)
        ):
            return False
        if (
            self._selected_key_id is not None
            and not self._commit_key(render_change=render_change)
        ):
            return False
        return self.commit_revision_metadata(render_change=render_change)

    def has_description_input(self) -> bool:
        card = self._selected_card()
        if card is None:
            return False
        return bool(self.description_edit.toPlainText().strip())

    def _render_description_workflow(
        self,
        document: Stack,
        card: Card,
        revision: CardRevision,
    ) -> None:
        reference_count = len(revision.references)
        reference_suffix = (
            f" + {reference_count} References"
            if reference_count > 1
            else " + Reference"
            if reference_count == 1
            else ""
        )
        style_suffix = " + Style" if revision.style_id is not None else ""
        width, height = output_dimensions(
            revision.generate_resolution,
            document.aspect_ratio,
        )
        self._generate_using_text = (
            f"Using: Description{style_suffix}{reference_suffix}\n"
            f"Resolution: {revision.generate_resolution.value} "
            f"square-equivalent ({width} x {height})"
        )
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

    def set_refine_capabilities(
        self,
        *,
        can_refine: bool,
        refine_reason: str,
        busy: bool,
        refining: bool,
    ) -> None:
        self._refine_reason = refine_reason
        self.refine_background_button.setEnabled(can_refine and not busy)
        self.refine_background_button.setText(
            "Refining…" if refining else "Refine"
        )
        self._refresh_generation_tooltips()

    def _refresh_generation_tooltips(self) -> None:
        self.generate_background_button.setToolTip(
            self._tooltip_with_using(
                self._generate_reason,
                self._generate_using_text,
            )
        )
        self.refine_background_button.setToolTip(
            self._tooltip_with_using(
                self._refine_reason,
                self._refine_using_text,
            )
        )

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
        self._set_error(self.refine_error, "")
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
        self._render_inputs_changed()
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

    def _style_draft_changed(self) -> None:
        if (
            self.style_name_edit.hasFocus()
            or self.style_prompt_edit.hasFocus()
        ):
            self._render_inputs_changed()

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
        self._render_inputs_changed()
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
        self._render_inputs_changed()
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
                            ("Has", interaction.conditions.requires),
                            ("Lacks", interaction.conditions.forbids),
                            ("Loses", interaction.key_changes.remove),
                            ("Gains", interaction.key_changes.grant),
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
            for card, revision_number, interaction, _roles in usages:
                text = self._key_usage_text(
                    card,
                    revision_number,
                    interaction,
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
    ) -> str:
        return (
            f"{card.name} · Version {revision_number}\n"
            f"{interaction.label}"
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
            card, revision_number, interaction, _roles = usage
            text = self._key_usage_text(
                card,
                revision_number,
                interaction,
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
        references = revision.references
        combos = (self.reference_combo, self.additional_reference_combo)
        resolved_ids = {
            reference.target_card_id
            for reference in references
            if isinstance(reference, ResolvedCardReference)
        }
        for position, combo in enumerate(combos):
            reference = references[position] if position < len(references) else None
            with QSignalBlocker(combo):
                combo.clear()
                combo.addItem("No reference", None)
                current_id = (
                    reference.target_card_id
                    if isinstance(reference, ResolvedCardReference)
                    else None
                )
                for candidate in document.cards:
                    if candidate.id == card.id:
                        continue
                    if candidate.id in resolved_ids and candidate.id != current_id:
                        continue
                    combo.addItem(candidate.name, candidate.id)
                if isinstance(reference, ResolvedCardReference):
                    combo.setCurrentIndex(
                        self._combo_index_for_data(combo, reference.target_card_id)
                    )
                elif isinstance(reference, UnresolvedCardReference):
                    name = reference.target_name or "Unknown card"
                    combo.addItem(f"Missing: {name}", reference)
                    combo.setCurrentIndex(combo.count() - 1)
                else:
                    combo.setCurrentIndex(0)
        self.additional_reference_combo.setEnabled(bool(references))

    def _reference_changed(self, position: int, index: int) -> None:
        if self._rendering or index < 0:
            return
        card = self._selected_card()
        if card is None:
            return
        combo = (
            self.reference_combo
            if position == 1
            else self.additional_reference_combo
        )
        value = combo.itemData(index)
        if isinstance(value, UUID):
            reference = ResolvedCardReference(target_card_id=value)
        elif isinstance(value, UnresolvedCardReference):
            reference = value
        else:
            reference = None
        current = (
            card.active_revision.references[position - 1]
            if position <= len(card.active_revision.references)
            else None
        )
        if reference == current:
            return
        self._render_inputs_changed()
        self._execute(
            SetRevisionReferenceCommand(
                card_id=card.id,
                revision_id=card.active_revision.id,
                reference=reference,
                position=position,
            ),
            error_label=self.reference_error,
            undo_message="Reference changed",
        )

    def _render_resolution(
        self,
        document: Stack,
        revision: CardRevision,
    ) -> None:
        with QSignalBlocker(self.resolution_combo):
            self.resolution_combo.clear()
            for resolution in GenerateResolution:
                width, height = output_dimensions(
                    resolution,
                    document.aspect_ratio,
                )
                self.resolution_combo.addItem(
                    f"{resolution.value} ({width} x {height})",
                    resolution,
                )
            self.resolution_combo.setCurrentIndex(
                self._combo_index_for_data(
                    self.resolution_combo,
                    revision.generate_resolution,
                )
            )

    def _revision_resolution_changed(self, index: int) -> None:
        if self._rendering or index < 0:
            return
        card = self._selected_card()
        if card is None:
            return
        try:
            resolution = GenerateResolution(
                self.resolution_combo.itemData(index)
            )
        except (TypeError, ValueError):
            self.render(self.controller.document, self.selected_card_id)
            return
        if resolution is card.active_revision.generate_resolution:
            return
        self._render_inputs_changed()
        self._execute(
            SetRevisionGenerateResolutionCommand(
                card_id=card.id,
                revision_id=card.active_revision.id,
                resolution=resolution,
            ),
            undo_message="Generate resolution changed",
        )

    def _render_refine(
        self,
        document: Stack,
        revision: CardRevision,
        *,
        source_size: tuple[int, int] | None,
    ) -> None:
        current_resolution = self.refine_resolution_combo.currentData()
        self._refine_source_size = source_size
        available: tuple[GenerateResolution, ...] = ()
        error = ""
        if revision.background is None:
            error = "Generate an image before refining."
        elif source_size is None:
            error = "The current image is unavailable or unreadable."
        else:
            available = higher_output_resolutions(
                source_size[0],
                source_size[1],
                document.aspect_ratio,
            )
            if not available:
                error = (
                    "The current image is already at the maximum Refine "
                    "resolution."
                )
        with QSignalBlocker(self.refine_resolution_combo):
            self.refine_resolution_combo.clear()
            for resolution in available:
                width, height = output_dimensions(
                    resolution,
                    document.aspect_ratio,
                )
                self.refine_resolution_combo.addItem(
                    f"{resolution.value} ({width} x {height})",
                    resolution,
                )
            selected_index = self._combo_index_for_data(
                self.refine_resolution_combo,
                current_resolution,
            )
            self.refine_resolution_combo.setCurrentIndex(
                selected_index if selected_index >= 0 else 0
            )
        self.refine_resolution_combo.setEnabled(bool(available))
        self._set_error(self.refine_error, error)
        resolution = self.refine_resolution_combo.currentData()
        try:
            transformation = RefineTransformation(
                self.refine_transformation_combo.currentData()
            )
        except (TypeError, ValueError):
            transformation = None
        lineage_count = (
            len(image_edit_lineage(revision.background.provenance))
            if revision.background is not None
            else 0
        )
        style_suffix = " + Style" if revision.style_id is not None else ""
        lineage_suffix = (
            f" + {lineage_count} accepted Edit"
            f"{'s' if lineage_count != 1 else ''}"
            if lineage_count
            else ""
        )
        using = [
            f"Using: Current image + Description{style_suffix}{lineage_suffix}",
        ]
        if isinstance(transformation, RefineTransformation):
            using.append(
                f"Transformation: {transformation.value.title()} "
                f"({transformation.strength:.2f})"
            )
        if isinstance(resolution, GenerateResolution):
            width, height = output_dimensions(
                resolution,
                document.aspect_ratio,
            )
            using.append(
                f"Resolution: {resolution.value} square-equivalent "
                f"({width} x {height})"
            )
        self._refine_using_text = "\n".join(using)
        self._refresh_generation_tooltips()

    def _request_refine_background(self) -> None:
        try:
            transformation = RefineTransformation(
                self.refine_transformation_combo.currentData()
            )
        except (TypeError, ValueError):
            transformation = None
        resolution = self.refine_resolution_combo.currentData()
        if not isinstance(transformation, RefineTransformation):
            self._set_error(
                self.refine_error,
                "Select a supported Refine transformation.",
            )
            return
        if not isinstance(resolution, GenerateResolution):
            if not self.refine_error.text():
                self._set_error(
                    self.refine_error,
                    "Select a higher Refine output resolution.",
                )
            return
        self.refine_background_requested.emit(transformation, resolution)

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
        self.hotspot_when_label.setVisible(has_interaction)
        self.hotspot_when_panel.setVisible(has_interaction)
        self.hotspot_then_label.setVisible(has_interaction)
        self.hotspot_then_panel.setVisible(has_interaction)
        self.hotspot_summary.setVisible(has_interaction)
        self.condition_table.setEnabled(has_interaction)
        self.add_condition_button.setEnabled(has_interaction)
        self.key_change_table.setEnabled(has_interaction)
        self.add_key_change_button.setEnabled(has_interaction)
        self.hotspot_destination_combo.setEnabled(has_interaction)
        self.delete_hotspot_button.setEnabled(has_interaction)
        current_row = self.hotspot_list.currentRow()
        self.move_hotspot_up_button.setEnabled(has_interaction and current_row > 0)
        self.move_hotspot_down_button.setEnabled(
            has_interaction and current_row < self.hotspot_list.count() - 1
        )
        self._render_condition_rows(document, interaction)
        self._render_key_change_rows(document, interaction)
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
            self._finish_rule_table(
                self.condition_table,
                self.no_conditions_label,
                (),
            )
            return
        rows = (
            *(("Has", key_id) for key_id in interaction.conditions.requires),
            *(("Lacks", key_id) for key_id in interaction.conditions.forbids),
        )
        row_heights: list[int] = []
        for row, (state, key_id) in enumerate(rows):
            self.condition_table.insertRow(row)
            key_combo = self._key_combo(document, key_id)
            state_combo = QComboBox()
            state_combo.addItem("Has", "requires")
            state_combo.addItem("Lacks", "forbids")
            state_combo.setCurrentIndex(0 if state == "Has" else 1)
            remove_button = _compact_text_button(
                "−",
                object_name=f"removeConditionButton{row}",
                accessible_name="Remove condition",
                tooltip="Remove condition",
                prominent=False,
            )
            control_extent = max(
                key_combo.sizeHint().height(),
                state_combo.sizeHint().height(),
            )
            remove_button.setFixedSize(20, 20)
            row_height = control_extent + 4
            self.condition_table.setRowHeight(row, row_height)
            row_heights.append(row_height)
            self.condition_table.setCellWidget(row, 0, state_combo)
            self.condition_table.setCellWidget(row, 1, key_combo)
            self.condition_table.setCellWidget(
                row,
                2,
                _centered_cell_widget(remove_button),
            )
            role = "requires" if state == "Has" else "forbids"
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
        self._finish_rule_table(
            self.condition_table,
            self.no_conditions_label,
            tuple(row_heights),
        )

    def _render_key_change_rows(
        self,
        document: Stack,
        interaction: Interaction | None,
    ) -> None:
        self.key_change_table.setRowCount(0)
        if interaction is None:
            self._finish_rule_table(
                self.key_change_table,
                self.no_key_changes_label,
                (),
            )
            return
        rows = (
            *(("Lose", key_id) for key_id in interaction.key_changes.remove),
            *(("Gain", key_id) for key_id in interaction.key_changes.grant),
        )
        row_heights: list[int] = []
        for row, (change, key_id) in enumerate(rows):
            self.key_change_table.insertRow(row)
            change_combo = QComboBox()
            change_combo.addItem("Lose", "remove")
            change_combo.addItem("Gain", "grant")
            change_combo.setCurrentIndex(0 if change == "Lose" else 1)
            key_combo = self._key_combo(document, key_id)
            remove_button = _compact_text_button(
                "−",
                object_name=f"removeKeyChangeButton{row}",
                accessible_name="Remove key change",
                tooltip="Remove key change",
                prominent=False,
            )
            control_extent = max(
                change_combo.sizeHint().height(),
                key_combo.sizeHint().height(),
            )
            remove_button.setFixedSize(20, 20)
            row_height = control_extent + 4
            self.key_change_table.setRowHeight(row, row_height)
            row_heights.append(row_height)
            self.key_change_table.setCellWidget(row, 0, change_combo)
            self.key_change_table.setCellWidget(row, 1, key_combo)
            self.key_change_table.setCellWidget(
                row,
                2,
                _centered_cell_widget(remove_button),
            )
            role = "remove" if change == "Lose" else "grant"
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
        self._finish_rule_table(
            self.key_change_table,
            self.no_key_changes_label,
            tuple(row_heights),
        )

    @staticmethod
    def _finish_rule_table(
        table: QTableWidget,
        placeholder: QLabel,
        row_heights: tuple[int, ...],
    ) -> None:
        has_rows = bool(row_heights)
        table.setVisible(has_rows)
        placeholder.setVisible(not has_rows)
        if not has_rows:
            return
        remove_width = max(
            table.cellWidget(row, 2).findChild(QToolButton).width()
            for row in range(table.rowCount())
        )
        table.setColumnWidth(2, remove_width + 4)
        visible_rows = min(len(row_heights), 3)
        content_height = (
            table.horizontalHeader().sizeHint().height() + sum(row_heights[:visible_rows]) + 2
        )
        table.setFixedHeight(content_height)
        table.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
            if len(row_heights) > visible_rows
            else Qt.ScrollBarPolicy.ScrollBarAlwaysOff
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
                f"has {document.key_by_id(key_id).name}"
                for key_id in interaction.conditions.requires
            ),
            *(
                f"lacks {document.key_by_id(key_id).name}"
                for key_id in interaction.conditions.forbids
            ),
        ]
        action_parts: list[str] = []
        action_parts.extend(
            f"lose {document.key_by_id(key_id).name}"
            for key_id in interaction.key_changes.remove
        )
        action_parts.extend(
            f"gain {document.key_by_id(key_id).name}"
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
            f"When the runner {' and '.join(condition_parts)}"
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
        if interaction is None:
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
        except (CommandError, DocumentMutationBlockedError, ValidationError) as error:
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
