"""Minimal revision-local Background and Hotspots inspector."""

from __future__ import annotations

from uuid import UUID

from pydantic import ValidationError
from PySide6.QtCore import QItemSelectionModel, QSignalBlocker, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QFocusEvent, QStandardItemModel
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QStyle,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from hotcards.application.commands import (
    AddInteractionCommand,
    ChangeHotspotDestinationCommand,
    ChangeHotspotSoundCommand,
    CommandError,
    DeleteInteractionCommand,
    DocumentCommand,
    EditRevisionDescriptionCommand,
    ReorderHotspotCommand,
    SetHotspotConditionsCommand,
    SetHotspotKeyChangesCommand,
    SetRevisionGenerateOutputSizeCommand,
    SetRevisionReferenceCommand,
    SetRevisionStyleCommand,
)
from hotcards.application.document_controller import (
    DocumentController,
    DocumentMutationBlockedError,
)
from hotcards.domain.image_dimensions import (
    AspectRatio,
    ResolutionTier,
    higher_output_tiers,
    output_dimensions,
    validate_exact_output_dimensions,
)
from hotcards.domain.models import (
    Card,
    CardRevision,
    CurrentSourceSize,
    EditOutputSize,
    ExactOutputSize,
    GenerateOutputSize,
    HotspotConditions,
    HotspotKeyChanges,
    Interaction,
    PresetOutputSize,
    RefineTransformation,
    ResolvedCardReference,
    Stack,
    UnresolvedCardReference,
    selected_output_dimensions,
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


class _AdjacentCardComboBox(QComboBox):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._popup_anchor_data: object | None = None

    def set_popup_anchor_data(self, value: object | None) -> None:
        self._popup_anchor_data = value

    def showPopup(self) -> None:
        super().showPopup()
        if self.currentData() is not None or self._popup_anchor_data is None:
            return
        index = self.findData(self._popup_anchor_data)
        if index < 0:
            return
        model_index = self.model().index(
            index,
            self.modelColumn(),
            self.rootModelIndex(),
        )
        self.view().selectionModel().setCurrentIndex(
            model_index,
            QItemSelectionModel.SelectionFlag.ClearAndSelect,
        )
        self.view().scrollTo(
            model_index,
            QAbstractItemView.ScrollHint.PositionAtCenter,
        )


def _adjacent_card_id(
    document: Stack,
    current_card_id: UUID,
    eligible_card_ids: set[UUID],
) -> UUID | None:
    current_index = next(
        (
            index
            for index, candidate in enumerate(document.cards)
            if candidate.id == current_card_id
        ),
        -1,
    )
    if current_index < 0:
        return None
    for adjacent_index in (current_index + 1, current_index - 1):
        if (
            0 <= adjacent_index < len(document.cards)
            and document.cards[adjacent_index].id in eligible_card_ids
        ):
            return document.cards[adjacent_index].id
    return None


def _rule_panel(object_name: str) -> tuple[QGroupBox, QVBoxLayout]:
    panel = QGroupBox()
    panel.setObjectName(object_name)
    layout = QVBoxLayout(panel)
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


_SOURCE_SIMILARITY_TOOLTIPS = {
    RefineTransformation.REIMAGINE: (
        "Give the Description the most freedom to change the current image's "
        "content and composition."
    ),
    RefineTransformation.BALANCED: ("Balance the Description with the current image."),
    RefineTransformation.PRESERVE: (
        "Keep the current image's content and composition as close as possible "
        "while applying the Description."
    ),
}


def _tier_label(tier: ResolutionTier) -> str:
    return tier.label


def _current_tier(
    source_size: tuple[int, int] | None,
    aspect_ratio: AspectRatio,
) -> ResolutionTier | None:
    if source_size is None:
        return None
    return next(
        (tier for tier in ResolutionTier if output_dimensions(tier, aspect_ratio) == source_size),
        None,
    )


def _sort_size_rows(
    rows: list[tuple[str, object, tuple[int, int]]],
) -> list[tuple[str, object, tuple[int, int]]]:
    return sorted(
        rows,
        key=lambda row: (
            row[2][0] * row[2][1],
            row[2][0],
            row[2][1],
            row[0],
        ),
    )


def _exact_size_error(
    source_size: tuple[int, int] | None,
    aspect_ratio: AspectRatio,
) -> str | None:
    if source_size is None:
        return None
    try:
        validate_exact_output_dimensions(
            source_size[0],
            source_size[1],
            aspect_ratio,
        )
    except ValueError as error:
        return str(error)
    return None


def _add_resolution_item(
    combo: QComboBox,
    *,
    label: str,
    value: object,
    dimensions: tuple[int, int],
    unavailable_reason: str | None = None,
) -> None:
    combo.addItem(label, value)
    model = combo.model()
    if not isinstance(model, QStandardItemModel):
        return
    item = model.item(combo.count() - 1)
    if item is None:
        return
    dimensions_text = f"{dimensions[0]} × {dimensions[1]}"
    if unavailable_reason is None:
        item.setToolTip(dimensions_text)
        return
    item.setEnabled(False)
    item.setToolTip(f"{dimensions_text}\nUnavailable: {unavailable_reason}")


def _sync_combo_tooltip(combo: QComboBox) -> None:
    tooltip = combo.itemData(combo.currentIndex(), Qt.ItemDataRole.ToolTipRole)
    combo.setToolTip(tooltip if isinstance(tooltip, str) else "")


def _first_selectable_combo_index(combo: QComboBox) -> int:
    model = combo.model()
    if not isinstance(model, QStandardItemModel):
        return 0 if combo.count() else -1
    for index in range(combo.count()):
        item = model.item(index)
        if item is not None and item.isEnabled():
            return index
    return -1


class Inspector(QWidget):
    """Render the selected active revision and issue typed commands."""

    document_changed = Signal(object)
    generate_background_requested = Signal()
    refine_background_requested = Signal(object, object)
    edit_background_requested = Signal(str, object)
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
        self._generate_using_text = ""
        self._generate_reason = ""
        self._refine_using_text = ""
        self._refine_reason = ""
        self._refine_source_size: tuple[int, int] | None = None
        self._edit_using_text = ""
        self._edit_reason = ""
        self._edit_source_size: tuple[int, int] | None = None
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
        self._build_transform_tab()
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

        self.description_label = QLabel("Description")
        self.description_label.setObjectName("descriptionLabel")
        layout.addWidget(self.description_label)
        self.description_edit = _CommitPlainTextEdit()
        self.description_edit.setObjectName("descriptionEdit")
        self.description_edit.setPlaceholderText(
            "Describe the image to generate. Refer to selected References as image 1 and image 2."
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
        self.style_selection_error = QLabel()
        self.style_selection_error.setObjectName("styleSelectionValidationError")
        self.style_selection_error.setWordWrap(True)
        self.style_selection_error.setVisible(False)
        layout.addWidget(self.style_selection_error)

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
        self.reference_combo = _AdjacentCardComboBox()
        self.reference_combo.setObjectName("referenceCombo")
        self.reference_combo.setAccessibleName("Reference card 1")
        self.reference_combo.setToolTip(
            "Primary Reference; its image is sent to the model as image 1"
        )
        first_reference_row.addWidget(self.reference_combo, 1)
        reference_layout.addLayout(first_reference_row)
        second_reference_row = QHBoxLayout()
        second_reference_row.addWidget(QLabel("2."))
        self.additional_reference_combo = _AdjacentCardComboBox()
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
        self.resolution_combo.setAccessibleName("Generate output size")
        self.resolution_combo.setToolTip("Generate output tier or exact current-image size")
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

    def _build_transform_tab(self) -> None:
        scroll = QScrollArea()
        scroll.setObjectName("transformInspectorTab")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        page = QWidget()
        page.setObjectName("transformInspectorContent")
        layout = QVBoxLayout(page)

        self.reinterpret_section_label = QLabel("Reinterpret")
        self.reinterpret_section_label.setObjectName("reinterpretSectionLabel")
        layout.addWidget(self.reinterpret_section_label)
        refine_group = QGroupBox()
        refine_group.setObjectName("refineTransformGroup")
        refine_layout = QVBoxLayout(refine_group)

        self.refine_transformation_label = QLabel("Source Similarity")
        refine_layout.addWidget(self.refine_transformation_label)
        self.refine_transformation_combo = QComboBox()
        self.refine_transformation_combo.setObjectName("refineTransformationCombo")
        self.refine_transformation_combo.setAccessibleName("Reinterpret source similarity")
        for transformation in RefineTransformation:
            self.refine_transformation_combo.addItem(
                transformation.value.title(),
                transformation,
            )
            self.refine_transformation_combo.setItemData(
                self.refine_transformation_combo.count() - 1,
                _SOURCE_SIMILARITY_TOOLTIPS[transformation],
                Qt.ItemDataRole.ToolTipRole,
            )
        self.refine_transformation_combo.setCurrentIndex(
            self._combo_index_for_data(
                self.refine_transformation_combo,
                RefineTransformation.BALANCED,
            )
        )
        _sync_combo_tooltip(self.refine_transformation_combo)
        refine_layout.addWidget(self.refine_transformation_combo)

        self.refine_resolution_label = QLabel("Resolution")
        refine_layout.addWidget(self.refine_resolution_label)
        self.refine_resolution_combo = QComboBox()
        self.refine_resolution_combo.setObjectName("refineResolutionCombo")
        self.refine_resolution_combo.setAccessibleName("Reinterpret resolution")
        refine_layout.addWidget(self.refine_resolution_combo)
        self.refine_error = QLabel()
        self.refine_error.setObjectName("refineValidationError")
        self.refine_error.setWordWrap(True)
        self.refine_error.setVisible(False)
        refine_layout.addWidget(self.refine_error)

        self.refine_background_button = QPushButton("Reinterpret")
        self.refine_background_button.setObjectName("refineBackgroundButton")
        refine_layout.addWidget(self.refine_background_button)
        self._focus_commit_targets.add(self.refine_background_button)
        layout.addWidget(refine_group)

        layout.addSpacing(8)
        self.edit_section_label = QLabel("Edit")
        self.edit_section_label.setObjectName("editSectionLabel")
        layout.addWidget(self.edit_section_label)
        edit_group = QGroupBox()
        edit_group.setObjectName("editTransformGroup")
        edit_layout = QVBoxLayout(edit_group)
        self.edit_instruction_label = QLabel("Edit Instruction")
        edit_layout.addWidget(self.edit_instruction_label)
        self.edit_instruction_edit = QPlainTextEdit()
        self.edit_instruction_edit.setObjectName("editInstructionEdit")
        self.edit_instruction_edit.setAccessibleName("Edit Instruction")
        self.edit_instruction_edit.setPlaceholderText(
            "Describe the change to make, everything else will be preserved"
        )
        self.edit_instruction_edit.setMaximumHeight(110)
        edit_layout.addWidget(self.edit_instruction_edit)
        self.edit_instruction_error = QLabel()
        self.edit_instruction_error.setObjectName("editInstructionValidationError")
        self.edit_instruction_error.setWordWrap(True)
        self.edit_instruction_error.setVisible(False)
        edit_layout.addWidget(self.edit_instruction_error)

        self.edit_resolution_label = QLabel("Resolution")
        edit_layout.addWidget(self.edit_resolution_label)
        self.edit_resolution_combo = QComboBox()
        self.edit_resolution_combo.setObjectName("editResolutionCombo")
        self.edit_resolution_combo.setAccessibleName("Edit resolution")
        edit_layout.addWidget(self.edit_resolution_combo)
        self.edit_output_error = QLabel()
        self.edit_output_error.setObjectName("editOutputValidationError")
        self.edit_output_error.setWordWrap(True)
        self.edit_output_error.setVisible(False)
        edit_layout.addWidget(self.edit_output_error)

        self.edit_background_button = QPushButton("Edit")
        self.edit_background_button.setObjectName("editBackgroundButton")
        edit_layout.addWidget(self.edit_background_button)
        self._focus_commit_targets.add(self.edit_background_button)
        layout.addWidget(edit_group)
        layout.addStretch(1)

        scroll.setWidget(page)
        self._transform_tab_index = self.inspector_tabs.addTab(scroll, "Transform")

    def _build_hotspots_tab(self) -> None:
        scroll = QScrollArea()
        scroll.setObjectName("hotspotsInspectorTab")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        page = QWidget()
        page.setObjectName("hotspotsInspectorContent")
        layout = QVBoxLayout(page)

        self.hotspots_placeholder = QLabel("No hotspots yet.")
        self.hotspots_placeholder.setWordWrap(True)
        layout.addWidget(self.hotspots_placeholder)
        self.hotspot_list = QListWidget()
        self.hotspot_list.setObjectName("hotspotList")
        self.hotspot_list.setWordWrap(True)
        self.hotspot_list.setFixedHeight(120)
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
        controls.addWidget(self.delete_hotspot_button)
        controls.addWidget(self.add_hotspot_button)
        layout.addLayout(controls)

        self.hotspot_when_label = QLabel("When")
        self.hotspot_when_label.setObjectName("hotspotWhenLabel")
        layout.addWidget(self.hotspot_when_label)
        self.hotspot_when_panel, when_layout = _rule_panel("hotspotWhenPanel")
        self.condition_controls_layout = QVBoxLayout()
        self.condition_controls_layout.setContentsMargins(0, 0, 0, 0)
        self.condition_controls_layout.setSpacing(3)
        self.condition_rows_widget = QWidget()
        self.condition_rows_widget.setObjectName("hotspotConditionRows")
        self.condition_rows_layout = QVBoxLayout(self.condition_rows_widget)
        self.condition_rows_layout.setContentsMargins(0, 0, 0, 0)
        self.condition_controls_layout.addWidget(self.condition_rows_widget)
        self.add_condition_button = _compact_text_button(
            "+",
            object_name="addConditionButton",
            accessible_name="Add condition",
            tooltip="Add a condition",
            prominent=False,
        )
        self.add_condition_button.setFixedSize(20, 20)
        self.condition_add_row = QWidget()
        self.condition_add_row.setObjectName("addConditionRow")
        condition_add_layout = QHBoxLayout(self.condition_add_row)
        condition_add_layout.setContentsMargins(0, 0, 0, 0)
        condition_add_layout.setSpacing(4)
        condition_add_layout.addWidget(self.add_condition_button)
        self.add_condition_label = QLabel("Key condition")
        condition_add_layout.addWidget(self.add_condition_label)
        condition_add_layout.addStretch(1)
        self.condition_controls_layout.addWidget(self.condition_add_row)
        when_layout.addLayout(self.condition_controls_layout)
        layout.addWidget(self.hotspot_when_panel)

        self.hotspot_then_label = QLabel("Then")
        self.hotspot_then_label.setObjectName("hotspotThenLabel")
        layout.addWidget(self.hotspot_then_label)
        self.hotspot_then_panel, then_layout = _rule_panel("hotspotThenPanel")
        self.key_change_controls_layout = QVBoxLayout()
        self.key_change_controls_layout.setContentsMargins(0, 0, 0, 0)
        self.key_change_controls_layout.setSpacing(3)
        self.key_change_rows_widget = QWidget()
        self.key_change_rows_widget.setObjectName("hotspotKeyChangeRows")
        self.key_change_rows_layout = QVBoxLayout(self.key_change_rows_widget)
        self.key_change_rows_layout.setContentsMargins(0, 0, 0, 0)
        self.key_change_controls_layout.addWidget(self.key_change_rows_widget)
        self.add_key_change_button = _compact_text_button(
            "+",
            object_name="addKeyChangeButton",
            accessible_name="Add key change",
            tooltip="Add a key change",
            prominent=False,
        )
        self.add_key_change_button.setFixedSize(20, 20)
        self.key_change_add_row = QWidget()
        self.key_change_add_row.setObjectName("addKeyChangeRow")
        key_change_add_layout = QHBoxLayout(self.key_change_add_row)
        key_change_add_layout.setContentsMargins(0, 0, 0, 0)
        key_change_add_layout.setSpacing(4)
        key_change_add_layout.addWidget(self.add_key_change_button)
        self.add_key_change_label = QLabel("Key change")
        key_change_add_layout.addWidget(self.add_key_change_label)
        key_change_add_layout.addStretch(1)
        self.key_change_controls_layout.addWidget(self.key_change_add_row)
        then_layout.addLayout(self.key_change_controls_layout)

        self.hotspot_target_label = QLabel("Go to")
        self.hotspot_target_label.setObjectName("hotspotTargetLabel")
        then_layout.addWidget(self.hotspot_target_label)
        self.hotspot_destination_combo = _AdjacentCardComboBox()
        self.hotspot_destination_combo.setObjectName("hotspotDestinationCombo")
        self.hotspot_destination_combo.setAccessibleName("Hotspot destination")
        then_layout.addWidget(self.hotspot_destination_combo)
        self.hotspot_sound_label = QLabel("Play")
        self.hotspot_sound_label.setObjectName("hotspotSoundLabel")
        then_layout.addWidget(self.hotspot_sound_label)
        self.hotspot_sound_combo = QComboBox()
        self.hotspot_sound_combo.setObjectName("hotspotSoundCombo")
        self.hotspot_sound_combo.setAccessibleName("Hotspot Sound")
        then_layout.addWidget(self.hotspot_sound_combo)
        layout.addWidget(self.hotspot_then_panel)

        self.hotspot_error = QLabel()
        self.hotspot_error.setObjectName("hotspotValidationError")
        self.hotspot_error.setWordWrap(True)
        self.hotspot_error.setVisible(False)
        layout.addWidget(self.hotspot_error)
        layout.addStretch(1)
        scroll.setWidget(page)
        self.hotspot_scroll = scroll
        self._hotspots_tab_index = self.inspector_tabs.addTab(scroll, "Hotspots")

    @property
    def hotspots_active(self) -> bool:
        return self.inspector_tabs.currentIndex() == self._hotspots_tab_index

    def _connect_signals(self) -> None:
        self.description_edit.editing_finished.connect(self._description_editing_finished)
        self.description_edit.textChanged.connect(self._render_inputs_changed)
        self.generate_background_button.clicked.connect(self._request_generate_background)
        self.refine_background_button.clicked.connect(self._request_refine_background)
        self.refine_transformation_combo.currentIndexChanged.connect(
            self._refine_similarity_changed
        )
        self.refine_resolution_combo.currentIndexChanged.connect(self._refine_output_changed)
        self.edit_instruction_edit.textChanged.connect(self._edit_inputs_changed)
        self.edit_resolution_combo.currentIndexChanged.connect(self._edit_output_changed)
        self.edit_background_button.clicked.connect(self._request_edit_background)
        self.style_combo.currentIndexChanged.connect(self._revision_style_changed)
        self.reference_combo.currentIndexChanged.connect(
            lambda index: self._reference_changed(1, index)
        )
        self.additional_reference_combo.currentIndexChanged.connect(
            lambda index: self._reference_changed(2, index)
        )
        self.resolution_combo.currentIndexChanged.connect(self._revision_resolution_changed)
        self.hotspot_list.currentItemChanged.connect(self._hotspot_selection_changed)
        self.move_hotspot_up_button.clicked.connect(lambda: self._move_hotspot(-1))
        self.move_hotspot_down_button.clicked.connect(lambda: self._move_hotspot(1))
        self.add_hotspot_button.clicked.connect(self._add_hotspot)
        self.delete_hotspot_button.clicked.connect(self._delete_hotspot)
        self.add_condition_button.clicked.connect(self._add_condition)
        self.add_key_change_button.clicked.connect(self._add_key_change)
        self.hotspot_destination_combo.currentIndexChanged.connect(self._destination_changed)
        self.hotspot_sound_combo.currentIndexChanged.connect(self._sound_changed)

    def _render_inputs_changed(self) -> None:
        if not self._rendering:
            self.render_inputs_changed.emit()

    def _edit_inputs_changed(self) -> None:
        if self._rendering:
            return
        self._set_error(self.edit_instruction_error, "")
        self._set_error(self.edit_output_error, "")
        self._render_edit_tooltip()
        self.render_inputs_changed.emit()

    def _refine_similarity_changed(self, _index: int) -> None:
        _sync_combo_tooltip(self.refine_transformation_combo)
        self._render_inputs_changed()

    def _refine_output_changed(self, _index: int) -> None:
        if self._rendering:
            return
        if self.refine_resolution_combo.currentData() is None:
            fallback = _first_selectable_combo_index(self.refine_resolution_combo)
            if fallback >= 0:
                with QSignalBlocker(self.refine_resolution_combo):
                    self.refine_resolution_combo.setCurrentIndex(fallback)
        _sync_combo_tooltip(self.refine_resolution_combo)
        self._render_inputs_changed()

    def _edit_output_changed(self, _index: int) -> None:
        if self._rendering:
            return
        if self.edit_resolution_combo.currentData() is None:
            fallback = _first_selectable_combo_index(self.edit_resolution_combo)
            if fallback >= 0:
                with QSignalBlocker(self.edit_resolution_combo):
                    self.edit_resolution_combo.setCurrentIndex(fallback)
        _sync_combo_tooltip(self.edit_resolution_combo)
        self._edit_inputs_changed()

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
                self._set_error(self.edit_instruction_error, "")
                self._set_error(self.edit_output_error, "")
                self._set_error(self.edit_instruction_error, "")
                self._set_error(self.edit_output_error, "")
                self._set_error(self.style_selection_error, "")
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
            self._render_reference(document, card, revision)
            self._render_resolution(
                document,
                revision,
                source_size=refine_source_size,
                select_current=not same_revision,
            )
            self._render_description_workflow(document, card, revision)
            self._render_refine(
                document,
                revision,
                source_size=refine_source_size,
                select_current=not same_revision,
            )
            self._render_edit(
                document,
                revision,
                source_size=refine_source_size,
                select_current=not same_revision,
            )
            self._render_hotspots(document, revision)
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
        return self.commit_revision_metadata(render_change=render_change)

    def has_description_input(self) -> bool:
        card = self._selected_card()
        if card is None:
            return False
        return bool(self.description_edit.toPlainText().strip())

    def has_edit_instruction_input(self) -> bool:
        return bool(self.edit_instruction_edit.toPlainText().strip())

    def selected_edit_output_size(self) -> EditOutputSize | None:
        return self._selected_edit_output_size()

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
        selected_output_size = self.resolution_combo.currentData()
        if not isinstance(
            selected_output_size,
            (PresetOutputSize, ExactOutputSize),
        ):
            selected_output_size = revision.generate_output_size
        width, height = selected_output_dimensions(
            selected_output_size,
            document.aspect_ratio,
        )
        if isinstance(selected_output_size, PresetOutputSize):
            output_label = selected_output_size.tier.label
        else:
            output_label = "Exact"
        self._generate_using_text = (
            f"Using: Description{style_suffix}{reference_suffix}\n"
            f"Resolution: {output_label} ({width} × {height})"
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
        self._refine_reason = "" if can_refine and not busy else refine_reason
        self.refine_background_button.setEnabled(can_refine and not busy)
        self.refine_background_button.setText("Reinterpreting…" if refining else "Reinterpret")
        self._refresh_generation_tooltips()

    def set_edit_capabilities(
        self,
        *,
        can_edit: bool,
        edit_reason: str,
        busy: bool,
        editing: bool,
    ) -> None:
        self._edit_reason = "" if can_edit and not busy else edit_reason
        self.edit_background_button.setEnabled(can_edit and not busy)
        self.edit_background_button.setText("Editing…" if editing else "Edit")
        self._refresh_generation_tooltips()

    def set_edit_error(self, message: str) -> None:
        self._set_error(self.edit_instruction_error, message)

    def clear_edit_instruction(self) -> None:
        with QSignalBlocker(self.edit_instruction_edit):
            self.edit_instruction_edit.clear()
        self._set_error(self.edit_instruction_error, "")
        self._render_edit_tooltip()

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
        self.edit_background_button.setToolTip(
            self._tooltip_with_using(
                self._edit_reason,
                self._edit_using_text,
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
        self._set_error(self.style_selection_error, "")
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
            error_label=self.style_selection_error,
            undo_message="Style changed",
        )

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
                eligible_card_ids = {
                    value
                    for index in range(combo.count())
                    if isinstance((value := combo.itemData(index)), UUID)
                }
                combo.set_popup_anchor_data(
                    _adjacent_card_id(
                        document,
                        card.id,
                        eligible_card_ids,
                    )
                )
        self.additional_reference_combo.setEnabled(bool(references))

    def _reference_changed(self, position: int, index: int) -> None:
        if self._rendering or index < 0:
            return
        card = self._selected_card()
        if card is None:
            return
        combo = self.reference_combo if position == 1 else self.additional_reference_combo
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
        *,
        source_size: tuple[int, int] | None,
        select_current: bool,
    ) -> None:
        previous_selection = self.resolution_combo.currentData()
        current_tier = _current_tier(source_size, document.aspect_ratio)
        current_error = _exact_size_error(source_size, document.aspect_ratio)
        rows: list[tuple[str, object, tuple[int, int]]] = []
        for tier in ResolutionTier:
            dimensions = output_dimensions(tier, document.aspect_ratio)
            output_size: GenerateOutputSize = PresetOutputSize(tier=tier)
            if (
                isinstance(revision.generate_output_size, ExactOutputSize)
                and (
                    revision.generate_output_size.width,
                    revision.generate_output_size.height,
                )
                == dimensions
            ):
                output_size = revision.generate_output_size
            rows.append(
                (
                    _tier_label(tier),
                    output_size,
                    dimensions,
                )
            )
        exact_sizes: dict[tuple[int, int], tuple[ExactOutputSize, bool]] = {}
        if isinstance(revision.generate_output_size, ExactOutputSize):
            exact_sizes[
                (
                    revision.generate_output_size.width,
                    revision.generate_output_size.height,
                )
            ] = (revision.generate_output_size, False)
        if source_size is not None and current_tier is None and current_error is None:
            current_size = ExactOutputSize(
                width=source_size[0],
                height=source_size[1],
            )
            exact_sizes[source_size] = (current_size, True)
        for output_size, current in exact_sizes.values():
            dimensions = (output_size.width, output_size.height)
            if any(existing[2] == dimensions for existing in rows):
                continue
            label = "Current" if current else "Exact"
            rows.append(
                (
                    label,
                    output_size,
                    dimensions,
                )
            )
        if source_size is not None and current_tier is None and current_error is not None:
            rows.append(
                (
                    "Current",
                    None,
                    source_size,
                )
            )
        sorted_rows = _sort_size_rows(rows)
        with QSignalBlocker(self.resolution_combo):
            self.resolution_combo.clear()
            for label, output_size, dimensions in sorted_rows:
                _add_resolution_item(
                    self.resolution_combo,
                    label=label,
                    value=output_size,
                    dimensions=dimensions,
                    unavailable_reason=(
                        f"{current_error}. Choose a named tier."
                        if output_size is None and current_error is not None
                        else None
                    ),
                )
            if select_current and source_size is not None and current_error is None:
                selected_index = next(
                    (
                        index
                        for index, (_label, output_size, dimensions) in enumerate(sorted_rows)
                        if output_size is not None and dimensions == source_size
                    ),
                    -1,
                )
            elif (
                source_size is not None
                and isinstance(previous_selection, (PresetOutputSize, ExactOutputSize))
                and selected_output_dimensions(
                    previous_selection,
                    document.aspect_ratio,
                )
                == source_size
            ):
                selected_index = self._combo_index_for_data(
                    self.resolution_combo,
                    previous_selection,
                )
            else:
                selected_index = self._combo_index_for_data(
                    self.resolution_combo,
                    revision.generate_output_size,
                )
            self.resolution_combo.setCurrentIndex(
                selected_index
                if selected_index >= 0
                else _first_selectable_combo_index(self.resolution_combo)
            )
        _sync_combo_tooltip(self.resolution_combo)

    def _revision_resolution_changed(self, index: int) -> None:
        if self._rendering or index < 0:
            return
        card = self._selected_card()
        if card is None:
            return
        output_size = self.resolution_combo.itemData(index)
        if not isinstance(output_size, (PresetOutputSize, ExactOutputSize)):
            self.render(self.controller.document, self.selected_card_id)
            return
        _sync_combo_tooltip(self.resolution_combo)
        if output_size == card.active_revision.generate_output_size:
            return
        self._render_inputs_changed()
        self._execute(
            SetRevisionGenerateOutputSizeCommand(
                card_id=card.id,
                revision_id=card.active_revision.id,
                output_size=output_size,
            )
        )

    def _request_generate_background(self) -> None:
        card = self._selected_card()
        output_size = self.resolution_combo.currentData()
        if card is None or not isinstance(
            output_size,
            (PresetOutputSize, ExactOutputSize),
        ):
            return
        if output_size != card.active_revision.generate_output_size and not self._execute(
            SetRevisionGenerateOutputSizeCommand(
                card_id=card.id,
                revision_id=card.active_revision.id,
                output_size=output_size,
            )
        ):
            return
        self.generate_background_requested.emit()

    def _render_refine(
        self,
        document: Stack,
        revision: CardRevision,
        *,
        source_size: tuple[int, int] | None,
        select_current: bool,
    ) -> None:
        previous_selection = self.refine_resolution_combo.currentData()
        self._refine_source_size = source_size
        error = ""
        current_error = _exact_size_error(source_size, document.aspect_ratio)
        if revision.background is None:
            error = "Generate an image before reinterpreting."
        elif source_size is None:
            error = "The current image is unavailable or unreadable."
        with QSignalBlocker(self.refine_resolution_combo):
            self.refine_resolution_combo.clear()
            current_output_size = None
            current_tier = _current_tier(source_size, document.aspect_ratio)
            if source_size is not None:
                if current_error is None:
                    current_output_size = CurrentSourceSize(
                        width=source_size[0],
                        height=source_size[1],
                    )
                rows: list[tuple[str, object, tuple[int, int]]] = []
                if current_output_size is not None:
                    rows.append(
                        (
                            _tier_label(current_tier) if current_tier is not None else "Current",
                            current_output_size,
                            source_size,
                        )
                    )
                elif current_tier is None:
                    rows.append(
                        (
                            "Current",
                            None,
                            source_size,
                        )
                    )
                rows.extend(
                    (
                        _tier_label(tier),
                        PresetOutputSize(tier=tier),
                        output_dimensions(tier, document.aspect_ratio),
                    )
                    for tier in higher_output_tiers(
                        source_size[0],
                        source_size[1],
                        document.aspect_ratio,
                    )
                )
                for label, data, dimensions in _sort_size_rows(rows):
                    _add_resolution_item(
                        self.refine_resolution_combo,
                        label=label,
                        value=data,
                        dimensions=dimensions,
                        unavailable_reason=(
                            f"{current_error}. Choose a higher named tier."
                            if data is None and current_error is not None
                            else None
                        ),
                    )
            use_current = current_output_size is not None and (
                select_current
                or not isinstance(
                    previous_selection,
                    (PresetOutputSize, CurrentSourceSize),
                )
                or isinstance(previous_selection, CurrentSourceSize)
            )
            selection = (
                current_output_size
                if use_current
                else (
                    previous_selection if isinstance(previous_selection, PresetOutputSize) else None
                )
            )
            selected_index = (
                self._combo_index_for_data(
                    self.refine_resolution_combo,
                    selection,
                )
                if selection is not None
                else -1
            )
            self.refine_resolution_combo.setCurrentIndex(
                selected_index
                if selected_index >= 0
                else _first_selectable_combo_index(self.refine_resolution_combo)
            )
        self.refine_resolution_combo.setEnabled(
            _first_selectable_combo_index(self.refine_resolution_combo) >= 0
        )
        _sync_combo_tooltip(self.refine_resolution_combo)
        self._set_error(self.refine_error, error)
        style_suffix = ", and Style" if revision.style_id is not None else ""
        self._refine_using_text = (
            f"Create a new version using the current image, Description{style_suffix}."
        )
        self._refresh_generation_tooltips()

    def _request_refine_background(self) -> None:
        try:
            transformation = RefineTransformation(self.refine_transformation_combo.currentData())
        except (TypeError, ValueError):
            transformation = None
        output_size = self.refine_resolution_combo.currentData()
        if not isinstance(transformation, RefineTransformation):
            self._set_error(
                self.refine_error,
                "Select a supported Source Similarity.",
            )
            return
        if not isinstance(output_size, (PresetOutputSize, CurrentSourceSize)):
            if not self.refine_error.text():
                self._set_error(
                    self.refine_error,
                    "Select a Reinterpret resolution.",
                )
            return
        self.refine_background_requested.emit(transformation, output_size)

    def _render_edit(
        self,
        document: Stack,
        revision: CardRevision,
        *,
        source_size: tuple[int, int] | None,
        select_current: bool,
    ) -> None:
        previous_selection = self.edit_resolution_combo.currentData()
        self._edit_source_size = source_size
        error = ""
        current_error = _exact_size_error(source_size, document.aspect_ratio)
        if revision.background is None:
            error = "Generate an image before editing."
        elif source_size is None:
            error = "The current image is unavailable or unreadable."
        with QSignalBlocker(self.edit_resolution_combo):
            self.edit_resolution_combo.clear()
            if source_size is not None:
                current_output_size = (
                    CurrentSourceSize(
                        width=source_size[0],
                        height=source_size[1],
                    )
                    if current_error is None
                    else None
                )
                current_tier = _current_tier(
                    source_size,
                    document.aspect_ratio,
                )
                if current_tier is None:
                    _add_resolution_item(
                        self.edit_resolution_combo,
                        label="Current",
                        value=current_output_size,
                        dimensions=source_size,
                        unavailable_reason=(
                            f"{current_error}. Choose a higher named tier."
                            if current_output_size is None and current_error is not None
                            else None
                        ),
                    )
                else:
                    _add_resolution_item(
                        self.edit_resolution_combo,
                        label=_tier_label(current_tier),
                        value=current_output_size,
                        dimensions=source_size,
                    )
                for tier in higher_output_tiers(
                    source_size[0],
                    source_size[1],
                    document.aspect_ratio,
                ):
                    _add_resolution_item(
                        self.edit_resolution_combo,
                        label=_tier_label(tier),
                        value=PresetOutputSize(tier=tier),
                        dimensions=output_dimensions(tier, document.aspect_ratio),
                    )
                use_current = current_output_size is not None and (
                    select_current
                    or not isinstance(
                        previous_selection,
                        (PresetOutputSize, CurrentSourceSize),
                    )
                    or isinstance(previous_selection, CurrentSourceSize)
                )
                selection = (
                    current_output_size
                    if use_current
                    else (
                        previous_selection
                        if isinstance(previous_selection, PresetOutputSize)
                        else None
                    )
                )
                selected_index = (
                    self._combo_index_for_data(
                        self.edit_resolution_combo,
                        selection,
                    )
                    if selection is not None
                    else -1
                )
                fallback_index = _first_selectable_combo_index(self.edit_resolution_combo)
                self.edit_resolution_combo.setCurrentIndex(
                    selected_index
                    if selected_index >= 0
                    else (
                        fallback_index
                        if fallback_index >= 0
                        else (0 if self.edit_resolution_combo.count() else -1)
                    )
                )
        self.edit_resolution_combo.setEnabled(
            _first_selectable_combo_index(self.edit_resolution_combo) >= 0
        )
        _sync_combo_tooltip(self.edit_resolution_combo)
        self._set_error(self.edit_output_error, error)
        self._render_edit_tooltip()

    def _selected_edit_output_size(self) -> EditOutputSize | None:
        selection = self.edit_resolution_combo.currentData()
        if isinstance(selection, (CurrentSourceSize, PresetOutputSize)):
            return selection
        return None

    def _render_edit_tooltip(self) -> None:
        self._edit_using_text = "Create a new version with only the requested change."
        self._refresh_generation_tooltips()

    def _request_edit_background(self) -> None:
        instruction = self.edit_instruction_edit.toPlainText().strip()
        if not instruction:
            self._set_error(
                self.edit_instruction_error,
                "Enter an Edit Instruction.",
            )
            return
        output_size = self._selected_edit_output_size()
        if output_size is None:
            self._set_error(
                self.edit_output_error,
                "Select the current size or a higher Edit output tier.",
            )
            return
        self._set_error(self.edit_instruction_error, "")
        self._set_error(self.edit_output_error, "")
        self.edit_background_requested.emit(
            instruction,
            output_size,
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
        self.hotspot_when_label.setVisible(has_interaction)
        self.hotspot_when_panel.setVisible(has_interaction)
        self.hotspot_then_label.setVisible(has_interaction)
        self.hotspot_then_panel.setVisible(has_interaction)
        self.condition_rows_widget.setEnabled(has_interaction)
        self.key_change_rows_widget.setEnabled(has_interaction)
        condition_key_ids = (
            set(interaction.conditions.requires) | set(interaction.conditions.forbids)
            if interaction is not None
            else set()
        )
        change_key_ids = (
            set(interaction.key_changes.remove) | set(interaction.key_changes.grant)
            if interaction is not None
            else set()
        )
        has_condition_key = any(key.id not in condition_key_ids for key in document.keys)
        has_change_key = any(key.id not in change_key_ids for key in document.keys)
        self.add_condition_button.setEnabled(has_interaction and has_condition_key)
        self.add_key_change_button.setEnabled(has_interaction and has_change_key)
        self.add_condition_label.setEnabled(has_interaction and has_condition_key)
        self.add_key_change_label.setEnabled(has_interaction and has_change_key)
        self.add_condition_button.setToolTip(
            "Add a condition"
            if has_condition_key
            else "Add a Key in the Keys window before adding a condition"
            if not document.keys
            else "Every Key already has a condition"
        )
        self.add_key_change_button.setToolTip(
            "Add a key change"
            if has_change_key
            else "Add a Key in the Keys window before adding a key change"
            if not document.keys
            else "Every Key already has a key change"
        )
        self.hotspot_destination_combo.setEnabled(has_interaction)
        self.hotspot_sound_combo.setEnabled(has_interaction)
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
            if interaction is None:
                self.hotspot_destination_combo.setCurrentIndex(-1)
            elif interaction.action is not None and isinstance(
                interaction.action.target, ResolvedCardReference
            ):
                self.hotspot_destination_combo.setCurrentIndex(
                    self._combo_index_for_data(
                        self.hotspot_destination_combo,
                        interaction.action.target.target_card_id,
                    )
                )
            elif interaction.action is not None and isinstance(
                interaction.action.target, UnresolvedCardReference
            ):
                self.hotspot_destination_combo.setCurrentIndex(
                    self._combo_index_for_data(
                        self.hotspot_destination_combo,
                        interaction.action.target,
                    )
                )
            else:
                self.hotspot_destination_combo.setCurrentIndex(0)
            eligible_card_ids = {
                value
                for index in range(self.hotspot_destination_combo.count())
                if isinstance(
                    (value := self.hotspot_destination_combo.itemData(index)),
                    UUID,
                )
            }
            selected_card_id = self.selected_card_id
            self.hotspot_destination_combo.set_popup_anchor_data(
                _adjacent_card_id(
                    document,
                    selected_card_id,
                    eligible_card_ids,
                )
                if selected_card_id is not None
                else None
            )
        with QSignalBlocker(self.hotspot_sound_combo):
            self.hotspot_sound_combo.clear()
            self.hotspot_sound_combo.addItem("No Sound", None)
            for sound in document.sounds:
                self.hotspot_sound_combo.addItem(sound.name, sound.id)
                if sound.generated is None:
                    self.hotspot_sound_combo.setItemData(
                        self.hotspot_sound_combo.count() - 1,
                        "This Sound has not been generated yet.",
                        Qt.ItemDataRole.ToolTipRole,
                    )
            self.hotspot_sound_combo.setCurrentIndex(
                -1
                if interaction is None
                else self._combo_index_for_data(
                    self.hotspot_sound_combo,
                    interaction.sound_id,
                )
            )

    def _render_condition_rows(
        self,
        document: Stack,
        interaction: Interaction | None,
    ) -> None:
        self._clear_rule_rows(self.condition_rows_layout)
        rows = (
            *(("Has", key_id) for key_id in interaction.conditions.requires),
            *(("Lacks", key_id) for key_id in interaction.conditions.forbids),
        ) if interaction is not None else ()
        for row, (state, key_id) in enumerate(rows):
            key_combo = self._key_combo(document, key_id)
            key_combo.setObjectName(f"conditionKeyCombo{row}")
            state_combo = QComboBox()
            state_combo.setObjectName(f"conditionStateCombo{row}")
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
            remove_button.setFixedSize(20, 20)
            row_widget = QWidget()
            row_widget.setObjectName(f"conditionRow{row}")
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(3)
            row_layout.addWidget(state_combo)
            row_layout.addWidget(key_combo, 1)
            row_layout.addWidget(remove_button, 0, Qt.AlignmentFlag.AlignVCenter)
            self.condition_rows_layout.addWidget(row_widget)
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
        self.condition_rows_widget.setVisible(bool(rows))

    def _render_key_change_rows(
        self,
        document: Stack,
        interaction: Interaction | None,
    ) -> None:
        self._clear_rule_rows(self.key_change_rows_layout)
        rows = (
            *(("Lose", key_id) for key_id in interaction.key_changes.remove),
            *(("Gain", key_id) for key_id in interaction.key_changes.grant),
        ) if interaction is not None else ()
        for row, (change, key_id) in enumerate(rows):
            change_combo = QComboBox()
            change_combo.setObjectName(f"keyChangeActionCombo{row}")
            change_combo.addItem("Lose", "remove")
            change_combo.addItem("Gain", "grant")
            change_combo.setCurrentIndex(0 if change == "Lose" else 1)
            key_combo = self._key_combo(document, key_id)
            key_combo.setObjectName(f"keyChangeKeyCombo{row}")
            remove_button = _compact_text_button(
                "−",
                object_name=f"removeKeyChangeButton{row}",
                accessible_name="Remove key change",
                tooltip="Remove key change",
                prominent=False,
            )
            remove_button.setFixedSize(20, 20)
            row_widget = QWidget()
            row_widget.setObjectName(f"keyChangeRow{row}")
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(3)
            row_layout.addWidget(change_combo)
            row_layout.addWidget(key_combo, 1)
            row_layout.addWidget(remove_button, 0, Qt.AlignmentFlag.AlignVCenter)
            self.key_change_rows_layout.addWidget(row_widget)
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
        self.key_change_rows_widget.setVisible(bool(rows))

    @staticmethod
    def _clear_rule_rows(layout: QVBoxLayout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    @staticmethod
    def _key_combo(document: Stack, selected_key_id: UUID) -> QComboBox:
        combo = QComboBox()
        for key in document.keys:
            combo.addItem(key.name, key.id)
            combo.setItemData(
                combo.count() - 1,
                key.name,
                Qt.ItemDataRole.ToolTipRole,
            )
        combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        combo.setMinimumContentsLength(8)
        combo.setMinimumWidth(0)
        combo.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Fixed,
        )
        combo.setCurrentIndex(Inspector._combo_index_for_data(combo, selected_key_id))
        _sync_combo_tooltip(combo)
        combo.currentIndexChanged.connect(lambda _index: _sync_combo_tooltip(combo))
        return combo

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
        used = set(interaction.conditions.requires) | set(interaction.conditions.forbids)
        available = [key for key in self.controller.document.keys if key.id not in used]
        if not available:
            self.set_hotspot_error("Add a Key in the Keys window before assigning one.")
            return
        self._append_condition(available[-1].id, "requires")

    def _add_key_change(self) -> None:
        interaction = self._selected_interaction()
        if interaction is None:
            return
        used = set(interaction.key_changes.remove) | set(interaction.key_changes.grant)
        available = [key for key in self.controller.document.keys if key.id not in used]
        if not available:
            self.set_hotspot_error("Add a Key in the Keys window before assigning one.")
            return
        self._append_key_change(available[-1].id, "grant")

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
                conditions=interaction.conditions.model_copy(update={role: (*values, key_id)}),
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
            candidate for candidate in getattr(interaction.conditions, role) if candidate != key_id
        )
        self._execute(
            SetHotspotConditionsCommand(
                card_id=card.id,
                revision_id=card.active_revision.id,
                interaction_id=interaction.id,
                conditions=interaction.conditions.model_copy(update={role: values}),
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
                key_changes=interaction.key_changes.model_copy(update={role: (*values, key_id)}),
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
            candidate for candidate in getattr(interaction.key_changes, role) if candidate != key_id
        )
        self._execute(
            SetHotspotKeyChangesCommand(
                card_id=card.id,
                revision_id=card.active_revision.id,
                interaction_id=interaction.id,
                key_changes=interaction.key_changes.model_copy(update={role: values}),
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
        self._execute(
            ChangeHotspotDestinationCommand(
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
        )

    def _sound_changed(self, index: int) -> None:
        if self._rendering or index < 0:
            return
        card = self._selected_card()
        interaction = self._selected_interaction()
        if card is None or interaction is None:
            return
        value = self.hotspot_sound_combo.itemData(index)
        sound_id = value if isinstance(value, UUID) else None
        if interaction.sound_id == sound_id:
            return
        self._execute(
            ChangeHotspotSoundCommand(
                card_id=card.id,
                revision_id=card.active_revision.id,
                interaction_id=interaction.id,
                sound_id=sound_id,
            ),
            undo_message="Hotspot Sound changed",
        )

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
