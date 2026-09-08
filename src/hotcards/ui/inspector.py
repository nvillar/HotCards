"""Minimal revision-local Background and Hotspots inspector."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from pydantic import ValidationError
from PySide6.QtCore import QModelIndex, QSignalBlocker, QSize, Qt, Signal
from PySide6.QtGui import (
    QCloseEvent,
    QFocusEvent,
    QHideEvent,
    QIcon,
    QKeyEvent,
    QKeySequence,
    QPainter,
    QPalette,
    QStandardItemModel,
)
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from hotcards.application.commands import (
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
    ResolvedCardReference,
    Stack,
    UnresolvedCardReference,
    image_edit_lineage,
    selected_output_dimensions,
)
from hotcards.ui.card_picker import CardPickerWindow
from hotcards.ui.card_thumbnails import ImagePathResolver, card_thumbnail_icon
from hotcards.ui.sound_picker import SoundPickerWindow


class _CommitPlainTextEdit(QPlainTextEdit):
    editing_finished = Signal(object, object)

    def focusOutEvent(self, event: QFocusEvent) -> None:
        super().focusOutEvent(event)
        self.editing_finished.emit(QApplication.focusWidget(), event.reason())


class _EditDraftPlainTextEdit(QPlainTextEdit):
    undo_requested = Signal()
    redo_requested = Signal()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.matches(QKeySequence.StandardKey.Undo):
            self.undo_requested.emit()
            event.accept()
            return
        if event.matches(QKeySequence.StandardKey.Redo):
            self.redo_requested.emit()
            event.accept()
            return
        super().keyPressEvent(event)


class _EditHistoryDelegate(QStyledItemDelegate):
    def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex) -> QSize:
        return super().sizeHint(option, index) + QSize(0, 12)

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex) -> None:
        super().paint(painter, option, index)
        model = index.model()
        if model is not None and index.row() < model.rowCount(index.parent()) - 1:
            painter.save()
            painter.setPen(option.palette.color(QPalette.ColorRole.Mid))
            painter.drawLine(
                option.rect.left() + 4,
                option.rect.bottom(),
                option.rect.right() - 4,
                option.rect.bottom(),
            )
            painter.restore()


class _EditHistoryList(QListWidget):
    instruction_requested = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.setItemDelegate(_EditHistoryDelegate(self))
        self.itemClicked.connect(self._recall)

    def _recall(self, item: QListWidgetItem) -> None:
        instruction = item.data(Qt.ItemDataRole.UserRole)
        if isinstance(instruction, str):
            self.instruction_requested.emit(instruction)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        # Handle keyboard activation separately: itemActivated can also accompany
        # a mouse click on platforms with single-click activation.
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            item = self.currentItem()
            if item is not None:
                self._recall(item)
            event.accept()
            return
        super().keyPressEvent(event)


@dataclass(frozen=True, slots=True)
class _ImageAuthoringFields:
    description_label: QLabel
    description_edit: _CommitPlainTextEdit
    description_error: QLabel
    style_label: QLabel
    style_combo: QComboBox
    style_error: QLabel


def _image_authoring_fields(layout: QVBoxLayout) -> _ImageAuthoringFields:
    description_label = QLabel("Description")
    description_label.setObjectName("descriptionLabel")
    layout.addWidget(description_label)
    description_edit = _CommitPlainTextEdit()
    description_edit.setObjectName("descriptionEdit")
    description_edit.setPlaceholderText(
        "Describe the image to generate. Refer to References as image 1 and image 2."
    )
    description_edit.setToolTip("Used to generate an image.")
    description_edit.setAccessibleName("Description")
    editor_height = round((description_edit.fontMetrics().lineSpacing() * 10 + 20) * 1.25)
    description_edit.setMinimumHeight(editor_height)
    layout.addWidget(description_edit, 1)
    description_error = QLabel()
    description_error.setObjectName("descriptionValidationError")
    description_error.setWordWrap(True)
    description_error.hide()
    layout.addWidget(description_error)
    layout.addSpacing(8)
    style_label = QLabel("Style")
    style_label.setObjectName("styleLabel")
    layout.addWidget(style_label)
    style_combo = QComboBox()
    style_combo.setObjectName("styleCombo")
    style_combo.setAccessibleName("Style")
    style_combo.setToolTip("Rendering treatment for Generate and visual continuity in Edit.")
    layout.addWidget(style_combo)
    style_error = QLabel()
    style_error.setObjectName("styleSelectionValidationError")
    style_error.setWordWrap(True)
    style_error.hide()
    layout.addWidget(style_error)
    return _ImageAuthoringFields(
        description_label,
        description_edit,
        description_error,
        style_label,
        style_combo,
        style_error,
    )


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


def _compact_button_offset_container(
    button: QToolButton,
    *,
    top_margin: int,
) -> QWidget:
    container = QWidget()
    layout = QVBoxLayout(container)
    layout.setContentsMargins(0, top_margin, 0, 0)
    layout.setSpacing(0)
    layout.addWidget(button)
    container.setFixedSize(button.width(), button.height() + top_margin)
    return container


def _set_compact_button_top_margin(container: QWidget, top_margin: int) -> None:
    layout = container.layout()
    if layout is not None:
        layout.setContentsMargins(0, top_margin, 0, 0)
    container.setFixedHeight(20 + top_margin)


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
    edit_background_requested = Signal(str, object)
    hotspot_selected = Signal(object)
    hotspot_drawing_requested = Signal()
    sound_preview_requested = Signal(object)
    sound_preview_stop_requested = Signal()
    change_applied = Signal(str, object)
    render_inputs_changed = Signal()

    def __init__(
        self,
        controller: DocumentController,
        parent: QWidget | None = None,
        *,
        image_path_resolver: ImagePathResolver | None = None,
        render_on_change: bool = True,
    ) -> None:
        super().__init__(parent)
        self.controller = controller
        self._render_on_change = render_on_change
        self.selected_card_id: UUID | None = None
        self._rendered_revision_id: UUID | None = None
        self._rendered_background_id: UUID | None = None
        self._generate_using_text = ""
        self._generate_reason = ""
        self._image_source_size: tuple[int, int] | None = None
        self._edit_using_text = ""
        self._edit_reason = ""
        self._image_path_resolver = image_path_resolver
        self._card_picker_context: tuple[str, UUID, UUID, object] | None = None
        self._sound_picker_context: tuple[UUID, UUID, UUID] | None = None
        self._sound_picker_preview_active = False
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
        self._build_edit_tab()
        self._build_hotspots_tab()
        self._description_errors = (self.description_error,)
        self._style_errors = (self.style_selection_error,)
        self.card_picker = CardPickerWindow(
            self,
            image_path_resolver=image_path_resolver,
        )
        self.sound_picker = SoundPickerWindow(self)
        self._connect_signals()

    def _build_background_tab(self) -> None:
        scroll = QScrollArea()
        scroll.setObjectName("backgroundInspectorTab")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        page = QWidget()
        page.setObjectName("backgroundInspectorContent")
        layout = QVBoxLayout(page)

        self._generate_fields = _image_authoring_fields(layout)
        self.description_label = self._generate_fields.description_label
        self.description_edit = self._generate_fields.description_edit
        self.description_error = self._generate_fields.description_error
        self.style_label = self._generate_fields.style_label
        self.style_combo = self._generate_fields.style_combo
        self.style_selection_error = self._generate_fields.style_error

        layout.addSpacing(8)
        self.reference_label = QLabel("References")
        self.reference_label.setObjectName("referenceLabel")
        self.reference_label.setToolTip("Use image 1 and image 2 in the Description.")
        layout.addWidget(self.reference_label)
        self.reference_panel = QWidget()
        self.reference_panel.setObjectName("referencePanel")
        reference_layout = QVBoxLayout(self.reference_panel)
        reference_layout.setContentsMargins(0, 0, 0, 0)
        reference_layout.setSpacing(4)
        first_reference_row = QHBoxLayout()
        first_reference_row.addWidget(QLabel("1."))
        self.reference_button = QPushButton("Choose Reference…")
        self.reference_button.setObjectName("referenceCardButton")
        self.reference_button.setAccessibleName("Choose Reference card 1")
        self.reference_button.setToolTip(
            "Generate only: this Reference is sent to the model as image 1"
        )
        self.reference_button.setIconSize(QSize(48, 32))
        first_reference_row.addWidget(
            self.reference_button,
            1,
            Qt.AlignmentFlag.AlignVCenter,
        )
        self.reference_remove_button = _compact_text_button(
            "−",
            object_name="removeReferenceCardButton",
            accessible_name="Remove Reference card 1",
            tooltip="Remove Reference card 1",
            prominent=False,
        )
        self.reference_remove_button.setFixedSize(20, 20)
        self.reference_remove_button.setAttribute(Qt.WidgetAttribute.WA_LayoutUsesWidgetRect)
        self.reference_remove_container = _compact_button_offset_container(
            self.reference_remove_button,
            top_margin=2,
        )
        first_reference_row.addWidget(
            self.reference_remove_container,
            0,
            Qt.AlignmentFlag.AlignVCenter,
        )
        reference_layout.addLayout(first_reference_row)
        second_reference_row = QHBoxLayout()
        second_reference_row.addWidget(QLabel("2."))
        self.additional_reference_button = QPushButton("Choose Reference…")
        self.additional_reference_button.setObjectName("additionalReferenceCardButton")
        self.additional_reference_button.setAccessibleName("Choose Reference card 2")
        self.additional_reference_button.setToolTip(
            "Generate only: this Reference is sent to the model as image 2"
        )
        self.additional_reference_button.setIconSize(QSize(48, 32))
        second_reference_row.addWidget(
            self.additional_reference_button,
            1,
            Qt.AlignmentFlag.AlignVCenter,
        )
        self.additional_reference_remove_button = _compact_text_button(
            "−",
            object_name="removeAdditionalReferenceCardButton",
            accessible_name="Remove Reference card 2",
            tooltip="Remove Reference card 2",
            prominent=False,
        )
        self.additional_reference_remove_button.setFixedSize(20, 20)
        self.additional_reference_remove_button.setAttribute(
            Qt.WidgetAttribute.WA_LayoutUsesWidgetRect
        )
        self.additional_reference_remove_container = _compact_button_offset_container(
            self.additional_reference_remove_button,
            top_margin=2,
        )
        second_reference_row.addWidget(
            self.additional_reference_remove_container,
            0,
            Qt.AlignmentFlag.AlignVCenter,
        )
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
        self._generate_tab_index = self.inspector_tabs.addTab(scroll, "Generate")

    def _build_edit_tab(self) -> None:
        scroll = QScrollArea()
        scroll.setObjectName("editInspectorTab")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        page = QWidget()
        page.setObjectName("editInspectorContent")
        layout = QVBoxLayout(page)

        self.edit_instruction_label = QLabel("Edit Instruction")
        layout.addWidget(self.edit_instruction_label)
        self.edit_instruction_edit = _EditDraftPlainTextEdit()
        self.edit_instruction_edit.setObjectName("editInstructionEdit")
        self.edit_instruction_edit.setAccessibleName("Edit Instruction")
        self.edit_instruction_edit.setPlaceholderText("Describe the change to make")
        self.edit_instruction_edit.setMaximumHeight(110)
        self.edit_instruction_edit.setUndoRedoEnabled(False)
        layout.addWidget(self.edit_instruction_edit)
        self.edit_instruction_error = QLabel()
        self.edit_instruction_error.setObjectName("editInstructionValidationError")
        self.edit_instruction_error.setWordWrap(True)
        self.edit_instruction_error.setVisible(False)
        layout.addWidget(self.edit_instruction_error)

        self.edit_resolution_label = QLabel("Resolution")
        layout.addWidget(self.edit_resolution_label)
        self.edit_resolution_combo = QComboBox()
        self.edit_resolution_combo.setObjectName("editResolutionCombo")
        self.edit_resolution_combo.setAccessibleName("Edit resolution")
        layout.addWidget(self.edit_resolution_combo)
        self.edit_output_error = QLabel()
        self.edit_output_error.setObjectName("editOutputValidationError")
        self.edit_output_error.setWordWrap(True)
        self.edit_output_error.setVisible(False)
        layout.addWidget(self.edit_output_error)

        self.edit_background_button = QPushButton("Edit")
        self.edit_background_button.setObjectName("editBackgroundButton")
        layout.addWidget(self.edit_background_button)
        self._focus_commit_targets.add(self.edit_background_button)

        layout.addSpacing(8)
        self.edit_history_label = QLabel("Edit History")
        self.edit_history_label.setObjectName("editHistoryLabel")
        layout.addWidget(self.edit_history_label)
        self.edit_history_list = _EditHistoryList()
        self.edit_history_list.setObjectName("editHistoryList")
        self.edit_history_list.setAccessibleName("Edit History")
        self.edit_history_list.setToolTip(
            "Click an accepted Edit, or press Enter or Space, to reuse its instruction."
        )
        self.edit_history_list.setWordWrap(True)
        self.edit_history_list.setTextElideMode(Qt.TextElideMode.ElideNone)
        self.edit_history_list.setMinimumHeight(160)
        layout.addWidget(self.edit_history_list, 1)

        scroll.setWidget(page)
        self._edit_tab_index = self.inspector_tabs.addTab(scroll, "Edit")

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
            accessible_name="Draw hotspot",
            tooltip="Draw a new hotspot",
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

        self.hotspot_target_label = QLabel("Go to card")
        self.hotspot_target_label.setObjectName("hotspotTargetLabel")
        then_layout.addWidget(self.hotspot_target_label)
        self.hotspot_destination_row = QWidget()
        self.hotspot_destination_row.setObjectName("hotspotDestinationRow")
        hotspot_destination_layout = QHBoxLayout(self.hotspot_destination_row)
        hotspot_destination_layout.setContentsMargins(0, 0, 0, 0)
        hotspot_destination_layout.setSpacing(4)
        self.hotspot_destination_button = QPushButton("Choose Destination…")
        self.hotspot_destination_button.setObjectName("hotspotDestinationButton")
        self.hotspot_destination_button.setAccessibleName("Choose hotspot destination")
        self.hotspot_destination_button.setIconSize(QSize(48, 32))
        hotspot_destination_layout.addWidget(
            self.hotspot_destination_button,
            1,
            Qt.AlignmentFlag.AlignVCenter,
        )
        self.hotspot_destination_remove_button = _compact_text_button(
            "−",
            object_name="removeHotspotDestinationButton",
            accessible_name="Remove hotspot destination",
            tooltip="Remove hotspot destination",
            prominent=False,
        )
        self.hotspot_destination_remove_button.setFixedSize(20, 20)
        self.hotspot_destination_remove_button.setAttribute(
            Qt.WidgetAttribute.WA_LayoutUsesWidgetRect
        )
        self.hotspot_destination_remove_container = _compact_button_offset_container(
            self.hotspot_destination_remove_button,
            top_margin=2,
        )
        hotspot_destination_layout.addWidget(
            self.hotspot_destination_remove_container,
            0,
            Qt.AlignmentFlag.AlignVCenter,
        )
        then_layout.addWidget(self.hotspot_destination_row)
        self.hotspot_sound_label = QLabel("Play sound")
        self.hotspot_sound_label.setObjectName("hotspotSoundLabel")
        then_layout.addWidget(self.hotspot_sound_label)
        self.hotspot_sound_row = QWidget()
        self.hotspot_sound_row.setObjectName("hotspotSoundRow")
        hotspot_sound_layout = QHBoxLayout(self.hotspot_sound_row)
        hotspot_sound_layout.setContentsMargins(0, 0, 0, 0)
        hotspot_sound_layout.setSpacing(4)
        self.hotspot_sound_button = QPushButton("Choose Sound…")
        self.hotspot_sound_button.setObjectName("hotspotSoundButton")
        self.hotspot_sound_button.setAccessibleName("Choose hotspot Sound")
        hotspot_sound_layout.addWidget(
            self.hotspot_sound_button,
            1,
            Qt.AlignmentFlag.AlignVCenter,
        )
        self.hotspot_sound_remove_button = _compact_text_button(
            "−",
            object_name="removeHotspotSoundButton",
            accessible_name="Remove hotspot Sound",
            tooltip="Remove hotspot Sound",
            prominent=False,
        )
        self.hotspot_sound_remove_button.setFixedSize(20, 20)
        self.hotspot_sound_remove_button.setAttribute(Qt.WidgetAttribute.WA_LayoutUsesWidgetRect)
        self.hotspot_sound_remove_container = _compact_button_offset_container(
            self.hotspot_sound_remove_button,
            top_margin=2,
        )
        hotspot_sound_layout.addWidget(
            self.hotspot_sound_remove_container,
            0,
            Qt.AlignmentFlag.AlignVCenter,
        )
        then_layout.addWidget(self.hotspot_sound_row)
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

    def hideEvent(self, event: QHideEvent) -> None:
        self.card_picker.hide()
        self.sound_picker.hide()
        self._card_picker_context = None
        self._sound_picker_context = None
        super().hideEvent(event)

    def closeEvent(self, event: QCloseEvent) -> None:
        self.card_picker.close()
        self.sound_picker.close()
        self._card_picker_context = None
        self._sound_picker_context = None
        super().closeEvent(event)

    def _connect_signals(self) -> None:
        self.description_edit.editing_finished.connect(self._description_editing_finished)
        self.style_combo.currentIndexChanged.connect(self._revision_style_changed)
        self.description_edit.document().contentsChanged.connect(self._render_inputs_changed)
        self.generate_background_button.clicked.connect(self._request_generate_background)
        self.edit_instruction_edit.textChanged.connect(self._edit_instruction_changed)
        self.edit_resolution_combo.currentIndexChanged.connect(self._edit_output_changed)
        self.edit_background_button.clicked.connect(self._request_edit_background)
        self.edit_history_list.instruction_requested.connect(self._recall_edit_instruction)
        self.reference_button.clicked.connect(lambda: self._open_reference_picker(1))
        self.additional_reference_button.clicked.connect(lambda: self._open_reference_picker(2))
        self.reference_remove_button.clicked.connect(lambda: self._set_reference(1, None))
        self.additional_reference_remove_button.clicked.connect(
            lambda: self._set_reference(2, None)
        )
        self.resolution_combo.currentIndexChanged.connect(self._revision_resolution_changed)
        self.hotspot_list.currentItemChanged.connect(self._hotspot_selection_changed)
        self.move_hotspot_up_button.clicked.connect(lambda: self._move_hotspot(-1))
        self.move_hotspot_down_button.clicked.connect(lambda: self._move_hotspot(1))
        self.add_hotspot_button.clicked.connect(self._request_hotspot_drawing)
        self.delete_hotspot_button.clicked.connect(self._delete_hotspot)
        self.add_condition_button.clicked.connect(self._add_condition)
        self.add_key_change_button.clicked.connect(self._add_key_change)
        self.hotspot_destination_button.clicked.connect(self._open_destination_picker)
        self.hotspot_destination_remove_button.clicked.connect(
            lambda: self._change_current_destination(None)
        )
        self.hotspot_sound_button.clicked.connect(self._open_sound_picker)
        self.hotspot_sound_remove_button.clicked.connect(lambda: self._change_current_sound(None))
        self.card_picker.card_selected.connect(self._card_picked)
        self.card_picker.dismissed.connect(self._discard_card_picker_context)
        self.sound_picker.sound_selected.connect(self._sound_picked)
        self.sound_picker.preview_requested.connect(self._preview_sound)
        self.sound_picker.dismissed.connect(self._sound_picker_dismissed)

    def _render_inputs_changed(self) -> None:
        if not self._rendering:
            self.render_inputs_changed.emit()

    def _edit_instruction_changed(self) -> None:
        if self._rendering:
            return
        self._replace_edit_draft(self.edit_instruction_edit.toPlainText())
        self._edit_inputs_changed()

    def _replace_edit_draft(self, instruction: str) -> bool:
        card = self._selected_card()
        if card is None:
            return False
        try:
            self.controller.replace_edit_draft(
                card.id,
                card.active_revision.id,
                instruction,
            )
        except (DocumentMutationBlockedError, ValidationError, ValueError) as error:
            self._set_error(self.edit_instruction_error, str(error))
            self._render_edit_instruction(card.active_revision.edit_draft.instruction)
            return False
        return True

    def _edit_inputs_changed(self) -> None:
        if self._rendering:
            return
        self._set_error(self.edit_instruction_error, "")
        self._set_error(self.edit_output_error, "")
        self._render_edit_tooltip()
        self.render_inputs_changed.emit()

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
        image_source_size: tuple[int, int] | None = None,
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
                self._image_source_size = None
                self._rendered_revision_id = None
                self._rendered_background_id = None
                self._render_edit_history(None)
                self._render_edit_instruction("")
                self.pages.setCurrentIndex(0)
                self._set_error(self._description_errors, "")
                self._set_error(self.reference_error, "")
                self._set_error(self.edit_instruction_error, "")
                self._set_error(self.edit_output_error, "")
                self._set_error(self._style_errors, "")
                self.set_hotspot_error("")
                self.description_edit.clear()
                self.hotspot_list.clear()
                self._render_hotspot_properties(document, None)
                return
            self.pages.setCurrentIndex(1)
            revision = card.active_revision
            same_revision = previous_card_id == card.id and previous_revision_id == revision.id
            background_id = revision.background.id if revision.background is not None else None
            same_image = same_revision and self._rendered_background_id == background_id
            if not same_revision:
                self._set_error(self._description_errors, "")
                self._set_error(self.reference_error, "")
                self.set_hotspot_error("")
            self._rendered_revision_id = revision.id
            self._rendered_background_id = background_id
            description = (
                description_draft
                if preserve_description and same_revision
                else revision.description
            )
            if self.description_edit.toPlainText() != description:
                self.description_edit.setPlainText(description)
            self._render_edit_instruction(revision.edit_draft.instruction)
            self._render_style_selector(document, revision)
            self._render_reference(document, card, revision)
            self._render_resolution(
                document,
                revision,
                source_size=image_source_size,
                select_current=not same_image,
            )
            self._render_description_workflow(document, card, revision)
            self._render_edit(
                document,
                revision,
                source_size=image_source_size,
                select_current=not same_image,
            )
            self._render_edit_history(revision)
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
            error_label=self._description_errors,
            render_change=render_change,
        )

    def commit_card_metadata(self, *, render_change: bool = True) -> bool:
        """Commit every visible authoring draft before a context change or save."""
        return self.commit_revision_metadata(render_change=render_change)

    def has_description_input(self) -> bool:
        return self.selected_card_id is not None and bool(
            self.description_edit.toPlainText().strip()
        )

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
        busy: bool,
        generating: bool,
    ) -> None:
        self._generate_reason = generate_reason
        self.generate_background_button.setEnabled(can_generate and not busy)
        self.generate_background_button.setText("Generating…" if generating else "Generate Image")
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

    def set_edit_instruction(self, instruction: str) -> None:
        self._render_edit_instruction(instruction)
        self._replace_edit_draft(instruction)
        self._edit_inputs_changed()

    def _render_edit_instruction(self, instruction: str) -> None:
        with QSignalBlocker(self.edit_instruction_edit):
            self.edit_instruction_edit.setPlainText(instruction)
        self._set_error(self.edit_instruction_error, "")
        self._render_edit_tooltip()

    def clear_edit_instruction(self) -> None:
        self.set_edit_instruction("")

    def _render_edit_history(self, revision: CardRevision | None) -> None:
        lineage = (
            image_edit_lineage(revision.background.provenance)
            if revision is not None and revision.background is not None
            else ()
        )
        with QSignalBlocker(self.edit_history_list):
            self.edit_history_list.clear()
            for edit in lineage:
                item = QListWidgetItem(edit.instruction)
                item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
                item.setData(Qt.ItemDataRole.UserRole, edit.instruction)
                item.setToolTip(edit.instruction)
                self.edit_history_list.addItem(item)

    def _recall_edit_instruction(self, instruction: str) -> None:
        if self._rendering:
            return
        self.set_edit_instruction(instruction)

    def _refresh_generation_tooltips(self) -> None:
        self.generate_background_button.setToolTip(
            self._tooltip_with_using(
                self._generate_reason,
                self._generate_using_text,
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
        self.card_picker.hide()
        self.sound_picker.hide()
        self._card_picker_context = None
        self._sound_picker_context = None
        self.selected_card_id = None
        self._rendered_revision_id = None
        self._rendered_background_id = None
        self._render_edit_history(None)
        self._render_edit_instruction("")
        self._set_error(self._description_errors, "")
        self._set_error(self.reference_error, "")
        self._set_error(self._style_errors, "")
        self.set_hotspot_error("")

    def _render_style_selector(
        self,
        document: Stack,
        revision: CardRevision,
    ) -> None:
        combo = self.style_combo
        with QSignalBlocker(combo):
            combo.clear()
            combo.addItem("No Style", None)
            for style in document.styles:
                combo.addItem(style.name, style.id)
            combo.setCurrentIndex(
                self._combo_index_for_data(
                    combo,
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
            error_label=self._style_errors,
            undo_message="Style changed",
        )

    def _render_reference(
        self,
        document: Stack,
        card: Card,
        revision: CardRevision,
    ) -> None:
        references = revision.references
        controls = (
            (self.reference_button, self.reference_remove_button),
            (
                self.additional_reference_button,
                self.additional_reference_remove_button,
            ),
        )
        for position, (button, remove_button) in enumerate(controls):
            reference = references[position] if position < len(references) else None
            self._render_card_picker_control(
                button,
                remove_button,
                document,
                reference,
                empty_text="Choose Reference…",
            )
        self.additional_reference_button.setEnabled(bool(references))
        self.additional_reference_remove_button.setEnabled(len(references) > 1)

    def _open_reference_picker(self, position: int) -> None:
        card = self._selected_card()
        if card is None:
            return
        references = card.active_revision.references
        reference = references[position - 1] if position <= len(references) else None
        current_id = (
            reference.target_card_id if isinstance(reference, ResolvedCardReference) else None
        )
        used_ids = {
            candidate.target_card_id
            for candidate in references
            if isinstance(candidate, ResolvedCardReference)
            and candidate.target_card_id != current_id
        }
        enabled_ids = {
            candidate.id
            for candidate in self.controller.document.cards
            if candidate.id != card.id and candidate.id not in used_ids
        }
        self._card_picker_context = (
            "reference",
            card.id,
            card.active_revision.id,
            position,
        )
        anchor = self.reference_button if position == 1 else self.additional_reference_button
        self.card_picker.open_for(
            anchor,
            self.controller.document.cards,
            title=f"Select Reference {position}",
            enabled_card_ids=enabled_ids,
            selected_card_id=current_id,
        )

    def _set_reference(
        self,
        position: int,
        reference: ResolvedCardReference | UnresolvedCardReference | None,
    ) -> None:
        if self._rendering:
            return
        card = self._selected_card()
        if card is None:
            return
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

    def _render_card_picker_control(
        self,
        button: QPushButton,
        remove_button: QToolButton,
        document: Stack,
        reference: ResolvedCardReference | UnresolvedCardReference | None,
        *,
        empty_text: str,
    ) -> None:
        button.setIcon(QIcon())
        has_thumbnail = False
        if isinstance(reference, ResolvedCardReference):
            selected = next(
                (
                    candidate
                    for candidate in document.cards
                    if candidate.id == reference.target_card_id
                ),
                None,
            )
            if selected is not None:
                button.setText(selected.name)
                button.setIcon(
                    card_thumbnail_icon(
                        selected,
                        size=QSize(48, 32),
                        image_path_resolver=self._image_path_resolver,
                    )
                )
                has_thumbnail = True
            else:
                button.setText("Missing card")
        elif isinstance(reference, UnresolvedCardReference):
            button.setText(
                f"Missing: {reference.target_name}" if reference.target_name else "Unresolved card"
            )
        else:
            button.setText(empty_text)
        remove_container = remove_button.parentWidget()
        if remove_container is not None:
            _set_compact_button_top_margin(
                remove_container,
                4 if has_thumbnail else 2,
            )
        remove_button.setEnabled(reference is not None)

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
            self.render(
                self.controller.document,
                self.selected_card_id,
                image_source_size=self._image_source_size,
            )
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

    def _render_edit(
        self,
        document: Stack,
        revision: CardRevision,
        *,
        source_size: tuple[int, int] | None,
        select_current: bool,
    ) -> None:
        previous_selection = self.edit_resolution_combo.currentData()
        self._image_source_size = source_size
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
        self._edit_using_text = "Apply only the requested change to the current image."
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
        self.hotspot_destination_button.setEnabled(has_interaction)
        self.hotspot_sound_button.setEnabled(has_interaction)
        self.delete_hotspot_button.setEnabled(has_interaction)
        current_row = self.hotspot_list.currentRow()
        self.move_hotspot_up_button.setEnabled(has_interaction and current_row > 0)
        self.move_hotspot_down_button.setEnabled(
            has_interaction and current_row < self.hotspot_list.count() - 1
        )
        self._render_condition_rows(document, interaction)
        self._render_key_change_rows(document, interaction)
        destination = (
            interaction.action.target
            if interaction is not None and interaction.action is not None
            else None
        )
        self._render_card_picker_control(
            self.hotspot_destination_button,
            self.hotspot_destination_remove_button,
            document,
            destination,
            empty_text="Choose Destination…",
        )
        self.hotspot_destination_remove_button.setEnabled(
            has_interaction and destination is not None
        )
        sound = (
            document.sound_by_id(interaction.sound_id)
            if interaction is not None and interaction.sound_id is not None
            else None
        )
        self.hotspot_sound_button.setText(sound.name if sound is not None else "Choose Sound…")
        self.hotspot_sound_remove_button.setEnabled(has_interaction and sound is not None)

    def _render_condition_rows(
        self,
        document: Stack,
        interaction: Interaction | None,
    ) -> None:
        self._clear_rule_rows(self.condition_rows_layout)
        rows = (
            (
                *(("Has", key_id) for key_id in interaction.conditions.requires),
                *(("Lacks", key_id) for key_id in interaction.conditions.forbids),
            )
            if interaction is not None
            else ()
        )
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
            (
                *(("Lose", key_id) for key_id in interaction.key_changes.remove),
                *(("Gain", key_id) for key_id in interaction.key_changes.grant),
            )
            if interaction is not None
            else ()
        )
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
        combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
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

    def _request_hotspot_drawing(self) -> None:
        self.hotspot_drawing_requested.emit()

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
            self._render_hotspot_properties(self.controller.document, interaction)
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
            self._render_hotspot_properties(self.controller.document, interaction)
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

    def _open_destination_picker(self) -> None:
        card = self._selected_card()
        interaction = self._selected_interaction()
        if card is None or interaction is None:
            return
        current_id = (
            interaction.action.target.target_card_id
            if interaction.action is not None
            and isinstance(interaction.action.target, ResolvedCardReference)
            else None
        )
        self._card_picker_context = (
            "destination",
            card.id,
            card.active_revision.id,
            interaction.id,
        )
        self.card_picker.open_for(
            self.hotspot_destination_button,
            self.controller.document.cards,
            title="Select Hotspot Destination",
            enabled_card_ids={candidate.id for candidate in self.controller.document.cards},
            selected_card_id=current_id,
        )

    def _change_current_destination(self, destination: object) -> None:
        if self._rendering:
            return
        card = self._selected_card()
        interaction = self._selected_interaction()
        if card is None or interaction is None:
            return
        self._apply_destination_change(
            card.id,
            card.active_revision.id,
            interaction.id,
            destination,
        )

    def _card_picked(self, card_id: object) -> None:
        context = self._card_picker_context
        self._card_picker_context = None
        if not isinstance(card_id, UUID) or context is None:
            return
        kind, card_context_id, revision_context_id, target = context
        card = self._selected_card()
        if (
            card is None
            or card.id != card_context_id
            or card.active_revision.id != revision_context_id
        ):
            return
        if kind == "reference" and isinstance(target, int):
            self._set_reference(
                target,
                ResolvedCardReference(target_card_id=card_id),
            )
        elif kind == "destination" and isinstance(target, UUID):
            interaction = self._selected_interaction()
            if interaction is not None and interaction.id == target:
                self._apply_destination_change(
                    card.id,
                    card.active_revision.id,
                    interaction.id,
                    card_id,
                )

    def _discard_card_picker_context(self) -> None:
        self._card_picker_context = None

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

    def _open_sound_picker(self) -> None:
        card = self._selected_card()
        interaction = self._selected_interaction()
        if card is None or interaction is None:
            return
        self._sound_picker_context = (
            card.id,
            card.active_revision.id,
            interaction.id,
        )
        self.sound_picker.open_for(
            self.hotspot_sound_button,
            list(self.controller.document.sounds),
            selected_sound_id=interaction.sound_id,
        )

    def _sound_picked(self, sound_id: object) -> None:
        context = self._sound_picker_context
        self._sound_picker_context = None
        self._stop_sound_picker_preview()
        if not isinstance(sound_id, UUID) or context is None:
            return
        card_id, revision_id, interaction_id = context
        card = self._selected_card()
        interaction = self._selected_interaction()
        if (
            card is None
            or interaction is None
            or card.id != card_id
            or card.active_revision.id != revision_id
            or interaction.id != interaction_id
            or not any(sound.id == sound_id for sound in self.controller.document.sounds)
        ):
            return
        self._change_current_sound(sound_id)

    def _change_current_sound(self, sound_id: UUID | None) -> None:
        card = self._selected_card()
        interaction = self._selected_interaction()
        if card is None or interaction is None:
            return
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

    def _preview_sound(self, sound_id: object) -> None:
        if not isinstance(sound_id, UUID):
            return
        context = self._sound_picker_context
        card = self._selected_card()
        interaction = self._selected_interaction()
        if (
            context is None
            or card is None
            or interaction is None
            or context != (card.id, card.active_revision.id, interaction.id)
            or not any(
                sound.id == sound_id and sound.generated is not None
                for sound in self.controller.document.sounds
            )
        ):
            return
        self._sound_picker_preview_active = True
        self.sound_preview_requested.emit(sound_id)

    def _sound_picker_dismissed(self) -> None:
        self._sound_picker_context = None
        self._stop_sound_picker_preview()

    def _stop_sound_picker_preview(self) -> None:
        if not self._sound_picker_preview_active:
            return
        self._sound_picker_preview_active = False
        self.sound_preview_stop_requested.emit()

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
        error_label: QLabel | tuple[QLabel, ...] | None = None,
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
                self.render(
                    self.controller.document,
                    self.selected_card_id,
                    image_source_size=self._image_source_size,
                )
            return False
        self._set_error(target_error, "")
        if render_change:
            if self._render_on_change:
                self.render(
                    changed,
                    self.selected_card_id,
                    image_source_size=self._image_source_size,
                )
            self.document_changed.emit(changed)
        token = self.controller.current_undo_token
        if undo_message is not None and token is not None and token != previous_token:
            self.change_applied.emit(undo_message, token)
        return True

    @staticmethod
    def _set_error(labels: QLabel | tuple[QLabel, ...], message: str) -> None:
        for label in (labels,) if isinstance(labels, QLabel) else labels:
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
