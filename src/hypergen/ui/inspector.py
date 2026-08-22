"""Minimal revision-local Background and Hotspots inspector."""

from __future__ import annotations

from uuid import UUID

from pydantic import ValidationError
from PySide6.QtCore import QSignalBlocker, Qt, QTimer, Signal
from PySide6.QtGui import QFocusEvent
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
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
    RenameInteractionCommand,
    ReorderHotspotCommand,
    SetRevisionStyleCommand,
)
from hypergen.application.document_controller import DocumentController
from hypergen.domain.models import (
    Card,
    CardRevision,
    HotspotSet,
    Interaction,
    NavigateAction,
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
    return button


class Inspector(QWidget):
    """Render the selected active revision and issue typed commands."""

    document_changed = Signal(object)
    generate_background_requested = Signal()
    import_background_requested = Signal()
    clear_background_requested = Signal()
    enrich_scene_requested = Signal()
    remap_hotspots_requested = Signal()
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
        self._rendering = False
        self.setObjectName("inspector")
        self.setMinimumWidth(300)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

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

        self.scene_edit = _CommitPlainTextEdit()
        self.scene_edit.setObjectName("sceneDescriptionEdit")
        self.scene_edit.setPlaceholderText("Description")
        self.scene_edit.setAccessibleName("Description")
        self.scene_edit.setMaximumHeight(180)
        layout.addWidget(self.scene_edit)
        self.description_error = QLabel()
        self.description_error.setObjectName("descriptionValidationError")
        self.description_error.setWordWrap(True)
        self.description_error.setVisible(False)
        layout.addWidget(self.description_error)

        self.enrich_scene_button = QPushButton("Enrich")
        self.enrich_scene_button.setObjectName("enrichSceneButton")
        layout.addWidget(self.enrich_scene_button)

        layout.addSpacing(8)
        self.generate_background_button = QPushButton("Generate Image")
        self.generate_background_button.setObjectName("generateBackgroundButton")
        layout.addWidget(self.generate_background_button)

        self.style_combo = QComboBox()
        self.style_combo.setObjectName("generationStyleCombo")
        self.style_combo.setAccessibleName("Generate style")
        self.style_combo.setToolTip("Style used when generating this revision's image")
        layout.addWidget(self.style_combo)
        self.style_error = QLabel()
        self.style_error.setObjectName("styleValidationError")
        self.style_error.setWordWrap(True)
        self.style_error.setVisible(False)
        layout.addWidget(self.style_error)

        self.import_background_button = QPushButton("Import Image...")
        self.import_background_button.setObjectName("importBackgroundButton")
        layout.addWidget(self.import_background_button)
        self.clear_background_button = QPushButton("Clear Image")
        self.clear_background_button.setObjectName("clearBackgroundButton")
        layout.addWidget(self.clear_background_button)

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
        self.remap_hotspots_button = QPushButton("Remap")
        self.remap_hotspots_button.setObjectName("remapHotspotsButton")
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
        controls.addWidget(self.move_hotspot_up_button)
        controls.addWidget(self.move_hotspot_down_button)
        controls.addWidget(self.remap_hotspots_button)
        controls.addStretch(1)
        controls.addWidget(self.add_hotspot_button)
        controls.addWidget(self.delete_hotspot_button)
        layout.addLayout(controls)

        self.hotspot_label_edit = QLineEdit()
        self.hotspot_label_edit.setObjectName("hotspotLabelEdit")
        self.hotspot_label_edit.setPlaceholderText("Hotspot name")
        self.hotspot_label_edit.setAccessibleName("Hotspot name")
        layout.addWidget(self.hotspot_label_edit)

        self.hotspot_destination_combo = QComboBox()
        self.hotspot_destination_combo.setObjectName("hotspotDestinationCombo")
        self.hotspot_destination_combo.setAccessibleName("Hotspot destination")
        layout.addWidget(self.hotspot_destination_combo)

        self.hotspot_error = QLabel()
        self.hotspot_error.setObjectName("hotspotValidationError")
        self.hotspot_error.setWordWrap(True)
        self.hotspot_error.setVisible(False)
        layout.addWidget(self.hotspot_error)
        self.inspector_tabs.addTab(page, "Hotspots")

    def _connect_signals(self) -> None:
        self.scene_edit.editing_finished.connect(self.commit_revision_metadata)
        self.scene_edit.textChanged.connect(self.render_inputs_changed)
        self.enrich_scene_button.clicked.connect(self.enrich_scene_requested)
        self.generate_background_button.clicked.connect(
            self.generate_background_requested
        )
        self.style_combo.currentIndexChanged.connect(self._style_changed)
        self.style_combo.currentIndexChanged.connect(self.render_inputs_changed)
        self.import_background_button.clicked.connect(
            self.import_background_requested
        )
        self.clear_background_button.clicked.connect(
            self.clear_background_requested
        )
        self.hotspot_list.currentItemChanged.connect(
            self._hotspot_selection_changed
        )
        self.move_hotspot_up_button.clicked.connect(
            lambda: self._move_hotspot(-1)
        )
        self.move_hotspot_down_button.clicked.connect(
            lambda: self._move_hotspot(1)
        )
        self.remap_hotspots_button.clicked.connect(
            self.remap_hotspots_requested
        )
        self.add_hotspot_button.clicked.connect(self._add_hotspot)
        self.delete_hotspot_button.clicked.connect(self._delete_hotspot)
        self.hotspot_label_edit.editingFinished.connect(
            self._commit_hotspot_label
        )
        self.hotspot_destination_combo.currentIndexChanged.connect(
            self._destination_changed
        )

    def render(self, document: Stack, selected_card_id: UUID | None) -> None:
        previous_card_id = self.selected_card_id
        previous_revision_id = self._rendered_revision_id
        selected_interaction_id = self.selected_interaction_id
        preserve_description = self.scene_edit.hasFocus()
        description_draft = self.scene_edit.toPlainText()
        preserve_label = self.hotspot_label_edit.hasFocus()
        label_draft = self.hotspot_label_edit.text()
        self._rendering = True
        try:
            card = next(
                (
                    candidate
                    for candidate in document.cards
                    if candidate.id == selected_card_id
                ),
                None,
            )
            self.selected_card_id = card.id if card is not None else None
            if card is None:
                self._rendered_revision_id = None
                self.pages.setCurrentIndex(0)
                self._set_error(self.description_error, "")
                self._set_error(self.style_error, "")
                self.set_hotspot_error("")
                self.scene_edit.clear()
                self.style_combo.clear()
                self.hotspot_list.clear()
                self._render_hotspot_properties(document, None)
                return
            self.pages.setCurrentIndex(1)
            revision = card.active_revision
            same_revision = (
                previous_card_id == card.id
                and previous_revision_id == revision.id
            )
            if not same_revision:
                self._set_error(self.description_error, "")
                self._set_error(self.style_error, "")
                self.set_hotspot_error("")
            self._rendered_revision_id = revision.id
            self.scene_edit.setPlainText(
                description_draft
                if preserve_description and same_revision
                else revision.description
            )
            with QSignalBlocker(self.style_combo):
                self.style_combo.clear()
                self.style_combo.addItem("No style", None)
                for style in document.styles:
                    self.style_combo.addItem(style.name, style.id)
                self.style_combo.setCurrentIndex(
                    self._combo_index_for_data(
                        self.style_combo,
                        revision.style_id,
                    )
                )
            self._render_hotspots(document, revision)
            if (
                preserve_label
                and same_revision
                and self.selected_interaction_id == selected_interaction_id
            ):
                self.hotspot_label_edit.setText(label_draft)
            self.clear_background_button.setEnabled(revision.background is not None)
        finally:
            self._rendering = False

    def commit_revision_metadata(self) -> bool:
        if self._rendering or self.selected_card_id is None:
            return False
        card = self._selected_card()
        if card is None:
            return False
        value = self.scene_edit.toPlainText()
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

    def commit_card_metadata(self) -> bool:
        """Compatibility alias while MainWindow is migrated."""
        return self.commit_revision_metadata()

    def has_render_prompt_input(self) -> bool:
        card = self._selected_card()
        if card is None:
            return False
        style_id = self.style_combo.currentData()
        style = next(
            (
                candidate
                for candidate in self.controller.document.styles
                if candidate.id == style_id
            ),
            None,
        )
        return bool(
            self.scene_edit.toPlainText().strip()
            or (style is not None and style.prompt.strip())
        )

    def has_description_input(self) -> bool:
        return bool(
            self.selected_card_id is not None
            and self.scene_edit.toPlainText().strip()
        )

    def set_background_capabilities(
        self,
        *,
        can_generate: bool,
        generate_reason: str,
        can_import: bool,
        import_reason: str,
        can_clear: bool = False,
        clear_reason: str = "",
        busy: bool,
        generating: bool,
    ) -> None:
        self.generate_background_button.setEnabled(can_generate and not busy)
        self.generate_background_button.setText(
            "Generating…" if generating else "Generate Image"
        )
        self.generate_background_button.setToolTip(generate_reason)
        self.import_background_button.setEnabled(can_import and not busy)
        self.import_background_button.setToolTip(import_reason)
        self.clear_background_button.setEnabled(can_clear and not busy)
        self.clear_background_button.setToolTip(clear_reason)

    def set_scene_enrichment_capabilities(
        self,
        *,
        can_enrich: bool,
        reason: str,
        busy: bool,
    ) -> None:
        self.enrich_scene_button.setEnabled(can_enrich)
        self.enrich_scene_button.setText("Enriching…" if busy else "Enrich")
        self.enrich_scene_button.setToolTip(reason)

    def set_hotspot_remap_capabilities(
        self,
        *,
        can_remap: bool,
        reason: str,
        busy: bool,
    ) -> None:
        self.remap_hotspots_button.setEnabled(can_remap)
        self.remap_hotspots_button.setText("Remapping…" if busy else "Remap")
        self.remap_hotspots_button.setToolTip(reason)

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
        self._set_error(self.style_error, "")
        self.set_hotspot_error("")

    def _render_hotspots(
        self,
        document: Stack,
        revision: CardRevision,
    ) -> None:
        desired_id = self.selected_interaction_id
        interactions = (
            revision.hotspot_set.interactions
            if revision.hotspot_set is not None
            else ()
        )
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
                    if self.hotspot_list.item(row).data(Qt.ItemDataRole.UserRole)
                    == desired_id
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
        self.hotspot_label_edit.setEnabled(has_interaction)
        self.hotspot_destination_combo.setEnabled(has_interaction)
        self.delete_hotspot_button.setEnabled(has_interaction)
        current_row = self.hotspot_list.currentRow()
        self.move_hotspot_up_button.setEnabled(has_interaction and current_row > 0)
        self.move_hotspot_down_button.setEnabled(
            has_interaction and current_row < self.hotspot_list.count() - 1
        )
        with QSignalBlocker(self.hotspot_label_edit):
            self.hotspot_label_edit.setText(
                interaction.label if interaction is not None else ""
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
                target = interaction.action.target
                self.hotspot_destination_combo.setItemText(
                    0,
                    (
                        f"Unresolved ({target.target_name})"
                        if target.target_name
                        else "Unresolved"
                    ),
                )
                self.hotspot_destination_combo.setCurrentIndex(0)

    def _style_changed(self, index: int) -> None:
        if self._rendering or index < 0:
            return
        card = self._selected_card()
        if card is None:
            return
        style_id = self.style_combo.itemData(index)
        if style_id == card.active_revision.style_id:
            return
        self._execute(
            SetRevisionStyleCommand(
                card_id=card.id,
                revision_id=card.active_revision.id,
                style_id=style_id if isinstance(style_id, UUID) else None,
            ),
            error_label=self.style_error,
        )

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
        self._render_hotspot_properties(
            self.controller.document,
            self._selected_interaction(),
        )
        self.hotspot_selected.emit(interaction_id)

    def _add_hotspot(self) -> None:
        card = self._selected_card()
        if card is None:
            return
        hotspot_set = card.active_revision.hotspot_set or HotspotSet()
        used = {interaction.label.casefold() for interaction in hotspot_set.interactions}
        number = 1
        while f"Hotspot {number}".casefold() in used:
            number += 1
        interaction = Interaction(
            label=f"Hotspot {number}",
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

    def _commit_hotspot_label(self) -> None:
        card = self._selected_card()
        interaction = self._selected_interaction()
        if (
            self._rendering
            or card is None
            or interaction is None
            or self.hotspot_label_edit.text() == interaction.label
        ):
            return
        self._execute(
            RenameInteractionCommand(
                card_id=card.id,
                revision_id=card.active_revision.id,
                interaction_id=interaction.id,
                label=self.hotspot_label_edit.text(),
            )
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
        target_error = (
            error_label if error_label is not None else self.hotspot_error
        )
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
        if (
            undo_message is not None
            and token is not None
            and token != previous_token
        ):
            self.change_applied.emit(undo_message, token)
        return True

    @staticmethod
    def _set_error(label: QLabel, message: str) -> None:
        label.setText(message)
        label.setVisible(bool(message))

    def _selected_card(self) -> Card | None:
        return next(
            (
                card
                for card in self.controller.document.cards
                if card.id == self.selected_card_id
            ),
            None,
        )

    def _selected_interaction(self) -> Interaction | None:
        card = self._selected_card()
        interaction_id = self.selected_interaction_id
        if (
            card is None
            or interaction_id is None
            or card.active_revision.hotspot_set is None
        ):
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
            (
                index
                for index in range(combo.count())
                if combo.itemData(index) == value
            ),
            -1,
        )


__all__ = ["Inspector"]
