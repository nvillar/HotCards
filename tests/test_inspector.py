"""Focused offscreen tests for revision-local authoring controls."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QRect, QSize, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPalette, QPixmap
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QGroupBox,
    QLabel,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QToolButton,
)
from shiboken6 import isValid

from hotcards.application.commands import (
    ActivateRevisionCommand,
    DeleteCardCommand,
    RenameCardCommand,
    ReplaceRevisionBackgroundCommand,
    UpdateStyleCommand,
)
from hotcards.application.document_controller import DocumentController
from hotcards.domain.image_dimensions import (
    AspectRatio,
    ResolutionTier,
    higher_output_tiers,
    output_dimensions,
)
from hotcards.domain.models import (
    AcceptedEdit,
    Card,
    CardRevision,
    CurrentSourceSize,
    DerivedImageSourceSnapshot,
    DuplicateOperation,
    EditOperation,
    ExactOutputSize,
    GeneratedBackground,
    GeneratedSoundAsset,
    GenerateInputs,
    GenerateOperation,
    HotspotConditions,
    HotspotKeyChanges,
    HotspotSet,
    ImageOperationSettings,
    ImageOriginFacts,
    ImageProvenance,
    ImageSourceSnapshot,
    Interaction,
    KeyDefinition,
    NavigateAction,
    Point,
    Polygon,
    PresetOutputSize,
    ResolvedCardReference,
    SoundDefinition,
    SoundGenerationProvenance,
    Stack,
    StyleDefinition,
    UnresolvedCardReference,
    image_edit_lineage,
)
from hotcards.ui.inspector import Inspector
from hotcards.ui.utility_windows import KeyManagerWindow, StyleManagerWindow


@pytest.fixture
def application(
    qt_application: QApplication, monkeypatch: pytest.MonkeyPatch
) -> Iterator[QApplication]:
    inspectors: list[Inspector] = []
    initialize = Inspector.__init__

    def initialize_owned(inspector: Inspector, *args: object, **kwargs: object) -> None:
        initialize(inspector, *args, **kwargs)
        inspectors.append(inspector)

    monkeypatch.setattr(Inspector, "__init__", initialize_owned)
    try:
        yield qt_application
    finally:
        for inspector in reversed(inspectors):
            if isValid(inspector):
                inspector.close()
                inspector.deleteLater()
                QCoreApplication.sendPostedEvents(inspector, QEvent.Type.DeferredDelete)


def _polygon() -> Polygon:
    return Polygon(
        points=(
            Point(x=0.1, y=0.1),
            Point(x=0.4, y=0.1),
            Point(x=0.2, y=0.4),
        )
    )


def _interaction(label: str = "Door") -> Interaction:
    return Interaction(
        label=label,
        action=NavigateAction(target=UnresolvedCardReference()),
        polygons=(_polygon(),),
    )


def _background(size: tuple[int, int] = (512, 384)) -> GeneratedBackground:
    asset_id = uuid4()
    generated_at = datetime.now(UTC)
    return GeneratedBackground(
        id=asset_id,
        image_path=f"assets/cards/card/image-{asset_id}.png",
        provenance=ImageProvenance(
            authoring=GenerateOperation(
                inputs=GenerateInputs(
                    description="A courtyard",
                    output_size=ExactOutputSize(width=size[0], height=size[1]),
                ),
            ),
            origin=ImageOriginFacts(
                render_prompt="A courtyard",
                settings=ImageOperationSettings(
                    model_identifier="test",
                    mflux_version="test",
                    seed=7,
                    width=size[0],
                    height=size[1],
                    step_count=4,
                    generated_at=generated_at,
                    duration_seconds=1,
                ),
            ),
        ),
        created_at=generated_at,
    )


def _image_result(
    card: Card,
    operation: str,
    *,
    size: tuple[int, int] = (512, 384),
    instruction: str = "Open the gate.",
) -> GeneratedBackground:
    result = _background(size)
    if operation == "generate":
        return result
    revision = card.active_revision
    background = revision.background
    assert background is not None
    source_settings = background.provenance.settings
    source = DerivedImageSourceSnapshot(
        card_id=card.id,
        revision_id=revision.id,
        background_id=background.id,
        width=source_settings.width,
        height=source_settings.height,
        seed=source_settings.seed,
    )
    output_size = (
        CurrentSourceSize(width=size[0], height=size[1])
        if size == (source.width, source.height)
        else PresetOutputSize(tier=ResolutionTier.LARGE)
    )
    settings = result.provenance.settings
    assert operation == "edit"
    expanded_prompt = f"{instruction}\n\nHidden Style addendum"
    provenance = ImageProvenance(
        origin=ImageOriginFacts(
            render_prompt=expanded_prompt,
            settings=settings.model_copy(update={"seed": source.seed + 1}),
            edit_lineage=(
                *image_edit_lineage(background.provenance),
                AcceptedEdit(instruction=instruction, expanded_prompt=expanded_prompt),
            ),
        ),
        authoring=EditOperation(source=source, output_size=output_size, prompt_token_count=20),
    )
    return result.model_copy(update={"provenance": provenance})


def _card_with_edits(
    instructions: tuple[str, ...],
    *,
    size: tuple[int, int] = (512, 384),
) -> Card:
    revision = CardRevision(
        description="A courtyard",
        background=_background(size),
        hotspot_set=HotspotSet(interactions=(_interaction(),)),
    )
    card = Card(name="Courtyard", revisions=(revision,))
    for instruction in instructions:
        result = _image_result(card, "edit", instruction=instruction, size=size)
        revision = card.active_revision.model_copy(update={"background": result})
        card = card.model_copy(update={"revisions": (revision,)})
    return card


def _picker_item(inspector: Inspector, card_id: object) -> QListWidgetItem:
    item = next(
        (
            inspector.card_picker.card_list.item(index)
            for index in range(inspector.card_picker.card_list.count())
            if inspector.card_picker.card_list.item(index).data(Qt.ItemDataRole.UserRole) == card_id
        ),
        None,
    )
    assert item is not None
    return item


def _sound_picker_item(inspector: Inspector, sound_id: object) -> QListWidgetItem:
    item = next(
        (
            inspector.sound_picker.sound_list.item(index)
            for index in range(inspector.sound_picker.sound_list.count())
            if inspector.sound_picker.sound_list.item(index).data(Qt.ItemDataRole.UserRole)
            == sound_id
        ),
        None,
    )
    assert item is not None
    return item


def _choose_card(
    inspector: Inspector,
    button: QPushButton,
    card_id: object,
    application: QApplication,
) -> None:
    button.click()
    application.processEvents()
    item = _picker_item(inspector, card_id)
    assert item.flags() & Qt.ItemFlag.ItemIsEnabled
    inspector.card_picker.card_list.itemClicked.emit(item)
    application.processEvents()


def test_inspector_has_minimal_background_and_hotspot_hierarchy(
    application: QApplication,
) -> None:
    card = Card(
        name="Courtyard",
        revisions=(
            CardRevision(
                description="A moonlit courtyard",
                hotspot_set=HotspotSet(interactions=(_interaction(),)),
            ),
        ),
    )
    controller = DocumentController(Stack(name="Demo", cards=(card,)))
    inspector = Inspector(controller)
    inspector.render(controller.document, card.id)

    assert inspector.inspector_tabs.count() == 3
    assert inspector.inspector_tabs.tabText(0) == "Generate"
    assert inspector.inspector_tabs.tabText(1) == "Edit"
    assert inspector.inspector_tabs.tabText(2) == "Hotspots"
    root_layout = inspector.layout()
    assert root_layout is not None
    assert root_layout.contentsMargins().top() == 16
    assert root_layout.contentsMargins().bottom() == 16
    assert root_layout.contentsMargins().left() == inspector.style().pixelMetric(
        QStyle.PixelMetric.PM_LayoutLeftMargin
    )
    assert "image 1 and image 2" in inspector.description_edit.placeholderText()
    assert inspector.description_edit.minimumHeight() == round(
        (inspector.description_edit.fontMetrics().lineSpacing() * 10 + 20) * 1.25
    )
    assert inspector.description_edit.maximumHeight() > (inspector.description_edit.minimumHeight())
    assert inspector.generate_background_button.text() == "Generate Image"
    assert not isinstance(inspector.reference_panel, QFrame)
    assert inspector.reference_panel.layout().contentsMargins().isNull()
    for index in (
        inspector._generate_tab_index,
        inspector._edit_tab_index,
    ):
        tab = inspector.inspector_tabs.widget(index)
        assert not tab.findChildren(QGroupBox)
        assert all(
            label.text() not in {"New Image", "Evolve", "Edit"}
            for label in tab.findChildren(QLabel)
        )
    assert inspector.reference_panel.parentWidget() is inspector.description_edit.parentWidget()
    assert inspector.generate_background_button.parentWidget() is (
        inspector.description_edit.parentWidget()
    )
    content_layout = inspector.description_edit.parentWidget().layout()
    assert content_layout is not None
    assert content_layout.stretch(content_layout.indexOf(inspector.description_edit)) == 1
    field_positions = [
        content_layout.indexOf(widget)
        for widget in (
            inspector.description_edit,
            inspector.style_label,
            inspector.style_combo,
            inspector.reference_label,
            inspector.reference_panel,
            inspector.resolution_label,
            inspector.resolution_combo,
            inspector.generate_background_button,
        )
    ]
    assert all(position >= 0 for position in field_positions)
    assert field_positions == sorted(field_positions)
    assert content_layout.contentsMargins() == (
        inspector.edit_background_button.parentWidget().layout().contentsMargins()
    )
    for widget in (
        inspector.description_edit,
        inspector.style_combo,
        inspector.reference_panel,
        inspector.generate_background_button,
    ):
        assert inspector.inspector_tabs.widget(0).isAncestorOf(widget)
        assert not inspector.inspector_tabs.widget(1).isAncestorOf(widget)
    assert inspector.inspector_tabs.widget(1).isAncestorOf(inspector.edit_instruction_edit)
    assert inspector.style_combo.currentText() == "No Style"
    assert inspector.hotspot_target_label.text() == "Go to card"
    assert inspector.hotspot_target_label.font().pointSizeF() == (
        inspector.description_label.font().pointSizeF()
    )
    assert inspector.hotspot_when_label.text() == "When"
    assert inspector.hotspot_then_label.text() == "Then"
    assert isinstance(inspector.hotspot_when_panel, QGroupBox)
    assert isinstance(inspector.hotspot_then_panel, QGroupBox)
    assert not inspector.hotspot_when_panel.title()
    assert not inspector.hotspot_then_panel.title()
    assert not inspector.hotspot_when_panel.styleSheet()
    assert not inspector.hotspot_then_panel.styleSheet()
    assert inspector.hotspot_when_label.font().pointSizeF() == (
        inspector.edit_instruction_label.font().pointSizeF()
    )
    assert inspector.hotspot_then_label.font().pointSizeF() == (
        inspector.edit_instruction_label.font().pointSizeF()
    )
    assert inspector.hotspot_destination_button.text() == "Unresolved card"
    for button in (
        inspector.hotspot_destination_remove_button,
        inspector.reference_remove_button,
        inspector.additional_reference_remove_button,
        inspector.hotspot_sound_remove_button,
    ):
        assert button.text() == "−"
        assert button.size() == QSize(20, 20)
        assert button.testAttribute(Qt.WidgetAttribute.WA_LayoutUsesWidgetRect)
    hotspot_layout = inspector.hotspot_list.parentWidget().layout()
    assert hotspot_layout is not None
    assert inspector.hotspot_list.parentWidget().objectName() == "hotspotsInspectorContent"
    assert hotspot_layout.contentsMargins() == (
        inspector.edit_instruction_edit.parentWidget().layout().contentsMargins()
    )
    then_layout = inspector.hotspot_then_panel.layout()
    assert then_layout is not None
    assert inspector.key_change_controls_layout.indexOf(inspector.key_change_rows_widget) < (
        inspector.key_change_controls_layout.indexOf(inspector.key_change_add_row)
    )
    assert then_layout.indexOf(inspector.key_change_controls_layout) < (
        then_layout.indexOf(inspector.hotspot_target_label)
    )
    assert then_layout.indexOf(inspector.hotspot_target_label) < (
        then_layout.indexOf(inspector.hotspot_destination_row)
    )
    assert inspector.hotspot_sound_label.text() == "Play sound"
    assert then_layout.indexOf(inspector.hotspot_destination_row) < (
        then_layout.indexOf(inspector.hotspot_sound_label)
    )
    assert then_layout.indexOf(inspector.hotspot_sound_label) < (
        then_layout.indexOf(inspector.hotspot_sound_row)
    )
    assert inspector.hotspot_sound_button.text() == "Choose Sound…"
    assert inspector.condition_rows_layout.contentsMargins().isNull()
    assert inspector.key_change_rows_layout.contentsMargins().isNull()
    assert inspector.condition_controls_layout.spacing() == 3
    assert inspector.key_change_controls_layout.spacing() == 3
    assert inspector.hotspot_list.minimumHeight() == 120
    assert inspector.hotspot_list.maximumHeight() == 120
    assert hotspot_layout.stretch(hotspot_layout.indexOf(inspector.hotspot_list)) == 0
    hotspot_control_sizes = {
        button.size()
        for button in (
            inspector.move_hotspot_up_button,
            inspector.move_hotspot_down_button,
            inspector.add_hotspot_button,
            inspector.delete_hotspot_button,
        )
    }
    assert len(hotspot_control_sizes) == 1
    assert inspector.add_hotspot_button.font().pointSizeF() > (
        inspector.move_hotspot_up_button.font().pointSizeF()
    )
    hotspot_controls = hotspot_layout.itemAt(
        hotspot_layout.indexOf(inspector.hotspot_list) + 1
    ).layout()
    assert hotspot_controls is not None
    assert hotspot_controls.indexOf(inspector.delete_hotspot_button) < hotspot_controls.indexOf(
        inspector.add_hotspot_button
    )


def test_edit_uses_current_or_higher_size_and_emits_exact_inputs(
    application: QApplication,
) -> None:
    revision = CardRevision(
        description="A legacy courtyard",
        background=_background(),
    )
    card = Card(name="Courtyard", revisions=(revision,))
    controller = DocumentController(Stack(name="Demo", cards=(card,)))
    inspector = Inspector(controller)
    requests: list[tuple[object, object]] = []
    inspector.edit_background_requested.connect(
        lambda instruction, output: requests.append((instruction, output))
    )

    inspector.render(
        controller.document,
        card.id,
        image_source_size=(768, 576),
    )

    assert inspector.edit_resolution_combo.count() == 2
    assert inspector.edit_resolution_combo.itemText(0) == "Large"
    assert inspector.edit_resolution_combo.itemData(0) == CurrentSourceSize(
        width=768,
        height=576,
    )
    assert inspector.edit_resolution_combo.itemData(0, Qt.ItemDataRole.ToolTipRole) == "768 × 576"
    assert inspector.edit_resolution_combo.itemText(1) == "Full"
    inspector.edit_instruction_edit.setPlainText("  Open the gate.  ")
    inspector.edit_background_button.click()

    assert len(requests) == 1
    instruction, output = requests[0]
    assert instruction == "Open the gate."
    assert output.mode == "current"
    assert (output.width, output.height) == (768, 576)
    assert inspector.edit_background_button.toolTip() == (
        "Apply only the requested change to the current image."
    )


@pytest.mark.parametrize("operation", ("generate", "edit"))
@pytest.mark.parametrize("source_size", ((512, 384), (592, 448)))
@pytest.mark.parametrize("enlarge", (False, True))
def test_same_version_image_replacement_refreshes_sizes_history_not_drafts_or_hotspots(
    application: QApplication,
    operation: str,
    source_size: tuple[int, int],
    enlarge: bool,
) -> None:
    instructions = ("Make the gate blue.", "Make the gate blue.")
    card = _card_with_edits(instructions, size=source_size)
    controller = DocumentController(Stack(name="Demo", cards=(card,)))
    inspector = Inspector(controller)
    inspector.show()
    inspector.render(controller.document, card.id, image_source_size=source_size)
    inspector.select_interaction(card.active_revision.hotspot_set.interactions[0].id)
    selected = inspector.selected_interaction_id
    inspector.description_edit.setFocus()
    inspector.description_edit.setPlainText("A focused Description draft")
    inspector.set_edit_instruction("A pending instruction\nwith exact line breaks.")
    application.processEvents()
    draft = controller.edit_draft(card.id, card.active_revision.id)
    controls = (
        inspector.resolution_combo,
        inspector.edit_resolution_combo,
    )
    full = PresetOutputSize(tier=ResolutionTier.FULL)
    for combo in controls:
        combo.setCurrentIndex(inspector._combo_index_for_data(combo, full))
    choice_token = controller.current_undo_token
    inspector.render(controller.document, card.id, image_source_size=source_size)
    assert [combo.currentData() for combo in controls] == [full] * 2
    assert controller.current_undo_token == choice_token

    size = (768, 576) if enlarge else source_size
    current_card = controller.document.cards[0]
    result = _image_result(current_card, operation, size=size)
    controller.execute(
        ReplaceRevisionBackgroundCommand(
            card_id=card.id,
            revision_id=card.active_revision.id,
            background=result,
        )
    )
    token = controller.current_undo_token
    assert token != choice_token
    expected = (
        ()
        if operation == "generate"
        else ((*instructions, "Open the gate.") if operation == "edit" else instructions)
    )
    for transition, current_size, history in (
        ("applied", size, expected),
        ("undo", source_size, instructions),
        ("redo", size, expected),
    ):
        if transition == "undo":
            assert controller.undo()
        elif transition == "redo":
            assert controller.redo()
        expected_token = choice_token if transition == "undo" else token
        assert controller.current_undo_token == expected_token
        inspector.render(controller.document, card.id, image_source_size=current_size)
        assert controller.document.cards[0].active_revision.id == card.active_revision.id
        assert inspector.resolution_combo.toolTip() == (f"{current_size[0]} × {current_size[1]}")
        assert inspector.edit_resolution_combo.currentData() == CurrentSourceSize(
            width=current_size[0], height=current_size[1]
        )
        assert [
            inspector.edit_history_list.item(index).data(Qt.ItemDataRole.UserRole)
            for index in range(inspector.edit_history_list.count())
        ] == list(history)
        assert inspector.description_edit.toPlainText() == "A focused Description draft"
        assert controller.edit_draft(card.id, card.active_revision.id) == draft
        assert inspector.edit_instruction_edit.toPlainText() == draft.instruction
        assert inspector.selected_interaction_id == selected
        inspector.render(controller.document, card.id, image_source_size=current_size)
        assert controller.current_undo_token == expected_token
    inspector.close()


@pytest.mark.parametrize("operation", ("generate", "edit"))
@pytest.mark.parametrize("duplicate", (False, True))
def test_edit_history_uses_only_active_image_authored_lineage(
    application: QApplication, operation: str, duplicate: bool
) -> None:
    instructions = ("Open the gate.\nKeep its hinges.", "Paint it blue.", "Paint it blue.")
    card = _card_with_edits(instructions)
    result = _image_result(card, operation, instruction="Add a flower.")
    if duplicate:
        result = result.model_copy(
            update={
                "id": uuid4(),
                "provenance": ImageProvenance(
                    origin=result.provenance.origin,
                    authoring=DuplicateOperation(
                        source=ImageSourceSnapshot(
                            card_id=uuid4(),
                            revision_id=uuid4(),
                            background_id=result.id,
                        ),
                        original_authoring=result.provenance.authoring,
                    ),
                ),
            }
        )
    revision = CardRevision(description="Not history", background=result)
    card = card.model_copy(
        update={"revisions": (*card.revisions, revision), "active_revision_id": revision.id}
    )
    other = Card(name="No image")
    controller = DocumentController(Stack(name="Demo", cards=(card, other)))
    inspector = Inspector(controller)
    inspector.render(controller.document, card.id)
    inspector.set_edit_instruction("Do not overwrite this editor.")
    draft = controller.edit_draft(card.id, revision.id)
    expected = () if operation == "generate" else (*instructions, "Add a flower.")
    assert inspector.edit_history_list.wordWrap()
    assert inspector.edit_history_label.text() == "Edit History"
    assert not inspector.edit_history_list.isHidden()
    assert [
        inspector.edit_history_list.item(index).text()
        for index in range(inspector.edit_history_list.count())
    ] == list(expected)
    for index, instruction in enumerate(expected):
        item = inspector.edit_history_list.item(index)
        assert item.data(Qt.ItemDataRole.UserRole) == instruction
        assert item.toolTip() == instruction
        assert "Hidden Style" not in item.text()
    inspector.render(controller.document, card.id)
    assert controller.edit_draft(card.id, revision.id) == draft

    controller.execute(ActivateRevisionCommand(card_id=card.id, revision_id=card.revisions[0].id))
    inspector.render(controller.document, card.id)
    assert inspector.edit_history_list.count() == len(instructions)
    assert inspector.edit_instruction_edit.toPlainText() == ""
    inspector.render(controller.document, other.id)
    assert inspector.edit_history_list.count() == 0
    assert not inspector.edit_history_list.isHidden()
    inspector.render(controller.document, card.id)
    assert inspector.edit_history_list.count() == len(instructions)
    controller.execute(ActivateRevisionCommand(card_id=card.id, revision_id=revision.id))
    inspector.render(controller.document, card.id)
    assert inspector.edit_instruction_edit.toPlainText() == draft.instruction
    inspector.reset_context()
    assert inspector.edit_history_list.count() == 0
    inspector.render(controller.document, None)
    assert inspector.edit_history_list.count() == 0


@pytest.mark.parametrize("activation", ("click", "return", "enter", "space"))
def test_edit_history_recall_is_exact_accessible_and_has_no_document_side_effects(
    application: QApplication, activation: str
) -> None:
    instruction = "Open the gate.\nKeep the tree and every one of its long shadows."
    card = _card_with_edits((instruction, instruction))
    controller = DocumentController(Stack(name="Demo", cards=(card,)))
    inspector = Inspector(controller)
    inspector.render(controller.document, card.id)
    inspector.inspector_tabs.setCurrentIndex(inspector._edit_tab_index)
    inspector.resize(340, 800)
    inspector.show()
    inspector.set_edit_instruction("A different draft")
    application.processEvents()
    history = inspector.edit_history_list
    requests: list[object] = []
    inspector.generate_background_requested.connect(lambda: requests.append("generate"))
    inspector.edit_background_requested.connect(lambda *_args: requests.append("edit"))
    inspector.document_changed.connect(lambda *_args: requests.append("document"))
    inspector.hotspot_selected.connect(lambda *_args: requests.append("hotspot"))
    draft = controller.edit_draft(card.id, card.active_revision.id)
    document = controller.document
    token = controller.current_undo_token
    history.setCurrentRow(0)
    QTest.keyClick(history, Qt.Key.Key_Down)
    assert controller.edit_draft(card.id, card.active_revision.id) == draft
    assert history.currentRow() == 1
    if activation == "click":
        item = history.item(1)
        QTest.mouseClick(
            history.viewport(),
            Qt.MouseButton.LeftButton,
            pos=history.visualItemRect(item).center(),
        )
        history.itemActivated.emit(item)
    else:
        key = {
            "return": Qt.Key.Key_Return,
            "enter": Qt.Key.Key_Enter,
            "space": Qt.Key.Key_Space,
        }[activation]
        QTest.keyClick(history, key)
    assert inspector.edit_instruction_edit.toPlainText() == instruction
    recalled = controller.edit_draft(card.id, card.active_revision.id)
    assert recalled.generation_id != draft.generation_id
    refresh_signals: list[object] = []
    history.currentItemChanged.connect(lambda *_args: refresh_signals.append("selection"))
    history.instruction_requested.connect(lambda *_args: refresh_signals.append("recall"))
    inspector.render(controller.document, card.id)
    assert controller.edit_draft(card.id, card.active_revision.id) == recalled
    assert refresh_signals == []
    assert controller.document != document
    assert controller.current_undo_token == token
    assert requests == []
    assert inspector.inspector_tabs.currentIndex() == inspector._edit_tab_index
    inspector.close()


def test_edit_history_wraps_full_instructions_as_inspector_resizes(
    application: QApplication,
) -> None:
    instruction = (
        "Repaint the gate blue, keeping the existing shapes, reflections, and shadows. " * 12
    ).strip()
    card = _card_with_edits((instruction,))
    inspector = Inspector(DocumentController(Stack(name="Demo", cards=(card,))))
    inspector.render(inspector.controller.document, card.id)
    inspector.inspector_tabs.setCurrentIndex(inspector._edit_tab_index)
    inspector.resize(340, 850)
    inspector.show()
    application.processEvents()
    history = inspector.edit_history_list
    item = history.item(0)
    narrow_height = history.visualItemRect(item).height()
    assert narrow_height > history.fontMetrics().lineSpacing() * 5
    assert history.horizontalScrollBar().maximum() == 0
    assert history.textElideMode() == Qt.TextElideMode.ElideNone
    inspector.resize(700, 850)
    application.processEvents()
    assert history.visualItemRect(item).height() < narrow_height
    assert item.text() == instruction
    assert item.data(Qt.ItemDataRole.UserRole) == instruction
    inspector.close()


def test_empty_history_keeps_edit_controls_at_the_same_top_positions(
    application: QApplication,
) -> None:
    edited = _card_with_edits(("Open the gate.",))
    blank = Card(
        name="No edits",
        revisions=(CardRevision(description="A courtyard", background=_background()),),
    )
    controller = DocumentController(Stack(name="Demo", cards=(blank, edited)))
    inspector = Inspector(controller)
    inspector.resize(380, 900)
    inspector.inspector_tabs.setCurrentIndex(inspector._edit_tab_index)
    inspector.render(controller.document, blank.id, image_source_size=(512, 384))
    inspector.show()
    application.processEvents()
    controls = (
        inspector.edit_instruction_label,
        inspector.edit_instruction_edit,
        inspector.edit_resolution_label,
        inspector.edit_resolution_combo,
        inspector.edit_background_button,
        inspector.edit_history_label,
        inspector.edit_history_list,
    )
    empty_positions = [widget.geometry() for widget in controls]
    assert inspector.edit_history_list.isVisible()
    assert inspector.edit_history_list.count() == 0
    assert inspector.edit_instruction_label.y() < 30
    assert not hasattr(inspector, "edit_history_empty_label")

    inspector.render(controller.document, edited.id, image_source_size=(512, 384))
    application.processEvents()
    assert inspector.edit_history_list.count() == 1
    assert [widget.geometry() for widget in controls] == empty_positions
    inspector.render(controller.document, blank.id, image_source_size=(512, 384))
    application.processEvents()
    assert [widget.geometry() for widget in controls] == empty_positions
    assert inspector.edit_history_list.isVisible()
    inspector.close()


def test_history_delegate_adds_space_and_draws_only_between_entries(
    application: QApplication,
) -> None:
    card = _card_with_edits(("Open the gate.", "Paint it blue."))
    inspector = Inspector(DocumentController(Stack(name="Demo", cards=(card,))))
    inspector.render(inspector.controller.document, card.id)
    history = inspector.edit_history_list
    delegate = history.itemDelegate()
    ordinary = QStyledItemDelegate(history)
    option = QStyleOptionViewItem()
    option.initFrom(history)
    option.widget = history
    option.rect = QRect(0, 0, 320, 70)
    separator_color = QColor(93, 81, 69)
    option.palette.setColor(QPalette.ColorRole.Mid, separator_color)
    for row in range(2):
        index = history.model().index(row, 0)
        assert delegate.sizeHint(option, index).height() == (
            ordinary.sizeHint(option, index).height() + 12
        )
        image = QImage(option.rect.size(), QImage.Format.Format_ARGB32)
        image.fill(Qt.GlobalColor.white)
        painter = QPainter(image)
        delegate.paint(painter, option, index)
        painter.end()
        has_separator = image.pixelColor(10, option.rect.bottom()) == separator_color
        assert has_separator == (row == 0)
    inspector.close()


def test_key_manager_manages_global_names_and_lists_hotspot_usages(
    application: QApplication,
) -> None:
    red_key = KeyDefinition(name="Red key")
    interaction = Interaction(
        key_changes=HotspotKeyChanges(remove=(red_key.id,)),
        conditions=HotspotConditions(requires=(red_key.id,)),
        polygons=(_polygon(),),
    )
    card = Card(
        name="Castle",
        revisions=(CardRevision(hotspot_set=HotspotSet(interactions=(interaction,))),),
    )
    controller = DocumentController(Stack(name="Demo", keys=(red_key,), cards=(card,)))
    parent = Inspector(controller)
    manager = KeyManagerWindow(controller, parent)

    assert manager.key_list.currentItem().text() == "Red key — 1 use"
    assert manager.name_edit.text() == "Red key"
    assert not manager.delete_button.isEnabled()
    assert manager.usage_list.count() == 1
    assert manager.usage_list.item(0).text() == "Castle V1: Lose Red key"
    assert not manager.usage_list.styleSheet()

    manager.name_edit.setText("Ruby key")
    usage_item = manager.usage_list.item(0)
    manager._editing_finished(
        manager.usage_list,
        Qt.FocusReason.MouseFocusReason,
    )
    assert controller.document.keys[0].name == "Ruby key"
    assert manager.usage_list.item(0) is usage_item
    assert manager.usage_list.item(0).text() == "Castle V1: Lose Ruby key"
    hotspot_set = controller.document.cards[0].active_revision.hotspot_set
    assert hotspot_set is not None
    assert hotspot_set.interactions[0].label == "Lose Ruby key"

    requested: list[tuple[object, object, object]] = []
    manager.hotspot_usage_requested.connect(
        lambda card_id, revision_id, interaction_id: requested.append(
            (card_id, revision_id, interaction_id)
        )
    )
    manager.usage_list.setCurrentRow(0)
    manager.usage_list.itemDoubleClicked.emit(manager.usage_list.item(0))
    assert requested == [(card.id, card.active_revision.id, interaction.id)]
    assert not hasattr(manager, "show_usage_button")

    manager.add_button.click()
    assert [key.name for key in controller.document.keys] == [
        "Ruby key",
        "New Key",
    ]
    assert manager.delete_button.isEnabled()
    manager.delete_button.click()
    assert [key.name for key in controller.document.keys] == ["Ruby key"]
    manager.close()


def test_key_manager_invalid_name_drafts_are_cancelable_and_do_not_block_delete(
    application: QApplication,
) -> None:
    first = KeyDefinition(name="First")
    second = KeyDefinition(name="Second")
    controller = DocumentController(Stack(name="Demo", keys=(first, second)))
    parent = Inspector(controller)
    manager = KeyManagerWindow(controller, parent)
    discarded: list[str] = []
    manager.invalid_name_discarded.connect(discarded.append)

    manager.name_edit.setText("Second")
    assert not manager.error_label.isHidden()
    assert "unique" in manager.error_label.text()
    assert controller.document.key_by_id(first.id).name == "First"

    QTest.keyClick(manager.name_edit, Qt.Key.Key_Return)
    assert manager.name_edit.text() == "Second"
    assert discarded == []

    QTest.keyClick(manager.name_edit, Qt.Key.Key_Escape)
    assert manager.name_edit.text() == "First"
    assert manager.error_label.isHidden()
    assert discarded == []

    manager.name_edit.setText(" ")
    manager.done_button.click()
    application.processEvents()
    assert discarded == ["First"]
    assert controller.document.key_by_id(first.id).name == "First"

    manager = KeyManagerWindow(controller, parent)
    manager.name_edit.setText("Second")
    manager.delete_button.click()
    assert [key.name for key in controller.document.keys] == ["Second"]
    manager.close()

    controller = DocumentController(Stack(name="Demo", keys=(first, second)))
    manager = KeyManagerWindow(controller, parent)
    manager.name_edit.setText("Renamed")
    manager.delete_button.click()
    assert [key.name for key in controller.document.keys] == ["Second"]
    controller.undo()
    assert controller.document.key_by_id(first.id).name == "Renamed"
    manager.close()


def test_hotspot_pipeline_edits_conditions_changes_and_navigation(
    application: QApplication,
) -> None:
    red_key = KeyDefinition(name="Red key")
    door_open = KeyDefinition(name="Door open")
    destination = Card(name="Castle")
    interaction = Interaction(
        conditions=HotspotConditions(requires=(red_key.id,)),
        key_changes=HotspotKeyChanges(
            remove=(red_key.id,),
            grant=(door_open.id,),
        ),
        polygons=(_polygon(),),
        action=NavigateAction(target=ResolvedCardReference(target_card_id=destination.id)),
    )
    source = Card(
        name="Source",
        revisions=(CardRevision(hotspot_set=HotspotSet(interactions=(interaction,))),),
    )
    controller = DocumentController(
        Stack(
            name="Demo",
            keys=(red_key, door_open),
            cards=(source, destination),
        )
    )
    inspector = Inspector(controller)
    inspector.render(controller.document, source.id)

    assert not hasattr(inspector, "hotspot_name_edit")
    assert inspector.hotspot_list.currentItem().text() == "Go to Castle"
    assert inspector.condition_rows_layout.count() == 1
    condition_row = inspector.condition_rows_layout.itemAt(0).widget()
    assert condition_row is not None
    condition_layout = condition_row.layout()
    assert condition_layout is not None
    condition_state = condition_layout.itemAt(0).widget()
    condition_key = condition_layout.itemAt(1).widget()
    assert isinstance(condition_state, QComboBox)
    assert isinstance(condition_key, QComboBox)
    assert condition_state.currentText() == "Has"
    assert condition_key.currentData() == red_key.id
    condition_remove = condition_layout.itemAt(2).widget()
    assert isinstance(condition_remove, QToolButton)
    assert condition_layout.itemAt(2).alignment() == Qt.AlignmentFlag.AlignVCenter
    assert condition_remove.width() == condition_remove.height()
    assert condition_remove.size() == QSize(20, 20)
    assert inspector.key_change_rows_layout.count() == 2
    remove_change_row = inspector.key_change_rows_layout.itemAt(0).widget()
    grant_change_row = inspector.key_change_rows_layout.itemAt(1).widget()
    assert remove_change_row is not None
    assert grant_change_row is not None
    remove_change_layout = remove_change_row.layout()
    grant_change_layout = grant_change_row.layout()
    assert remove_change_layout is not None
    assert grant_change_layout is not None
    remove_change = remove_change_layout.itemAt(0).widget()
    grant_change = grant_change_layout.itemAt(0).widget()
    assert isinstance(remove_change, QComboBox)
    assert isinstance(grant_change, QComboBox)
    assert remove_change.currentText() == "Lose"
    assert grant_change.currentText() == "Gain"
    key_change_remove = remove_change_layout.itemAt(2).widget()
    assert isinstance(key_change_remove, QToolButton)
    assert remove_change_layout.itemAt(2).alignment() == Qt.AlignmentFlag.AlignVCenter
    assert key_change_remove.width() == key_change_remove.height()
    assert key_change_remove.size() == QSize(20, 20)
    assert inspector.hotspot_target_label.text() == "Go to card"
    grant_key = grant_change_layout.itemAt(1).widget()
    assert isinstance(grant_key, QComboBox)
    grant_key.setCurrentIndex(
        next(index for index in range(grant_key.count()) if grant_key.itemData(index) == red_key.id)
    )
    assert "both gained and lost" in inspector.hotspot_error.text()
    assert not inspector.hotspot_error.isHidden()
    changed = controller.document.cards[0].active_revision.hotspot_set
    assert changed is not None
    assert changed.interactions[0].key_changes.grant == (door_open.id,)

    inspector.hotspot_destination_remove_button.click()
    application.processEvents()
    changed = controller.document.cards[0].active_revision.hotspot_set
    assert changed is not None
    assert changed.interactions[0].action is None


def test_empty_hotspot_rule_sections_show_only_disabled_add_actions_without_keys(
    application: QApplication,
) -> None:
    interaction = Interaction(polygons=(_polygon(),))
    card = Card(
        name="Card",
        revisions=(CardRevision(hotspot_set=HotspotSet(interactions=(interaction,))),),
    )
    inspector = Inspector(DocumentController(Stack(name="Demo", cards=(card,))))

    inspector.render(inspector.controller.document, card.id)

    assert inspector.condition_rows_widget.isHidden()
    assert inspector.key_change_rows_widget.isHidden()
    assert inspector.add_condition_button.isVisibleTo(inspector.hotspot_when_panel)
    assert inspector.add_key_change_button.isVisibleTo(inspector.hotspot_then_panel)
    assert not inspector.add_condition_button.isEnabled()
    assert not inspector.add_key_change_button.isEnabled()
    assert "Keys window" in inspector.add_condition_button.toolTip()
    assert "Keys window" in inspector.add_key_change_button.toolTip()
    assert inspector.hotspot_destination_button.text() == "Choose Destination…"
    assert not inspector.hotspot_destination_remove_button.isEnabled()


def test_hotspot_sound_picker_assigns_and_removes_catalog_sound(
    application: QApplication,
) -> None:
    sound = SoundDefinition(name="Door knock")
    interaction = Interaction(polygons=(_polygon(),))
    card = Card(
        name="Card",
        revisions=(CardRevision(hotspot_set=HotspotSet(interactions=(interaction,))),),
    )
    controller = DocumentController(Stack(name="Demo", sounds=(sound,), cards=(card,)))
    inspector = Inspector(controller)
    inspector.render(controller.document, card.id)

    inspector.hotspot_sound_button.click()
    application.processEvents()
    tile = inspector.sound_picker.tile(sound.id)
    assert tile is not None
    assert not tile.play_button.isEnabled()
    QTest.mouseClick(tile.name_label, Qt.MouseButton.LeftButton)
    application.processEvents()

    hotspot_set = controller.document.cards[0].active_revision.hotspot_set
    assert hotspot_set is not None
    assert hotspot_set.interactions[0].sound_id == sound.id
    assert inspector.hotspot_sound_button.text() == sound.name
    assert inspector.hotspot_sound_remove_button.isEnabled()

    inspector.hotspot_sound_remove_button.click()
    application.processEvents()

    hotspot_set = controller.document.cards[0].active_revision.hotspot_set
    assert hotspot_set is not None
    assert hotspot_set.interactions[0].sound_id is None


def test_hotspot_sound_picker_searches_previews_and_stops_on_cancel(
    application: QApplication,
) -> None:
    generated_at = datetime.now(UTC)
    generated = SoundDefinition(
        name="Door knock",
        generated=GeneratedSoundAsset(
            audio_path="assets/sounds/door.wav",
            provenance=SoundGenerationProvenance(
                prompt="A door knock",
                duration_seconds=2,
                seed=1,
                generation_duration_milliseconds=10,
            ),
            created_at=generated_at,
        ),
    )
    ungenerated = SoundDefinition(name="Wind")
    interaction = Interaction(polygons=(_polygon(),))
    card = Card(
        name="Card",
        revisions=(CardRevision(hotspot_set=HotspotSet(interactions=(interaction,))),),
    )
    controller = DocumentController(
        Stack(name="Demo", sounds=(generated, ungenerated), cards=(card,))
    )
    inspector = Inspector(controller)
    inspector.render(controller.document, card.id)
    inspector.show()
    previews: list[object] = []
    stops: list[bool] = []
    inspector.sound_preview_requested.connect(previews.append)
    inspector.sound_preview_stop_requested.connect(lambda: stops.append(True))

    inspector.hotspot_sound_button.click()
    application.processEvents()

    assert inspector.sound_picker.windowTitle() == "Select Sound"
    assert inspector.sound_picker.windowModality() == Qt.WindowModality.NonModal
    assert inspector.sound_picker.windowFlags() & Qt.WindowType.Tool
    assert inspector.sound_picker.cancel_button.text() == "Cancel"
    viewport_width = inspector.sound_picker.sound_list.viewport().width()
    column_width = inspector.sound_picker.sound_list.gridSize().width()
    assert viewport_width >= column_width * 3
    assert viewport_width < column_width * 4
    generated_tile = inspector.sound_picker.tile(generated.id)
    ungenerated_tile = inspector.sound_picker.tile(ungenerated.id)
    assert generated_tile is not None
    assert ungenerated_tile is not None
    assert generated_tile.play_button.isEnabled()
    assert not ungenerated_tile.play_button.isEnabled()

    inspector.sound_picker.search_edit.setText("missing")
    inspector.sound_picker.search_edit.returnPressed.emit()
    hotspot_set = controller.document.cards[0].active_revision.hotspot_set
    assert hotspot_set is not None
    assert hotspot_set.interactions[0].sound_id is None

    inspector.sound_picker.search_edit.setText("door")
    assert not _sound_picker_item(inspector, generated.id).isHidden()
    assert _sound_picker_item(inspector, ungenerated.id).isHidden()
    generated_tile.play_button.click()
    assert previews == [generated.id]

    inspector.sound_picker.cancel_button.click()
    application.processEvents()

    assert not inspector.sound_picker.isVisible()
    assert stops == [True]
    hotspot_set = controller.document.cards[0].active_revision.hotspot_set
    assert hotspot_set is not None
    assert hotspot_set.interactions[0].sound_id is None

    resized = QSize(620, 410)
    inspector.sound_picker.resize(resized)
    inspector.hotspot_sound_button.click()
    application.processEvents()
    assert inspector.sound_picker.size() == resized
    generated_tile = inspector.sound_picker.tile(generated.id)
    assert generated_tile is not None
    generated_tile.play_button.click()
    inspector.hide()
    application.processEvents()
    assert previews == [generated.id, generated.id]
    assert stops == [True, True]
    assert not inspector.sound_picker.isVisible()
    inspector.close()


def test_hotspot_key_add_actions_use_latest_eligible_catalog_key(
    application: QApplication,
) -> None:
    first_key = KeyDefinition(name="First")
    latest_key = KeyDefinition(name="Latest")
    interaction = Interaction(polygons=(_polygon(),))
    card = Card(
        name="Card",
        revisions=(CardRevision(hotspot_set=HotspotSet(interactions=(interaction,))),),
    )
    controller = DocumentController(Stack(name="Demo", keys=(first_key, latest_key), cards=(card,)))
    inspector = Inspector(controller)
    inspector.render(controller.document, card.id)

    inspector.add_condition_button.click()
    inspector.add_key_change_button.click()

    changed = controller.document.cards[0].active_revision.hotspot_set
    assert changed is not None
    assert changed.interactions[0].conditions.requires == (latest_key.id,)
    assert changed.interactions[0].key_changes.grant == (latest_key.id,)
    assert inspector.condition_rows_layout.count() == 1
    assert inspector.key_change_rows_layout.count() == 1


def test_hotspot_key_rows_fit_long_names_with_visible_remove_controls(
    application: QApplication,
) -> None:
    key = KeyDefinition(
        name="A very long authored Key name that must not widen the inspector panel"
    )
    interaction = Interaction(
        conditions=HotspotConditions(requires=(key.id,)),
        key_changes=HotspotKeyChanges(grant=(key.id,)),
        polygons=(_polygon(),),
    )
    card = Card(
        name="Card",
        revisions=(CardRevision(hotspot_set=HotspotSet(interactions=(interaction,))),),
    )
    inspector = Inspector(DocumentController(Stack(name="Demo", keys=(key,), cards=(card,))))
    inspector.resize(320, 700)
    inspector.render(inspector.controller.document, card.id)
    inspector.show()
    application.processEvents()

    for rows_layout in (
        inspector.condition_rows_layout,
        inspector.key_change_rows_layout,
    ):
        row = rows_layout.itemAt(0).widget()
        assert row is not None
        row_layout = row.layout()
        assert row_layout is not None
        key_combo = row_layout.itemAt(1).widget()
        remove_button = row_layout.itemAt(2).widget()
        assert isinstance(key_combo, QComboBox)
        assert isinstance(remove_button, QToolButton)
        assert row_layout.spacing() == 3
        assert key_combo.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Ignored
        assert key_combo.toolTip() == key.name
        assert remove_button.geometry().right() <= row.contentsRect().right()

    inspector.close()


@pytest.mark.parametrize("rule", ("condition", "key_change"))
def test_invalid_key_rule_preserves_image_sizes_drafts_and_selection(
    application: QApplication, rule: str
) -> None:
    first, second = KeyDefinition(name="First"), KeyDefinition(name="Second")
    interaction = Interaction(
        conditions=HotspotConditions(requires=(first.id,), forbids=(second.id,)),
        key_changes=HotspotKeyChanges(grant=(first.id,), remove=(second.id,)),
        polygons=(_polygon(),),
    )
    card = Card(
        name="Card",
        revisions=(
            CardRevision(
                description="A courtyard",
                background=_background((640, 480)),
                hotspot_set=HotspotSet(interactions=(interaction,)),
            ),
        ),
    )
    controller = DocumentController(Stack(name="Demo", keys=(first, second), cards=(card,)))
    inspector = Inspector(controller)
    inspector.render(controller.document, card.id, image_source_size=(640, 480))
    full = PresetOutputSize(tier=ResolutionTier.FULL)
    inspector.resolution_combo.setCurrentIndex(
        inspector._combo_index_for_data(inspector.resolution_combo, full)
    )
    inspector.edit_resolution_combo.setCurrentIndex(
        inspector._combo_index_for_data(inspector.edit_resolution_combo, full)
    )
    inspector.set_edit_instruction("Keep this unfinished Edit.")
    document = controller.document
    token = controller.current_undo_token
    if rule == "condition":
        inspector._change_condition(second.id, "forbids", first.id, "forbids")
    else:
        inspector._change_key_change(second.id, "remove", first.id, "remove")
    assert inspector.hotspot_error.text()
    assert controller.document == document
    assert controller.current_undo_token == token
    assert inspector.selected_interaction_id == interaction.id
    assert inspector.resolution_combo.currentData() == full
    assert inspector.edit_resolution_combo.currentData() == full
    assert inspector.edit_instruction_edit.toPlainText() == "Keep this unfinished Edit."
    assert inspector._image_source_size == (640, 480)
    assert any(
        inspector.edit_resolution_combo.itemData(index) == CurrentSourceSize(width=640, height=480)
        for index in range(inspector.edit_resolution_combo.count())
    )


def test_description_edits_target_active_revision(
    application: QApplication,
) -> None:
    card = Card(
        name="Card",
        revisions=(CardRevision(description="Old"),),
    )
    controller = DocumentController(Stack(name="Demo", cards=(card,)))
    inspector = Inspector(controller)
    inspector.render(controller.document, card.id)

    inspector.description_edit.setPlainText("New description")
    assert inspector.description_edit.toPlainText() == "New description"
    assert controller.document.cards[0].active_revision.description == "Old"
    assert inspector.commit_revision_metadata()
    token = controller.current_undo_token
    assert inspector.commit_revision_metadata()
    assert controller.current_undo_token == token

    revision = controller.document.cards[0].active_revision
    assert revision.description == "New description"
    assert controller.undo()
    assert controller.document.cards[0].active_revision.description == "Old"
    inspector.render(controller.document, card.id)
    assert inspector.description_edit.toPlainText() == "Old"
    assert controller.redo()
    inspector.render(controller.document, card.id)
    assert inspector.description_edit.toPlainText() == "New description"


def test_generate_style_selection_is_undoable(
    application: QApplication,
) -> None:
    style = StyleDefinition(name="Ink", prompt_text="Black ink")
    card = Card(name="Card")
    controller = DocumentController(
        Stack(name="Demo", styles=(style,), new_card_style_id=None, cards=(card,))
    )
    inspector = Inspector(controller)
    inspector.render(controller.document, card.id)
    changes: list[object] = []
    inspector.document_changed.connect(changes.append)
    applied: list[tuple[str, object]] = []
    inspector.change_applied.connect(lambda message, token: applied.append((message, token)))
    inspector.style_combo.setCurrentIndex(
        inspector._combo_index_for_data(inspector.style_combo, style.id)
    )
    assert len(changes) == 1
    assert controller.document.cards[0].active_revision.style_id == style.id
    assert controller.document.new_card_style_id == style.id
    assert inspector.style_combo.currentData() == style.id
    assert applied == [("Style changed", controller.current_undo_token)]
    assert controller.undo_if_current(applied[-1][1])
    inspector.render(controller.document, card.id)
    assert inspector.style_combo.currentData() is None
    assert controller.document.new_card_style_id is None
    assert not controller.can_undo
    assert controller.redo()
    inspector.render(controller.document, card.id)
    assert inspector.style_combo.currentData() == style.id
    inspector.style_combo.setCurrentIndex(inspector.style_combo.findData(None))
    assert len(changes) == 2
    assert controller.document.cards[0].active_revision.style_id is None
    assert controller.document.new_card_style_id is None


@pytest.mark.parametrize("aspect_ratio", tuple(AspectRatio))
def test_resolution_selector_uses_tier_names_and_dimension_tooltips(
    application: QApplication,
    aspect_ratio: AspectRatio,
) -> None:
    card = Card(name="Card")
    controller = DocumentController(Stack(name="Demo", aspect_ratio=aspect_ratio, cards=(card,)))
    inspector = Inspector(controller)
    inspector.render(controller.document, card.id)

    assert [
        inspector.resolution_combo.itemText(index)
        for index in range(inspector.resolution_combo.count())
    ] == [tier.label for tier in ResolutionTier]
    assert [
        inspector.resolution_combo.itemData(index, Qt.ItemDataRole.ToolTipRole)
        for index in range(inspector.resolution_combo.count())
    ] == [
        f"{output_dimensions(tier, aspect_ratio)[0]} × {output_dimensions(tier, aspect_ratio)[1]}"
        for tier in ResolutionTier
    ]
    assert inspector.resolution_combo.currentData() == PresetOutputSize(tier=ResolutionTier.MEDIUM)
    width, height = output_dimensions(
        ResolutionTier.MEDIUM,
        aspect_ratio,
    )
    assert (
        f"Resolution: Medium ({width} × {height})" in inspector.generate_background_button.toolTip()
    )


def test_resolution_selector_is_revision_local_and_undoable(
    application: QApplication,
) -> None:
    first = CardRevision()
    second = CardRevision(generate_output_size=PresetOutputSize(tier=ResolutionTier.SMALL))
    card = Card(
        name="Card",
        revisions=(first, second),
        active_revision_id=first.id,
    )
    autosaves: list[Stack] = []
    controller = DocumentController(
        Stack(name="Demo", cards=(card,)),
        autosave_hook=autosaves.append,
    )
    inspector = Inspector(controller)
    inspector.render(controller.document, card.id)
    applied: list[tuple[str, object]] = []
    inspector.change_applied.connect(lambda message, token: applied.append((message, token)))

    inspector.resolution_combo.setCurrentIndex(
        inspector._combo_index_for_data(
            inspector.resolution_combo,
            PresetOutputSize(tier=ResolutionTier.FULL),
        )
    )

    revisions = controller.document.cards[0].revisions
    assert revisions[0].generate_output_size == PresetOutputSize(tier=ResolutionTier.FULL)
    assert revisions[1].generate_output_size == PresetOutputSize(tier=ResolutionTier.SMALL)
    assert autosaves[-1].cards[0].active_revision.generate_output_size == PresetOutputSize(
        tier=ResolutionTier.FULL
    )
    assert not applied
    undo_token = controller.current_undo_token
    assert undo_token is not None
    assert controller.undo_if_current(undo_token)
    inspector.render(controller.document, card.id)
    assert inspector.resolution_combo.currentData() == PresetOutputSize(tier=ResolutionTier.MEDIUM)
    assert controller.redo()
    inspector.render(controller.document, card.id)
    assert inspector.resolution_combo.currentData() == PresetOutputSize(tier=ResolutionTier.FULL)


def test_generate_selects_current_size_on_navigation_and_syncs_it_on_request(
    application: QApplication,
) -> None:
    revision = CardRevision(
        generate_output_size=PresetOutputSize(tier=ResolutionTier.MEDIUM),
        background=_background(),
    )
    card = Card(name="Card", revisions=(revision,))
    controller = DocumentController(Stack(name="Demo", cards=(card,)))
    inspector = Inspector(controller)

    inspector.render(
        controller.document,
        card.id,
        image_source_size=(1024, 768),
    )

    assert inspector.resolution_combo.itemText(3) == "Full"
    assert inspector.resolution_combo.currentData() == PresetOutputSize(tier=ResolutionTier.FULL)
    assert "Resolution: Full (1024 × 768)" in (inspector.generate_background_button.toolTip())
    assert controller.document.cards[0].active_revision.generate_output_size == PresetOutputSize(
        tier=ResolutionTier.MEDIUM
    )

    inspector.render(
        controller.document,
        card.id,
        image_source_size=(1024, 768),
    )
    assert inspector.resolution_combo.currentData() == PresetOutputSize(tier=ResolutionTier.FULL)

    requested: list[None] = []
    inspector.generate_background_requested.connect(lambda: requested.append(None))
    inspector.generate_background_button.click()

    assert requested == [None]
    assert controller.document.cards[0].active_revision.generate_output_size == PresetOutputSize(
        tier=ResolutionTier.FULL
    )


def test_generate_output_size_inserts_selectable_exact_current_size(
    application: QApplication,
) -> None:
    revision = CardRevision(background=_background())
    card = Card(name="Card", revisions=(revision,))
    controller = DocumentController(Stack(name="Demo", cards=(card,)))
    inspector = Inspector(controller)
    inspector.render(
        controller.document,
        card.id,
        image_source_size=(592, 448),
    )

    labels = [
        inspector.resolution_combo.itemText(index)
        for index in range(inspector.resolution_combo.count())
    ]
    assert labels == [
        "Small",
        "Medium",
        "Current",
        "Large",
        "Full",
    ]
    assert inspector.resolution_combo.currentText() == "Current"
    assert inspector.resolution_combo.toolTip() == "592 × 448"
    inspector.generate_background_button.click()
    assert controller.document.cards[0].active_revision.generate_output_size == (
        ExactOutputSize(width=592, height=448)
    )


def test_card_navigation_selects_each_current_resolution(
    application: QApplication,
) -> None:
    first = Card(name="First", revisions=(CardRevision(background=_background()),))
    second = Card(name="Second", revisions=(CardRevision(background=_background()),))
    controller = DocumentController(Stack(name="Demo", cards=(first, second)))
    inspector = Inspector(controller)

    inspector.render(
        controller.document,
        first.id,
        image_source_size=(512, 384),
    )
    inspector.resolution_combo.setCurrentIndex(
        inspector._combo_index_for_data(
            inspector.resolution_combo,
            PresetOutputSize(tier=ResolutionTier.FULL),
        )
    )
    inspector.render(
        controller.document,
        second.id,
        image_source_size=(768, 576),
    )

    current = CurrentSourceSize(width=768, height=576)
    assert inspector.resolution_combo.currentData() == PresetOutputSize(tier=ResolutionTier.LARGE)
    assert inspector.edit_resolution_combo.currentData() == current


@pytest.mark.parametrize("source_size", ((641, 480), (1008, 784)))
def test_generate_shows_invalid_current_size_without_selecting_or_persisting_it(
    application: QApplication,
    source_size: tuple[int, int],
) -> None:
    revision = CardRevision(background=_background())
    card = Card(name="Card", revisions=(revision,))
    controller = DocumentController(Stack(name="Demo", cards=(card,)))
    inspector = Inspector(controller)
    inspector.render(
        controller.document,
        card.id,
        image_source_size=source_size,
    )

    labels = [
        inspector.resolution_combo.itemText(index)
        for index in range(inspector.resolution_combo.count())
    ]
    unavailable_index = labels.index("Current")
    item = inspector.resolution_combo.model().item(unavailable_index)
    assert item is not None
    assert not item.isEnabled()
    assert inspector.resolution_combo.currentData() == PresetOutputSize(tier=ResolutionTier.MEDIUM)
    assert "Unavailable" in item.toolTip()
    assert f"{source_size[0]} × {source_size[1]}" in item.toolTip()

    inspector.resolution_combo.setCurrentIndex(unavailable_index)

    assert inspector.resolution_combo.currentData() == PresetOutputSize(tier=ResolutionTier.MEDIUM)
    assert controller.document.cards[0].active_revision.generate_output_size == (
        PresetOutputSize(tier=ResolutionTier.MEDIUM)
    )


def test_edit_resolution_inserts_nonstandard_current_and_only_higher_tiers(
    application: QApplication,
) -> None:
    revision = CardRevision(background=_background())
    card = Card(name="Card", revisions=(revision,))
    controller = DocumentController(Stack(name="Demo", cards=(card,)))
    inspector = Inspector(controller)

    inspector.render(
        controller.document,
        card.id,
        image_source_size=(640, 480),
    )

    assert [
        inspector.edit_resolution_combo.itemText(index)
        for index in range(inspector.edit_resolution_combo.count())
    ] == [
        "Current",
        "Large",
        "Full",
    ]


@pytest.mark.parametrize("source_size", ((641, 480), (1008, 784)))
def test_edit_shows_invalid_current_size_as_unavailable(
    application: QApplication,
    source_size: tuple[int, int],
) -> None:
    revision = CardRevision(
        description="A courtyard",
        background=_background(),
    )
    card = Card(name="Card", revisions=(revision,))
    controller = DocumentController(Stack(name="Demo", cards=(card,)))
    inspector = Inspector(controller)
    inspector.render(
        controller.document,
        card.id,
        image_source_size=source_size,
    )
    higher_tiers = higher_output_tiers(
        source_size[0],
        source_size[1],
        AspectRatio.LANDSCAPE,
    )

    edit_labels = [
        inspector.edit_resolution_combo.itemText(index)
        for index in range(inspector.edit_resolution_combo.count())
    ]
    edit_unavailable = edit_labels.index("Current")
    edit_item = inspector.edit_resolution_combo.model().item(edit_unavailable)
    assert edit_item is not None
    assert not edit_item.isEnabled()
    if higher_tiers:
        assert inspector.edit_resolution_combo.currentData() == (
            PresetOutputSize(tier=higher_tiers[0])
        )
        inspector.edit_resolution_combo.setCurrentIndex(edit_unavailable)
        assert inspector.edit_resolution_combo.currentData() == (
            PresetOutputSize(tier=higher_tiers[0])
        )
        assert inspector.edit_resolution_combo.isEnabled()
    else:
        assert inspector.edit_resolution_combo.currentData() is None
        assert not inspector.edit_resolution_combo.isEnabled()


def test_reference_selector_assigns_one_card_with_undo(
    application: QApplication,
) -> None:
    source = Card(name="Source")
    portrait = Card(name="Portrait")
    controller = DocumentController(Stack(name="Demo", cards=(source, portrait)))
    inspector = Inspector(controller)
    inspector.render(controller.document, source.id)
    applied: list[tuple[str, object]] = []
    inspector.change_applied.connect(lambda message, token: applied.append((message, token)))

    _choose_card(inspector, inspector.reference_button, portrait.id, application)

    assert controller.document.cards[0].active_revision.references == (
        ResolvedCardReference(target_card_id=portrait.id),
    )
    assert inspector.reference_remove_container.height() == 24
    inspector.reference_button.click()
    application.processEvents()
    assert inspector.card_picker.windowTitle() == "Select Reference 1"
    assert (
        inspector.card_picker.card_list.currentItem().data(Qt.ItemDataRole.UserRole) == portrait.id
    )
    assert inspector.card_picker.card_list.currentItem().isSelected()
    inspector.card_picker.hide()
    assert applied[-1][0] == "Reference changed"
    assert controller.undo_if_current(applied[-1][1])  # type: ignore[arg-type]
    inspector.render(controller.document, source.id)
    assert controller.document.cards[0].active_revision.references == ()


def test_card_picker_searches_names_and_disables_ineligible_references(
    application: QApplication,
) -> None:
    previous = Card(name="Previous")
    interaction = Interaction(polygons=(_polygon(),))
    source = Card(
        name="Source",
        revisions=(
            CardRevision(
                hotspot_set=HotspotSet(interactions=(interaction,)),
            ),
        ),
    )
    following = Card(name="Following")
    controller = DocumentController(Stack(name="Demo", cards=(previous, source, following)))
    inspector = Inspector(controller)
    inspector.render(controller.document, source.id)
    inspector.show()
    application.processEvents()

    inspector.reference_button.click()
    application.processEvents()
    assert inspector.card_picker.windowModality() == Qt.WindowModality.NonModal
    assert inspector.card_picker.windowFlags() & Qt.WindowType.Tool
    assert not inspector.card_picker.windowFlags() & Qt.WindowType.FramelessWindowHint
    viewport_width = inspector.card_picker.card_list.viewport().width()
    column_width = inspector.card_picker.card_list.gridSize().width()
    assert viewport_width >= column_width * 3
    assert viewport_width < column_width * 4
    resized = QSize(620, 410)
    inspector.card_picker.resize(resized)
    inspector.card_picker.hide()
    inspector.reference_button.click()
    application.processEvents()
    assert inspector.card_picker.size() == resized
    assert [
        inspector.card_picker.card_list.item(index).text()
        for index in range(inspector.card_picker.card_list.count())
    ] == ["Previous", "Source", "Following"]
    source_item = _picker_item(inspector, source.id)
    assert not source_item.flags() & Qt.ItemFlag.ItemIsEnabled
    assert source_item.toolTip() == ""
    inspector.card_picker.card_list.itemClicked.emit(source_item)
    assert controller.document.cards[1].active_revision.references == ()
    assert _picker_item(inspector, previous.id).flags() & Qt.ItemFlag.ItemIsEnabled
    assert _picker_item(inspector, following.id).flags() & Qt.ItemFlag.ItemIsEnabled

    inspector.card_picker.search_edit.setText("source")
    inspector.card_picker.search_edit.returnPressed.emit()
    assert controller.document.cards[1].active_revision.references == ()

    inspector.card_picker.search_edit.setText("follow")
    assert _picker_item(inspector, previous.id).isHidden()
    assert _picker_item(inspector, source.id).isHidden()
    assert not _picker_item(inspector, following.id).isHidden()
    inspector.close()


def test_hiding_inspector_closes_picker_and_discards_its_context(
    application: QApplication,
) -> None:
    source = Card(name="Source")
    reference = Card(name="Reference")
    controller = DocumentController(Stack(name="Demo", cards=(source, reference)))
    inspector = Inspector(controller)
    inspector.render(controller.document, source.id)
    inspector.show()
    inspector.reference_button.click()
    application.processEvents()
    assert inspector.card_picker.isVisible()

    inspector.hide()
    application.processEvents()
    inspector.card_picker.card_selected.emit(reference.id)

    assert not inspector.card_picker.isVisible()
    assert controller.document.cards[0].active_revision.references == ()
    inspector.close()


def test_card_picker_cancel_closes_without_changing_selection(
    application: QApplication,
) -> None:
    source = Card(name="Source")
    reference = Card(name="Reference")
    controller = DocumentController(Stack(name="Demo", cards=(source, reference)))
    inspector = Inspector(controller)
    inspector.render(controller.document, source.id)
    inspector.reference_button.click()
    application.processEvents()

    assert inspector.card_picker.cancel_button.text() == "Cancel"
    assert inspector.card_picker.cancel_button.accessibleName() == "Cancel"
    inspector.card_picker.cancel_button.click()

    assert not inspector.card_picker.isVisible()
    assert controller.document.cards[0].active_revision.references == ()
    inspector.close()


def test_card_picker_uses_active_revision_thumbnail(
    application: QApplication,
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "reference.png"
    image = QPixmap(240, 160)
    image.fill(QColor("red"))
    assert image.save(str(image_path))
    source = Card(name="Source")
    reference = Card(
        name="Reference",
        revisions=(CardRevision(background=_background()),),
    )
    controller = DocumentController(Stack(name="Demo", cards=(source, reference)))
    inspector = Inspector(
        controller,
        image_path_resolver=lambda _path: image_path,
    )
    inspector.render(controller.document, source.id)

    inspector.reference_button.click()
    application.processEvents()

    icon = _picker_item(inspector, reference.id).icon().pixmap(QSize(144, 96))
    assert icon.toImage().pixelColor(10, 10) == QColor("red")
    inspector.close()


def test_hotspot_picker_allows_current_card(
    application: QApplication,
) -> None:
    interaction = Interaction(polygons=(_polygon(),))
    source = Card(
        name="Source",
        revisions=(CardRevision(hotspot_set=HotspotSet(interactions=(interaction,))),),
    )
    controller = DocumentController(Stack(name="Demo", cards=(source,)))
    inspector = Inspector(controller)
    inspector.render(controller.document, source.id)
    inspector.show()
    application.processEvents()

    inspector.hotspot_destination_button.click()
    application.processEvents()

    assert _picker_item(inspector, source.id).flags() & Qt.ItemFlag.ItemIsEnabled
    inspector.close()


def test_second_reference_is_ordered_unique_and_promoted_when_first_clears(
    application: QApplication,
) -> None:
    source = Card(name="Source")
    portrait = Card(name="Portrait")
    room = Card(name="Room")
    controller = DocumentController(Stack(name="Demo", cards=(source, portrait, room)))
    inspector = Inspector(controller)
    inspector.render(controller.document, source.id)

    assert inspector.reference_label.text() == "References"
    assert not inspector.additional_reference_button.isEnabled()
    _choose_card(
        inspector,
        inspector.reference_button,
        portrait.id,
        application,
    )
    assert inspector.additional_reference_button.isEnabled()

    inspector.additional_reference_button.click()
    application.processEvents()
    assert (
        not _picker_item(
            inspector,
            portrait.id,
        ).flags()
        & Qt.ItemFlag.ItemIsEnabled
    )
    inspector.card_picker.hide()
    _choose_card(
        inspector,
        inspector.additional_reference_button,
        room.id,
        application,
    )
    assert controller.document.cards[0].active_revision.references == (
        ResolvedCardReference(target_card_id=portrait.id),
        ResolvedCardReference(target_card_id=room.id),
    )

    inspector.reference_remove_button.click()
    assert controller.document.cards[0].active_revision.references == (
        ResolvedCardReference(target_card_id=room.id),
    )
    assert inspector.reference_button.text() == "Room"
    assert inspector.additional_reference_button.text() == "Choose Reference…"


def test_style_manager_edits_global_definition_and_deletes_with_undo(
    application: QApplication,
) -> None:
    ink = StyleDefinition(name="Ink", prompt_text="Rendered in ink.")
    revision = CardRevision(style_id=ink.id)
    source = Card(name="Source", revisions=(revision,))
    other = Card(
        name="Other",
        revisions=(CardRevision(style_id=ink.id),),
    )
    controller = DocumentController(
        Stack(
            name="Demo",
            styles=(ink,),
            new_card_style_id=ink.id,
            cards=(source, other),
        )
    )
    parent = Inspector(controller)
    manager = StyleManagerWindow(controller, parent)
    applied: list[tuple[str, object]] = []
    manager.change_applied.connect(lambda message, token: applied.append((message, token)))

    assert manager.style_list.currentItem().text() == "Ink"
    assert manager.name_edit.text() == "Ink"
    assert manager.prompt_edit.toPlainText() == "Rendered in ink."
    style_layout = manager.layout()
    assert style_layout is not None
    assert style_layout.indexOf(manager.name_edit) < style_layout.indexOf(manager.prompt_edit)
    assert manager.name_edit.sizePolicy().horizontalPolicy() == (
        manager.prompt_edit.sizePolicy().horizontalPolicy()
    )
    manager.name_edit.setText("Etching")
    manager.prompt_edit.setPlainText("Fine etched linework.")
    assert manager.commit_pending_edits(render_change=True)

    changed_style = controller.document.styles[0]
    assert changed_style.id == ink.id
    assert changed_style.name == "Etching"
    assert changed_style.prompt_text == "Fine etched linework."
    assert all(card.active_revision.style_id == ink.id for card in controller.document.cards)

    manager.add_button.click()
    assert len(controller.document.styles) == 2
    assert manager.name_edit.text() == "New Style"
    assert manager.name_edit.selectedText() == "New Style"

    manager.delete_button.click()
    assert len(controller.document.styles) == 1
    assert controller.document.styles[0].id == ink.id
    assert applied[-1][0] == "Style deleted"
    assert controller.undo_if_current(applied[-1][1])  # type: ignore[arg-type]
    assert len(controller.document.styles) == 2

    manager.render(controller.document)
    manager.style_list.setCurrentRow(0)
    manager.delete_button.click()
    assert controller.document.styles[0].name == "New Style"
    assert controller.document.new_card_style_id is None
    assert all(card.active_revision.style_id is None for card in controller.document.cards)
    manager.close()


@pytest.mark.parametrize("field", ("name", "prompt"))
def test_live_style_drafts_signal_without_programmatic_render_noise(
    application: QApplication,
    field: str,
) -> None:
    style = StyleDefinition(name="Ink", prompt_text="Rendered in ink.")
    revision = CardRevision(style_id=style.id)
    card = Card(name="Card", revisions=(revision,))
    controller = DocumentController(
        Stack(
            name="Demo",
            styles=(style,),
            new_card_style_id=style.id,
            cards=(card,),
        )
    )
    parent = Inspector(controller)
    manager = StyleManagerWindow(controller, parent)
    changes: list[None] = []
    manager.inputs_changed.connect(lambda: changes.append(None))
    manager.show()
    editor = manager.name_edit if field == "name" else manager.prompt_edit
    editor.setFocus()
    application.processEvents()

    manager.render(controller.document)
    assert changes == []

    if field == "name":
        manager.name_edit.setText("Etching")
    else:
        manager.prompt_edit.setPlainText("Fine etched linework.")

    assert changes == [None]
    assert controller.document.style_by_id(style.id) == style
    manager.close()


def test_style_name_draft_does_not_overwrite_authoritative_prompt_change(
    application: QApplication,
) -> None:
    style = StyleDefinition(name="Ink", prompt_text="Original prompt.")
    controller = DocumentController(
        Stack(
            name="Demo",
            styles=(style,),
            new_card_style_id=style.id,
        )
    )
    parent = Inspector(controller)
    manager = StyleManagerWindow(controller, parent)

    manager.name_edit.setText("Etching")
    changed = controller.execute(
        UpdateStyleCommand(
            style_id=style.id,
            name=style.name,
            prompt_text="Changed elsewhere.",
        )
    )
    manager.render(changed)

    assert manager.name_edit.text() == "Etching"
    assert manager.prompt_edit.toPlainText() == "Changed elsewhere."
    assert manager.commit_pending_edits(render_change=True)
    committed = controller.document.style_by_id(style.id)
    assert committed is not None
    assert committed.name == "Etching"
    assert committed.prompt_text == "Changed elsewhere."
    manager.close()


def test_deleted_reference_is_shown_as_unresolved(
    application: QApplication,
) -> None:
    destination = Card(name="Former portrait")
    revision = CardRevision(references=(ResolvedCardReference(target_card_id=destination.id),))
    source = Card(name="Source", revisions=(revision,))
    controller = DocumentController(Stack(name="Demo", cards=(source, destination)))
    inspector = Inspector(controller)
    inspector.render(controller.document, source.id)

    changed = controller.execute(DeleteCardCommand(card_id=destination.id))
    inspector.render(changed, source.id)

    assert inspector.reference_button.text() == "Missing: Former portrait"
    assert controller.document.cards[0].active_revision.references == (
        UnresolvedCardReference(target_name="Former portrait"),
    )


def test_render_preserves_focused_description_draft(
    application: QApplication,
) -> None:
    interaction = _interaction()
    revision = CardRevision(
        description="Saved description",
        hotspot_set=HotspotSet(interactions=(interaction,)),
    )
    card = Card(name="Card", revisions=(revision,))
    controller = DocumentController(Stack(name="Demo", cards=(card,)))
    inspector = Inspector(controller)
    inspector.show()
    inspector.render(controller.document, card.id)
    editor = inspector.description_edit
    editor.setFocus()
    editor.setPlainText("Uncommitted description")
    cursor = editor.textCursor()
    cursor.setPosition(4)
    editor.setTextCursor(cursor)
    application.processEvents()
    assert editor.hasFocus()
    changed = controller.execute(RenameCardCommand(card_id=card.id, name="Renamed"))
    inspector.render(changed, card.id)
    assert inspector.description_edit.toPlainText() == "Uncommitted description"
    assert editor.textCursor().position() == 4

    assert not hasattr(inspector, "hotspot_label_edit")
    inspector.close()


def test_switching_cards_shows_each_description(
    application: QApplication,
) -> None:
    first = Card(
        name="First",
        revisions=(CardRevision(description="First description"),),
    )
    second = Card(
        name="Second",
        revisions=(CardRevision(description="Second description"),),
    )
    controller = DocumentController(Stack(name="Demo", cards=(first, second)))
    inspector = Inspector(controller)
    inspector.render(controller.document, first.id)
    inspector.render(controller.document, second.id)
    assert inspector.description_edit.toPlainText() == "Second description"
    assert inspector.description_edit.toPlainText() == "Second description"


def test_using_labels_name_single_reference(
    application: QApplication,
) -> None:
    reference = Card(name="Reference")
    source = Card(
        name="Source",
        revisions=(
            CardRevision(
                description="A courtyard",
                references=(ResolvedCardReference(target_card_id=reference.id),),
            ),
        ),
    )
    controller = DocumentController(Stack(name="Demo", cards=(source, reference)))
    inspector = Inspector(controller)
    inspector.render(controller.document, source.id)

    assert inspector.reference_label.text() == "References"
    assert "Using: Description + Reference" in inspector.generate_background_button.toolTip()
    assert "image 1" in inspector.reference_button.toolTip()


def test_add_hotspot_requests_drawing_without_mutating_document(
    application: QApplication,
) -> None:
    card = Card(name="Card")
    controller = DocumentController(Stack(name="Demo", cards=(card,)))
    inspector = Inspector(controller)
    inspector.render(controller.document, card.id)

    requests: list[None] = []
    inspector.hotspot_drawing_requested.connect(lambda: requests.append(None))

    inspector.add_hotspot_button.click()

    assert requests == [None]
    assert controller.document.cards[0].active_revision.hotspot_set is None
    assert inspector.selected_interaction_id is None


def test_hotspot_properties_reorder_and_delete_use_commands(
    application: QApplication,
) -> None:
    first = _interaction("First")
    second = _interaction("Second")
    destination = Card(name="Destination")
    source = Card(
        name="Source",
        revisions=(
            CardRevision(
                hotspot_set=HotspotSet(interactions=(first, second)),
            ),
        ),
    )
    controller = DocumentController(Stack(name="Demo", cards=(source, destination)))
    inspector = Inspector(controller)
    inspector.render(controller.document, source.id)

    inspector.select_interaction(second.id)
    inspector.move_hotspot_up_button.click()
    assert [
        item.id
        for item in controller.document.cards[0].active_revision.hotspot_set.interactions  # type: ignore[union-attr]
    ] == [second.id, first.id]

    _choose_card(
        inspector,
        inspector.hotspot_destination_button,
        destination.id,
        application,
    )
    application.processEvents()
    changed = controller.document.cards[0].active_revision.hotspot_set
    assert changed is not None
    assert changed.interactions[0].label == "Go to Destination"
    assert changed.interactions[0].action.target == ResolvedCardReference(
        target_card_id=destination.id
    )

    applied: list[tuple[str, object]] = []
    inspector.change_applied.connect(lambda message, token: applied.append((message, token)))
    inspector.delete_hotspot_button.click()
    remaining = controller.document.cards[0].active_revision.hotspot_set
    assert remaining is not None
    assert [item.id for item in remaining.interactions] == [first.id]
    assert applied and applied[0][0] == "Hotspot deleted"
    assert controller.undo_if_current(applied[0][1])  # type: ignore[arg-type]


def test_generation_activity_is_reflected_on_the_initiating_button(
    application: QApplication,
) -> None:
    inspector = Inspector(DocumentController(Stack(name="Demo")))
    inspector.set_background_capabilities(
        can_generate=False,
        generate_reason="Generating",
        busy=True,
        generating=True,
    )

    assert inspector.generate_background_button.text() == "Generating…"


def test_image_prompt_controls_are_absent(
    application: QApplication,
) -> None:
    inspector = Inspector(DocumentController(Stack(name="Demo")))

    assert not hasattr(inspector, "image_prompt_button")
    assert not hasattr(inspector, "enrich_button")
    assert not hasattr(inspector, "prepare_image_prompt_requested")
