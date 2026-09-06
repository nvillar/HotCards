"""Focused offscreen tests for revision-local authoring controls."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QColor, QPixmap
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
    QToolButton,
    QWidget,
)

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
    Card,
    CardRevision,
    CurrentSourceSize,
    DerivedImageSourceSnapshot,
    DirectGenerateProvenance,
    DuplicateProvenance,
    EditPreserveOptions,
    EditProvenance,
    ExactOutputSize,
    GeneratedBackground,
    GeneratedSoundAsset,
    GenerateInputs,
    HotspotConditions,
    HotspotKeyChanges,
    HotspotSet,
    ImageOperationSettings,
    ImageSourceSnapshot,
    Interaction,
    KeyDefinition,
    NavigateAction,
    Point,
    Polygon,
    PresetOutputSize,
    RefineProvenance,
    RefineTransformation,
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


@pytest.fixture(scope="module")
def application() -> QApplication:
    return QApplication.instance() or QApplication([])


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
        provenance=DirectGenerateProvenance(
            inputs=GenerateInputs(
                description="A courtyard",
                output_size=ExactOutputSize(width=size[0], height=size[1]),
            ),
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
        edit_lineage=image_edit_lineage(background.provenance),
    )
    output_size = (
        CurrentSourceSize(width=size[0], height=size[1])
        if size == (source.width, source.height)
        else PresetOutputSize(tier=ResolutionTier.LARGE)
    )
    settings = result.provenance.settings
    if operation == "refine":
        provenance = RefineProvenance(
            source=source,
            description=revision.description,
            render_prompt="Current Description and accepted Edit instructions.",
            output_size=output_size,
            transformation=RefineTransformation.BALANCED,
            strength=0.5,
            settings=settings.model_copy(update={"seed": source.seed}),
        )
    else:
        assert operation == "edit"
        provenance = EditProvenance(
            source=source,
            instruction=instruction,
            preserve=EditPreserveOptions(),
            expanded_prompt=f"{instruction}\n\nHidden Style addendum",
            output_size=output_size,
            prompt_token_count=20,
            settings=settings.model_copy(update={"seed": source.seed + 1}),
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
    assert not hasattr(inspector, "style_list")
    assert not hasattr(inspector, "key_list")
    assert not hasattr(inspector, "edit_preserve_checkboxes")
    assert all(
        label.text() != "Keys are global to this stack." for label in inspector.findChildren(QLabel)
    )
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
    assert not hasattr(inspector, "enrich_button")
    assert not hasattr(inspector, "description_toggle")
    assert not hasattr(inspector, "image_prompt_button")
    assert not isinstance(inspector.reference_panel, QFrame)
    assert inspector.reference_panel.layout().contentsMargins().isNull()
    new_image_group = inspector.reference_panel.parentWidget()
    assert isinstance(new_image_group, QGroupBox)
    assert not new_image_group.title()
    new_image_layout = new_image_group.layout()
    assert new_image_layout is not None
    content_layout = inspector.description_edit.parentWidget().layout()
    assert content_layout is not None
    assert content_layout.stretch(content_layout.indexOf(inspector.description_edit)) == 1
    assert content_layout.indexOf(inspector.description_edit) < (
        content_layout.indexOf(inspector.style_label)
    )
    assert content_layout.indexOf(inspector.style_label) < (
        content_layout.indexOf(inspector.style_combo)
    )
    assert content_layout.indexOf(inspector.style_combo) < (
        content_layout.indexOf(inspector.new_image_section_label)
    )
    assert content_layout.indexOf(inspector.new_image_section_label) < (
        content_layout.indexOf(new_image_group)
    )
    assert content_layout.indexOf(new_image_group) < (
        content_layout.indexOf(inspector.evolve_section_label)
    )
    assert new_image_layout.indexOf(inspector.reference_label) < (
        new_image_layout.indexOf(inspector.reference_panel)
    )
    assert new_image_layout.indexOf(inspector.reference_panel) < (
        new_image_layout.indexOf(inspector.resolution_label)
    )
    assert new_image_layout.indexOf(inspector.resolution_label) < (
        new_image_layout.indexOf(inspector.resolution_combo)
    )

    assert new_image_layout.indexOf(inspector.resolution_combo) < (
        new_image_layout.indexOf(inspector.generate_background_button)
    )
    assert inspector.new_image_section_label.text() == "New Image"
    assert inspector.new_image_section_label.font() == inspector.evolve_section_label.font()
    assert "New Image only" in inspector.reference_label.toolTip()
    assert "never sent to Evolve" in inspector.reference_label.toolTip()
    assert "Shared by New Image and Evolve" in inspector.description_edit.toolTip()
    assert new_image_layout.contentsMargins() == (
        inspector.refine_background_button.parentWidget().layout().contentsMargins()
    )
    for widget in (
        inspector.description_edit,
        inspector.style_combo,
        inspector.reference_panel,
        inspector.generate_background_button,
        inspector.refine_background_button,
    ):
        assert inspector.inspector_tabs.widget(0).isAncestorOf(widget)
        assert not inspector.inspector_tabs.widget(1).isAncestorOf(widget)
    assert inspector.inspector_tabs.widget(1).isAncestorOf(inspector.edit_instruction_edit)
    assert inspector.style_combo.currentText() == "No Style"
    assert not hasattr(inspector, "clear_background_button")
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
        inspector.evolve_section_label.font().pointSizeF()
    )
    assert inspector.hotspot_then_label.font().pointSizeF() == (
        inspector.edit_section_label.font().pointSizeF()
    )
    assert inspector.add_condition_button.text() == "+"
    assert inspector.add_condition_button.accessibleName() == "Add condition"
    assert inspector.add_condition_label.text() == "Key condition"
    assert inspector.add_key_change_button.text() == "+"
    assert inspector.add_key_change_button.accessibleName() == "Add key change"
    assert inspector.add_key_change_label.text() == "Key change"
    assert inspector.add_condition_button.size() == QSize(20, 20)
    assert inspector.add_key_change_button.size() == QSize(20, 20)
    assert inspector.hotspot_destination_button.text() == "Unresolved card"
    assert inspector.hotspot_destination_remove_button.text() == "−"
    assert inspector.reference_remove_button.text() == "−"
    assert inspector.additional_reference_remove_button.text() == "−"
    assert inspector.hotspot_destination_remove_button.size() == QSize(20, 20)
    assert inspector.reference_remove_button.size() == QSize(20, 20)
    assert inspector.additional_reference_remove_button.size() == QSize(20, 20)
    assert inspector.hotspot_destination_remove_button.testAttribute(
        Qt.WidgetAttribute.WA_LayoutUsesWidgetRect
    )
    assert inspector.reference_remove_button.testAttribute(
        Qt.WidgetAttribute.WA_LayoutUsesWidgetRect
    )
    assert inspector.additional_reference_remove_button.testAttribute(
        Qt.WidgetAttribute.WA_LayoutUsesWidgetRect
    )
    assert inspector.reference_remove_container.height() == 22
    assert inspector.additional_reference_remove_container.height() == 22
    assert not hasattr(inspector, "clear_all_keys_checkbox")
    assert not hasattr(inspector, "hotspot_summary")
    hotspot_layout = inspector.hotspot_list.parentWidget().layout()
    assert hotspot_layout is not None
    assert inspector.hotspot_list.parentWidget().objectName() == "hotspotsInspectorContent"
    assert hotspot_layout.contentsMargins() == (
        inspector.refine_background_button.parentWidget().parentWidget().layout().contentsMargins()
    )
    assert hotspot_layout.contentsMargins() == (
        inspector.edit_instruction_edit.parentWidget().parentWidget().layout().contentsMargins()
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
    assert inspector.hotspot_sound_remove_button.text() == "−"
    assert inspector.hotspot_sound_remove_button.size() == QSize(20, 20)
    assert inspector.hotspot_sound_remove_button.testAttribute(
        Qt.WidgetAttribute.WA_LayoutUsesWidgetRect
    )
    assert inspector.hotspot_destination_remove_container.height() == 22
    assert inspector.hotspot_sound_remove_container.height() == 22
    assert inspector.condition_rows_layout.contentsMargins().isNull()
    assert inspector.key_change_rows_layout.contentsMargins().isNull()
    assert inspector.condition_controls_layout.spacing() == 3
    assert inspector.key_change_controls_layout.spacing() == 3
    assert not hasattr(inspector, "condition_table")
    assert not hasattr(inspector, "key_change_table")
    assert not hasattr(inspector, "no_conditions_label")
    assert not hasattr(inspector, "no_key_changes_label")
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
    visible_copy = " ".join(label.text() for label in inspector.findChildren(QLabel))
    for obsolete in (
        "Inspector",
        "Scene",
        "Intent",
        "Details",
        "Areas",
        "Summarize",
        "Generate Hotspots",
    ):
        assert obsolete not in visible_copy


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
        refine_source_size=(768, 576),
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


def test_generate_evolve_has_source_similarity_resolution_and_action(
    application: QApplication,
) -> None:
    revision = CardRevision(
        description="A moonlit courtyard",
        background=_background(),
    )
    card = Card(name="Courtyard", revisions=(revision,))
    controller = DocumentController(Stack(name="Demo", cards=(card,)))
    inspector = Inspector(controller)
    inspector.render(
        controller.document,
        card.id,
        refine_source_size=(512, 384),
    )

    assert inspector.inspector_tabs.tabText(inspector._edit_tab_index) == "Edit"
    evolve_group = inspector.refine_background_button.parentWidget()
    edit_group = inspector.edit_background_button.parentWidget()
    assert isinstance(evolve_group, QGroupBox)
    assert isinstance(edit_group, QGroupBox)
    assert not evolve_group.title()
    assert not edit_group.title()
    assert inspector.evolve_section_label.text() == "Evolve"
    assert inspector.edit_section_label.text() == "Edit"
    assert inspector.evolve_section_label.font().pointSizeF() == (
        inspector.refine_transformation_label.font().pointSizeF()
    )
    assert inspector.edit_section_label.font().pointSizeF() == (
        inspector.edit_instruction_label.font().pointSizeF()
    )
    assert not hasattr(inspector, "refine_description_label")
    assert not hasattr(inspector, "edit_description_label")
    assert (
        RefineTransformation(inspector.refine_transformation_combo.currentData())
        is RefineTransformation.BALANCED
    )
    assert [
        inspector.refine_transformation_combo.itemText(index)
        for index in range(inspector.refine_transformation_combo.count())
    ] == [
        "Reimagine",
        "Balanced",
        "Preserve",
    ]
    assert inspector.refine_transformation_label.text() == "Source Similarity"
    assert "Balance the Description" in inspector.refine_transformation_combo.toolTip()
    for index in range(inspector.refine_transformation_combo.count()):
        tooltip = inspector.refine_transformation_combo.itemData(index, Qt.ItemDataRole.ToolTipRole)
        assert tooltip and not any(char.isdigit() for char in tooltip)
    assert [
        inspector.refine_resolution_combo.itemText(index)
        for index in range(inspector.refine_resolution_combo.count())
    ] == [
        "Medium",
        "Large",
        "Full",
    ]
    assert inspector.refine_resolution_label.text() == "Resolution"
    assert inspector.edit_resolution_label.text() == "Resolution"
    assert inspector.refine_resolution_combo.currentData() == CurrentSourceSize(
        width=512,
        height=384,
    )
    refine_layout = inspector.refine_background_button.parentWidget().layout()
    assert refine_layout is not None
    assert refine_layout.indexOf(inspector.refine_transformation_label) < refine_layout.indexOf(
        inspector.refine_transformation_combo
    )
    assert refine_layout.indexOf(inspector.refine_transformation_combo) < refine_layout.indexOf(
        inspector.refine_resolution_label
    )
    assert refine_layout.indexOf(inspector.refine_resolution_label) < refine_layout.indexOf(
        inspector.refine_resolution_combo
    )
    assert refine_layout.indexOf(inspector.refine_resolution_combo) < refine_layout.indexOf(
        inspector.refine_background_button
    )
    assert not hasattr(inspector, "refine_source_card_combo")
    assert not hasattr(inspector, "refine_source_revision_combo")

    requested: list[tuple[object, object]] = []
    inspector.refine_background_requested.connect(
        lambda transformation, output_size: requested.append((transformation, output_size))
    )
    inspector.set_refine_capabilities(
        can_refine=True,
        refine_reason="Ready to evolve",
        busy=False,
        refining=False,
    )
    inspector.refine_background_button.click()

    assert requested == [
        (
            RefineTransformation.BALANCED,
            CurrentSourceSize(width=512, height=384),
        )
    ]
    assert inspector.refine_background_button.text() == "Evolve"
    assert inspector.refine_background_button.toolTip() == (
        "Evolve the current image using Description."
    )
    for widget in inspector.findChildren(QWidget):
        assert "reinterpret" not in widget.toolTip().casefold()
        assert "reinterpret" not in widget.accessibleName().casefold()


def test_refine_resolution_inserts_nonstandard_current_size_by_area(
    application: QApplication,
) -> None:
    revision = CardRevision(
        description="A moonlit courtyard",
        background=_background(),
    )
    card = Card(name="Courtyard", revisions=(revision,))
    controller = DocumentController(Stack(name="Demo", cards=(card,)))
    inspector = Inspector(controller)

    inspector.render(
        controller.document,
        card.id,
        refine_source_size=(640, 480),
    )
    assert [
        inspector.refine_resolution_combo.itemText(index)
        for index in range(inspector.refine_resolution_combo.count())
    ] == [
        "Current",
        "Large",
        "Full",
    ]
    assert inspector.refine_resolution_combo.currentData() == CurrentSourceSize(
        width=640,
        height=480,
    )
    assert not inspector.refine_error.isVisible()


def test_refine_selection_becomes_current_when_new_image_matches_preset(
    application: QApplication,
) -> None:
    revision = CardRevision(
        description="A moonlit courtyard",
        background=_background(),
    )
    card = Card(name="Courtyard", revisions=(revision,))
    controller = DocumentController(Stack(name="Demo", cards=(card,)))
    inspector = Inspector(controller)
    inspector.render(
        controller.document,
        card.id,
        refine_source_size=(640, 480),
    )
    inspector.refine_resolution_combo.setCurrentIndex(
        inspector._combo_index_for_data(
            inspector.refine_resolution_combo,
            PresetOutputSize(tier=ResolutionTier.MEDIUM),
        )
    )

    inspector.render(
        controller.document,
        card.id,
        refine_source_size=(512, 384),
    )

    assert inspector.refine_resolution_combo.currentData() == CurrentSourceSize(
        width=512,
        height=384,
    )
    assert inspector.refine_resolution_combo.currentIndex() == 0


@pytest.mark.parametrize("operation", ("generate", "refine", "edit"))
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
    inspector.render(controller.document, card.id, refine_source_size=source_size)
    inspector.select_interaction(card.active_revision.hotspot_set.interactions[0].id)
    selected = inspector.selected_interaction_id
    inspector.description_edit.setFocus()
    inspector.description_edit.setPlainText("A focused Description draft")
    inspector.set_edit_instruction("A pending instruction\nwith exact line breaks.")
    application.processEvents()
    draft = inspector.edit_instruction_draft
    controls = (
        inspector.resolution_combo,
        inspector.refine_resolution_combo,
        inspector.edit_resolution_combo,
    )
    full = PresetOutputSize(tier=ResolutionTier.FULL)
    for combo in controls:
        combo.setCurrentIndex(inspector._combo_index_for_data(combo, full))
    choice_token = controller.current_undo_token
    inspector.render(controller.document, card.id, refine_source_size=source_size)
    assert [combo.currentData() for combo in controls] == [full] * 3
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
        inspector.render(controller.document, card.id, refine_source_size=current_size)
        assert controller.document.cards[0].active_revision.id == card.active_revision.id
        assert inspector.resolution_combo.toolTip() == (f"{current_size[0]} × {current_size[1]}")
        for combo in controls[1:]:
            assert combo.currentData() == CurrentSourceSize(
                width=current_size[0], height=current_size[1]
            )
        assert [
            inspector.edit_history_list.item(index).data(Qt.ItemDataRole.UserRole)
            for index in range(inspector.edit_history_list.count())
        ] == list(history)
        assert inspector.description_edit.toPlainText() == "A focused Description draft"
        assert inspector.edit_instruction_draft == draft
        assert inspector.selected_interaction_id == selected
        inspector.render(controller.document, card.id, refine_source_size=current_size)
        assert controller.current_undo_token == expected_token
    inspector.close()


@pytest.mark.parametrize("operation", ("generate", "refine", "edit"))
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
                "provenance": DuplicateProvenance(
                    source=ImageSourceSnapshot(
                        card_id=uuid4(),
                        revision_id=uuid4(),
                        background_id=result.id,
                    ),
                    original_provenance=result.provenance,
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
    draft = inspector.edit_instruction_draft
    expected = (
        ()
        if operation == "generate"
        else ((*instructions, "Add a flower.") if operation == "edit" else instructions)
    )
    assert inspector.edit_history_list.wordWrap()
    assert inspector.edit_history_label.text() == "Edit History"
    assert inspector.edit_history_empty_label.text() == "No accepted edits for this image."
    assert inspector.edit_history_empty_label.isHidden() == bool(expected)
    assert [
        inspector.edit_history_list.item(index).text()
        for index in range(inspector.edit_history_list.count())
    ] == [f"{number}. {text}" for number, text in enumerate(expected, start=1)]
    for index, instruction in enumerate(expected):
        item = inspector.edit_history_list.item(index)
        assert item.data(Qt.ItemDataRole.UserRole) == instruction
        assert item.toolTip() == instruction
        assert "Hidden Style" not in item.text()
    inspector.render(controller.document, card.id)
    assert inspector.edit_instruction_draft == draft

    controller.execute(ActivateRevisionCommand(card_id=card.id, revision_id=card.revisions[0].id))
    inspector.render(controller.document, card.id)
    assert inspector.edit_history_list.count() == len(instructions)
    assert inspector.edit_instruction_edit.toPlainText() == draft.text
    inspector.render(controller.document, other.id)
    assert inspector.edit_history_list.count() == 0
    assert not inspector.edit_history_empty_label.isHidden()
    inspector.render(controller.document, card.id)
    assert inspector.edit_history_list.count() == len(instructions)
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
    inspector.inspector_tabs.setCurrentIndex(1)
    inspector.resize(340, 800)
    inspector.show()
    inspector.set_edit_instruction("A different draft")
    application.processEvents()
    history = inspector.edit_history_list
    requests: list[object] = []
    inspector.generate_background_requested.connect(lambda: requests.append("generate"))
    inspector.refine_background_requested.connect(lambda *_args: requests.append("refine"))
    inspector.edit_background_requested.connect(lambda *_args: requests.append("edit"))
    inspector.document_changed.connect(lambda *_args: requests.append("document"))
    inspector.hotspot_selected.connect(lambda *_args: requests.append("hotspot"))
    draft = inspector.edit_instruction_draft
    document = controller.document
    token = controller.current_undo_token
    history.setCurrentRow(0)
    QTest.keyClick(history, Qt.Key.Key_Down)
    assert inspector.edit_instruction_draft == draft
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
    assert inspector.edit_instruction_draft.sequence == draft.sequence + 1
    recalled = inspector.edit_instruction_draft
    refresh_signals: list[object] = []
    history.currentItemChanged.connect(lambda *_args: refresh_signals.append("selection"))
    history.instruction_requested.connect(lambda *_args: refresh_signals.append("recall"))
    inspector.render(controller.document, card.id)
    assert inspector.edit_instruction_draft == recalled
    assert refresh_signals == []
    assert controller.document == document
    assert controller.current_undo_token == token
    assert requests == []
    assert inspector.inspector_tabs.currentIndex() == 1
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
    inspector.inspector_tabs.setCurrentIndex(1)
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
    assert item.text() == f"1. {instruction}"
    assert item.data(Qt.ItemDataRole.UserRole) == instruction
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
    assert inspector.commit_revision_metadata()

    revision = controller.document.cards[0].active_revision
    assert revision.description == "New description"
    assert controller.undo()
    assert controller.document.cards[0].active_revision.description == "Old"


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
        refine_source_size=(1024, 768),
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
        refine_source_size=(1024, 768),
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
        refine_source_size=(592, 448),
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
        refine_source_size=(512, 384),
    )
    inspector.resolution_combo.setCurrentIndex(
        inspector._combo_index_for_data(
            inspector.resolution_combo,
            PresetOutputSize(tier=ResolutionTier.FULL),
        )
    )
    inspector.refine_resolution_combo.setCurrentIndex(
        inspector._combo_index_for_data(
            inspector.refine_resolution_combo,
            PresetOutputSize(tier=ResolutionTier.LARGE),
        )
    )

    inspector.render(
        controller.document,
        second.id,
        refine_source_size=(768, 576),
    )

    current = CurrentSourceSize(width=768, height=576)
    assert inspector.resolution_combo.currentData() == PresetOutputSize(tier=ResolutionTier.LARGE)
    assert inspector.refine_resolution_combo.currentData() == current
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
        refine_source_size=source_size,
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
        refine_source_size=(640, 480),
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
def test_refine_and_edit_show_invalid_current_size_as_unavailable(
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
        refine_source_size=source_size,
    )
    refine_labels = [
        inspector.refine_resolution_combo.itemText(index)
        for index in range(inspector.refine_resolution_combo.count())
    ]
    refine_unavailable = refine_labels.index("Current")
    refine_item = inspector.refine_resolution_combo.model().item(refine_unavailable)
    assert refine_item is not None
    assert not refine_item.isEnabled()
    higher_tiers = higher_output_tiers(
        source_size[0],
        source_size[1],
        AspectRatio.LANDSCAPE,
    )
    if higher_tiers:
        assert inspector.refine_resolution_combo.currentData() == PresetOutputSize(
            tier=higher_tiers[0]
        )
        inspector.refine_resolution_combo.setCurrentIndex(refine_unavailable)
        assert inspector.refine_resolution_combo.currentData() == PresetOutputSize(
            tier=higher_tiers[0]
        )
        assert inspector.refine_resolution_combo.isEnabled()
    else:
        assert inspector.refine_resolution_combo.currentData() is None
        assert not inspector.refine_resolution_combo.isEnabled()

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


def test_style_selector_updates_revision_and_new_card_default_with_undo(
    application: QApplication,
) -> None:
    ink = StyleDefinition(name="Ink", prompt_text="Rendered in ink.")
    source = Card(name="Source")
    controller = DocumentController(
        Stack(
            name="Demo",
            styles=(ink,),
            new_card_style_id=None,
            cards=(source,),
        )
    )
    inspector = Inspector(controller)
    inspector.render(controller.document, source.id)
    applied: list[tuple[str, object]] = []
    inspector.change_applied.connect(lambda message, token: applied.append((message, token)))

    ink_index = inspector._combo_index_for_data(inspector.style_combo, ink.id)
    inspector.style_combo.setCurrentIndex(ink_index)

    revision = controller.document.cards[0].active_revision
    assert revision.style_id == ink.id
    assert controller.document.new_card_style_id == ink.id
    assert applied[-1][0] == "Style changed"
    assert controller.undo_if_current(applied[-1][1])  # type: ignore[arg-type]
    assert controller.document.cards[0].active_revision.style_id is None
    assert controller.document.new_card_style_id is None


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

    inspector.description_edit.setFocus()
    inspector.description_edit.setPlainText("Uncommitted description")
    application.processEvents()
    changed = controller.execute(RenameCardCommand(card_id=card.id, name="Renamed"))
    inspector.render(changed, card.id)
    assert inspector.description_edit.toPlainText() == "Uncommitted description"

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
        has_image=True,
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
