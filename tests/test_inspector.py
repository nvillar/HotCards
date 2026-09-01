"""Focused offscreen tests for revision-local authoring controls."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from uuid import uuid4

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QGroupBox,
    QLabel,
    QSizePolicy,
    QStyle,
    QToolButton,
)

from hotcards.application.commands import (
    DeleteCardCommand,
    RenameCardCommand,
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
    DirectGenerateProvenance,
    ExactOutputSize,
    GeneratedBackground,
    GenerateInputs,
    HotspotConditions,
    HotspotKeyChanges,
    HotspotSet,
    ImageOperationSettings,
    Interaction,
    KeyDefinition,
    NavigateAction,
    Point,
    Polygon,
    PresetOutputSize,
    RefineTransformation,
    ResolvedCardReference,
    SoundDefinition,
    Stack,
    StyleDefinition,
    UnresolvedCardReference,
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


def _background() -> GeneratedBackground:
    asset_id = uuid4()
    generated_at = datetime.now(UTC)
    return GeneratedBackground(
        id=asset_id,
        image_path=f"assets/cards/card/image-{asset_id}.png",
        provenance=DirectGenerateProvenance(
            inputs=GenerateInputs(description="A courtyard"),
            render_prompt="A courtyard",
            settings=ImageOperationSettings(
                model_identifier="test",
                mflux_version="test",
                seed=7,
                width=512,
                height=384,
                step_count=4,
                generated_at=generated_at,
                duration_seconds=1,
            ),
        ),
        created_at=generated_at,
    )


def _popup_focused_data(combo: QComboBox, application: QApplication) -> object:
    combo.showPopup()
    application.processEvents()
    current_index = combo.view().currentIndex()
    assert combo.view().selectionModel().selectedIndexes() == [current_index]
    value = combo.itemData(current_index.row())
    combo.hidePopup()
    return value


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
    assert inspector.inspector_tabs.tabText(1) == "Transform"
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
    content_layout = inspector.reference_panel.parentWidget().layout()
    assert content_layout is not None
    assert content_layout.stretch(content_layout.indexOf(inspector.description_edit)) == 1
    assert content_layout.indexOf(inspector.description_edit) < (
        content_layout.indexOf(inspector.style_label)
    )
    assert content_layout.indexOf(inspector.style_label) < (
        content_layout.indexOf(inspector.style_combo)
    )
    assert content_layout.indexOf(inspector.style_combo) < (
        content_layout.indexOf(inspector.reference_label)
    )
    assert content_layout.indexOf(inspector.reference_label) < (
        content_layout.indexOf(inspector.reference_panel)
    )
    assert content_layout.indexOf(inspector.reference_panel) < (
        content_layout.indexOf(inspector.resolution_label)
    )
    assert content_layout.indexOf(inspector.resolution_label) < (
        content_layout.indexOf(inspector.resolution_combo)
    )

    assert content_layout.indexOf(inspector.resolution_combo) < (
        content_layout.indexOf(inspector.generate_background_button)
    )
    assert inspector.style_combo.currentText() == "No Style"
    assert not hasattr(inspector, "clear_background_button")
    assert inspector.hotspot_target_label.text() == "Go to"
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
        inspector.reinterpret_section_label.font().pointSizeF()
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
    assert inspector.hotspot_destination_combo.findText("Create New Card...") == -1
    assert all(
        inspector.hotspot_destination_combo.itemData(index) != "create"
        for index in range(inspector.hotspot_destination_combo.count())
    )
    assert not hasattr(inspector, "clear_all_keys_checkbox")
    assert not hasattr(inspector, "hotspot_summary")
    hotspot_layout = inspector.hotspot_list.parentWidget().layout()
    assert hotspot_layout is not None
    assert inspector.hotspot_list.parentWidget().objectName() == "hotspotsInspectorContent"
    assert hotspot_layout.contentsMargins() == (
        inspector.refine_background_button.parentWidget().parentWidget().layout().contentsMargins()
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
        then_layout.indexOf(inspector.hotspot_destination_combo)
    )
    assert inspector.hotspot_sound_label.text() == "Play"
    assert then_layout.indexOf(inspector.hotspot_destination_combo) < (
        then_layout.indexOf(inspector.hotspot_sound_label)
    )
    assert then_layout.indexOf(inspector.hotspot_sound_label) < (
        then_layout.indexOf(inspector.hotspot_sound_combo)
    )
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


def test_transform_edit_uses_current_or_higher_size_and_emits_exact_inputs(
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
        "Create a new version with only the requested change."
    )


def test_transform_reinterpret_has_source_similarity_resolution_and_action(
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

    assert inspector.inspector_tabs.tabText(inspector._transform_tab_index) == "Transform"
    reinterpret_group = inspector.refine_background_button.parentWidget()
    edit_group = inspector.edit_background_button.parentWidget()
    assert isinstance(reinterpret_group, QGroupBox)
    assert isinstance(edit_group, QGroupBox)
    assert not reinterpret_group.title()
    assert not edit_group.title()
    assert inspector.reinterpret_section_label.text() == "Reinterpret"
    assert inspector.edit_section_label.text() == "Edit"
    assert inspector.reinterpret_section_label.font().pointSizeF() == (
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
        refine_reason="Ready to reinterpret",
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
    assert inspector.refine_background_button.text() == "Reinterpret"
    assert inspector.refine_background_button.toolTip() == (
        "Create a new version using the current image, Description."
    )


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
    assert inspector.hotspot_target_label.text() == "Go to"
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

    no_destination_index = next(
        index
        for index in range(inspector.hotspot_destination_combo.count())
        if inspector.hotspot_destination_combo.itemData(index) is None
    )
    inspector.hotspot_destination_combo.setCurrentIndex(no_destination_index)
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
    assert inspector.hotspot_destination_combo.currentText() == "No destination"


def test_hotspot_play_selector_assigns_catalog_sound(
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

    sound_index = next(
        index
        for index in range(inspector.hotspot_sound_combo.count())
        if inspector.hotspot_sound_combo.itemData(index) == sound.id
    )
    inspector.hotspot_sound_combo.setCurrentIndex(sound_index)

    hotspot_set = controller.document.cards[0].active_revision.hotspot_set
    assert hotspot_set is not None
    assert hotspot_set.interactions[0].sound_id == sound.id


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

    assert (
        inspector._combo_index_for_data(
            inspector.reference_combo,
            source.id,
        )
        == -1
    )
    identity_index = inspector._combo_index_for_data(
        inspector.reference_combo,
        portrait.id,
    )
    inspector.reference_combo.setCurrentIndex(identity_index)

    assert controller.document.cards[0].active_revision.references == (
        ResolvedCardReference(target_card_id=portrait.id),
    )
    assert applied[-1][0] == "Reference changed"
    assert controller.undo_if_current(applied[-1][1])  # type: ignore[arg-type]
    inspector.render(controller.document, source.id)
    assert controller.document.cards[0].active_revision.references == ()


def test_unassigned_card_selectors_focus_the_next_available_card(
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

    assert inspector.reference_combo.currentData() is None
    assert _popup_focused_data(inspector.reference_combo, application) == following.id
    assert inspector.reference_combo.currentData() is None
    assert inspector.hotspot_destination_combo.currentData() is None
    assert _popup_focused_data(inspector.hotspot_destination_combo, application) == following.id
    assert inspector.hotspot_destination_combo.currentData() is None

    inspector.reference_combo.setCurrentIndex(
        inspector._combo_index_for_data(inspector.reference_combo, previous.id)
    )
    assert inspector.additional_reference_combo.isEnabled()
    assert inspector.additional_reference_combo.currentData() is None
    assert _popup_focused_data(inspector.additional_reference_combo, application) == following.id
    assert inspector.additional_reference_combo.currentData() is None
    inspector.close()


def test_unassigned_card_selector_focuses_previous_card_at_end(
    application: QApplication,
) -> None:
    previous = Card(name="Previous")
    source = Card(name="Source")
    controller = DocumentController(Stack(name="Demo", cards=(previous, source)))
    inspector = Inspector(controller)
    inspector.render(controller.document, source.id)
    inspector.show()
    application.processEvents()

    assert _popup_focused_data(inspector.reference_combo, application) == previous.id
    assert inspector.reference_combo.currentData() is None
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
    assert not inspector.additional_reference_combo.isEnabled()
    inspector.reference_combo.setCurrentIndex(
        inspector._combo_index_for_data(inspector.reference_combo, portrait.id)
    )
    assert inspector.additional_reference_combo.isEnabled()
    assert (
        inspector._combo_index_for_data(
            inspector.additional_reference_combo,
            portrait.id,
        )
        == -1
    )

    inspector.additional_reference_combo.setCurrentIndex(
        inspector._combo_index_for_data(
            inspector.additional_reference_combo,
            room.id,
        )
    )
    assert controller.document.cards[0].active_revision.references == (
        ResolvedCardReference(target_card_id=portrait.id),
        ResolvedCardReference(target_card_id=room.id),
    )

    inspector.reference_combo.setCurrentIndex(0)
    assert controller.document.cards[0].active_revision.references == (
        ResolvedCardReference(target_card_id=room.id),
    )
    assert inspector.reference_combo.currentData() == room.id
    assert inspector.additional_reference_combo.currentData() is None


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

    assert inspector.reference_combo.currentText() == ("Missing: Former portrait")
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
    assert "image 1" in inspector.reference_combo.toolTip()


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

    destination_index = next(
        index
        for index in range(inspector.hotspot_destination_combo.count())
        if inspector.hotspot_destination_combo.itemData(index) == destination.id
    )
    inspector.hotspot_destination_combo.setCurrentIndex(destination_index)
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
