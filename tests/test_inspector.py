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
    QInputDialog,
    QLabel,
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
    GenerateResolution,
    output_dimensions,
)
from hotcards.domain.models import (
    Card,
    CardRevision,
    DirectGenerateProvenance,
    EditPreserveOptions,
    GeneratedBackground,
    GenerateInputs,
    HotspotConditions,
    HotspotKeyChanges,
    HotspotSet,
    ImageOperationSettings,
    Interaction,
    KeyDefinition,
    NavigateAction,
    RefineTransformation,
    ResolvedCardReference,
    Stack,
    StyleDefinition,
    UnresolvedCardReference,
)
from hotcards.ui.inspector import Inspector
from hotcards.ui.utility_windows import KeyManagerWindow, StyleManagerWindow


@pytest.fixture(scope="module")
def application() -> QApplication:
    return QApplication.instance() or QApplication([])


def _interaction(label: str = "Door") -> Interaction:
    return Interaction(
        label=label,
        action=NavigateAction(target=UnresolvedCardReference()),
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
                width=592,
                height=448,
                step_count=4,
                generated_at=generated_at,
                duration_seconds=1,
            ),
        ),
        created_at=generated_at,
    )


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

    assert inspector.inspector_tabs.count() == 4
    assert inspector.inspector_tabs.tabText(0) == "Generate"
    assert inspector.inspector_tabs.tabText(1) == "Refine"
    assert inspector.inspector_tabs.tabText(2) == "Edit"
    assert inspector.inspector_tabs.tabText(3) == "Hotspots"
    assert not hasattr(inspector, "style_list")
    assert not hasattr(inspector, "key_list")
    assert [checkbox.text() for checkbox in inspector.edit_preserve_checkboxes.values()] == [
        "Subject identity",
        "Pose and expression",
        "Composition and framing",
        "Background",
        "Lighting and color",
        "Existing text and logos",
    ]
    assert [checkbox.isChecked() for checkbox in inspector.edit_preserve_checkboxes.values()] == [
        True,
        True,
        True,
        False,
        False,
        False,
    ]
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
    assert inspector.add_condition_button.text() == "+ Add condition"
    assert inspector.add_condition_button.isFlat()
    assert inspector.add_key_change_button.text() == "+ Add key change"
    assert inspector.add_key_change_button.isFlat()
    assert not hasattr(inspector, "clear_all_keys_checkbox")
    hotspot_layout = inspector.hotspot_list.parentWidget().layout()
    assert hotspot_layout is not None
    then_layout = inspector.hotspot_then_panel.layout()
    assert then_layout is not None
    assert then_layout.indexOf(inspector.key_change_table) < (
        then_layout.indexOf(inspector.add_key_change_button)
    )
    assert then_layout.indexOf(inspector.add_key_change_button) < (
        then_layout.indexOf(inspector.hotspot_target_label)
    )
    assert then_layout.indexOf(inspector.hotspot_target_label) < (
        then_layout.indexOf(inspector.hotspot_destination_combo)
    )
    assert hotspot_layout.stretch(hotspot_layout.indexOf(inspector.hotspot_list)) == 1
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


def test_edit_tab_uses_current_or_higher_size_and_emits_exact_inputs(
    application: QApplication,
) -> None:
    revision = CardRevision(
        description="A legacy courtyard",
        background=_background(),
    )
    card = Card(name="Courtyard", revisions=(revision,))
    controller = DocumentController(Stack(name="Demo", cards=(card,)))
    inspector = Inspector(controller)
    requests: list[tuple[object, object, object]] = []
    inspector.edit_background_requested.connect(
        lambda instruction, preserve, output: requests.append((instruction, preserve, output))
    )

    inspector.render(
        controller.document,
        card.id,
        refine_source_size=(1024, 768),
    )

    assert inspector.edit_resolution_combo.count() == 2
    assert inspector.edit_resolution_combo.itemText(0) == ("Current (1024 x 768)")
    assert inspector.edit_resolution_combo.itemData(0) == "current"
    assert inspector.edit_resolution_combo.itemText(1) == ("1024 (1184 x 880)")
    inspector.edit_instruction_edit.setPlainText("  Open the gate.  ")
    inspector.edit_background_button.click()

    assert len(requests) == 1
    instruction, preserve, output = requests[0]
    assert instruction == "Open the gate."
    assert preserve == EditPreserveOptions(
        subject_identity=True,
        pose_and_expression=True,
        composition_and_framing=True,
    )
    assert output.mode == "current"
    assert (output.width, output.height) == (1024, 768)
    assert "Expanded prompt:" in inspector.edit_background_button.toolTip()
    assert "Token budget: 512" in inspector.edit_background_button.toolTip()


def test_refine_tab_has_transformation_resolution_and_action_only(
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
        refine_source_size=(592, 448),
    )

    assert inspector.inspector_tabs.tabText(inspector._refine_tab_index) == "Refine"
    assert (
        RefineTransformation(inspector.refine_transformation_combo.currentData())
        is RefineTransformation.BALANCED
    )
    assert [
        inspector.refine_transformation_combo.itemText(index)
        for index in range(inspector.refine_transformation_combo.count())
    ] == [
        "Reimagine (0.25)",
        "Balanced (0.50)",
        "Preserve (0.75)",
    ]
    assert [
        inspector.refine_resolution_combo.itemText(index)
        for index in range(inspector.refine_resolution_combo.count())
    ] == [
        "768 (880 x 672)",
        "1024 (1184 x 880)",
    ]
    content = inspector.refine_background_button.parentWidget()
    layout = content.layout()
    assert layout is not None
    assert layout.indexOf(inspector.refine_transformation_label) < layout.indexOf(
        inspector.refine_transformation_combo
    )
    assert layout.indexOf(inspector.refine_transformation_combo) < layout.indexOf(
        inspector.refine_resolution_label
    )
    assert layout.indexOf(inspector.refine_resolution_label) < layout.indexOf(
        inspector.refine_resolution_combo
    )
    assert layout.indexOf(inspector.refine_resolution_combo) < layout.indexOf(
        inspector.refine_background_button
    )
    assert not hasattr(inspector, "refine_source_card_combo")
    assert not hasattr(inspector, "refine_source_revision_combo")

    requested: list[tuple[object, object]] = []
    inspector.refine_background_requested.connect(
        lambda transformation, resolution: requested.append((transformation, resolution))
    )
    inspector.set_refine_capabilities(
        can_refine=True,
        refine_reason="Ready to refine",
        busy=False,
        refining=False,
    )
    inspector.refine_background_button.click()

    assert requested == [
        (
            RefineTransformation.BALANCED,
            GenerateResolution.RESOLUTION_768,
        )
    ]
    assert "Current image + Description" in (inspector.refine_background_button.toolTip())
    assert "Balanced (0.50)" in inspector.refine_background_button.toolTip()
    assert "880 x 672" in inspector.refine_background_button.toolTip()


def test_refine_resolution_handles_legacy_size_and_maximum(
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
        refine_source_size=(1024, 768),
    )
    assert inspector.refine_resolution_combo.count() == 1
    assert inspector.refine_resolution_combo.currentData() is GenerateResolution.RESOLUTION_1024
    assert not inspector.refine_error.isVisible()

    inspector.render(
        controller.document,
        card.id,
        refine_source_size=(1184, 880),
    )
    assert inspector.refine_resolution_combo.count() == 0
    assert not inspector.refine_resolution_combo.isEnabled()
    assert "maximum Refine resolution" in inspector.refine_error.text()


def test_key_manager_manages_global_names_and_lists_hotspot_usages(
    application: QApplication,
) -> None:
    red_key = KeyDefinition(name="Red key")
    interaction = Interaction(
        key_changes=HotspotKeyChanges(remove=(red_key.id,)),
        conditions=HotspotConditions(requires=(red_key.id,)),
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
    assert manager.usage_list.item(0).text() == (
        "Castle · Version 1\nLose Red key · Requires, Removes"
    )

    manager.name_edit.setText("Ruby key")
    usage_item = manager.usage_list.item(0)
    manager._editing_finished(
        manager.show_usage_button,
        Qt.FocusReason.MouseFocusReason,
    )
    assert controller.document.keys[0].name == "Ruby key"
    assert manager.usage_list.item(0) is usage_item
    assert manager.usage_list.item(0).text().splitlines()[1] == (
        "Lose Ruby key · Requires, Removes"
    )
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
    manager.show_usage_button.click()
    assert requested == [(card.id, card.active_revision.id, interaction.id)]

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
    assert inspector.hotspot_list.currentItem().text() == (
        "Lose Red key · Gain Door open\n→ Castle"
    )
    assert inspector.condition_table.rowCount() == 1
    assert inspector.condition_table.horizontalHeaderItem(0).text() == "State"
    assert inspector.condition_table.horizontalHeaderItem(1).text() == "Key"
    condition_state = inspector.condition_table.cellWidget(0, 0)
    condition_key = inspector.condition_table.cellWidget(0, 1)
    assert isinstance(condition_state, QComboBox)
    assert isinstance(condition_key, QComboBox)
    assert condition_state.currentText() == "Has"
    assert condition_key.currentData() == red_key.id
    condition_remove_cell = inspector.condition_table.cellWidget(0, 2)
    condition_remove = condition_remove_cell.findChild(QToolButton)
    assert condition_remove is not None
    assert condition_remove_cell.layout().itemAt(0).alignment() == (Qt.AlignmentFlag.AlignCenter)
    assert condition_remove.width() == condition_remove.height()
    assert condition_remove.size() == QSize(20, 20)
    assert inspector.condition_table.height() == (
        inspector.condition_table.horizontalHeader().sizeHint().height()
        + sum(
            inspector.condition_table.rowHeight(row)
            for row in range(inspector.condition_table.rowCount())
        )
        + 2
    )
    assert inspector.no_conditions_label.isHidden()
    assert inspector.key_change_table.rowCount() == 2
    remove_change = inspector.key_change_table.cellWidget(0, 0)
    grant_change = inspector.key_change_table.cellWidget(1, 0)
    assert isinstance(remove_change, QComboBox)
    assert isinstance(grant_change, QComboBox)
    assert remove_change.currentText() == "Lose"
    assert grant_change.currentText() == "Gain"
    key_change_remove_cell = inspector.key_change_table.cellWidget(0, 2)
    key_change_remove = key_change_remove_cell.findChild(QToolButton)
    assert key_change_remove is not None
    assert key_change_remove_cell.layout().itemAt(0).alignment() == (Qt.AlignmentFlag.AlignCenter)
    assert key_change_remove.width() == key_change_remove.height()
    assert key_change_remove.size() == QSize(20, 20)
    assert inspector.no_key_changes_label.isHidden()
    assert inspector.hotspot_target_label.text() == "Go to"
    assert inspector.hotspot_summary.text() == (
        "When the runner has Red key, lose Red key, then gain Door open, then go to Castle."
    )

    grant_key = inspector.key_change_table.cellWidget(1, 1)
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


def test_empty_hotspot_rule_sections_use_plain_language_placeholders(
    application: QApplication,
) -> None:
    interaction = Interaction()
    card = Card(
        name="Card",
        revisions=(CardRevision(hotspot_set=HotspotSet(interactions=(interaction,))),),
    )
    inspector = Inspector(DocumentController(Stack(name="Demo", cards=(card,))))

    inspector.render(inspector.controller.document, card.id)

    assert inspector.condition_table.isHidden()
    assert inspector.no_conditions_label.text() == "No conditions (always activates)"
    assert not inspector.no_conditions_label.isHidden()
    assert inspector.key_change_table.isHidden()
    assert inspector.no_key_changes_label.text() == "No key changes"
    assert not inspector.no_key_changes_label.isHidden()
    assert inspector.hotspot_destination_combo.currentText() == "No destination"


def test_contextual_key_creation_label_does_not_reserve_free_form_name(
    application: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key = KeyDefinition(name="Create New Key...")
    interaction = Interaction()
    card = Card(
        name="Card",
        revisions=(CardRevision(hotspot_set=HotspotSet(interactions=(interaction,))),),
    )
    controller = DocumentController(Stack(name="Demo", keys=(key,), cards=(card,)))
    inspector = Inspector(controller)
    inspector.render(controller.document, card.id)
    monkeypatch.setattr(
        QInputDialog,
        "getItem",
        lambda *_args, **_kwargs: ("Create New Key...", True),
    )

    inspector.add_condition_button.click()

    changed = controller.document.cards[0].active_revision.hotspot_set
    assert changed is not None
    assert changed.interactions[0].conditions.requires == (key.id,)


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
def test_resolution_selector_shows_actual_dimensions_for_every_preset(
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
    ] == [
        f"{resolution.value} "
        f"({output_dimensions(resolution, aspect_ratio)[0]} x "
        f"{output_dimensions(resolution, aspect_ratio)[1]})"
        for resolution in GenerateResolution
    ]
    assert inspector.resolution_combo.currentData() == GenerateResolution.RESOLUTION_512
    width, height = output_dimensions(
        GenerateResolution.RESOLUTION_512,
        aspect_ratio,
    )
    assert (
        f"Resolution: 512 square-equivalent ({width} x {height})"
        in inspector.generate_background_button.toolTip()
    )


def test_resolution_selector_is_revision_local_and_undoable(
    application: QApplication,
) -> None:
    first = CardRevision()
    second = CardRevision(generate_resolution=GenerateResolution.RESOLUTION_256)
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
            GenerateResolution.RESOLUTION_1024,
        )
    )

    revisions = controller.document.cards[0].revisions
    assert revisions[0].generate_resolution is GenerateResolution.RESOLUTION_1024
    assert revisions[1].generate_resolution is GenerateResolution.RESOLUTION_256
    assert (
        autosaves[-1].cards[0].active_revision.generate_resolution
        is GenerateResolution.RESOLUTION_1024
    )
    assert applied[-1][0] == "Generate resolution changed"
    assert controller.undo_if_current(applied[-1][1])  # type: ignore[arg-type]
    inspector.render(controller.document, card.id)
    assert inspector.resolution_combo.currentData() == GenerateResolution.RESOLUTION_512
    assert controller.redo()
    inspector.render(controller.document, card.id)
    assert inspector.resolution_combo.currentData() == GenerateResolution.RESOLUTION_1024


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


def test_add_hotspot_persists_and_selects_area_less_entry(
    application: QApplication,
) -> None:
    card = Card(name="Card")
    controller = DocumentController(Stack(name="Demo", cards=(card,)))
    inspector = Inspector(controller)
    inspector.render(controller.document, card.id)

    inspector.add_hotspot_button.click()

    hotspot_set = controller.document.cards[0].active_revision.hotspot_set
    assert hotspot_set is not None
    assert len(hotspot_set.interactions) == 1
    interaction = hotspot_set.interactions[0]
    assert interaction.label == "New Hotspot"
    assert interaction.polygons == ()
    assert inspector.selected_interaction_id == interaction.id


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
    assert changed.interactions[0].label == "Destination"
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
