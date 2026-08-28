"""Focused offscreen tests for revision-local authoring controls."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QInputDialog,
    QLabel,
    QRadioButton,
    QStyle,
    QToolButton,
)

from hypergen.application.commands import DeleteCardCommand, RenameCardCommand
from hypergen.application.document_controller import DocumentController
from hypergen.domain.models import (
    Card,
    CardRevision,
    HotspotConditions,
    HotspotKeyChanges,
    HotspotSet,
    ImagePrompt,
    Interaction,
    KeyDefinition,
    NavigateAction,
    ResolvedCardReference,
    Stack,
    StyleDefinition,
    UnresolvedCardReference,
)
from hypergen.generation.image_prompt_preparation import (
    IMAGE_PROMPT_PREPARATION_VERSION,
)
from hypergen.ui.inspector import Inspector


@pytest.fixture(scope="module")
def application() -> QApplication:
    return QApplication.instance() or QApplication([])


def _interaction(label: str = "Door") -> Interaction:
    return Interaction(
        label=label,
        action=NavigateAction(target=UnresolvedCardReference()),
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
    assert inspector.inspector_tabs.tabText(0) == "Image"
    assert inspector.inspector_tabs.tabText(1) == "Styles"
    assert inspector.inspector_tabs.tabText(2) == "Hotspots"
    assert inspector.inspector_tabs.tabText(3) == "Keys"
    assert all(
        label.text() != "Keys are global to this stack."
        for label in inspector.findChildren(QLabel)
    )
    root_layout = inspector.layout()
    assert root_layout is not None
    assert root_layout.contentsMargins().top() == 16
    assert root_layout.contentsMargins().bottom() == 16
    assert root_layout.contentsMargins().left() == inspector.style().pixelMetric(
        QStyle.PixelMetric.PM_LayoutLeftMargin
    )
    assert inspector.description_edit.placeholderText() == "Description"
    assert inspector.description_edit.minimumHeight() == round(
        (inspector.description_edit.fontMetrics().lineSpacing() * 10 + 20) * 1.25
    )
    assert inspector.description_edit.maximumHeight() > (inspector.description_edit.minimumHeight())
    assert inspector.enrich_button.text() == "Prepare Image Prompt"
    assert inspector.generate_background_button.text() == "Generate Image"
    assert inspector.description_toggle.isHidden()
    assert isinstance(inspector.description_button, QRadioButton)
    assert isinstance(inspector.image_prompt_button, QRadioButton)
    assert not isinstance(inspector.reference_panel, QFrame)
    assert inspector.reference_panel.layout().contentsMargins().isNull()
    content_layout = inspector.reference_panel.parentWidget().layout()
    assert content_layout is not None
    assert content_layout.stretch(content_layout.indexOf(inspector.description_edit)) == 1
    assert content_layout.indexOf(inspector.description_edit) < (
        content_layout.indexOf(inspector.description_toggle)
    )
    assert content_layout.indexOf(inspector.description_toggle) < (
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
        content_layout.indexOf(inspector.enrich_button)
    )
    assert content_layout.indexOf(inspector.enrich_button) < (
        content_layout.indexOf(inspector.generate_background_button)
    )
    assert inspector.style_combo.currentText() == "No Style"
    assert not hasattr(inspector, "clear_background_button")
    styles_layout = inspector.style_list.parentWidget().layout()
    assert styles_layout is not None
    assert styles_layout.stretch(styles_layout.indexOf(inspector.style_list)) == 1
    style_controls = styles_layout.itemAt(styles_layout.indexOf(inspector.style_list) + 1).layout()
    assert style_controls is not None
    assert style_controls.indexOf(inspector.delete_style_button) < style_controls.indexOf(
        inspector.add_style_button
    )
    assert styles_layout.indexOf(inspector.style_name_label) < (
        styles_layout.indexOf(inspector.style_name_edit)
    )
    assert styles_layout.indexOf(inspector.style_name_edit) < (
        styles_layout.indexOf(inspector.style_prompt_label)
    )
    assert styles_layout.indexOf(inspector.style_prompt_label) < (
        styles_layout.indexOf(inspector.style_prompt_edit)
    )
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
    keys_layout = inspector.key_list.parentWidget().layout()
    assert keys_layout is not None
    key_controls = keys_layout.itemAt(
        keys_layout.indexOf(inspector.key_list) + 1
    ).layout()
    assert key_controls is not None
    assert key_controls.indexOf(inspector.delete_key_button) < (
        key_controls.indexOf(inspector.add_key_button)
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


def test_keys_tab_manages_global_names_and_lists_hotspot_usages(
    application: QApplication,
) -> None:
    red_key = KeyDefinition(name="Red key")
    interaction = Interaction(
        key_changes=HotspotKeyChanges(remove=(red_key.id,)),
        conditions=HotspotConditions(requires=(red_key.id,)),
    )
    card = Card(
        name="Castle",
        revisions=(
            CardRevision(hotspot_set=HotspotSet(interactions=(interaction,))),
        ),
    )
    controller = DocumentController(
        Stack(name="Demo", keys=(red_key,), cards=(card,))
    )
    inspector = Inspector(controller)
    inspector.render(controller.document, card.id)

    assert inspector.key_list.currentItem().text() == "Red key — 1 use"
    assert inspector.key_name_edit.text() == "Red key"
    assert not inspector.delete_key_button.isEnabled()
    assert inspector.key_usage_list.count() == 1
    assert inspector.key_usage_list.item(0).text() == (
        "Castle · Version 1\n"
        "Lose Red key"
    )

    inspector.key_name_edit.setText("Ruby key")
    usage_item = inspector.key_usage_list.item(0)
    inspector._key_name_editing_finished(
        inspector.show_hotspot_usage_button,
        Qt.FocusReason.MouseFocusReason,
    )
    assert controller.document.keys[0].name == "Ruby key"
    assert inspector.key_usage_list.item(0) is usage_item
    assert inspector.key_usage_list.item(0).text().splitlines()[1] == (
        "Lose Ruby key"
    )
    hotspot_set = controller.document.cards[0].active_revision.hotspot_set
    assert hotspot_set is not None
    assert hotspot_set.interactions[0].label == "Lose Ruby key"

    requested: list[tuple[object, object, object]] = []
    inspector.hotspot_usage_requested.connect(
        lambda card_id, revision_id, interaction_id: requested.append(
            (card_id, revision_id, interaction_id)
        )
    )
    inspector.key_usage_list.setCurrentRow(0)
    inspector.show_hotspot_usage_button.click()
    assert requested == [
        (card.id, card.active_revision.id, interaction.id)
    ]

    inspector.add_key_button.click()
    assert [key.name for key in controller.document.keys] == [
        "Ruby key",
        "New Key",
    ]
    assert inspector.delete_key_button.isEnabled()
    inspector.delete_key_button.click()
    assert [key.name for key in controller.document.keys] == ["Ruby key"]


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
        action=NavigateAction(
            target=ResolvedCardReference(target_card_id=destination.id)
        ),
    )
    source = Card(
        name="Source",
        revisions=(
            CardRevision(hotspot_set=HotspotSet(interactions=(interaction,))),
        ),
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
    assert condition_remove_cell.layout().itemAt(0).alignment() == (
        Qt.AlignmentFlag.AlignCenter
    )
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
    assert key_change_remove_cell.layout().itemAt(0).alignment() == (
        Qt.AlignmentFlag.AlignCenter
    )
    assert key_change_remove.width() == key_change_remove.height()
    assert key_change_remove.size() == QSize(20, 20)
    assert inspector.no_key_changes_label.isHidden()
    assert inspector.hotspot_target_label.text() == "Go to"
    assert inspector.hotspot_summary.text() == (
        "When the runner has Red key, lose Red key, then gain Door open, "
        "then go to Castle."
    )

    grant_key = inspector.key_change_table.cellWidget(1, 1)
    assert isinstance(grant_key, QComboBox)
    grant_key.setCurrentIndex(
        next(
            index
            for index in range(grant_key.count())
            if grant_key.itemData(index) == red_key.id
        )
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
        revisions=(
            CardRevision(hotspot_set=HotspotSet(interactions=(interaction,))),
        ),
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
        revisions=(
            CardRevision(hotspot_set=HotspotSet(interactions=(interaction,))),
        ),
    )
    controller = DocumentController(
        Stack(name="Demo", keys=(key,), cards=(card,))
    )
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

    assert controller.document.cards[0].active_revision.reference == (
        ResolvedCardReference(target_card_id=portrait.id)
    )
    assert applied[-1][0] == "Reference changed"
    assert controller.undo_if_current(applied[-1][1])  # type: ignore[arg-type]
    inspector.render(controller.document, source.id)
    assert controller.document.cards[0].active_revision.reference is None


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


def test_styles_tab_edits_global_definition_and_deletes_with_undo(
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
    inspector = Inspector(controller)
    inspector.render(controller.document, source.id)
    applied: list[tuple[str, object]] = []
    inspector.change_applied.connect(lambda message, token: applied.append((message, token)))

    assert inspector.style_list.currentItem().text() == "Ink"
    assert inspector.style_name_edit.text() == "Ink"
    assert inspector.style_prompt_edit.toPlainText() == "Rendered in ink."
    inspector.style_name_edit.setText("Etching")
    inspector.style_prompt_edit.setPlainText("Fine etched linework.")
    assert inspector._commit_style()

    changed_style = controller.document.styles[0]
    assert changed_style.id == ink.id
    assert changed_style.name == "Etching"
    assert changed_style.prompt_text == "Fine etched linework."
    assert all(card.active_revision.style_id == ink.id for card in controller.document.cards)

    inspector.add_style_button.click()
    assert len(controller.document.styles) == 2
    assert inspector.style_name_edit.text() == "New Style"
    assert inspector.style_name_edit.selectedText() == "New Style"

    inspector.delete_style_button.click()
    assert len(controller.document.styles) == 1
    assert controller.document.styles[0].id == ink.id
    assert applied[-1][0] == "Style deleted"
    assert controller.undo_if_current(applied[-1][1])  # type: ignore[arg-type]
    assert len(controller.document.styles) == 2

    inspector.render(controller.document, source.id)
    inspector.style_list.setCurrentRow(0)
    inspector.delete_style_button.click()
    assert controller.document.styles[0].name == "New Style"
    assert controller.document.new_card_style_id is None
    assert all(card.active_revision.style_id is None for card in controller.document.cards)


def test_deleted_reference_is_shown_as_unresolved(
    application: QApplication,
) -> None:
    destination = Card(name="Former portrait")
    revision = CardRevision(reference=ResolvedCardReference(target_card_id=destination.id))
    source = Card(name="Source", revisions=(revision,))
    controller = DocumentController(Stack(name="Demo", cards=(source, destination)))
    inspector = Inspector(controller)
    inspector.render(controller.document, source.id)

    changed = controller.execute(DeleteCardCommand(card_id=destination.id))
    inspector.render(changed, source.id)

    assert inspector.reference_combo.currentText() == ("Missing: Former portrait")
    assert controller.document.cards[0].active_revision.reference == (
        UnresolvedCardReference(target_name="Former portrait")
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


def test_image_prompt_status_and_generation_source(
    application: QApplication,
) -> None:
    revision = CardRevision(
        description="A courtyard",
        image_prompt=ImagePrompt(
            text="A richly detailed courtyard",
            source_description="A courtyard",
        ),
    )
    card = Card(name="Card", revisions=(revision,))
    controller = DocumentController(Stack(name="Demo", cards=(card,)))
    inspector = Inspector(controller)
    inspector.render(controller.document, card.id)

    assert not inspector.description_toggle.isHidden()
    assert inspector.description_button.isChecked()
    assert inspector.description_edit.toPlainText() == ("A courtyard")
    assert "Using: Description" in inspector.enrich_button.toolTip()
    assert "Using: Image Prompt" in inspector.generate_background_button.toolTip()
    assert inspector.enrich_button.text() == "Image Prompt Current"
    assert not inspector.enrich_button.isEnabled()
    assert "Current" in inspector.enrich_button.toolTip()
    inspector.set_image_prompt_capabilities(
        can_enrich=True,
        reason="Ready to prepare Image Prompt",
        busy=False,
    )

    inspector.description_edit.setPlainText("A changed courtyard")
    assert inspector.enrich_button.text() == "Update Image Prompt"
    assert inspector.enrich_button.isEnabled()
    assert "Out of date" in inspector.enrich_button.toolTip()
    assert inspector.commit_revision_metadata()

    current = controller.document.cards[0].active_revision
    assert current.image_prompt is not None
    assert current.image_prompt.text == "A richly detailed courtyard"

    inspector.image_prompt_button.click()
    inspector.description_edit.setPlainText("A manually revised courtyard")
    assert inspector.commit_revision_metadata()

    inspector.description_edit.clear()
    assert inspector.commit_revision_metadata()
    assert controller.document.cards[0].active_revision.image_prompt is None
    assert inspector.description_toggle.isHidden()
    assert inspector.description_edit.toPlainText() == "A changed courtyard"
    assert "Using: Image Prompt" in inspector.generate_background_button.toolTip()
    assert controller.undo()
    assert controller.document.cards[0].active_revision.image_prompt is not None


def test_manual_image_prompt_edit_refreshes_changed_description(
    application: QApplication,
) -> None:
    revision = CardRevision(
        description="A courtyard",
        image_prompt=ImagePrompt(
            text="A richly detailed courtyard",
            source_description="A courtyard",
            model_identifier="qwen3.5:9b-mlx",
            prompt_version=IMAGE_PROMPT_PREPARATION_VERSION,
        ),
    )
    card = Card(name="Card", revisions=(revision,))
    controller = DocumentController(Stack(name="Demo", cards=(card,)))
    inspector = Inspector(controller)
    inspector.render(controller.document, card.id)
    inspector.set_image_prompt_capabilities(
        can_enrich=True,
        reason="Ready to prepare Image Prompt",
        busy=False,
        model_identifier="qwen3.5:9b-mlx",
        prompt_version=IMAGE_PROMPT_PREPARATION_VERSION,
    )

    inspector.description_edit.setPlainText("A moonlit courtyard")
    assert inspector.commit_revision_metadata()
    assert not inspector.has_current_image_prompt()

    inspector.image_prompt_button.click()
    inspector.description_edit.setPlainText("A manually revised moonlit courtyard")

    assert inspector.has_current_image_prompt()
    assert inspector.enrich_button.text() == "Image Prompt Current"
    inspector.show()
    inspector.description_edit.setFocus()
    application.processEvents()
    assert inspector.description_edit.hasFocus()
    inspector.render(controller.document, card.id)
    assert (
        inspector.description_edit.toPlainText()
        == "A manually revised moonlit courtyard"
    )
    assert inspector.has_current_image_prompt()
    assert inspector.commit_revision_metadata(render_change=False)

    image_prompt = controller.document.cards[0].active_revision.image_prompt
    assert image_prompt is not None
    assert image_prompt.text == "A manually revised moonlit courtyard"
    assert image_prompt.source_description == "A moonlit courtyard"
    assert image_prompt.model_identifier == "qwen3.5:9b-mlx"
    assert image_prompt.prompt_version == IMAGE_PROMPT_PREPARATION_VERSION
    assert inspector.has_current_image_prompt()


def test_switching_cards_defaults_to_description(
    application: QApplication,
) -> None:
    first = Card(
        name="First",
        revisions=(
            CardRevision(
                description="First description",
                image_prompt=ImagePrompt(
                    text="First prompt",
                    source_description="First description",
                ),
            ),
        ),
    )
    second = Card(
        name="Second",
        revisions=(
            CardRevision(
                description="Second description",
                image_prompt=ImagePrompt(
                    text="Second prompt",
                    source_description="Second description",
                ),
            ),
        ),
    )
    controller = DocumentController(Stack(name="Demo", cards=(first, second)))
    inspector = Inspector(controller)
    inspector.render(controller.document, first.id)
    inspector.image_prompt_button.click()
    assert inspector.image_prompt_button.isChecked()

    inspector.render(controller.document, second.id)

    assert inspector.description_button.isChecked()
    assert inspector.description_edit.toPlainText() == "Second description"


def test_changing_model_marks_image_prompt_out_of_date(
    application: QApplication,
) -> None:
    revision = CardRevision(
        description="A courtyard",
        image_prompt=ImagePrompt(
            text="A richly detailed courtyard",
            source_description="A courtyard",
            model_identifier="qwen3.5:9b-mlx",
        ),
    )
    card = Card(name="Card", revisions=(revision,))
    controller = DocumentController(Stack(name="Demo", cards=(card,)))
    inspector = Inspector(controller)
    inspector.render(controller.document, card.id)

    inspector.set_image_prompt_capabilities(
        can_enrich=True,
        reason="Ready to prepare Image Prompt",
        busy=False,
        model_identifier="qwen3.5:9b-mlx",
    )
    assert inspector.enrich_button.text() == "Image Prompt Current"
    assert not inspector.enrich_button.isEnabled()

    inspector.set_image_prompt_capabilities(
        can_enrich=True,
        reason="Ready to prepare Image Prompt",
        busy=False,
        model_identifier="llama3.2:latest",
    )

    assert inspector.enrich_button.text() == "Update Image Prompt"
    assert inspector.enrich_button.isEnabled()
    assert "Out of date" in inspector.enrich_button.toolTip()

    inspector.set_image_prompt_capabilities(
        can_enrich=True,
        reason="Ready to prepare Image Prompt",
        busy=False,
        model_identifier="qwen3.5:9b-mlx",
    )
    assert inspector.enrich_button.text() == "Image Prompt Current"
    assert not inspector.enrich_button.isEnabled()


def test_using_labels_name_single_reference(
    application: QApplication,
) -> None:
    reference = Card(name="Reference")
    source = Card(
        name="Source",
        revisions=(
            CardRevision(
                description="A courtyard",
                reference=ResolvedCardReference(target_card_id=reference.id),
            ),
        ),
    )
    controller = DocumentController(Stack(name="Demo", cards=(source, reference)))
    inspector = Inspector(controller)
    inspector.render(controller.document, source.id)

    assert inspector.reference_label.text() == "Reference"
    assert "Using: Description + Reference" in inspector.enrich_button.toolTip()
    assert "Using: Image Prompt + Reference" in inspector.generate_background_button.toolTip()


def test_generate_requires_a_current_non_empty_image_prompt(
    application: QApplication,
) -> None:
    revision = CardRevision(
        description="A courtyard",
        image_prompt=ImagePrompt(
            text="A richly detailed courtyard",
            source_description="A courtyard",
        ),
    )
    card = Card(name="Card", revisions=(revision,))
    controller = DocumentController(Stack(name="Demo", cards=(card,)))
    inspector = Inspector(controller)
    inspector.render(controller.document, card.id)

    assert inspector.has_current_image_prompt()
    assert inspector.has_description_input()

    inspector.image_prompt_button.click()
    inspector.description_edit.clear()
    assert not inspector.has_current_image_prompt()
    assert inspector.commit_revision_metadata()
    assert controller.document.cards[0].active_revision.image_prompt is None


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


def test_ai_activity_is_reflected_on_the_initiating_buttons(
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
    inspector.set_image_prompt_capabilities(
        can_enrich=False,
        reason="Preparing Image Prompt",
        busy=True,
    )

    assert inspector.generate_background_button.text() == "Generating…"
    assert inspector.enrich_button.text() == "Preparing Image Prompt…"


def test_image_prompt_has_no_separate_proposal_editor(
    application: QApplication,
) -> None:
    inspector = Inspector(DocumentController(Stack(name="Demo")))

    assert not hasattr(inspector, "image_prompt_review")
    assert not hasattr(inspector, "apply_image_prompt_button")
    assert not hasattr(inspector, "discard_image_prompt_button")
