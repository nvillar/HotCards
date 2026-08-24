"""Focused offscreen tests for revision-local authoring controls."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QLabel,
    QRadioButton,
    QStyle,
)

from hypergen.application.commands import DeleteCardCommand, RenameCardCommand
from hypergen.application.document_controller import DocumentController
from hypergen.domain.models import (
    Card,
    CardRevision,
    HotspotSet,
    ImagePrompt,
    Interaction,
    NavigateAction,
    ResolvedCardReference,
    Stack,
    UnresolvedCardReference,
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

    assert inspector.inspector_tabs.count() == 2
    assert inspector.inspector_tabs.tabText(0) == "Background"
    assert inspector.inspector_tabs.tabText(1) == "Hotspots"
    root_layout = inspector.layout()
    assert root_layout is not None
    assert root_layout.contentsMargins().top() == 16
    assert root_layout.contentsMargins().bottom() == 16
    assert root_layout.contentsMargins().left() == inspector.style().pixelMetric(
        QStyle.PixelMetric.PM_LayoutLeftMargin
    )
    assert inspector.description_edit.placeholderText() == "Description"
    assert inspector.description_edit.minimumHeight() == round(
        (
            inspector.description_edit.fontMetrics().lineSpacing() * 10
            + 20
        )
        * 1.25
    )
    assert inspector.description_edit.maximumHeight() > (
        inspector.description_edit.minimumHeight()
    )
    assert inspector.enrich_button.text() == "Prepare Image Prompt"
    assert inspector.generate_background_button.text() == "Generate Image"
    assert inspector.description_toggle.isHidden()
    assert isinstance(inspector.description_button, QRadioButton)
    assert isinstance(inspector.image_prompt_button, QRadioButton)
    assert not isinstance(inspector.reference_panel, QFrame)
    assert inspector.reference_panel.layout().contentsMargins().isNull()
    content_layout = inspector.reference_panel.parentWidget().layout()
    assert content_layout is not None
    assert content_layout.stretch(
        content_layout.indexOf(inspector.description_edit)
    ) == 1
    assert content_layout.indexOf(inspector.description_edit) < (
        content_layout.indexOf(inspector.description_toggle)
    )
    assert content_layout.indexOf(inspector.description_toggle) < (
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
    assert not hasattr(inspector, "style_combo")
    assert not hasattr(inspector, "clear_background_button")
    assert inspector.hotspot_target_label.text() == "Hotspot Target"
    assert inspector.hotspot_target_label.font().pointSizeF() == (
        inspector.description_label.font().pointSizeF()
    )
    hotspot_layout = inspector.hotspot_list.parentWidget().layout()
    assert hotspot_layout is not None
    assert hotspot_layout.indexOf(inspector.hotspot_target_label) < (
        hotspot_layout.indexOf(inspector.hotspot_destination_combo)
    )
    assert hotspot_layout.stretch(
        hotspot_layout.indexOf(inspector.hotspot_list)
    ) == 1
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
    assert hotspot_controls.indexOf(
        inspector.delete_hotspot_button
    ) < hotspot_controls.indexOf(inspector.add_hotspot_button)

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
    controller = DocumentController(
        Stack(name="Demo", cards=(source, portrait))
    )
    inspector = Inspector(controller)
    inspector.render(controller.document, source.id)
    applied: list[tuple[str, object]] = []
    inspector.change_applied.connect(
        lambda message, token: applied.append((message, token))
    )

    assert inspector._combo_index_for_data(
        inspector.reference_combo,
        source.id,
    ) == -1
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


def test_deleted_reference_is_shown_as_unresolved(
    application: QApplication,
) -> None:
    destination = Card(name="Former portrait")
    revision = CardRevision(
        reference=ResolvedCardReference(target_card_id=destination.id)
    )
    source = Card(name="Source", revisions=(revision,))
    controller = DocumentController(
        Stack(name="Demo", cards=(source, destination))
    )
    inspector = Inspector(controller)
    inspector.render(controller.document, source.id)

    changed = controller.execute(DeleteCardCommand(card_id=destination.id))
    inspector.render(changed, source.id)

    assert inspector.reference_combo.currentText() == (
        "Missing: Former portrait"
    )
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
    assert inspector.description_edit.toPlainText() == (
        "A courtyard"
    )
    assert "Using: Description" in inspector.enrich_button.toolTip()
    assert (
        "Using: Image Prompt"
        in inspector.generate_background_button.toolTip()
    )
    assert inspector.enrich_button.text() == "Image Prompt Current ✓"
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
    controller = DocumentController(
        Stack(name="Demo", cards=(first, second))
    )
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
    assert inspector.enrich_button.text() == "Image Prompt Current ✓"
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
    assert inspector.enrich_button.text() == "Image Prompt Current ✓"
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
    controller = DocumentController(
        Stack(name="Demo", cards=(source, reference))
    )
    inspector = Inspector(controller)
    inspector.render(controller.document, source.id)

    assert inspector.reference_label.text() == "Reference"
    assert "Using: Description + Reference" in inspector.enrich_button.toolTip()
    assert (
        "Using: Image Prompt + Reference"
        in inspector.generate_background_button.toolTip()
    )


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
    assert interaction.label == "Unresolved"
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
