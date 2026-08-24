"""Focused offscreen tests for revision-local authoring controls."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication, QLabel, QRadioButton

from hypergen.application.commands import DeleteCardCommand, RenameCardCommand
from hypergen.application.document_controller import DocumentController
from hypergen.domain.models import (
    Card,
    CardRevision,
    EnrichedDescription,
    HotspotSet,
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
    assert inspector.description_edit.placeholderText() == "Description"
    assert inspector.description_edit.minimumHeight() == (
        inspector.description_edit.maximumHeight()
    )
    assert inspector.description_edit.minimumHeight() >= (
        inspector.description_edit.fontMetrics().lineSpacing() * 10
    )
    assert inspector.enrich_scene_button.text() == "Enrich Description"
    assert inspector.generate_background_button.text() == "Generate Image"
    assert inspector.description_toggle.isHidden()
    assert isinstance(inspector.original_description_button, QRadioButton)
    assert isinstance(inspector.enriched_description_button, QRadioButton)
    content_layout = inspector.references_group.parentWidget().layout()
    assert content_layout is not None
    assert content_layout.indexOf(inspector.description_edit) < (
        content_layout.indexOf(inspector.description_toggle)
    )
    assert content_layout.indexOf(inspector.description_toggle) < (
        content_layout.indexOf(inspector.references_group)
    )
    assert content_layout.indexOf(inspector.references_group) < (
        content_layout.indexOf(inspector.enrich_scene_button)
    )
    assert content_layout.indexOf(inspector.enrich_scene_button) < (
        content_layout.indexOf(inspector.generate_background_button)
    )
    assert not hasattr(inspector, "style_combo")
    assert not hasattr(inspector, "clear_background_button")

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


def test_reference_selectors_assign_distinct_cards_with_undo(
    application: QApplication,
) -> None:
    source = Card(name="Source")
    portrait = Card(name="Portrait")
    landscape = Card(name="Landscape")
    controller = DocumentController(
        Stack(name="Demo", cards=(source, portrait, landscape))
    )
    inspector = Inspector(controller)
    inspector.render(controller.document, source.id)
    applied: list[tuple[str, object]] = []
    inspector.change_applied.connect(
        lambda message, token: applied.append((message, token))
    )

    assert inspector._combo_index_for_data(
        inspector.subject_reference_combo,
        source.id,
    ) == -1
    identity_index = inspector._combo_index_for_data(
        inspector.subject_reference_combo,
        portrait.id,
    )
    inspector.subject_reference_combo.setCurrentIndex(identity_index)

    assert controller.document.cards[0].active_revision.subject == (
        ResolvedCardReference(target_card_id=portrait.id)
    )
    assert inspector._combo_index_for_data(
        inspector.style_reference_combo,
        portrait.id,
    ) >= 0
    assert applied[-1][0] == "Subject reference changed"
    assert controller.undo_if_current(applied[-1][1])  # type: ignore[arg-type]
    inspector.render(controller.document, source.id)
    assert controller.document.cards[0].active_revision.subject is None


def test_deleted_reference_is_shown_as_unresolved(
    application: QApplication,
) -> None:
    destination = Card(name="Former portrait")
    revision = CardRevision(
        subject=ResolvedCardReference(target_card_id=destination.id)
    )
    source = Card(name="Source", revisions=(revision,))
    controller = DocumentController(
        Stack(name="Demo", cards=(source, destination))
    )
    inspector = Inspector(controller)
    inspector.render(controller.document, source.id)

    changed = controller.execute(DeleteCardCommand(card_id=destination.id))
    inspector.render(changed, source.id)

    assert inspector.subject_reference_combo.currentText() == (
        "Missing: Former portrait"
    )
    assert controller.document.cards[0].active_revision.subject == (
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


def test_enriched_description_status_and_generation_source(
    application: QApplication,
) -> None:
    revision = CardRevision(
        description="A courtyard",
        enriched_description=EnrichedDescription(
            text="A richly detailed courtyard",
            source_description="A courtyard",
        ),
    )
    card = Card(name="Card", revisions=(revision,))
    controller = DocumentController(Stack(name="Demo", cards=(card,)))
    inspector = Inspector(controller)
    inspector.render(controller.document, card.id)

    assert not inspector.description_toggle.isHidden()
    assert inspector.enriched_description_button.isChecked()
    assert inspector.description_edit.toPlainText() == (
        "A richly detailed courtyard"
    )
    assert "Using: Description" in inspector.enrich_scene_button.toolTip()
    assert (
        "Using: Description + Enriched Description"
        in inspector.generate_background_button.toolTip()
    )
    assert inspector.enrich_scene_button.text() == "Description Enriched ✓"
    assert not inspector.enrich_scene_button.isEnabled()
    assert "Current" in inspector.enrich_scene_button.toolTip()
    inspector.set_scene_enrichment_capabilities(
        can_enrich=True,
        reason="Ready to enrich Description",
        busy=False,
    )

    inspector.original_description_button.click()
    assert inspector.description_edit.toPlainText() == "A courtyard"
    inspector.description_edit.setPlainText("A changed courtyard")
    assert inspector.enrich_scene_button.text() == "Re-enrich Description"
    assert inspector.enrich_scene_button.isEnabled()
    assert "Out of date" in inspector.enrich_scene_button.toolTip()
    assert inspector.commit_revision_metadata()

    current = controller.document.cards[0].active_revision
    assert current.enriched_description is not None
    assert current.enriched_description.text == "A richly detailed courtyard"

    inspector.enriched_description_button.click()
    inspector.description_edit.setPlainText("A manually revised courtyard")
    assert inspector.commit_revision_metadata()

    inspector.description_edit.clear()
    assert inspector.commit_revision_metadata()
    assert controller.document.cards[0].active_revision.enriched_description is None
    assert inspector.description_toggle.isHidden()
    assert inspector.description_edit.toPlainText() == "A changed courtyard"
    assert "Using: Description" in inspector.generate_background_button.toolTip()
    assert controller.undo()
    assert controller.document.cards[0].active_revision.enriched_description is not None


def test_using_labels_name_assigned_reference_roles(
    application: QApplication,
) -> None:
    subject = Card(name="Subject")
    style = Card(name="Style")
    source = Card(
        name="Source",
        revisions=(
            CardRevision(
                description="A courtyard",
                subject=ResolvedCardReference(target_card_id=subject.id),
                style=ResolvedCardReference(target_card_id=style.id),
            ),
        ),
    )
    controller = DocumentController(
        Stack(name="Demo", cards=(source, subject, style))
    )
    inspector = Inspector(controller)
    inspector.render(controller.document, source.id)

    expected = "Using: Description + References (Subject, Style)"
    assert inspector.references_group.title() == "References (2)"
    assert expected in inspector.enrich_scene_button.toolTip()
    assert expected in inspector.generate_background_button.toolTip()


def test_generate_accepts_either_description_source(
    application: QApplication,
) -> None:
    revision = CardRevision(
        enriched_description=EnrichedDescription(
            text="A richly detailed courtyard",
            source_description="",
        ),
    )
    card = Card(name="Card", revisions=(revision,))
    controller = DocumentController(Stack(name="Demo", cards=(card,)))
    inspector = Inspector(controller)
    inspector.render(controller.document, card.id)

    assert inspector.has_render_prompt_input()
    assert not inspector.has_description_input()

    inspector.description_edit.clear()
    assert not inspector.has_render_prompt_input()
    assert inspector.commit_revision_metadata()
    assert controller.document.cards[0].active_revision.enriched_description is None


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
    inspector.set_scene_enrichment_capabilities(
        can_enrich=False,
        reason="Enriching",
        busy=True,
    )

    assert inspector.generate_background_button.text() == "Generating…"
    assert inspector.enrich_scene_button.text() == "Enriching…"


def test_enrichment_has_no_proposal_editor(
    application: QApplication,
) -> None:
    inspector = Inspector(DocumentController(Stack(name="Demo")))

    assert not hasattr(inspector, "enrichment_review")
    assert not hasattr(inspector, "apply_enrichment_button")
    assert not hasattr(inspector, "discard_enrichment_button")
