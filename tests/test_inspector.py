"""Focused offscreen tests for revision-local authoring controls."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication, QLabel, QMessageBox

from hypergen.application.commands import RenameCardCommand
from hypergen.application.document_controller import DocumentController
from hypergen.domain.models import (
    Card,
    CardRevision,
    GenerationStyle,
    HotspotSet,
    Interaction,
    NavigateAction,
    ResolvedCardReference,
    Stack,
    UnresolvedCardReference,
)
from hypergen.ui.inspector import Inspector
from hypergen.ui.styles_dialog import StylesDialog


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
    style = GenerationStyle(name="Watercolor", prompt="Soft watercolor")
    card = Card(
        name="Courtyard",
        revisions=(
            CardRevision(
                description="A moonlit courtyard",
                style_id=style.id,
                hotspot_set=HotspotSet(interactions=(_interaction(),)),
            ),
        ),
    )
    controller = DocumentController(
        Stack(name="Demo", styles=(style,), cards=(card,))
    )
    inspector = Inspector(controller)
    inspector.render(controller.document, card.id)

    assert inspector.inspector_tabs.count() == 2
    assert inspector.inspector_tabs.tabText(0) == "Background"
    assert inspector.inspector_tabs.tabText(1) == "Hotspots"
    assert inspector.scene_edit.placeholderText() == "Description"
    assert inspector.enrich_scene_button.text() == "Enrich"
    assert inspector.generate_background_button.text() == "Generate Image"
    assert inspector.style_combo.currentText() == "Watercolor"
    assert inspector.import_background_button.text() == "Import Image..."
    assert inspector.clear_background_button.text() == "Clear Image"
    assert inspector.remap_hotspots_button.text() == "Remap"

    visible_copy = " ".join(
        label.text() for label in inspector.findChildren(QLabel)
    )
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


def test_description_and_style_edits_target_active_revision(
    application: QApplication,
) -> None:
    first = GenerationStyle(name="Ink", prompt="Black ink")
    second = GenerationStyle(name="Paint", prompt="Thick paint")
    card = Card(
        name="Card",
        revisions=(CardRevision(description="Old", style_id=first.id),),
    )
    controller = DocumentController(
        Stack(name="Demo", styles=(first, second), cards=(card,))
    )
    inspector = Inspector(controller)
    inspector.render(controller.document, card.id)

    inspector.scene_edit.setPlainText("New description")
    assert inspector.commit_revision_metadata()
    inspector.style_combo.setCurrentIndex(2)

    revision = controller.document.cards[0].active_revision
    assert revision.description == "New description"
    assert revision.style_id == second.id
    assert controller.undo()
    assert controller.document.cards[0].active_revision.style_id == first.id


def test_render_preserves_focused_description_and_hotspot_label_drafts(
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

    inspector.scene_edit.setFocus()
    inspector.scene_edit.setPlainText("Uncommitted description")
    application.processEvents()
    changed = controller.execute(
        RenameCardCommand(card_id=card.id, name="Renamed")
    )
    inspector.render(changed, card.id)
    assert inspector.scene_edit.toPlainText() == "Uncommitted description"

    inspector.hotspot_label_edit.setFocus()
    inspector.hotspot_label_edit.setText("Uncommitted label")
    application.processEvents()
    inspector.render(controller.document, card.id)
    assert inspector.hotspot_label_edit.text() == "Uncommitted label"
    inspector.close()


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
    assert interaction.label == "Hotspot 1"
    assert interaction.polygons == ()
    assert inspector.selected_interaction_id == interaction.id
    assert inspector.hotspot_label_edit.isEnabled()


def test_hotspot_properties_reorder_and_delete_use_commands(
    application: QApplication,
    monkeypatch: pytest.MonkeyPatch,
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
    controller = DocumentController(
        Stack(name="Demo", cards=(source, destination))
    )
    inspector = Inspector(controller)
    inspector.render(controller.document, source.id)

    inspector.select_interaction(second.id)
    inspector.move_hotspot_up_button.click()
    assert [
        item.label
        for item in controller.document.cards[0].active_revision.hotspot_set.interactions  # type: ignore[union-attr]
    ] == ["Second", "First"]

    inspector.hotspot_label_edit.setText("Renamed")
    inspector.hotspot_label_edit.editingFinished.emit()
    destination_index = next(
        index
        for index in range(inspector.hotspot_destination_combo.count())
        if inspector.hotspot_destination_combo.itemData(index) == destination.id
    )
    inspector.hotspot_destination_combo.setCurrentIndex(destination_index)
    application.processEvents()
    changed = controller.document.cards[0].active_revision.hotspot_set
    assert changed is not None
    assert changed.interactions[0].label == "Renamed"
    assert changed.interactions[0].action.target == ResolvedCardReference(
        target_card_id=destination.id
    )

    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *args, **kwargs: QMessageBox.StandardButton.Yes,
    )
    inspector.delete_hotspot_button.click()
    remaining = controller.document.cards[0].active_revision.hotspot_set
    assert remaining is not None
    assert [item.label for item in remaining.interactions] == ["First"]


def test_status_messages_are_dismissible_and_undo_is_actionable(
    application: QApplication,
) -> None:
    inspector = Inspector(DocumentController(Stack(name="Demo")))
    undo_requests: list[bool] = []
    inspector.undo_requested.connect(lambda: undo_requests.append(True))

    inspector.set_background_status("Image imported", detail="Done")
    assert not inspector.background_status_message.isHidden()
    inspector.background_status_message.dismiss_button.click()
    assert not inspector.background_status_message.isVisible()

    inspector.show_undo("Image imported")
    assert inspector.undo_message.action_button.text() == "Undo"
    inspector.undo_message.action_button.click()
    assert undo_requests == [True]


def test_styles_dialog_manages_styles_and_blocks_in_use_deletion(
    application: QApplication,
) -> None:
    style = GenerationStyle(name="Ink", prompt="Black ink")
    card = Card(
        name="Card",
        revisions=(CardRevision(style_id=style.id),),
    )
    controller = DocumentController(
        Stack(name="Demo", styles=(style,), cards=(card,))
    )
    dialog = StylesDialog(controller)

    assert dialog.style_list.count() == 1
    dialog._add_style()
    assert [item.name for item in controller.document.styles] == ["Ink", "Style 1"]
    dialog.name_edit.setText("Gouache")
    dialog.prompt_edit.setPlainText("Opaque paint")
    dialog._commit_selected()
    assert controller.document.styles[-1].name == "Gouache"
    assert controller.document.styles[-1].prompt == "Opaque paint"

    dialog.style_list.setCurrentRow(0)
    dialog._delete_style()
    assert not dialog.error_label.isHidden()
    assert "used by a revision" in dialog.error_label.text()
    assert len(controller.document.styles) == 2
