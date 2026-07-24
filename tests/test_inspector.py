"""Focused offscreen tests for the tabbed authoring inspector."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PIL import Image
from PySide6.QtWidgets import QApplication, QFrame

from hypergen.application.document_controller import DocumentController
from hypergen.domain.models import (
    Card,
    ImageGenerationInputs,
    ImageGenerationMetadata,
    ImageOrigin,
    ImageRevision,
    Stack,
)
from hypergen.ui.inspector import Inspector


@pytest.fixture(scope="module")
def application() -> QApplication:
    return QApplication.instance() or QApplication([])


def generated_revision(image_path: str) -> ImageRevision:
    metadata = ImageGenerationMetadata(
        inputs=ImageGenerationInputs(
            scene_description="The surface of the sun",
            global_style="Scientific storybook",
        ),
        render_prompt="The surface of the sun\n\nScientific storybook",
        model_identifier="flux2-klein-4b",
        mflux_version="0.18.0",
        seed=42,
        width=1024,
        height=768,
        step_count=4,
        generated_at=datetime.now(UTC),
        duration_seconds=1.0,
    )
    return ImageRevision(
        image_path=image_path,
        origin=ImageOrigin.GENERATED,
        generation_metadata=metadata,
        created_at=metadata.generated_at,
    )


def test_card_tab_hides_revision_summary_and_collapses_details_and_style(
    application: QApplication,
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "sun.png"
    Image.new("RGB", (1024, 768), "orange").save(image_path)
    revision = generated_revision("assets/cards/sun/background.png")
    card = Card(
        name="Sun",
        scene_description="The surface of the sun",
        image_revisions=(revision,),
        active_revision_id=revision.id,
    )
    controller = DocumentController(
        Stack(name="Demo", global_style="Scientific storybook", cards=(card,))
    )
    inspector = Inspector(
        controller,
        image_path_resolver=lambda _relative: image_path,
    )
    inspector.render(controller.document, card.id)

    assert inspector.inspector_tabs.count() == 2
    assert not hasattr(inspector, "background_value")
    assert not hasattr(inspector, "revision_metadata")
    assert inspector.revision_details.isHidden()
    inspector.revision_details_button.click()
    assert not inspector.revision_details.isHidden()
    details = json.loads(inspector.revision_details.text())
    assert details["render_prompt"] == (
        "The surface of the sun\n\nScientific storybook"
    )
    thumbnail = inspector.revision_thumbnail.pixmap()
    assert thumbnail is not None and not thumbnail.isNull()
    assert inspector.style_details.isHidden()
    inspector.style_details_button.click()
    assert not inspector.style_details.isHidden()
    inspector.close()


def test_section_headings_are_bold(
    application: QApplication,
) -> None:
    card = Card(name="Card")
    controller = DocumentController(Stack(name="Demo", cards=(card,)))
    inspector = Inspector(controller)
    inspector.render(controller.document, card.id)

    for heading in (
        inspector.scene_heading,
        inspector.background_heading,
        inspector.intent_heading,
        inspector.hotspots_heading,
    ):
        assert heading.font().bold()
    assert inspector.style_details_button.font().bold()
    inspector.close()


def test_interactivity_tab_uses_compact_controls_and_on_demand_help(
    application: QApplication,
) -> None:
    card = Card(name="Card", interaction_description="Tap the fox")
    controller = DocumentController(Stack(name="Demo", cards=(card,)))
    inspector = Inspector(controller)
    inspector.render(controller.document, card.id)

    assert inspector.inspector_tabs.tabText(1) == "Interactivity (0)"
    assert inspector.interactions_edit.toPlainText() == "Tap the fox"
    assert inspector.summarize_hotspots_button.text() == "Summarize Hotspots"
    for button, accessible_name in (
        (inspector.move_hotspot_up_button, "Move hotspot up"),
        (inspector.move_hotspot_down_button, "Move hotspot down"),
        (inspector.delete_hotspot_button, "Delete hotspot"),
    ):
        assert not button.icon().isNull()
        assert button.accessibleName() == accessible_name
        assert not button.text()
    assert "Click to add vertices" in inspector.hotspot_help_button.toolTip()
    inspector.close()


def test_summarize_hotspots_follows_intent_text(
    application: QApplication,
) -> None:
    card = Card(name="Card", interaction_description="Tap the fox")
    controller = DocumentController(Stack(name="Demo", cards=(card,)))
    inspector = Inspector(controller)
    inspector.render(controller.document, card.id)
    page = inspector.inspector_tabs.widget(1)
    layout = page.layout()
    assert layout is not None
    intent_actions_index = next(
        index
        for index in range(layout.count())
        if layout.itemAt(index).layout() is inspector.intent_actions
    )

    assert layout.indexOf(inspector.interactions_edit) < intent_actions_index
    inspector.close()


def test_hotspot_controls_follow_selection_edit_generation_hierarchy(
    application: QApplication,
) -> None:
    card = Card(name="Card")
    controller = DocumentController(Stack(name="Demo", cards=(card,)))
    inspector = Inspector(controller)
    inspector.render(controller.document, card.id)
    page = inspector.inspector_tabs.widget(1)
    layout = page.layout()
    assert layout is not None
    order_actions_index = next(
        index
        for index in range(layout.count())
        if layout.itemAt(index).layout() is inspector.hotspot_order_actions
    )
    form_index = next(
        index
        for index in range(layout.count())
        if layout.itemAt(index).layout() is inspector.hotspot_form
    )

    assert layout.indexOf(inspector.hotspot_list) < order_actions_index < form_index
    assert form_index < layout.indexOf(inspector.generate_hotspots_button)
    assert layout.indexOf(inspector.generate_hotspots_button) < layout.indexOf(
        inspector.hotspot_candidate_widget
    )
    assert layout.indexOf(inspector.hotspot_candidate_widget) < layout.indexOf(
        inspector.add_hotspot_button
    )
    inspector.close()


def test_card_tab_uses_revision_edit_generation_hierarchy(
    application: QApplication,
) -> None:
    card = Card(name="Card")
    controller = DocumentController(Stack(name="Demo", cards=(card,)))
    inspector = Inspector(controller)
    inspector.render(controller.document, card.id)
    inspector.resize(inspector.minimumWidth(), 700)
    inspector.show()
    application.processEvents()

    for button in (
        inspector.generate_background_button,
        inspector.import_background_button,
    ):
        assert button.width() >= button.minimumSizeHint().width()
    assert inspector.generate_background_button.text() == "Generate Background"
    assert inspector.import_background_button.text() == "Import Background..."
    assert not inspector.delete_revision_button.icon().isNull()
    assert inspector.delete_revision_button.accessibleName() == "Delete revision"
    assert not inspector.delete_revision_button.text()

    layout = inspector.card_layout
    revision_selector_index = next(
        index
        for index in range(layout.count())
        if layout.itemAt(index).layout() is inspector.revision_selector
    )
    assert revision_selector_index < layout.indexOf(
        inspector.generate_background_button
    )
    assert layout.indexOf(inspector.generate_background_button) < layout.indexOf(
        inspector.style_details_button
    )
    assert layout.indexOf(inspector.style_details_button) < layout.indexOf(
        inspector.draft_widget
    )
    assert layout.indexOf(inspector.draft_widget) < layout.indexOf(
        inspector.import_background_button
    )
    inspector.close()


def test_scene_actions_follow_text_and_start_card_is_last_control(
    application: QApplication,
) -> None:
    card = Card(name="Card", scene_description="A scene")
    controller = DocumentController(Stack(name="Demo", cards=(card,)))
    inspector = Inspector(controller)
    inspector.render(controller.document, card.id)
    layout = inspector.card_layout
    scene_actions_index = next(
        index
        for index in range(layout.count())
        if layout.itemAt(index).layout() is inspector.scene_actions
    )

    assert layout.indexOf(inspector.scene_edit) < scene_actions_index
    assert [
        inspector.enrich_scene_button.text(),
        inspector.describe_image_button.text(),
    ] == ["Enrich", "Describe Image"]
    assert layout.indexOf(inspector.start_card_check) == layout.count() - 2
    inspector.close()


def test_workflow_status_messages_are_highlighted_and_dismissible(
    application: QApplication,
) -> None:
    card = Card(name="Card")
    controller = DocumentController(Stack(name="Demo", cards=(card,)))
    inspector = Inspector(controller)
    inspector.render(controller.document, card.id)

    statuses = (
        (
            inspector.set_background_status,
            inspector.background_status_message,
            inspector.background_status,
        ),
        (
            inspector.set_scene_enrichment_status,
            inspector.scene_enrichment_status_message,
            inspector.scene_enrichment_status,
        ),
        (
            inspector.set_intent_status,
            inspector.intent_status_message,
            inspector.intent_status,
        ),
        (
            inspector.set_hotspot_generation_status,
            inspector.hotspot_generation_status_message,
            inspector.hotspot_generation_status,
        ),
    )
    for set_status, message_widget, label in statuses:
        set_status("Action completed", detail="Result detail")
        assert not message_widget.isHidden()
        assert message_widget.frameShape() == QFrame.Shape.StyledPanel
        assert label.text() == "Action completed"
        assert label.toolTip() == "Result detail"
        assert not message_widget.dismiss_button.icon().isNull()
        message_widget.dismiss_button.click()
        assert message_widget.isHidden()
        assert not label.text()

    inspector.close()
