"""Focused offscreen tests for the tabbed authoring inspector."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PIL import Image
from PySide6.QtWidgets import QApplication

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
    assert inspector.move_hotspot_up_button.text() == "↑"
    assert inspector.move_hotspot_down_button.text() == "↓"
    assert inspector.delete_hotspot_button.text() == "🗑"
    assert "Click to add vertices" in inspector.hotspot_help_button.toolTip()
    inspector.close()


def test_hotspot_generation_controls_follow_list_before_manual_actions(
    application: QApplication,
) -> None:
    card = Card(name="Card")
    controller = DocumentController(Stack(name="Demo", cards=(card,)))
    inspector = Inspector(controller)
    inspector.render(controller.document, card.id)
    page = inspector.inspector_tabs.widget(1)
    layout = page.layout()
    assert layout is not None
    manual_actions_index = next(
        index
        for index in range(layout.count())
        if layout.itemAt(index).layout() is inspector.hotspot_actions
    )

    assert layout.indexOf(inspector.hotspot_list) < layout.indexOf(
        inspector.generate_hotspots_button
    )
    assert layout.indexOf(
        inspector.generate_hotspots_button
    ) < layout.indexOf(inspector.hotspot_candidate_widget)
    assert layout.indexOf(inspector.hotspot_candidate_widget) < manual_actions_index
    inspector.close()


def test_card_tab_keeps_revision_button_labels_readable(
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
        inspector.delete_revision_button,
    ):
        assert button.width() >= button.minimumSizeHint().width()

    style_row, _column, _row_span, _column_span = (
        inspector.revision_controls.getItemPosition(
            inspector.revision_controls.indexOf(inspector.style_details_button)
        )
    )
    generate_row, _column, _row_span, _column_span = (
        inspector.revision_controls.getItemPosition(
            inspector.revision_controls.indexOf(inspector.generate_background_button)
        )
    )
    import_row, _column, _row_span, import_column_span = (
        inspector.revision_controls.getItemPosition(
            inspector.revision_controls.indexOf(inspector.import_background_button)
        )
    )
    assert style_row == generate_row
    assert import_row > generate_row
    assert import_column_span == 2
    inspector.close()
