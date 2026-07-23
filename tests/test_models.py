"""Tests for serialized stack-domain boundaries."""

import json
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from hypergen.domain.models import (
    CanvasSize,
    Card,
    HotspotSet,
    ImageGenerationInputs,
    ImageGenerationMetadata,
    ImageOrigin,
    ImageRevision,
    Interaction,
    NavigateAction,
    Point,
    Polygon,
    ResolvedCardReference,
    RunOverlayMode,
    Stack,
    UnresolvedCardReference,
)


def image_metadata() -> ImageGenerationMetadata:
    return ImageGenerationMetadata(
        inputs=ImageGenerationInputs(
            scene_description="A moonlit courtyard",
            global_style="Ink and watercolor",
        ),
        render_prompt="A moonlit courtyard\n\nInk and watercolor",
        model_identifier="flux2-klein-4b",
        mflux_version="0.18.0",
        dependency_versions={"mlx": "0.31.2"},
        seed=42,
        width=1024,
        height=768,
        step_count=4,
        generated_at=datetime.now(UTC),
        duration_seconds=8.5,
    )


def test_stack_defaults_match_document_contract() -> None:
    stack = Stack(name="Castle")

    assert stack.schema_version == 1
    assert (stack.canvas.width, stack.canvas.height) == (1024, 768)
    assert stack.run_overlay_mode is RunOverlayMode.HIDDEN
    assert stack.cards == ()


def test_stack_serializes_only_the_global_style_key() -> None:
    stack = Stack(name="Castle", global_style="Ink wash")

    values = stack.model_dump(mode="json")

    assert values["global_style"] == "Ink wash"
    assert "art_direction" not in values
    with pytest.raises(ValidationError, match="art_direction"):
        Stack.model_validate({"name": "Castle", "art_direction": "Legacy"})


def test_revalidated_dump_preserves_pydantic_field_selection() -> None:
    stack = Stack(name="Castle")

    assert stack.model_dump(exclude_unset=True) == {"name": "Castle"}


def test_only_navigate_actions_are_accepted() -> None:
    with pytest.raises(ValidationError, match="navigate"):
        Interaction.model_validate_json(
            json.dumps(
                {
                    "label": "Gate",
                    "action": {
                        "type": "external_url",
                        "target": {"type": "unresolved", "target_name": "Garden"},
                    },
                    "polygons": [
                        {
                            "points": [
                                {"x": 0.1, "y": 0.1},
                                {"x": 0.2, "y": 0.1},
                                {"x": 0.2, "y": 0.2},
                            ]
                        }
                    ],
                }
            )
        )


def test_card_references_are_discriminated_and_consistent() -> None:
    card_id = uuid4()
    resolved = NavigateAction(
        target=ResolvedCardReference(target_card_id=card_id),
    )
    unresolved = NavigateAction(
        target=UnresolvedCardReference(target_name="Former garden"),
    )

    assert resolved.target.target_card_id == card_id
    assert unresolved.target.target_name == "Former garden"

    with pytest.raises(ValidationError):
        NavigateAction.model_validate(
            {
                "target": {
                    "type": "resolved",
                    "target_card_id": None,
                    "target_name": "Invalid mixed state",
                }
            }
        )


def test_hotspot_set_distinguishes_never_applied_from_applied_empty() -> None:
    never_applied = ImageRevision(
        image_path="assets/cards/card/image-revision.png",
        origin=ImageOrigin.IMPORTED,
        source_filename="courtyard.png",
        created_at=datetime.now(UTC),
    )
    applied_empty = ImageRevision(
        image_path="assets/cards/card/image-revision.png",
        origin=ImageOrigin.IMPORTED,
        source_filename="courtyard.png",
        hotspot_set=HotspotSet(),
        created_at=datetime.now(UTC),
    )

    assert never_applied.hotspot_set is None
    assert applied_empty.hotspot_set is not None
    assert applied_empty.hotspot_set.interactions == ()


def test_hotspots_are_nested_in_their_image_revision() -> None:
    interaction = Interaction(
        label="Gate",
        action=NavigateAction(
            target=UnresolvedCardReference(target_name="Garden"),
        ),
        polygons=(
            Polygon(
                points=(
                    Point(x=0.1, y=0.1),
                    Point(x=0.3, y=0.1),
                    Point(x=0.2, y=0.4),
                )
            ),
        ),
    )
    revision = ImageRevision(
        image_path="assets/cards/card/image-revision.png",
        origin=ImageOrigin.GENERATED,
        generation_metadata=image_metadata(),
        hotspot_set=HotspotSet(interactions=(interaction,)),
        created_at=datetime.now(UTC),
    )
    card = Card(
        name="Courtyard",
        image_revisions=(revision,),
        active_revision_id=revision.id,
    )
    stack = Stack(name="Castle", cards=(card,), start_card_id=card.id)

    loaded = Stack.model_validate_json(stack.model_dump_json())

    assert loaded == stack
    assert loaded.cards[0].image_revisions[0].hotspot_set is not None
    assert loaded.cards[0].image_revisions[0].hotspot_set.interactions[0] == interaction


def test_generated_revision_requires_reproducibility_metadata() -> None:
    with pytest.raises(ValidationError, match="generation_metadata"):
        ImageRevision(
            image_path="assets/cards/card/image-revision.png",
            origin=ImageOrigin.GENERATED,
            created_at=datetime.now(UTC),
        )


def test_active_revision_and_start_card_must_exist() -> None:
    with pytest.raises(ValidationError, match="active_revision_id"):
        Card(name="Courtyard", active_revision_id=uuid4())

    with pytest.raises(ValidationError, match="start_card_id"):
        Stack(name="Castle", start_card_id=uuid4())


def test_unknown_serialized_fields_are_rejected() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        Stack.model_validate({"name": "Castle", "database_id": 7})


def test_serialized_python_values_are_not_silently_coerced() -> None:
    with pytest.raises(ValidationError):
        CanvasSize.model_validate({"width": "1024", "height": "768"})

    with pytest.raises(ValidationError):
        Stack.model_validate({"name": "Castle", "schema_version": True})


def test_hotspot_sets_are_copied_when_attached_to_revisions() -> None:
    shared = HotspotSet()
    first = ImageRevision(
        image_path="assets/cards/card/image-first.png",
        origin=ImageOrigin.IMPORTED,
        hotspot_set=shared,
        created_at=datetime.now(UTC),
    )
    second = ImageRevision(
        image_path="assets/cards/card/image-second.png",
        origin=ImageOrigin.IMPORTED,
        hotspot_set=shared,
        created_at=datetime.now(UTC),
    )

    assert first.hotspot_set is not shared
    assert second.hotspot_set is not shared
    assert first.hotspot_set is not second.hotspot_set


def test_nonfinite_metadata_is_rejected_before_json_serialization() -> None:
    values = image_metadata().model_dump()
    values["duration_seconds"] = float("inf")
    with pytest.raises(ValidationError, match="finite"):
        ImageGenerationMetadata.model_validate(values)

    values = image_metadata().model_dump()
    values["effective_settings"] = {"guidance": float("nan")}
    with pytest.raises(ValidationError, match="finite"):
        ImageGenerationMetadata.model_validate(values)

    metadata = image_metadata()
    metadata.effective_settings["nested"] = {"bad": float("nan")}
    with pytest.raises(ValidationError, match="finite"):
        metadata.model_dump_json()


def test_duplicate_card_ids_and_dangling_resolved_targets_are_rejected() -> None:
    duplicate_id = uuid4()
    with pytest.raises(ValidationError, match="card IDs"):
        Stack(
            name="Castle",
            cards=(Card(id=duplicate_id, name="One"), Card(id=duplicate_id, name="Two")),
        )

    interaction = Interaction(
        label="Missing room",
        action=NavigateAction(target=ResolvedCardReference(target_card_id=uuid4())),
        polygons=(
            Polygon(
                points=(
                    Point(x=0.1, y=0.1),
                    Point(x=0.3, y=0.1),
                    Point(x=0.2, y=0.4),
                )
            ),
        ),
    )
    revision = ImageRevision(
        image_path="assets/cards/card/image.png",
        origin=ImageOrigin.IMPORTED,
        hotspot_set=HotspotSet(interactions=(interaction,)),
        created_at=datetime.now(UTC),
    )
    card = Card(name="Courtyard", image_revisions=(revision,), active_revision_id=revision.id)

    with pytest.raises(ValidationError, match="resolved card references"):
        Stack(name="Castle", cards=(card,))


def test_image_revision_cannot_be_attached_to_multiple_cards() -> None:
    revision = ImageRevision(
        image_path="assets/cards/shared/image.png",
        origin=ImageOrigin.IMPORTED,
        hotspot_set=HotspotSet(),
        created_at=datetime.now(UTC),
    )

    with pytest.raises(ValidationError, match="image revision IDs"):
        Stack(
            name="Castle",
            cards=(
                Card(name="One", image_revisions=(revision,), active_revision_id=revision.id),
                Card(name="Two", image_revisions=(revision,), active_revision_id=revision.id),
            ),
        )
