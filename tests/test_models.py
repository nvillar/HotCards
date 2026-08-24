"""Tests for serialized stack-domain boundaries."""

import json
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from hypergen.domain.models import (
    CanvasSize,
    Card,
    CardRevision,
    GeneratedBackground,
    HotspotSet,
    ImageGenerationInputs,
    ImageGenerationMetadata,
    ImagePrompt,
    ImageReferenceSnapshot,
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
            description="A moonlit courtyard",
            image_prompt="A moonlit courtyard",
        ),
        render_prompt="A moonlit courtyard",
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

    assert stack.schema_version == 6
    assert (stack.canvas.width, stack.canvas.height) == (1024, 768)
    assert stack.run_overlay_mode is RunOverlayMode.HIDDEN
    assert stack.cards == ()


def test_revision_uses_current_image_prompt_when_available() -> None:
    reference = ImageReferenceSnapshot(
        card_id=uuid4(),
        revision_id=uuid4(),
        background_id=uuid4(),
    )
    image_prompt = ImagePrompt(
        text="A richer courtyard",
        source_description="A courtyard",
        reference=reference,
        model_identifier="qwen3.5:9b-mlx",
        prompt_version="image-prompt-preparation-v5",
    )
    revision = CardRevision(
        description="A courtyard",
        image_prompt=image_prompt,
    )

    inputs = ImageGenerationInputs(
        description=revision.description,
        image_prompt=image_prompt.text,
        reference=reference,
    )
    assert inputs.effective_description == "A richer courtyard"
    assert image_prompt.is_current(
        source_description="A courtyard",
        reference=reference,
        model_identifier="qwen3.5:9b-mlx",
        prompt_version="image-prompt-preparation-v5",
    )
    assert not image_prompt.is_current(
        source_description="A courtyard",
        reference=reference,
        model_identifier="llama3.2:latest",
        prompt_version="image-prompt-preparation-v5",
    )
    assert not image_prompt.is_current(
        source_description="A changed courtyard",
        reference=reference,
        prompt_version="image-prompt-preparation-v5",
    )
    assert not image_prompt.is_current(
        source_description="A courtyard",
        reference=reference,
        model_identifier="qwen3.5:9b-mlx",
        prompt_version="image-prompt-preparation-v0",
    )
    assert not image_prompt.is_current(
        source_description="A courtyard",
        reference=None,
        model_identifier="qwen3.5:9b-mlx",
        prompt_version="image-prompt-preparation-v5",
    )
    assert not ImagePrompt(
        text="A legacy prompt",
        source_description="A courtyard",
    ).is_current(
        source_description="A courtyard",
        model_identifier="qwen3.5:9b-mlx",
        prompt_version="image-prompt-preparation-v5",
    )


def test_stack_serializes_one_optional_reference() -> None:
    destination = Card(name="Portrait")
    revision = CardRevision(
        reference=ResolvedCardReference(target_card_id=destination.id)
    )
    source = Card(name="Source", revisions=(revision,))
    stack = Stack(name="Castle", cards=(source, destination))

    values = stack.model_dump(mode="json")

    serialized_revision = values["cards"][0]["revisions"][0]
    assert serialized_revision["reference"]["target_card_id"] == str(destination.id)
    assert "subject" not in serialized_revision
    assert "style" not in serialized_revision
    assert "setting" not in serialized_revision


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
    never_applied = CardRevision()
    applied_empty = CardRevision(
        hotspot_set=HotspotSet(),
    )

    assert never_applied.hotspot_set is None
    assert applied_empty.hotspot_set is not None
    assert applied_empty.hotspot_set.interactions == ()


def test_stack_rejects_self_references_and_accepts_one_reference() -> None:
    source = Card(name="Source")
    with pytest.raises(ValidationError, match="own card"):
        Stack(
            name="Castle",
            cards=(
                source.model_copy(
                    update={
                        "revisions": (
                            source.active_revision.model_copy(
                                update={
                                    "reference": ResolvedCardReference(
                                        target_card_id=source.id
                                    )
                                }
                            ),
                        )
                    }
                ),
            ),
        )
    reference = Card(name="Reference")
    revision = CardRevision(
        reference=ResolvedCardReference(target_card_id=reference.id),
    )
    stack = Stack(
        name="Castle",
        cards=(Card(name="Source", revisions=(revision,)), reference),
    )

    assert stack.cards[0].active_revision.reference == (
        ResolvedCardReference(target_card_id=reference.id)
    )


def test_hotspot_labels_are_derived_from_destinations() -> None:
    destination = Card(name="Castle Gate")
    resolved = Interaction(
        label="Author-entered value",
        action=NavigateAction(
            target=ResolvedCardReference(target_card_id=destination.id)
        ),
    )
    unresolved = Interaction(
        label="Former label",
        action=NavigateAction(
            target=UnresolvedCardReference(target_name="Former room")
        ),
    )
    source = Card(
        name="Source",
        revisions=(
            CardRevision(
                hotspot_set=HotspotSet(
                    interactions=(resolved, unresolved)
                )
            ),
        ),
    )

    stack = Stack(name="Castle", cards=(source, destination))

    interactions = stack.cards[0].active_revision.hotspot_set
    assert interactions is not None
    assert [item.label for item in interactions.interactions] == [
        "Castle Gate",
        "Unresolved",
    ]


def test_generation_inputs_capture_exact_reference_source_state() -> None:
    snapshot = ImageReferenceSnapshot(
        card_id=uuid4(),
        revision_id=uuid4(),
        background_id=uuid4(),
    )
    inputs = ImageGenerationInputs(
        description="Portrait at dusk",
        image_prompt="Portrait at dusk",
        reference=snapshot,
    )

    assert inputs.reference == snapshot


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
    asset_id = uuid4()
    revision = CardRevision(
        background=GeneratedBackground(
            id=asset_id,
            image_path=f"assets/cards/card/image-{asset_id}.png",
            generation_metadata=image_metadata(),
            created_at=datetime.now(UTC),
        ),
        hotspot_set=HotspotSet(interactions=(interaction,)),
    )
    card = Card(
        name="Courtyard",
        revisions=(revision,),
        active_revision_id=revision.id,
    )
    stack = Stack(name="Castle", cards=(card,), start_card_id=card.id)

    loaded = Stack.model_validate_json(stack.model_dump_json())

    assert loaded == stack
    assert loaded.cards[0].revisions[0].hotspot_set is not None
    loaded_interaction = (
        loaded.cards[0].revisions[0].hotspot_set.interactions[0]
    )
    assert loaded_interaction.id == interaction.id
    assert loaded_interaction.label == "Unresolved"
    assert loaded_interaction.action == interaction.action
    assert loaded_interaction.polygons == interaction.polygons


def test_generated_background_requires_reproducibility_metadata() -> None:
    with pytest.raises(ValidationError, match="generation_metadata"):
        GeneratedBackground(
            image_path="assets/cards/card/image-revision.png",
            created_at=datetime.now(UTC),
        )


def test_active_revision_and_start_card_must_exist() -> None:
    revision = CardRevision()
    with pytest.raises(ValidationError, match="active_revision_id"):
        Card(
            name="Courtyard",
            revisions=(revision,),
            active_revision_id=uuid4(),
        )

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
    first = CardRevision(hotspot_set=shared)
    second = CardRevision(hotspot_set=shared)

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
    revision = CardRevision(
        hotspot_set=HotspotSet(interactions=(interaction,)),
    )
    card = Card(name="Courtyard", revisions=(revision,), active_revision_id=revision.id)

    with pytest.raises(ValidationError, match="resolved card references"):
        Stack(name="Castle", cards=(card,))


def test_revision_cannot_be_attached_to_multiple_cards() -> None:
    revision = CardRevision(hotspot_set=HotspotSet())

    with pytest.raises(ValidationError, match="revision IDs"):
        Stack(
            name="Castle",
            cards=(
                Card(name="One", revisions=(revision,), active_revision_id=revision.id),
                Card(name="Two", revisions=(revision,), active_revision_id=revision.id),
            ),
        )


def test_blank_hotspot_is_valid_but_has_no_hit_geometry() -> None:
    interaction = Interaction(
        action=NavigateAction(target=UnresolvedCardReference()),
    )

    assert interaction.label == "Unresolved"
    assert interaction.polygons == ()
