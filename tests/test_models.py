"""Tests for serialized stack-domain boundaries."""

import json
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from pydantic import TypeAdapter, ValidationError

from hotcards.domain.image_dimensions import (
    AspectRatio,
    GenerateResolution,
)
from hotcards.domain.models import (
    BUILT_IN_STYLES,
    CURRENT_SCHEMA_VERSION,
    HYPERCARD_STYLE_ID,
    AcceptedEdit,
    CanvasSize,
    Card,
    CardRevision,
    DirectGenerateProvenance,
    EditPreserveOptions,
    EditProvenance,
    GeneratedBackground,
    GenerateInputs,
    HotspotConditions,
    HotspotKeyChanges,
    HotspotSet,
    ImageOperationSettings,
    ImageProvenance,
    ImageReferenceSnapshot,
    ImageSourceSnapshot,
    Interaction,
    KeyDefinition,
    LegacyGenerateProvenance,
    NavigateAction,
    Point,
    Polygon,
    RefineProvenance,
    RefineTransformation,
    ResolvedCardReference,
    RunOverlayMode,
    Stack,
    StyleDefinition,
    StyleSnapshot,
    UnresolvedCardReference,
)


def image_settings() -> ImageOperationSettings:
    return ImageOperationSettings(
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


def image_provenance() -> DirectGenerateProvenance:
    return DirectGenerateProvenance(
        inputs=GenerateInputs(
            description="A moonlit courtyard",
        ),
        render_prompt="A moonlit courtyard",
        settings=image_settings(),
    )


def test_stack_defaults_match_document_contract() -> None:
    stack = Stack(name="Castle")

    assert stack.schema_version == CURRENT_SCHEMA_VERSION
    assert stack.aspect_ratio is AspectRatio.LANDSCAPE
    assert (stack.canvas.width, stack.canvas.height) == (1024, 768)
    assert stack.model_dump(mode="json")["aspect_ratio"] == "4:3"
    assert "canvas" not in stack.model_dump(mode="json")
    assert stack.run_overlay_mode is RunOverlayMode.HIDDEN
    assert stack.styles == BUILT_IN_STYLES
    assert stack.new_card_style_id == HYPERCARD_STYLE_ID
    assert stack.keys == ()
    assert stack.cards == ()
    with pytest.raises(ValidationError, match="frozen"):
        stack.aspect_ratio = AspectRatio.SQUARE


def test_built_in_style_ids_remain_stable_across_product_renames() -> None:
    assert tuple(style.id for style in BUILT_IN_STYLES) == (
        UUID("2372dddb-99e0-5e62-b740-405d0f7dab2a"),
        UUID("a23ac4cf-0358-500a-a4fb-1f120f1f9e47"),
        UUID("7baf1057-786a-587e-a52f-c63b513519c6"),
        UUID("72b678e8-49b0-50d8-af4b-735b31def4c0"),
        UUID("5f602dd8-d459-5633-bd1a-f9b1c86fe473"),
        UUID("61a9ca01-5458-5607-9940-8088955640a2"),
        UUID("1e0aeb8e-7112-5ddc-becf-f6053ce31ffb"),
        UUID("a73faa84-f09b-5832-b8a7-9af17fcb7c86"),
        UUID("ea72f569-5ac1-5906-88b8-a0b7171b4d15"),
        UUID("eda5058a-d463-5fd7-82dc-f50e7a571efa"),
    )


def test_stack_style_references_use_stable_ids_and_unique_names() -> None:
    custom = StyleDefinition(name="Custom", prompt_text="  Rendered in ink.  ")
    revision = CardRevision(style_id=custom.id)
    card = Card(name="Card", revisions=(revision,))
    stack = Stack(
        name="Castle",
        styles=(custom,),
        new_card_style_id=custom.id,
        cards=(card,),
    )

    assert stack.styles[0].prompt_text == "Rendered in ink."
    assert stack.style_by_id(custom.id) == custom
    assert stack.style_by_id(None) is None

    with pytest.raises(ValidationError, match="Style names"):
        Stack(
            name="Castle",
            styles=(
                custom,
                StyleDefinition(name=" custom ", prompt_text="Other"),
            ),
            new_card_style_id=None,
        )
    with pytest.raises(ValidationError, match="revision style_id"):
        Stack(
            name="Castle",
            cards=(Card(name="Card", revisions=(CardRevision(style_id=uuid4()),)),),
        )


def test_obsolete_image_prompt_fields_are_rejected() -> None:
    with pytest.raises(ValidationError, match="image_prompt"):
        CardRevision.model_validate(
            {
                "description": "A courtyard",
                "image_prompt": {"text": "Legacy prompt"},
            }
        )
    with pytest.raises(ValidationError, match="image_prompt"):
        GenerateInputs.model_validate(
            {
                "description": "A courtyard",
                "image_prompt": "Legacy prompt",
            }
        )


def test_stack_serializes_ordered_references() -> None:
    destinations = (Card(name="Portrait"), Card(name="Studio"))
    revision = CardRevision(
        references=tuple(
            ResolvedCardReference(target_card_id=destination.id)
            for destination in destinations
        )
    )
    source = Card(name="Source", revisions=(revision,))
    stack = Stack(name="Castle", cards=(source, *destinations))

    values = stack.model_dump(mode="json")

    serialized_revision = values["cards"][0]["revisions"][0]
    assert tuple(
        reference["target_card_id"]
        for reference in serialized_revision["references"]
    ) == tuple(str(destination.id) for destination in destinations)
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
    assert UnresolvedCardReference(target_name="   ").target_name is None

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


def test_stack_rejects_self_and_duplicate_references() -> None:
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
                                    "references": (
                                        ResolvedCardReference(
                                            target_card_id=source.id
                                        ),
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
        references=(ResolvedCardReference(target_card_id=reference.id),),
    )
    stack = Stack(
        name="Castle",
        cards=(Card(name="Source", revisions=(revision,)), reference),
    )

    assert stack.cards[0].active_revision.references == (
        ResolvedCardReference(target_card_id=reference.id),
    )
    duplicate = CardRevision(
        references=(
            ResolvedCardReference(target_card_id=reference.id),
            ResolvedCardReference(target_card_id=reference.id),
        )
    )
    with pytest.raises(ValidationError, match="same card twice"):
        Stack(
            name="Castle",
            cards=(Card(name="Source", revisions=(duplicate,)), reference),
        )
    with pytest.raises(ValidationError, match="at most 2"):
        CardRevision(
            references=(
                UnresolvedCardReference(target_name="One"),
                UnresolvedCardReference(target_name="Two"),
                UnresolvedCardReference(target_name="Three"),
            )
        )


def test_hotspot_labels_are_derived_from_actions() -> None:
    destination = Card(name="Castle Gate")
    red_key = KeyDefinition(name="Red key")
    door_open = KeyDefinition(name="Door open")
    resolved = Interaction(
        action=NavigateAction(target=ResolvedCardReference(target_card_id=destination.id)),
        key_changes=HotspotKeyChanges(
            remove=(red_key.id,),
            grant=(door_open.id,),
        ),
    )
    unresolved = Interaction(
        action=NavigateAction(target=UnresolvedCardReference(target_name="Former room")),
    )
    actionless = Interaction()
    source = Card(
        name="Source",
        revisions=(
            CardRevision(
                hotspot_set=HotspotSet(interactions=(resolved, unresolved, actionless))
            ),
        ),
    )

    stack = Stack(
        name="Castle",
        keys=(red_key, door_open),
        cards=(source, destination),
    )

    interactions = stack.cards[0].active_revision.hotspot_set
    assert interactions is not None
    assert [item.label for item in interactions.interactions] == [
        "Lose Red key · Gain Door open → Castle Gate",
        "Former room",
        "New Hotspot",
    ]


def test_hotspot_key_contract_is_closed_and_references_stack_keys() -> None:
    red_key = KeyDefinition(name=" Red key ")
    interaction = Interaction(
        conditions=HotspotConditions(requires=(red_key.id,)),
        key_changes=HotspotKeyChanges(remove=(red_key.id,)),
    )
    stack = Stack(
        name="Castle",
        keys=(red_key,),
        cards=(
            Card(
                name="Card",
                revisions=(
                    CardRevision(hotspot_set=HotspotSet(interactions=(interaction,))),
                ),
            ),
        ),
    )

    assert stack.keys[0].name == "Red key"
    assert stack.key_by_id(red_key.id) == red_key
    with pytest.raises(ValidationError, match="both have and lack"):
        HotspotConditions(requires=(red_key.id,), forbids=(red_key.id,))
    with pytest.raises(ValidationError, match="both gained and lost"):
        HotspotKeyChanges(remove=(red_key.id,), grant=(red_key.id,))
    with pytest.raises(ValidationError, match="hotspot key references"):
        Stack(
            name="Castle",
            cards=(
                Card(
                    name="Card",
                    revisions=(
                        CardRevision(
                            hotspot_set=HotspotSet(
                                interactions=(
                                    Interaction(
                                        conditions=HotspotConditions(
                                            requires=(uuid4(),)
                                        )
                                    ),
                                )
                            )
                        ),
                    ),
                ),
            ),
        )
    with pytest.raises(ValidationError, match="Key names"):
        Stack(
            name="Castle",
            keys=(red_key, KeyDefinition(name="red KEY")),
        )


def test_generation_inputs_capture_exact_reference_source_state() -> None:
    snapshot = ImageReferenceSnapshot(
        card_id=uuid4(),
        revision_id=uuid4(),
        background_id=uuid4(),
    )
    inputs = GenerateInputs(
        description="Portrait at dusk",
        references=(snapshot,),
    )

    assert inputs.references == (snapshot,)


def test_generation_inputs_capture_exact_style_snapshot() -> None:
    style = StyleSnapshot(
        style_id=HYPERCARD_STYLE_ID,
        name="HyperCard",
        prompt_text="Pure black and white pixels.",
    )
    inputs = GenerateInputs(
        description="Portrait at dusk",
        style=style,
    )

    assert inputs.style == style


def test_generate_resolution_is_revision_local_and_strict() -> None:
    revision = CardRevision()

    assert revision.generate_resolution is GenerateResolution.RESOLUTION_512
    assert revision.model_dump(mode="json")["generate_resolution"] == 512
    with pytest.raises(ValidationError, match="generate_resolution"):
        CardRevision.model_validate(
            {"generate_resolution": 300},
        )


def test_image_provenance_union_is_discriminated_strict_and_round_trips() -> None:
    source = ImageSourceSnapshot(
        card_id=uuid4(),
        revision_id=uuid4(),
        background_id=uuid4(),
    )
    preserve = EditPreserveOptions(
        subject_identity=True,
        composition=True,
    )
    accepted_edit = AcceptedEdit(
        instruction="Open the gate.",
        preserve=preserve,
        expanded_prompt="Open the gate. Preserve the subject and composition.",
    )
    provenances: tuple[ImageProvenance, ...] = (
        image_provenance(),
        LegacyGenerateProvenance(
            render_prompt="Historical exact prompt",
            settings=image_settings(),
        ),
        RefineProvenance(
            source=source,
            description="A moonlit courtyard",
            edit_lineage=(accepted_edit,),
            render_prompt="A moonlit courtyard. Open the gate.",
            resolution=GenerateResolution.RESOLUTION_768,
            transformation=RefineTransformation.BALANCED,
            strength=0.50,
            settings=image_settings(),
        ),
        EditProvenance(
            source=source,
            instruction=accepted_edit.instruction,
            preserve=accepted_edit.preserve,
            expanded_prompt=accepted_edit.expanded_prompt,
            resolution=GenerateResolution.RESOLUTION_1024,
            edit_lineage=(accepted_edit,),
            settings=image_settings(),
        ),
    )
    adapter = TypeAdapter(ImageProvenance)

    for provenance in provenances:
        assert adapter.validate_json(adapter.dump_json(provenance)) == provenance

    with pytest.raises(ValidationError, match="union_tag_invalid"):
        adapter.validate_python(
            {
                "operation": "unknown",
                "render_prompt": "Prompt",
                "settings": image_settings(),
            }
        )
    with pytest.raises(ValidationError, match="extra_forbidden"):
        adapter.validate_python(
            {
                **image_provenance().model_dump(mode="python"),
                "instruction": "Not a Generate field",
            }
        )


def test_direct_generate_provenance_requires_authored_inputs_and_aligned_output() -> None:
    with pytest.raises(ValidationError, match="nonempty Description"):
        GenerateInputs(description=" \n ")

    values = image_settings().model_dump()
    values["width"] = 1025
    with pytest.raises(ValidationError, match="multiples of 16"):
        ImageOperationSettings.model_validate(values)


def test_refine_and_edit_provenance_enforce_operation_invariants() -> None:
    source = ImageSourceSnapshot(
        card_id=uuid4(),
        revision_id=uuid4(),
        background_id=uuid4(),
    )
    preserve = EditPreserveOptions(subject_identity=True)
    accepted = AcceptedEdit(
        instruction="Open the gate.",
        preserve=preserve,
        expanded_prompt="Open the gate. Preserve subject identity.",
    )

    with pytest.raises(ValidationError, match="balanced Refine strength"):
        RefineProvenance(
            source=source,
            description="A courtyard",
            render_prompt="A courtyard",
            resolution=GenerateResolution.RESOLUTION_768,
            transformation=RefineTransformation.BALANCED,
            strength=0.25,
            settings=image_settings(),
        )
    with pytest.raises(ValidationError, match="lineage"):
        EditProvenance(
            source=source,
            instruction="Close the gate.",
            preserve=preserve,
            expanded_prompt="Close the gate. Preserve subject identity.",
            resolution=GenerateResolution.RESOLUTION_768,
            edit_lineage=(accepted,),
            settings=image_settings(),
        )


def test_derived_provenance_requires_a_source_revision_in_the_declared_card() -> None:
    source = CardRevision(background=GeneratedBackground(
        image_path="assets/cards/source/image-source.png",
        provenance=image_provenance(),
        created_at=datetime.now(UTC),
    ))
    source_card = Card(name="Source", revisions=(source,))
    derived = CardRevision(
        background=GeneratedBackground(
            image_path="assets/cards/derived/image-derived.png",
            provenance=RefineProvenance(
                source=ImageSourceSnapshot(
                    card_id=source_card.id,
                    revision_id=source.id,
                    background_id=source.background.id,
                ),
                description="A refined courtyard",
                render_prompt="A refined courtyard",
                resolution=GenerateResolution.RESOLUTION_768,
                transformation=RefineTransformation.PRESERVE,
                strength=0.75,
                settings=image_settings(),
            ),
            created_at=datetime.now(UTC),
        )
    )
    dependent_card = Card(name="Dependent", revisions=(derived,))

    Stack(name="Valid", cards=(source_card, dependent_card))
    wrong_source = derived.provenance
    assert isinstance(wrong_source, RefineProvenance)
    invalid_provenance = wrong_source.model_copy(
        update={
            "source": wrong_source.source.model_copy(
                update={"card_id": dependent_card.id}
            )
        }
    )
    invalid_derived = derived.model_copy(
        update={
            "background": derived.background.model_copy(
                update={"provenance": invalid_provenance}
            )
        }
    )
    with pytest.raises(ValidationError, match="source card"):
        Stack(
            name="Invalid",
            cards=(
                source_card,
                dependent_card.model_copy(update={"revisions": (invalid_derived,)}),
            ),
        )
    mismatched_source = wrong_source.model_copy(
        update={
            "source": wrong_source.source.model_copy(
                update={"background_id": uuid4()}
            )
        }
    )
    mismatched_derived = derived.model_copy(
        update={
            "background": derived.background.model_copy(
                update={"provenance": mismatched_source}
            )
        }
    )
    with pytest.raises(ValidationError, match="source background"):
        Stack(
            name="Invalid",
            cards=(
                source_card,
                dependent_card.model_copy(update={"revisions": (mismatched_derived,)}),
            ),
        )


def test_derived_image_source_lineage_rejects_cycles() -> None:
    first_card_id, second_card_id = uuid4(), uuid4()
    first_revision_id, second_revision_id = uuid4(), uuid4()
    first_background_id, second_background_id = uuid4(), uuid4()
    first_revision = CardRevision(
        id=first_revision_id,
        background=GeneratedBackground(
            id=first_background_id,
            image_path="assets/cards/first.png",
            provenance=RefineProvenance(
                source=ImageSourceSnapshot(
                    card_id=second_card_id,
                    revision_id=second_revision_id,
                    background_id=second_background_id,
                ),
                description="First",
                render_prompt="First",
                resolution=GenerateResolution.RESOLUTION_512,
                transformation=RefineTransformation.BALANCED,
                strength=0.50,
                settings=image_settings(),
            ),
            created_at=datetime.now(UTC),
        ),
    )
    second_revision = CardRevision(
        id=second_revision_id,
        background=GeneratedBackground(
            id=second_background_id,
            image_path="assets/cards/second.png",
            provenance=RefineProvenance(
                source=ImageSourceSnapshot(
                    card_id=first_card_id,
                    revision_id=first_revision_id,
                    background_id=first_background_id,
                ),
                description="Second",
                render_prompt="Second",
                resolution=GenerateResolution.RESOLUTION_512,
                transformation=RefineTransformation.BALANCED,
                strength=0.50,
                settings=image_settings(),
            ),
            created_at=datetime.now(UTC),
        ),
    )

    with pytest.raises(ValidationError, match="lineage cannot contain cycles"):
        Stack(
            name="Invalid",
            cards=(
                Card(id=first_card_id, name="First", revisions=(first_revision,)),
                Card(id=second_card_id, name="Second", revisions=(second_revision,)),
            ),
        )


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
            provenance=image_provenance(),
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
    loaded_interaction = loaded.cards[0].revisions[0].hotspot_set.interactions[0]
    assert loaded_interaction.id == interaction.id
    assert loaded_interaction.label == "Garden"
    assert loaded_interaction.action == interaction.action
    assert loaded_interaction.polygons == interaction.polygons


def test_generated_background_requires_reproducibility_metadata() -> None:
    with pytest.raises(ValidationError, match="provenance"):
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
    values = image_settings().model_dump()
    values["duration_seconds"] = float("inf")
    with pytest.raises(ValidationError, match="finite"):
        ImageOperationSettings.model_validate(values)

    values = image_settings().model_dump()
    values["guidance"] = float("nan")
    with pytest.raises(ValidationError, match="finite"):
        ImageOperationSettings.model_validate(values)


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
    interaction = Interaction()

    assert interaction.label == "New Hotspot"
    assert interaction.polygons == ()
