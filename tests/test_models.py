"""Tests for serialized stack-domain boundaries."""

import json
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from pydantic import TypeAdapter, ValidationError

from hotcards.domain.image_dimensions import (
    AspectRatio,
    GenerateResolution,
    output_dimensions,
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
    DuplicateProvenance,
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


def image_settings(
    *,
    width: int = 592,
    height: int = 448,
) -> ImageOperationSettings:
    return ImageOperationSettings(
        model_identifier="flux2-klein-4b",
        mflux_version="0.18.0",
        dependency_versions={"mlx": "0.31.2"},
        seed=42,
        width=width,
        height=height,
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


@pytest.mark.parametrize("aspect_ratio", tuple(AspectRatio))
@pytest.mark.parametrize("resolution", tuple(GenerateResolution))
def test_stack_requires_generate_dimensions_for_every_schema_combination(
    aspect_ratio: AspectRatio,
    resolution: GenerateResolution,
) -> None:
    width, height = output_dimensions(resolution, aspect_ratio)
    background = GeneratedBackground(
        image_path="assets/cards/card/image.png",
        provenance=DirectGenerateProvenance(
            inputs=GenerateInputs(
                description="A courtyard",
                resolution=resolution,
            ),
            render_prompt="A courtyard",
            settings=image_settings(width=width, height=height),
        ),
        created_at=datetime.now(UTC),
    )
    revision = CardRevision(background=background)
    card = Card(name="Card", revisions=(revision,))

    Stack(name="Valid", aspect_ratio=aspect_ratio, cards=(card,))

    invalid_settings = background.provenance.settings.model_copy(
        update={"width": width + 16}
    )
    invalid_background = background.model_copy(
        update={
            "provenance": background.provenance.model_copy(
                update={"settings": invalid_settings}
            )
        }
    )
    with pytest.raises(ValidationError, match="direct Generate dimensions"):
        Stack(
            name="Invalid",
            aspect_ratio=aspect_ratio,
            cards=(
                card.model_copy(
                    update={
                        "revisions": (
                            revision.model_copy(
                                update={"background": invalid_background}
                            ),
                        )
                    }
                ),
            ),
        )


def test_legacy_generate_preserves_historic_nonpreset_dimensions() -> None:
    background = GeneratedBackground(
        image_path="assets/cards/card/legacy.png",
        provenance=LegacyGenerateProvenance(
            render_prompt="Historical exact prompt",
            settings=image_settings(width=1001, height=777),
        ),
        created_at=datetime.now(UTC),
    )

    Stack(
        name="Historical",
        aspect_ratio=AspectRatio.LANDSCAPE,
        cards=(Card(name="Card", revisions=(CardRevision(background=background),)),),
    )


def test_image_provenance_union_is_discriminated_strict_and_round_trips() -> None:
    source = ImageSourceSnapshot(
        card_id=uuid4(),
        revision_id=uuid4(),
        background_id=uuid4(),
    )
    preserve = EditPreserveOptions(
        subject_identity=True,
        composition_and_framing=True,
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
            resolution=GenerateResolution.RESOLUTION_512,
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
            settings=image_settings(width=1184, height=880),
        ),
        DuplicateProvenance(
            source=source,
            original_provenance=image_provenance(),
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


def test_duplicate_provenance_is_flattened_and_inherits_original_edit_lineage() -> None:
    source = ImageSourceSnapshot(
        card_id=uuid4(),
        revision_id=uuid4(),
        background_id=uuid4(),
    )
    accepted_edit = AcceptedEdit(
        instruction="Open the gate.",
        preserve=EditPreserveOptions(subject_identity=True),
        expanded_prompt="Open the gate. Preserve subject identity.",
    )
    original = EditProvenance(
        source=source,
        instruction=accepted_edit.instruction,
        preserve=accepted_edit.preserve,
        expanded_prompt=accepted_edit.expanded_prompt,
        resolution=GenerateResolution.RESOLUTION_512,
        edit_lineage=(accepted_edit,),
        settings=image_settings(),
    )
    duplicate = DuplicateProvenance(
        source=source,
        original_provenance=original,
    )

    assert duplicate.settings == original.settings
    with pytest.raises(ValidationError, match="union_tag_invalid"):
        DuplicateProvenance.model_validate(
            {
                "source": source.model_dump(mode="python"),
                "original_provenance": duplicate.model_dump(mode="python"),
            }
        )


def test_derived_operation_inherits_lineage_from_independent_duplicate() -> None:
    original_source = ImageSourceSnapshot(
        card_id=uuid4(),
        revision_id=uuid4(),
        background_id=uuid4(),
    )
    accepted_edit = AcceptedEdit(
        instruction="Open the gate.",
        preserve=EditPreserveOptions(subject_identity=True),
        expanded_prompt="Open the gate. Preserve subject identity.",
    )
    original = EditProvenance(
        source=original_source,
        instruction=accepted_edit.instruction,
        preserve=accepted_edit.preserve,
        expanded_prompt=accepted_edit.expanded_prompt,
        resolution=GenerateResolution.RESOLUTION_512,
        edit_lineage=(accepted_edit,),
        settings=image_settings(),
    )
    duplicate_card_id = uuid4()
    duplicate_revision = CardRevision(
        background=GeneratedBackground(
            image_path="assets/cards/duplicate.png",
            provenance=DuplicateProvenance(
                source=original_source,
                original_provenance=original,
            ),
            created_at=datetime.now(UTC),
        )
    )
    assert duplicate_revision.background is not None
    refined = CardRevision(
        background=GeneratedBackground(
            image_path="assets/cards/refined.png",
            provenance=RefineProvenance(
                source=ImageSourceSnapshot(
                    card_id=duplicate_card_id,
                    revision_id=duplicate_revision.id,
                    background_id=duplicate_revision.background.id,
                ),
                description="Refined duplicate",
                edit_lineage=(accepted_edit,),
                render_prompt="Refined duplicate",
                resolution=GenerateResolution.RESOLUTION_512,
                transformation=RefineTransformation.BALANCED,
                strength=0.50,
                settings=image_settings(),
            ),
            created_at=datetime.now(UTC),
        )
    )

    Stack(
        name="Valid",
        cards=(
            Card(
                id=duplicate_card_id,
                name="Duplicate",
                revisions=(duplicate_revision, refined),
            ),
        ),
    )

    assert refined.background is not None
    invalid_refined = refined.model_copy(
        update={
            "background": refined.background.model_copy(
                update={
                    "provenance": refined.background.provenance.model_copy(
                        update={"edit_lineage": ()}
                    )
                }
            )
        }
    )
    with pytest.raises(ValidationError, match="Refine lineage"):
        Stack(
            name="Invalid",
            cards=(
                Card(
                    id=duplicate_card_id,
                    name="Duplicate",
                    revisions=(duplicate_revision, invalid_refined),
                ),
            ),
        )


def test_direct_generate_provenance_requires_authored_inputs() -> None:
    with pytest.raises(ValidationError, match="nonempty Description"):
        GenerateInputs(description=" \n ")


def test_edit_preserve_options_use_only_approved_serialized_categories() -> None:
    preserve = EditPreserveOptions(
        subject_identity=True,
        pose_and_expression=True,
        composition_and_framing=True,
        background=True,
        lighting_and_color=True,
        existing_text_and_logos=True,
    )

    assert preserve.model_dump(mode="json") == {
        "subject_identity": True,
        "pose_and_expression": True,
        "composition_and_framing": True,
        "background": True,
        "lighting_and_color": True,
        "existing_text_and_logos": True,
    }
    for obsolete_name in (
        "composition",
        "camera",
        "lighting",
        "color_palette",
        "visual_style",
    ):
        with pytest.raises(ValidationError, match="extra_forbidden"):
            EditPreserveOptions.model_validate({obsolete_name: True})


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
            resolution=GenerateResolution.RESOLUTION_512,
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
            settings=image_settings(width=880, height=672),
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
                resolution=GenerateResolution.RESOLUTION_512,
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


def test_multistep_refine_and_edit_lineage_must_match_resolved_sources() -> None:
    card_id = uuid4()
    edit_one = AcceptedEdit(
        instruction="Open the gate.",
        preserve=EditPreserveOptions(subject_identity=True),
        expanded_prompt="Open the gate. Preserve subject identity.",
    )
    edit_two = AcceptedEdit(
        instruction="Add ivy.",
        preserve=EditPreserveOptions(composition_and_framing=True),
        expanded_prompt="Add ivy. Preserve composition.",
    )
    edit_three = AcceptedEdit(
        instruction="Light the lanterns.",
        preserve=EditPreserveOptions(lighting_and_color=True),
        expanded_prompt="Light the lanterns. Preserve the color palette.",
    )
    direct = CardRevision(
        background=GeneratedBackground(
            image_path="assets/cards/card/direct.png",
            provenance=image_provenance(),
            created_at=datetime.now(UTC),
        )
    )
    assert direct.background is not None
    first_edit = CardRevision(
        background=GeneratedBackground(
            image_path="assets/cards/card/edit-one.png",
            provenance=EditProvenance(
                source=ImageSourceSnapshot(
                    card_id=card_id,
                    revision_id=direct.id,
                    background_id=direct.background.id,
                ),
                instruction=edit_one.instruction,
                preserve=edit_one.preserve,
                expanded_prompt=edit_one.expanded_prompt,
                resolution=GenerateResolution.RESOLUTION_512,
                edit_lineage=(edit_one,),
                settings=image_settings(),
            ),
            created_at=datetime.now(UTC),
        )
    )
    assert first_edit.background is not None
    second_edit = CardRevision(
        background=GeneratedBackground(
            image_path="assets/cards/card/edit-two.png",
            provenance=EditProvenance(
                source=ImageSourceSnapshot(
                    card_id=card_id,
                    revision_id=first_edit.id,
                    background_id=first_edit.background.id,
                ),
                instruction=edit_two.instruction,
                preserve=edit_two.preserve,
                expanded_prompt=edit_two.expanded_prompt,
                resolution=GenerateResolution.RESOLUTION_512,
                edit_lineage=(edit_one, edit_two),
                settings=image_settings(),
            ),
            created_at=datetime.now(UTC),
        )
    )
    assert second_edit.background is not None
    refined = CardRevision(
        background=GeneratedBackground(
            image_path="assets/cards/card/refined.png",
            provenance=RefineProvenance(
                source=ImageSourceSnapshot(
                    card_id=card_id,
                    revision_id=second_edit.id,
                    background_id=second_edit.background.id,
                ),
                description="A refined courtyard",
                edit_lineage=(edit_one, edit_two),
                render_prompt="A refined courtyard",
                resolution=GenerateResolution.RESOLUTION_512,
                transformation=RefineTransformation.BALANCED,
                strength=0.50,
                settings=image_settings(),
            ),
            created_at=datetime.now(UTC),
        )
    )
    assert refined.background is not None
    third_edit = CardRevision(
        background=GeneratedBackground(
            image_path="assets/cards/card/edit-three.png",
            provenance=EditProvenance(
                source=ImageSourceSnapshot(
                    card_id=card_id,
                    revision_id=refined.id,
                    background_id=refined.background.id,
                ),
                instruction=edit_three.instruction,
                preserve=edit_three.preserve,
                expanded_prompt=edit_three.expanded_prompt,
                resolution=GenerateResolution.RESOLUTION_512,
                edit_lineage=(edit_one, edit_two, edit_three),
                settings=image_settings(),
            ),
            created_at=datetime.now(UTC),
        )
    )
    valid_card = Card(
        id=card_id,
        name="Card",
        revisions=(direct, first_edit, second_edit, refined, third_edit),
    )

    Stack(name="Valid", cards=(valid_card,))

    for invalid_revision, operation_name in (
        (refined, "Refine"),
        (third_edit, "Edit"),
    ):
        assert invalid_revision.background is not None
        invalid_provenance = invalid_revision.background.provenance.model_copy(
            update={
                "settings": invalid_revision.background.provenance.settings.model_copy(
                    update={"height": 464}
                )
            }
        )
        invalid_background = invalid_revision.background.model_copy(
            update={"provenance": invalid_provenance}
        )
        revisions = tuple(
            revision.model_copy(update={"background": invalid_background})
            if revision.id == invalid_revision.id
            else revision
            for revision in valid_card.revisions
        )
        with pytest.raises(
            ValidationError,
            match=f"{operation_name} dimensions",
        ):
            Stack(
                name="Invalid",
                cards=(valid_card.model_copy(update={"revisions": revisions}),),
            )

    assert isinstance(refined.provenance, RefineProvenance)
    dropped_refine = refined.model_copy(
        update={
            "background": refined.background.model_copy(
                update={
                    "provenance": refined.provenance.model_copy(
                        update={"edit_lineage": (edit_two,)}
                    )
                }
            )
        }
    )
    with pytest.raises(ValidationError, match="Refine lineage"):
        Stack(
            name="Invalid",
            cards=(
                valid_card.model_copy(
                    update={
                        "revisions": (
                            direct,
                            first_edit,
                            second_edit,
                            dropped_refine,
                            third_edit,
                        )
                    }
                ),
            ),
        )

    assert isinstance(third_edit.provenance, EditProvenance)
    fabricated_edit = AcceptedEdit(
        instruction="Invented.",
        preserve=EditPreserveOptions(),
        expanded_prompt="Invented.",
    )
    for invalid_lineage in (
        (edit_two, edit_three),
        (edit_two, edit_one, edit_three),
        (edit_one, fabricated_edit, edit_two, edit_three),
    ):
        invalid_third_edit = third_edit.model_copy(
            update={
                "background": third_edit.background.model_copy(
                    update={
                        "provenance": third_edit.provenance.model_copy(
                            update={"edit_lineage": invalid_lineage}
                        )
                    }
                )
            }
        )
        with pytest.raises(ValidationError, match="Edit lineage"):
            Stack(
                name="Invalid",
                cards=(
                    valid_card.model_copy(
                        update={
                            "revisions": (
                                direct,
                                first_edit,
                                second_edit,
                                refined,
                                invalid_third_edit,
                            )
                        }
                    ),
                ),
            )


def test_legacy_generate_source_has_an_empty_inherited_edit_lineage() -> None:
    card_id = uuid4()
    legacy = CardRevision(
        background=GeneratedBackground(
            image_path="assets/cards/card/legacy.png",
            provenance=LegacyGenerateProvenance(
                render_prompt="Historical exact prompt",
                settings=image_settings(width=1001, height=777),
            ),
            created_at=datetime.now(UTC),
        )
    )
    assert legacy.background is not None
    refined = CardRevision(
        background=GeneratedBackground(
            image_path="assets/cards/card/refined.png",
            provenance=RefineProvenance(
                source=ImageSourceSnapshot(
                    card_id=card_id,
                    revision_id=legacy.id,
                    background_id=legacy.background.id,
                ),
                description="Refined",
                render_prompt="Refined",
                resolution=GenerateResolution.RESOLUTION_512,
                transformation=RefineTransformation.BALANCED,
                strength=0.50,
                settings=image_settings(),
            ),
            created_at=datetime.now(UTC),
        )
    )

    Stack(
        name="Valid",
        cards=(Card(id=card_id, name="Card", revisions=(legacy, refined)),),
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
