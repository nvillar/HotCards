"""Tests for serialized stack-domain boundaries."""

import json
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from pydantic import TypeAdapter, ValidationError

from hotcards.domain.image_dimensions import (
    AspectRatio,
    ResolutionTier,
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
    CurrentSourceSize,
    DerivedImageSourceSnapshot,
    DirectGenerateProvenance,
    DuplicateProvenance,
    EditDraft,
    EditPreserveOptions,
    EditProvenance,
    ExactOutputSize,
    GeneratedBackground,
    GeneratedSoundAsset,
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
    PresetOutputSize,
    RefineProvenance,
    RefineTransformation,
    ResolvedCardReference,
    RunOverlayMode,
    SoundDefinition,
    SoundGenerationProvenance,
    Stack,
    StyleDefinition,
    StyleSnapshot,
    UnresolvedCardReference,
    image_edit_lineage,
    image_operation_settings,
    original_image_provenance,
)


def generated_sound() -> GeneratedSoundAsset:
    return GeneratedSoundAsset(
        audio_path=(
            "assets/sounds/27dd6a14-1a2a-4b13-9705-b56d70474a17/"
            "sound-47e14ba7-3e57-48e3-9aac-edf495305fb1.wav"
        ),
        provenance=SoundGenerationProvenance(
            prompt="A wooden knock",
            duration_seconds=2,
            seed=42,
            generation_duration_milliseconds=1234,
        ),
        created_at=datetime.now(UTC),
    )


def hotspot_polygon(offset: float = 0.0) -> Polygon:
    return Polygon(
        points=(
            Point(x=0.1 + offset, y=0.1),
            Point(x=0.3 + offset, y=0.1),
            Point(x=0.2 + offset, y=0.4),
        )
    )


def image_settings(
    *,
    width: int = 512,
    height: int = 384,
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


def derived_source_snapshot(
    *,
    card_id: UUID | None = None,
    revision_id: UUID | None = None,
    background_id: UUID | None = None,
    width: int = 512,
    height: int = 384,
    edit_lineage: tuple[AcceptedEdit, ...] = (),
) -> DerivedImageSourceSnapshot:
    return DerivedImageSourceSnapshot(
        card_id=card_id or uuid4(),
        revision_id=revision_id or uuid4(),
        background_id=background_id or uuid4(),
        width=width,
        height=height,
        seed=42,
        edit_lineage=edit_lineage,
    )


def test_stack_round_trips_sound_catalog_and_hotspot_reference() -> None:
    sound = SoundDefinition(
        name="Door knock",
        prompt="  A wooden knock  ",
        generated=generated_sound(),
    )
    revision = CardRevision(
        hotspot_set=HotspotSet(
            interactions=(Interaction(sound_id=sound.id, polygons=(hotspot_polygon(),)),)
        )
    )
    stack = Stack(
        name="Sound stack",
        sounds=(sound,),
        cards=(Card(name="Door", revisions=(revision,)),),
    )

    loaded = Stack.model_validate_json(stack.model_dump_json())

    assert loaded.sounds[0].prompt == "A wooden knock"
    assert loaded.cards[0].active_revision.hotspot_set is not None
    assert loaded.cards[0].active_revision.hotspot_set.interactions[0].sound_id == sound.id


def test_stack_rejects_unknown_or_ambiguously_named_sounds() -> None:
    revision = CardRevision(
        hotspot_set=HotspotSet(
            interactions=(Interaction(sound_id=uuid4(), polygons=(hotspot_polygon(),)),)
        )
    )
    with pytest.raises(ValidationError, match="must identify Sounds"):
        Stack(name="Unknown sound", cards=(Card(name="Card", revisions=(revision,)),))

    with pytest.raises(ValidationError, match="Sound names must be unique"):
        Stack(
            name="Duplicate sounds",
            sounds=(
                SoundDefinition(name=" Knock "),
                SoundDefinition(name="knock"),
            ),
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


def test_edit_drafts_preserve_raw_text_and_use_independent_generations() -> None:
    first = CardRevision(edit_draft=EditDraft(instruction="  Keep raw text.\n"))
    second = CardRevision()
    stack = Stack(name="Drafts", cards=(Card(name="Card", revisions=(first, second)),))

    loaded = Stack.model_validate_json(stack.model_dump_json())

    assert loaded.cards[0].revisions[0].edit_draft == first.edit_draft
    assert loaded.cards[0].revisions[0].edit_draft.instruction == "  Keep raw text.\n"
    assert (
        loaded.cards[0].revisions[0].edit_draft.generation_id
        != loaded.cards[0].revisions[1].edit_draft.generation_id
    )


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
            ResolvedCardReference(target_card_id=destination.id) for destination in destinations
        )
    )
    source = Card(name="Source", revisions=(revision,))
    stack = Stack(name="Castle", cards=(source, *destinations))

    values = stack.model_dump(mode="json")

    serialized_revision = values["cards"][0]["revisions"][0]
    assert tuple(
        reference["target_card_id"] for reference in serialized_revision["references"]
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
                                    "references": (ResolvedCardReference(target_card_id=source.id),)
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
    chime = SoundDefinition(name="Chime")
    resolved = Interaction(
        action=NavigateAction(target=ResolvedCardReference(target_card_id=destination.id)),
        sound_id=chime.id,
        key_changes=HotspotKeyChanges(
            remove=(red_key.id,),
            grant=(door_open.id,),
        ),
        polygons=(hotspot_polygon(),),
    )
    sound_only = Interaction(
        sound_id=chime.id,
        key_changes=HotspotKeyChanges(
            remove=(red_key.id,),
            grant=(door_open.id,),
        ),
        polygons=(hotspot_polygon(0.1),),
    )
    grant_only = Interaction(
        key_changes=HotspotKeyChanges(
            remove=(red_key.id,),
            grant=(door_open.id,),
        ),
        polygons=(hotspot_polygon(0.2),),
    )
    remove_only = Interaction(
        key_changes=HotspotKeyChanges(remove=(red_key.id,)),
        polygons=(hotspot_polygon(0.3),),
    )
    unresolved = Interaction(
        action=NavigateAction(target=UnresolvedCardReference(target_name="Former room")),
        polygons=(hotspot_polygon(0.4),),
    )
    actionless = Interaction(polygons=(hotspot_polygon(0.5),))
    source = Card(
        name="Source",
        revisions=(
            CardRevision(
                hotspot_set=HotspotSet(
                    interactions=(
                        resolved,
                        sound_only,
                        grant_only,
                        remove_only,
                        unresolved,
                        actionless,
                    )
                )
            ),
        ),
    )

    stack = Stack(
        name="Castle",
        keys=(red_key, door_open),
        sounds=(chime,),
        cards=(source, destination),
    )

    interactions = stack.cards[0].active_revision.hotspot_set
    assert interactions is not None
    assert [item.label for item in interactions.interactions] == [
        "Go to Castle Gate",
        "Play Chime",
        "Gain Door open",
        "Lose Red key",
        "Go to Former room",
        "New Hotspot",
    ]


def test_hotspot_key_contract_is_closed_and_references_stack_keys() -> None:
    red_key = KeyDefinition(name=" Red key ")
    interaction = Interaction(
        conditions=HotspotConditions(requires=(red_key.id,)),
        key_changes=HotspotKeyChanges(remove=(red_key.id,)),
        polygons=(hotspot_polygon(),),
    )
    stack = Stack(
        name="Castle",
        keys=(red_key,),
        cards=(
            Card(
                name="Card",
                revisions=(CardRevision(hotspot_set=HotspotSet(interactions=(interaction,))),),
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
                                        conditions=HotspotConditions(requires=(uuid4(),)),
                                        polygons=(hotspot_polygon(),),
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


def test_generate_output_size_is_revision_local_and_strict() -> None:
    revision = CardRevision()

    assert revision.generate_output_size == PresetOutputSize(tier=ResolutionTier.MEDIUM)
    assert revision.model_dump(mode="json")["generate_output_size"] == {
        "mode": "preset",
        "tier": 512,
    }
    with pytest.raises(ValidationError, match="generate_output_size"):
        CardRevision.model_validate(
            {"generate_output_size": {"mode": "preset", "tier": 300}},
        )


def test_exact_generate_output_size_is_aligned_and_matches_stack_ratio() -> None:
    revision = CardRevision(generate_output_size=ExactOutputSize(width=640, height=480))
    Stack(
        name="Valid",
        aspect_ratio=AspectRatio.LANDSCAPE,
        cards=(Card(name="Card", revisions=(revision,)),),
    )

    with pytest.raises(ValidationError, match="aligned"):
        ExactOutputSize(width=641, height=480)
    with pytest.raises(ValidationError, match="stack aspect ratio"):
        Stack(
            name="Invalid",
            aspect_ratio=AspectRatio.LANDSCAPE,
            cards=(
                Card(
                    name="Card",
                    revisions=(
                        CardRevision(
                            generate_output_size=ExactOutputSize(
                                width=640,
                                height=496,
                            )
                        ),
                    ),
                ),
            ),
        )


@pytest.mark.parametrize("duplicate", (False, True))
def test_direct_generate_exact_provenance_requires_stack_compatible_aspect(
    duplicate: bool,
) -> None:
    direct = DirectGenerateProvenance(
        inputs=GenerateInputs(
            description="A courtyard",
            output_size=ExactOutputSize(width=640, height=496),
        ),
        render_prompt="A courtyard",
        settings=image_settings(width=640, height=496),
    )
    provenance = (
        DuplicateProvenance(
            source=ImageSourceSnapshot(
                card_id=uuid4(),
                revision_id=uuid4(),
                background_id=uuid4(),
            ),
            original_provenance=direct,
        )
        if duplicate
        else direct
    )
    background = GeneratedBackground(
        image_path="assets/cards/card/image.png",
        provenance=provenance,
        created_at=datetime.now(UTC),
    )

    with pytest.raises(ValidationError, match="4:3 stack aspect ratio"):
        Stack(
            name="Invalid",
            aspect_ratio=AspectRatio.LANDSCAPE,
            cards=(
                Card(
                    name="Card",
                    revisions=(CardRevision(background=background),),
                ),
            ),
        )


def test_direct_generate_exact_provenance_accepts_ratio_rounded_size() -> None:
    direct = DirectGenerateProvenance(
        inputs=GenerateInputs(
            description="A courtyard",
            output_size=ExactOutputSize(width=592, height=448),
        ),
        render_prompt="A courtyard",
        settings=image_settings(width=592, height=448),
    )
    background = GeneratedBackground(
        image_path="assets/cards/card/image.png",
        provenance=direct,
        created_at=datetime.now(UTC),
    )

    Stack(
        name="Valid",
        aspect_ratio=AspectRatio.LANDSCAPE,
        cards=(Card(name="Card", revisions=(CardRevision(background=background),)),),
    )


@pytest.mark.parametrize("aspect_ratio", tuple(AspectRatio))
@pytest.mark.parametrize("resolution", tuple(ResolutionTier))
def test_stack_requires_generate_dimensions_for_every_schema_combination(
    aspect_ratio: AspectRatio,
    resolution: ResolutionTier,
) -> None:
    width, height = output_dimensions(resolution, aspect_ratio)
    background = GeneratedBackground(
        image_path="assets/cards/card/image.png",
        provenance=DirectGenerateProvenance(
            inputs=GenerateInputs(
                description="A courtyard",
                output_size=PresetOutputSize(tier=resolution),
            ),
            render_prompt="A courtyard",
            settings=image_settings(width=width, height=height),
        ),
        created_at=datetime.now(UTC),
    )
    revision = CardRevision(background=background)
    card = Card(name="Card", revisions=(revision,))

    Stack(name="Valid", aspect_ratio=aspect_ratio, cards=(card,))

    invalid_settings = background.provenance.settings.model_copy(update={"width": width + 16})
    invalid_background = background.model_copy(
        update={
            "provenance": background.provenance.model_copy(update={"settings": invalid_settings})
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
                            revision.model_copy(update={"background": invalid_background}),
                        )
                    }
                ),
            ),
        )


@pytest.mark.parametrize("duplicate", (False, True))
def test_legacy_generate_preserves_historic_nonpreset_dimensions(duplicate: bool) -> None:
    legacy = LegacyGenerateProvenance(
        render_prompt="Historical exact prompt\n\nKeep authored  spacing.",
        references=(
            ImageReferenceSnapshot(card_id=uuid4(), revision_id=uuid4(), background_id=uuid4()),
        ),
        settings=image_settings(width=1007, height=783),
    )
    background = GeneratedBackground(
        image_path="assets/cards/card/legacy.png",
        provenance=(
            DuplicateProvenance(
                source=ImageSourceSnapshot(
                    card_id=uuid4(), revision_id=uuid4(), background_id=uuid4()
                ),
                original_provenance=legacy,
            )
            if duplicate
            else legacy
        ),
        created_at=datetime.now(UTC),
    )

    stack = Stack(
        name="Historical",
        aspect_ratio=AspectRatio.LANDSCAPE,
        cards=(Card(name="Card", revisions=(CardRevision(background=background),)),),
    )
    loaded = Stack.model_validate_json(stack.model_dump_json())
    assert original_image_provenance(loaded.cards[0].active_revision.provenance) == legacy


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
            source=derived_source_snapshot(edit_lineage=(accepted_edit,)),
            description="A moonlit courtyard",
            render_prompt="A moonlit courtyard. Open the gate.",
            output_size=CurrentSourceSize(width=512, height=384),
            transformation=RefineTransformation.BALANCED,
            strength=0.50,
            settings=image_settings(),
        ),
        EditProvenance(
            source=derived_source_snapshot(),
            instruction=accepted_edit.instruction,
            preserve=accepted_edit.preserve,
            expanded_prompt=accepted_edit.expanded_prompt,
            output_size=PresetOutputSize(tier=ResolutionTier.FULL),
            prompt_token_count=24,
            settings=image_settings(width=1024, height=768),
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
        source=derived_source_snapshot(),
        instruction=accepted_edit.instruction,
        preserve=accepted_edit.preserve,
        expanded_prompt=accepted_edit.expanded_prompt,
        output_size=CurrentSourceSize(width=512, height=384),
        prompt_token_count=24,
        settings=image_settings(),
    )
    duplicate = DuplicateProvenance(
        source=source,
        original_provenance=original,
    )

    assert duplicate.settings == original.settings
    assert original_image_provenance(duplicate) == original
    assert image_edit_lineage(duplicate) == (accepted_edit,)
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
        source=derived_source_snapshot(),
        instruction=accepted_edit.instruction,
        preserve=accepted_edit.preserve,
        expanded_prompt=accepted_edit.expanded_prompt,
        output_size=CurrentSourceSize(width=512, height=384),
        prompt_token_count=24,
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
                source=derived_source_snapshot(
                    card_id=duplicate_card_id,
                    revision_id=duplicate_revision.id,
                    background_id=duplicate_revision.background.id,
                    edit_lineage=image_edit_lineage(duplicate_revision.background.provenance),
                ),
                description="Refined duplicate",
                render_prompt="Refined duplicate",
                output_size=CurrentSourceSize(width=512, height=384),
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

    assert refined.provenance is not None
    assert image_edit_lineage(refined.provenance) == (accepted_edit,)
    independent = Stack(
        name="Independent",
        cards=(Card(id=duplicate_card_id, name="Duplicate", revisions=(refined,)),),
    )
    assert Stack.model_validate_json(independent.model_dump_json()) == independent


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
    source = derived_source_snapshot()
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
            output_size=PresetOutputSize(tier=ResolutionTier.MEDIUM),
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
            output_size=PresetOutputSize(tier=ResolutionTier.LARGE),
            edit_lineage=(accepted,),
            prompt_token_count=24,
            settings=image_settings(width=768, height=576),
        )
    with pytest.raises(ValidationError, match="512-token budget"):
        EditProvenance(
            source=source,
            instruction=accepted.instruction,
            preserve=accepted.preserve,
            expanded_prompt=accepted.expanded_prompt,
            output_size=PresetOutputSize(tier=ResolutionTier.MEDIUM),
            prompt_token_count=513,
            settings=image_settings(),
        )


def test_edit_current_output_size_matches_exact_source_dimensions() -> None:
    card_id = uuid4()
    source = CardRevision(
        background=GeneratedBackground(
            image_path="assets/cards/card/legacy.png",
            provenance=LegacyGenerateProvenance(
                render_prompt="Legacy source",
                settings=image_settings(width=1008, height=752),
            ),
            created_at=datetime.now(UTC),
        )
    )
    assert source.background is not None
    accepted = AcceptedEdit(
        instruction="Open the gate.",
        preserve=EditPreserveOptions(subject_identity=True),
        expanded_prompt="Open the gate.",
    )

    def edited_revision(width: int, height: int) -> CardRevision:
        return CardRevision(
            background=GeneratedBackground(
                image_path="assets/cards/card/edited.png",
                provenance=EditProvenance(
                    source=derived_source_snapshot(
                        card_id=card_id,
                        revision_id=source.id,
                        background_id=source.background.id,
                        width=1008,
                        height=752,
                    ),
                    instruction=accepted.instruction,
                    preserve=accepted.preserve,
                    expanded_prompt=accepted.expanded_prompt,
                    output_size=CurrentSourceSize(
                        width=width,
                        height=height,
                    ),
                    prompt_token_count=12,
                    settings=image_settings(width=width, height=height),
                ),
                created_at=datetime.now(UTC),
            )
        )

    matching = edited_revision(1008, 752)
    Stack(
        name="Valid",
        cards=(
            Card(
                id=card_id,
                name="Card",
                revisions=(source, matching),
                active_revision_id=matching.id,
            ),
        ),
    )

    with pytest.raises(ValidationError, match="current-size derived output"):
        edited_revision(992, 752)


@pytest.mark.parametrize(
    ("source_tier", "output_tier", "duplicate_source"),
    [
        (ResolutionTier.FULL, ResolutionTier.SMALL, False),
        (ResolutionTier.FULL, ResolutionTier.MEDIUM, False),
        (ResolutionTier.FULL, ResolutionTier.LARGE, True),
        (ResolutionTier.FULL, ResolutionTier.FULL, False),
        (ResolutionTier.MEDIUM, ResolutionTier.MEDIUM, True),
    ],
)
def test_edit_preset_output_must_be_strictly_larger_than_source(
    source_tier: ResolutionTier,
    output_tier: ResolutionTier,
    duplicate_source: bool,
) -> None:
    card_id = uuid4()
    source_width, source_height = output_dimensions(
        source_tier,
        AspectRatio.LANDSCAPE,
    )
    direct = DirectGenerateProvenance(
        inputs=GenerateInputs(
            description="Source",
            output_size=PresetOutputSize(tier=source_tier),
        ),
        render_prompt="Source",
        settings=image_settings(width=source_width, height=source_height),
    )
    source_provenance = (
        DuplicateProvenance(
            source=ImageSourceSnapshot(
                card_id=uuid4(),
                revision_id=uuid4(),
                background_id=uuid4(),
            ),
            original_provenance=direct,
        )
        if duplicate_source
        else direct
    )
    source = CardRevision(
        background=GeneratedBackground(
            image_path="assets/cards/card/source.png",
            provenance=source_provenance,
            created_at=datetime.now(UTC),
        )
    )
    assert source.background is not None
    accepted = AcceptedEdit(
        instruction="Open the gate.",
        preserve=EditPreserveOptions(subject_identity=True),
        expanded_prompt="Open the gate.",
    )
    output_width, output_height = output_dimensions(
        output_tier,
        AspectRatio.LANDSCAPE,
    )
    with pytest.raises(ValidationError, match="more pixels"):
        EditProvenance(
            source=derived_source_snapshot(
                card_id=card_id,
                revision_id=source.id,
                background_id=source.background.id,
                width=source_width,
                height=source_height,
                edit_lineage=image_edit_lineage(source_provenance),
            ),
            instruction=accepted.instruction,
            preserve=accepted.preserve,
            expanded_prompt=accepted.expanded_prompt,
            output_size=PresetOutputSize(tier=output_tier),
            prompt_token_count=12,
            settings=image_settings(width=output_width, height=output_height),
        )


@pytest.mark.parametrize("duplicate_source", (False, True))
def test_edit_preset_output_accepts_strictly_larger_source_area(
    duplicate_source: bool,
) -> None:
    card_id = uuid4()
    direct = image_provenance()
    source_provenance = (
        DuplicateProvenance(
            source=ImageSourceSnapshot(
                card_id=uuid4(),
                revision_id=uuid4(),
                background_id=uuid4(),
            ),
            original_provenance=direct,
        )
        if duplicate_source
        else direct
    )
    source = CardRevision(
        background=GeneratedBackground(
            image_path="assets/cards/card/source.png",
            provenance=source_provenance,
            created_at=datetime.now(UTC),
        )
    )
    assert source.background is not None
    accepted = AcceptedEdit(
        instruction="Open the gate.",
        preserve=EditPreserveOptions(subject_identity=True),
        expanded_prompt="Open the gate.",
    )
    edited = CardRevision(
        background=GeneratedBackground(
            image_path="assets/cards/card/edited.png",
            provenance=EditProvenance(
                source=derived_source_snapshot(
                    card_id=card_id,
                    revision_id=source.id,
                    background_id=source.background.id,
                ),
                instruction=accepted.instruction,
                preserve=accepted.preserve,
                expanded_prompt=accepted.expanded_prompt,
                output_size=PresetOutputSize(tier=ResolutionTier.LARGE),
                prompt_token_count=12,
                settings=image_settings(width=768, height=576),
            ),
            created_at=datetime.now(UTC),
        )
    )

    Stack(
        name="Valid",
        cards=(
            Card(
                id=card_id,
                name="Card",
                revisions=(source, edited),
                active_revision_id=edited.id,
            ),
        ),
    )


def test_derived_provenance_survives_deleted_source_cards_and_backgrounds() -> None:
    source = CardRevision(
        background=GeneratedBackground(
            image_path="assets/cards/source/image-source.png",
            provenance=image_provenance(),
            created_at=datetime.now(UTC),
        )
    )
    source_card = Card(name="Source", revisions=(source,))
    derived = CardRevision(
        background=GeneratedBackground(
            image_path="assets/cards/derived/image-derived.png",
            provenance=RefineProvenance(
                source=derived_source_snapshot(
                    card_id=source_card.id,
                    revision_id=source.id,
                    background_id=source.background.id,
                ),
                description="A refined courtyard",
                render_prompt="A refined courtyard",
                output_size=CurrentSourceSize(width=512, height=384),
                transformation=RefineTransformation.PRESERVE,
                strength=0.75,
                settings=image_settings(),
            ),
            created_at=datetime.now(UTC),
        )
    )
    dependent_card = Card(name="Dependent", revisions=(derived,))

    Stack(name="Valid", cards=(source_card, dependent_card))
    independent = Stack(name="Independent", cards=(dependent_card,))
    loaded = Stack.model_validate_json(independent.model_dump_json())
    assert loaded == independent
    assert loaded.cards[0].active_revision.provenance == derived.provenance


def test_multistep_refine_and_edit_lineage_has_one_canonical_stored_sequence() -> None:
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
                source=derived_source_snapshot(
                    card_id=card_id,
                    revision_id=direct.id,
                    background_id=direct.background.id,
                ),
                instruction=edit_one.instruction,
                preserve=edit_one.preserve,
                expanded_prompt=edit_one.expanded_prompt,
                output_size=CurrentSourceSize(width=512, height=384),
                prompt_token_count=24,
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
                source=derived_source_snapshot(
                    card_id=card_id,
                    revision_id=first_edit.id,
                    background_id=first_edit.background.id,
                    edit_lineage=image_edit_lineage(first_edit.provenance),
                ),
                instruction=edit_two.instruction,
                preserve=edit_two.preserve,
                expanded_prompt=edit_two.expanded_prompt,
                output_size=CurrentSourceSize(width=512, height=384),
                prompt_token_count=24,
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
                source=derived_source_snapshot(
                    card_id=card_id,
                    revision_id=second_edit.id,
                    background_id=second_edit.background.id,
                    edit_lineage=image_edit_lineage(second_edit.provenance),
                ),
                description="A refined courtyard",
                render_prompt="A refined courtyard",
                output_size=CurrentSourceSize(width=512, height=384),
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
                source=derived_source_snapshot(
                    card_id=card_id,
                    revision_id=refined.id,
                    background_id=refined.background.id,
                    edit_lineage=image_edit_lineage(refined.provenance),
                ),
                instruction=edit_three.instruction,
                preserve=edit_three.preserve,
                expanded_prompt=edit_three.expanded_prompt,
                output_size=CurrentSourceSize(width=512, height=384),
                prompt_token_count=24,
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

    for revision, inherited, resulting in (
        (first_edit, (), (edit_one,)),
        (second_edit, (edit_one,), (edit_one, edit_two)),
        (refined, (edit_one, edit_two), (edit_one, edit_two)),
        (third_edit, (edit_one, edit_two), (edit_one, edit_two, edit_three)),
    ):
        provenance = revision.provenance
        assert isinstance(provenance, (EditProvenance, RefineProvenance))
        assert provenance.source.edit_lineage == inherited
        assert image_edit_lineage(provenance) == resulting
        payload = provenance.model_dump(mode="json")
        assert "edit_lineage" not in payload
        assert payload["source"]["edit_lineage"] == [
            accepted.model_dump(mode="json") for accepted in inherited
        ]
        with pytest.raises(ValidationError, match="extra_forbidden"):
            type(provenance).model_validate_json(json.dumps({**payload, "edit_lineage": []}))
        independent = Stack(
            name="Independent",
            cards=(Card(id=card_id, name="Card", revisions=(revision,)),),
        )
        assert Stack.model_validate_json(independent.model_dump_json()) == independent


def test_legacy_generate_source_has_an_empty_inherited_edit_lineage() -> None:
    card_id = uuid4()
    legacy = CardRevision(
        background=GeneratedBackground(
            image_path="assets/cards/card/legacy.png",
            provenance=LegacyGenerateProvenance(
                render_prompt="Historical exact prompt",
                settings=image_settings(width=503, height=391),
            ),
            created_at=datetime.now(UTC),
        )
    )
    assert legacy.background is not None
    refined = CardRevision(
        background=GeneratedBackground(
            image_path="assets/cards/card/refined.png",
            provenance=RefineProvenance(
                source=derived_source_snapshot(
                    card_id=card_id,
                    revision_id=legacy.id,
                    background_id=legacy.background.id,
                    width=503,
                    height=391,
                    edit_lineage=image_edit_lineage(legacy.provenance),
                ),
                description="Refined",
                render_prompt="Refined",
                output_size=PresetOutputSize(tier=ResolutionTier.FULL),
                transformation=RefineTransformation.BALANCED,
                strength=0.50,
                settings=image_settings(width=1024, height=768),
            ),
            created_at=datetime.now(UTC),
        )
    )

    Stack(
        name="Valid",
        cards=(Card(id=card_id, name="Card", revisions=(legacy, refined)),),
    )
    assert image_edit_lineage(refined.provenance) == ()
    assert image_operation_settings(legacy.provenance).width == 503


def test_historical_source_ids_do_not_form_a_live_revision_graph() -> None:
    first_card_id, second_card_id = uuid4(), uuid4()
    first_revision_id, second_revision_id = uuid4(), uuid4()
    first_background_id, second_background_id = uuid4(), uuid4()
    first_revision = CardRevision(
        id=first_revision_id,
        background=GeneratedBackground(
            id=first_background_id,
            image_path="assets/cards/first.png",
            provenance=RefineProvenance(
                source=derived_source_snapshot(
                    card_id=second_card_id,
                    revision_id=second_revision_id,
                    background_id=second_background_id,
                ),
                description="First",
                render_prompt="First",
                output_size=CurrentSourceSize(width=512, height=384),
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
                source=derived_source_snapshot(
                    card_id=first_card_id,
                    revision_id=first_revision_id,
                    background_id=first_background_id,
                ),
                description="Second",
                render_prompt="Second",
                output_size=CurrentSourceSize(width=512, height=384),
                transformation=RefineTransformation.BALANCED,
                strength=0.50,
                settings=image_settings(),
            ),
            created_at=datetime.now(UTC),
        ),
    )

    stack = Stack(
        name="Independent",
        cards=(
            Card(id=first_card_id, name="First", revisions=(first_revision,)),
            Card(id=second_card_id, name="Second", revisions=(second_revision,)),
        ),
    )
    assert Stack.model_validate_json(stack.model_dump_json()) == stack


def derived_provenance(
    operation: str,
    source: DerivedImageSourceSnapshot,
) -> RefineProvenance | EditProvenance:
    output_size = CurrentSourceSize(width=512, height=384)
    if operation == "refine":
        return RefineProvenance(
            source=source,
            description="Courtyard",
            render_prompt="Courtyard",
            output_size=output_size,
            transformation=RefineTransformation.BALANCED,
            strength=0.5,
            settings=image_settings(),
        )
    return EditProvenance(
        source=source,
        instruction="Open the gate.",
        preserve=EditPreserveOptions(background=True),
        expanded_prompt="Open the gate. Preserve the background.",
        output_size=output_size,
        prompt_token_count=24,
        settings=image_settings(),
    )


@pytest.mark.parametrize("operation", ("refine", "edit"))
def test_same_version_derived_background_is_independent_and_round_trips(operation: str) -> None:
    card_id, revision_id = uuid4(), uuid4()
    source = derived_source_snapshot(card_id=card_id, revision_id=revision_id)
    provenance = derived_provenance(operation, source)
    background = GeneratedBackground(
        image_path="assets/cards/result.png",
        provenance=provenance,
        created_at=datetime.now(UTC),
    )
    stack = Stack(
        name="Independent",
        cards=(
            Card(
                id=card_id,
                name="Card",
                revisions=(CardRevision(id=revision_id, background=background),),
            ),
        ),
    )
    assert Stack.model_validate_json(stack.model_dump_json()) == stack
    with pytest.raises(ValidationError, match="identity must differ"):
        GeneratedBackground.model_validate(
            {**background.model_dump(mode="python"), "id": source.background_id}
        )


@pytest.mark.parametrize("operation", ("refine", "edit"))
@pytest.mark.parametrize(
    ("change", "message"),
    (
        ({"source": {"width": 496}}, "match its source dimensions"),
        ({"settings": {"width": 496}}, "match the selected output size"),
        ({"output_size": {"mode": "preset", "tier": ResolutionTier.MEDIUM}}, "more pixels"),
        ({"source": {"height": 0}}, "greater than 0"),
        ({"source": {"seed": True}}, "valid integer"),
        ({"source": {"edit_lineage": ({"instruction": "Missing facts"},)}}, "Field required"),
        ({"source": {"image_path": "old.png"}}, "extra_forbidden"),
        ({"source": {"original_provenance": {}}}, "extra_forbidden"),
    ),
)
def test_derived_provenance_validates_local_source_and_output_facts(
    operation: str, change: dict[str, dict[str, object]], message: str
) -> None:
    provenance = derived_provenance(operation, derived_source_snapshot())
    payload = provenance.model_dump(mode="python")
    for field, values in change.items():
        if field == "output_size":
            payload[field] = values
        else:
            payload[field].update(values)
    with pytest.raises(ValidationError, match=message):
        type(provenance).model_validate(payload)


def test_refine_provenance_requires_captured_source_seed() -> None:
    provenance = derived_provenance("refine", derived_source_snapshot())
    payload = provenance.model_dump(mode="python")
    payload["settings"]["seed"] = 43
    with pytest.raises(ValidationError, match="reuse its captured source seed"):
        RefineProvenance.model_validate(payload)


@pytest.mark.parametrize("operation", ("refine", "edit"))
@pytest.mark.parametrize("duplicate", (False, True))
def test_derived_presets_require_correct_dimensions_even_inside_duplicates(
    operation: str, duplicate: bool
) -> None:
    provenance = derived_provenance(operation, derived_source_snapshot())
    payload = provenance.model_dump(mode="python")
    payload["output_size"] = {"mode": "preset", "tier": ResolutionTier.LARGE}
    payload["settings"].update(width=768, height=592)
    invalid = type(provenance).model_validate(payload)
    wrapped = (
        DuplicateProvenance(
            source=ImageSourceSnapshot(card_id=uuid4(), revision_id=uuid4(), background_id=uuid4()),
            original_provenance=invalid,
        )
        if duplicate
        else invalid
    )
    with pytest.raises(ValidationError, match="dimensions must match"):
        Stack(
            name="Invalid",
            cards=(
                Card(
                    name="Card",
                    revisions=(
                        CardRevision(
                            background=GeneratedBackground(
                                image_path="assets/cards/invalid.png",
                                provenance=wrapped,
                                created_at=datetime.now(UTC),
                            )
                        ),
                    ),
                ),
            ),
        )


@pytest.mark.parametrize("operation", ("refine", "edit"))
def test_derived_source_requires_all_captured_facts(operation: str) -> None:
    provenance = derived_provenance(operation, derived_source_snapshot())
    for field in ("width", "height", "seed", "edit_lineage"):
        payload = provenance.model_dump(mode="python")
        del payload["source"][field]
        with pytest.raises(ValidationError, match=f"source.{field}"):
            type(provenance).model_validate(payload)


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
    assert loaded_interaction.label == "Go to Garden"
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


def test_hotspot_requires_exactly_one_polygon() -> None:
    with pytest.raises(ValidationError, match="polygons"):
        Interaction()
    with pytest.raises(ValidationError, match="at least 1"):
        Interaction(polygons=())
    with pytest.raises(ValidationError, match="at most 1"):
        Interaction(polygons=(hotspot_polygon(), hotspot_polygon(0.2)))
