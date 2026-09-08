"""Tests for serialized stack-domain boundaries."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from hotcards.domain.image_dimensions import AspectRatio, ResolutionTier, output_dimensions
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
    DuplicateOperation,
    EditDraft,
    EditOperation,
    ExactOutputSize,
    GeneratedBackground,
    GeneratedSoundAsset,
    GenerateInputs,
    GenerateOperation,
    HotspotConditions,
    HotspotKeyChanges,
    HotspotSet,
    ImageOperationSettings,
    ImageOriginFacts,
    ImageProvenance,
    ImageReferenceSnapshot,
    ImageSourceSnapshot,
    Interaction,
    KeyDefinition,
    NavigateAction,
    Point,
    Polygon,
    PresetOutputSize,
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


def hotspot_polygon(offset: float = 0.0) -> Polygon:
    return Polygon(
        points=(
            Point(x=0.1 + offset, y=0.1),
            Point(x=0.3 + offset, y=0.1),
            Point(x=0.2 + offset, y=0.4),
        )
    )


def image_settings(width: int = 512, height: int = 384) -> ImageOperationSettings:
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


def source_snapshot(**updates: object) -> DerivedImageSourceSnapshot:
    return DerivedImageSourceSnapshot(
        **{
            "card_id": uuid4(),
            "revision_id": uuid4(),
            "background_id": uuid4(),
            "width": 512,
            "height": 384,
            "seed": 42,
            **updates,
        }
    )


def generate_provenance(
    width: int = 512,
    height: int = 384,
    output_size: PresetOutputSize | ExactOutputSize | None = None,
) -> ImageProvenance:
    return ImageProvenance(
        origin=ImageOriginFacts(
            render_prompt="A moonlit courtyard", settings=image_settings(width, height)
        ),
        authoring=GenerateOperation(
            inputs=GenerateInputs(
                description="A moonlit courtyard",
                output_size=output_size or PresetOutputSize(tier=ResolutionTier.MEDIUM),
            )
        ),
    )


def edit_provenance(
    *,
    source: DerivedImageSourceSnapshot | None = None,
    inherited: tuple[AcceptedEdit, ...] = (),
    instruction: str = "Open the gate.",
    width: int = 512,
    height: int = 384,
    output_size: CurrentSourceSize | PresetOutputSize | None = None,
) -> ImageProvenance:
    return ImageProvenance(
        origin=ImageOriginFacts(
            render_prompt=instruction,
            settings=image_settings(width, height),
            edit_lineage=(
                *inherited,
                AcceptedEdit(instruction=instruction, expanded_prompt=instruction),
            ),
        ),
        authoring=EditOperation(
            source=source or source_snapshot(),
            output_size=output_size or CurrentSourceSize(width=width, height=height),
            prompt_token_count=24,
        ),
    )


def duplicate_provenance(provenance: ImageProvenance) -> ImageProvenance:
    return ImageProvenance(
        origin=provenance.origin,
        authoring=DuplicateOperation(
            source=ImageSourceSnapshot(card_id=uuid4(), revision_id=uuid4(), background_id=uuid4()),
            original_authoring=original_image_provenance(provenance).authoring,
        ),
    )


def background(provenance: ImageProvenance) -> GeneratedBackground:
    return GeneratedBackground(
        image_path="assets/cards/card/image.png",
        provenance=provenance,
        created_at=datetime.now(UTC),
    )


def image_stack(
    provenance: ImageProvenance, aspect_ratio: AspectRatio = AspectRatio.LANDSCAPE
) -> Stack:
    return Stack(
        name="Images",
        aspect_ratio=aspect_ratio,
        cards=(Card(name="Card", revisions=(CardRevision(background=background(provenance)),)),),
    )


def test_stack_defaults_and_dump_field_selection() -> None:
    stack = Stack(name="Castle")
    assert CURRENT_SCHEMA_VERSION == stack.schema_version == 15
    assert stack.aspect_ratio is AspectRatio.LANDSCAPE
    assert stack.canvas == CanvasSize(width=1024, height=768)
    assert stack.run_overlay_mode is RunOverlayMode.HIDDEN
    assert stack.styles == BUILT_IN_STYLES
    assert stack.new_card_style_id == HYPERCARD_STYLE_ID
    assert stack.keys == stack.sounds == stack.cards == ()
    assert stack.model_dump(exclude_unset=True) == {"name": "Castle"}
    assert stack.model_dump(mode="json")["aspect_ratio"] == "4:3"
    assert "canvas" not in stack.model_dump()
    with pytest.raises(ValidationError, match="frozen"):
        stack.aspect_ratio = AspectRatio.SQUARE


@pytest.mark.parametrize("version", (True, 0, 13, 14, 16, "15"))
def test_only_current_schema_is_accepted(version: object) -> None:
    with pytest.raises(ValidationError):
        Stack.model_validate({"name": "Stack", "schema_version": version})


def test_edit_drafts_round_trip_identical_raw_text_with_independent_generations() -> None:
    first = CardRevision(edit_draft=EditDraft(instruction="  Keep raw text.\n"))
    second = CardRevision(edit_draft=EditDraft(instruction=first.edit_draft.instruction))
    stack = Stack(name="Drafts", cards=(Card(name="Card", revisions=(first, second)),))
    loaded = Stack.model_validate_json(stack.model_dump_json())
    assert loaded == stack
    drafts = [revision.edit_draft for revision in loaded.cards[0].revisions]
    assert drafts[0].instruction == drafts[1].instruction == "  Keep raw text.\n"
    assert drafts[0].generation_id != drafts[1].generation_id


def test_built_in_style_ids_remain_stable_across_product_renames() -> None:
    assert tuple(str(style.id) for style in BUILT_IN_STYLES) == (
        "2372dddb-99e0-5e62-b740-405d0f7dab2a",
        "a23ac4cf-0358-500a-a4fb-1f120f1f9e47",
        "7baf1057-786a-587e-a52f-c63b513519c6",
        "72b678e8-49b0-50d8-af4b-735b31def4c0",
        "5f602dd8-d459-5633-bd1a-f9b1c86fe473",
        "61a9ca01-5458-5607-9940-8088955640a2",
        "1e0aeb8e-7112-5ddc-becf-f6053ce31ffb",
        "a73faa84-f09b-5832-b8a7-9af17fcb7c86",
        "ea72f569-5ac1-5906-88b8-a0b7171b4d15",
        "eda5058a-d463-5fd7-82dc-f50e7a571efa",
    )


def test_styles_reference_stable_ids_and_reject_unknown_or_duplicate_selections() -> None:
    style = StyleDefinition(name=" Ink ", prompt_text="  Ink lines  ")
    revision = CardRevision(style_id=style.id)
    stack = Stack(
        name="Styles",
        styles=(style,),
        new_card_style_id=style.id,
        cards=(Card(name="Card", revisions=(revision,)),),
    )
    assert Stack.model_validate_json(stack.model_dump_json()) == stack
    assert stack.style_by_id(style.id).prompt_text == "Ink lines"
    assert stack.style_by_id(None) is None
    with pytest.raises(ValidationError, match="Style names"):
        Stack(name="Invalid", styles=(style, StyleDefinition(name="INK")), new_card_style_id=None)
    with pytest.raises(ValidationError, match="Style IDs"):
        Stack(name="Invalid", styles=(style, style), new_card_style_id=None)
    with pytest.raises(ValidationError, match="new_card_style_id"):
        Stack(name="Invalid", new_card_style_id=uuid4())
    with pytest.raises(ValidationError, match="revision style_id"):
        Stack(name="Invalid", cards=(Card(name="Card", revisions=(revision,)),))


def test_sound_asset_metadata_and_hotspot_reference_round_trip() -> None:
    generated = GeneratedSoundAsset(
        audio_path="assets/sounds/sound/sound-generated.wav",
        provenance=SoundGenerationProvenance(
            prompt="A wooden knock",
            duration_seconds=2,
            seed=42,
            generation_duration_milliseconds=1234,
        ),
        created_at=datetime.now(UTC),
    )
    sound = SoundDefinition(name="Knock", prompt="  A wooden knock  ", generated=generated)
    revision = CardRevision(
        hotspot_set=HotspotSet(
            interactions=(Interaction(sound_id=sound.id, polygons=(hotspot_polygon(),)),)
        )
    )
    stack = Stack(name="Sounds", sounds=(sound,), cards=(Card(name="Door", revisions=(revision,)),))
    loaded = Stack.model_validate_json(stack.model_dump_json())
    assert loaded == stack
    assert loaded.sound_by_id(sound.id).prompt == "A wooden knock"
    assert loaded.sounds[0].generated == generated
    metadata = loaded.sounds[0].generated.provenance
    assert (metadata.seed, metadata.generation_duration_milliseconds) == (42, 1234)
    assert (metadata.sample_rate, metadata.channels, metadata.steps, metadata.cfg) == (
        44100,
        2,
        8,
        1.0,
    )
    assert loaded.cards[0].active_revision.hotspot_set.interactions[0].sound_id == sound.id
    with pytest.raises(ValidationError, match="must identify Sounds"):
        Stack(name="Missing sound", cards=stack.cards)
    with pytest.raises(ValidationError, match="Sound names"):
        Stack(name="Duplicate", sounds=(sound, SoundDefinition(name=" knock ")))
    with pytest.raises(ValidationError, match="Sound IDs"):
        Stack(name="Duplicate", sounds=(sound, sound))


def test_references_are_ordered_discriminated_and_limited_to_two() -> None:
    destinations = (Card(name="First"), Card(name="Second"))
    references = tuple(ResolvedCardReference(target_card_id=card.id) for card in destinations)
    revision = CardRevision(references=references)
    stack = Stack(name="Stack", cards=(Card(name="Source", revisions=(revision,)), *destinations))
    loaded = Stack.model_validate_json(stack.model_dump_json())
    assert loaded.cards[0].active_revision.references == references
    assert UnresolvedCardReference(target_name="  ").target_name is None
    assert UnresolvedCardReference(target_name="Former room").target_name == "Former room"
    with pytest.raises(ValidationError):
        NavigateAction.model_validate(
            {"target": {"type": "resolved", "target_card_id": None, "target_name": "Mixed state"}}
        )
    with pytest.raises(ValidationError, match="at most 2"):
        CardRevision(references=(*references, UnresolvedCardReference()))
    with pytest.raises(ValidationError, match="same card twice"):
        Stack(
            name="Invalid",
            cards=(
                Card(name="Source", revisions=(CardRevision(references=(references[0],) * 2),)),
                *destinations,
            ),
        )
    with pytest.raises(ValidationError, match="own card"):
        Stack(
            name="Invalid",
            cards=(
                Card(id=destinations[0].id, name="Source", revisions=(revision,)),
                destinations[1],
            ),
        )
    with pytest.raises(ValidationError, match="must identify a card"):
        Stack(name="Invalid", cards=(stack.cards[0],))


def test_hotspot_sets_belong_to_revisions_and_distinguish_empty_from_unapplied() -> None:
    hotspot = Interaction(polygons=(hotspot_polygon(),))
    shared = HotspotSet(interactions=(hotspot,))
    first, second, blank = (
        CardRevision(hotspot_set=shared),
        CardRevision(hotspot_set=shared),
        CardRevision(),
    )
    assert first.hotspot_set is not shared
    assert first.hotspot_set is not second.hotspot_set
    assert blank.hotspot_set is None
    assert CardRevision(hotspot_set=HotspotSet()).hotspot_set.interactions == ()
    stack = Stack(name="Stack", cards=(Card(name="Card", revisions=(first, second, blank)),))
    assert Stack.model_validate_json(stack.model_dump_json()) == stack
    with pytest.raises(ValidationError, match="interaction IDs"):
        HotspotSet(interactions=(hotspot, hotspot))


@pytest.mark.parametrize("count", (0, 2))
def test_hotspots_require_exactly_one_polygon(count: int) -> None:
    with pytest.raises(ValidationError):
        Interaction(polygons=(hotspot_polygon(),) * count)


def test_hotspot_labels_follow_navigation_sound_gain_lose_priority() -> None:
    destination = Card(name="Gate")
    red, opened = KeyDefinition(name="Red key"), KeyDefinition(name="Door open")
    sound = SoundDefinition(name="Chime")
    changes = HotspotKeyChanges(remove=(red.id,), grant=(opened.id,))
    rows = (
        Interaction(
            action=NavigateAction(target=ResolvedCardReference(target_card_id=destination.id)),
            sound_id=sound.id,
            key_changes=changes,
            polygons=(hotspot_polygon(),),
        ),
        Interaction(sound_id=sound.id, key_changes=changes, polygons=(hotspot_polygon(),)),
        Interaction(key_changes=changes, polygons=(hotspot_polygon(),)),
        Interaction(key_changes=HotspotKeyChanges(remove=(red.id,)), polygons=(hotspot_polygon(),)),
        Interaction(
            action=NavigateAction(target=UnresolvedCardReference(target_name="Former room")),
            polygons=(hotspot_polygon(),),
        ),
        Interaction(polygons=(hotspot_polygon(),)),
    )
    stack = Stack(
        name="Stack",
        keys=(red, opened),
        sounds=(sound,),
        cards=(
            Card(
                name="Source", revisions=(CardRevision(hotspot_set=HotspotSet(interactions=rows)),)
            ),
            destination,
        ),
    )
    assert [row.label for row in stack.cards[0].active_revision.hotspot_set.interactions] == [
        "Go to Gate",
        "Play Chime",
        "Gain Door open",
        "Lose Red key",
        "Go to Former room",
        "New Hotspot",
    ]


def test_key_contract_is_closed_and_references_stack_keys() -> None:
    key = KeyDefinition(name=" Red key ")
    hotspot = Interaction(
        conditions=HotspotConditions(requires=(key.id,)),
        key_changes=HotspotKeyChanges(remove=(key.id,)),
        polygons=(hotspot_polygon(),),
    )
    card = Card(
        name="Card", revisions=(CardRevision(hotspot_set=HotspotSet(interactions=(hotspot,))),)
    )
    stack = Stack(name="Stack", keys=(key,), cards=(card,))
    assert stack.key_by_id(key.id).name == "Red key"
    with pytest.raises(ValidationError, match="both have and lack"):
        HotspotConditions(requires=(key.id,), forbids=(key.id,))
    with pytest.raises(ValidationError, match="both gained and lost"):
        HotspotKeyChanges(remove=(key.id,), grant=(key.id,))
    with pytest.raises(ValidationError, match="Key names"):
        Stack(name="Stack", keys=(key, KeyDefinition(name="red KEY")))
    with pytest.raises(ValidationError, match="Key IDs"):
        Stack(name="Stack", keys=(key, key))
    with pytest.raises(ValidationError, match="hotspot key references"):
        Stack(name="Stack", cards=(card,))
    for cls, role in (
        (HotspotConditions, "requires"),
        (HotspotConditions, "forbids"),
        (HotspotKeyChanges, "remove"),
        (HotspotKeyChanges, "grant"),
    ):
        with pytest.raises(ValidationError, match="only once"):
            cls(**{role: (key.id, key.id)})


def test_generate_inputs_capture_exact_ordered_reference_and_style_snapshots() -> None:
    references = tuple(
        ImageReferenceSnapshot(card_id=uuid4(), revision_id=uuid4(), background_id=uuid4())
        for _ in range(2)
    )
    style = StyleSnapshot(
        style_id=HYPERCARD_STYLE_ID, name="Ink", prompt_text="Exact ink treatment"
    )
    inputs = GenerateInputs(description="Portrait at dusk", references=references, style=style)
    assert GenerateInputs.model_validate_json(inputs.model_dump_json()) == inputs
    assert inputs.references == references
    assert inputs.style == style
    with pytest.raises(ValidationError, match="nonempty Description"):
        GenerateInputs(description=" \n ")


def test_revision_output_size_is_local_typed_and_aspect_compatible() -> None:
    default = CardRevision()
    assert default.generate_output_size == PresetOutputSize(tier=ResolutionTier.MEDIUM)
    assert default.model_dump(mode="json")["generate_output_size"] == {
        "mode": "preset",
        "tier": 512,
    }
    exact = CardRevision(generate_output_size=ExactOutputSize(width=592, height=448))
    Stack(name="Stack", cards=(Card(name="Card", revisions=(default, exact)),))
    with pytest.raises(ValidationError, match="generate_output_size"):
        CardRevision.model_validate({"generate_output_size": {"mode": "preset", "tier": 300}})
    with pytest.raises(ValidationError, match="aligned"):
        ExactOutputSize(width=641, height=480)
    with pytest.raises(ValidationError, match="stack aspect ratio"):
        Stack(
            name="Invalid",
            cards=(
                Card(
                    name="Card",
                    revisions=(
                        CardRevision(generate_output_size=ExactOutputSize(width=640, height=496)),
                    ),
                ),
            ),
        )


@pytest.mark.parametrize("aspect_ratio", tuple(AspectRatio))
@pytest.mark.parametrize("tier", tuple(ResolutionTier))
def test_generate_dimensions_match_every_format_and_tier(
    aspect_ratio: AspectRatio, tier: ResolutionTier
) -> None:
    width, height = output_dimensions(tier, aspect_ratio)
    provenance = generate_provenance(width, height, PresetOutputSize(tier=tier))
    stack = image_stack(provenance, aspect_ratio)
    assert Stack.model_validate_json(stack.model_dump_json()) == stack
    invalid = provenance.model_copy(
        update={
            "origin": provenance.origin.model_copy(
                update={"settings": provenance.settings.model_copy(update={"width": width + 16})}
            )
        }
    )
    with pytest.raises(ValidationError, match="direct Generate dimensions"):
        image_stack(invalid, aspect_ratio)


@pytest.mark.parametrize("duplicate", (False, True))
@pytest.mark.parametrize(("width", "height", "valid"), ((592, 448, True), (640, 496, False)))
def test_exact_generate_receipts_require_stack_compatible_dimensions(
    duplicate: bool, width: int, height: int, valid: bool
) -> None:
    provenance = generate_provenance(width, height, ExactOutputSize(width=width, height=height))
    if duplicate:
        provenance = duplicate_provenance(provenance)
    if valid:
        assert image_stack(provenance).cards[0].active_revision.provenance == provenance
    else:
        with pytest.raises(ValidationError, match="stack aspect ratio"):
            image_stack(provenance)


@pytest.mark.parametrize("kind", ("neutral", "generate", "edit"))
@pytest.mark.parametrize("duplicate", (False, True))
def test_all_images_share_origin_facts_and_strict_optional_authoring(
    kind: str, duplicate: bool
) -> None:
    provenance = {
        "neutral": ImageProvenance(
            origin=ImageOriginFacts(
                render_prompt="Exact prompt\n\nKeep  authored spacing.",
                settings=image_settings(1007, 783),
                edit_lineage=(
                    AcceptedEdit(instruction="Open gate.", expanded_prompt="Open gate. Ink."),
                ),
            )
        ),
        "generate": generate_provenance(),
        "edit": edit_provenance(),
    }[kind]
    if duplicate:
        provenance = duplicate_provenance(provenance)
    stack = image_stack(provenance)
    loaded = Stack.model_validate_json(stack.model_dump_json())
    assert loaded == stack
    assert image_operation_settings(provenance) == provenance.origin.settings
    assert image_edit_lineage(provenance) == provenance.origin.edit_lineage
    original = original_image_provenance(provenance)
    assert original.origin == provenance.origin
    if kind == "neutral":
        assert original.authoring is None
        assert original.origin.settings.width == 1007


def test_authoring_receipts_reject_unknown_operations_and_mixed_fields() -> None:
    payload = generate_provenance().model_dump(mode="python")
    payload["authoring"]["operation"] = "unknown"
    with pytest.raises(ValidationError, match="union_tag_invalid"):
        ImageProvenance.model_validate(payload)
    payload = generate_provenance().model_dump(mode="python")
    payload["authoring"]["instruction"] = "Not a Generate input"
    with pytest.raises(ValidationError, match="extra_forbidden"):
        ImageProvenance.model_validate(payload)
    duplicate = duplicate_provenance(edit_provenance())
    payload = duplicate.model_dump(mode="python")
    payload["authoring"]["original_authoring"] = payload["authoring"].copy()
    with pytest.raises(ValidationError, match="union_tag_invalid"):
        ImageProvenance.model_validate(payload)


def test_generate_resets_lineage_and_edit_requires_exact_accepted_final_prompt() -> None:
    payload = generate_provenance().model_dump(mode="python")
    payload["origin"]["edit_lineage"] = edit_provenance().origin.edit_lineage
    with pytest.raises(ValidationError, match="empty Edit lineage"):
        ImageProvenance.model_validate(payload)
    payload = edit_provenance().model_dump(mode="python")
    payload["origin"]["edit_lineage"] = ()
    with pytest.raises(ValidationError, match="accepted instruction"):
        ImageProvenance.model_validate(payload)
    payload = edit_provenance().model_dump(mode="python")
    payload["origin"]["render_prompt"] = "Not the accepted prompt"
    with pytest.raises(ValidationError, match="exact origin prompt"):
        ImageProvenance.model_validate(payload)
    with pytest.raises(ValidationError, match="512-token budget"):
        EditOperation(
            source=source_snapshot(),
            output_size=CurrentSourceSize(width=512, height=384),
            prompt_token_count=513,
        )


def test_multistep_edit_and_duplicate_store_one_canonical_ordered_lineage() -> None:
    first = edit_provenance(instruction="Open the gate.")
    duplicate = duplicate_provenance(first)
    second = edit_provenance(inherited=image_edit_lineage(duplicate), instruction="Open the gate.")
    third = edit_provenance(inherited=image_edit_lineage(second), instruction="Light the lanterns.")
    assert [entry.instruction for entry in image_edit_lineage(third)] == [
        "Open the gate.",
        "Open the gate.",
        "Light the lanterns.",
    ]
    payload = third.model_dump(mode="json")
    assert "edit_lineage" not in payload["authoring"]["source"]
    assert "edit_lineage" not in payload["authoring"]
    assert len(payload["origin"]["edit_lineage"]) == 3
    for provenance in (first, duplicate, second, third):
        stack = image_stack(provenance)
        assert Stack.model_validate_json(stack.model_dump_json()) == stack


def test_origin_and_accepted_edit_preserve_exact_text_without_normalization() -> None:
    text = "  Open  the gate.\n"
    provenance = edit_provenance(instruction=text)
    loaded = ImageProvenance.model_validate_json(provenance.model_dump_json())
    assert loaded.origin.render_prompt == text
    assert loaded.origin.edit_lineage[0].instruction == text
    assert loaded.origin.edit_lineage[0].expanded_prompt == text
    with pytest.raises(ValidationError, match="nonempty"):
        ImageOriginFacts(render_prompt=" \n", settings=image_settings())
    with pytest.raises(ValidationError, match="nonempty"):
        AcceptedEdit(instruction=" \n", expanded_prompt="Prompt")


@pytest.mark.parametrize("duplicate", (False, True))
def test_edit_source_attribution_is_independent_of_retained_cards_and_revisions(
    duplicate: bool,
) -> None:
    card_id, revision_id = uuid4(), uuid4()
    source = source_snapshot(card_id=card_id, revision_id=revision_id)
    provenance = edit_provenance(source=source)
    if duplicate:
        provenance = duplicate_provenance(provenance)
    result = background(provenance)
    stack = Stack(
        name="Independent",
        cards=(
            Card(
                id=card_id,
                name="Card",
                revisions=(CardRevision(id=revision_id, background=result),),
            ),
        ),
    )
    assert Stack.model_validate_json(stack.model_dump_json()) == stack
    with pytest.raises(ValidationError, match="identity must differ"):
        GeneratedBackground.model_validate({**result.model_dump(), "id": source.background_id})


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"width": 496}, "match its source dimensions"),
        ({"height": 0}, "greater than 0"),
        ({"seed": True}, "valid integer"),
        ({"image_path": "old.png"}, "extra_forbidden"),
        ({"original_provenance": {}}, "extra_forbidden"),
        ({"edit_lineage": ()}, "extra_forbidden"),
    ],
)
def test_edit_source_facts_are_strict_and_local(change: dict[str, object], message: str) -> None:
    payload = edit_provenance().model_dump(mode="python")
    payload["authoring"]["source"].update(change)
    with pytest.raises(ValidationError, match=message):
        ImageProvenance.model_validate(payload)


@pytest.mark.parametrize(
    "field", ("width", "height", "seed", "card_id", "revision_id", "background_id")
)
def test_edit_source_requires_every_captured_fact(field: str) -> None:
    payload = edit_provenance().model_dump(mode="python")
    del payload["authoring"]["source"][field]
    with pytest.raises(ValidationError, match="Field required"):
        ImageProvenance.model_validate(payload)


@pytest.mark.parametrize(
    ("source_width", "source_height", "tier", "valid"),
    [
        (512, 384, ResolutionTier.MEDIUM, False),
        (1024, 768, ResolutionTier.LARGE, False),
        (503, 391, ResolutionTier.LARGE, True),
        (1024, 192, ResolutionTier.LARGE, True),
    ],
)
def test_edit_presets_use_strictly_larger_source_area(
    source_width: int, source_height: int, tier: ResolutionTier, valid: bool
) -> None:
    width, height = output_dimensions(tier, AspectRatio.LANDSCAPE)
    source = source_snapshot(width=source_width, height=source_height)
    if not valid:
        with pytest.raises(ValidationError, match="more pixels"):
            edit_provenance(
                source=source, width=width, height=height, output_size=PresetOutputSize(tier=tier)
            )
    else:
        provenance = edit_provenance(
            source=source, width=width, height=height, output_size=PresetOutputSize(tier=tier)
        )
        assert Stack.model_validate_json(image_stack(provenance).model_dump_json())


@pytest.mark.parametrize("duplicate", (False, True))
def test_edit_presets_require_correct_stack_dimensions_inside_duplicates(duplicate: bool) -> None:
    invalid = edit_provenance(
        width=768, height=592, output_size=PresetOutputSize(tier=ResolutionTier.LARGE)
    )
    if duplicate:
        invalid = duplicate_provenance(invalid)
    with pytest.raises(ValidationError, match="Edit dimensions must match"):
        image_stack(invalid)


def test_edit_current_dimensions_match_source_but_seed_is_fresh() -> None:
    provenance = edit_provenance(source=source_snapshot(seed=7))
    assert provenance.settings.seed != provenance.authoring.source.seed
    payload = provenance.model_dump(mode="python")
    payload["origin"]["settings"]["width"] = 496
    with pytest.raises(ValidationError, match="selected output size"):
        ImageProvenance.model_validate(payload)
    with pytest.raises(ValidationError, match="aligned"):
        CurrentSourceSize(width=503, height=391)


@pytest.mark.parametrize(
    ("field", "value"),
    [("duration_seconds", float("inf")), ("guidance", float("nan"))],
)
@pytest.mark.parametrize("serializer", ("model_dump", "model_dump_json"))
def test_invalid_copied_nested_metadata_is_rejected_at_serialization(
    field: str, value: float, serializer: str
) -> None:
    provenance = generate_provenance()
    invalid_settings = provenance.settings.model_copy(update={field: value})
    invalid = provenance.model_copy(
        update={"origin": provenance.origin.model_copy(update={"settings": invalid_settings})}
    )
    stack = image_stack(invalid)
    for value_to_serialize in (invalid_settings, stack):
        with pytest.raises(ValidationError, match="finite"):
            getattr(value_to_serialize, serializer)()


def test_card_revision_and_start_identities_must_be_unique_and_known() -> None:
    first = Card(name="First")
    assert first.active_revision_id == first.revisions[0].id
    with pytest.raises(ValidationError, match="active_revision_id"):
        Card(name="Invalid", revisions=first.revisions, active_revision_id=uuid4())
    with pytest.raises(ValidationError, match="start_card_id"):
        Stack(name="Invalid", cards=(first,), start_card_id=uuid4())
    with pytest.raises(ValidationError, match="card IDs"):
        Stack(name="Invalid", cards=(first, first))
    with pytest.raises(ValidationError, match="revision IDs"):
        Card(name="Invalid", revisions=first.revisions * 2)
    with pytest.raises(ValidationError, match="revision IDs"):
        Stack(name="Invalid", cards=(first, Card(name="Second", revisions=first.revisions)))
    with pytest.raises(ValidationError, match="must identify a card"):
        Stack(
            name="Invalid",
            cards=(
                Card(
                    name="Card",
                    revisions=(
                        CardRevision(
                            hotspot_set=HotspotSet(
                                interactions=(
                                    Interaction(
                                        action=NavigateAction(
                                            target=ResolvedCardReference(target_card_id=uuid4())
                                        ),
                                        polygons=(hotspot_polygon(),),
                                    ),
                                )
                            ),
                        ),
                    ),
                ),
            ),
        )


@pytest.mark.parametrize(
    ("model", "payload"),
    [
        (Stack, {"name": "Stack", "database_id": 7}),
        (CanvasSize, {"width": "1024", "height": "768"}),
        (GenerateInputs, {"description": "Scene", "image_prompt": "Other"}),
        (CardRevision, {"image_prompt": {"text": "Other"}}),
        (NavigateAction, {"type": "external_url", "target": {"type": "unresolved"}}),
        (GeneratedBackground, {"image_path": "image.png", "created_at": datetime.now(UTC)}),
    ],
)
def test_unknown_fields_coercion_and_missing_metadata_are_rejected(
    model: type, payload: dict[str, object]
) -> None:
    with pytest.raises(ValidationError):
        model.model_validate(payload)
