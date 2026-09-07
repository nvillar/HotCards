"""Focused tests for typed document commands."""

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from hotcards.application.commands import (
    ActivateRevisionCommand,
    AddInteractionCommand,
    AddKeyCommand,
    AddSoundCommand,
    AddStyleCommand,
    ApplyEditResultCommand,
    ChangeHotspotDestinationCommand,
    ChangeHotspotSoundCommand,
    CommandError,
    CreateCardCommand,
    CreateImageRevisionCommand,
    CreateKeyAndAddHotspotReferenceCommand,
    DeleteCardCommand,
    DeleteInteractionCommand,
    DeleteKeyCommand,
    DeleteRevisionCommand,
    DeleteSoundCommand,
    DeleteStyleCommand,
    DuplicateCardCommand,
    DuplicateRevisionCommand,
    EditRevisionDescriptionCommand,
    RenameCardCommand,
    RenameKeyCommand,
    ReorderCardCommand,
    ReorderHotspotCommand,
    ReplaceGeneratedSoundCommand,
    ReplaceHotspotSetCommand,
    ReplacePolygonCommand,
    ReplaceRevisionBackgroundCommand,
    SetHotspotConditionsCommand,
    SetHotspotKeyChangesCommand,
    SetRevisionGenerateOutputSizeCommand,
    SetRevisionReferenceCommand,
    SetRevisionStyleCommand,
    SetStartCardCommand,
    UpdateSoundCommand,
    UpdateStyleCommand,
    next_duplicate_card_name,
)
from hotcards.application.document_controller import DocumentController
from hotcards.domain.image_dimensions import AspectRatio, ResolutionTier
from hotcards.domain.models import (
    Card,
    CardRevision,
    CurrentSourceSize,
    DerivedImageSourceSnapshot,
    DirectGenerateProvenance,
    DuplicateProvenance,
    EditDraft,
    GeneratedBackground,
    GeneratedSoundAsset,
    GenerateInputs,
    HotspotConditions,
    HotspotKeyChanges,
    HotspotSet,
    ImageOperationSettings,
    ImageSourceSnapshot,
    Interaction,
    KeyDefinition,
    NavigateAction,
    Point,
    Polygon,
    PresetOutputSize,
    RefineProvenance,
    RefineTransformation,
    ResolvedCardReference,
    SoundGenerationProvenance,
    Stack,
    StyleDefinition,
    UnresolvedCardReference,
    image_edit_lineage,
    image_operation_settings,
)


def polygon(offset: float = 0.0) -> Polygon:
    return Polygon(
        points=(
            Point(x=0.1 + offset, y=0.1),
            Point(x=0.3 + offset, y=0.1),
            Point(x=0.2 + offset, y=0.4),
        )
    )


def interaction(label: str, target_name: str) -> Interaction:
    return Interaction(
        label=label,
        action=NavigateAction(target=UnresolvedCardReference(target_name=target_name)),
        polygons=(polygon(),),
    )


def card_with_revision(*interactions: Interaction) -> tuple[Card, CardRevision]:
    revision = CardRevision(
        hotspot_set=HotspotSet(interactions=interactions),
    )
    return (
        Card(
            name="Source",
            revisions=(revision,),
            active_revision_id=revision.id,
        ),
        revision,
    )


def operation_settings() -> ImageOperationSettings:
    return ImageOperationSettings(
        model_identifier="test",
        mflux_version="test",
        seed=1,
        width=512,
        height=384,
        step_count=4,
        generated_at=datetime.now(UTC),
        duration_seconds=1,
    )


def generated_background(description: str = "A card") -> GeneratedBackground:
    asset_id = uuid4()
    return GeneratedBackground(
        id=asset_id,
        image_path=f"assets/cards/card/image-{asset_id}.png",
        provenance=DirectGenerateProvenance(
            inputs=GenerateInputs(description=description),
            render_prompt=description,
            settings=operation_settings(),
        ),
        created_at=datetime.now(UTC),
    )


def refined_background(
    *,
    source_card_id: UUID,
    source_revision: CardRevision,
) -> GeneratedBackground:
    assert source_revision.background is not None
    asset_id = uuid4()
    return GeneratedBackground(
        id=asset_id,
        image_path=f"assets/cards/card/image-{asset_id}.png",
        provenance=RefineProvenance(
            source=DerivedImageSourceSnapshot(
                card_id=source_card_id,
                revision_id=source_revision.id,
                background_id=source_revision.background.id,
                width=512,
                height=384,
                seed=image_operation_settings(source_revision.background.provenance).seed,
                edit_lineage=image_edit_lineage(source_revision.background.provenance),
            ),
            description="Refined card",
            render_prompt="Refined card",
            output_size=CurrentSourceSize(width=512, height=384),
            transformation=RefineTransformation.BALANCED,
            strength=0.5,
            settings=operation_settings(),
        ),
        created_at=datetime.now(UTC),
    )


def test_card_create_rename_reorder_and_start_selection() -> None:
    first = Card(name="First")
    document = Stack(name="Stack", cards=(first,))
    created_id = uuid4()

    document = CreateCardCommand(
        name="Second",
        card_id=created_id,
        index=0,
    ).apply(document)
    assert [card.name for card in document.cards] == ["Second", "First"]

    document = RenameCardCommand(card_id=created_id, name="Renamed").apply(document)
    document = ReorderCardCommand(card_id=created_id, new_index=1).apply(document)
    document = SetStartCardCommand(card_id=created_id).apply(document)

    assert [card.name for card in document.cards] == ["First", "Renamed"]
    assert document.start_card_id == created_id
    assert SetStartCardCommand(card_id=None).apply(document).start_card_id is None


def test_duplicate_card_copies_only_active_revision_with_independent_ids() -> None:
    key = KeyDefinition(name="Key")
    style = StyleDefinition(name="Style", prompt_text="Treatment")
    destination = Card(name="Destination")
    inactive = CardRevision(description="Inactive")
    background = generated_background("Active")
    source_id = uuid4()
    self_interaction = Interaction(
        conditions=HotspotConditions(requires=(key.id,)),
        key_changes=HotspotKeyChanges(grant=(key.id,)),
        action=NavigateAction(target=ResolvedCardReference(target_card_id=source_id)),
        polygons=(polygon(),),
    )
    external_interaction = Interaction(
        action=NavigateAction(target=ResolvedCardReference(target_card_id=destination.id)),
        polygons=(polygon(0.2),),
    )
    active = CardRevision(
        description="Active",
        background=background,
        hotspot_set=HotspotSet(interactions=(self_interaction, external_interaction)),
        references=(ResolvedCardReference(target_card_id=destination.id),),
        style_id=style.id,
        generate_output_size=PresetOutputSize(tier=ResolutionTier.FULL),
        edit_draft=EditDraft(instruction="Unfinished duplicate source"),
    )
    source = Card(
        id=source_id,
        name="Source",
        revisions=(inactive, active),
        active_revision_id=active.id,
    )
    document = Stack(
        name="Stack",
        styles=(style,),
        new_card_style_id=style.id,
        keys=(key,),
        cards=(source, destination),
    )
    duplicate_card_id = uuid4()
    duplicate_revision_id = uuid4()
    duplicate_background_id = uuid4()
    duplicate_interaction_ids = (uuid4(), uuid4())

    changed = DuplicateCardCommand(
        source_card_id=source.id,
        name="Source Copy",
        card_id=duplicate_card_id,
        revision_id=duplicate_revision_id,
        background_id=duplicate_background_id,
        background_image_path=(
            f"assets/cards/{duplicate_card_id}/image-{duplicate_background_id}.png"
        ),
        interaction_ids=duplicate_interaction_ids,
    ).apply(document)

    assert [card.name for card in changed.cards] == [
        "Source",
        "Source Copy",
        "Destination",
    ]
    duplicate = changed.cards[1]
    assert duplicate.id == duplicate_card_id
    assert len(duplicate.revisions) == 1
    revision = duplicate.active_revision
    assert revision.id == duplicate_revision_id
    assert revision.description == active.description
    assert revision.references == active.references
    assert revision.style_id == style.id
    assert revision.generate_output_size == PresetOutputSize(tier=ResolutionTier.FULL)
    assert revision.edit_draft.instruction == ""
    assert revision.edit_draft.generation_id != active.edit_draft.generation_id
    assert revision.hotspot_set is not None
    assert (
        tuple(copied.id for copied in revision.hotspot_set.interactions)
        == duplicate_interaction_ids
    )
    copied_self, copied_external = revision.hotspot_set.interactions
    assert copied_self.conditions == self_interaction.conditions
    assert copied_self.key_changes == self_interaction.key_changes
    assert copied_self.polygons == self_interaction.polygons
    assert copied_self.action == NavigateAction(
        target=ResolvedCardReference(target_card_id=duplicate.id)
    )
    assert copied_external.action == external_interaction.action
    assert revision.background is not None
    assert revision.background.id == duplicate_background_id
    assert revision.background.id != background.id
    assert isinstance(revision.background.provenance, DuplicateProvenance)
    assert revision.background.provenance.source == ImageSourceSnapshot(
        card_id=source.id,
        revision_id=active.id,
        background_id=background.id,
    )
    assert revision.background.provenance.original_provenance == background.provenance


def test_duplicate_card_name_is_case_insensitively_unique() -> None:
    source = Card(name="Scene")
    document = Stack(
        name="Stack",
        cards=(
            source,
            Card(name="scene copy"),
            Card(name="SCENE COPY 2"),
        ),
    )

    assert next_duplicate_card_name(document, source.name) == "Scene Copy 3"


def test_deleting_blank_duplicate_does_not_change_source() -> None:
    source = Card(name="Source")
    document = Stack(name="Stack", cards=(source,))
    duplicate = DuplicateCardCommand(
        source_card_id=source.id,
        name="Source Copy",
    ).apply(document)

    changed = DeleteCardCommand(card_id=duplicate.cards[1].id).apply(duplicate)

    assert changed.cards == (source,)


def test_duplicate_card_flattens_provenance_and_survives_source_deletion() -> None:
    source_card_id = uuid4()
    direct = CardRevision(background=generated_background("Direct"))
    refined = CardRevision(
        background=refined_background(
            source_card_id=source_card_id,
            source_revision=direct,
        )
    )
    source = Card(
        id=source_card_id,
        name="Source",
        revisions=(direct, refined),
        active_revision_id=refined.id,
    )
    document = Stack(name="Stack", cards=(source,))
    first_card_id, first_revision_id, first_background_id = (
        uuid4(),
        uuid4(),
        uuid4(),
    )
    first = DuplicateCardCommand(
        source_card_id=source.id,
        name="Source Copy",
        card_id=first_card_id,
        revision_id=first_revision_id,
        background_id=first_background_id,
        background_image_path=(f"assets/cards/{first_card_id}/image-{first_background_id}.png"),
    ).apply(document)
    first_duplicate = first.cards[1]
    second_card_id, second_revision_id, second_background_id = (
        uuid4(),
        uuid4(),
        uuid4(),
    )
    changed = DuplicateCardCommand(
        source_card_id=first_duplicate.id,
        name="Source Copy Copy",
        card_id=second_card_id,
        revision_id=second_revision_id,
        background_id=second_background_id,
        background_image_path=(f"assets/cards/{second_card_id}/image-{second_background_id}.png"),
    ).apply(first)

    first_provenance = first_duplicate.active_revision.provenance
    second_provenance = changed.cards[2].active_revision.provenance
    assert isinstance(first_provenance, DuplicateProvenance)
    assert isinstance(second_provenance, DuplicateProvenance)
    assert isinstance(second_provenance.original_provenance, RefineProvenance)
    assert second_provenance.original_provenance == first_provenance.original_provenance
    assert second_provenance.source.card_id == first_duplicate.id

    without_source = DeleteCardCommand(card_id=source.id).apply(changed)
    assert [card.name for card in without_source.cards] == [
        "Source Copy",
        "Source Copy Copy",
    ]


def test_commands_preserve_immutable_stack_aspect_ratio() -> None:
    document = Stack(name="Stack", aspect_ratio=AspectRatio.PORTRAIT)

    changed = CreateCardCommand(name="Card").apply(document)
    changed = RenameCardCommand(
        card_id=changed.cards[0].id,
        name="Renamed",
    ).apply(changed)

    assert changed.aspect_ratio is AspectRatio.PORTRAIT
    assert changed.canvas == document.canvas


def test_style_lifecycle_and_new_card_default_are_typed_changes() -> None:
    original = StyleDefinition(name="Original", prompt_text="Original treatment")
    document = Stack(
        name="Stack",
        styles=(original,),
        new_card_style_id=original.id,
    )
    created_style_id = uuid4()

    document = AddStyleCommand(
        name="Custom",
        prompt_text="Custom treatment",
        style_id=created_style_id,
    ).apply(document)
    document = UpdateStyleCommand(
        style_id=created_style_id,
        name="Edited",
        prompt_text="Edited treatment",
    ).apply(document)
    document = CreateCardCommand(name="First").apply(document)
    first = document.cards[0]
    assert first.active_revision.style_id == original.id

    document = SetRevisionStyleCommand(
        card_id=first.id,
        revision_id=first.active_revision.id,
        style_id=created_style_id,
    ).apply(document)
    document = CreateCardCommand(name="Second").apply(document)
    assert document.cards[0].active_revision.style_id == created_style_id
    assert document.cards[1].active_revision.style_id == created_style_id
    assert document.new_card_style_id == created_style_id

    document = DeleteStyleCommand(style_id=created_style_id).apply(document)
    assert [style.name for style in document.styles] == ["Original"]
    assert document.new_card_style_id is None
    assert all(card.active_revision.style_id is None for card in document.cards)


def test_key_lifecycle_and_hotspot_behavior_are_typed_changes() -> None:
    key_id = uuid4()
    interaction = Interaction(polygons=(polygon(),))
    source, revision = card_with_revision(interaction)
    document = Stack(name="Stack", cards=(source,))

    document = AddKeyCommand(name="Red key", key_id=key_id).apply(document)
    document = RenameKeyCommand(key_id=key_id, name="Ruby key").apply(document)
    document = SetHotspotConditionsCommand(
        card_id=source.id,
        revision_id=revision.id,
        interaction_id=interaction.id,
        conditions=HotspotConditions(forbids=(key_id,)),
    ).apply(document)
    document = SetHotspotKeyChangesCommand(
        card_id=source.id,
        revision_id=revision.id,
        interaction_id=interaction.id,
        key_changes=HotspotKeyChanges(grant=(key_id,)),
    ).apply(document)

    changed = document.cards[0].active_revision.hotspot_set
    assert changed is not None
    assert document.keys == (KeyDefinition(id=key_id, name="Ruby key"),)
    assert changed.interactions[0].label == "Gain Ruby key"
    assert changed.interactions[0].conditions.forbids == (key_id,)
    assert changed.interactions[0].key_changes.grant == (key_id,)
    with pytest.raises(CommandError, match="still used"):
        DeleteKeyCommand(key_id=key_id).apply(document)

    document = SetHotspotConditionsCommand(
        card_id=source.id,
        revision_id=revision.id,
        interaction_id=interaction.id,
        conditions=HotspotConditions(),
    ).apply(document)
    document = SetHotspotKeyChangesCommand(
        card_id=source.id,
        revision_id=revision.id,
        interaction_id=interaction.id,
        key_changes=HotspotKeyChanges(),
    ).apply(document)
    document = DeleteKeyCommand(key_id=key_id).apply(document)
    assert document.keys == ()


def test_sound_lifecycle_and_hotspot_assignment_are_typed_changes() -> None:
    sound_id = uuid4()
    interaction = Interaction(polygons=(polygon(),))
    source, revision = card_with_revision(interaction)
    document = Stack(name="Stack", cards=(source,))

    document = AddSoundCommand(
        name="Door",
        prompt="Heavy wooden door closes",
        sound_id=sound_id,
    ).apply(document)
    document = UpdateSoundCommand(
        sound_id=sound_id,
        name="Door close",
        prompt="A dry wooden door slam",
        duration_seconds=3,
    ).apply(document)
    document = ChangeHotspotSoundCommand(
        card_id=source.id,
        revision_id=revision.id,
        interaction_id=interaction.id,
        sound_id=sound_id,
    ).apply(document)
    generated = GeneratedSoundAsset(
        audio_path=f"assets/sounds/{sound_id}/sound-{uuid4()}.wav",
        provenance=SoundGenerationProvenance(
            prompt="A dry wooden door slam",
            duration_seconds=3,
            seed=42,
            generation_duration_milliseconds=900,
        ),
        created_at=datetime.now(UTC),
    )
    document = ReplaceGeneratedSoundCommand(
        sound_id=sound_id,
        generated=generated,
    ).apply(document)

    sound = document.sound_by_id(sound_id)
    assert sound.name == "Door close"
    assert sound.duration_seconds == 3
    assert sound.generated == generated
    hotspot_set = document.cards[0].active_revision.hotspot_set
    assert hotspot_set is not None
    assert hotspot_set.interactions[0].sound_id == sound_id
    with pytest.raises(CommandError, match="still used"):
        DeleteSoundCommand(sound_id=sound_id).apply(document)

    document = ChangeHotspotSoundCommand(
        card_id=source.id,
        revision_id=revision.id,
        interaction_id=interaction.id,
        sound_id=None,
    ).apply(document)
    document = DeleteSoundCommand(sound_id=sound_id).apply(document)
    assert document.sounds == ()


def test_create_key_and_hotspot_reference_is_atomic() -> None:
    interaction = Interaction(polygons=(polygon(),))
    source, revision = card_with_revision(interaction)
    document = Stack(name="Stack", cards=(source,))
    command = CreateKeyAndAddHotspotReferenceCommand(
        card_id=source.id,
        revision_id=revision.id,
        interaction_id=interaction.id,
        name="Visited castle",
        role="grant",
    )

    changed = command.apply(document)

    assert changed.keys == (KeyDefinition(id=command.key_id, name="Visited castle"),)
    hotspot_set = changed.cards[0].active_revision.hotspot_set
    assert hotspot_set is not None
    assert hotspot_set.interactions[0].key_changes.grant == (command.key_id,)


def test_revision_generate_output_size_is_typed_and_copied_completely() -> None:
    style = StyleDefinition(name="Ink", prompt_text="Rendered in ink")
    revision = CardRevision(
        style_id=style.id,
        generate_output_size=PresetOutputSize(tier=ResolutionTier.LARGE),
        edit_draft=EditDraft(instruction="Do not duplicate"),
    )
    card = Card(name="Card", revisions=(revision,))
    document = Stack(
        name="Stack",
        styles=(style,),
        new_card_style_id=style.id,
        cards=(card,),
    )

    changed = SetRevisionGenerateOutputSizeCommand(
        card_id=card.id,
        revision_id=revision.id,
        output_size=PresetOutputSize(tier=ResolutionTier.FULL),
    ).apply(document)
    changed = DuplicateRevisionCommand(
        card_id=card.id,
        source_revision_id=revision.id,
    ).apply(changed)

    assert changed.cards[0].active_revision.style_id == style.id
    assert changed.cards[0].active_revision.generate_output_size == PresetOutputSize(
        tier=ResolutionTier.FULL
    )
    assert changed.cards[0].active_revision.edit_draft.instruction == ""
    assert (
        changed.cards[0].active_revision.edit_draft.generation_id
        != revision.edit_draft.generation_id
    )


def test_create_image_revision_preserves_resolution_and_uses_stable_command_ids() -> None:
    previous = CardRevision(
        description="Before",
        generate_output_size=PresetOutputSize(tier=ResolutionTier.LARGE),
    )
    generated = previous.model_copy(
        update={
            "description": "Generated",
            "background": generated_background("Generated"),
        }
    )
    card = Card(
        name="Card",
        revisions=(generated,),
        active_revision_id=generated.id,
    )
    document = Stack(name="Stack", cards=(card,))

    command = CreateImageRevisionCommand(
        card_id=card.id,
        revision_id=generated.id,
        previous_revision=previous,
    )
    changed = command.apply(document)
    assert changed == command.apply(document)
    assert changed.cards[0].revisions[0] == previous
    assert changed.cards[0].active_revision == generated.model_copy(
        update={
            "id": command.new_revision_id,
            "edit_draft": generated.edit_draft.model_copy(
                update={"generation_id": command.new_edit_draft_generation_id}
            ),
        }
    )

    assert tuple(revision.generate_output_size for revision in changed.cards[0].revisions) == (
        PresetOutputSize(tier=ResolutionTier.LARGE),
        PresetOutputSize(tier=ResolutionTier.LARGE),
    )


def test_create_image_revision_keeps_newer_draft_on_original_and_starts_result_empty() -> None:
    previous = CardRevision(edit_draft=EditDraft(instruction="Submitted instruction"))
    newer_draft = EditDraft(instruction="A newer unfinished instruction")
    result = previous.model_copy(
        update={
            "background": generated_background("Result"),
            "edit_draft": newer_draft,
        }
    )
    card = Card(name="Card", revisions=(result,), active_revision_id=result.id)
    command = CreateImageRevisionCommand(
        card_id=card.id,
        revision_id=result.id,
        previous_revision=previous,
    )

    changed = command.apply(Stack(name="Stack", cards=(card,)))

    restored, versioned_result = changed.cards[0].revisions
    assert restored.edit_draft == newer_draft
    assert versioned_result.edit_draft.instruction == ""
    assert versioned_result.edit_draft.generation_id == command.new_edit_draft_generation_id


@pytest.mark.parametrize("hotspot_set", (None, HotspotSet()))
def test_replace_refined_background_preserves_complete_revision(
    hotspot_set: HotspotSet | None,
) -> None:
    style = StyleDefinition(name="Ink", prompt_text="Rendered in ink")
    reference = Card(name="Reference")
    source = CardRevision(
        description="Before",
        hotspot_set=hotspot_set,
        references=(ResolvedCardReference(target_card_id=reference.id),),
        style_id=style.id,
        generate_output_size=PresetOutputSize(tier=ResolutionTier.LARGE),
        background=generated_background("Source"),
    )
    card = Card(name="Card", revisions=(source,))
    document = Stack(
        name="Stack",
        styles=(style,),
        new_card_style_id=style.id,
        cards=(card, reference),
    )
    new_background = refined_background(
        source_card_id=card.id,
        source_revision=source,
    )
    command = ReplaceRevisionBackgroundCommand(
        card_id=card.id,
        revision_id=source.id,
        background=new_background,
    )

    changed = command.apply(document)

    changed_card = changed.cards[0]
    assert len(changed_card.revisions) == 1
    assert changed_card.active_revision.id == source.id
    assert changed_card.active_revision.description == source.description
    assert changed_card.active_revision.style_id == source.style_id
    assert changed_card.active_revision.references == source.references
    assert changed_card.active_revision.hotspot_set == hotspot_set
    assert changed_card.active_revision.generate_output_size == PresetOutputSize(
        tier=ResolutionTier.LARGE
    )
    assert changed_card.active_revision.background == new_background


def test_apply_edit_result_clears_only_the_matching_submitted_draft() -> None:
    submitted = EditDraft(instruction="Open the gate.")
    revision = CardRevision(
        background=generated_background("Source"),
        edit_draft=submitted,
    )
    card = Card(name="Card", revisions=(revision,), active_revision_id=revision.id)
    result = generated_background("Edited")
    command = ApplyEditResultCommand(
        card_id=card.id,
        revision_id=revision.id,
        background=result,
        submitted_draft_generation_id=submitted.generation_id,
    )
    document = Stack(name="Stack", cards=(card,))

    changed = command.apply(document)

    assert changed == command.apply(document)
    assert changed.cards[0].active_revision.background == result
    assert changed.cards[0].active_revision.edit_draft == EditDraft(
        generation_id=command.cleared_draft_generation_id
    )


def test_apply_edit_result_preserves_a_newer_draft_generation() -> None:
    submitted = EditDraft(instruction="Open the gate.")
    newer = EditDraft(instruction="Add ivy.")
    revision = CardRevision(
        background=generated_background("Source"),
        edit_draft=newer,
    )
    card = Card(name="Card", revisions=(revision,), active_revision_id=revision.id)
    result = generated_background("Edited")

    changed = ApplyEditResultCommand(
        card_id=card.id,
        revision_id=revision.id,
        background=result,
        submitted_draft_generation_id=submitted.generation_id,
    ).apply(Stack(name="Stack", cards=(card,)))

    assert changed.cards[0].active_revision.background == result
    assert changed.cards[0].active_revision.edit_draft == newer


def test_revision_description_and_reference_edits_are_typed_changes() -> None:
    card = Card(name="Card")
    reference = Card(name="Reference")
    document = Stack(name="Stack", cards=(card, reference))
    revision_id = card.active_revision_id
    assert revision_id is not None

    document = EditRevisionDescriptionCommand(
        card_id=card.id,
        revision_id=revision_id,
        value="A quiet library",
    ).apply(document)
    document = SetRevisionReferenceCommand(
        card_id=card.id,
        revision_id=revision_id,
        reference=ResolvedCardReference(target_card_id=reference.id),
    ).apply(document)

    changed = document.cards[0].active_revision
    assert changed.description == "A quiet library"
    assert changed.references == (ResolvedCardReference(target_card_id=reference.id),)


def test_revision_activation_and_complete_hotspot_replacement() -> None:
    source, first_revision = card_with_revision(interaction("Door", "Hall"))
    second_revision = CardRevision()
    source = source.model_copy(update={"revisions": (first_revision, second_revision)})
    document = Stack(name="Stack", cards=(source,))
    replacement = HotspotSet(
        interactions=(
            interaction("Window", "Garden"),
            interaction("Stairs", "Tower"),
        )
    )

    document = ActivateRevisionCommand(
        card_id=source.id,
        revision_id=second_revision.id,
    ).apply(document)
    document = ReplaceHotspotSetCommand(
        card_id=source.id,
        revision_id=second_revision.id,
        hotspot_set=replacement,
    ).apply(document)

    changed_revision = document.cards[0].revisions[1]
    assert document.cards[0].active_revision_id == second_revision.id
    assert changed_revision.hotspot_set is not None
    assert [item.id for item in changed_revision.hotspot_set.interactions] == [
        item.id for item in replacement.interactions
    ]
    assert [item.label for item in changed_revision.hotspot_set.interactions] == [
        "Go to Garden",
        "Go to Tower",
    ]
    assert (
        ReplaceHotspotSetCommand(
            card_id=source.id,
            revision_id=second_revision.id,
            hotspot_set=None,
        )
        .apply(document)
        .cards[0]
        .revisions[1]
        .hotspot_set
        is None
    )


def test_duplicate_and_delete_revisions_choose_safe_active_revision() -> None:
    card = Card(name="Card")
    document = Stack(name="Stack", cards=(card,))
    first = card.active_revision
    duplicate = DuplicateRevisionCommand(
        card_id=card.id,
        source_revision_id=first.id,
    )
    document = duplicate.apply(document)
    assert document.cards[0].active_revision_id == duplicate.revision_id
    assert len(document.cards[0].revisions) == 2

    document = DeleteRevisionCommand(
        card_id=card.id,
        revision_id=duplicate.revision_id,
    ).apply(document)
    assert document.cards[0].active_revision_id == first.id
    assert document.cards[0].revisions == (first,)

    with pytest.raises(CommandError, match="at least one revision"):
        DeleteRevisionCommand(
            card_id=card.id,
            revision_id=first.id,
        ).apply(document)


def test_source_revision_deletion_and_replacement_leave_derived_revision_independent() -> None:
    source = CardRevision(background=generated_background("Source"))
    card_id = uuid4()
    derived = CardRevision(
        background=refined_background(
            source_card_id=card_id,
            source_revision=source,
        )
    )
    card = Card(
        id=card_id,
        name="Evolution",
        revisions=(source, derived),
        active_revision_id=derived.id,
    )
    document = Stack(name="Stack", cards=(card,))

    controller = DocumentController(document)
    controller.execute(
        ReplaceRevisionBackgroundCommand(
            card_id=card.id, revision_id=source.id, background=generated_background("Replacement")
        )
    )
    assert controller.document.cards[0].revisions[1] == derived
    controller.execute(DeleteRevisionCommand(card_id=card.id, revision_id=source.id))
    assert controller.document.cards[0].revisions == (derived,)
    assert Stack.model_validate_json(controller.document.model_dump_json()) == controller.document
    controller.undo()
    controller.undo()
    assert controller.document == document


def test_whole_card_deletion_leaves_external_derived_backgrounds_independent() -> None:
    source = CardRevision(background=generated_background("Source"))
    source_card_id = uuid4()
    internal = CardRevision(
        background=refined_background(
            source_card_id=source_card_id,
            source_revision=source,
        )
    )
    source_card = Card(
        id=source_card_id,
        name="Source",
        revisions=(source, internal),
        active_revision_id=internal.id,
    )
    internal_document = Stack(name="Stack", cards=(source_card,))

    assert DeleteCardCommand(card_id=source_card.id).apply(internal_document).cards == ()

    dependent = Card(
        name="Dependent",
        revisions=(
            CardRevision(
                background=refined_background(
                    source_card_id=source_card.id,
                    source_revision=source,
                )
            ),
        ),
    )
    external_document = Stack(
        name="Stack",
        cards=(source_card, dependent),
    )

    deleted = DeleteCardCommand(card_id=source_card.id).apply(external_document)
    assert deleted.cards == (dependent,)
    assert Stack.model_validate_json(deleted.model_dump_json()) == deleted


def test_derived_background_can_replace_its_own_version_and_undo() -> None:
    source = CardRevision(background=generated_background("Source"), description="Source")
    card = Card(name="Card", revisions=(source,))
    controller = DocumentController(Stack(name="Stack", cards=(card,)))
    background = refined_background(source_card_id=card.id, source_revision=source)

    controller.execute(
        ReplaceRevisionBackgroundCommand(
            card_id=card.id, revision_id=source.id, background=background
        )
    )

    changed = controller.document.cards[0].active_revision
    assert changed.id == source.id
    assert len(controller.document.cards[0].revisions) == 1
    assert changed.background == background
    assert changed.background.id != source.background.id
    assert Stack.model_validate_json(controller.document.model_dump_json()) == controller.document
    controller.undo()
    assert controller.document.cards == (card,)
    controller.redo()
    assert controller.document.cards[0].active_revision == changed


def test_polygon_destination_and_hotspot_order_changes() -> None:
    first = interaction("Door", "Hall")
    second = interaction("Window", "Garden")
    source, revision = card_with_revision(first, second)
    destination = Card(name="Hall")
    document = Stack(name="Stack", cards=(source, destination))
    replacement = polygon(0.2)

    document = ReplacePolygonCommand(
        card_id=source.id,
        revision_id=revision.id,
        interaction_id=first.id,
        polygon_index=0,
        polygon=replacement,
    ).apply(document)
    document = ChangeHotspotDestinationCommand(
        card_id=source.id,
        revision_id=revision.id,
        interaction_id=first.id,
        destination=ResolvedCardReference(target_card_id=destination.id),
    ).apply(document)
    document = ReorderHotspotCommand(
        card_id=source.id,
        revision_id=revision.id,
        interaction_id=first.id,
        new_index=1,
    ).apply(document)

    hotspots = document.cards[0].revisions[0].hotspot_set
    assert hotspots is not None
    assert [item.id for item in hotspots.interactions] == [second.id, first.id]
    assert hotspots.interactions[1].polygons == (replacement,)
    assert hotspots.interactions[1].action.target == ResolvedCardReference(
        target_card_id=destination.id
    )
    assert hotspots.interactions[1].label == "Go to Hall"

    renamed = RenameCardCommand(
        card_id=destination.id,
        name="Great Hall",
    ).apply(document)
    renamed_hotspots = renamed.cards[0].active_revision.hotspot_set
    assert renamed_hotspots is not None
    assert renamed_hotspots.interactions[1].label == "Go to Great Hall"


def test_interaction_lifecycle() -> None:
    first = interaction("Door", "Hall")
    second = interaction("Window", "Garden")
    source, revision = card_with_revision(first)
    document = Stack(name="Stack", cards=(source,))
    document = AddInteractionCommand(
        card_id=source.id,
        revision_id=revision.id,
        interaction=second,
    ).apply(document)
    document = DeleteInteractionCommand(
        card_id=source.id,
        revision_id=revision.id,
        interaction_id=second.id,
    ).apply(document)

    hotspot_set = document.cards[0].revisions[0].hotspot_set
    assert hotspot_set is not None
    assert len(hotspot_set.interactions) == 1
    assert hotspot_set.interactions[0].label == "Go to Hall"


def test_targeted_undo_uses_command_identity_not_repeated_document_values() -> None:
    card = Card(name="Original")
    controller = DocumentController(Stack(name="Stack", cards=(card,)))

    controller.execute(RenameCardCommand(card_id=card.id, name="Repeated"))
    old_token = controller.current_undo_token
    assert old_token is not None
    controller.execute(RenameCardCommand(card_id=card.id, name="Different"))
    controller.execute(RenameCardCommand(card_id=card.id, name="Repeated"))

    assert not controller.undo_if_current(old_token)
    current_token = controller.current_undo_token
    assert current_token is not None
    assert controller.undo_if_current(current_token)
    assert controller.document.cards[0].name == "Different"


def test_delete_card_converts_all_inbound_references_and_clears_start() -> None:
    destination = Card(name="Former Hall")
    inbound = interaction("Door", "ignored").model_copy(
        update={
            "action": NavigateAction(target=ResolvedCardReference(target_card_id=destination.id))
        }
    )
    source, _ = card_with_revision(inbound)
    source = source.model_copy(
        update={
            "revisions": (
                source.active_revision.model_copy(
                    update={"references": (ResolvedCardReference(target_card_id=destination.id),)}
                ),
            )
        }
    )
    document = Stack(
        name="Stack",
        cards=(source, destination),
        start_card_id=destination.id,
    )

    changed = DeleteCardCommand(card_id=destination.id).apply(document)

    assert [card.id for card in changed.cards] == [source.id]
    assert changed.start_card_id is None
    hotspot_set = changed.cards[0].revisions[0].hotspot_set
    assert hotspot_set is not None
    target = hotspot_set.interactions[0].action.target
    assert target == UnresolvedCardReference(target_name="Former Hall")
    assert hotspot_set.interactions[0].label == "Go to Former Hall"
    assert changed.cards[0].active_revision.references == (
        UnresolvedCardReference(target_name="Former Hall"),
    )


def test_commands_reject_missing_targets_and_invalid_domain_results() -> None:
    card = Card(name="Card")
    document = Stack(name="Stack", cards=(card,))

    with pytest.raises(CommandError, match="does not exist"):
        RenameCardCommand(card_id=uuid4(), name="Missing").apply(document)
    with pytest.raises(CommandError, match="out of range"):
        ReorderCardCommand(card_id=card.id, new_index=3).apply(document)
    with pytest.raises(ValidationError):
        CreateCardCommand(name=" card ").apply(document)


def test_reference_assignment_and_background_replacement_are_guarded() -> None:
    card = Card(name="Card")
    reference = Card(name="Reference")
    revision_id = card.active_revision_id
    assert revision_id is not None
    document = Stack(name="Stack", cards=(card, reference))
    document = SetRevisionReferenceCommand(
        card_id=card.id,
        revision_id=revision_id,
        reference=ResolvedCardReference(target_card_id=reference.id),
    ).apply(document)

    with pytest.raises(ValidationError, match="own card"):
        SetRevisionReferenceCommand(
            card_id=card.id,
            revision_id=revision_id,
            reference=ResolvedCardReference(target_card_id=card.id),
        ).apply(document)

    asset_id = uuid4()
    generated_at = datetime.now(UTC)
    background = GeneratedBackground(
        id=asset_id,
        image_path=f"assets/cards/{card.id}/image-{asset_id}.png",
        provenance=DirectGenerateProvenance(
            inputs=GenerateInputs(
                description="A card",
            ),
            render_prompt="A card",
            settings=ImageOperationSettings(
                model_identifier="test",
                mflux_version="test",
                seed=1,
                width=512,
                height=384,
                step_count=4,
                generated_at=generated_at,
                duration_seconds=1,
            ),
        ),
        created_at=generated_at,
    )
    document = ReplaceRevisionBackgroundCommand(
        card_id=card.id,
        revision_id=revision_id,
        background=background,
    ).apply(document)
    assert document.cards[0].active_revision.background == background

    document = SetRevisionReferenceCommand(
        card_id=card.id,
        revision_id=revision_id,
        reference=None,
    ).apply(document)
    assert document.cards[0].active_revision.references == ()
