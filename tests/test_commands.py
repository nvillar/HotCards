"""Focused tests for typed document commands."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from hotcards.application.commands import (
    ActivateRevisionCommand,
    AddInteractionCommand,
    AddKeyCommand,
    AddPolygonCommand,
    AddStyleCommand,
    ChangeHotspotDestinationCommand,
    CommandError,
    CreateCardCommand,
    CreateKeyAndAddHotspotReferenceCommand,
    DeleteCardCommand,
    DeleteInteractionCommand,
    DeleteKeyCommand,
    DeletePolygonCommand,
    DeleteRevisionCommand,
    DeleteStyleCommand,
    DuplicateRevisionCommand,
    EditRevisionDescriptionCommand,
    RenameCardCommand,
    RenameKeyCommand,
    ReorderCardCommand,
    ReorderHotspotCommand,
    ReplaceHotspotSetCommand,
    ReplaceInteractionPolygonsCommand,
    ReplacePolygonCommand,
    ReplaceRevisionBackgroundCommand,
    SetHotspotConditionsCommand,
    SetHotspotKeyChangesCommand,
    SetRevisionReferenceCommand,
    SetRevisionStyleCommand,
    SetStartCardCommand,
    UpdateStyleCommand,
)
from hotcards.application.document_controller import DocumentController
from hotcards.domain.models import (
    Card,
    CardRevision,
    GeneratedBackground,
    HotspotConditions,
    HotspotKeyChanges,
    HotspotSet,
    ImageGenerationInputs,
    ImageGenerationMetadata,
    Interaction,
    KeyDefinition,
    NavigateAction,
    Point,
    Polygon,
    ResolvedCardReference,
    Stack,
    StyleDefinition,
    UnresolvedCardReference,
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
    interaction = Interaction()
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


def test_create_key_and_hotspot_reference_is_atomic() -> None:
    interaction = Interaction()
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

    assert changed.keys == (
        KeyDefinition(id=command.key_id, name="Visited castle"),
    )
    hotspot_set = changed.cards[0].active_revision.hotspot_set
    assert hotspot_set is not None
    assert hotspot_set.interactions[0].key_changes.grant == (command.key_id,)


def test_duplicate_revision_copies_style_selection() -> None:
    style = StyleDefinition(name="Ink", prompt_text="Rendered in ink")
    revision = CardRevision(style_id=style.id)
    card = Card(name="Card", revisions=(revision,))
    document = Stack(
        name="Stack",
        styles=(style,),
        new_card_style_id=style.id,
        cards=(card,),
    )

    changed = DuplicateRevisionCommand(
        card_id=card.id,
        source_revision_id=revision.id,
    ).apply(document)

    assert changed.cards[0].active_revision.style_id == style.id


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
    assert changed.references == (
        ResolvedCardReference(target_card_id=reference.id),
    )


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
        "Garden",
        "Tower",
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


def test_polygon_destination_and_hotspot_order_changes() -> None:
    first = interaction("Door", "Hall")
    second = interaction("Window", "Garden")
    source, revision = card_with_revision(first, second)
    destination = Card(name="Hall")
    document = Stack(name="Stack", cards=(source, destination))
    replacement = polygon(0.2)
    extra = polygon(0.4)

    document = ReplacePolygonCommand(
        card_id=source.id,
        revision_id=revision.id,
        interaction_id=first.id,
        polygon_index=0,
        polygon=replacement,
    ).apply(document)
    document = ReplaceInteractionPolygonsCommand(
        card_id=source.id,
        revision_id=revision.id,
        interaction_id=first.id,
        polygons=(replacement, extra),
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
    assert hotspots.interactions[1].polygons == (replacement, extra)
    assert hotspots.interactions[1].action.target == ResolvedCardReference(
        target_card_id=destination.id
    )
    assert hotspots.interactions[1].label == "Hall"

    renamed = RenameCardCommand(
        card_id=destination.id,
        name="Great Hall",
    ).apply(document)
    renamed_hotspots = renamed.cards[0].active_revision.hotspot_set
    assert renamed_hotspots is not None
    assert renamed_hotspots.interactions[1].label == "Great Hall"


def test_interaction_and_polygon_component_lifecycle() -> None:
    first = interaction("Door", "Hall")
    second = interaction("Window", "Garden")
    source, revision = card_with_revision(first)
    document = Stack(name="Stack", cards=(source,))
    extra = polygon(0.4)

    document = AddInteractionCommand(
        card_id=source.id,
        revision_id=revision.id,
        interaction=second,
    ).apply(document)
    document = AddPolygonCommand(
        card_id=source.id,
        revision_id=revision.id,
        interaction_id=first.id,
        polygon=extra,
    ).apply(document)
    document = DeletePolygonCommand(
        card_id=source.id,
        revision_id=revision.id,
        interaction_id=first.id,
        polygon_index=0,
    ).apply(document)
    document = DeleteInteractionCommand(
        card_id=source.id,
        revision_id=revision.id,
        interaction_id=second.id,
    ).apply(document)

    hotspot_set = document.cards[0].revisions[0].hotspot_set
    assert hotspot_set is not None
    assert len(hotspot_set.interactions) == 1
    assert hotspot_set.interactions[0].label == "Hall"
    assert hotspot_set.interactions[0].polygons == (extra,)


def test_polygon_component_deletion_can_leave_an_area_less_interaction() -> None:
    first = interaction("Door", "Hall")
    source, revision = card_with_revision(first)
    document = Stack(name="Stack", cards=(source,))

    changed = DeletePolygonCommand(
        card_id=source.id,
        revision_id=revision.id,
        interaction_id=first.id,
        polygon_index=0,
    ).apply(document)

    assert changed.cards[0].active_revision.hotspot_set is not None
    assert changed.cards[0].active_revision.hotspot_set.interactions[0].polygons == ()


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
                    update={
                        "references": (
                            ResolvedCardReference(
                                target_card_id=destination.id
                            ),
                        )
                    }
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
    assert hotspot_set.interactions[0].label == "Former Hall"
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
        generation_metadata=ImageGenerationMetadata(
            inputs=ImageGenerationInputs(
                description="A card",
            ),
            render_prompt="A card",
            model_identifier="test",
            mflux_version="test",
            seed=1,
            width=1024,
            height=768,
            step_count=4,
            generated_at=generated_at,
            duration_seconds=1,
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
