"""Focused tests for typed document commands."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from hypergen.application.commands import (
    ActivateRevisionCommand,
    AddImageRevisionCommand,
    AddInteractionCommand,
    AddPolygonCommand,
    ChangeHotspotDestinationCommand,
    CommandError,
    CreateCardCommand,
    DeleteCardCommand,
    DeleteImageRevisionCommand,
    DeleteInteractionCommand,
    DeletePolygonCommand,
    EditCardTextCommand,
    EditGlobalStyleCommand,
    RenameCardCommand,
    RenameInteractionCommand,
    ReorderCardCommand,
    ReorderHotspotCommand,
    ReplaceHotspotSetCommand,
    ReplaceInteractionPolygonsCommand,
    ReplacePolygonCommand,
    SetStartCardCommand,
)
from hypergen.domain.models import (
    Card,
    HotspotSet,
    ImageOrigin,
    ImageRevision,
    Interaction,
    NavigateAction,
    Point,
    Polygon,
    ResolvedCardReference,
    Stack,
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


def card_with_revision(*interactions: Interaction) -> tuple[Card, ImageRevision]:
    revision = ImageRevision(
        image_path="assets/cards/source/image.png",
        origin=ImageOrigin.IMPORTED,
        hotspot_set=HotspotSet(interactions=interactions),
        created_at=datetime.now(UTC),
    )
    return (
        Card(
            name="Source",
            image_revisions=(revision,),
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


def test_text_edits_are_individual_typed_field_changes() -> None:
    card = Card(name="Card")
    document = Stack(name="Stack", cards=(card,))

    document = EditCardTextCommand(
        card_id=card.id,
        field="scene_description",
        value="A quiet library",
    ).apply(document)
    document = EditCardTextCommand(
        card_id=card.id,
        field="interaction_description",
        value="The ladder can be climbed",
    ).apply(document)
    document = EditCardTextCommand(
        card_id=card.id,
        field="card_style",
        value="Woodcut",
    ).apply(document)

    changed = document.cards[0]
    assert changed.scene_description == "A quiet library"
    assert changed.interaction_description == "The ladder can be climbed"
    assert changed.card_style == "Woodcut"

    document = EditGlobalStyleCommand(value="Watercolor").apply(document)
    assert document.global_style == "Watercolor"


def test_revision_activation_and_complete_hotspot_replacement() -> None:
    source, first_revision = card_with_revision(interaction("Door", "Hall"))
    second_revision = ImageRevision(
        image_path="assets/cards/source/image-two.png",
        origin=ImageOrigin.IMPORTED,
        created_at=datetime.now(UTC),
    )
    source = source.model_copy(update={"image_revisions": (first_revision, second_revision)})
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

    changed_revision = document.cards[0].image_revisions[1]
    assert document.cards[0].active_revision_id == second_revision.id
    assert changed_revision.hotspot_set == replacement
    assert (
        ReplaceHotspotSetCommand(
            card_id=source.id,
            revision_id=second_revision.id,
            hotspot_set=None,
        )
        .apply(document)
        .cards[0]
        .image_revisions[1]
        .hotspot_set
        is None
    )


def test_add_and_delete_image_revisions_choose_safe_active_revision() -> None:
    card = Card(name="Card")
    document = Stack(name="Stack", cards=(card,))
    first = ImageRevision(
        image_path=f"assets/cards/{card.id}/image-{uuid4()}.png",
        origin=ImageOrigin.IMPORTED,
        created_at=datetime.now(UTC),
    )
    first = first.model_copy(
        update={
            "image_path": f"assets/cards/{card.id}/image-{first.id}.png",
        }
    )
    second = ImageRevision(
        image_path=f"assets/cards/{card.id}/image-{uuid4()}.png",
        origin=ImageOrigin.IMPORTED,
        created_at=datetime.now(UTC),
    )
    second = second.model_copy(
        update={
            "image_path": f"assets/cards/{card.id}/image-{second.id}.png",
        }
    )

    document = AddImageRevisionCommand(card_id=card.id, revision=first).apply(document)
    document = AddImageRevisionCommand(card_id=card.id, revision=second).apply(document)
    assert document.cards[0].active_revision_id == second.id

    document = DeleteImageRevisionCommand(
        card_id=card.id,
        revision_id=second.id,
    ).apply(document)
    assert document.cards[0].active_revision_id == first.id
    assert document.cards[0].image_revisions == (first,)

    document = DeleteImageRevisionCommand(
        card_id=card.id,
        revision_id=first.id,
    ).apply(document)
    assert document.cards[0].active_revision_id is None
    assert document.cards[0].image_revisions == ()


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

    hotspots = document.cards[0].image_revisions[0].hotspot_set
    assert hotspots is not None
    assert [item.id for item in hotspots.interactions] == [second.id, first.id]
    assert hotspots.interactions[1].polygons == (replacement, extra)
    assert hotspots.interactions[1].action.target == ResolvedCardReference(
        target_card_id=destination.id
    )


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
    document = RenameInteractionCommand(
        card_id=source.id,
        revision_id=revision.id,
        interaction_id=first.id,
        label="Archway",
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

    hotspot_set = document.cards[0].image_revisions[0].hotspot_set
    assert hotspot_set is not None
    assert len(hotspot_set.interactions) == 1
    assert hotspot_set.interactions[0].label == "Archway"
    assert hotspot_set.interactions[0].polygons == (extra,)


def test_polygon_component_deletion_preserves_valid_interactions() -> None:
    first = interaction("Door", "Hall")
    source, revision = card_with_revision(first)
    document = Stack(name="Stack", cards=(source,))

    with pytest.raises(CommandError, match="only polygon"):
        DeletePolygonCommand(
            card_id=source.id,
            revision_id=revision.id,
            interaction_id=first.id,
            polygon_index=0,
        ).apply(document)


def test_delete_card_converts_all_inbound_references_and_clears_start() -> None:
    destination = Card(name="Former Hall")
    inbound = interaction("Door", "ignored").model_copy(
        update={
            "action": NavigateAction(target=ResolvedCardReference(target_card_id=destination.id))
        }
    )
    source, _ = card_with_revision(inbound)
    document = Stack(
        name="Stack",
        cards=(source, destination),
        start_card_id=destination.id,
    )

    changed = DeleteCardCommand(card_id=destination.id).apply(document)

    assert [card.id for card in changed.cards] == [source.id]
    assert changed.start_card_id is None
    hotspot_set = changed.cards[0].image_revisions[0].hotspot_set
    assert hotspot_set is not None
    target = hotspot_set.interactions[0].action.target
    assert target == UnresolvedCardReference(target_name="Former Hall")


def test_commands_reject_missing_targets_and_invalid_domain_results() -> None:
    card = Card(name="Card")
    document = Stack(name="Stack", cards=(card,))

    with pytest.raises(CommandError, match="does not exist"):
        RenameCardCommand(card_id=uuid4(), name="Missing").apply(document)
    with pytest.raises(CommandError, match="out of range"):
        ReorderCardCommand(card_id=card.id, new_index=3).apply(document)
    with pytest.raises(ValidationError):
        CreateCardCommand(name=" card ").apply(document)
