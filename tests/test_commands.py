"""Focused tests for typed document commands."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from hypergen.application.commands import (
    ActivateRevisionCommand,
    AddInteractionCommand,
    AddPolygonCommand,
    AddStyleCommand,
    ChangeHotspotDestinationCommand,
    CommandError,
    CreateCardCommand,
    DeleteCardCommand,
    DeleteImageRevisionCommand,
    DeleteInteractionCommand,
    DeletePolygonCommand,
    DeleteStyleCommand,
    DuplicateRevisionCommand,
    EditRevisionDescriptionCommand,
    EditStyleCommand,
    RenameCardCommand,
    RenameInteractionCommand,
    ReorderCardCommand,
    ReorderHotspotCommand,
    ReplaceHotspotSetCommand,
    ReplaceInteractionPolygonsCommand,
    ReplacePolygonCommand,
    ReplaceRevisionBackgroundCommand,
    SetRevisionStyleCommand,
    SetStartCardCommand,
)
from hypergen.application.document_controller import DocumentController
from hypergen.domain.models import (
    Card,
    CardRevision,
    GenerationStyle,
    HotspotSet,
    ImportedBackground,
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


def test_revision_description_and_style_edits_are_typed_changes() -> None:
    card = Card(name="Card")
    style = GenerationStyle(name="Woodcut", prompt="Strong carved lines")
    document = Stack(name="Stack", styles=(style,), cards=(card,))
    revision_id = card.active_revision_id
    assert revision_id is not None

    document = EditRevisionDescriptionCommand(
        card_id=card.id,
        revision_id=revision_id,
        value="A quiet library",
    ).apply(document)
    document = SetRevisionStyleCommand(
        card_id=card.id,
        revision_id=revision_id,
        style_id=style.id,
    ).apply(document)
    document = EditStyleCommand(
        style_id=style.id,
        name="Detailed woodcut",
        prompt="Fine carved lines",
    ).apply(document)

    changed = document.cards[0].active_revision
    assert changed.description == "A quiet library"
    assert changed.style_id == style.id
    assert document.styles[0].name == "Detailed woodcut"
    assert document.styles[0].prompt == "Fine carved lines"


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
    assert changed_revision.hotspot_set == replacement
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

    document = DeleteImageRevisionCommand(
        card_id=card.id,
        revision_id=duplicate.revision_id,
    ).apply(document)
    assert document.cards[0].active_revision_id == first.id
    assert document.cards[0].revisions == (first,)

    with pytest.raises(CommandError, match="at least one revision"):
        DeleteImageRevisionCommand(
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

    hotspot_set = document.cards[0].revisions[0].hotspot_set
    assert hotspot_set is not None
    assert len(hotspot_set.interactions) == 1
    assert hotspot_set.interactions[0].label == "Archway"
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


def test_commands_reject_missing_targets_and_invalid_domain_results() -> None:
    card = Card(name="Card")
    document = Stack(name="Stack", cards=(card,))

    with pytest.raises(CommandError, match="does not exist"):
        RenameCardCommand(card_id=uuid4(), name="Missing").apply(document)
    with pytest.raises(CommandError, match="out of range"):
        ReorderCardCommand(card_id=card.id, new_index=3).apply(document)
    with pytest.raises(ValidationError):
        CreateCardCommand(name=" card ").apply(document)


def test_style_deletion_and_background_replacement_are_guarded() -> None:
    style = GenerationStyle(name="Watercolor", prompt="Soft washes")
    card = Card(name="Card")
    revision_id = card.active_revision_id
    assert revision_id is not None
    document = Stack(name="Stack", styles=(style,), cards=(card,))
    document = SetRevisionStyleCommand(
        card_id=card.id,
        revision_id=revision_id,
        style_id=style.id,
    ).apply(document)

    with pytest.raises(CommandError, match="used by a revision"):
        DeleteStyleCommand(style_id=style.id).apply(document)

    background = ImportedBackground(
        image_path=f"assets/cards/{card.id}/image-{uuid4()}.png",
        created_at=datetime.now(UTC),
    )
    document = ReplaceRevisionBackgroundCommand(
        card_id=card.id,
        revision_id=revision_id,
        background=background,
    ).apply(document)
    assert document.cards[0].active_revision.background == background

    document = SetRevisionStyleCommand(
        card_id=card.id,
        revision_id=revision_id,
        style_id=None,
    ).apply(document)
    document = DeleteStyleCommand(style_id=style.id).apply(document)
    assert document.styles == ()


def test_add_style_rejects_duplicate_names_case_insensitively() -> None:
    style = GenerationStyle(name="Watercolor", prompt="Soft washes")
    document = AddStyleCommand(style=style).apply(Stack(name="Stack"))

    with pytest.raises(CommandError, match="already in use"):
        AddStyleCommand(
            style=GenerationStyle(name="watercolor", prompt="Other"),
        ).apply(document)
