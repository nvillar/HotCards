"""Focused tests for authoritative document history and autosave signaling."""

from hypergen.application.commands import (
    AddInteractionCommand,
    ChangeHotspotDestinationCommand,
    CreateCardAndResolveCommand,
    CreateCardCommand,
    DeleteCardCommand,
    EditRevisionDescriptionCommand,
    RenameCardCommand,
    ReplaceHotspotSetCommand,
)
from hypergen.application.document_controller import DocumentController
from hypergen.domain.models import (
    Card,
    CardRevision,
    HotspotSet,
    Interaction,
    NavigateAction,
    Point,
    Polygon,
    ResolvedCardReference,
    Stack,
    UnresolvedCardReference,
)


def interaction(target_name: str = "Retained destination") -> Interaction:
    return Interaction(
        label="Door",
        action=NavigateAction(target=UnresolvedCardReference(target_name=target_name)),
        polygons=(
            Polygon(
                points=(
                    Point(x=0.1, y=0.1),
                    Point(x=0.4, y=0.1),
                    Point(x=0.2, y=0.5),
                )
            ),
        ),
    )


def document_with_hotspot() -> tuple[Stack, Card, CardRevision, Interaction]:
    hotspot = interaction()
    revision = CardRevision(
        description="Original description",
        hotspot_set=HotspotSet(interactions=(hotspot,)),
    )
    source = Card(
        name="Source",
        revisions=(revision,),
        active_revision_id=revision.id,
    )
    return Stack(name="Stack", cards=(source,), start_card_id=source.id), source, revision, hotspot


def hotspot_target(controller: DocumentController) -> object:
    hotspot_set = controller.document.cards[0].revisions[0].hotspot_set
    assert hotspot_set is not None
    return hotspot_set.interactions[0].action.target


def test_execute_undo_redo_and_new_command_invalidates_redo() -> None:
    card = Card(name="Original")
    controller = DocumentController(Stack(name="Stack", cards=(card,)))

    controller.execute(RenameCardCommand(card_id=card.id, name="First edit"))
    assert controller.document.cards[0].name == "First edit"
    assert controller.can_undo
    assert controller.undo()
    assert controller.document.cards[0].name == "Original"
    assert controller.can_redo
    assert controller.redo()
    assert controller.document.cards[0].name == "First edit"

    assert controller.undo()
    controller.execute(RenameCardCommand(card_id=card.id, name="Different edit"))
    assert not controller.can_redo
    assert not controller.redo()


def test_text_edits_have_sensible_per_commit_undo_boundaries() -> None:
    card = Card(name="Card")
    controller = DocumentController(Stack(name="Stack", cards=(card,)))

    controller.execute(
        EditRevisionDescriptionCommand(
            card_id=card.id,
            revision_id=card.active_revision.id,
            value="First committed edit",
        )
    )
    controller.execute(
        EditRevisionDescriptionCommand(
            card_id=card.id,
            revision_id=card.active_revision.id,
            value="Second committed edit",
        )
    )

    assert controller.undo()
    assert (
        controller.document.cards[0].active_revision.description
        == "First committed edit"
    )
    assert controller.undo()
    assert controller.document.cards[0].active_revision.description == ""


def test_complete_hotspot_replacement_is_one_atomic_undo_step() -> None:
    document, source, revision, original = document_with_hotspot()
    controller = DocumentController(document)
    replacement = HotspotSet(interactions=(interaction("One"), interaction("Two")))

    controller.execute(
        ReplaceHotspotSetCommand(
            card_id=source.id,
            revision_id=revision.id,
            hotspot_set=replacement,
        )
    )
    changed = controller.document.cards[0].revisions[0].hotspot_set
    assert changed is not None
    assert [item.id for item in changed.interactions] == [
        item.id for item in replacement.interactions
    ]
    assert all(item.label == "Unresolved" for item in changed.interactions)

    assert controller.undo()
    restored = controller.document.cards[0].revisions[0].hotspot_set
    assert restored is not None
    assert restored.interactions[0].id == original.id
    assert restored.interactions[0].label == "Unresolved"
    assert restored.interactions[0].action == original.action
    assert restored.interactions[0].polygons == original.polygons
    assert controller.redo()
    redone = controller.document.cards[0].revisions[0].hotspot_set
    assert redone is not None
    assert [item.id for item in redone.interactions] == [
        item.id for item in replacement.interactions
    ]
    assert all(item.label == "Unresolved" for item in redone.interactions)


def test_destination_resolution_is_one_undoable_command() -> None:
    document, source, revision, hotspot = document_with_hotspot()
    destination = Card(name="Existing destination")
    controller = DocumentController(
        document.model_copy(update={"cards": (*document.cards, destination)})
    )

    controller.execute(
        ChangeHotspotDestinationCommand(
            card_id=source.id,
            revision_id=revision.id,
            interaction_id=hotspot.id,
            destination=ResolvedCardReference(target_card_id=destination.id),
        )
    )
    assert hotspot_target(controller) == ResolvedCardReference(target_card_id=destination.id)

    assert controller.undo()
    assert hotspot_target(controller) == UnresolvedCardReference(target_name="Retained destination")


def test_manual_interaction_add_has_an_undo_boundary() -> None:
    document, source, revision, _hotspot = document_with_hotspot()
    controller = DocumentController(document)
    added = interaction("New destination")

    controller.execute(
        AddInteractionCommand(
            card_id=source.id,
            revision_id=revision.id,
            interaction=added,
        )
    )
    hotspot_set = controller.document.cards[0].revisions[0].hotspot_set
    assert hotspot_set is not None
    assert hotspot_set.interactions[-1].label == "Unresolved"
    assert controller.undo()
    hotspot_set = controller.document.cards[0].revisions[0].hotspot_set
    assert hotspot_set is not None
    assert len(hotspot_set.interactions) == 1


def test_deleting_start_destination_is_valid_and_undo_restores_inbound_link() -> None:
    document, source, revision, hotspot = document_with_hotspot()
    destination = Card(name="Deleted destination")
    resolved_hotspot = hotspot.model_copy(
        update={
            "action": NavigateAction(target=ResolvedCardReference(target_card_id=destination.id))
        }
    )
    resolved_revision = revision.model_copy(
        update={"hotspot_set": HotspotSet(interactions=(resolved_hotspot,))}
    )
    resolved_source = source.model_copy(update={"revisions": (resolved_revision,)})
    controller = DocumentController(
        Stack(
            name="Stack",
            cards=(resolved_source, destination),
            start_card_id=destination.id,
        )
    )

    controller.execute(DeleteCardCommand(card_id=destination.id))
    assert controller.document.start_card_id is None
    assert hotspot_target(controller) == UnresolvedCardReference(target_name="Deleted destination")

    assert controller.undo()
    assert controller.document.start_card_id == destination.id
    assert hotspot_target(controller) == ResolvedCardReference(target_card_id=destination.id)


def test_create_card_and_resolve_is_atomic_with_explicit_name() -> None:
    document, source, revision, hotspot = document_with_hotspot()
    controller = DocumentController(document)
    command = CreateCardAndResolveCommand(
        source_card_id=source.id,
        revision_id=revision.id,
        interaction_id=hotspot.id,
        card_name="Explicit destination",
    )

    controller.execute(command)
    assert [card.name for card in controller.document.cards] == [
        "Source",
        "Explicit destination",
    ]
    assert hotspot_target(controller) == ResolvedCardReference(target_card_id=command.new_card_id)

    assert controller.undo()
    assert [card.name for card in controller.document.cards] == ["Source"]
    assert hotspot_target(controller) == UnresolvedCardReference(target_name="Retained destination")

    assert controller.redo()
    assert controller.document.cards[1].id == command.new_card_id
    assert hotspot_target(controller) == ResolvedCardReference(target_card_id=command.new_card_id)


def test_create_card_and_resolve_defaults_to_retained_target_name() -> None:
    document, source, revision, hotspot = document_with_hotspot()
    controller = DocumentController(document)

    controller.execute(
        CreateCardAndResolveCommand(
            source_card_id=source.id,
            revision_id=revision.id,
            interaction_id=hotspot.id,
        )
    )

    assert controller.document.cards[1].name == "Retained destination"


def test_autosave_hook_signals_execute_undo_and_redo_with_snapshots() -> None:
    signals: list[Stack] = []
    controller = DocumentController(Stack(name="Stack"), autosave_hook=signals.append)

    controller.execute(CreateCardCommand(name="Card"))
    assert controller.undo()
    assert controller.redo()

    assert [len(stack.cards) for stack in signals] == [1, 0, 1]
    assert signals[-1] is not controller.document


def test_replace_document_clears_session_history_without_autosave() -> None:
    signals: list[Stack] = []
    controller = DocumentController(Stack(name="First"), autosave_hook=signals.append)
    controller.execute(CreateCardCommand(name="Old card"))
    assert controller.can_undo
    signals.clear()

    replaced = controller.replace_document(Stack(name="Second"))

    assert replaced.name == "Second"
    assert controller.document.name == "Second"
    assert not controller.can_undo
    assert not controller.can_redo
    assert signals == []
