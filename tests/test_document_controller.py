"""Focused tests for authoritative document history and autosave signaling."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

import pytest
from helpers import generated_background
from pydantic import ValidationError

from hotcards.application.commands import (
    AddInteractionCommand,
    ChangeHotspotDestinationCommand,
    CreateCardCommand,
    DeleteCardCommand,
    DeleteSoundCommand,
    DuplicateRevisionCommand,
    EditRevisionDescriptionCommand,
    RenameCardCommand,
    ReplaceGeneratedSoundCommand,
    ReplaceHotspotSetCommand,
)
from hotcards.application.document_controller import (
    DocumentController,
    DocumentMutationBlockedError,
    OwnedSoundAsset,
)
from hotcards.domain.models import (
    Card,
    CardRevision,
    EditDraft,
    GeneratedSoundAsset,
    HotspotSet,
    Interaction,
    NavigateAction,
    Point,
    Polygon,
    ResolvedCardReference,
    SoundDefinition,
    SoundGenerationProvenance,
    Stack,
    UnresolvedCardReference,
)
from hotcards.storage.stack_store import StackStoreTransactionError


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


def test_owned_sound_is_released_only_after_history_cannot_restore_it() -> None:
    sound = SoundDefinition(name="Knock", prompt="A wooden knock")
    controller = DocumentController(Stack(name="Sounds", sounds=(sound,)))
    releases: list[OwnedSoundAsset] = []
    controller.set_owned_asset_release_hook(
        lambda assets: releases.extend(
            asset for asset in assets if isinstance(asset, OwnedSoundAsset)
        )
    )
    generated = GeneratedSoundAsset(
        audio_path=f"assets/sounds/{sound.id}/sound-{uuid4()}.wav",
        provenance=SoundGenerationProvenance(
            prompt="A wooden knock",
            duration_seconds=2,
            seed=42,
            generation_duration_milliseconds=800,
        ),
        created_at=datetime.now(UTC),
    )
    owned = OwnedSoundAsset(
        bundle_path=Path("Sounds.hotcards"),
        relative_path=generated.audio_path,
        sound_id=sound.id,
        asset_id=generated.id,
        device=1,
        inode=2,
        directory_device=1,
        directory_inode=3,
    )
    controller.execute_persisted(
        ReplaceGeneratedSoundCommand(sound_id=sound.id, generated=generated),
        lambda _stack: None,
        owned_assets=(owned,),
    )
    controller.execute(DeleteSoundCommand(sound_id=sound.id))

    assert releases == []

    controller.clear_history()

    assert releases == [owned]


def hotspot_target(controller: DocumentController) -> object:
    hotspot_set = controller.document.cards[0].revisions[0].hotspot_set
    assert hotspot_set is not None
    return hotspot_set.interactions[0].action.target


@pytest.mark.parametrize("boundary", ("construct", "replace"))
def test_document_snapshots_isolate_nested_input_and_reader_mutations(
    boundary: Literal["construct", "replace"],
) -> None:
    card_id = uuid4()
    background = generated_background(card_id)
    revision = CardRevision(background=background)
    card = Card(id=card_id, name="Card", revisions=(revision,))
    document = Stack(name="Stack", cards=(card,))
    if boundary == "construct":
        controller = DocumentController(document)
        first_read = controller.document
    else:
        controller = DocumentController(Stack(name="Previous"))
        first_read = controller.replace_document(document)
    second_read = controller.document

    input_background = document.cards[0].active_revision.background
    assert input_background is not None
    input_background.provenance.settings.dependency_versions["mflux"] = "input mutation"
    first_background = first_read.cards[0].active_revision.background
    assert first_background is not None
    first_background.provenance.settings.dependency_versions["mflux"] = "reader mutation"

    controller.execute(RenameCardCommand(card_id=card.id, name="Renamed"))
    assert controller.undo()
    assert controller.redo()
    for snapshot in (second_read, controller.document):
        snapshot_background = snapshot.cards[0].active_revision.background
        assert snapshot_background is not None
        assert snapshot_background.provenance.settings.dependency_versions == {"mflux": "test"}


@pytest.mark.parametrize("boundary", ("construct", "replace", "execute", "persist"))
def test_document_input_boundaries_reject_invalid_nested_models_without_state_changes(
    boundary: Literal["construct", "replace", "execute", "persist"],
) -> None:
    card = Card(name="Original")
    signals: list[Stack] = []
    persisted: list[Stack] = []
    controller = DocumentController(
        Stack(name="Stack", cards=(card,)), autosave_hook=signals.append
    )
    controller.execute(RenameCardCommand(card_id=card.id, name="Renamed"))
    before = controller.document
    token = controller.current_undo_token
    signals.clear()
    invalid_card = before.cards[0].model_copy(update={"name": ""})
    invalid = before.model_copy(update={"cards": (invalid_card,)})

    class InvalidDocumentCommand:
        def apply(self, _document: Stack) -> Stack:
            return invalid

    with pytest.raises(ValidationError, match="cards.0.name"):
        if boundary == "construct":
            DocumentController(invalid)
        elif boundary == "replace":
            controller.replace_document(invalid)
        elif boundary == "execute":
            controller.execute(InvalidDocumentCommand())
        else:
            controller.execute_persisted(InvalidDocumentCommand(), persisted.append)

    assert controller.document == before
    assert controller.current_undo_token == token
    assert signals == []
    assert persisted == []
    assert controller.undo()
    assert controller.document.cards[0].name == "Original"


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
    assert controller.document.cards[0].active_revision.description == "First committed edit"
    assert controller.undo()
    assert controller.document.cards[0].active_revision.description == ""


def test_retained_history_tokens_cover_redo_and_prune_abandoned_history() -> None:
    card = Card(name="Card")
    controller = DocumentController(Stack(name="Stack", cards=(card,)))
    controller.execute(RenameCardCommand(card_id=card.id, name="First"))
    first = controller.current_undo_token
    controller.execute(RenameCardCommand(card_id=card.id, name="Second"))
    second = controller.current_undo_token
    assert controller.retained_history_tokens == {first, second}
    assert controller.undo()
    assert controller.retained_history_tokens == {first, second}
    controller.execute(RenameCardCommand(card_id=card.id, name="New branch"))
    assert controller.retained_history_tokens == {first, controller.current_undo_token}
    controller.clear_history()
    assert controller.retained_history_tokens == set()
    controller.execute(RenameCardCommand(card_id=card.id, name="New history"))
    token = controller.current_undo_token
    assert controller.retained_history_tokens == {token}
    controller.replace_document(controller.document)
    assert controller.retained_history_tokens == set()
    controller.execute(RenameCardCommand(card_id=card.id, name="Replacement edit"))
    assert controller.current_undo_token != token


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
    assert [item.label for item in changed.interactions] == [
        "Go to One",
        "Go to Two",
    ]

    assert controller.undo()
    restored = controller.document.cards[0].revisions[0].hotspot_set
    assert restored is not None
    assert restored.interactions[0].id == original.id
    assert restored.interactions[0].label == "Go to Retained destination"
    assert restored.interactions[0].action == original.action
    assert restored.interactions[0].polygons == original.polygons
    assert controller.redo()
    redone = controller.document.cards[0].revisions[0].hotspot_set
    assert redone is not None
    assert [item.id for item in redone.interactions] == [
        item.id for item in replacement.interactions
    ]
    assert [item.label for item in redone.interactions] == [
        "Go to One",
        "Go to Two",
    ]


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
    assert hotspot_set.interactions[-1].label == "Go to New destination"
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


def test_autosave_hook_signals_execute_undo_and_redo_with_snapshots() -> None:
    card_id = uuid4()
    revision = CardRevision(background=generated_background(card_id))
    card = Card(id=card_id, name="Original", revisions=(revision,))
    signals: list[Stack] = []
    controller = DocumentController(
        Stack(name="Stack", cards=(card,)),
        autosave_hook=signals.append,
    )

    controller.execute(RenameCardCommand(card_id=card.id, name="Renamed"))
    assert controller.undo()
    assert controller.redo()

    assert [stack.cards[0].name for stack in signals] == ["Renamed", "Original", "Renamed"]
    saved_background = signals[0].cards[0].active_revision.background
    assert saved_background is not None
    saved_background.provenance.settings.dependency_versions["mflux"] = "snapshot mutation"

    for document in (*signals[1:], controller.document):
        background = document.cards[0].active_revision.background
        assert background is not None
        assert background.provenance.settings.dependency_versions == {"mflux": "test"}
    assert controller.undo()
    restored_background = controller.document.cards[0].active_revision.background
    assert restored_background is not None
    assert restored_background.provenance.settings.dependency_versions == {"mflux": "test"}


def test_pending_persisted_change_blocks_mutations_until_confirmed() -> None:
    card = Card(name="Original")
    before = Stack(name="Stack", cards=(card,))
    controller = DocumentController(before)
    command = CreateCardCommand(name="Observed duplicate")
    observed_after: Stack | None = None

    def fail_indeterminate(candidate: Stack) -> None:
        nonlocal observed_after
        observed_after = candidate
        raise StackStoreTransactionError(
            RuntimeError("manifest fsync failed"),
            observed_stack=candidate,
            durability_indeterminate=True,
        )

    with pytest.raises(
        StackStoreTransactionError,
        match="durability remains indeterminate",
    ):
        controller.execute_persisted(command, fail_indeterminate)

    assert observed_after is not None
    assert controller.document == observed_after
    assert controller.mutation_blocked
    assert not controller.can_undo
    assert not controller.can_redo

    blocked_operations = (
        lambda: controller.execute(RenameCardCommand(card_id=card.id, name="Blocked")),
        controller.undo,
        controller.redo,
        controller.clear_history,
        lambda: controller.replace_document(Stack(name="Replacement")),
    )
    for operation in blocked_operations:
        with pytest.raises(
            DocumentMutationBlockedError,
            match="Save the stack to finish the pending asset change",
        ):
            operation()
        assert controller.document == observed_after
        assert controller.mutation_blocked

    controller.confirm_persisted_document(observed_after)

    assert not controller.mutation_blocked
    assert controller.can_undo
    assert controller.undo()
    assert controller.document == before
    assert not controller.undo()


def test_observed_before_indeterminate_failure_does_not_block_mutations() -> None:
    card = Card(name="Original")
    before = Stack(name="Stack", cards=(card,))
    controller = DocumentController(before)

    def fail_indeterminate(_candidate: Stack) -> None:
        raise StackStoreTransactionError(
            RuntimeError("manifest fsync failed"),
            observed_stack=before,
            durability_indeterminate=True,
        )

    with pytest.raises(
        StackStoreTransactionError,
        match="durability remains indeterminate",
    ):
        controller.execute_persisted(
            CreateCardCommand(name="Unobserved duplicate"),
            fail_indeterminate,
        )

    assert controller.document == before
    assert not controller.mutation_blocked
    changed = controller.execute(RenameCardCommand(card_id=card.id, name="Allowed"))
    assert changed.cards[0].name == "Allowed"


def test_persisted_change_blocks_reentrant_commands_without_losing_history() -> None:
    card = Card(name="Original")
    before = Stack(name="Stack", cards=(card,))
    controller = DocumentController(before)

    def mutate_during_persistence(_candidate: Stack) -> None:
        controller.execute(RenameCardCommand(card_id=card.id, name="Intervening"))

    with pytest.raises(DocumentMutationBlockedError, match="current document change"):
        controller.execute_persisted(
            RenameCardCommand(card_id=card.id, name="Candidate"),
            mutate_during_persistence,
        )
    assert controller.document == before
    assert not controller.can_undo
    changed = controller.execute(RenameCardCommand(card_id=card.id, name="Later"))
    assert changed.cards[0].name == "Later"


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


def test_replace_document_discards_draft_state_even_when_ids_are_reused() -> None:
    card = Card(name="Old")
    controller = DocumentController(Stack(name="Old", cards=(card,)))
    controller.replace_edit_draft(
        card.id,
        card.active_revision.id,
        "Old project draft",
    )
    replacement_revision = card.active_revision.model_copy(
        update={"edit_draft": EditDraft(instruction="Replacement draft")}
    )
    replacement_card = card.model_copy(
        update={"name": "Replacement", "revisions": (replacement_revision,)}
    )

    controller.replace_document(Stack(name="Replacement", cards=(replacement_card,)))

    assert controller.edit_draft(card.id, card.active_revision.id) == (
        replacement_revision.edit_draft
    )
    assert not controller.can_undo_edit_draft(card.id, card.active_revision.id)
    assert not controller.can_redo_edit_draft(card.id, card.active_revision.id)


def test_draft_replacement_autosaves_without_changing_document_history() -> None:
    card = Card(name="Card")
    signals: list[Stack] = []
    controller = DocumentController(
        Stack(name="Stack", cards=(card,)),
        autosave_hook=signals.append,
    )
    controller.execute(RenameCardCommand(card_id=card.id, name="Renamed"))
    assert controller.undo()
    redo_token = controller.current_redo_token
    signals.clear()

    draft = controller.replace_edit_draft(
        card.id,
        card.active_revision.id,
        "  Keep raw text.\n",
    )

    assert controller.document.cards[0].active_revision.edit_draft == draft
    assert draft.instruction == "  Keep raw text.\n"
    assert controller.current_redo_token == redo_token
    assert controller.can_redo
    assert not controller.can_undo
    assert signals == [controller.document]


def test_draft_undo_and_redo_are_context_local_and_create_fresh_generations() -> None:
    first_card = Card(name="First")
    second_card = Card(name="Second")
    controller = DocumentController(Stack(name="Stack", cards=(first_card, second_card)))
    first = controller.replace_edit_draft(
        first_card.id,
        first_card.active_revision.id,
        "First",
    )
    second = controller.replace_edit_draft(
        first_card.id,
        first_card.active_revision.id,
        "Second",
    )
    other = controller.replace_edit_draft(
        second_card.id,
        second_card.active_revision.id,
        "Other",
    )

    assert controller.undo_edit_draft(first_card.id, first_card.active_revision.id)
    restored = controller.edit_draft(first_card.id, first_card.active_revision.id)
    assert restored.instruction == "First"
    assert restored.generation_id not in {first.generation_id, second.generation_id}
    assert controller.edit_draft(second_card.id, second_card.active_revision.id) == other

    assert controller.redo_edit_draft(first_card.id, first_card.active_revision.id)
    redone = controller.edit_draft(first_card.id, first_card.active_revision.id)
    assert redone.instruction == "Second"
    assert redone.generation_id not in {
        first.generation_id,
        second.generation_id,
        restored.generation_id,
    }
    assert not controller.can_undo


def test_clearing_history_discards_draft_undo_without_clearing_the_draft() -> None:
    card = Card(name="Card")
    controller = DocumentController(Stack(name="Stack", cards=(card,)))
    draft = controller.replace_edit_draft(
        card.id,
        card.active_revision.id,
        "Keep this draft",
    )
    assert controller.can_undo_edit_draft(card.id, card.active_revision.id)

    controller.clear_history()

    assert controller.edit_draft(card.id, card.active_revision.id) == draft
    assert not controller.can_undo_edit_draft(card.id, card.active_revision.id)
    assert not controller.can_redo_edit_draft(card.id, card.active_revision.id)


def test_document_undo_and_redo_preserve_newer_revision_draft() -> None:
    card = Card(name="Original")
    controller = DocumentController(Stack(name="Stack", cards=(card,)))
    controller.execute(RenameCardCommand(card_id=card.id, name="Renamed"))
    draft = controller.replace_edit_draft(
        card.id,
        card.active_revision.id,
        "Newer draft",
    )

    assert controller.undo()
    assert controller.document.cards[0].name == "Original"
    assert controller.document.cards[0].active_revision.edit_draft == draft
    assert controller.redo()
    assert controller.document.cards[0].name == "Renamed"
    assert controller.document.cards[0].active_revision.edit_draft == draft


def test_revision_draft_survives_undo_and_redo_of_its_creation() -> None:
    card = Card(name="Card")
    controller = DocumentController(Stack(name="Stack", cards=(card,)))
    command = DuplicateRevisionCommand(
        card_id=card.id,
        source_revision_id=card.active_revision.id,
    )
    controller.execute(command)
    draft = controller.replace_edit_draft(
        card.id,
        command.revision_id,
        "Draft on duplicated version",
    )

    assert controller.undo()
    assert [revision.id for revision in controller.document.cards[0].revisions] == [
        card.active_revision.id
    ]
    assert controller.redo()
    restored = controller.document.cards[0].active_revision
    assert restored.id == command.revision_id
    assert restored.edit_draft == draft


def test_document_draft_transition_distinguishes_same_text_generations() -> None:
    card = Card(name="Original")
    controller = DocumentController(Stack(name="Stack", cards=(card,)))
    submitted = controller.replace_edit_draft(
        card.id,
        card.active_revision.id,
        "Submitted",
    )
    consumed = EditDraft()

    class ConsumeDraftCommand:
        def apply(self, document: Stack) -> Stack:
            current_card = document.cards[0]
            revision = current_card.active_revision.model_copy(update={"edit_draft": consumed})
            changed_card = current_card.model_copy(
                update={"name": "Edited", "revisions": (revision,)}
            )
            return document.model_copy(update={"cards": (changed_card,)})

    controller.execute(ConsumeDraftCommand())
    assert controller.edit_draft(card.id, card.active_revision.id) == consumed
    assert controller.undo()
    assert controller.edit_draft(card.id, card.active_revision.id) == submitted
    assert controller.redo()
    assert controller.edit_draft(card.id, card.active_revision.id) == consumed

    assert controller.undo()
    newer = controller.replace_edit_draft(
        card.id,
        card.active_revision.id,
        "Submitted",
    )
    assert newer.instruction == submitted.instruction
    assert newer.generation_id != submitted.generation_id
    assert controller.redo()
    assert controller.document.cards[0].name == "Edited"
    assert controller.edit_draft(card.id, card.active_revision.id) == newer
