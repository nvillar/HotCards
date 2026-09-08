"""Typed mutations for an authoritative in-memory stack document."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol
from uuid import UUID, uuid4

from pydantic import BaseModel

from hotcards.domain.models import (
    Background,
    Card,
    CardReference,
    CardRevision,
    DuplicateOperation,
    EditDraft,
    GeneratedBackground,
    GeneratedSoundAsset,
    GenerateOutputSize,
    HotspotConditions,
    HotspotKeyChanges,
    HotspotSet,
    ImageProvenance,
    ImageSourceSnapshot,
    Interaction,
    KeyDefinition,
    NavigateAction,
    Polygon,
    ResolvedCardReference,
    RunOverlayMode,
    SoundDefinition,
    Stack,
    StyleDefinition,
    UnresolvedCardReference,
    original_image_provenance,
)
from hotcards.domain.validation import normalize_card_name


class CommandError(ValueError):
    """A command cannot be applied to the current document."""


class DocumentCommand(Protocol):
    """A typed, atomic stack mutation."""

    def apply(self, document: Stack) -> Stack:
        """Return the valid document produced by this command."""


def validated_copy(document: Stack) -> Stack:
    """Return an independently owned, recursively validated document."""
    return Stack.model_validate(BaseModel.model_dump(document, mode="python", round_trip=True))


def _card_index(document: Stack, card_id: UUID) -> int:
    for index, card in enumerate(document.cards):
        if card.id == card_id:
            return index
    raise CommandError(f"card {card_id} does not exist")


def _revision_index(card: Card, revision_id: UUID) -> int:
    for index, revision in enumerate(card.revisions):
        if revision.id == revision_id:
            return index
    raise CommandError(f"revision {revision_id} does not exist on card {card.id}")


def _style_index(document: Stack, style_id: UUID) -> int:
    for index, style in enumerate(document.styles):
        if style.id == style_id:
            return index
    raise CommandError(f"Style {style_id} does not exist")


def _key_index(document: Stack, key_id: UUID) -> int:
    for index, key in enumerate(document.keys):
        if key.id == key_id:
            return index
    raise CommandError(f"Key {key_id} does not exist")


def _sound_index(document: Stack, sound_id: UUID) -> int:
    for index, sound in enumerate(document.sounds):
        if sound.id == sound_id:
            return index
    raise CommandError(f"Sound {sound_id} does not exist")


def _interaction_index(revision: CardRevision, interaction_id: UUID) -> int:
    if revision.hotspot_set is None:
        raise CommandError(f"revision {revision.id} has no applied hotspot set")
    for index, interaction in enumerate(revision.hotspot_set.interactions):
        if interaction.id == interaction_id:
            return index
    raise CommandError(f"interaction {interaction_id} does not exist on revision {revision.id}")


def _replace_card(document: Stack, card_index: int, card: Card) -> Stack:
    cards = list(document.cards)
    cards[card_index] = card
    return document.model_copy(update={"cards": tuple(cards)})


def _replace_revision(card: Card, revision_index: int, revision: CardRevision) -> Card:
    revisions = list(card.revisions)
    revisions[revision_index] = revision
    return card.model_copy(update={"revisions": tuple(revisions)})


def _replace_interaction(
    document: Stack,
    *,
    card_id: UUID,
    revision_id: UUID,
    interaction_id: UUID,
    replacement: Interaction,
) -> Stack:
    card_index = _card_index(document, card_id)
    card = document.cards[card_index]
    revision_index = _revision_index(card, revision_id)
    revision = card.revisions[revision_index]
    interaction_index = _interaction_index(revision, interaction_id)
    assert revision.hotspot_set is not None
    interactions = list(revision.hotspot_set.interactions)
    interactions[interaction_index] = replacement
    hotspot_set = revision.hotspot_set.model_copy(update={"interactions": tuple(interactions)})
    revision = revision.model_copy(update={"hotspot_set": hotspot_set})
    card = _replace_revision(card, revision_index, revision)
    return _replace_card(document, card_index, card)


def _interaction(
    document: Stack,
    *,
    card_id: UUID,
    revision_id: UUID,
    interaction_id: UUID,
) -> Interaction:
    card = document.cards[_card_index(document, card_id)]
    revision = card.revisions[_revision_index(card, revision_id)]
    interaction_index = _interaction_index(revision, interaction_id)
    assert revision.hotspot_set is not None
    return revision.hotspot_set.interactions[interaction_index]


@dataclass(frozen=True, slots=True)
class CreateCardCommand:
    """Create one blank card, optionally at a specific position."""

    name: str
    card_id: UUID = field(default_factory=uuid4)
    index: int | None = None

    def apply(self, document: Stack) -> Stack:
        revision = CardRevision(style_id=document.new_card_style_id)
        card = Card(
            id=self.card_id,
            name=self.name,
            revisions=(revision,),
            active_revision_id=revision.id,
        )
        cards = list(document.cards)
        index = len(cards) if self.index is None else self.index
        if not 0 <= index <= len(cards):
            raise CommandError(f"card insertion index {index} is out of range")
        cards.insert(index, card)
        return validated_copy(document.model_copy(update={"cards": tuple(cards)}))


@dataclass(frozen=True, slots=True)
class RenameCardCommand:
    """Rename one card."""

    card_id: UUID
    name: str

    def apply(self, document: Stack) -> Stack:
        index = _card_index(document, self.card_id)
        card = document.cards[index].model_copy(update={"name": self.name})
        return validated_copy(_replace_card(document, index, card))


@dataclass(frozen=True, slots=True)
class ReorderCardCommand:
    """Move one card to a final zero-based position."""

    card_id: UUID
    new_index: int

    def apply(self, document: Stack) -> Stack:
        old_index = _card_index(document, self.card_id)
        if not 0 <= self.new_index < len(document.cards):
            raise CommandError(f"card index {self.new_index} is out of range")
        cards = list(document.cards)
        card = cards.pop(old_index)
        cards.insert(self.new_index, card)
        return validated_copy(document.model_copy(update={"cards": tuple(cards)}))


def _replace_deleted_target(
    interaction: Interaction,
    *,
    deleted_card_id: UUID,
    deleted_card_name: str,
) -> Interaction:
    if interaction.action is None:
        return interaction
    target = interaction.action.target
    if not (isinstance(target, ResolvedCardReference) and target.target_card_id == deleted_card_id):
        return interaction
    action = NavigateAction(target=UnresolvedCardReference(target_name=deleted_card_name))
    return interaction.model_copy(update={"action": action})


def _unresolve_inbound_references(
    card: Card,
    *,
    deleted_card_id: UUID,
    deleted_card_name: str,
) -> Card:
    revisions: list[CardRevision] = []
    for revision in card.revisions:
        updates: dict[str, object] = {}
        references = tuple(
            UnresolvedCardReference(target_name=deleted_card_name)
            if (
                isinstance(reference, ResolvedCardReference)
                and reference.target_card_id == deleted_card_id
            )
            else reference
            for reference in revision.references
        )
        if references != revision.references:
            updates["references"] = references
        if revision.hotspot_set is None:
            revisions.append(revision.model_copy(update=updates))
            continue
        interactions = tuple(
            _replace_deleted_target(
                interaction,
                deleted_card_id=deleted_card_id,
                deleted_card_name=deleted_card_name,
            )
            for interaction in revision.hotspot_set.interactions
        )
        hotspot_set = revision.hotspot_set.model_copy(update={"interactions": interactions})
        updates["hotspot_set"] = hotspot_set
        revisions.append(revision.model_copy(update=updates))
    return card.model_copy(update={"revisions": tuple(revisions)})


def next_duplicate_card_name(document: Stack, source_name: str) -> str:
    """Return the first case-insensitively unique copy name for a card."""
    existing_names = {normalize_card_name(card.name) for card in document.cards}
    candidate = f"{source_name} Copy"
    if normalize_card_name(candidate) not in existing_names:
        return candidate
    copy_number = 2
    while normalize_card_name(f"{source_name} Copy {copy_number}") in existing_names:
        copy_number += 1
    return f"{source_name} Copy {copy_number}"


def _duplicate_interaction(
    interaction: Interaction,
    *,
    source_card_id: UUID,
    duplicate_card_id: UUID,
    interaction_id: UUID,
) -> Interaction:
    action = interaction.action
    if (
        action is not None
        and isinstance(action.target, ResolvedCardReference)
        and action.target.target_card_id == source_card_id
    ):
        action = NavigateAction(target=ResolvedCardReference(target_card_id=duplicate_card_id))
    return interaction.model_copy(
        deep=True,
        update={"id": interaction_id, "action": action},
    )


@dataclass(frozen=True, slots=True)
class DuplicateCardCommand:
    """Insert one independent card copied from a source card's active revision."""

    source_card_id: UUID
    name: str
    card_id: UUID = field(default_factory=uuid4)
    revision_id: UUID = field(default_factory=uuid4)
    edit_draft_generation_id: UUID = field(default_factory=uuid4)
    background_id: UUID | None = None
    background_image_path: str | None = None
    interaction_ids: tuple[UUID, ...] = field(default_factory=tuple)

    def apply(self, document: Stack) -> Stack:
        source_index = _card_index(document, self.source_card_id)
        source_card = document.cards[source_index]
        source_revision = source_card.active_revision
        if any(card.id == self.card_id for card in document.cards):
            raise CommandError(f"card {self.card_id} already exists")
        if any(
            revision.id == self.revision_id
            for card in document.cards
            for revision in card.revisions
        ):
            raise CommandError(f"revision {self.revision_id} already exists")

        source_interactions = (
            source_revision.hotspot_set.interactions
            if source_revision.hotspot_set is not None
            else ()
        )
        if len(self.interaction_ids) != len(source_interactions):
            raise CommandError("duplicate interaction IDs must match the active hotspot set")
        if len(self.interaction_ids) != len(set(self.interaction_ids)):
            raise CommandError("duplicate interaction IDs must be unique")
        source_interaction_ids = {interaction.id for interaction in source_interactions}
        if source_interaction_ids & set(self.interaction_ids):
            raise CommandError("duplicate interactions must use new IDs")
        hotspot_set = (
            HotspotSet(
                interactions=tuple(
                    _duplicate_interaction(
                        interaction,
                        source_card_id=source_card.id,
                        duplicate_card_id=self.card_id,
                        interaction_id=interaction_id,
                    )
                    for interaction, interaction_id in zip(
                        source_interactions,
                        self.interaction_ids,
                        strict=True,
                    )
                )
            )
            if source_revision.hotspot_set is not None
            else None
        )

        source_background = source_revision.background
        if source_background is None:
            if self.background_id is not None or self.background_image_path is not None:
                raise CommandError(
                    "a blank source revision cannot have a duplicate background asset"
                )
            background = None
        else:
            if self.background_id is None or self.background_image_path is None:
                raise CommandError(
                    "a generated source revision requires a duplicate background asset"
                )
            if self.background_id == source_background.id:
                raise CommandError("duplicate background must use a new ID")
            background = GeneratedBackground(
                id=self.background_id,
                image_path=self.background_image_path,
                provenance=ImageProvenance(
                    origin=source_background.provenance.origin,
                    authoring=DuplicateOperation(
                        source=ImageSourceSnapshot(
                            card_id=source_card.id,
                            revision_id=source_revision.id,
                            background_id=source_background.id,
                        ),
                        original_authoring=original_image_provenance(
                            source_background.provenance
                        ).authoring,
                    ),
                ),
                created_at=source_background.created_at,
            )

        revision = source_revision.model_copy(
            deep=True,
            update={
                "id": self.revision_id,
                "background": background,
                "hotspot_set": hotspot_set,
                "edit_draft": EditDraft(
                    generation_id=self.edit_draft_generation_id,
                ),
            },
        )
        duplicate = Card(
            id=self.card_id,
            name=self.name,
            revisions=(revision,),
            active_revision_id=revision.id,
        )
        cards = list(document.cards)
        cards.insert(source_index + 1, duplicate)
        return validated_copy(document.model_copy(update={"cards": tuple(cards)}))


@dataclass(frozen=True, slots=True)
class DeleteCardCommand:
    """Delete a card and preserve inbound hotspots as unresolved references."""

    card_id: UUID

    def apply(self, document: Stack) -> Stack:
        index = _card_index(document, self.card_id)
        deleted_card = document.cards[index]
        cards = tuple(
            _unresolve_inbound_references(
                card,
                deleted_card_id=deleted_card.id,
                deleted_card_name=deleted_card.name,
            )
            for card_index, card in enumerate(document.cards)
            if card_index != index
        )
        start_card_id = (
            None if document.start_card_id == deleted_card.id else document.start_card_id
        )
        return validated_copy(
            document.model_copy(update={"cards": cards, "start_card_id": start_card_id})
        )


@dataclass(frozen=True, slots=True)
class SetStartCardCommand:
    """Select a start card, or explicitly clear the selection."""

    card_id: UUID | None

    def apply(self, document: Stack) -> Stack:
        if self.card_id is not None:
            _card_index(document, self.card_id)
        return validated_copy(document.model_copy(update={"start_card_id": self.card_id}))


@dataclass(frozen=True, slots=True)
class SetRunOverlayModeCommand:
    """Set the portable hotspot overlay behavior used in Run mode."""

    mode: RunOverlayMode

    def apply(self, document: Stack) -> Stack:
        return validated_copy(document.model_copy(update={"run_overlay_mode": self.mode}))


@dataclass(frozen=True, slots=True)
class AddStyleCommand:
    """Append one stack-owned Style definition."""

    name: str
    prompt_text: str = ""
    style_id: UUID = field(default_factory=uuid4)

    def apply(self, document: Stack) -> Stack:
        if any(style.id == self.style_id for style in document.styles):
            raise CommandError(f"Style {self.style_id} already exists")
        style = StyleDefinition(
            id=self.style_id,
            name=self.name,
            prompt_text=self.prompt_text,
        )
        return validated_copy(document.model_copy(update={"styles": (*document.styles, style)}))


@dataclass(frozen=True, slots=True)
class UpdateStyleCommand:
    """Update one Style's editable global name and prompt text."""

    style_id: UUID
    name: str
    prompt_text: str

    def apply(self, document: Stack) -> Stack:
        index = _style_index(document, self.style_id)
        styles = list(document.styles)
        styles[index] = styles[index].model_copy(
            update={"name": self.name, "prompt_text": self.prompt_text}
        )
        return validated_copy(document.model_copy(update={"styles": tuple(styles)}))


@dataclass(frozen=True, slots=True)
class DeleteStyleCommand:
    """Delete one Style and clear every selection that used it."""

    style_id: UUID

    def apply(self, document: Stack) -> Stack:
        index = _style_index(document, self.style_id)
        styles = list(document.styles)
        styles.pop(index)
        cards = tuple(
            card.model_copy(
                update={
                    "revisions": tuple(
                        revision.model_copy(update={"style_id": None})
                        if revision.style_id == self.style_id
                        else revision
                        for revision in card.revisions
                    )
                }
            )
            for card in document.cards
        )
        default_style_id = (
            None if document.new_card_style_id == self.style_id else document.new_card_style_id
        )
        return validated_copy(
            document.model_copy(
                update={
                    "styles": tuple(styles),
                    "cards": cards,
                    "new_card_style_id": default_style_id,
                }
            )
        )


@dataclass(frozen=True, slots=True)
class AddKeyCommand:
    """Append one stack-owned Key definition."""

    name: str
    key_id: UUID = field(default_factory=uuid4)

    def apply(self, document: Stack) -> Stack:
        if any(key.id == self.key_id for key in document.keys):
            raise CommandError(f"Key {self.key_id} already exists")
        key = KeyDefinition(id=self.key_id, name=self.name)
        return validated_copy(document.model_copy(update={"keys": (*document.keys, key)}))


@dataclass(frozen=True, slots=True)
class RenameKeyCommand:
    """Rename one stack-owned Key without changing hotspot references."""

    key_id: UUID
    name: str

    def apply(self, document: Stack) -> Stack:
        index = _key_index(document, self.key_id)
        keys = list(document.keys)
        keys[index] = keys[index].model_copy(update={"name": self.name})
        return validated_copy(document.model_copy(update={"keys": tuple(keys)}))


@dataclass(frozen=True, slots=True)
class DeleteKeyCommand:
    """Delete one unused stack-owned Key."""

    key_id: UUID

    def apply(self, document: Stack) -> Stack:
        index = _key_index(document, self.key_id)
        for card in document.cards:
            for revision in card.revisions:
                if revision.hotspot_set is None:
                    continue
                for interaction in revision.hotspot_set.interactions:
                    referenced = (
                        *interaction.conditions.requires,
                        *interaction.conditions.forbids,
                        *interaction.key_changes.remove,
                        *interaction.key_changes.grant,
                    )
                    if self.key_id in referenced:
                        raise CommandError(
                            f'Key "{document.keys[index].name}" is still used by hotspots'
                        )
        keys = list(document.keys)
        keys.pop(index)
        return validated_copy(document.model_copy(update={"keys": tuple(keys)}))


@dataclass(frozen=True, slots=True)
class AddSoundCommand:
    """Append one stack-owned Sound definition."""

    name: str
    prompt: str = ""
    duration_seconds: int = 2
    sound_id: UUID = field(default_factory=uuid4)

    def apply(self, document: Stack) -> Stack:
        if any(sound.id == self.sound_id for sound in document.sounds):
            raise CommandError(f"Sound {self.sound_id} already exists")
        sound = SoundDefinition(
            id=self.sound_id,
            name=self.name,
            prompt=self.prompt,
            duration_seconds=self.duration_seconds,
        )
        return validated_copy(document.model_copy(update={"sounds": (*document.sounds, sound)}))


@dataclass(frozen=True, slots=True)
class UpdateSoundCommand:
    """Update one Sound's editable name, prompt, and duration."""

    sound_id: UUID
    name: str
    prompt: str
    duration_seconds: int

    def apply(self, document: Stack) -> Stack:
        index = _sound_index(document, self.sound_id)
        sounds = list(document.sounds)
        sounds[index] = sounds[index].model_copy(
            update={
                "name": self.name,
                "prompt": self.prompt,
                "duration_seconds": self.duration_seconds,
            }
        )
        return validated_copy(document.model_copy(update={"sounds": tuple(sounds)}))


@dataclass(frozen=True, slots=True)
class ReplaceGeneratedSoundCommand:
    """Replace one Sound's generated asset through one undo boundary."""

    sound_id: UUID
    generated: GeneratedSoundAsset

    def apply(self, document: Stack) -> Stack:
        index = _sound_index(document, self.sound_id)
        sounds = list(document.sounds)
        sounds[index] = sounds[index].model_copy(update={"generated": self.generated})
        return validated_copy(document.model_copy(update={"sounds": tuple(sounds)}))


@dataclass(frozen=True, slots=True)
class DeleteSoundCommand:
    """Delete one Sound that is not referenced by any hotspot."""

    sound_id: UUID

    def apply(self, document: Stack) -> Stack:
        index = _sound_index(document, self.sound_id)
        for card in document.cards:
            for revision in card.revisions:
                if revision.hotspot_set is None:
                    continue
                if any(
                    interaction.sound_id == self.sound_id
                    for interaction in revision.hotspot_set.interactions
                ):
                    raise CommandError(
                        f'Sound "{document.sounds[index].name}" is still used by hotspots'
                    )
        sounds = list(document.sounds)
        sounds.pop(index)
        return validated_copy(document.model_copy(update={"sounds": tuple(sounds)}))


@dataclass(frozen=True, slots=True)
class EditRevisionDescriptionCommand:
    """Commit one revision Description as one editing undo boundary."""

    card_id: UUID
    revision_id: UUID
    value: str

    def apply(self, document: Stack) -> Stack:
        card_index = _card_index(document, self.card_id)
        card = document.cards[card_index]
        revision_index = _revision_index(card, self.revision_id)
        revision = card.revisions[revision_index].model_copy(update={"description": self.value})
        card = _replace_revision(card, revision_index, revision)
        return validated_copy(_replace_card(document, card_index, card))


@dataclass(frozen=True, slots=True)
class SetRevisionReferenceCommand:
    """Assign or clear one ordered revision image Reference."""

    card_id: UUID
    revision_id: UUID
    reference: CardReference | None
    position: int = 1

    def apply(self, document: Stack) -> Stack:
        card_index = _card_index(document, self.card_id)
        card = document.cards[card_index]
        revision_index = _revision_index(card, self.revision_id)
        revision = card.revisions[revision_index]
        if not 1 <= self.position <= 2:
            raise CommandError("Reference position must be 1 or 2")
        references = list(revision.references)
        index = self.position - 1
        if self.reference is None:
            if index < len(references):
                references.pop(index)
        elif index < len(references):
            references[index] = self.reference
        elif index == len(references):
            references.append(self.reference)
        else:
            raise CommandError("set Reference 1 before Reference 2")
        revision = revision.model_copy(update={"references": tuple(references)})
        card = _replace_revision(card, revision_index, revision)
        return validated_copy(_replace_card(document, card_index, card))


@dataclass(frozen=True, slots=True)
class SetRevisionStyleCommand:
    """Select a revision Style and make it the default for new cards."""

    card_id: UUID
    revision_id: UUID
    style_id: UUID | None

    def apply(self, document: Stack) -> Stack:
        if self.style_id is not None:
            _style_index(document, self.style_id)
        card_index = _card_index(document, self.card_id)
        card = document.cards[card_index]
        revision_index = _revision_index(card, self.revision_id)
        revision = card.revisions[revision_index].model_copy(update={"style_id": self.style_id})
        card = _replace_revision(card, revision_index, revision)
        changed = _replace_card(document, card_index, card)
        return validated_copy(changed.model_copy(update={"new_card_style_id": self.style_id}))


@dataclass(frozen=True, slots=True)
class SetRevisionGenerateOutputSizeCommand:
    """Select a revision's next direct Generate output size."""

    card_id: UUID
    revision_id: UUID
    output_size: GenerateOutputSize

    def apply(self, document: Stack) -> Stack:
        card_index = _card_index(document, self.card_id)
        card = document.cards[card_index]
        revision_index = _revision_index(card, self.revision_id)
        revision = card.revisions[revision_index].model_copy(
            update={"generate_output_size": self.output_size}
        )
        card = _replace_revision(card, revision_index, revision)
        return validated_copy(_replace_card(document, card_index, card))


@dataclass(frozen=True, slots=True)
class ActivateRevisionCommand:
    """Select a card's active revision."""

    card_id: UUID
    revision_id: UUID

    def apply(self, document: Stack) -> Stack:
        card_index = _card_index(document, self.card_id)
        card = document.cards[card_index]
        _revision_index(card, self.revision_id)
        card = card.model_copy(update={"active_revision_id": self.revision_id})
        return validated_copy(_replace_card(document, card_index, card))


@dataclass(frozen=True, slots=True)
class DuplicateRevisionCommand:
    """Append and activate a complete copy of one revision."""

    card_id: UUID
    source_revision_id: UUID
    revision_id: UUID = field(default_factory=uuid4)
    edit_draft_generation_id: UUID = field(default_factory=uuid4)

    def apply(self, document: Stack) -> Stack:
        card_index = _card_index(document, self.card_id)
        card = document.cards[card_index]
        source = card.revisions[_revision_index(card, self.source_revision_id)]
        if any(revision.id == self.revision_id for revision in card.revisions):
            raise CommandError(f"revision {self.revision_id} already exists on card {card.id}")
        revision = source.model_copy(
            deep=True,
            update={
                "id": self.revision_id,
                "edit_draft": EditDraft(
                    generation_id=self.edit_draft_generation_id,
                ),
            },
        )
        card = card.model_copy(
            update={
                "revisions": (*card.revisions, revision),
                "active_revision_id": revision.id,
            }
        )
        return validated_copy(_replace_card(document, card_index, card))


@dataclass(frozen=True, slots=True)
class CreateImageRevisionCommand:
    """Move a directly applied result into a new complete revision."""

    card_id: UUID
    revision_id: UUID
    previous_revision: CardRevision
    new_revision_id: UUID = field(default_factory=uuid4)
    new_edit_draft_generation_id: UUID = field(default_factory=uuid4)

    def apply(self, document: Stack) -> Stack:
        card_index = _card_index(document, self.card_id)
        card = document.cards[card_index]
        source_index = _revision_index(card, self.revision_id)
        if self.previous_revision.id != self.revision_id:
            raise CommandError("the previous revision does not match the applied image result")
        if any(revision.id == self.new_revision_id for revision in card.revisions):
            raise CommandError(f"revision {self.new_revision_id} already exists on card {card.id}")
        current_revision = card.revisions[source_index]
        result_revision = current_revision.model_copy(
            deep=True,
            update={
                "id": self.new_revision_id,
                "edit_draft": EditDraft(
                    generation_id=self.new_edit_draft_generation_id,
                ),
            },
        )
        revisions = list(card.revisions)
        revisions[source_index] = self.previous_revision.model_copy(
            deep=True,
            update={"edit_draft": current_revision.edit_draft},
        )
        revisions.append(result_revision)
        card = card.model_copy(
            update={
                "revisions": tuple(revisions),
                "active_revision_id": result_revision.id,
            }
        )
        return validated_copy(_replace_card(document, card_index, card))


@dataclass(frozen=True, slots=True)
class DeleteRevisionCommand:
    """Remove one revision and activate the nearest remaining revision."""

    card_id: UUID
    revision_id: UUID

    def apply(self, document: Stack) -> Stack:
        card_index = _card_index(document, self.card_id)
        card = document.cards[card_index]
        revision_index = _revision_index(card, self.revision_id)
        if len(card.revisions) == 1:
            raise CommandError("a card must retain at least one revision")
        revisions = list(card.revisions)
        revisions.pop(revision_index)
        active_revision_id = card.active_revision_id
        if active_revision_id == self.revision_id:
            replacement_index = min(revision_index, len(revisions) - 1)
            active_revision_id = revisions[replacement_index].id
        card = card.model_copy(
            update={
                "revisions": tuple(revisions),
                "active_revision_id": active_revision_id,
            }
        )
        return validated_copy(_replace_card(document, card_index, card))


@dataclass(frozen=True, slots=True)
class ReplaceRevisionBackgroundCommand:
    """Replace or clear one revision background without changing its other data."""

    card_id: UUID
    revision_id: UUID
    background: Background | None

    def apply(self, document: Stack) -> Stack:
        card_index = _card_index(document, self.card_id)
        card = document.cards[card_index]
        revision_index = _revision_index(card, self.revision_id)
        current_revision = card.revisions[revision_index]
        revision = current_revision.model_copy(
            update={
                "background": (
                    self.background.model_copy(deep=True) if self.background is not None else None
                )
            }
        )
        card = _replace_revision(card, revision_index, revision)
        return validated_copy(_replace_card(document, card_index, card))


@dataclass(frozen=True, slots=True)
class ApplyEditResultCommand:
    """Replace a background and consume only the exact submitted Edit draft."""

    card_id: UUID
    revision_id: UUID
    background: Background
    submitted_draft_generation_id: UUID
    cleared_draft_generation_id: UUID = field(default_factory=uuid4)

    def apply(self, document: Stack) -> Stack:
        card_index = _card_index(document, self.card_id)
        card = document.cards[card_index]
        revision_index = _revision_index(card, self.revision_id)
        current_revision = card.revisions[revision_index]
        updates: dict[str, object] = {
            "background": self.background.model_copy(deep=True),
        }
        if current_revision.edit_draft.generation_id == self.submitted_draft_generation_id:
            updates["edit_draft"] = EditDraft(
                generation_id=self.cleared_draft_generation_id,
            )
        revision = current_revision.model_copy(update=updates)
        card = _replace_revision(card, revision_index, revision)
        return validated_copy(_replace_card(document, card_index, card))


@dataclass(frozen=True, slots=True)
class ReplaceHotspotSetCommand:
    """Replace one revision's complete applied hotspot set atomically."""

    card_id: UUID
    revision_id: UUID
    hotspot_set: HotspotSet | None

    def apply(self, document: Stack) -> Stack:
        card_index = _card_index(document, self.card_id)
        card = document.cards[card_index]
        revision_index = _revision_index(card, self.revision_id)
        revision = card.revisions[revision_index].model_copy(
            update={
                "hotspot_set": (
                    self.hotspot_set.model_copy(deep=True) if self.hotspot_set is not None else None
                )
            }
        )
        card = _replace_revision(card, revision_index, revision)
        return validated_copy(_replace_card(document, card_index, card))


@dataclass(frozen=True, slots=True)
class AddInteractionCommand:
    """Add one complete interaction to an applied hotspot set."""

    card_id: UUID
    revision_id: UUID
    interaction: Interaction

    def apply(self, document: Stack) -> Stack:
        card_index = _card_index(document, self.card_id)
        card = document.cards[card_index]
        revision_index = _revision_index(card, self.revision_id)
        revision = card.revisions[revision_index]
        hotspot_set = revision.hotspot_set or HotspotSet()
        interactions = (*hotspot_set.interactions, self.interaction)
        hotspot_set = hotspot_set.model_copy(update={"interactions": interactions})
        revision = revision.model_copy(update={"hotspot_set": hotspot_set})
        card = _replace_revision(card, revision_index, revision)
        return validated_copy(_replace_card(document, card_index, card))


@dataclass(frozen=True, slots=True)
class DeleteInteractionCommand:
    """Delete one interaction while retaining the applied hotspot set."""

    card_id: UUID
    revision_id: UUID
    interaction_id: UUID

    def apply(self, document: Stack) -> Stack:
        card_index = _card_index(document, self.card_id)
        card = document.cards[card_index]
        revision_index = _revision_index(card, self.revision_id)
        revision = card.revisions[revision_index]
        interaction_index = _interaction_index(revision, self.interaction_id)
        assert revision.hotspot_set is not None
        interactions = list(revision.hotspot_set.interactions)
        interactions.pop(interaction_index)
        hotspot_set = revision.hotspot_set.model_copy(update={"interactions": tuple(interactions)})
        revision = revision.model_copy(update={"hotspot_set": hotspot_set})
        card = _replace_revision(card, revision_index, revision)
        return validated_copy(_replace_card(document, card_index, card))


@dataclass(frozen=True, slots=True)
class ReplacePolygonCommand:
    """Replace the polygon of one interaction."""

    card_id: UUID
    revision_id: UUID
    interaction_id: UUID
    polygon_index: int
    polygon: Polygon

    def apply(self, document: Stack) -> Stack:
        interaction = _interaction(
            document,
            card_id=self.card_id,
            revision_id=self.revision_id,
            interaction_id=self.interaction_id,
        )
        if not 0 <= self.polygon_index < len(interaction.polygons):
            raise CommandError(f"polygon index {self.polygon_index} is out of range")
        replacement = interaction.model_copy(update={"polygons": (self.polygon,)})
        changed = _replace_interaction(
            document,
            card_id=self.card_id,
            revision_id=self.revision_id,
            interaction_id=self.interaction_id,
            replacement=replacement,
        )
        return validated_copy(changed)


@dataclass(frozen=True, slots=True)
class SetHotspotConditionsCommand:
    """Replace one hotspot's complete all-of condition."""

    card_id: UUID
    revision_id: UUID
    interaction_id: UUID
    conditions: HotspotConditions

    def apply(self, document: Stack) -> Stack:
        interaction = _interaction(
            document,
            card_id=self.card_id,
            revision_id=self.revision_id,
            interaction_id=self.interaction_id,
        ).model_copy(update={"conditions": self.conditions})
        return validated_copy(
            _replace_interaction(
                document,
                card_id=self.card_id,
                revision_id=self.revision_id,
                interaction_id=self.interaction_id,
                replacement=interaction,
            )
        )


@dataclass(frozen=True, slots=True)
class SetHotspotKeyChangesCommand:
    """Replace one hotspot's complete atomic key transition."""

    card_id: UUID
    revision_id: UUID
    interaction_id: UUID
    key_changes: HotspotKeyChanges

    def apply(self, document: Stack) -> Stack:
        interaction = _interaction(
            document,
            card_id=self.card_id,
            revision_id=self.revision_id,
            interaction_id=self.interaction_id,
        ).model_copy(update={"key_changes": self.key_changes})
        return validated_copy(
            _replace_interaction(
                document,
                card_id=self.card_id,
                revision_id=self.revision_id,
                interaction_id=self.interaction_id,
                replacement=interaction,
            )
        )


@dataclass(frozen=True, slots=True)
class ChangeHotspotSoundCommand:
    """Change or clear one hotspot's Sound action."""

    card_id: UUID
    revision_id: UUID
    interaction_id: UUID
    sound_id: UUID | None

    def apply(self, document: Stack) -> Stack:
        if self.sound_id is not None:
            _sound_index(document, self.sound_id)
        interaction = _interaction(
            document,
            card_id=self.card_id,
            revision_id=self.revision_id,
            interaction_id=self.interaction_id,
        ).model_copy(update={"sound_id": self.sound_id})
        return validated_copy(
            _replace_interaction(
                document,
                card_id=self.card_id,
                revision_id=self.revision_id,
                interaction_id=self.interaction_id,
                replacement=interaction,
            )
        )


@dataclass(frozen=True, slots=True)
class ChangeHotspotDestinationCommand:
    """Change or clear one hotspot's navigation destination."""

    card_id: UUID
    revision_id: UUID
    interaction_id: UUID
    destination: ResolvedCardReference | UnresolvedCardReference | None

    def apply(self, document: Stack) -> Stack:
        interaction = _interaction(
            document,
            card_id=self.card_id,
            revision_id=self.revision_id,
            interaction_id=self.interaction_id,
        )
        changed = interaction.model_copy(
            update={
                "action": (
                    NavigateAction(target=self.destination)
                    if self.destination is not None
                    else None
                )
            }
        )
        return validated_copy(
            _replace_interaction(
                document,
                card_id=self.card_id,
                revision_id=self.revision_id,
                interaction_id=self.interaction_id,
                replacement=changed,
            )
        )


@dataclass(frozen=True, slots=True)
class ReorderHotspotCommand:
    """Move one hotspot to a final zero-based stacking position."""

    card_id: UUID
    revision_id: UUID
    interaction_id: UUID
    new_index: int

    def apply(self, document: Stack) -> Stack:
        card_index = _card_index(document, self.card_id)
        card = document.cards[card_index]
        revision_index = _revision_index(card, self.revision_id)
        revision = card.revisions[revision_index]
        old_index = _interaction_index(revision, self.interaction_id)
        assert revision.hotspot_set is not None
        if not 0 <= self.new_index < len(revision.hotspot_set.interactions):
            raise CommandError(f"hotspot index {self.new_index} is out of range")
        interactions = list(revision.hotspot_set.interactions)
        interaction = interactions.pop(old_index)
        interactions.insert(self.new_index, interaction)
        hotspot_set = revision.hotspot_set.model_copy(update={"interactions": tuple(interactions)})
        revision = revision.model_copy(update={"hotspot_set": hotspot_set})
        card = _replace_revision(card, revision_index, revision)
        return validated_copy(_replace_card(document, card_index, card))


__all__ = [
    "ActivateRevisionCommand",
    "AddKeyCommand",
    "AddSoundCommand",
    "AddStyleCommand",
    "AddInteractionCommand",
    "ChangeHotspotDestinationCommand",
    "ChangeHotspotSoundCommand",
    "CommandError",
    "CreateCardCommand",
    "CreateImageRevisionCommand",
    "DeleteCardCommand",
    "DeleteInteractionCommand",
    "DeleteKeyCommand",
    "DeleteRevisionCommand",
    "DeleteSoundCommand",
    "DeleteStyleCommand",
    "DocumentCommand",
    "DuplicateCardCommand",
    "DuplicateRevisionCommand",
    "EditRevisionDescriptionCommand",
    "SetRevisionStyleCommand",
    "RenameCardCommand",
    "RenameKeyCommand",
    "ReorderCardCommand",
    "ReorderHotspotCommand",
    "ReplaceHotspotSetCommand",
    "ReplacePolygonCommand",
    "ReplaceRevisionBackgroundCommand",
    "ReplaceGeneratedSoundCommand",
    "SetRunOverlayModeCommand",
    "SetHotspotConditionsCommand",
    "SetHotspotKeyChangesCommand",
    "SetRevisionReferenceCommand",
    "SetRevisionGenerateOutputSizeCommand",
    "SetStartCardCommand",
    "UpdateStyleCommand",
    "UpdateSoundCommand",
    "next_duplicate_card_name",
]
