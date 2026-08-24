"""Typed mutations for an authoritative in-memory stack document."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol
from uuid import UUID, uuid4

from hypergen.domain.models import (
    Background,
    Card,
    CardReference,
    CardRevision,
    HotspotSet,
    ImagePrompt,
    Interaction,
    NavigateAction,
    Polygon,
    ResolvedCardReference,
    RunOverlayMode,
    Stack,
    UnresolvedCardReference,
)


class CommandError(ValueError):
    """A command cannot be applied to the current document."""


class DocumentCommand(Protocol):
    """A typed, atomic stack mutation."""

    def apply(self, document: Stack) -> Stack:
        """Return the valid document produced by this command."""


def validated_copy(document: Stack) -> Stack:
    """Return an independently owned, recursively validated document."""
    return Stack.model_validate(document.model_dump(mode="python", round_trip=True))


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
        card = Card(id=self.card_id, name=self.name)
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
        reference = revision.reference
        if (
            isinstance(reference, ResolvedCardReference)
            and reference.target_card_id == deleted_card_id
        ):
            updates["reference"] = UnresolvedCardReference(
                target_name=deleted_card_name
            )
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
class EditRevisionDescriptionCommand:
    """Commit one revision Description as one editing undo boundary."""

    card_id: UUID
    revision_id: UUID
    value: str

    def apply(self, document: Stack) -> Stack:
        card_index = _card_index(document, self.card_id)
        card = document.cards[card_index]
        revision_index = _revision_index(card, self.revision_id)
        revision = card.revisions[revision_index].model_copy(
            update={"description": self.value}
        )
        card = _replace_revision(card, revision_index, revision)
        return validated_copy(_replace_card(document, card_index, card))


@dataclass(frozen=True, slots=True)
class SetRevisionImagePromptCommand:
    """Set or clear one revision's prepared Image Prompt."""

    card_id: UUID
    revision_id: UUID
    value: ImagePrompt | None

    def apply(self, document: Stack) -> Stack:
        card_index = _card_index(document, self.card_id)
        card = document.cards[card_index]
        revision_index = _revision_index(card, self.revision_id)
        revision = card.revisions[revision_index].model_copy(
            update={"image_prompt": self.value}
        )
        card = _replace_revision(card, revision_index, revision)
        return validated_copy(_replace_card(document, card_index, card))


@dataclass(frozen=True, slots=True)
class SetRevisionReferenceCommand:
    """Assign or clear the revision's optional image Reference."""

    card_id: UUID
    revision_id: UUID
    reference: CardReference | None

    def apply(self, document: Stack) -> Stack:
        card_index = _card_index(document, self.card_id)
        card = document.cards[card_index]
        revision_index = _revision_index(card, self.revision_id)
        revision = card.revisions[revision_index].model_copy(
            update={"reference": self.reference}
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

    def apply(self, document: Stack) -> Stack:
        card_index = _card_index(document, self.card_id)
        card = document.cards[card_index]
        source = card.revisions[_revision_index(card, self.source_revision_id)]
        if any(revision.id == self.revision_id for revision in card.revisions):
            raise CommandError(f"revision {self.revision_id} already exists on card {card.id}")
        revision = source.model_copy(deep=True, update={"id": self.revision_id})
        card = card.model_copy(
            update={
                "revisions": (*card.revisions, revision),
                "active_revision_id": revision.id,
            }
        )
        return validated_copy(_replace_card(document, card_index, card))


@dataclass(frozen=True, slots=True)
class CreateGeneratedRevisionCommand:
    """Move a directly applied result into a new complete revision."""

    card_id: UUID
    revision_id: UUID
    previous_revision: CardRevision
    new_revision_id: UUID = field(default_factory=uuid4)

    def apply(self, document: Stack) -> Stack:
        card_index = _card_index(document, self.card_id)
        card = document.cards[card_index]
        source_index = _revision_index(card, self.revision_id)
        if self.previous_revision.id != self.revision_id:
            raise CommandError("the previous revision does not match the generated result")
        if any(revision.id == self.new_revision_id for revision in card.revisions):
            raise CommandError(
                f"revision {self.new_revision_id} already exists on card {card.id}"
            )
        generated_revision = card.revisions[source_index].model_copy(
            deep=True,
            update={"id": self.new_revision_id},
        )
        revisions = list(card.revisions)
        revisions[source_index] = self.previous_revision.model_copy(deep=True)
        revisions.append(generated_revision)
        card = card.model_copy(
            update={
                "revisions": tuple(revisions),
                "active_revision_id": generated_revision.id,
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
        revision = card.revisions[revision_index].model_copy(
            update={
                "background": (
                    self.background.model_copy(deep=True)
                    if self.background is not None
                    else None
                )
            }
        )
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
        hotspot_set = revision.hotspot_set.model_copy(
            update={"interactions": tuple(interactions)}
        )
        revision = revision.model_copy(update={"hotspot_set": hotspot_set})
        card = _replace_revision(card, revision_index, revision)
        return validated_copy(_replace_card(document, card_index, card))


@dataclass(frozen=True, slots=True)
class ReplaceInteractionPolygonsCommand:
    """Replace all polygon components of one interaction."""

    card_id: UUID
    revision_id: UUID
    interaction_id: UUID
    polygons: tuple[Polygon, ...]

    def apply(self, document: Stack) -> Stack:
        interaction = _interaction(
            document,
            card_id=self.card_id,
            revision_id=self.revision_id,
            interaction_id=self.interaction_id,
        ).model_copy(update={"polygons": self.polygons})
        changed = _replace_interaction(
            document,
            card_id=self.card_id,
            revision_id=self.revision_id,
            interaction_id=self.interaction_id,
            replacement=interaction,
        )
        return validated_copy(changed)


@dataclass(frozen=True, slots=True)
class AddPolygonCommand:
    """Append one polygon component to an interaction."""

    card_id: UUID
    revision_id: UUID
    interaction_id: UUID
    polygon: Polygon

    def apply(self, document: Stack) -> Stack:
        interaction = _interaction(
            document,
            card_id=self.card_id,
            revision_id=self.revision_id,
            interaction_id=self.interaction_id,
        )
        return ReplaceInteractionPolygonsCommand(
            card_id=self.card_id,
            revision_id=self.revision_id,
            interaction_id=self.interaction_id,
            polygons=(*interaction.polygons, self.polygon),
        ).apply(document)


@dataclass(frozen=True, slots=True)
class DeletePolygonCommand:
    """Delete one polygon component, retaining an area-less interaction."""

    card_id: UUID
    revision_id: UUID
    interaction_id: UUID
    polygon_index: int

    def apply(self, document: Stack) -> Stack:
        interaction = _interaction(
            document,
            card_id=self.card_id,
            revision_id=self.revision_id,
            interaction_id=self.interaction_id,
        )
        if not 0 <= self.polygon_index < len(interaction.polygons):
            raise CommandError(f"polygon index {self.polygon_index} is out of range")
        polygons = list(interaction.polygons)
        polygons.pop(self.polygon_index)
        return ReplaceInteractionPolygonsCommand(
            card_id=self.card_id,
            revision_id=self.revision_id,
            interaction_id=self.interaction_id,
            polygons=tuple(polygons),
        ).apply(document)


@dataclass(frozen=True, slots=True)
class ReplacePolygonCommand:
    """Replace one polygon component of an interaction."""

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
        polygons = list(interaction.polygons)
        polygons[self.polygon_index] = self.polygon
        return ReplaceInteractionPolygonsCommand(
            card_id=self.card_id,
            revision_id=self.revision_id,
            interaction_id=self.interaction_id,
            polygons=tuple(polygons),
        ).apply(document)


@dataclass(frozen=True, slots=True)
class ChangeHotspotDestinationCommand:
    """Change one hotspot's resolved or unresolved navigation destination."""

    card_id: UUID
    revision_id: UUID
    interaction_id: UUID
    destination: ResolvedCardReference | UnresolvedCardReference

    def apply(self, document: Stack) -> Stack:
        interaction = _interaction(
            document,
            card_id=self.card_id,
            revision_id=self.revision_id,
            interaction_id=self.interaction_id,
        )
        changed = interaction.model_copy(update={"action": NavigateAction(target=self.destination)})
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


@dataclass(frozen=True, slots=True)
class CreateCardAndResolveCommand:
    """Create a blank card and resolve one unresolved hotspot to it atomically."""

    source_card_id: UUID
    revision_id: UUID
    interaction_id: UUID
    card_name: str | None = None
    new_card_id: UUID = field(default_factory=uuid4)
    index: int | None = None

    def apply(self, document: Stack) -> Stack:
        interaction = _interaction(
            document,
            card_id=self.source_card_id,
            revision_id=self.revision_id,
            interaction_id=self.interaction_id,
        )
        target = interaction.action.target
        name = (
            self.card_name
            if self.card_name is not None
            else target.target_name
            if isinstance(target, UnresolvedCardReference)
            else None
        )
        if name is None:
            raise CommandError(
                "card_name is required when the hotspot has no unresolved target name"
            )
        changed = CreateCardCommand(
            name=name,
            card_id=self.new_card_id,
            index=self.index,
        ).apply(document)
        return ChangeHotspotDestinationCommand(
            card_id=self.source_card_id,
            revision_id=self.revision_id,
            interaction_id=self.interaction_id,
            destination=ResolvedCardReference(target_card_id=self.new_card_id),
        ).apply(changed)


__all__ = [
    "ActivateRevisionCommand",
    "AddInteractionCommand",
    "AddPolygonCommand",
    "ChangeHotspotDestinationCommand",
    "CommandError",
    "CreateCardAndResolveCommand",
    "CreateCardCommand",
    "CreateGeneratedRevisionCommand",
    "DeleteCardCommand",
    "DeleteInteractionCommand",
    "DeleteRevisionCommand",
    "DeletePolygonCommand",
    "DocumentCommand",
    "DuplicateRevisionCommand",
    "EditRevisionDescriptionCommand",
    "SetRevisionImagePromptCommand",
    "RenameCardCommand",
    "ReorderCardCommand",
    "ReorderHotspotCommand",
    "ReplaceHotspotSetCommand",
    "ReplaceInteractionPolygonsCommand",
    "ReplacePolygonCommand",
    "ReplaceRevisionBackgroundCommand",
    "SetRunOverlayModeCommand",
    "SetRevisionReferenceCommand",
    "SetStartCardCommand",
]
