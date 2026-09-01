"""Session-only deterministic stack playback state."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from hotcards.domain.models import (
    HotspotSet,
    Interaction,
    ResolvedCardReference,
    Stack,
)


@dataclass(frozen=True, slots=True)
class RunSessionState:
    """Current player location, keys, Back history, and transient feedback."""

    current_card_id: UUID | None
    history: tuple[UUID, ...]
    keys: frozenset[UUID]
    warning: str | None
    notice: str | None


class RunSession:
    """Execute conditional hotspots without mutating the authored document."""

    def __init__(self) -> None:
        self._current_card_id: UUID | None = None
        self._entry_card_id: UUID | None = None
        self._history: list[UUID] = []
        self._keys: set[UUID] = set()
        self._start_warning: str | None = None
        self._warning: str | None = None
        self._notice: str | None = None

    @property
    def state(self) -> RunSessionState:
        return RunSessionState(
            current_card_id=self._current_card_id,
            history=tuple(self._history),
            keys=frozenset(self._keys),
            warning=self._warning,
            notice=self._notice,
        )

    def start(
        self,
        document: Stack,
        preferred_card_id: UUID | None = None,
    ) -> RunSessionState:
        """Start at the preferred card while retaining the configured restart target."""
        card_ids = {card.id for card in document.cards}
        if preferred_card_id in card_ids:
            current_card_id = preferred_card_id
            warning = None
        elif document.start_card_id in card_ids:
            current_card_id = document.start_card_id
            warning = None
        elif document.cards:
            current_card_id = document.cards[0].id
            warning = (
                f'No start card is configured; previewing "{document.cards[0].name}".'
            )
        else:
            current_card_id = None
            warning = "This stack has no cards to run."
        if document.start_card_id not in card_ids and preferred_card_id in card_ids:
            card = next(
                card
                for card in document.cards
                if card.id == preferred_card_id
            )
            warning = (
                f'No start card is configured; previewing "{card.name}".'
            )
        self._entry_card_id = (
            document.start_card_id
            if document.start_card_id in card_ids
            else current_card_id
        )
        self._current_card_id = current_card_id
        self._history.clear()
        self._keys.clear()
        self._start_warning = warning
        self._warning = warning
        self._notice = None
        return self.state

    def active_hotspot_set(self, document: Stack) -> HotspotSet | None:
        """Return current hotspots whose conditions pass and actions have effects."""
        card = next(
            (card for card in document.cards if card.id == self._current_card_id),
            None,
        )
        if card is None or card.active_revision.hotspot_set is None:
            return None
        return HotspotSet(
            interactions=tuple(
                interaction
                for interaction in card.active_revision.hotspot_set.interactions
                if self._is_active(interaction) and self._has_effect(interaction)
            )
        )

    def activate(self, document: Stack, interaction_id: UUID) -> RunSessionState:
        """Execute one current hotspot's atomic key transition, then navigate."""
        self._warning = None
        self._notice = None
        card = next(
            (
                card
                for card in document.cards
                if card.id == self._current_card_id
            ),
            None,
        )
        if card is None:
            self._warning = "The current card is no longer available."
            return self.state
        revision = card.active_revision
        interactions = (
            revision.hotspot_set.interactions
            if revision.hotspot_set is not None
            else ()
        )
        interaction = next(
            (
                interaction
                for interaction in interactions
                if interaction.id == interaction_id
            ),
            None,
        )
        if interaction is None:
            self._warning = "That hotspot is no longer available on this card."
            return self.state
        if not self._is_active(interaction) or not self._has_effect(interaction):
            return self.state
        changes = interaction.key_changes
        self._keys.difference_update(changes.remove)
        self._keys.update(changes.grant)
        if interaction.action is None:
            self._notice = self._key_change_notice(document, interaction)
            return self.state
        target = interaction.action.target
        if not isinstance(target, ResolvedCardReference):
            self._warning = (
                f'Link to "{target.target_name}" is unresolved.'
                if target.target_name
                else "This hotspot does not have a destination yet."
            )
            return self.state
        destination = next(
            (
                candidate
                for candidate in document.cards
                if candidate.id == target.target_card_id
            ),
            None,
        )
        if destination is None:
            self._warning = "The linked destination card is missing."
            return self.state
        if destination.id != self._current_card_id:
            assert self._current_card_id is not None
            self._history.append(self._current_card_id)
            self._current_card_id = destination.id
        self._warning = None
        return self.state

    def activation_sound_id(
        self,
        document: Stack,
        interaction_id: UUID,
    ) -> UUID | None:
        """Return the Sound for one currently eligible hotspot activation."""
        interaction = self._eligible_interaction(document, interaction_id)
        return interaction.sound_id if interaction is not None else None

    def activation_has_navigation(
        self,
        document: Stack,
        interaction_id: UUID,
    ) -> bool:
        """Return whether one currently eligible activation attempts navigation."""
        interaction = self._eligible_interaction(document, interaction_id)
        return interaction is not None and interaction.action is not None

    def _eligible_interaction(
        self,
        document: Stack,
        interaction_id: UUID,
    ) -> Interaction | None:
        card = next(
            (card for card in document.cards if card.id == self._current_card_id),
            None,
        )
        if card is None or card.active_revision.hotspot_set is None:
            return None
        interaction = next(
            (
                candidate
                for candidate in card.active_revision.hotspot_set.interactions
                if candidate.id == interaction_id
            ),
            None,
        )
        if (
            interaction is None
            or not self._is_active(interaction)
            or not self._has_effect(interaction)
        ):
            return None
        return interaction

    def back(self) -> RunSessionState:
        if not self._history:
            self._warning = "There is no previous card."
            return self.state
        self._current_card_id = self._history.pop()
        self._warning = None
        self._notice = None
        return self.state

    def restart(self) -> RunSessionState:
        self._current_card_id = self._entry_card_id
        self._history.clear()
        self._keys.clear()
        self._warning = self._start_warning
        self._notice = None
        return self.state

    def clear(self) -> None:
        self._current_card_id = None
        self._entry_card_id = None
        self._history.clear()
        self._keys.clear()
        self._start_warning = None
        self._warning = None
        self._notice = None

    def _is_active(self, interaction: Interaction) -> bool:
        conditions = interaction.conditions
        return set(conditions.requires) <= self._keys and not (
            set(conditions.forbids) & self._keys
        )

    @staticmethod
    def _has_effect(interaction: Interaction) -> bool:
        changes = interaction.key_changes
        return (
            interaction.action is not None
            or interaction.sound_id is not None
            or bool(changes.remove)
            or bool(changes.grant)
        )

    @staticmethod
    def _key_change_notice(document: Stack, interaction: Interaction) -> str:
        changes = interaction.key_changes
        parts = [
            *(
                f"Removed {document.key_by_id(key_id).name}"
                for key_id in changes.remove
            ),
            *(
                f"Granted {document.key_by_id(key_id).name}"
                for key_id in changes.grant
            ),
        ]
        return " · ".join(parts)


__all__ = ["RunSession", "RunSessionState"]
