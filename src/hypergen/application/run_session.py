"""Session-only deterministic stack playback state."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from hypergen.domain.models import ResolvedCardReference, Stack


@dataclass(frozen=True, slots=True)
class RunSessionState:
    """Current player location, Back history, and non-blocking warning."""

    current_card_id: UUID | None
    history: tuple[UUID, ...]
    warning: str | None


class RunSession:
    """Navigate applied UUID links without mutating the authored document."""

    def __init__(self) -> None:
        self._current_card_id: UUID | None = None
        self._entry_card_id: UUID | None = None
        self._history: list[UUID] = []
        self._start_warning: str | None = None
        self._warning: str | None = None

    @property
    def state(self) -> RunSessionState:
        return RunSessionState(
            current_card_id=self._current_card_id,
            history=tuple(self._history),
            warning=self._warning,
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
        self._start_warning = warning
        self._warning = warning
        return self.state

    def navigate(self, document: Stack, interaction_id: UUID) -> RunSessionState:
        """Follow one interaction from the current active revision."""
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

    def back(self) -> RunSessionState:
        if not self._history:
            self._warning = "There is no previous card."
            return self.state
        self._current_card_id = self._history.pop()
        self._warning = None
        return self.state

    def restart(self) -> RunSessionState:
        self._current_card_id = self._entry_card_id
        self._history.clear()
        self._warning = self._start_warning
        return self.state

    def clear(self) -> None:
        self._current_card_id = None
        self._entry_card_id = None
        self._history.clear()
        self._start_warning = None
        self._warning = None


__all__ = ["RunSession", "RunSessionState"]
