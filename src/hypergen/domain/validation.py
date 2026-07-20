"""Document-level validation helpers."""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable
from typing import Protocol


class NamedCard(Protocol):
    """Structural card type used by name validators."""

    @property
    def name(self) -> str: ...


def normalize_card_name(name: str) -> str:
    """Normalize an author-facing card name for exact matching and uniqueness."""
    return unicodedata.normalize("NFKC", name).strip().casefold()


def require_unique_card_names(cards: Iterable[NamedCard]) -> None:
    """Reject case-insensitively duplicate normalized card names."""
    seen: dict[str, str] = {}
    for card in cards:
        normalized = normalize_card_name(card.name)
        if normalized in seen:
            raise ValueError(
                f"card names must be unique ignoring case: {seen[normalized]!r} and {card.name!r}"
            )
        seen[normalized] = card.name
