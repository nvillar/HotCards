"""Deterministic applied-hotspot to Intent composition."""

from __future__ import annotations

from uuid import UUID

from hypergen.domain.models import ResolvedCardReference, Stack


class HotspotIntentError(ValueError):
    """Applied hotspots cannot currently be summarized."""


def compose_hotspot_intent(document: Stack, card_id: UUID) -> str:
    """Describe every applied hotspot label and destination in author order."""
    card = next(
        (candidate for candidate in document.cards if candidate.id == card_id),
        None,
    )
    if card is None:
        raise HotspotIntentError(f"card {card_id} no longer exists")
    revision = next(
        (
            revision
            for revision in card.image_revisions
            if revision.id == card.active_revision_id
        ),
        None,
    )
    if revision is None:
        raise HotspotIntentError(
            "apply a background before summarizing hotspots"
        )
    if revision.hotspot_set is None or not revision.hotspot_set.interactions:
        raise HotspotIntentError(
            "the active revision has no applied hotspots to summarize"
        )
    card_names = {candidate.id: candidate.name for candidate in document.cards}
    sentences: list[str] = []
    for interaction in revision.hotspot_set.interactions:
        target = interaction.action.target
        if isinstance(target, ResolvedCardReference):
            destination = f'navigates to "{card_names[target.target_card_id]}"'
        elif target.target_name:
            destination = (
                f'navigates to unresolved destination "{target.target_name}"'
            )
        else:
            destination = "has an unresolved destination"
        sentences.append(f'Selecting "{interaction.label}" {destination}.')
    return "\n".join(sentences)


__all__ = [
    "HotspotIntentError",
    "compose_hotspot_intent",
]
