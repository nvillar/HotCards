"""Tests for deterministic applied-hotspot Intent composition."""

from __future__ import annotations

from datetime import UTC, datetime

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
from hypergen.generation.hotspot_intent import compose_hotspot_intent


def polygon() -> Polygon:
    return Polygon(
        points=(
            Point(x=0.1, y=0.1),
            Point(x=0.3, y=0.1),
            Point(x=0.2, y=0.4),
        )
    )


def test_summary_preserves_order_labels_and_destination_states() -> None:
    destination = Card(name="Castle")
    interactions = (
        Interaction(
            label="Stone gate",
            action=NavigateAction(
                target=ResolvedCardReference(target_card_id=destination.id)
            ),
            polygons=(polygon(),),
        ),
        Interaction(
            label="Forest path",
            action=NavigateAction(
                target=UnresolvedCardReference(target_name="Deep Woods")
            ),
            polygons=(polygon(),),
        ),
        Interaction(
            label="Unknown symbol",
            action=NavigateAction(target=UnresolvedCardReference()),
            polygons=(polygon(),),
        ),
    )
    revision = ImageRevision(
        image_path="assets/cards/source/image.png",
        origin=ImageOrigin.IMPORTED,
        hotspot_set=HotspotSet(interactions=interactions),
        created_at=datetime.now(UTC),
    )
    source = Card(
        name="Source",
        image_revisions=(revision,),
        active_revision_id=revision.id,
    )
    stack = Stack(name="Demo", cards=(source, destination))

    summary = compose_hotspot_intent(stack, source.id)

    assert summary.splitlines() == [
        'Selecting "Stone gate" navigates to "Castle".',
        (
            'Selecting "Forest path" navigates to unresolved destination '
            '"Deep Woods".'
        ),
        'Selecting "Unknown symbol" has an unresolved destination.',
    ]
