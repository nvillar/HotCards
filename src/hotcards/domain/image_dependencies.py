"""Queries for revision-local derived image dependencies."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from typing import Literal
from uuid import UUID

from hotcards.domain.models import EditProvenance, RefineProvenance, Stack


@dataclass(frozen=True, slots=True)
class ImageSourceDependency:
    """One derived revision that requires a source revision to remain."""

    source_card_id: UUID
    source_revision_id: UUID
    dependent_card_id: UUID
    dependent_card_name: str
    dependent_revision_id: UUID
    dependent_revision_number: int
    operation: Literal["refine", "edit"]


def image_source_dependencies(
    document: Stack,
    source_revision_ids: Collection[UUID],
    *,
    excluding_revision_ids: Collection[UUID] = (),
) -> tuple[ImageSourceDependency, ...]:
    """Return derived revisions that depend on any selected source revision."""
    source_ids = frozenset(source_revision_ids)
    excluded_ids = frozenset(excluding_revision_ids)
    dependencies: list[ImageSourceDependency] = []
    for card in document.cards:
        for revision_number, revision in enumerate(card.revisions, start=1):
            if revision.id in excluded_ids:
                continue
            provenance = revision.provenance
            if not isinstance(provenance, (RefineProvenance, EditProvenance)):
                continue
            if provenance.source.revision_id not in source_ids:
                continue
            dependencies.append(
                ImageSourceDependency(
                    source_card_id=provenance.source.card_id,
                    source_revision_id=provenance.source.revision_id,
                    dependent_card_id=card.id,
                    dependent_card_name=card.name,
                    dependent_revision_id=revision.id,
                    dependent_revision_number=revision_number,
                    operation=provenance.operation,
                )
            )
    return tuple(dependencies)


__all__ = ["ImageSourceDependency", "image_source_dependencies"]
