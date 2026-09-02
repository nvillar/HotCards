"""Generation result context for revision-aware UI actions."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from hotcards.application.document_controller import UndoToken
from hotcards.domain.models import CardRevision


@dataclass(frozen=True, slots=True)
class EditedRevisionChange:
    """Identify one completed Edit while its creation remains undoable."""

    token: UndoToken
    card_id: UUID
    revision_id: UUID
    instruction: str


@dataclass(frozen=True, slots=True)
class GeneratedRevisionChange:
    """Identify one directly applied result while it remains undoable."""

    message: str
    token: UndoToken
    card_id: UUID
    revision_id: UUID
    previous_revision: CardRevision


__all__ = ["EditedRevisionChange", "GeneratedRevisionChange"]
