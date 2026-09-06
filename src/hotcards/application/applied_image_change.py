"""Durably applied image context for history-bound version and Undo actions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal
from uuid import UUID

from hotcards.application.document_controller import UndoToken
from hotcards.domain.models import CardRevision


@dataclass(frozen=True, slots=True)
class ImageOperation:
    """Identify an image operation without an authored Edit instruction."""

    kind: Literal["generate", "refine"]


@dataclass(frozen=True, slots=True)
class EditImageOperation:
    """Retain the exact authored instruction, never the expanded model prompt."""

    instruction: str
    kind: Literal["edit"] = field(default="edit", init=False)


@dataclass(frozen=True, slots=True)
class AppliedImageChange:
    """One image replacement, independent of a later version-creation command."""

    token: UndoToken
    card_id: UUID
    revision_id: UUID
    previous_revision: CardRevision
    operation: ImageOperation | EditImageOperation

    @property
    def message(self) -> str:
        return image_operation_message(self.operation)


def image_operation_message(operation: ImageOperation | EditImageOperation) -> str:
    return {
        "generate": "Image generated",
        "refine": "Image reinterpreted",
        "edit": "Image edited",
    }[operation.kind]


__all__ = ["AppliedImageChange", "EditImageOperation", "ImageOperation"]
