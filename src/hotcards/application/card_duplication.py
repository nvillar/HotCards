"""Storage-safe active-card duplication workflow."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID, uuid4

from pydantic import ValidationError

from hotcards.application.commands import (
    CommandError,
    DuplicateCardCommand,
    next_duplicate_card_name,
)
from hotcards.application.document_controller import DocumentController, UndoToken
from hotcards.application.document_session import DocumentSession, DocumentSessionError
from hotcards.domain.models import Stack
from hotcards.storage.stack_store import StackStoreError


class CardDuplicationError(ValueError):
    """A card could not be duplicated without leaving partial state."""


@dataclass(frozen=True, slots=True)
class DuplicatedCardChange:
    """One completed duplicate and its undo identity."""

    document: Stack
    source_card_id: UUID
    duplicate_card_id: UUID
    token: UndoToken


class CardDuplicationWorkflow:
    """Copy one active complete revision and its owned image asset atomically."""

    def __init__(
        self,
        controller: DocumentController,
        session: DocumentSession,
    ) -> None:
        self.controller = controller
        self.session = session

    def duplicate(self, source_card_id: UUID) -> DuplicatedCardChange:
        """Duplicate one source card immediately after itself."""
        document = self.controller.document
        source_card = next(
            (card for card in document.cards if card.id == source_card_id),
            None,
        )
        if source_card is None:
            raise CardDuplicationError(f"card {source_card_id} does not exist")
        store = self.session.store
        if store is None:
            raise CardDuplicationError(
                "save the stack before duplicating a card"
            )

        source_revision = source_card.active_revision
        source_background = source_revision.background
        duplicate_card_id = uuid4()
        duplicate_revision_id = uuid4()
        duplicate_background_id = uuid4() if source_background is not None else None
        duplicate_image_path = (
            store.image_asset_path(duplicate_card_id, duplicate_background_id)
            if duplicate_background_id is not None
            else None
        )
        command = DuplicateCardCommand(
            source_card_id=source_card.id,
            name=next_duplicate_card_name(document, source_card.name),
            card_id=duplicate_card_id,
            revision_id=duplicate_revision_id,
            background_id=duplicate_background_id,
            background_image_path=duplicate_image_path,
            interaction_ids=tuple(
                uuid4()
                for _interaction in (
                    source_revision.hotspot_set.interactions
                    if source_revision.hotspot_set is not None
                    else ()
                )
            ),
        )

        try:
            command.apply(document)
        except (CommandError, ValidationError) as error:
            raise CardDuplicationError(str(error)) from error

        stored_image_path: str | None = None
        try:
            if source_background is not None:
                assert duplicate_background_id is not None
                stored_image_path = store.copy_image_asset(
                    source_background.image_path,
                    source_card_id=source_card.id,
                    source_asset_id=source_background.id,
                    card_id=duplicate_card_id,
                    asset_id=duplicate_background_id,
                )
                if stored_image_path != duplicate_image_path:
                    raise CardDuplicationError(
                        "duplicate image asset path did not match its reserved path"
                    )
            changed = self.session.execute_persisted(command)
        except (
            CardDuplicationError,
            CommandError,
            DocumentSessionError,
            StackStoreError,
            ValidationError,
        ) as error:
            if stored_image_path is not None and duplicate_background_id is not None:
                try:
                    store.remove_image_asset_if_unreferenced(
                        stored_image_path,
                        card_id=duplicate_card_id,
                        asset_id=duplicate_background_id,
                        stack=self.controller.document,
                    )
                except StackStoreError as cleanup_error:
                    raise CardDuplicationError(
                        f"{error}; could not roll back duplicate asset: {cleanup_error}"
                    ) from error
            if isinstance(error, CardDuplicationError):
                raise
            raise CardDuplicationError(str(error)) from error

        token = self.controller.current_undo_token
        if token is None:
            raise CardDuplicationError("duplicate did not create an undoable change")
        return DuplicatedCardChange(
            document=changed,
            source_card_id=source_card.id,
            duplicate_card_id=duplicate_card_id,
            token=token,
        )


__all__ = [
    "CardDuplicationError",
    "CardDuplicationWorkflow",
    "DuplicatedCardChange",
]
