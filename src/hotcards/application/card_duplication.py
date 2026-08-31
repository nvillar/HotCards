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
from hotcards.application.document_controller import (
    DocumentController,
    OwnedImageAsset,
    UndoToken,
)
from hotcards.application.document_session import DocumentSession, DocumentSessionError
from hotcards.domain.models import Stack
from hotcards.storage.stack_store import StackStoreTransactionError


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
        if not self.session.flush():
            raise CardDuplicationError(
                self.session.state.error or "the current stack could not be saved"
            )
        document = self.controller.document
        source_card = next(
            card for card in document.cards if card.id == source_card_id
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

        try:
            if source_background is not None:
                assert duplicate_background_id is not None
                assert duplicate_image_path is not None
                owned_assets: list[OwnedImageAsset] = []

                def persist_duplicate(candidate: Stack) -> None:
                    try:
                        stored_asset = store.copy_image_asset_and_save(
                            source_background.image_path,
                            source_card_id=source_card.id,
                            source_asset_id=source_background.id,
                            destination_card_id=duplicate_card_id,
                            destination_asset_id=duplicate_background_id,
                            previous_stack=document,
                            changed_stack=candidate,
                        )
                    except StackStoreTransactionError as error:
                        if error.owned_asset is not None:
                            owned_assets.append(
                                OwnedImageAsset(
                                    bundle_path=store.bundle_path,
                                    relative_path=error.owned_asset.relative_path,
                                    card_id=duplicate_card_id,
                                    asset_id=duplicate_background_id,
                                    device=error.owned_asset.device,
                                    inode=error.owned_asset.inode,
                                )
                            )
                        raise
                    owned_assets.append(
                        OwnedImageAsset(
                            bundle_path=store.bundle_path,
                            relative_path=stored_asset.relative_path,
                            card_id=duplicate_card_id,
                            asset_id=duplicate_background_id,
                            device=stored_asset.device,
                            inode=stored_asset.inode,
                        )
                    )

                changed = self.session.execute_persisted(
                    command,
                    persist=persist_duplicate,
                    owned_assets=owned_assets,
                )
            else:
                changed = self.session.execute_persisted(
                    command,
                    persist=lambda candidate: store.save_stack_transaction(
                        document,
                        candidate,
                    ),
                )
        except (
            CardDuplicationError,
            CommandError,
            DocumentSessionError,
            ValidationError,
        ) as error:
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
