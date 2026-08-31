"""Authoritative in-memory stack controller with session-only history."""

from __future__ import annotations

from collections.abc import Callable, Collection
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from hotcards.application.commands import DocumentCommand, validated_copy
from hotcards.domain.models import Stack

AutosaveHook = Callable[[Stack], None]
PersistenceHook = Callable[[Stack], None]


@dataclass(frozen=True, slots=True)
class UndoToken:
    """Opaque identity for one current undo-stack entry."""

    sequence: int


@dataclass(frozen=True, slots=True)
class OwnedImageAsset:
    """One duplicate-owned image eligible for history-aware reclamation."""

    bundle_path: Path
    relative_path: str
    card_id: UUID
    asset_id: UUID
    device: int
    inode: int


OwnedAssetReleaseHook = Callable[[tuple[OwnedImageAsset, ...]], None]


@dataclass(frozen=True, slots=True)
class _HistoryEntry:
    before: Stack
    after: Stack
    token: UndoToken


class DocumentController:
    """Own one stack document and mutate it only through typed commands."""

    def __init__(
        self,
        document: Stack,
        *,
        autosave_hook: AutosaveHook | None = None,
    ) -> None:
        self._document = validated_copy(document)
        self._autosave_hook = autosave_hook
        self._owned_asset_release_hook: OwnedAssetReleaseHook | None = None
        self._undo_stack: list[_HistoryEntry] = []
        self._redo_stack: list[_HistoryEntry] = []
        self._owned_assets: dict[tuple[Path, str], OwnedImageAsset] = {}
        self._next_undo_sequence = 1

    @property
    def document(self) -> Stack:
        """Return an independently owned snapshot of the current document."""
        return validated_copy(self._document)

    @property
    def can_undo(self) -> bool:
        """Return whether this session has a command to undo."""
        return bool(self._undo_stack)

    @property
    def can_redo(self) -> bool:
        """Return whether this session has an undone command to redo."""
        return bool(self._redo_stack)

    @property
    def current_undo_token(self) -> UndoToken | None:
        """Identify the latest command while it remains directly undoable."""
        return self._undo_stack[-1].token if self._undo_stack else None

    @property
    def current_redo_token(self) -> UndoToken | None:
        """Identify the next command while it remains directly redoable."""
        return self._redo_stack[-1].token if self._redo_stack else None

    def set_autosave_hook(self, hook: AutosaveHook | None) -> None:
        """Replace the callback signaled after each effective document change."""
        self._autosave_hook = hook

    def set_owned_asset_release_hook(
        self,
        hook: OwnedAssetReleaseHook | None,
    ) -> None:
        """Set the callback for duplicate-owned assets leaving all history."""
        self._owned_asset_release_hook = hook

    def replace_document(self, document: Stack) -> Stack:
        """Replace the active document and start a fresh session history."""
        self.detach_owned_assets()
        self._document = validated_copy(document)
        self._undo_stack.clear()
        self._redo_stack.clear()
        return self.document

    def execute(self, command: DocumentCommand) -> Stack:
        """Apply one command and record one session undo boundary."""
        before = self._document
        after = validated_copy(command.apply(validated_copy(before)))
        self._record_change(before, after)
        if after != before:
            self._signal_autosave()
            self._release_unreachable_owned_assets()
        return self.document

    def execute_persisted(
        self,
        command: DocumentCommand,
        persist: PersistenceHook,
        *,
        owned_assets: Collection[OwnedImageAsset] = (),
    ) -> Stack:
        """Persist a command result before exposing it or recording history."""
        before = self._document
        after = validated_copy(command.apply(validated_copy(before)))
        if after == before:
            return self.document
        try:
            persist(validated_copy(after))
        except Exception as error:
            persisted_after = getattr(error, "persisted_stack", None) == after
            retained_owned_asset = getattr(error, "owned_asset", None) is not None
            if persisted_after:
                self._record_change(before, after)
            if persisted_after or retained_owned_asset:
                for asset in tuple(owned_assets):
                    self._owned_assets[(asset.bundle_path, asset.relative_path)] = asset
                self._release_unreachable_owned_assets()
            raise
        self._record_change(before, after)
        for asset in tuple(owned_assets):
            self._owned_assets[(asset.bundle_path, asset.relative_path)] = asset
        self._release_unreachable_owned_assets()
        return self.document

    def _record_change(self, before: Stack, after: Stack) -> None:
        if after != before:
            self._document = after
            token = UndoToken(self._next_undo_sequence)
            self._next_undo_sequence += 1
            self._undo_stack.append(
                _HistoryEntry(before=before, after=after, token=token)
            )
            self._redo_stack.clear()

    def undo(self) -> bool:
        """Undo the latest command in this session."""
        if not self._undo_stack:
            return False
        entry = self._undo_stack.pop()
        self._document = entry.before
        self._redo_stack.append(entry)
        self._signal_autosave()
        self._release_unreachable_owned_assets()
        return True

    def undo_if_current(self, token: UndoToken) -> bool:
        """Undo only when the identified change is still the latest mutation."""
        if not self._undo_stack or self._undo_stack[-1].token != token:
            return False
        return self.undo()

    def redo(self) -> bool:
        """Redo the latest command undone in this session."""
        if not self._redo_stack:
            return False
        entry = self._redo_stack.pop()
        self._document = entry.after
        self._undo_stack.append(entry)
        self._signal_autosave()
        self._release_unreachable_owned_assets()
        return True

    def clear_history(self) -> None:
        """Discard undo and redo state without changing the document."""
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._release_unreachable_owned_assets()

    def register_owned_assets(
        self,
        assets: tuple[OwnedImageAsset, ...],
    ) -> None:
        """Track persisted duplicate assets loaded into the active document."""
        for asset in assets:
            self._owned_assets[(asset.bundle_path, asset.relative_path)] = asset
        self._release_unreachable_owned_assets()

    def detach_owned_assets(self) -> None:
        """Release unreachable assets and forget those retained by this document."""
        self._undo_stack.clear()
        self._redo_stack.clear()
        current_paths = self._background_paths(self._document)
        released = tuple(
            asset
            for asset in self._owned_assets.values()
            if asset.relative_path not in current_paths
        )
        self._owned_assets.clear()
        self._signal_owned_asset_release(released)

    def _signal_autosave(self) -> None:
        if self._autosave_hook is not None:
            self._autosave_hook(self.document)

    def _release_unreachable_owned_assets(self) -> None:
        reachable_paths = set(self._background_paths(self._document))
        for entry in (*self._undo_stack, *self._redo_stack):
            reachable_paths.update(self._background_paths(entry.before))
            reachable_paths.update(self._background_paths(entry.after))
        released = tuple(
            asset
            for asset in self._owned_assets.values()
            if asset.relative_path not in reachable_paths
        )
        for asset in released:
            self._owned_assets.pop((asset.bundle_path, asset.relative_path), None)
        self._signal_owned_asset_release(released)

    def _signal_owned_asset_release(
        self,
        assets: tuple[OwnedImageAsset, ...],
    ) -> None:
        if assets and self._owned_asset_release_hook is not None:
            self._owned_asset_release_hook(assets)

    @staticmethod
    def _background_paths(document: Stack) -> frozenset[str]:
        return frozenset(
            revision.background.image_path
            for card in document.cards
            for revision in card.revisions
            if revision.background is not None
        )


__all__ = [
    "AutosaveHook",
    "DocumentController",
    "OwnedImageAsset",
    "UndoToken",
]
