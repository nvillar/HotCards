"""Bound document lifecycle and debounced autosave coordination."""

from __future__ import annotations

from collections.abc import Callable, Collection
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, QTimer, Signal, Slot

from hotcards.application.commands import DocumentCommand
from hotcards.application.document_controller import (
    DocumentController,
    OwnedImageAsset,
)
from hotcards.domain.models import DuplicateProvenance, Stack
from hotcards.storage.stack_store import (
    StackStore,
    StackStoreError,
    StoredImageAsset,
)


class DocumentSessionError(ValueError):
    """A document lifecycle operation could not complete safely."""

    def __init__(self, message: str, *, committed: bool = False) -> None:
        super().__init__(message)
        self.committed = committed


@dataclass(frozen=True, slots=True)
class DocumentSessionState:
    """UI-facing state for one bound or unbound document session."""

    bundle_path: Path | None
    dirty: bool
    error: str | None
    mutation_blocked: bool = False


class DocumentSession(QObject):
    """Bind one controller to bundle storage and debounce accepted mutations."""

    document_replaced = Signal(object)
    state_changed = Signal(object)

    def __init__(
        self,
        controller: DocumentController,
        *,
        debounce_milliseconds: int = 500,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        if debounce_milliseconds < 0:
            raise ValueError("autosave debounce must not be negative")
        self.controller = controller
        self._store: StackStore | None = None
        self._pending_snapshot: Stack | None = None
        self._dirty = False
        self._error: str | None = None
        self._released_assets: dict[tuple[Path, str], OwnedImageAsset] = {}
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(debounce_milliseconds)
        self._timer.timeout.connect(self.flush)
        controller.set_autosave_hook(self.schedule_autosave)
        controller.set_owned_asset_release_hook(self._queue_owned_asset_cleanup)

    @property
    def store(self) -> StackStore | None:
        return self._store

    @property
    def state(self) -> DocumentSessionState:
        return DocumentSessionState(
            bundle_path=self._store.bundle_path if self._store is not None else None,
            dirty=self._dirty,
            error=self._error,
            mutation_blocked=self.controller.mutation_blocked,
        )

    def create(self, stack: Stack, bundle_path: Path) -> Stack:
        """Create, save, and bind a new bundle before exposing its document."""
        try:
            if bundle_path.exists() and (
                not bundle_path.is_dir() or any(bundle_path.iterdir())
            ):
                raise DocumentSessionError(f"bundle already exists: {bundle_path}")
        except OSError as error:
            raise DocumentSessionError(
                f"could not inspect bundle destination {bundle_path}: {error}"
            ) from error
        if not self.flush():
            raise DocumentSessionError(self._error or "current document could not be saved")
        store = StackStore(bundle_path)
        try:
            store.create(stack)
        except StackStoreError as error:
            raise DocumentSessionError(str(error)) from error
        return self._bind(store, stack, replace_document=True)

    def open(self, bundle_path: Path) -> Stack:
        """Validate a bundle, preserve the current session on failure, then bind it."""
        store = StackStore(bundle_path)
        reopening_active_bundle = (
            self._store is not None
            and self._store.bundle_path.resolve() == bundle_path.resolve()
        )
        if reopening_active_bundle and not self.flush():
            raise DocumentSessionError(self._error or "current document could not be saved")
        try:
            stack = store.load()
        except StackStoreError as error:
            raise DocumentSessionError(str(error)) from error
        if not reopening_active_bundle and not self.flush():
            raise DocumentSessionError(self._error or "current document could not be saved")
        return self._bind(store, stack, replace_document=True)

    def save_as(self, bundle_path: Path) -> Stack:
        """Create an independent bundle and bind subsequent autosaves to it."""
        if not self.flush():
            raise DocumentSessionError(self._error or "current document could not be saved")
        stack = self.controller.document
        try:
            if self._store is None:
                if bundle_path.exists():
                    raise StackStoreError(
                        f"refusing to overwrite existing bundle: {bundle_path}"
                    )
                store = StackStore(bundle_path)
                store.save(stack)
            else:
                store = self._store.clone_to(bundle_path, stack)
        except StackStoreError as error:
            raise DocumentSessionError(str(error)) from error
        self.controller.clear_history()
        return self._bind(store, stack, replace_document=False)

    def execute_persisted(
        self,
        command: DocumentCommand,
        *,
        persist: Callable[[Stack], None] | None = None,
        owned_assets: Collection[OwnedImageAsset] = (),
    ) -> Stack:
        """Apply one command only after its complete snapshot is durably saved."""
        if self._store is None:
            raise DocumentSessionError("Document is not bound to a .hotcards bundle")
        self._timer.stop()
        before = self.controller.document
        try:
            document = self.controller.execute_persisted(
                command,
                persist or self._store.save,
                owned_assets=owned_assets,
            )
        except StackStoreError as error:
            persisted_stack = getattr(error, "persisted_stack", None)
            durability_indeterminate = bool(
                getattr(error, "durability_indeterminate", False)
            )
            committed = (
                self.controller.document != before
                and persisted_stack == self.controller.document
            )
            if committed:
                self._pending_snapshot = None
            elif durability_indeterminate:
                self._pending_snapshot = self.controller.document
            self._error = str(error)
            self._dirty = durability_indeterminate or (
                not committed and self._pending_snapshot is not None
            )
            self._emit_state()
            raise DocumentSessionError(str(error), committed=committed) from error
        self._pending_snapshot = None
        self._dirty = False
        self._error = None
        self._cleanup_released_assets()
        self._emit_state()
        return document

    @Slot(object)
    def schedule_autosave(self, snapshot: object) -> None:
        """Retain only the latest authoritative snapshot for the next save."""
        if not isinstance(snapshot, Stack):
            raise TypeError("autosave snapshots must be Stack instances")
        self._pending_snapshot = snapshot.model_copy(deep=True)
        self._dirty = True
        self._error = None
        if self._store is None:
            self._error = "Document is not bound to a .hotcards bundle"
        else:
            self._timer.start()
        self._emit_state()

    @Slot()
    def flush(self) -> bool:
        """Synchronously persist the latest pending snapshot, if any."""
        self._timer.stop()
        if self._pending_snapshot is None:
            self._cleanup_released_assets()
            self._emit_state()
            return not self._dirty
        if self._store is None:
            self._error = "Document is not bound to a .hotcards bundle"
            self._emit_state()
            return False
        try:
            persisted_snapshot = self._pending_snapshot
            self._store.save(persisted_snapshot)
        except StackStoreError as error:
            self._error = str(error)
            self._dirty = True
            self._emit_state()
            return False
        self.controller.confirm_persisted_document(persisted_snapshot)
        self._pending_snapshot = None
        self._dirty = False
        self._error = None
        self._cleanup_released_assets()
        self._emit_state()
        return True

    def close_history(self) -> bool:
        """Discard session history and reclaim any now-unreachable owned assets."""
        self.controller.detach_owned_assets()
        self._cleanup_released_assets()
        cleaned = not self._released_assets
        if not cleaned:
            self._released_assets.clear()
        self._emit_state()
        return cleaned

    def _bind(
        self,
        store: StackStore,
        stack: Stack,
        *,
        replace_document: bool,
    ) -> Stack:
        self._timer.stop()
        self.controller.detach_owned_assets()
        self._cleanup_released_assets()
        if self._released_assets:
            self._released_assets.clear()
            raise DocumentSessionError(
                self._error or "duplicate-owned assets could not be cleaned up"
            )
        self._store = store
        self._pending_snapshot = None
        self._dirty = False
        self._error = None
        document = (
            self.controller.replace_document(stack)
            if replace_document
            else self.controller.document
        )
        self.controller.register_owned_assets(
            self._duplicate_owned_assets(store, document)
        )
        if replace_document:
            self.document_replaced.emit(document)
        self._emit_state()
        return document

    def _emit_state(self) -> None:
        self.state_changed.emit(self.state)

    def _queue_owned_asset_cleanup(
        self,
        assets: tuple[OwnedImageAsset, ...],
    ) -> None:
        for asset in assets:
            self._released_assets[(asset.bundle_path, asset.relative_path)] = asset
        if self._pending_snapshot is None:
            self._cleanup_released_assets()
            self._emit_state()

    def _cleanup_released_assets(self) -> None:
        had_released_assets = bool(self._released_assets)
        cleanup_error: str | None = None
        for key, asset in tuple(self._released_assets.items()):
            store = StackStore(asset.bundle_path)
            try:
                store.remove_owned_image_asset_if_unreferenced(
                    StoredImageAsset(
                        relative_path=asset.relative_path,
                        device=asset.device,
                        inode=asset.inode,
                        directory_device=asset.directory_device,
                        directory_inode=asset.directory_inode,
                    ),
                    card_id=asset.card_id,
                    asset_id=asset.asset_id,
                    stack=Stack(name="Owned asset cleanup"),
                )
            except StackStoreError as error:
                cleanup_error = str(error)
                continue
            self._released_assets.pop(key, None)
        if cleanup_error is not None:
            self._error = cleanup_error
        elif had_released_assets and not self._released_assets:
            self._error = None

    @staticmethod
    def _duplicate_owned_assets(
        store: StackStore,
        document: Stack,
    ) -> tuple[OwnedImageAsset, ...]:
        assets: list[OwnedImageAsset] = []
        for card in document.cards:
            for revision in card.revisions:
                background = revision.background
                if background is None or not isinstance(
                    background.provenance,
                    DuplicateProvenance,
                ):
                    continue
                stored = store.stored_image_asset(
                    background.image_path,
                    card_id=card.id,
                    asset_id=background.id,
                )
                assets.append(
                    OwnedImageAsset(
                        bundle_path=store.bundle_path,
                        relative_path=stored.relative_path,
                        card_id=card.id,
                        asset_id=background.id,
                        device=stored.device,
                        inode=stored.inode,
                        directory_device=stored.directory_device,
                        directory_inode=stored.directory_inode,
                    )
                )
        return tuple(assets)


__all__ = [
    "DocumentSession",
    "DocumentSessionError",
    "DocumentSessionState",
]
