"""Bound document lifecycle and debounced autosave coordination."""

from __future__ import annotations

from collections.abc import Callable, Collection
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, QTimer, Signal, Slot

from hotcards.application.commands import DocumentCommand
from hotcards.application.document_controller import (
    DocumentController,
    OwnedAsset,
    OwnedImageAsset,
    OwnedSoundAsset,
)
from hotcards.domain.models import (
    DirectGenerateProvenance,
    DuplicateProvenance,
    EditProvenance,
    RefineProvenance,
    Stack,
)
from hotcards.storage.stack_store import (
    StackStore,
    StackStoreError,
    StoredImageAsset,
    StoredSoundAsset,
    StoredStackDocument,
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


@dataclass(frozen=True, slots=True)
class _BindingCandidate:
    store: StackStore
    document: Stack
    stored_document: StoredStackDocument
    owned_assets: tuple[OwnedAsset, ...]


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
        self._released_assets: dict[tuple[Path, str], OwnedAsset] = {}
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
            if bundle_path.exists() and (not bundle_path.is_dir() or any(bundle_path.iterdir())):
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
        return self._bind_latest(
            store,
            replace_document=True,
            expected_document=stack,
        )

    def open(self, bundle_path: Path) -> Stack:
        """Validate a bundle, preserve the current session on failure, then bind it."""
        store = StackStore(bundle_path)
        reopening_active_bundle = (
            self._store is not None and self._store.bundle_path.resolve() == bundle_path.resolve()
        )
        if reopening_active_bundle and not self.flush():
            raise DocumentSessionError(self._error or "current document could not be saved")
        if not reopening_active_bundle:
            self._preflight_binding(store)
            if not self.flush():
                raise DocumentSessionError(self._error or "current document could not be saved")
        return self._bind_latest(store, replace_document=True)

    def save_as(self, bundle_path: Path) -> Stack:
        """Create an independent bundle and bind subsequent autosaves to it."""
        if not self.flush():
            raise DocumentSessionError(self._error or "current document could not be saved")
        stack = self.controller.document
        try:
            if self._store is None:
                store = StackStore(bundle_path)
                store.create(stack)
            else:
                store = self._store.clone_to(bundle_path, stack)
        except StackStoreError as error:
            raise DocumentSessionError(str(error)) from error
        return self._bind_latest(
            store,
            replace_document=False,
            expected_document=stack,
        )

    def execute_persisted(
        self,
        command: DocumentCommand,
        *,
        persist: Callable[[Stack], None] | None = None,
        owned_assets: Collection[OwnedAsset] = (),
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
            durability_indeterminate = bool(getattr(error, "durability_indeterminate", False))
            committed = (
                self.controller.document != before and persisted_stack == self.controller.document
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
        candidate: _BindingCandidate,
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
        self._store = candidate.store
        self._pending_snapshot = None
        self._dirty = False
        self._error = None
        document = (
            self.controller.replace_document(candidate.document)
            if replace_document
            else self.controller.document
        )
        self.controller.register_owned_assets(candidate.owned_assets)
        if replace_document:
            self.document_replaced.emit(document)
        self._emit_state()
        return document

    def _preflight_binding(
        self,
        store: StackStore,
        *,
        expected_document: Stack | None = None,
    ) -> _BindingCandidate:
        try:
            stored_document = store.load_document()
            document = stored_document.stack
            if expected_document is not None and document != expected_document:
                raise StackStoreError("candidate stack document changed before binding")
            owned_assets = self._app_owned_assets(store, document)
        except StackStoreError as error:
            raise DocumentSessionError(str(error)) from error
        return _BindingCandidate(
            store=store,
            document=document,
            stored_document=stored_document,
            owned_assets=owned_assets,
        )

    def _confirm_binding(self, candidate: _BindingCandidate) -> _BindingCandidate:
        confirmed = self._preflight_binding(
            candidate.store,
            expected_document=candidate.document,
        )
        if confirmed.stored_document != candidate.stored_document:
            raise DocumentSessionError("candidate stack document identity changed before binding")
        if confirmed.owned_assets != candidate.owned_assets:
            raise DocumentSessionError(
                "candidate duplicate-owned asset identity changed before binding"
            )
        return confirmed

    def _bind_latest(
        self,
        store: StackStore,
        *,
        replace_document: bool,
        expected_document: Stack | None = None,
    ) -> Stack:
        additional_paths = (
            (self._store.bundle_path,)
            if self._store is not None and self._store is not store
            else ()
        )

        def bind() -> Stack:
            candidate = self._confirm_binding(
                self._preflight_binding(
                    store,
                    expected_document=expected_document,
                )
            )
            return self._bind(candidate, replace_document=replace_document)

        return store.run_locked(
            bind,
            additional_bundle_paths=additional_paths,
        )

    def _emit_state(self) -> None:
        self.state_changed.emit(self.state)

    def _queue_owned_asset_cleanup(
        self,
        assets: tuple[OwnedAsset, ...],
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
                if isinstance(asset, OwnedImageAsset):
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
                else:
                    store.remove_owned_sound_asset_if_unreferenced(
                        StoredSoundAsset(
                            relative_path=asset.relative_path,
                            device=asset.device,
                            inode=asset.inode,
                            directory_device=asset.directory_device,
                            directory_inode=asset.directory_inode,
                        ),
                        sound_id=asset.sound_id,
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
    def _app_owned_assets(
        store: StackStore,
        document: Stack,
    ) -> tuple[OwnedAsset, ...]:
        assets: list[OwnedAsset] = []
        seen_paths: set[str] = set()
        for card in document.cards:
            for revision in card.revisions:
                background = revision.background
                if background is None or not isinstance(
                    background.provenance,
                    (
                        DirectGenerateProvenance,
                        RefineProvenance,
                        EditProvenance,
                        DuplicateProvenance,
                    ),
                ):
                    continue
                if background.image_path in seen_paths:
                    continue
                stored = store.stored_image_asset(
                    background.image_path,
                    card_id=card.id,
                    asset_id=background.id,
                )
                seen_paths.add(background.image_path)
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
        for sound in document.sounds:
            generated = sound.generated
            if generated is None or generated.audio_path in seen_paths:
                continue
            stored = store.stored_sound_asset(
                generated.audio_path,
                sound_id=sound.id,
                asset_id=generated.id,
            )
            seen_paths.add(generated.audio_path)
            assets.append(
                OwnedSoundAsset(
                    bundle_path=store.bundle_path,
                    relative_path=stored.relative_path,
                    sound_id=sound.id,
                    asset_id=generated.id,
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
