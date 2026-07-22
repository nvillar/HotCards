"""Bound document lifecycle and debounced autosave coordination."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, QTimer, Signal, Slot

from hypergen.application.document_controller import DocumentController
from hypergen.domain.models import Stack
from hypergen.storage.stack_store import StackStore, StackStoreError


class DocumentSessionError(ValueError):
    """A document lifecycle operation could not complete safely."""


@dataclass(frozen=True, slots=True)
class DocumentSessionState:
    """UI-facing state for one bound or unbound document session."""

    bundle_path: Path | None
    dirty: bool
    error: str | None


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
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(debounce_milliseconds)
        self._timer.timeout.connect(self.flush)
        controller.set_autosave_hook(self.schedule_autosave)

    @property
    def store(self) -> StackStore | None:
        return self._store

    @property
    def state(self) -> DocumentSessionState:
        return DocumentSessionState(
            bundle_path=self._store.bundle_path if self._store is not None else None,
            dirty=self._dirty,
            error=self._error,
        )

    def create(self, stack: Stack, bundle_path: Path) -> Stack:
        """Create, save, and bind a new bundle before exposing its document."""
        if bundle_path.exists():
            raise DocumentSessionError(f"bundle already exists: {bundle_path}")
        if not self.flush():
            raise DocumentSessionError(self._error or "current document could not be saved")
        store = StackStore(bundle_path)
        try:
            store.save(stack)
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
        return self._bind(store, stack, replace_document=False)

    @Slot(object)
    def schedule_autosave(self, snapshot: object) -> None:
        """Retain only the latest authoritative snapshot for the next save."""
        if not isinstance(snapshot, Stack):
            raise TypeError("autosave snapshots must be Stack instances")
        self._pending_snapshot = snapshot.model_copy(deep=True)
        self._dirty = True
        self._error = None
        if self._store is None:
            self._error = "Document is not bound to a .hypergen bundle"
        else:
            self._timer.start()
        self._emit_state()

    @Slot()
    def flush(self) -> bool:
        """Synchronously persist the latest pending snapshot, if any."""
        self._timer.stop()
        if self._pending_snapshot is None:
            return not self._dirty
        if self._store is None:
            self._error = "Document is not bound to a .hypergen bundle"
            self._emit_state()
            return False
        try:
            self._store.save(self._pending_snapshot)
        except StackStoreError as error:
            self._error = str(error)
            self._dirty = True
            self._emit_state()
            return False
        self._pending_snapshot = None
        self._dirty = False
        self._error = None
        self._emit_state()
        return True

    def _bind(
        self,
        store: StackStore,
        stack: Stack,
        *,
        replace_document: bool,
    ) -> Stack:
        self._timer.stop()
        self._store = store
        self._pending_snapshot = None
        self._dirty = False
        self._error = None
        document = (
            self.controller.replace_document(stack)
            if replace_document
            else self.controller.document
        )
        if replace_document:
            self.document_replaced.emit(document)
        self._emit_state()
        return document

    def _emit_state(self) -> None:
        self.state_changed.emit(self.state)


__all__ = [
    "DocumentSession",
    "DocumentSessionError",
    "DocumentSessionState",
]
