"""Authoritative in-memory stack controller with session-only history."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from hypergen.application.commands import DocumentCommand, validated_copy
from hypergen.domain.models import Stack

AutosaveHook = Callable[[Stack], None]


@dataclass(frozen=True, slots=True)
class _HistoryEntry:
    before: Stack
    after: Stack


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
        self._undo_stack: list[_HistoryEntry] = []
        self._redo_stack: list[_HistoryEntry] = []

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

    def set_autosave_hook(self, hook: AutosaveHook | None) -> None:
        """Replace the callback signaled after each effective document change."""
        self._autosave_hook = hook

    def execute(self, command: DocumentCommand) -> Stack:
        """Apply one command and record one session undo boundary."""
        before = self._document
        after = validated_copy(command.apply(validated_copy(before)))
        if after != before:
            self._document = after
            self._undo_stack.append(_HistoryEntry(before=before, after=after))
            self._redo_stack.clear()
            self._signal_autosave()
        return self.document

    def undo(self) -> bool:
        """Undo the latest command in this session."""
        if not self._undo_stack:
            return False
        entry = self._undo_stack.pop()
        self._document = entry.before
        self._redo_stack.append(entry)
        self._signal_autosave()
        return True

    def redo(self) -> bool:
        """Redo the latest command undone in this session."""
        if not self._redo_stack:
            return False
        entry = self._redo_stack.pop()
        self._document = entry.after
        self._undo_stack.append(entry)
        self._signal_autosave()
        return True

    def clear_history(self) -> None:
        """Discard undo and redo state without changing the document."""
        self._undo_stack.clear()
        self._redo_stack.clear()

    def _signal_autosave(self) -> None:
        if self._autosave_hook is not None:
            self._autosave_hook(self.document)


__all__ = ["AutosaveHook", "DocumentController"]
