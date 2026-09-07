"""Authoritative in-memory stack controller with session-only history."""

from __future__ import annotations

from collections.abc import Callable, Collection
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from hotcards.application.commands import DocumentCommand, validated_copy
from hotcards.domain.models import EditDraft, Stack

AutosaveHook = Callable[[Stack], None]
PersistenceHook = Callable[[Stack], None]
PENDING_DURABILITY_MESSAGE = (
    "Save the stack to finish the pending asset change before making another change."
)


class DocumentMutationBlockedError(ValueError):
    """A document mutation is blocked until pending durability is resolved."""


@dataclass(frozen=True, slots=True)
class UndoToken:
    """Opaque identity for one current undo-stack entry."""

    sequence: int


HistoryRecordedHook = Callable[[UndoToken], None]


@dataclass(frozen=True, slots=True)
class OwnedImageAsset:
    """One app-owned image eligible for history-aware reclamation."""

    bundle_path: Path
    relative_path: str
    card_id: UUID
    asset_id: UUID
    device: int
    inode: int
    directory_device: int
    directory_inode: int


@dataclass(frozen=True, slots=True)
class OwnedSoundAsset:
    """One generated sound eligible for history-aware reclamation."""

    bundle_path: Path
    relative_path: str
    sound_id: UUID
    asset_id: UUID
    device: int
    inode: int
    directory_device: int
    directory_inode: int


OwnedAsset = OwnedImageAsset | OwnedSoundAsset
OwnedAssetReleaseHook = Callable[[tuple[OwnedAsset, ...]], None]


@dataclass(frozen=True, slots=True)
class _HistoryEntry:
    before: Stack
    after: Stack
    token: UndoToken


@dataclass(frozen=True, slots=True)
class _PendingPersistedChange:
    before: Stack
    after: Stack
    on_recorded: HistoryRecordedHook | None = None


DraftContext = tuple[UUID, UUID]


@dataclass(frozen=True, slots=True)
class _DraftHistoryEntry:
    before_instruction: str
    after_instruction: str


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
        self._owned_assets: dict[tuple[Path, str], OwnedAsset] = {}
        self._durability_pending_assets: set[tuple[Path, str]] = set()
        self._pending_persisted_change: _PendingPersistedChange | None = None
        self._persistence_in_progress = False
        self._next_undo_sequence = 1
        self._edit_drafts = self._document_edit_drafts(self._document)
        self._draft_undo_stacks: dict[DraftContext, list[_DraftHistoryEntry]] = {}
        self._draft_redo_stacks: dict[DraftContext, list[_DraftHistoryEntry]] = {}

    @property
    def document(self) -> Stack:
        """Return an independently owned snapshot of the current document."""
        return validated_copy(self._document)

    @property
    def can_undo(self) -> bool:
        """Return whether this session has a command to undo."""
        return not self.mutation_blocked and bool(self._undo_stack)

    @property
    def can_redo(self) -> bool:
        """Return whether this session has an undone command to redo."""
        return not self.mutation_blocked and bool(self._redo_stack)

    @property
    def mutation_blocked(self) -> bool:
        """Return whether a visible change still lacks durable persistence."""
        return self._pending_persisted_change is not None

    @property
    def mutation_blocked_reason(self) -> str | None:
        """Explain why document mutations are temporarily blocked."""
        return PENDING_DURABILITY_MESSAGE if self.mutation_blocked else None

    @property
    def current_undo_token(self) -> UndoToken | None:
        """Identify the latest command while it remains directly undoable."""
        return self._undo_stack[-1].token if self._undo_stack else None

    @property
    def current_redo_token(self) -> UndoToken | None:
        """Identify the next command while it remains directly redoable."""
        return self._redo_stack[-1].token if self._redo_stack else None

    def edit_draft(self, card_id: UUID, revision_id: UUID) -> EditDraft:
        """Return the authoritative draft for one revision in the current document."""
        context = (card_id, revision_id)
        if context not in self._current_draft_contexts():
            raise ValueError(f"revision {revision_id} does not exist on card {card_id}")
        return self._edit_drafts[context].model_copy(deep=True)

    def replace_edit_draft(
        self,
        card_id: UUID,
        revision_id: UUID,
        instruction: str,
    ) -> EditDraft:
        """Replace one draft without adding to or invalidating document history."""
        self._require_mutation_allowed()
        context = (card_id, revision_id)
        before = self.edit_draft(card_id, revision_id)
        after = EditDraft(instruction=instruction)
        self._set_current_edit_draft(context, after)
        self._draft_undo_stacks.setdefault(context, []).append(
            _DraftHistoryEntry(
                before_instruction=before.instruction,
                after_instruction=after.instruction,
            )
        )
        self._draft_redo_stacks.pop(context, None)
        self._signal_autosave()
        return after.model_copy(deep=True)

    def can_undo_edit_draft(self, card_id: UUID, revision_id: UUID) -> bool:
        """Return whether the current revision has a draft replacement to undo."""
        context = (card_id, revision_id)
        return context in self._current_draft_contexts() and bool(
            self._draft_undo_stacks.get(context)
        )

    def can_redo_edit_draft(self, card_id: UUID, revision_id: UUID) -> bool:
        """Return whether the current revision has a draft replacement to redo."""
        context = (card_id, revision_id)
        return context in self._current_draft_contexts() and bool(
            self._draft_redo_stacks.get(context)
        )

    def undo_edit_draft(self, card_id: UUID, revision_id: UUID) -> bool:
        """Undo one revision-local draft replacement with a fresh generation."""
        self._require_mutation_allowed()
        context = (card_id, revision_id)
        self.edit_draft(card_id, revision_id)
        undo_stack = self._draft_undo_stacks.get(context)
        if not undo_stack:
            return False
        entry = undo_stack.pop()
        self._set_current_edit_draft(
            context,
            EditDraft(instruction=entry.before_instruction),
        )
        self._draft_redo_stacks.setdefault(context, []).append(entry)
        self._signal_autosave()
        return True

    def redo_edit_draft(self, card_id: UUID, revision_id: UUID) -> bool:
        """Redo one revision-local draft replacement with a fresh generation."""
        self._require_mutation_allowed()
        context = (card_id, revision_id)
        self.edit_draft(card_id, revision_id)
        redo_stack = self._draft_redo_stacks.get(context)
        if not redo_stack:
            return False
        entry = redo_stack.pop()
        self._set_current_edit_draft(
            context,
            EditDraft(instruction=entry.after_instruction),
        )
        self._draft_undo_stacks.setdefault(context, []).append(entry)
        self._signal_autosave()
        return True

    @property
    def retained_history_tokens(self) -> frozenset[UndoToken]:
        """Identify session metadata still reachable through Undo or Redo."""
        return frozenset(entry.token for entry in (*self._undo_stack, *self._redo_stack))

    def is_history_token_applied(self, token: UndoToken) -> bool:
        """Distinguish an applied change from one retained only for Redo."""
        return any(entry.token == token for entry in self._undo_stack)

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
        self._require_mutation_allowed()
        self.detach_owned_assets()
        self._document = validated_copy(document)
        self._edit_drafts = self._document_edit_drafts(self._document)
        self._draft_undo_stacks.clear()
        self._draft_redo_stacks.clear()
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._pending_persisted_change = None
        return self.document

    def execute(self, command: DocumentCommand) -> Stack:
        """Apply one command and record one session undo boundary."""
        self._require_mutation_allowed()
        before = self._document
        after = self._prepare_command_result(
            before,
            validated_copy(command.apply(validated_copy(before))),
        )
        self._adopt_committed_drafts(before, after)
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
        owned_assets: Collection[OwnedAsset] = (),
        on_recorded: HistoryRecordedHook | None = None,
    ) -> Stack:
        """Persist a command result before exposing it or recording history."""
        self._require_mutation_allowed()
        before = self._document
        after = self._prepare_command_result(
            before,
            validated_copy(command.apply(validated_copy(before))),
        )
        if after == before:
            return self.document
        try:
            self._persistence_in_progress = True
            try:
                persist(validated_copy(after))
            finally:
                self._persistence_in_progress = False
        except Exception as error:
            persisted_after = getattr(error, "persisted_stack", None) == after
            durability_indeterminate = bool(getattr(error, "durability_indeterminate", False))
            observed_stack = getattr(error, "observed_stack", None)
            retained_owned_asset = getattr(error, "owned_asset", None) is not None
            if persisted_after:
                self._adopt_committed_drafts(before, after)
                self._record_change(before, after, on_recorded=on_recorded)
            elif durability_indeterminate:
                if isinstance(observed_stack, Stack):
                    observed = validated_copy(observed_stack)
                    if observed == after:
                        self._adopt_committed_drafts(before, after)
                    self._document = self._merge_authoritative_drafts(observed)
                self._pending_persisted_change = (
                    _PendingPersistedChange(before=before, after=after, on_recorded=on_recorded)
                    if observed_stack == after
                    else None
                )
            if persisted_after or retained_owned_asset or durability_indeterminate:
                for asset in tuple(owned_assets):
                    key = (asset.bundle_path, asset.relative_path)
                    self._owned_assets[key] = asset
                    if durability_indeterminate:
                        self._durability_pending_assets.add(key)
                if not durability_indeterminate:
                    self._release_unreachable_owned_assets()
            raise
        self._pending_persisted_change = None
        self._adopt_committed_drafts(before, after)
        self._record_change(before, after, on_recorded=on_recorded)
        for asset in tuple(owned_assets):
            self._owned_assets[(asset.bundle_path, asset.relative_path)] = asset
        self._durability_pending_assets.clear()
        self._release_unreachable_owned_assets()
        return self.document

    def confirm_persisted_document(self, document: Stack) -> None:
        """Finalize history and asset cleanup after one durable retry save."""
        persisted = validated_copy(document)
        pending = self._pending_persisted_change
        if pending is not None and persisted == pending.after and self._document == pending.after:
            self._adopt_committed_drafts(pending.before, pending.after)
            self._record_change(
                pending.before,
                pending.after,
                on_recorded=pending.on_recorded,
            )
        self._pending_persisted_change = None
        self._durability_pending_assets.clear()
        self._release_unreachable_owned_assets()

    def _record_change(
        self,
        before: Stack,
        after: Stack,
        *,
        on_recorded: HistoryRecordedHook | None = None,
    ) -> None:
        if after != before:
            self._document = after
            token = UndoToken(self._next_undo_sequence)
            self._next_undo_sequence += 1
            self._undo_stack.append(_HistoryEntry(before=before, after=after, token=token))
            self._redo_stack.clear()
            self._prune_draft_state()
            if on_recorded is not None:
                on_recorded(token)

    def undo(self) -> bool:
        """Undo the latest command in this session."""
        self._require_mutation_allowed()
        if not self._undo_stack:
            return False
        entry = self._undo_stack.pop()
        self._document = self._restore_history_document(
            target=entry.before,
            source=entry.after,
        )
        self._redo_stack.append(entry)
        self._signal_autosave()
        self._release_unreachable_owned_assets()
        return True

    def undo_if_current(self, token: UndoToken) -> bool:
        """Undo only when the identified change is still the latest mutation."""
        self._require_mutation_allowed()
        if not self._undo_stack or self._undo_stack[-1].token != token:
            return False
        return self.undo()

    def redo(self) -> bool:
        """Redo the latest command undone in this session."""
        self._require_mutation_allowed()
        if not self._redo_stack:
            return False
        entry = self._redo_stack.pop()
        self._document = self._restore_history_document(
            target=entry.after,
            source=entry.before,
        )
        self._undo_stack.append(entry)
        self._signal_autosave()
        self._release_unreachable_owned_assets()
        return True

    def clear_history(self) -> None:
        """Discard undo and redo state without changing the document."""
        self._require_mutation_allowed()
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._draft_undo_stacks.clear()
        self._draft_redo_stacks.clear()
        self._prune_draft_state()
        self._release_unreachable_owned_assets()

    def register_owned_assets(
        self,
        assets: tuple[OwnedAsset, ...],
    ) -> None:
        """Track persisted duplicate assets loaded into the active document."""
        for asset in assets:
            self._owned_assets[(asset.bundle_path, asset.relative_path)] = asset
        self._release_unreachable_owned_assets()

    def detach_owned_assets(self) -> None:
        """Release unreachable assets and forget those retained by this document."""
        self._require_mutation_allowed()
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._draft_undo_stacks.clear()
        self._draft_redo_stacks.clear()
        self._prune_draft_state()
        current_paths = self._asset_paths(self._document)
        released = tuple(
            asset
            for key, asset in self._owned_assets.items()
            if key not in self._durability_pending_assets
            and asset.relative_path not in current_paths
        )
        self._owned_assets = {
            key: asset
            for key, asset in self._owned_assets.items()
            if key in self._durability_pending_assets
        }
        self._signal_owned_asset_release(released)

    def _set_current_edit_draft(
        self,
        context: DraftContext,
        draft: EditDraft,
    ) -> None:
        self._edit_drafts[context] = draft
        self._document = self._replace_document_drafts(
            self._document,
            {context: draft},
        )

    def _prepare_command_result(self, before: Stack, candidate: Stack) -> Stack:
        before_drafts = self._document_edit_drafts(before)
        candidate_drafts = self._document_edit_drafts(candidate)
        replacements: dict[DraftContext, EditDraft] = {}
        for context, candidate_draft in candidate_drafts.items():
            before_draft = before_drafts.get(context)
            if before_draft is not None and candidate_draft == before_draft:
                replacements[context] = self._edit_drafts.get(context, candidate_draft)
            else:
                replacements[context] = candidate_draft
        return self._replace_document_drafts(candidate, replacements)

    def _adopt_committed_drafts(self, before: Stack, after: Stack) -> None:
        before_drafts = self._document_edit_drafts(before)
        after_drafts = self._document_edit_drafts(after)
        for context, draft in after_drafts.items():
            previous = before_drafts.get(context)
            if previous is None or draft != previous:
                self._edit_drafts[context] = draft
            if previous is not None and draft != previous:
                self._draft_undo_stacks.pop(context, None)
                self._draft_redo_stacks.pop(context, None)

    def _merge_authoritative_drafts(self, document: Stack) -> Stack:
        replacements = {
            context: self._edit_drafts.get(context, draft)
            for context, draft in self._document_edit_drafts(document).items()
        }
        for context, draft in replacements.items():
            self._edit_drafts.setdefault(context, draft)
        return self._replace_document_drafts(document, replacements)

    def _restore_history_document(self, *, target: Stack, source: Stack) -> Stack:
        target_drafts = self._document_edit_drafts(target)
        source_drafts = self._document_edit_drafts(source)
        replacements: dict[DraftContext, EditDraft] = {}
        for context, target_draft in target_drafts.items():
            current_draft = self._edit_drafts.get(context)
            source_draft = source_drafts.get(context)
            if (
                source_draft is not None
                and source_draft != target_draft
                and current_draft == source_draft
            ):
                current_draft = target_draft
                self._edit_drafts[context] = target_draft
            elif current_draft is None:
                current_draft = target_draft
                self._edit_drafts[context] = target_draft
            replacements[context] = current_draft
        return self._replace_document_drafts(target, replacements)

    def _prune_draft_state(self) -> None:
        retained_contexts = set(self._current_draft_contexts())
        for entry in (*self._undo_stack, *self._redo_stack):
            retained_contexts.update(self._document_edit_drafts(entry.before))
            retained_contexts.update(self._document_edit_drafts(entry.after))
        self._edit_drafts = {
            context: draft
            for context, draft in self._edit_drafts.items()
            if context in retained_contexts
        }
        self._draft_undo_stacks = {
            context: entries
            for context, entries in self._draft_undo_stacks.items()
            if context in retained_contexts
        }
        self._draft_redo_stacks = {
            context: entries
            for context, entries in self._draft_redo_stacks.items()
            if context in retained_contexts
        }

    def _current_draft_contexts(self) -> frozenset[DraftContext]:
        return frozenset(self._document_edit_drafts(self._document))

    @staticmethod
    def _document_edit_drafts(document: Stack) -> dict[DraftContext, EditDraft]:
        return {
            (card.id, revision.id): revision.edit_draft
            for card in document.cards
            for revision in card.revisions
        }

    @staticmethod
    def _replace_document_drafts(
        document: Stack,
        replacements: dict[DraftContext, EditDraft],
    ) -> Stack:
        cards = []
        changed = False
        for card in document.cards:
            revisions = []
            card_changed = False
            for revision in card.revisions:
                draft = replacements.get((card.id, revision.id), revision.edit_draft)
                if draft != revision.edit_draft:
                    revision = revision.model_copy(update={"edit_draft": draft})
                    card_changed = True
                revisions.append(revision)
            if card_changed:
                card = card.model_copy(update={"revisions": tuple(revisions)})
                changed = True
            cards.append(card)
        if not changed:
            return document
        return validated_copy(document.model_copy(update={"cards": tuple(cards)}))

    def _require_mutation_allowed(self) -> None:
        if self._persistence_in_progress:
            raise DocumentMutationBlockedError("Wait for the current document change to finish.")
        if self.mutation_blocked:
            raise DocumentMutationBlockedError(PENDING_DURABILITY_MESSAGE)

    def _signal_autosave(self) -> None:
        if self._autosave_hook is not None:
            self._autosave_hook(self.document)

    def _release_unreachable_owned_assets(self) -> None:
        reachable_paths = set(self._asset_paths(self._document))
        for entry in (*self._undo_stack, *self._redo_stack):
            reachable_paths.update(self._asset_paths(entry.before))
            reachable_paths.update(self._asset_paths(entry.after))
        released = tuple(
            asset
            for key, asset in self._owned_assets.items()
            if key not in self._durability_pending_assets
            and asset.relative_path not in reachable_paths
        )
        for asset in released:
            self._owned_assets.pop((asset.bundle_path, asset.relative_path), None)
        self._signal_owned_asset_release(released)

    def _signal_owned_asset_release(
        self,
        assets: tuple[OwnedAsset, ...],
    ) -> None:
        if assets and self._owned_asset_release_hook is not None:
            self._owned_asset_release_hook(assets)

    @staticmethod
    def _asset_paths(document: Stack) -> frozenset[str]:
        image_paths = {
            revision.background.image_path
            for card in document.cards
            for revision in card.revisions
            if revision.background is not None
        }
        sound_paths = {
            sound.generated.audio_path for sound in document.sounds if sound.generated is not None
        }
        return frozenset((*image_paths, *sound_paths))


__all__ = [
    "AutosaveHook",
    "DocumentController",
    "DocumentMutationBlockedError",
    "OwnedAsset",
    "OwnedImageAsset",
    "OwnedSoundAsset",
    "PENDING_DURABILITY_MESSAGE",
    "UndoToken",
]
