"""Tests for storage-safe active-card duplication."""

from __future__ import annotations

import hashlib
import os
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from PIL import Image

import hotcards.storage.stack_store as stack_store_module
from hotcards.application.card_duplication import (
    CardDuplicationError,
    CardDuplicationWorkflow,
)
from hotcards.application.commands import (
    CommandError,
    DeleteCardCommand,
    DuplicateCardCommand,
    RenameCardCommand,
    ReplaceRevisionBackgroundCommand,
)
from hotcards.application.document_controller import (
    DocumentController,
    DocumentMutationBlockedError,
)
from hotcards.application.document_session import DocumentSession
from hotcards.domain.models import (
    Card,
    DirectGenerateProvenance,
    DuplicateProvenance,
    GeneratedBackground,
    GenerateInputs,
    ImageOperationSettings,
    Stack,
)
from hotcards.storage.stack_store import (
    StackStore,
    StackStoreError,
    StackStoreTransactionError,
)


def _checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bound_source(
    tmp_path: Path,
    *,
    with_background: bool,
) -> tuple[CardDuplicationWorkflow, DocumentController, DocumentSession, Card]:
    card = Card(name="Scene")
    controller = DocumentController(
        Stack(name="Stack", cards=(card,), start_card_id=card.id)
    )
    session = DocumentSession(controller)
    session.create(controller.document, tmp_path / "Stack.hotcards")
    if with_background:
        assert session.store is not None
        source_png = tmp_path / "source.png"
        Image.new("RGB", (19, 13), (12, 34, 56)).save(source_png, format="PNG")
        asset_id = uuid4()
        image_path = session.store.store_image_asset(
            source_png,
            card_id=card.id,
            asset_id=asset_id,
        )
        generated_at = datetime.now(UTC)
        background = GeneratedBackground(
            id=asset_id,
            image_path=image_path,
            provenance=DirectGenerateProvenance(
                inputs=GenerateInputs(description="Scene"),
                render_prompt="Scene",
                settings=ImageOperationSettings(
                    model_identifier="test",
                    mflux_version="test",
                    seed=7,
                    width=592,
                    height=448,
                    step_count=4,
                    generated_at=generated_at,
                    duration_seconds=1,
                ),
            ),
            created_at=generated_at,
        )
        controller.execute(
            ReplaceRevisionBackgroundCommand(
                card_id=card.id,
                revision_id=card.active_revision.id,
                background=background,
            )
        )
        assert session.flush()
        card = controller.document.cards[0]
    return (
        CardDuplicationWorkflow(controller, session),
        controller,
        session,
        card,
    )


@pytest.mark.parametrize("with_background", [False, True])
def test_duplicate_is_persisted_undoable_and_redoable(
    tmp_path: Path,
    with_background: bool,
) -> None:
    workflow, controller, session, source = _bound_source(
        tmp_path,
        with_background=with_background,
    )

    change = workflow.duplicate(source.id)

    assert [card.name for card in change.document.cards] == [
        "Scene",
        "Scene Copy",
    ]
    duplicate = change.document.cards[1]
    assert len(duplicate.revisions) == 1
    assert duplicate.active_revision.id != source.active_revision.id
    assert session.store is not None
    assert session.store.load() == change.document
    if with_background:
        source_background = source.active_revision.background
        duplicate_background = duplicate.active_revision.background
        assert source_background is not None
        assert duplicate_background is not None
        assert source_background.id != duplicate_background.id
        assert source_background.image_path != duplicate_background.image_path
        assert _checksum(
            session.store.asset_path(source_background.image_path)
        ) == _checksum(session.store.asset_path(duplicate_background.image_path))
        assert isinstance(duplicate_background.provenance, DuplicateProvenance)

    assert controller.undo_if_current(change.token)
    assert [card.name for card in controller.document.cards] == ["Scene"]
    assert controller.redo()
    assert controller.document == change.document
    if with_background:
        duplicate_background = controller.document.cards[1].active_revision.background
        assert duplicate_background is not None
        assert session.store.asset_path(duplicate_background.image_path).is_file()


def test_duplicate_remains_valid_after_source_card_is_deleted(
    tmp_path: Path,
) -> None:
    workflow, controller, session, source = _bound_source(
        tmp_path,
        with_background=True,
    )
    duplicate_change = workflow.duplicate(source.id)
    duplicate = duplicate_change.document.cards[1]
    duplicate_background = duplicate.active_revision.background
    assert duplicate_background is not None

    controller.execute(DeleteCardCommand(card_id=source.id))
    assert session.flush()

    loaded = StackStore(session.state.bundle_path).load()
    assert loaded.cards == (duplicate,)
    assert StackStore(session.state.bundle_path).asset_path(
        duplicate_background.image_path
    ).is_file()

    assert controller.undo()
    assert controller.undo_if_current(duplicate_change.token)
    assert controller.redo()
    assert controller.redo()
    assert controller.document.cards == (duplicate,)


def test_duplicate_save_failure_rolls_back_document_and_new_asset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow, controller, session, source = _bound_source(
        tmp_path,
        with_background=True,
    )
    assert session.store is not None
    before = controller.document
    before_assets = set(session.store.bundle_path.rglob("*.png"))

    def fail_command(
        _command: object,
        **_kwargs: object,
    ) -> Stack:
        raise CommandError("command rejected")

    monkeypatch.setattr(session, "execute_persisted", fail_command)

    with pytest.raises(CardDuplicationError, match="command rejected"):
        workflow.duplicate(source.id)

    assert controller.document == before
    assert set(session.store.bundle_path.rglob("*.png")) == before_assets


@pytest.mark.parametrize(
    "boundary",
    [
        "destination-created",
        "asset-file-fsynced",
        "asset-directory-fsynced",
        "cards-directory-fsynced",
        "manifest-written",
        "manifest-file-fsynced",
        "manifest-replaced",
        "manifest-directory-fsynced",
    ],
)
def test_duplicate_transaction_fault_restores_manifest_and_owned_asset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    workflow, controller, session, source = _bound_source(
        tmp_path,
        with_background=True,
    )
    assert session.store is not None
    before = controller.document
    before_assets = set(session.store.bundle_path.rglob("*.png"))

    previous_manifest = session.store.stack_path.read_bytes()
    sentinel = session.store.bundle_path / "preserve.txt"
    sentinel.write_bytes(b"pre-existing")
    injected = False

    def fail_boundary(name: str) -> None:
        nonlocal injected
        if name == boundary and not injected:
            injected = True
            raise OSError(f"fault after {boundary}")

    monkeypatch.setattr(stack_store_module, "_io_checkpoint", fail_boundary)

    with pytest.raises(CardDuplicationError, match=boundary):
        workflow.duplicate(source.id)

    assert injected
    assert controller.document == before
    assert session.store.stack_path.read_bytes() == previous_manifest
    assert session.store.load() == before
    assert set(session.store.bundle_path.rglob("*.png")) == before_assets
    assert sentinel.read_bytes() == b"pre-existing"


def test_blank_duplicate_post_replace_failure_restores_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow, controller, session, source = _bound_source(
        tmp_path,
        with_background=False,
    )
    assert session.store is not None
    before = controller.document
    previous_manifest = session.store.stack_path.read_bytes()
    injected = False

    def fail_after_replace(name: str) -> None:
        nonlocal injected
        if name == "manifest-replaced" and not injected:
            injected = True
            raise OSError("fault after manifest replace")

    monkeypatch.setattr(stack_store_module, "_io_checkpoint", fail_after_replace)

    with pytest.raises(CardDuplicationError, match="manifest replace"):
        workflow.duplicate(source.id)

    assert injected
    assert controller.document == before
    assert session.store.stack_path.read_bytes() == previous_manifest
    assert session.store.load() == before


def test_blank_duplicate_wraps_manifest_open_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow, controller, session, source = _bound_source(
        tmp_path,
        with_background=False,
    )
    real_open = stack_store_module.os.open

    def fail_stack_open(
        path: object,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if path == "stack.json" and dir_fd is not None:
            raise OSError("manifest unavailable")
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(stack_store_module.os, "open", fail_stack_open)

    with pytest.raises(CardDuplicationError, match="manifest unavailable"):
        workflow.duplicate(source.id)

    assert len(controller.document.cards) == 1


def test_indeterminate_observed_duplicate_becomes_history_only_after_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow, controller, session, source = _bound_source(
        tmp_path,
        with_background=True,
    )
    assert session.store is not None
    before = controller.document
    before_token = controller.current_undo_token
    real_write_manifest = stack_store_module._write_manifest_at
    injected = False

    def fail_after_replace(name: str) -> None:
        nonlocal injected
        if name == "manifest-replaced" and not injected:
            injected = True
            raise OSError("post-replace failure")

    def fail_rollback_manifest(
        bundle_fd: int,
        payload: bytes,
        *,
        prefix: str,
        checkpoints: bool,
    ) -> str:
        if not checkpoints:
            raise OSError("rollback write failure")
        return real_write_manifest(
            bundle_fd,
            payload,
            prefix=prefix,
            checkpoints=checkpoints,
        )

    monkeypatch.setattr(stack_store_module, "_io_checkpoint", fail_after_replace)
    monkeypatch.setattr(
        stack_store_module,
        "_write_manifest_at",
        fail_rollback_manifest,
    )

    with pytest.raises(
        CardDuplicationError,
        match="rollback also failed.*rollback write failure",
    ):
        workflow.duplicate(source.id)

    assert injected
    assert len(controller.document.cards) == 2
    assert session.store.load() == controller.document
    assert controller.current_undo_token == before_token
    assert controller.mutation_blocked
    assert session.state.mutation_blocked
    assert session.state.dirty
    assert "durability remains indeterminate" in (session.state.error or "")
    duplicate_background = controller.document.cards[1].active_revision.background
    assert duplicate_background is not None
    assert session.store.asset_path(duplicate_background.image_path).is_file()

    pending_document = controller.document
    blocked_operations = (
        lambda: controller.execute(
            RenameCardCommand(card_id=source.id, name="Blocked edit")
        ),
        controller.undo,
        controller.redo,
    )
    for operation in blocked_operations:
        with pytest.raises(
            DocumentMutationBlockedError,
            match="Save the stack to finish the pending image change",
        ):
            operation()
        assert controller.document == pending_document
        assert controller.current_undo_token == before_token
        assert controller.mutation_blocked

    assert session.flush()
    assert not session.state.dirty
    assert not session.state.mutation_blocked
    duplicate_token = controller.current_undo_token
    assert duplicate_token is not None
    assert duplicate_token != before_token
    assert session.store.load() == controller.document

    controller.execute(RenameCardCommand(card_id=source.id, name="After retry"))
    edit_token = controller.current_undo_token
    assert edit_token is not None
    assert edit_token != duplicate_token
    assert controller.undo()
    assert controller.document == pending_document
    assert controller.undo()
    assert controller.document == before


def test_duplicate_copy_failure_leaves_document_and_assets_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow, controller, session, source = _bound_source(
        tmp_path,
        with_background=True,
    )
    assert session.store is not None
    before = controller.document
    before_assets = set(session.store.bundle_path.rglob("*.png"))

    def fail_copy(*_args: object, **_kwargs: object) -> str:
        raise StackStoreError("copy failed")

    monkeypatch.setattr(
        session.store,
        "copy_image_asset_and_save",
        fail_copy,
    )

    with pytest.raises(CardDuplicationError, match="copy failed"):
        workflow.duplicate(source.id)

    assert controller.document == before
    assert set(session.store.bundle_path.rglob("*.png")) == before_assets


def test_duplicate_rejects_unsafe_source_asset_path_without_orphan(
    tmp_path: Path,
) -> None:
    workflow, controller, session, source = _bound_source(
        tmp_path,
        with_background=True,
    )
    assert source.active_revision.background is not None
    unsafe_background = source.active_revision.background.model_copy(
        update={"image_path": "../escape.png"}
    )
    unsafe_revision = source.active_revision.model_copy(
        update={"background": unsafe_background}
    )
    unsafe_card = source.model_copy(
        update={
            "revisions": (unsafe_revision,),
            "active_revision_id": unsafe_revision.id,
        }
    )
    controller.replace_document(
        controller.document.model_copy(update={"cards": (unsafe_card,)})
    )
    assert session.store is not None
    before_assets = set(session.store.bundle_path.rglob("*.png"))

    with pytest.raises(CardDuplicationError, match="unsafe traversal"):
        workflow.duplicate(source.id)

    assert controller.document.cards == (unsafe_card,)
    assert set(session.store.bundle_path.rglob("*.png")) == before_assets


def test_duplicate_rejects_source_path_from_another_asset_namespace(
    tmp_path: Path,
) -> None:
    workflow, controller, session, source = _bound_source(
        tmp_path,
        with_background=True,
    )
    assert source.active_revision.background is not None
    mismatched_background = source.active_revision.background.model_copy(
        update={
            "image_path": (
                f"assets/cards/{uuid4()}/"
                f"image-{source.active_revision.background.id}.png"
            )
        }
    )
    mismatched_revision = source.active_revision.model_copy(
        update={"background": mismatched_background}
    )
    mismatched_card = source.model_copy(
        update={
            "revisions": (mismatched_revision,),
            "active_revision_id": mismatched_revision.id,
        }
    )
    controller.replace_document(
        controller.document.model_copy(update={"cards": (mismatched_card,)})
    )
    assert session.store is not None
    before_assets = set(session.store.bundle_path.rglob("*.png"))

    with pytest.raises(CardDuplicationError, match="does not match"):
        workflow.duplicate(source.id)

    assert controller.document.cards == (mismatched_card,)
    assert set(session.store.bundle_path.rglob("*.png")) == before_assets


def test_secure_duplicate_rejects_symlink_source_without_reading_outside(
    tmp_path: Path,
) -> None:
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        pytest.skip("secure dirfd flags are unavailable")
    workflow, controller, session, source = _bound_source(
        tmp_path,
        with_background=True,
    )
    assert session.store is not None
    source_background = source.active_revision.background
    assert source_background is not None
    source_path = session.store.asset_path(source_background.image_path)
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside bytes")
    source_path.unlink()
    source_path.symlink_to(outside)
    before = controller.document

    with pytest.raises(CardDuplicationError, match="securely open source image"):
        workflow.duplicate(source.id)

    assert controller.document == before
    assert outside.read_bytes() == b"outside bytes"
    assert list(outside.parent.glob("image-*.png")) == []


def test_secure_duplicate_detects_source_swap_after_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        pytest.skip("secure dirfd flags are unavailable")
    workflow, controller, session, source = _bound_source(
        tmp_path,
        with_background=True,
    )
    assert session.store is not None
    source_background = source.active_revision.background
    assert source_background is not None
    source_path = session.store.asset_path(source_background.image_path)
    source_bytes = source_path.read_bytes()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside bytes")
    swapped = False

    def swap_source(name: str) -> None:
        nonlocal swapped
        if name == "source-opened" and not swapped:
            swapped = True
            source_path.unlink()
            source_path.symlink_to(outside)

    monkeypatch.setattr(stack_store_module, "_io_checkpoint", swap_source)
    try:
        with pytest.raises(CardDuplicationError, match="symbolic links"):
            workflow.duplicate(source.id)
    finally:
        if source_path.is_symlink():
            source_path.unlink()
            source_path.write_bytes(source_bytes)

    assert swapped
    assert outside.read_bytes() == b"outside bytes"
    assert session.store.load() == controller.document
    assert len(list(session.store.bundle_path.rglob("*.png"))) == 1


def test_secure_duplicate_parent_swap_never_writes_outside_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        pytest.skip("secure dirfd flags are unavailable")
    workflow, controller, session, source = _bound_source(
        tmp_path,
        with_background=True,
    )
    assert session.store is not None
    assets = session.store.bundle_path / "assets"
    cards = assets / "cards"
    held_cards = assets / "cards-held"
    outside = tmp_path / "outside-cards"
    outside.mkdir()
    swapped = False

    def swap_parent(name: str) -> None:
        nonlocal swapped
        if name == "destination-directory-opened" and not swapped:
            swapped = True
            cards.rename(held_cards)
            cards.symlink_to(outside, target_is_directory=True)

    monkeypatch.setattr(stack_store_module, "_io_checkpoint", swap_parent)
    try:
        with pytest.raises(CardDuplicationError, match="securely open asset directory"):
            workflow.duplicate(source.id)
    finally:
        if cards.is_symlink():
            cards.unlink()
        if held_cards.exists():
            held_cards.rename(cards)

    assert swapped
    assert list(outside.iterdir()) == []
    assert session.store.load() == controller.document
    assert len(list(session.store.bundle_path.rglob("*.png"))) == 1


def test_secure_duplicate_destination_swap_never_writes_or_deletes_outside(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        pytest.skip("secure dirfd flags are unavailable")
    workflow, controller, session, source = _bound_source(
        tmp_path,
        with_background=True,
    )
    assert session.store is not None
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside bytes")
    swapped_path: Path | None = None

    def swap_destination(name: str) -> None:
        nonlocal swapped_path
        if name != "destination-created" or swapped_path is not None:
            return
        cards_directory = session.store.bundle_path / "assets" / "cards"
        destination_directories = [
            path
            for path in cards_directory.iterdir()
            if path.name != str(source.id)
        ]
        assert len(destination_directories) == 1
        image_paths = list(destination_directories[0].glob("image-*.png"))
        assert len(image_paths) == 1
        swapped_path = image_paths[0]
        swapped_path.unlink()
        swapped_path.symlink_to(outside)

    monkeypatch.setattr(
        stack_store_module,
        "_io_checkpoint",
        swap_destination,
    )
    try:
        with pytest.raises(CardDuplicationError, match="symbolic links"):
            workflow.duplicate(source.id)
    finally:
        if swapped_path is not None and swapped_path.is_symlink():
            parent = swapped_path.parent
            swapped_path.unlink()
            parent.rmdir()

    assert swapped_path is not None
    assert outside.read_bytes() == b"outside bytes"
    assert session.store.load() == controller.document
    assert len(list(session.store.bundle_path.rglob("*.png"))) == 1


def test_duplicate_cards_directory_fsync_failure_precedes_manifest_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow, controller, session, source = _bound_source(
        tmp_path,
        with_background=True,
    )
    assert session.store is not None
    before = controller.document
    previous_manifest = session.store.stack_path.read_bytes()
    cards_path = session.store.bundle_path / "assets" / "cards"
    cards_stat = cards_path.stat()
    real_fsync = stack_store_module.os.fsync
    checkpoints: list[str] = []
    injected = False

    def record_checkpoint(name: str) -> None:
        checkpoints.append(name)

    def fail_cards_fsync(fd: int) -> None:
        nonlocal injected
        descriptor_stat = os.fstat(fd)
        if (
            not injected
            and descriptor_stat.st_dev == cards_stat.st_dev
            and descriptor_stat.st_ino == cards_stat.st_ino
        ):
            injected = True
            raise OSError("cards directory fsync failure")
        real_fsync(fd)

    monkeypatch.setattr(stack_store_module, "_io_checkpoint", record_checkpoint)
    monkeypatch.setattr(stack_store_module.os, "fsync", fail_cards_fsync)

    with pytest.raises(CardDuplicationError, match="cards directory fsync failure"):
        workflow.duplicate(source.id)

    assert injected
    assert "manifest-written" not in checkpoints
    assert controller.document == before
    assert session.store.stack_path.read_bytes() == previous_manifest
    assert session.store.load() == before
    assert len(list(session.store.bundle_path.rglob("*.png"))) == 1


def test_precommit_failure_uses_already_durable_previous_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow, controller, session, source = _bound_source(
        tmp_path,
        with_background=True,
    )
    assert session.store is not None
    before = controller.document
    previous_manifest = session.store.stack_path.read_bytes()
    before_assets = set(session.store.bundle_path.rglob("*.png"))
    real_write_manifest = stack_store_module._write_manifest_at
    injected = False
    rollback_attempted = False

    def fail_before_manifest(name: str) -> None:
        nonlocal injected
        if name == "cards-directory-fsynced" and not injected:
            injected = True
            raise OSError("pre-manifest failure")

    def fail_rollback_manifest(
        bundle_fd: int,
        payload: bytes,
        *,
        prefix: str,
        checkpoints: bool,
    ) -> str:
        nonlocal rollback_attempted
        if not checkpoints:
            rollback_attempted = True
            raise OSError("rollback write failure")
        return real_write_manifest(
            bundle_fd,
            payload,
            prefix=prefix,
            checkpoints=checkpoints,
        )

    monkeypatch.setattr(stack_store_module, "_io_checkpoint", fail_before_manifest)
    monkeypatch.setattr(
        stack_store_module,
        "_write_manifest_at",
        fail_rollback_manifest,
    )

    with pytest.raises(CardDuplicationError, match="pre-manifest failure"):
        workflow.duplicate(source.id)

    assert injected
    assert not rollback_attempted
    assert controller.document == before
    assert session.store.stack_path.read_bytes() == previous_manifest
    assert session.store.load() == before
    assert set(session.store.bundle_path.rglob("*.png")) == before_assets
    assert session.close_history()


def test_manifest_fsync_recovery_stabilizes_previous_state_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow, controller, session, source = _bound_source(
        tmp_path,
        with_background=True,
    )
    assert session.store is not None
    before = controller.document
    previous_manifest = session.store.stack_path.read_bytes()
    before_assets = set(session.store.bundle_path.rglob("*.png"))
    before_token = controller.current_undo_token
    bundle_stat = session.store.bundle_path.stat()
    real_fsync = stack_store_module.os.fsync
    failed_bundle_fsyncs = 0

    def fail_changed_and_rollback_fsync(fd: int) -> None:
        nonlocal failed_bundle_fsyncs
        descriptor_stat = os.fstat(fd)
        if (
            descriptor_stat.st_dev == bundle_stat.st_dev
            and descriptor_stat.st_ino == bundle_stat.st_ino
            and failed_bundle_fsyncs < 2
        ):
            failed_bundle_fsyncs += 1
            raise OSError(f"bundle directory fsync failure {failed_bundle_fsyncs}")
        real_fsync(fd)

    monkeypatch.setattr(
        stack_store_module.os,
        "fsync",
        fail_changed_and_rollback_fsync,
    )

    with pytest.raises(
        CardDuplicationError,
        match="bundle directory fsync failure 1.*bundle directory fsync failure 2",
    ):
        workflow.duplicate(source.id)

    assert failed_bundle_fsyncs == 2
    assert controller.document == before
    assert session.store.stack_path.read_bytes() == previous_manifest
    assert session.store.load() == before
    assert set(session.store.bundle_path.rglob("*.png")) == before_assets
    assert not session.state.dirty
    assert controller.current_undo_token == before_token


def test_repeated_manifest_fsync_failure_stays_dirty_until_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow, controller, session, source = _bound_source(
        tmp_path,
        with_background=True,
    )
    assert session.store is not None
    before = controller.document
    previous_manifest = session.store.stack_path.read_bytes()
    before_assets = set(session.store.bundle_path.rglob("*.png"))
    before_token = controller.current_undo_token
    bundle_stat = session.store.bundle_path.stat()
    real_fsync = stack_store_module.os.fsync
    failed_bundle_fsyncs = 0

    def fail_all_transaction_manifest_fsyncs(fd: int) -> None:
        nonlocal failed_bundle_fsyncs
        descriptor_stat = os.fstat(fd)
        if (
            descriptor_stat.st_dev == bundle_stat.st_dev
            and descriptor_stat.st_ino == bundle_stat.st_ino
            and failed_bundle_fsyncs < 3
        ):
            failed_bundle_fsyncs += 1
            raise OSError(f"bundle directory fsync failure {failed_bundle_fsyncs}")
        real_fsync(fd)

    monkeypatch.setattr(
        stack_store_module.os,
        "fsync",
        fail_all_transaction_manifest_fsyncs,
    )

    with pytest.raises(
        CardDuplicationError,
        match="manifest durability remains indeterminate",
    ):
        workflow.duplicate(source.id)

    assert failed_bundle_fsyncs == 3
    assert controller.document == before
    assert controller.current_undo_token == before_token
    assert not controller.mutation_blocked
    assert not session.state.mutation_blocked
    assert session.state.dirty
    assert "manifest durability remains indeterminate" in (session.state.error or "")
    assert session.store.stack_path.read_bytes() == previous_manifest
    assert session.store.load() == before
    retained_assets = set(session.store.bundle_path.rglob("*.png")) - before_assets
    assert len(retained_assets) == 1

    controller.clear_history()
    assert set(session.store.bundle_path.rglob("*.png")) - before_assets == retained_assets

    monkeypatch.setattr(stack_store_module.os, "fsync", real_fsync)
    assert session.flush()
    assert not session.state.dirty
    assert session.state.error is None
    assert set(session.store.bundle_path.rglob("*.png")) == before_assets
    assert controller.current_undo_token is None


def test_blank_duplicate_repeated_manifest_fsync_failure_stays_dirty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow, controller, session, source = _bound_source(
        tmp_path,
        with_background=False,
    )
    assert session.store is not None
    before = controller.document
    before_token = controller.current_undo_token
    bundle_stat = session.store.bundle_path.stat()
    real_fsync = stack_store_module.os.fsync
    failed_bundle_fsyncs = 0

    def fail_all_transaction_manifest_fsyncs(fd: int) -> None:
        nonlocal failed_bundle_fsyncs
        descriptor_stat = os.fstat(fd)
        if (
            descriptor_stat.st_dev == bundle_stat.st_dev
            and descriptor_stat.st_ino == bundle_stat.st_ino
            and failed_bundle_fsyncs < 3
        ):
            failed_bundle_fsyncs += 1
            raise OSError(f"bundle directory fsync failure {failed_bundle_fsyncs}")
        real_fsync(fd)

    monkeypatch.setattr(
        stack_store_module.os,
        "fsync",
        fail_all_transaction_manifest_fsyncs,
    )

    with pytest.raises(
        CardDuplicationError,
        match="manifest durability remains indeterminate",
    ):
        workflow.duplicate(source.id)

    assert failed_bundle_fsyncs == 3
    assert controller.document == before
    assert controller.current_undo_token == before_token
    assert session.store.load() == before
    assert session.state.dirty
    assert "manifest durability remains indeterminate" in (session.state.error or "")

    monkeypatch.setattr(stack_store_module.os, "fsync", real_fsync)
    assert session.flush()
    assert not session.state.dirty
    assert session.state.error is None
    assert controller.current_undo_token == before_token


def test_rollback_directory_cleanup_preserves_swapped_foreign_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        pytest.skip("secure dirfd flags are unavailable")
    workflow, controller, session, source = _bound_source(
        tmp_path,
        with_background=True,
    )
    assert session.store is not None
    before = controller.document
    cards_directory = session.store.bundle_path / "assets" / "cards"
    owned_renamed = cards_directory / "owned-renamed"
    foreign_directory: Path | None = None
    foreign_inode: int | None = None
    operation_failed = False
    swapped = False

    def fail_then_swap(name: str) -> None:
        nonlocal foreign_directory, foreign_inode, operation_failed, swapped
        if name == "cards-directory-fsynced" and not operation_failed:
            operation_failed = True
            raise OSError("force rollback")
        if name != "destination-directory-cleanup" or swapped:
            return
        destination_directories = [
            path
            for path in cards_directory.iterdir()
            if path.name != str(source.id)
        ]
        assert len(destination_directories) == 1
        original_directory = destination_directories[0]
        original_directory.rename(owned_renamed)
        original_directory.mkdir()
        (original_directory / "foreign.txt").write_bytes(b"foreign")
        foreign_directory = original_directory
        foreign_inode = original_directory.stat().st_ino
        swapped = True

    monkeypatch.setattr(stack_store_module, "_io_checkpoint", fail_then_swap)

    with pytest.raises(CardDuplicationError, match="force rollback"):
        workflow.duplicate(source.id)

    assert operation_failed
    assert swapped
    assert controller.document == before
    assert session.store.load() == before
    assert foreign_directory is not None
    assert foreign_directory.is_dir()
    assert foreign_directory.stat().st_ino == foreign_inode
    assert (foreign_directory / "foreign.txt").read_bytes() == b"foreign"
    assert owned_renamed.is_dir()
    assert list(owned_renamed.iterdir()) == []
    assert len(list(session.store.bundle_path.rglob("*.png"))) == 1

    (foreign_directory / "foreign.txt").unlink()
    foreign_directory.rmdir()
    owned_renamed.rmdir()


def test_rollback_file_quarantine_preserves_post_precheck_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        pytest.skip("secure dirfd flags are unavailable")
    _workflow, controller, session, source = _bound_source(
        tmp_path,
        with_background=True,
    )
    assert session.store is not None
    source_background = source.active_revision.background
    assert source_background is not None
    duplicate_card_id = uuid4()
    duplicate_revision_id = uuid4()
    duplicate_background_id = uuid4()
    duplicate_path = session.store.image_asset_path(
        duplicate_card_id,
        duplicate_background_id,
    )
    command = DuplicateCardCommand(
        source_card_id=source.id,
        name="Scene Copy",
        card_id=duplicate_card_id,
        revision_id=duplicate_revision_id,
        background_id=duplicate_background_id,
        background_image_path=duplicate_path,
    )
    before = controller.document
    changed = command.apply(before)
    destination_directory = (
        session.store.bundle_path / "assets" / "cards" / str(duplicate_card_id)
    )
    destination_file = destination_directory / Path(duplicate_path).name
    held_owned_file = destination_directory / "owned-held.png"
    operation_failed = False
    swapped = False

    def fail_then_swap(name: str) -> None:
        nonlocal operation_failed, swapped
        if name == "cards-directory-fsynced" and not operation_failed:
            operation_failed = True
            raise OSError("force rollback")
        if name == "owned-file-cleanup-prechecked" and not swapped:
            destination_file.rename(held_owned_file)
            destination_file.write_bytes(b"foreign bytes")
            swapped = True

    monkeypatch.setattr(stack_store_module, "_io_checkpoint", fail_then_swap)

    with pytest.raises(
        StackStoreTransactionError,
        match="foreign identity.*preserved",
    ) as captured:
        session.store.copy_image_asset_and_save(
            source_background.image_path,
            source_card_id=source.id,
            source_asset_id=source_background.id,
            destination_card_id=duplicate_card_id,
            destination_asset_id=duplicate_background_id,
            previous_stack=before,
            changed_stack=changed,
        )

    assert operation_failed
    assert swapped
    assert captured.value.owned_asset is not None
    assert session.store.load() == before
    assert held_owned_file.is_file()
    quarantines = list(
        destination_directory.glob(
            f".{destination_file.name}.owned-file-*.tmp"
        )
    )
    assert len(quarantines) == 1
    assert quarantines[0].read_bytes() == b"foreign bytes"

    held_owned_file.unlink()
    quarantines[0].unlink()
    destination_directory.rmdir()


def test_history_directory_quarantine_preserves_post_precheck_swap_for_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        pytest.skip("secure dirfd flags are unavailable")
    workflow, controller, session, source = _bound_source(
        tmp_path,
        with_background=True,
    )
    assert session.store is not None
    change = workflow.duplicate(source.id)
    duplicate = change.document.cards[1]
    duplicate_background = duplicate.active_revision.background
    assert duplicate_background is not None
    duplicate_path = session.store.asset_path(duplicate_background.image_path)
    duplicate_directory = duplicate_path.parent
    held_owned_directory = duplicate_directory.with_name("owned-held-directory")
    swapped = False

    assert controller.undo_if_current(change.token)
    controller.execute(RenameCardCommand(card_id=source.id, name="Renamed"))

    def swap_directory(name: str) -> None:
        nonlocal swapped
        if name != "owned-directory-cleanup-prechecked" or swapped:
            return
        duplicate_directory.rename(held_owned_directory)
        duplicate_directory.mkdir()
        (duplicate_directory / "foreign.txt").write_bytes(b"foreign directory")
        swapped = True

    monkeypatch.setattr(stack_store_module, "_io_checkpoint", swap_directory)

    assert session.flush()
    assert swapped
    assert "changed identity or contents" in (session.state.error or "")
    assert held_owned_directory.is_dir()
    assert list(held_owned_directory.iterdir()) == []
    quarantines = list(
        duplicate_directory.parent.glob(
            f".{duplicate_directory.name}.owned-directory-*.tmp"
        )
    )
    assert len(quarantines) == 1
    assert (quarantines[0] / "foreign.txt").read_bytes() == b"foreign directory"

    (quarantines[0] / "foreign.txt").unlink()
    quarantines[0].rmdir()
    held_owned_directory.rename(duplicate_directory)
    monkeypatch.setattr(stack_store_module, "_io_checkpoint", lambda _name: None)

    assert session.flush()
    assert session.state.error is None
    assert not duplicate_directory.exists()


def test_undo_then_new_command_reclaims_duplicate_owned_asset(
    tmp_path: Path,
) -> None:
    workflow, controller, session, source = _bound_source(
        tmp_path,
        with_background=True,
    )
    assert session.store is not None
    change = workflow.duplicate(source.id)
    duplicate_background = change.document.cards[1].active_revision.background
    source_background = source.active_revision.background
    assert duplicate_background is not None
    assert source_background is not None
    duplicate_path = session.store.asset_path(duplicate_background.image_path)
    source_path = session.store.asset_path(source_background.image_path)

    assert controller.undo_if_current(change.token)
    assert duplicate_path.is_file()
    controller.execute(RenameCardCommand(card_id=source.id, name="Renamed"))
    assert session.flush()

    assert not duplicate_path.exists()
    assert source_path.is_file()
    assert not controller.can_redo


def test_history_clear_retains_current_duplicate_but_reclaims_undone_copy(
    tmp_path: Path,
) -> None:
    workflow, controller, session, source = _bound_source(
        tmp_path,
        with_background=True,
    )
    assert session.store is not None
    change = workflow.duplicate(source.id)
    duplicate_background = change.document.cards[1].active_revision.background
    assert duplicate_background is not None
    duplicate_path = session.store.asset_path(duplicate_background.image_path)

    controller.clear_history()
    assert duplicate_path.is_file()

    second_change = workflow.duplicate(source.id)
    second_background = second_change.document.cards[1].active_revision.background
    assert second_background is not None
    second_path = session.store.asset_path(second_background.image_path)
    assert controller.undo_if_current(second_change.token)
    assert second_path.is_file()
    controller.clear_history()
    assert session.flush()

    assert not second_path.exists()
    assert duplicate_path.is_file()


def test_deleted_duplicate_asset_is_retained_only_while_undo_can_restore_it(
    tmp_path: Path,
) -> None:
    workflow, controller, session, source = _bound_source(
        tmp_path,
        with_background=True,
    )
    assert session.store is not None
    change = workflow.duplicate(source.id)
    duplicate = change.document.cards[1]
    duplicate_background = duplicate.active_revision.background
    assert duplicate_background is not None
    duplicate_path = session.store.asset_path(duplicate_background.image_path)

    controller.execute(DeleteCardCommand(card_id=duplicate.id))
    assert session.flush()
    assert duplicate_path.is_file()
    controller.clear_history()

    assert not duplicate_path.exists()
    source_background = source.active_revision.background
    assert source_background is not None
    assert session.store.asset_path(source_background.image_path).is_file()


def test_project_replacement_reclaims_only_undone_duplicate_asset(
    tmp_path: Path,
) -> None:
    workflow, controller, session, source = _bound_source(
        tmp_path,
        with_background=True,
    )
    assert session.store is not None
    original_store = session.store
    source_background = source.active_revision.background
    assert source_background is not None
    source_path = original_store.asset_path(source_background.image_path)
    change = workflow.duplicate(source.id)
    duplicate_background = change.document.cards[1].active_revision.background
    assert duplicate_background is not None
    duplicate_path = original_store.asset_path(duplicate_background.image_path)
    assert controller.undo_if_current(change.token)
    assert session.flush()
    replacement_store = StackStore(tmp_path / "Replacement.hotcards")
    replacement_store.create(Stack(name="Replacement"))

    session.open(replacement_store.bundle_path)

    assert not duplicate_path.exists()
    assert source_path.is_file()
    assert controller.document.name == "Replacement"


def test_session_close_reclaims_undone_duplicate_asset(
    tmp_path: Path,
) -> None:
    workflow, controller, session, source = _bound_source(
        tmp_path,
        with_background=True,
    )
    assert session.store is not None
    source_background = source.active_revision.background
    assert source_background is not None
    source_path = session.store.asset_path(source_background.image_path)
    change = workflow.duplicate(source.id)
    duplicate_background = change.document.cards[1].active_revision.background
    assert duplicate_background is not None
    duplicate_path = session.store.asset_path(duplicate_background.image_path)
    assert controller.undo_if_current(change.token)
    assert session.flush()

    assert session.close_history()

    assert not duplicate_path.exists()
    assert source_path.is_file()


def test_history_cleanup_refuses_foreign_replacement_at_owned_path(
    tmp_path: Path,
) -> None:
    workflow, controller, session, source = _bound_source(
        tmp_path,
        with_background=True,
    )
    assert session.store is not None
    source_background = source.active_revision.background
    assert source_background is not None
    source_path = session.store.asset_path(source_background.image_path)
    change = workflow.duplicate(source.id)
    duplicate_background = change.document.cards[1].active_revision.background
    assert duplicate_background is not None
    duplicate_path = session.store.asset_path(duplicate_background.image_path)
    assert controller.undo_if_current(change.token)
    duplicate_path.unlink()
    duplicate_path.write_bytes(b"foreign replacement")

    controller.execute(RenameCardCommand(card_id=source.id, name="Renamed"))

    assert session.flush()
    assert "identity changed" in (session.state.error or "")
    assert duplicate_path.read_bytes() == b"foreign replacement"
    assert source_path.is_file()
    assert not session.close_history()
    assert session.close_history()
