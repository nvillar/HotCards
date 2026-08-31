"""Tests for storage-safe active-card duplication."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from PIL import Image

from hotcards.application.card_duplication import (
    CardDuplicationError,
    CardDuplicationWorkflow,
)
from hotcards.application.commands import (
    CommandError,
    DeleteCardCommand,
    ReplaceRevisionBackgroundCommand,
)
from hotcards.application.document_controller import DocumentController
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
from hotcards.storage.stack_store import StackStore, StackStoreError


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

    def fail_command(_command: object) -> Stack:
        raise CommandError("command rejected")

    monkeypatch.setattr(session, "execute_persisted", fail_command)

    with pytest.raises(CardDuplicationError, match="command rejected"):
        workflow.duplicate(source.id)

    assert controller.document == before
    assert set(session.store.bundle_path.rglob("*.png")) == before_assets


def test_duplicate_persistence_failure_removes_only_new_asset(
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

    def fail_save(_stack: Stack) -> None:
        raise StackStoreError("disk unavailable")

    monkeypatch.setattr(session.store, "save", fail_save)

    with pytest.raises(CardDuplicationError, match="disk unavailable"):
        workflow.duplicate(source.id)

    assert controller.document == before
    assert set(session.store.bundle_path.rglob("*.png")) == before_assets


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

    monkeypatch.setattr(session.store, "copy_image_asset", fail_copy)

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
