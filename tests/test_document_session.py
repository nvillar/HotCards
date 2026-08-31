"""Tests for bound document lifecycle and debounced autosave."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PIL import Image
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from hotcards.application.commands import CreateCardCommand, RenameCardCommand
from hotcards.application.document_controller import DocumentController, OwnedImageAsset
from hotcards.application.document_session import DocumentSession, DocumentSessionError
from hotcards.domain.models import (
    Card,
    CardRevision,
    DirectGenerateProvenance,
    DuplicateProvenance,
    GeneratedBackground,
    GenerateInputs,
    ImageOperationSettings,
    ImageSourceSnapshot,
    Stack,
)
from hotcards.storage.stack_store import StackStore, StackStoreError


@pytest.fixture(scope="module")
def application() -> QApplication:
    return QApplication.instance() or QApplication([])


def _duplicate_bundle(
    tmp_path: Path,
    *,
    symlink_owned_image: bool,
) -> tuple[StackStore, Stack]:
    store = StackStore(tmp_path / "Candidate.hotcards")
    card = Card(name="Candidate")
    asset_id = uuid4()
    generated_at = datetime.now(UTC)
    source_png = tmp_path / "candidate.png"
    Image.new("RGB", (19, 13), (12, 34, 56)).save(source_png, format="PNG")
    image_path = store.store_image_asset(
        source_png,
        card_id=card.id,
        asset_id=asset_id,
    )
    revision = CardRevision(
        background=GeneratedBackground(
            id=asset_id,
            image_path=image_path,
            provenance=DuplicateProvenance(
                source=ImageSourceSnapshot(
                    card_id=uuid4(),
                    revision_id=uuid4(),
                    background_id=uuid4(),
                ),
                original_provenance=DirectGenerateProvenance(
                    inputs=GenerateInputs(description="Candidate"),
                    render_prompt="Candidate",
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
            ),
            created_at=generated_at,
        ),
    )
    card = card.model_copy(
        update={
            "revisions": (revision,),
            "active_revision_id": revision.id,
        }
    )
    stack = Stack(name="Candidate", cards=(card,), start_card_id=card.id)
    store.save(stack)

    if symlink_owned_image:
        target = store.bundle_path.joinpath(*image_path.split("/"))
        alternate = store.bundle_path / "assets" / "alternate.png"
        alternate.write_bytes(target.read_bytes())
        target.unlink()
        target.symlink_to(Path("..") / ".." / "alternate.png")
    assert store.load() == stack
    return store, stack


def test_create_binds_bundle_before_mutations_and_flushes_autosave(
    tmp_path: Path,
) -> None:
    controller = DocumentController(Stack(name="Welcome"))
    session = DocumentSession(controller)
    first_card = Card(name="Card 1")
    created = Stack(
        name="Garden",
        cards=(first_card,),
        start_card_id=first_card.id,
    )
    bundle = tmp_path / "Garden.hotcards"

    session.create(created, bundle)
    assert session.store is not None
    assert session.store.load() == created
    assert not session.state.dirty

    controller.execute(RenameCardCommand(card_id=first_card.id, name="Pergola"))
    assert session.state.dirty
    assert session.flush()
    assert StackStore(bundle).load().cards[0].name == "Pergola"
    assert not session.state.dirty


def test_autosave_runs_after_debounce(
    application: QApplication,
    tmp_path: Path,
) -> None:
    controller = DocumentController(Stack(name="Welcome"))
    session = DocumentSession(controller, debounce_milliseconds=10)
    bundle = tmp_path / "Debounced.hotcards"
    session.create(Stack(name="Debounced"), bundle)

    controller.execute(CreateCardCommand(name="Saved shortly"))
    assert session.state.dirty
    for _attempt in range(100):
        if not session.state.dirty:
            break
        QTest.qWait(10)

    assert [card.name for card in StackStore(bundle).load().cards] == ["Saved shortly"]
    assert not session.state.dirty


def test_failed_save_remains_dirty_and_can_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = DocumentController(Stack(name="Welcome"))
    session = DocumentSession(controller)
    session.create(Stack(name="Retry"), tmp_path / "Retry.hotcards")
    controller.execute(CreateCardCommand(name="Pending"))
    assert session.store is not None
    real_save = session.store.save

    def fail_save(_stack: Stack) -> None:
        raise StackStoreError("disk is unavailable")

    monkeypatch.setattr(session.store, "save", fail_save)
    assert not session.flush()
    assert session.state.dirty
    assert session.state.error == "disk is unavailable"

    monkeypatch.setattr(session.store, "save", real_save)
    assert session.flush()
    assert not session.state.dirty
    assert session.state.error is None


def test_invalid_open_preserves_current_document(tmp_path: Path) -> None:
    controller = DocumentController(Stack(name="Welcome"))
    session = DocumentSession(controller)
    session.create(Stack(name="Current"), tmp_path / "Current.hotcards")
    invalid = tmp_path / "Invalid.hotcards"
    invalid.mkdir()
    (invalid / "stack.json").write_text("not json")

    with pytest.raises(DocumentSessionError, match="could not read"):
        session.open(invalid)

    assert controller.document.name == "Current"
    assert session.state.bundle_path == tmp_path / "Current.hotcards"


def test_secure_duplicate_asset_preflight_preserves_active_session(
    tmp_path: Path,
) -> None:
    first = Card(name="First")
    second = Card(name="Second")
    active = Stack(
        name="Active",
        cards=(first, second),
        start_card_id=first.id,
    )
    controller = DocumentController(active)
    session = DocumentSession(controller)
    active_store = StackStore(tmp_path / "Active.hotcards")
    session.create(active, active_store.bundle_path)
    controller.execute(RenameCardCommand(card_id=second.id, name="Pending edit"))
    before_document = controller.document
    before_store = session.store
    before_state = session.state
    before_token = controller.current_undo_token
    before_pending = session._pending_snapshot
    candidate_store, candidate_stack = _duplicate_bundle(
        tmp_path,
        symlink_owned_image=True,
    )
    replaced: list[Stack] = []
    states: list[object] = []
    session.document_replaced.connect(replaced.append)
    session.state_changed.connect(states.append)

    with pytest.raises(
        DocumentSessionError,
        match="could not securely open owned image",
    ):
        session.open(candidate_store.bundle_path)

    assert session.store is before_store
    assert session.state == before_state
    assert session._pending_snapshot == before_pending
    assert controller.document == before_document
    assert controller.current_undo_token == before_token
    assert controller.can_undo
    assert replaced == []
    assert states == []

    controller.execute(RenameCardCommand(card_id=first.id, name="Still active"))
    assert session.flush()
    assert active_store.load() == controller.document
    assert candidate_store.load() == candidate_stack


def test_candidate_owned_asset_is_revalidated_before_active_flush(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    card = Card(name="Active")
    active = Stack(name="Active", cards=(card,), start_card_id=card.id)
    controller = DocumentController(active)
    session = DocumentSession(controller)
    active_store = StackStore(tmp_path / "Active.hotcards")
    session.create(active, active_store.bundle_path)
    before_store = session.store
    controller.execute(RenameCardCommand(card_id=card.id, name="Pending edit"))
    before_document = controller.document
    before_state = session.state
    before_token = controller.current_undo_token
    before_pending = session._pending_snapshot
    candidate_store, _candidate_stack = _duplicate_bundle(
        tmp_path,
        symlink_owned_image=False,
    )
    real_classifier = session._duplicate_owned_assets
    candidate_classifications = 0

    def swap_after_first_classification(
        store: StackStore,
        document: Stack,
    ) -> tuple[OwnedImageAsset, ...]:
        nonlocal candidate_classifications
        assets = real_classifier(store, document)
        if store.bundle_path == candidate_store.bundle_path:
            candidate_classifications += 1
            if candidate_classifications == 1:
                owned_path = candidate_store.bundle_path.joinpath(
                    *assets[0].relative_path.split("/")
                )
                alternate = candidate_store.bundle_path / "assets" / "alternate.png"
                alternate.write_bytes(owned_path.read_bytes())
                owned_path.unlink()
                owned_path.symlink_to(Path("..") / ".." / "alternate.png")
        return assets

    monkeypatch.setattr(
        session,
        "_duplicate_owned_assets",
        swap_after_first_classification,
    )

    with pytest.raises(
        DocumentSessionError,
        match="could not securely open owned image",
    ):
        session.open(candidate_store.bundle_path)

    assert candidate_classifications == 1
    assert session.store is before_store
    assert session.state == before_state
    assert session._pending_snapshot == before_pending
    assert controller.document == before_document
    assert controller.current_undo_token == before_token


def test_reopening_active_bundle_flushes_before_reload(tmp_path: Path) -> None:
    controller = DocumentController(Stack(name="Welcome"))
    session = DocumentSession(controller)
    bundle = tmp_path / "Current.hotcards"
    session.create(Stack(name="Current"), bundle)
    controller.execute(CreateCardCommand(name="Pending"))
    assert session.state.dirty

    session.open(bundle)

    assert [card.name for card in controller.document.cards] == ["Pending"]
    assert [card.name for card in StackStore(bundle).load().cards] == ["Pending"]
    assert not session.state.dirty


def test_save_as_copies_assets_and_rebinds_autosave(tmp_path: Path) -> None:
    source_image = tmp_path / "source.png"
    Image.new("RGB", (32, 24), "green").save(source_image)
    source_bundle = tmp_path / "Source.hotcards"
    source_store = StackStore(source_bundle)
    card = Card(name="Garden")
    asset_id = uuid4()
    generated_at = datetime.now(UTC)
    revision = CardRevision(
        background=GeneratedBackground(
            id=asset_id,
            image_path=source_store.store_image_asset(
                source_image,
                card_id=card.id,
                asset_id=asset_id,
            ),
            provenance=DirectGenerateProvenance(
                inputs=GenerateInputs(
                    description="A garden",
                ),
                render_prompt="A garden",
                settings=ImageOperationSettings(
                    model_identifier="test",
                    mflux_version="test",
                    seed=1,
                    width=592,
                    height=448,
                    step_count=4,
                    generated_at=generated_at,
                    duration_seconds=1,
                ),
            ),
            created_at=generated_at,
        ),
    )
    card = card.model_copy(
        update={
            "revisions": (revision,),
            "active_revision_id": revision.id,
        }
    )
    stack = Stack(name="Source", cards=(card,), start_card_id=card.id)
    source_store.save(stack)

    controller = DocumentController(Stack(name="Welcome"))
    session = DocumentSession(controller)
    session.open(source_bundle)
    destination = tmp_path / "Copy.hotcards"
    session.save_as(destination)

    copied = StackStore(destination).load()
    assert copied == stack
    assert StackStore(destination).asset_path(revision.image_path).is_file()
    assert session.state.bundle_path == destination

    controller.execute(RenameCardCommand(card_id=card.id, name="Copied Garden"))
    assert session.flush()
    assert StackStore(destination).load().cards[0].name == "Copied Garden"
    assert StackStore(source_bundle).load().cards[0].name == "Garden"


def test_save_as_refuses_existing_destination(tmp_path: Path) -> None:
    controller = DocumentController(Stack(name="Welcome"))
    session = DocumentSession(controller)
    session.create(Stack(name="Current"), tmp_path / "Current.hotcards")
    existing = tmp_path / "Existing.hotcards"
    existing.mkdir()

    with pytest.raises(DocumentSessionError, match="refusing to overwrite"):
        session.save_as(existing)


def test_save_as_clears_history_that_can_reference_uncloned_assets(
    tmp_path: Path,
) -> None:
    controller = DocumentController(Stack(name="Welcome"))
    session = DocumentSession(controller)
    session.create(Stack(name="Current"), tmp_path / "Current.hotcards")
    controller.execute(CreateCardCommand(name="Undo-only state"))
    assert controller.undo()
    assert controller.can_redo

    session.save_as(tmp_path / "Copy.hotcards")

    assert not controller.can_undo
    assert not controller.can_redo


def test_create_can_recover_empty_bundle_left_by_failed_attempt(tmp_path: Path) -> None:
    controller = DocumentController(Stack(name="Welcome"))
    session = DocumentSession(controller)
    bundle = tmp_path / "Retry.hotcards"
    bundle.mkdir()

    session.create(Stack(name="Retry"), bundle)

    assert StackStore(bundle).load().name == "Retry"


def test_create_refuses_nonempty_existing_bundle(tmp_path: Path) -> None:
    controller = DocumentController(Stack(name="Welcome"))
    session = DocumentSession(controller)
    bundle = tmp_path / "Existing.hotcards"
    StackStore(bundle).save(Stack(name="Existing"))

    with pytest.raises(DocumentSessionError, match="bundle already exists"):
        session.create(Stack(name="Replacement"), bundle)


def test_create_does_not_replace_stack_created_after_destination_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = DocumentController(Stack(name="Welcome"))
    session = DocumentSession(controller)
    bundle = tmp_path / "Race.hotcards"
    bundle.mkdir()

    def competing_create() -> bool:
        StackStore(bundle).create(Stack(name="Competing"))
        return True

    monkeypatch.setattr(session, "flush", competing_create)

    with pytest.raises(DocumentSessionError, match="refusing to overwrite"):
        session.create(Stack(name="Replacement"), bundle)

    assert StackStore(bundle).load().name == "Competing"
