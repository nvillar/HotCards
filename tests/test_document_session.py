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

from hypergen.application.commands import CreateCardCommand, RenameCardCommand
from hypergen.application.document_controller import DocumentController
from hypergen.application.document_session import DocumentSession, DocumentSessionError
from hypergen.domain.models import Card, ImageOrigin, ImageRevision, Stack
from hypergen.storage.stack_store import StackStore, StackStoreError


@pytest.fixture(scope="module")
def application() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_create_binds_bundle_before_mutations_and_flushes_autosave(
    tmp_path: Path,
) -> None:
    controller = DocumentController(Stack(name="Welcome"))
    session = DocumentSession(controller)
    first_card = Card(name="Card 1")
    created = Stack(
        name="Garden",
        art_direction="Pencil sketch",
        cards=(first_card,),
        start_card_id=first_card.id,
    )
    bundle = tmp_path / "Garden.hypergen"

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
    bundle = tmp_path / "Debounced.hypergen"
    session.create(Stack(name="Debounced"), bundle)

    controller.execute(CreateCardCommand(name="Saved shortly"))
    assert session.state.dirty
    QTest.qWait(30)

    assert [card.name for card in StackStore(bundle).load().cards] == ["Saved shortly"]
    assert not session.state.dirty


def test_failed_save_remains_dirty_and_can_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = DocumentController(Stack(name="Welcome"))
    session = DocumentSession(controller)
    session.create(Stack(name="Retry"), tmp_path / "Retry.hypergen")
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
    session.create(Stack(name="Current"), tmp_path / "Current.hypergen")
    invalid = tmp_path / "Invalid.hypergen"
    invalid.mkdir()
    (invalid / "stack.json").write_text("not json")

    with pytest.raises(DocumentSessionError, match="could not read"):
        session.open(invalid)

    assert controller.document.name == "Current"
    assert session.state.bundle_path == tmp_path / "Current.hypergen"


def test_reopening_active_bundle_flushes_before_reload(tmp_path: Path) -> None:
    controller = DocumentController(Stack(name="Welcome"))
    session = DocumentSession(controller)
    bundle = tmp_path / "Current.hypergen"
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
    source_bundle = tmp_path / "Source.hypergen"
    source_store = StackStore(source_bundle)
    card = Card(name="Garden")
    revision = ImageRevision(
        image_path=source_store.import_image(
            source_image,
            card_id=card.id,
            revision_id=(revision_id := uuid4()),
        ),
        id=revision_id,
        origin=ImageOrigin.IMPORTED,
        source_filename=source_image.name,
        created_at=datetime.now(UTC),
    )
    card = card.model_copy(
        update={
            "image_revisions": (revision,),
            "active_revision_id": revision.id,
        }
    )
    stack = Stack(name="Source", cards=(card,), start_card_id=card.id)
    source_store.save(stack)

    controller = DocumentController(Stack(name="Welcome"))
    session = DocumentSession(controller)
    session.open(source_bundle)
    destination = tmp_path / "Copy.hypergen"
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
    session.create(Stack(name="Current"), tmp_path / "Current.hypergen")
    existing = tmp_path / "Existing.hypergen"
    existing.mkdir()

    with pytest.raises(DocumentSessionError, match="refusing to overwrite"):
        session.save_as(existing)
