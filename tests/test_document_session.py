"""Tests for bound document lifecycle and debounced autosave."""

from __future__ import annotations

from pathlib import Path
from typing import Literal
from uuid import uuid4

import pytest
from helpers import DocumentSessionFactory, generated_background
from helpers import document_session_factory as document_session_factory
from PIL import Image
from PySide6.QtTest import QTest

from hotcards.application.commands import (
    CreateCardCommand,
    DeleteRevisionCommand,
    RenameCardCommand,
    ReplaceRevisionBackgroundCommand,
)
from hotcards.application.document_controller import DocumentController, OwnedImageAsset
from hotcards.application.document_session import DocumentSessionError
from hotcards.domain.image_dimensions import ResolutionTier
from hotcards.domain.models import (
    AcceptedEdit,
    Card,
    CardRevision,
    DerivedImageSourceSnapshot,
    DuplicateOperation,
    EditOperation,
    GeneratedBackground,
    GenerateOperation,
    ImageOriginFacts,
    ImageProvenance,
    ImageSourceSnapshot,
    PresetOutputSize,
    Stack,
)
from hotcards.storage.stack_store import StackStore, StackStoreError


def _duplicate_bundle(
    tmp_path: Path,
    *,
    symlink_owned_image: bool,
) -> tuple[StackStore, Stack]:
    store, stack, target = _owned_bundle(tmp_path, name="Candidate", operation="duplicate")

    if symlink_owned_image:
        alternate = store.bundle_path / "assets" / "alternate.png"
        alternate.write_bytes(target.read_bytes())
        target.unlink()
        target.symlink_to(Path("..") / ".." / "alternate.png")
    assert store.load() == stack
    return store, stack


def _owned_bundle(
    tmp_path: Path,
    *,
    name: str,
    operation: Literal["generate", "edit", "duplicate"],
) -> tuple[StackStore, Stack, Path]:
    store = StackStore(tmp_path / f"{name}.hotcards")
    card = Card(name=name)
    background = generated_background(card.id, description=name)
    direct = background.provenance
    assert isinstance(direct.authoring, GenerateOperation)
    source_revision: CardRevision | None = None
    if operation == "edit":
        source_background = generated_background(card.id, description=name)
        source_revision = CardRevision(background=source_background)
        source = DerivedImageSourceSnapshot(
            card_id=card.id,
            revision_id=source_revision.id,
            background_id=source_background.id,
            width=direct.settings.width,
            height=direct.settings.height,
            seed=direct.settings.seed,
        )
        provenance = ImageProvenance(
            origin=ImageOriginFacts(
                render_prompt="Add a lantern",
                settings=direct.settings.model_copy(update={"width": 768, "height": 576}),
                edit_lineage=(
                    AcceptedEdit(
                        instruction="Add a lantern",
                        expanded_prompt="Add a lantern",
                    ),
                ),
            ),
            authoring=EditOperation(
                source=source,
                output_size=PresetOutputSize(tier=ResolutionTier.LARGE),
                prompt_token_count=20,
            ),
        )
    elif operation == "duplicate":
        provenance = ImageProvenance(
            origin=direct.origin,
            authoring=DuplicateOperation(
                source=ImageSourceSnapshot(
                    card_id=uuid4(),
                    revision_id=uuid4(),
                    background_id=uuid4(),
                ),
                original_authoring=direct.authoring,
            ),
        )
    else:
        provenance = direct
    revision = CardRevision(
        background=GeneratedBackground(
            id=background.id,
            image_path=background.image_path,
            provenance=provenance,
            created_at=background.created_at,
        )
    )
    card = card.model_copy(
        update={
            "revisions": (
                (source_revision, revision) if source_revision is not None else (revision,)
            ),
            "active_revision_id": revision.id,
        }
    )
    for item in card.revisions:
        assert item.background is not None
        settings = item.background.provenance.settings
        source_png = tmp_path / f"{item.background.id}.png"
        Image.new("RGB", (settings.width, settings.height), (12, 34, 56)).save(source_png)
        image_path = store.store_image_asset(
            source_png,
            card_id=card.id,
            asset_id=item.background.id,
        )
        assert image_path == item.background.image_path
    stack = Stack(name=name, cards=(card,), start_card_id=card.id)
    store.save(stack)
    return store, stack, store.asset_path(background.image_path)


def test_create_binds_bundle_before_mutations_and_flushes_autosave(
    tmp_path: Path,
    document_session_factory: DocumentSessionFactory,
) -> None:
    controller = DocumentController(Stack(name="Welcome"))
    session = document_session_factory(controller)
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
    tmp_path: Path,
    document_session_factory: DocumentSessionFactory,
) -> None:
    controller = DocumentController(Stack(name="Welcome"))
    session = document_session_factory(controller, debounce_milliseconds=10)
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


def test_revision_edit_draft_survives_failed_autosave_retry_and_reopen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    document_session_factory: DocumentSessionFactory,
) -> None:
    card = Card(name="Card")
    controller = DocumentController(Stack(name="Welcome"))
    session = document_session_factory(controller)
    bundle = tmp_path / "Draft.hotcards"
    session.create(Stack(name="Draft", cards=(card,)), bundle)
    draft = controller.replace_edit_draft(
        card.id,
        card.active_revision.id,
        "  Unfinished instruction.\n",
    )
    assert session.store is not None
    real_save = session.store.save

    def fail_save(_stack: Stack) -> None:
        raise StackStoreError("disk is unavailable")

    monkeypatch.setattr(session.store, "save", fail_save)
    assert not session.flush()
    assert controller.edit_draft(card.id, card.active_revision.id) == draft
    assert session.state.dirty

    monkeypatch.setattr(session.store, "save", real_save)
    assert session.flush()
    reopened = StackStore(bundle).load()
    assert reopened.cards[0].active_revision.edit_draft == draft


def test_failed_save_remains_dirty_and_can_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    document_session_factory: DocumentSessionFactory,
) -> None:
    controller = DocumentController(Stack(name="Welcome"))
    session = document_session_factory(controller)
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


def test_invalid_open_preserves_current_document(
    tmp_path: Path, document_session_factory: DocumentSessionFactory
) -> None:
    controller = DocumentController(Stack(name="Welcome"))
    session = document_session_factory(controller)
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
    document_session_factory: DocumentSessionFactory,
) -> None:
    first = Card(name="First")
    second = Card(name="Second")
    active = Stack(
        name="Active",
        cards=(first, second),
        start_card_id=first.id,
    )
    controller = DocumentController(active)
    session = document_session_factory(controller)
    active_store = StackStore(tmp_path / "Active.hotcards")
    session.create(active, active_store.bundle_path)
    controller.execute(RenameCardCommand(card_id=second.id, name="Pending edit"))
    before_document = controller.document
    before_store = session.store
    before_state = session.state
    before_token = controller.current_undo_token
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
    assert controller.document == before_document
    assert controller.current_undo_token == before_token
    assert controller.can_undo
    assert replaced == []
    assert states == []

    controller.execute(RenameCardCommand(card_id=first.id, name="Still active"))
    assert session.flush()
    assert active_store.load() == controller.document
    assert candidate_store.load() == candidate_stack


def test_candidate_owned_asset_changed_after_early_preflight_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    document_session_factory: DocumentSessionFactory,
) -> None:
    card = Card(name="Active")
    active = Stack(name="Active", cards=(card,), start_card_id=card.id)
    controller = DocumentController(active)
    session = document_session_factory(controller)
    active_store = StackStore(tmp_path / "Active.hotcards")
    session.create(active, active_store.bundle_path)
    before_store = session.store
    controller.execute(RenameCardCommand(card_id=card.id, name="Pending edit"))
    before_document = controller.document
    before_token = controller.current_undo_token
    candidate_store, _candidate_stack = _duplicate_bundle(
        tmp_path,
        symlink_owned_image=False,
    )
    real_classifier = session._app_owned_assets
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
        "_app_owned_assets",
        swap_after_first_classification,
    )

    with pytest.raises(
        DocumentSessionError,
        match="could not securely open owned image",
    ):
        session.open(candidate_store.bundle_path)

    assert candidate_classifications == 1
    assert session.store is before_store
    assert session.state.bundle_path == active_store.bundle_path
    assert not session.state.dirty
    assert active_store.load() == before_document
    assert controller.document == before_document
    assert controller.current_undo_token == before_token


@pytest.mark.parametrize(
    "operation",
    ("generate", "edit", "duplicate"),
)
def test_open_and_history_close_retain_current_app_owned_backgrounds(
    tmp_path: Path,
    operation: Literal["generate", "edit", "duplicate"],
    document_session_factory: DocumentSessionFactory,
) -> None:
    store, stack, asset_path = _owned_bundle(
        tmp_path,
        name=operation,
        operation=operation,
    )
    controller = DocumentController(Stack(name="Welcome"))
    session = document_session_factory(controller)

    session.open(store.bundle_path)

    assert controller.document == stack
    controller.clear_history()
    assert asset_path.is_file()
    assert session.close_history()
    assert asset_path.is_file()


@pytest.mark.parametrize("operation", ("generate", "edit", "duplicate"))
def test_reopened_owned_background_is_removed_only_after_history_discards_it(
    tmp_path: Path,
    operation: Literal["generate", "edit", "duplicate"],
    document_session_factory: DocumentSessionFactory,
) -> None:
    store, stack, asset_path = _owned_bundle(
        tmp_path,
        name=operation,
        operation=operation,
    )
    unrelated = asset_path.parent / "unrelated.png"
    unrelated.write_bytes(b"unrelated")
    controller = DocumentController(Stack(name="Welcome"))
    session = document_session_factory(controller)
    session.open(store.bundle_path)
    card = controller.document.cards[0]
    revision = card.active_revision

    controller.execute(
        ReplaceRevisionBackgroundCommand(
            card_id=card.id,
            revision_id=revision.id,
            background=None,
        )
    )
    assert session.flush()
    assert asset_path.is_file()
    assert controller.undo()
    assert controller.document == stack
    assert asset_path.is_file()
    assert controller.redo()
    assert session.flush()
    controller.clear_history()

    assert not asset_path.exists()
    assert unrelated.read_bytes() == b"unrelated"
    assert store.load().cards[0].active_revision.background is None


def test_reopened_owned_background_is_removed_when_session_history_closes(
    tmp_path: Path,
    document_session_factory: DocumentSessionFactory,
) -> None:
    store, _stack, asset_path = _owned_bundle(
        tmp_path,
        name="edit-close",
        operation="edit",
    )
    controller = DocumentController(Stack(name="Welcome"))
    session = document_session_factory(controller)
    session.open(store.bundle_path)
    card = controller.document.cards[0]
    revision = card.active_revision
    controller.execute(
        ReplaceRevisionBackgroundCommand(
            card_id=card.id,
            revision_id=revision.id,
            background=None,
        )
    )
    assert session.flush()

    assert session.close_history()
    assert not asset_path.exists()


def test_edit_source_attribution_does_not_retain_deleted_source_assets(
    tmp_path: Path,
    document_session_factory: DocumentSessionFactory,
) -> None:
    store, _stack, asset_path = _owned_bundle(
        tmp_path,
        name="edit-source",
        operation="edit",
    )
    controller = DocumentController(Stack(name="Welcome"))
    session = document_session_factory(controller)
    session.open(store.bundle_path)
    assert session.store is not None
    card = controller.document.cards[0]
    source_revision = card.revisions[0]
    assert source_revision.background is not None
    source_path = session.store.asset_path(source_revision.background.image_path)
    result = card.active_revision.background
    assert result is not None
    assert isinstance(result.provenance.authoring, EditOperation)
    assert result.provenance.authoring.source.background_id == source_revision.background.id

    controller.execute(
        DeleteRevisionCommand(
            card_id=card.id,
            revision_id=source_revision.id,
        )
    )
    assert session.flush()
    assert source_path.is_file()
    assert controller.undo()
    assert controller.document.cards[0] == card
    assert controller.redo()
    assert session.flush()
    controller.clear_history()
    assert session.close_history()

    assert not source_path.exists()
    assert asset_path.is_file()
    reopened_card = store.load().cards[0]
    assert len(reopened_card.revisions) == 1
    assert reopened_card.active_revision.background == result


def test_open_reloads_candidate_changed_during_active_flush(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    document_session_factory: DocumentSessionFactory,
) -> None:
    active_card = Card(name="Active card")
    active = Stack(
        name="Active",
        cards=(active_card,),
        start_card_id=active_card.id,
    )
    controller = DocumentController(active)
    session = document_session_factory(controller)
    session.create(active, tmp_path / "Active.hotcards")
    assert session.store is not None
    active_store = session.store
    controller.execute(RenameCardCommand(card_id=active_card.id, name="Pending active edit"))

    candidate_store = StackStore(tmp_path / "Candidate.hotcards")
    candidate_card = Card(name="Candidate v1")
    candidate_v1 = Stack(
        name="Candidate v1",
        cards=(candidate_card,),
        start_card_id=candidate_card.id,
    )
    candidate_v2 = candidate_v1.model_copy(
        update={
            "name": "Candidate v2",
            "cards": (candidate_card.model_copy(update={"name": "Candidate v2 card"}),),
        }
    )
    candidate_store.save(candidate_v1)
    real_active_save = active_store.save

    def save_active_then_replace_candidate(snapshot: Stack) -> None:
        real_active_save(snapshot)
        candidate_store.save(candidate_v2)

    monkeypatch.setattr(active_store, "save", save_active_then_replace_candidate)
    replaced: list[Stack] = []
    session.document_replaced.connect(replaced.append)

    opened = session.open(candidate_store.bundle_path)

    assert opened == candidate_v2
    assert controller.document == candidate_v2
    assert session.store is not None
    assert session.store.bundle_path == candidate_store.bundle_path
    assert not session.state.dirty
    assert replaced == [candidate_v2]

    controller.execute(RenameCardCommand(card_id=candidate_card.id, name="Candidate v2 edited"))
    assert session.flush()
    persisted = candidate_store.load()
    assert persisted.name == "Candidate v2"
    assert persisted.cards[0].name == "Candidate v2 edited"


def test_open_rejects_candidate_made_invalid_during_active_flush(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    document_session_factory: DocumentSessionFactory,
) -> None:
    active_card = Card(name="Active card")
    active = Stack(
        name="Active",
        cards=(active_card,),
        start_card_id=active_card.id,
    )
    controller = DocumentController(active)
    session = document_session_factory(controller)
    session.create(active, tmp_path / "Active.hotcards")
    assert session.store is not None
    active_store = session.store
    controller.execute(RenameCardCommand(card_id=active_card.id, name="Persisted active edit"))
    candidate_store, candidate_stack = _duplicate_bundle(
        tmp_path,
        symlink_owned_image=False,
    )
    background = candidate_stack.cards[0].active_revision.background
    assert background is not None
    owned_path = candidate_store.bundle_path.joinpath(*background.image_path.split("/"))
    alternate = candidate_store.bundle_path / "assets" / "alternate.png"
    real_active_save = active_store.save

    def save_active_then_invalidate_candidate(snapshot: Stack) -> None:
        real_active_save(snapshot)
        alternate.write_bytes(owned_path.read_bytes())
        owned_path.unlink()
        owned_path.symlink_to(Path("..") / ".." / "alternate.png")

    monkeypatch.setattr(active_store, "save", save_active_then_invalidate_candidate)
    replaced: list[Stack] = []
    session.document_replaced.connect(replaced.append)

    with pytest.raises(
        DocumentSessionError,
        match="could not securely open owned image",
    ):
        session.open(candidate_store.bundle_path)

    assert session.store is active_store
    assert session.state.bundle_path == active_store.bundle_path
    assert controller.document.cards[0].name == "Persisted active edit"
    assert not session.state.dirty
    assert replaced == []
    assert candidate_store.load() == candidate_stack

    monkeypatch.setattr(active_store, "save", real_active_save)
    controller.execute(RenameCardCommand(card_id=active_card.id, name="Still active after failure"))
    assert session.flush()
    assert active_store.load().cards[0].name == "Still active after failure"
    assert candidate_store.load() == candidate_stack


def test_reopening_active_bundle_flushes_before_reload(
    tmp_path: Path, document_session_factory: DocumentSessionFactory
) -> None:
    controller = DocumentController(Stack(name="Welcome"))
    session = document_session_factory(controller)
    bundle = tmp_path / "Current.hotcards"
    session.create(Stack(name="Current"), bundle)
    controller.execute(CreateCardCommand(name="Pending"))
    assert session.state.dirty

    session.open(bundle)

    assert [card.name for card in controller.document.cards] == ["Pending"]
    assert [card.name for card in StackStore(bundle).load().cards] == ["Pending"]
    assert not session.state.dirty


def test_save_as_copies_assets_and_rebinds_autosave(
    tmp_path: Path, document_session_factory: DocumentSessionFactory
) -> None:
    source_store, stack, source_image = _owned_bundle(tmp_path, name="Source", operation="generate")
    card = stack.cards[0]
    revision = card.active_revision

    controller = DocumentController(Stack(name="Welcome"))
    session = document_session_factory(controller)
    session.open(source_store.bundle_path)
    destination = tmp_path / "Copy.hotcards"
    session.save_as(destination)

    copied = StackStore(destination).load()
    assert copied == stack
    assert (
        StackStore(destination).asset_path(revision.image_path).read_bytes()
        == source_image.read_bytes()
    )
    assert session.state.bundle_path == destination

    controller.execute(RenameCardCommand(card_id=card.id, name="Copied Garden"))
    assert session.flush()
    assert StackStore(destination).load().cards[0].name == "Copied Garden"
    assert source_store.load().cards[0].name == "Source"


def test_save_as_refuses_existing_destination(
    tmp_path: Path, document_session_factory: DocumentSessionFactory
) -> None:
    controller = DocumentController(Stack(name="Welcome"))
    session = document_session_factory(controller)
    session.create(Stack(name="Current"), tmp_path / "Current.hotcards")
    existing = tmp_path / "Existing.hotcards"
    existing.mkdir()

    with pytest.raises(DocumentSessionError, match="refusing to overwrite"):
        session.save_as(existing)


def test_save_as_refuses_destination_created_during_active_flush(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    document_session_factory: DocumentSessionFactory,
) -> None:
    active_card = Card(name="Active card")
    active = Stack(
        name="Active",
        cards=(active_card,),
        start_card_id=active_card.id,
    )
    controller = DocumentController(active)
    session = document_session_factory(controller)
    session.create(active, tmp_path / "Active.hotcards")
    assert session.store is not None
    active_store = session.store
    controller.execute(RenameCardCommand(card_id=active_card.id, name="Pending active edit"))
    destination = tmp_path / "Copy.hotcards"
    competing = Stack(name="External newer target")
    real_active_save = active_store.save

    def save_active_then_create_destination(snapshot: Stack) -> None:
        real_active_save(snapshot)
        StackStore(destination).create(competing)

    monkeypatch.setattr(active_store, "save", save_active_then_create_destination)

    with pytest.raises(DocumentSessionError, match="refusing to overwrite"):
        session.save_as(destination)

    assert session.store is active_store
    assert session.state.bundle_path == active_store.bundle_path
    assert controller.document.cards[0].name == "Pending active edit"
    assert not session.state.dirty
    assert StackStore(destination).load() == competing

    monkeypatch.setattr(active_store, "save", real_active_save)
    controller.execute(RenameCardCommand(card_id=active_card.id, name="Still active after race"))
    assert session.flush()
    assert active_store.load().cards[0].name == "Still active after race"
    assert StackStore(destination).load() == competing


def test_save_as_clears_history_that_can_reference_uncloned_assets(
    tmp_path: Path,
    document_session_factory: DocumentSessionFactory,
) -> None:
    controller = DocumentController(Stack(name="Welcome"))
    session = document_session_factory(controller)
    session.create(Stack(name="Current"), tmp_path / "Current.hotcards")
    controller.execute(CreateCardCommand(name="Undo-only state"))
    assert controller.undo()
    assert controller.can_redo

    session.save_as(tmp_path / "Copy.hotcards")

    assert not controller.can_undo
    assert not controller.can_redo


def test_binding_completes_and_retains_failed_asset_cleanup_for_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    document_session_factory: DocumentSessionFactory,
) -> None:
    store, stack, image_path = _owned_bundle(tmp_path, name="Original", operation="generate")
    controller = DocumentController(Stack(name="Welcome"))
    session = document_session_factory(controller)
    session.open(store.bundle_path)
    card = stack.cards[0]
    controller.execute(
        ReplaceRevisionBackgroundCommand(
            card_id=card.id, revision_id=card.active_revision.id, background=None
        )
    )
    assert session.flush()
    assert image_path.exists()  # Still reachable through Undo.
    candidate = Stack(name="Next")
    candidate_store = StackStore(tmp_path / "Next.hotcards")
    candidate_store.create(candidate)
    real_remove = StackStore.remove_owned_image_asset_if_unreferenced

    def fail_cleanup(*_args: object, **_kwargs: object) -> bool:
        raise StackStoreError("cleanup temporarily unavailable")

    monkeypatch.setattr(StackStore, "remove_owned_image_asset_if_unreferenced", fail_cleanup)
    assert session.open(candidate_store.bundle_path) == candidate
    assert controller.document == candidate
    assert not controller.can_undo
    assert session.state.bundle_path == candidate_store.bundle_path
    assert "cleanup temporarily unavailable" in (session.state.error or "")
    assert image_path.exists()
    monkeypatch.setattr(StackStore, "remove_owned_image_asset_if_unreferenced", real_remove)
    assert session.flush()
    assert not image_path.exists()
    assert session.state.error is None


def test_create_can_recover_empty_bundle_left_by_failed_attempt(
    tmp_path: Path, document_session_factory: DocumentSessionFactory
) -> None:
    controller = DocumentController(Stack(name="Welcome"))
    session = document_session_factory(controller)
    bundle = tmp_path / "Retry.hotcards"
    bundle.mkdir()

    session.create(Stack(name="Retry"), bundle)

    assert StackStore(bundle).load().name == "Retry"


def test_create_refuses_nonempty_existing_bundle(
    tmp_path: Path, document_session_factory: DocumentSessionFactory
) -> None:
    controller = DocumentController(Stack(name="Welcome"))
    session = document_session_factory(controller)
    bundle = tmp_path / "Existing.hotcards"
    StackStore(bundle).save(Stack(name="Existing"))

    with pytest.raises(DocumentSessionError, match="bundle already exists"):
        session.create(Stack(name="Replacement"), bundle)


def test_create_does_not_replace_stack_created_after_destination_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    document_session_factory: DocumentSessionFactory,
) -> None:
    controller = DocumentController(Stack(name="Welcome"))
    session = document_session_factory(controller)
    bundle = tmp_path / "Race.hotcards"
    bundle.mkdir()

    def competing_create() -> bool:
        StackStore(bundle).create(Stack(name="Competing"))
        return True

    monkeypatch.setattr(session, "flush", competing_create)

    with pytest.raises(DocumentSessionError, match="refusing to overwrite"):
        session.create(Stack(name="Replacement"), bundle)

    assert StackStore(bundle).load().name == "Competing"
