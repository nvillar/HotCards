"""Tests for safe, atomic HotCards stack bundles."""

import errno
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from PIL import Image

import hotcards.storage.stack_store as stack_store_module
from hotcards.domain.models import (
    CURRENT_SCHEMA_VERSION,
    Card,
    CardRevision,
    DirectGenerateProvenance,
    GeneratedBackground,
    GenerateInputs,
    HotspotSet,
    ImageOperationSettings,
    Interaction,
    NavigateAction,
    Point,
    Polygon,
    ResolvedCardReference,
    Stack,
)
from hotcards.storage.stack_store import StackStore, StackStoreError


def _write_png(path: Path) -> None:
    Image.new("RGB", (32, 24), "navy").save(path, format="PNG")


def _generated_background(asset_id: UUID, image_path: str) -> GeneratedBackground:
    generated_at = datetime.now(UTC)
    return GeneratedBackground(
        id=asset_id,
        image_path=image_path,
        provenance=DirectGenerateProvenance(
            inputs=GenerateInputs(
                description="A courtyard",
            ),
            render_prompt="A courtyard",
            settings=ImageOperationSettings(
                model_identifier="test",
                mflux_version="test",
                seed=1,
                width=1024,
                height=768,
                step_count=4,
                generated_at=generated_at,
                duration_seconds=1,
            ),
        ),
        created_at=generated_at,
    )


def _stack_with_asset(store: StackStore, source: Path) -> Stack:
    target_card = Card(name="Garden")
    source_card_id = uuid4()
    revision_id = uuid4()
    image_path = store.store_image_asset(
        source,
        card_id=source_card_id,
        revision_id=revision_id,
    )
    interaction = Interaction(
        label="Garden gate",
        action=NavigateAction(target=ResolvedCardReference(target_card_id=target_card.id)),
        polygons=(
            Polygon(
                points=(
                    Point(x=0.1, y=0.1),
                    Point(x=0.4, y=0.1),
                    Point(x=0.2, y=0.5),
                )
            ),
        ),
    )
    revision = CardRevision(
        id=revision_id,
        background=_generated_background(revision_id, image_path),
        hotspot_set=HotspotSet(interactions=(interaction,)),
    )
    source_card = Card(
        id=source_card_id,
        name="Courtyard",
        revisions=(revision,),
        active_revision_id=revision.id,
    )
    return Stack(
        name="Castle",
        cards=(source_card, target_card),
        start_card_id=source_card.id,
    )


def test_bundle_round_trip_preserves_document_and_relative_asset(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    _write_png(source)
    store = StackStore(tmp_path / "Castle.hotcards")
    stack = _stack_with_asset(store, source)

    store.save(stack)

    assert store.load() == stack
    image_path = stack.cards[0].revisions[0].image_path
    assert image_path is not None
    assert image_path.startswith("assets/cards/")
    assert not Path(image_path).is_absolute()
    payload = json.loads(store.stack_path.read_text())
    assert payload["schema_version"] == CURRENT_SCHEMA_VERSION
    assert payload["aspect_ratio"] == "4:3"
    assert "canvas" not in payload
    assert (
        payload["cards"][0]["revisions"][0]["background"]["provenance"]["operation"]
        == "generate"
    )


def test_failed_replace_preserves_active_stack_and_removes_temporary_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = StackStore(tmp_path / "Castle.hotcards")
    store.save(Stack(name="Original"))
    original = store.stack_path.read_bytes()

    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError("simulated interruption")

    monkeypatch.setattr("hotcards.storage.stack_store.os.replace", fail_replace)

    with pytest.raises(StackStoreError, match="simulated interruption"):
        store.save(Stack(name="Changed"))

    assert store.stack_path.read_bytes() == original
    assert list(store.bundle_path.glob(".stack-*.tmp")) == []


def test_create_never_replaces_existing_stack_document(tmp_path: Path) -> None:
    store = StackStore(tmp_path / "Existing.hotcards")
    store.create(Stack(name="Existing"))

    with pytest.raises(StackStoreError, match="refusing to overwrite"):
        store.create(Stack(name="Replacement"))

    assert store.load().name == "Existing"


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "/tmp/escape.png",
        "../escape.png",
        "assets/cards/../../escape.png",
        "other/image.png",
    ],
)
def test_load_rejects_unsafe_asset_paths(tmp_path: Path, unsafe_path: str) -> None:
    store = StackStore(tmp_path / "Castle.hotcards")
    store.bundle_path.mkdir()
    revision_id = uuid4()
    payload = Stack(
        name="Unsafe",
        cards=(
            Card(
                name="Card",
                revisions=(
                    CardRevision(
                        id=revision_id,
                        background=_generated_background(revision_id, unsafe_path),
                    ),
                ),
            ),
        ),
    ).model_dump(mode="json")
    store.stack_path.write_text(json.dumps(payload))

    with pytest.raises(StackStoreError, match="asset path"):
        store.load()


def test_save_requires_asset_before_json_reference(tmp_path: Path) -> None:
    store = StackStore(tmp_path / "Castle.hotcards")
    card_id = uuid4()
    revision_id = uuid4()
    stack = Stack(
        name="Missing",
        cards=(
            Card(
                id=card_id,
                name="Card",
                revisions=(
                    CardRevision(
                        id=revision_id,
                        background=_generated_background(
                            revision_id,
                            f"assets/cards/{card_id}/image-{revision_id}.png",
                        ),
                    ),
                ),
            ),
        ),
    )

    with pytest.raises(StackStoreError, match="missing image asset"):
        store.save(stack)

    assert not store.stack_path.exists()


def test_save_requires_asset_path_to_match_card_and_revision_ids(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    _write_png(source)
    store = StackStore(tmp_path / "Castle.hotcards")
    wrong_card_id = uuid4()
    revision_id = uuid4()
    image_path = store.store_image_asset(
        source,
        card_id=wrong_card_id,
        revision_id=revision_id,
    )
    stack = Stack(
        name="Mismatch",
        cards=(
            Card(
                name="Card",
                revisions=(
                    CardRevision(
                        id=revision_id,
                        background=_generated_background(revision_id, image_path),
                    ),
                ),
            ),
        ),
    )

    with pytest.raises(StackStoreError, match="does not match"):
        store.save(stack)


def test_store_image_asset_refuses_overwrite_and_invalid_content(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.png"
    _write_png(source)
    store = StackStore(tmp_path / "Castle.hotcards")
    card_id = uuid4()
    revision_id = uuid4()

    store.store_image_asset(source, card_id=card_id, revision_id=revision_id)

    with pytest.raises(StackStoreError, match="overwrite"):
        store.store_image_asset(source, card_id=card_id, revision_id=revision_id)

    invalid = tmp_path / "invalid.png"
    invalid.write_bytes(b"not an image")
    with pytest.raises(StackStoreError, match="decode"):
        store.store_image_asset(invalid, card_id=card_id, revision_id=uuid4())


def test_load_rejects_missing_unsupported_and_future_versions(
    tmp_path: Path,
) -> None:
    store = StackStore(tmp_path / "Castle.hotcards")
    store.bundle_path.mkdir()

    for payload, message in [
        ({"name": "Missing"}, "schema_version"),
        ({"schema_version": 3, "name": "Legacy"}, "schema_version"),
        ({"schema_version": 10, "name": "Previous"}, "schema_version"),
        ({"schema_version": 12, "name": "Future"}, "schema_version"),
    ]:
        store.stack_path.write_text(json.dumps(payload))
        with pytest.raises(StackStoreError, match=message):
            store.load()


def test_symlinked_asset_directory_cannot_escape_bundle(tmp_path: Path) -> None:
    store = StackStore(tmp_path / "Castle.hotcards")
    outside = tmp_path / "outside"
    outside.mkdir()
    asset_parent = store.bundle_path / "assets"
    asset_parent.mkdir(parents=True)
    (asset_parent / "cards").symlink_to(outside, target_is_directory=True)
    card_id = uuid4()
    revision_id = uuid4()
    source = tmp_path / "source.png"
    _write_png(source)

    with pytest.raises(StackStoreError, match="outside"):
        store.store_image_asset(source, card_id=card_id, revision_id=revision_id)


def test_clone_to_creates_independent_bundle_with_referenced_assets(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    _write_png(source)
    original = StackStore(tmp_path / "Original.hotcards")
    stack = _stack_with_asset(original, source)
    original.save(stack)

    destination = tmp_path / "Copy.hotcards"
    copied_store = original.clone_to(destination, stack)

    assert copied_store.load() == stack
    image_path = stack.cards[0].revisions[0].image_path
    assert image_path is not None
    assert copied_store.asset_path(image_path).is_file()
    copied_store.asset_path(image_path).unlink()
    assert original.asset_path(image_path).is_file()


def test_clone_to_refuses_existing_destination(tmp_path: Path) -> None:
    source_store = StackStore(tmp_path / "Source.hotcards")
    source_store.save(Stack(name="Source"))
    destination = tmp_path / "Existing.hotcards"
    destination.mkdir()

    with pytest.raises(StackStoreError, match="refusing to overwrite"):
        source_store.clone_to(destination, Stack(name="Source"))


def test_clone_to_wraps_destination_preparation_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_store = StackStore(tmp_path / "Source.hotcards")
    source_store.save(Stack(name="Source"))

    def fail_mkdtemp(**_kwargs: object) -> str:
        raise PermissionError("destination is read-only")

    monkeypatch.setattr("hotcards.storage.stack_store.tempfile.mkdtemp", fail_mkdtemp)

    with pytest.raises(StackStoreError, match="destination is read-only"):
        source_store.clone_to(
            tmp_path / "Copy.hotcards",
            Stack(name="Source"),
        )


def test_macos_privacy_denial_during_directory_fsync_is_tolerated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def deny_directory_open(_path: Path, _flags: int) -> int:
        raise PermissionError(errno.EPERM, "Operation not permitted")

    monkeypatch.setattr(stack_store_module.sys, "platform", "darwin")
    monkeypatch.setattr(stack_store_module.os, "open", deny_directory_open)

    stack_store_module._fsync_directory(tmp_path)

    assert "file data remains flushed" in caplog.text


def test_save_wraps_bundle_directory_preparation_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_mkdir(_path: Path) -> None:
        raise PermissionError(errno.EACCES, "protected folder")

    monkeypatch.setattr(stack_store_module, "_mkdir_durable", fail_mkdir)

    with pytest.raises(StackStoreError, match="protected folder"):
        StackStore(tmp_path / "Protected.hotcards").save(Stack(name="Protected"))
