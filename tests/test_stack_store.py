"""Tests for safe, atomic HyperGen stack bundles."""

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from PIL import Image

from hypergen.domain.models import (
    Card,
    HotspotSet,
    ImageOrigin,
    ImageRevision,
    Interaction,
    NavigateAction,
    Point,
    Polygon,
    ResolvedCardReference,
    Stack,
)
from hypergen.storage.stack_store import StackStore, StackStoreError


def _write_png(path: Path) -> None:
    Image.new("RGB", (32, 24), "navy").save(path, format="PNG")


def _stack_with_asset(store: StackStore, source: Path) -> Stack:
    target_card = Card(name="Garden")
    source_card_id = uuid4()
    revision_id = uuid4()
    image_path = store.import_image(
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
    revision = ImageRevision(
        id=revision_id,
        image_path=image_path,
        origin=ImageOrigin.IMPORTED,
        source_filename=source.name,
        hotspot_set=HotspotSet(interactions=(interaction,)),
        created_at=datetime.now(UTC),
    )
    source_card = Card(
        id=source_card_id,
        name="Courtyard",
        image_revisions=(revision,),
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
    store = StackStore(tmp_path / "Castle.hypergen")
    stack = _stack_with_asset(store, source)

    store.save(stack)

    assert store.load() == stack
    image_path = stack.cards[0].image_revisions[0].image_path
    assert image_path.startswith("assets/cards/")
    assert not Path(image_path).is_absolute()
    assert json.loads(store.stack_path.read_text())["schema_version"] == 1


def test_failed_replace_preserves_active_stack_and_removes_temporary_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = StackStore(tmp_path / "Castle.hypergen")
    store.save(Stack(name="Original"))
    original = store.stack_path.read_bytes()

    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError("simulated interruption")

    monkeypatch.setattr("hypergen.storage.stack_store.os.replace", fail_replace)

    with pytest.raises(StackStoreError, match="simulated interruption"):
        store.save(Stack(name="Changed"))

    assert store.stack_path.read_bytes() == original
    assert list(store.bundle_path.glob(".stack-*.tmp")) == []


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
    store = StackStore(tmp_path / "Castle.hypergen")
    store.bundle_path.mkdir()
    payload = Stack(
        name="Unsafe",
        cards=(
            Card(
                name="Card",
                image_revisions=(
                    ImageRevision(
                        image_path=unsafe_path,
                        origin=ImageOrigin.IMPORTED,
                        created_at=datetime.now(UTC),
                    ),
                ),
            ),
        ),
    ).model_dump(mode="json")
    store.stack_path.write_text(json.dumps(payload))

    with pytest.raises(StackStoreError, match="asset path"):
        store.load()


def test_save_requires_asset_before_json_reference(tmp_path: Path) -> None:
    store = StackStore(tmp_path / "Castle.hypergen")
    card_id = uuid4()
    revision_id = uuid4()
    stack = Stack(
        name="Missing",
        cards=(
            Card(
                id=card_id,
                name="Card",
                image_revisions=(
                    ImageRevision(
                        id=revision_id,
                        image_path=f"assets/cards/{card_id}/image-{revision_id}.png",
                        origin=ImageOrigin.IMPORTED,
                        created_at=datetime.now(UTC),
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
    store = StackStore(tmp_path / "Castle.hypergen")
    wrong_card_id = uuid4()
    revision_id = uuid4()
    image_path = store.import_image(
        source,
        card_id=wrong_card_id,
        revision_id=revision_id,
    )
    stack = Stack(
        name="Mismatch",
        cards=(
            Card(
                name="Card",
                image_revisions=(
                    ImageRevision(
                        id=revision_id,
                        image_path=image_path,
                        origin=ImageOrigin.IMPORTED,
                        created_at=datetime.now(UTC),
                    ),
                ),
            ),
        ),
    )

    with pytest.raises(StackStoreError, match="does not match"):
        store.save(stack)


def test_import_image_refuses_overwrite_and_invalid_content(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    _write_png(source)
    store = StackStore(tmp_path / "Castle.hypergen")
    card_id = uuid4()
    revision_id = uuid4()

    store.import_image(source, card_id=card_id, revision_id=revision_id)

    with pytest.raises(StackStoreError, match="overwrite"):
        store.import_image(source, card_id=card_id, revision_id=revision_id)

    invalid = tmp_path / "invalid.png"
    invalid.write_bytes(b"not an image")
    with pytest.raises(StackStoreError, match="decode"):
        store.import_image(invalid, card_id=card_id, revision_id=uuid4())


def test_load_rejects_missing_and_future_versions(tmp_path: Path) -> None:
    store = StackStore(tmp_path / "Castle.hypergen")
    store.bundle_path.mkdir()

    for payload, message in [
        ({"name": "Missing"}, "missing schema_version"),
        ({"schema_version": 2, "name": "Future"}, "newer than supported"),
    ]:
        store.stack_path.write_text(json.dumps(payload))
        with pytest.raises(StackStoreError, match=message):
            store.load()


def test_symlinked_asset_directory_cannot_escape_bundle(tmp_path: Path) -> None:
    store = StackStore(tmp_path / "Castle.hypergen")
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
        store.import_image(source, card_id=card_id, revision_id=revision_id)


def test_clone_to_creates_independent_bundle_with_referenced_assets(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    _write_png(source)
    original = StackStore(tmp_path / "Original.hypergen")
    stack = _stack_with_asset(original, source)
    original.save(stack)

    destination = tmp_path / "Copy.hypergen"
    copied_store = original.clone_to(destination, stack)

    assert copied_store.load() == stack
    image_path = stack.cards[0].image_revisions[0].image_path
    assert copied_store.asset_path(image_path).is_file()
    copied_store.asset_path(image_path).unlink()
    assert original.asset_path(image_path).is_file()


def test_clone_to_refuses_existing_destination(tmp_path: Path) -> None:
    source_store = StackStore(tmp_path / "Source.hypergen")
    source_store.save(Stack(name="Source"))
    destination = tmp_path / "Existing.hypergen"
    destination.mkdir()

    with pytest.raises(StackStoreError, match="refusing to overwrite"):
        source_store.clone_to(destination, Stack(name="Source"))


def test_clone_to_wraps_destination_preparation_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_store = StackStore(tmp_path / "Source.hypergen")
    source_store.save(Stack(name="Source"))

    def fail_mkdtemp(**_kwargs: object) -> str:
        raise PermissionError("destination is read-only")

    monkeypatch.setattr("hypergen.storage.stack_store.tempfile.mkdtemp", fail_mkdtemp)

    with pytest.raises(StackStoreError, match="destination is read-only"):
        source_store.clone_to(
            tmp_path / "Copy.hypergen",
            Stack(name="Source"),
        )
