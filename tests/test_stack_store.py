"""Tests for safe, atomic HotCards stack bundles."""

import errno
import json
import os
import subprocess
import sys
import wave
from dataclasses import asdict
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
    GeneratedSoundAsset,
    GenerateInputs,
    HotspotSet,
    ImageOperationSettings,
    Interaction,
    NavigateAction,
    Point,
    Polygon,
    ResolvedCardReference,
    SoundDefinition,
    SoundGenerationProvenance,
    Stack,
)
from hotcards.storage.stack_store import (
    ImageAssetExpectation,
    StackStore,
    StackStoreError,
    StackStoreTransactionError,
)


def _write_png(path: Path) -> None:
    Image.new("RGB", (512, 384), "navy").save(path, format="PNG")


def _write_wav(path: Path, *, duration_seconds: int = 2, channels: int = 2) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(44_100)
        output.writeframes(b"\0" * duration_seconds * 44_100 * channels * 2)


def _generated_sound(
    sound_id: UUID,
    asset_id: UUID,
    *,
    duration_seconds: int = 2,
) -> SoundDefinition:
    return SoundDefinition(
        id=sound_id,
        name="Knock",
        prompt="A wooden knock",
        duration_seconds=duration_seconds,
        generated=GeneratedSoundAsset(
            id=asset_id,
            audio_path=StackStore.sound_asset_path(sound_id, asset_id),
            provenance=SoundGenerationProvenance(
                prompt="A wooden knock",
                duration_seconds=duration_seconds,
                seed=42,
                generation_duration_milliseconds=700,
            ),
            created_at=datetime.now(UTC),
        ),
    )


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
                width=512,
                height=384,
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
        payload["cards"][0]["revisions"][0]["background"]["provenance"]["operation"] == "generate"
    )


def test_generated_sound_asset_and_manifest_are_committed_together(tmp_path: Path) -> None:
    source = tmp_path / "sound.wav"
    _write_wav(source)
    store = StackStore(tmp_path / "Sounds.hotcards")
    previous = Stack(name="Sounds")
    store.save(previous)
    sound_id = uuid4()
    asset_id = uuid4()
    changed = previous.model_copy(update={"sounds": (_generated_sound(sound_id, asset_id),)})

    owned = store.store_sound_asset_and_save(
        source,
        sound_id=sound_id,
        asset_id=asset_id,
        duration_seconds=2,
        previous_stack=previous,
        changed_stack=changed,
    )

    assert store.load() == changed
    assert owned.relative_path == StackStore.sound_asset_path(sound_id, asset_id)
    assert store.asset_path(owned.relative_path).is_file()
    with wave.open(str(store.asset_path(owned.relative_path)), "rb") as generated:
        assert generated.getnchannels() == 2
        assert generated.getframerate() == 44_100
        assert generated.getnframes() == 2 * 44_100
    assert (
        store.stored_sound_asset(
            owned.relative_path,
            sound_id=sound_id,
            asset_id=asset_id,
        )
        == owned
    )
    clone = store.clone_to(tmp_path / "Sounds Copy.hotcards", changed)
    assert clone.load() == changed
    with wave.open(str(clone.asset_path(owned.relative_path)), "rb") as generated:
        assert generated.getnframes() == 2 * 44_100


def test_sound_storage_rejects_wrong_audio_contract(tmp_path: Path) -> None:
    source = tmp_path / "mono.wav"
    _write_wav(source, channels=1)
    store = StackStore(tmp_path / "Sounds.hotcards")
    store.save(Stack(name="Sounds"))

    with pytest.raises(StackStoreError, match="16-bit PCM stereo"):
        store.store_sound_asset(
            source,
            sound_id=uuid4(),
            asset_id=uuid4(),
            duration_seconds=2,
        )


def test_sound_storage_rejects_truncated_payload_before_and_after_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "sound.wav"
    _write_wav(source)
    store = StackStore(tmp_path / "Sounds.hotcards")
    store.create(Stack(name="Sounds"))
    complete = source.read_bytes()
    source.write_bytes(complete[:-4])
    with pytest.raises(StackStoreError, match="truncated PCM"):
        store.store_sound_asset(source, sound_id=uuid4(), asset_id=uuid4(), duration_seconds=2)
    source.write_bytes(complete)

    def truncate_copy(source_file, destination_file) -> None:
        destination_file.write(source_file.read()[:-4])

    monkeypatch.setattr(stack_store_module.shutil, "copyfileobj", truncate_copy)
    with pytest.raises(StackStoreError, match="truncated PCM"):
        store.store_sound_asset(source, sound_id=uuid4(), asset_id=uuid4(), duration_seconds=2)
    assert not list(store.bundle_path.rglob("*.wav"))


@pytest.mark.parametrize("failures", [1, 3])
def test_sound_manifest_fsync_failure_is_never_reported_as_committed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failures: int
) -> None:
    source = tmp_path / "sound.wav"
    _write_wav(source)
    store = StackStore(tmp_path / "Sounds.hotcards")
    before = Stack(name="Sounds")
    store.create(before)
    sound_id, asset_id = uuid4(), uuid4()
    after = before.model_copy(update={"sounds": (_generated_sound(sound_id, asset_id),)})
    real_fsync = os.fsync
    bundle_stat = store.bundle_path.stat()
    replaced = False
    failed = 0

    def checkpoint(name: str) -> None:
        nonlocal replaced
        if name == "manifest-replaced":
            replaced = True

    def fail_fsync(descriptor: int) -> None:
        nonlocal failed
        current = os.fstat(descriptor)
        if replaced and current.st_ino == bundle_stat.st_ino and failed < failures:
            failed += 1
            raise OSError("manifest fsync interrupted")
        real_fsync(descriptor)

    monkeypatch.setattr(stack_store_module, "_io_checkpoint", checkpoint)
    monkeypatch.setattr(stack_store_module.os, "fsync", fail_fsync)
    with pytest.raises(StackStoreTransactionError) as caught:
        store.store_sound_asset_and_save(
            source,
            sound_id=sound_id,
            asset_id=asset_id,
            duration_seconds=2,
            previous_stack=before,
            changed_stack=after,
        )
    assert failed == failures
    assert caught.value.persisted_stack != after
    assert store.load() == before
    assert caught.value.durability_indeterminate == (failures == 3)
    assert bool(list(store.bundle_path.rglob("*.wav"))) == (failures == 3)


def test_owned_sound_is_removed_only_after_all_references_leave(tmp_path: Path) -> None:
    source = tmp_path / "sound.wav"
    _write_wav(source)
    store = StackStore(tmp_path / "Sounds.hotcards")
    previous = Stack(name="Sounds")
    store.save(previous)
    sound_id = uuid4()
    asset_id = uuid4()
    changed = previous.model_copy(update={"sounds": (_generated_sound(sound_id, asset_id),)})
    owned = store.store_sound_asset_and_save(
        source,
        sound_id=sound_id,
        asset_id=asset_id,
        duration_seconds=2,
        previous_stack=previous,
        changed_stack=changed,
    )

    with pytest.raises(StackStoreError, match="referenced sound"):
        store.remove_owned_sound_asset_if_unreferenced(
            owned,
            sound_id=sound_id,
            asset_id=asset_id,
            stack=changed,
        )
    store.save(previous)
    assert store.remove_owned_sound_asset_if_unreferenced(
        owned,
        sound_id=sound_id,
        asset_id=asset_id,
        stack=previous,
    )
    assert not store.asset_path(owned.relative_path).exists()


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


def test_secure_image_snapshot_pins_bytes_and_revalidates_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.png"
    Image.new("RGB", (41, 29), "navy").save(source, format="PNG")
    store = StackStore(tmp_path / "Snapshot.hotcards")
    card_id = uuid4()
    asset_id = uuid4()
    relative_path = store.store_image_asset(
        source,
        card_id=card_id,
        asset_id=asset_id,
    )
    store.create(Stack(name="Snapshot"))
    snapshots = tmp_path / "private"
    snapshots.mkdir(mode=0o700)

    snapshot = store.snapshot_image_asset(
        relative_path,
        card_id=card_id,
        asset_id=asset_id,
        destination_directory=snapshots,
    )

    assert snapshot.snapshot_path.parent == snapshots
    assert snapshot.snapshot_path != store.asset_path(relative_path)
    assert snapshot.snapshot_path.read_bytes() == store.asset_path(relative_path).read_bytes()
    assert (snapshot.width, snapshot.height) == (41, 29)
    store.require_image_asset_unchanged(
        snapshot,
        card_id=card_id,
        asset_id=asset_id,
    )
    assert snapshot.dispose()
    assert not snapshot.snapshot_path.exists()


def test_secure_image_snapshot_rejects_symlink_source(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside.png"
    _write_png(outside)
    store = StackStore(tmp_path / "Snapshot.hotcards")
    card_id = uuid4()
    asset_id = uuid4()
    relative_path = store.image_asset_path(card_id, asset_id)
    source = store.bundle_path.joinpath(*relative_path.split("/"))
    source.parent.mkdir(parents=True)
    source.symlink_to(outside)
    store.create(Stack(name="Snapshot"))
    snapshots = tmp_path / "private"
    snapshots.mkdir()

    with pytest.raises(
        StackStoreError,
        match="securely (snapshot|open asset directory)",
    ):
        store.snapshot_image_asset(
            relative_path,
            card_id=card_id,
            asset_id=asset_id,
            destination_directory=snapshots,
        )

    assert list(snapshots.iterdir()) == []
    assert outside.is_file()


def test_secure_image_snapshot_rejects_swapped_card_directory(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.png"
    _write_png(source)
    store = StackStore(tmp_path / "Snapshot.hotcards")
    card_id = uuid4()
    asset_id = uuid4()
    relative_path = store.store_image_asset(
        source,
        card_id=card_id,
        asset_id=asset_id,
    )
    store.create(Stack(name="Snapshot"))
    card_directory = store.asset_path(relative_path).parent
    owned_directory = card_directory.with_name(f"{card_directory.name}-owned")
    card_directory.rename(owned_directory)
    outside_directory = tmp_path / "outside-card"
    outside_directory.mkdir()
    _write_png(outside_directory / Path(relative_path).name)
    card_directory.symlink_to(outside_directory, target_is_directory=True)
    snapshots = tmp_path / "private"
    snapshots.mkdir()

    with pytest.raises(
        StackStoreError,
        match="securely (snapshot|open asset directory)",
    ):
        store.snapshot_image_asset(
            relative_path,
            card_id=card_id,
            asset_id=asset_id,
            destination_directory=snapshots,
        )

    assert (outside_directory / Path(relative_path).name).is_file()
    assert list(snapshots.iterdir()) == []


def test_secure_image_snapshot_detects_source_namespace_replacement(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.png"
    replacement = tmp_path / "replacement.png"
    Image.new("RGB", (32, 24), "navy").save(source, format="PNG")
    Image.new("RGB", (32, 24), "gold").save(replacement, format="PNG")
    store = StackStore(tmp_path / "Snapshot.hotcards")
    card_id = uuid4()
    asset_id = uuid4()
    relative_path = store.store_image_asset(
        source,
        card_id=card_id,
        asset_id=asset_id,
    )
    store.create(Stack(name="Snapshot"))
    snapshots = tmp_path / "private"
    snapshots.mkdir()
    snapshot = store.snapshot_image_asset(
        relative_path,
        card_id=card_id,
        asset_id=asset_id,
        destination_directory=snapshots,
    )
    logical_source = store.asset_path(relative_path)
    logical_source.unlink()
    replacement.replace(logical_source)

    with pytest.raises(StackStoreError, match="changed while Edit"):
        store.require_image_asset_unchanged(
            snapshot,
            card_id=card_id,
            asset_id=asset_id,
        )

    assert snapshot.snapshot_path.is_file()
    assert snapshot.dispose()


def test_source_revalidation_reopens_namespace_after_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.png"
    replacement = tmp_path / "replacement.png"
    Image.new("RGB", (32, 24), "navy").save(source, format="PNG")
    Image.new("RGB", (32, 24), "navy").save(replacement, format="PNG")
    store = StackStore(tmp_path / "Snapshot.hotcards")
    card_id = uuid4()
    asset_id = uuid4()
    relative_path = store.store_image_asset(
        source,
        card_id=card_id,
        asset_id=asset_id,
    )
    store.create(Stack(name="Snapshot"))
    snapshots = tmp_path / "private"
    snapshots.mkdir()
    snapshot = store.snapshot_image_asset(
        relative_path,
        card_id=card_id,
        asset_id=asset_id,
        destination_directory=snapshots,
    )
    logical_source = store.asset_path(relative_path)
    real_sha256_fd = stack_store_module._sha256_fd
    replaced = False

    def replace_after_hash(fd: int) -> str:
        nonlocal replaced
        digest = real_sha256_fd(fd)
        if not replaced:
            replaced = True
            os.replace(replacement, logical_source)
        return digest

    monkeypatch.setattr(stack_store_module, "_sha256_fd", replace_after_hash)

    with pytest.raises(StackStoreError, match="changed while Edit"):
        store.require_image_asset_unchanged(
            snapshot,
            card_id=card_id,
            asset_id=asset_id,
        )

    assert replaced
    assert snapshot.dispose()


def test_secure_image_snapshot_rejects_source_swapped_during_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.png"
    replacement = tmp_path / "replacement.png"
    Image.new("RGB", (32, 24), "navy").save(source, format="PNG")
    Image.new("RGB", (32, 24), "gold").save(replacement, format="PNG")
    store = StackStore(tmp_path / "Snapshot.hotcards")
    card_id = uuid4()
    asset_id = uuid4()
    relative_path = store.store_image_asset(
        source,
        card_id=card_id,
        asset_id=asset_id,
    )
    store.create(Stack(name="Snapshot"))
    logical_source = store.asset_path(relative_path)
    snapshots = tmp_path / "private"
    snapshots.mkdir()
    real_write_all = stack_store_module._write_all
    swapped = False

    def swap_source_after_copy_write(fd: int, payload: bytes) -> None:
        nonlocal swapped
        real_write_all(fd, payload)
        if not swapped:
            swapped = True
            logical_source.unlink()
            replacement.replace(logical_source)

    monkeypatch.setattr(stack_store_module, "_write_all", swap_source_after_copy_write)

    with pytest.raises(StackStoreError, match="changed while being copied"):
        store.snapshot_image_asset(
            relative_path,
            card_id=card_id,
            asset_id=asset_id,
            destination_directory=snapshots,
        )

    assert swapped
    assert list(snapshots.iterdir()) == []


def test_snapshot_disposal_preserves_replacement_inode(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    _write_png(source)
    store = StackStore(tmp_path / "Snapshot.hotcards")
    card_id = uuid4()
    asset_id = uuid4()
    relative_path = store.store_image_asset(
        source,
        card_id=card_id,
        asset_id=asset_id,
    )
    store.create(Stack(name="Snapshot"))
    snapshots = tmp_path / "private"
    snapshots.mkdir()
    snapshot = store.snapshot_image_asset(
        relative_path,
        card_id=card_id,
        asset_id=asset_id,
        destination_directory=snapshots,
    )
    owned_backup = snapshots / "owned-backup.png"
    snapshot.snapshot_path.rename(owned_backup)
    snapshot.snapshot_path.write_bytes(b"foreign")

    assert not snapshot.dispose()
    assert snapshot.snapshot_path.read_bytes() == b"foreign"
    assert owned_backup.is_file()


def test_snapshot_disposal_does_not_quarantine_foreign_symlink(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.png"
    _write_png(source)
    store = StackStore(tmp_path / "Snapshot.hotcards")
    card_id = uuid4()
    asset_id = uuid4()
    relative_path = store.store_image_asset(
        source,
        card_id=card_id,
        asset_id=asset_id,
    )
    store.create(Stack(name="Snapshot"))
    snapshots = tmp_path / "private"
    snapshots.mkdir()
    snapshot = store.snapshot_image_asset(
        relative_path,
        card_id=card_id,
        asset_id=asset_id,
        destination_directory=snapshots,
    )
    owned_backup = snapshots / "owned-backup.png"
    snapshot.snapshot_path.rename(owned_backup)
    foreign = tmp_path / "foreign"
    foreign.write_bytes(b"foreign")
    snapshot.snapshot_path.symlink_to(foreign)

    assert not snapshot.dispose()
    assert snapshot.snapshot_path.is_symlink()
    assert snapshot.snapshot_path.resolve() == foreign
    assert list(snapshots.glob(".image-cleanup-*")) == []
    assert owned_backup.is_file()


def test_source_revalidation_rejects_fifo_without_blocking(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    _write_png(source)
    store = StackStore(tmp_path / "Snapshot.hotcards")
    card_id = uuid4()
    asset_id = uuid4()
    relative_path = store.store_image_asset(
        source,
        card_id=card_id,
        asset_id=asset_id,
    )
    store.create(Stack(name="Snapshot"))
    snapshots = tmp_path / "private"
    snapshots.mkdir()
    snapshot = store.snapshot_image_asset(
        relative_path,
        card_id=card_id,
        asset_id=asset_id,
        destination_directory=snapshots,
    )
    logical_source = store.asset_path(relative_path)
    logical_source.unlink()
    os.mkfifo(logical_source)

    # A lost O_NONBLOCK flag must fail this test instead of hanging pytest.
    script = """
import sys
from pathlib import Path
from uuid import UUID
from hotcards.storage.stack_store import StackStore, StackStoreError, StoredImageSnapshot
snapshot = StoredImageSnapshot(**__import__('json').loads(sys.argv[2]))
try:
    StackStore(Path(sys.argv[1])).require_image_asset_unchanged(
        snapshot, card_id=UUID(sys.argv[3]), asset_id=UUID(sys.argv[4]))
except StackStoreError as error:
    assert 'no longer a regular file' in str(error), str(error)
else:
    raise AssertionError('FIFO was accepted')
"""
    subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(store.bundle_path),
            json.dumps(asdict(snapshot), default=str),
            str(card_id),
            str(asset_id),
        ],
        check=True,
        timeout=5,
    )

    assert snapshot.dispose()


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
        ({"schema_version": str(CURRENT_SCHEMA_VERSION)}, "schema_version"),
        ({"schema_version": True}, "schema_version"),
        ({"schema_version": CURRENT_SCHEMA_VERSION - 1}, "schema_version"),
        ({"schema_version": CURRENT_SCHEMA_VERSION + 1}, "schema_version"),
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
    original_bytes = original.asset_path(image_path).read_bytes()
    copied_store.asset_path(image_path).write_bytes(b"independent replacement")
    assert original.asset_path(image_path).read_bytes() == original_bytes


def test_external_image_and_manifest_commit_as_one_transaction(
    tmp_path: Path,
) -> None:
    source = tmp_path / "generated.png"
    _write_png(source)
    store = StackStore(tmp_path / "Generated.hotcards")
    card = Card(name="Card")
    previous = Stack(name="Stack", cards=(card,))
    store.save(previous)
    asset_id = uuid4()
    image_path = store.image_asset_path(card.id, asset_id)
    revision = card.active_revision.model_copy(
        update={"background": _generated_background(asset_id, image_path)}
    )
    changed = previous.model_copy(
        update={
            "cards": (
                card.model_copy(
                    update={
                        "revisions": (revision,),
                        "active_revision_id": revision.id,
                    }
                ),
            )
        }
    )

    stored = store.store_image_asset_and_save(
        source,
        destination_card_id=card.id,
        destination_asset_id=asset_id,
        previous_stack=previous,
        changed_stack=changed,
    )

    assert store.load() == changed
    assert stored.relative_path == image_path
    assert (
        store.stored_image_asset(
            image_path,
            card_id=card.id,
            asset_id=asset_id,
        )
        == stored
    )


def test_external_image_transaction_rolls_back_manifest_and_asset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "generated.png"
    _write_png(source)
    store = StackStore(tmp_path / "Generated.hotcards")
    card = Card(name="Card")
    previous = Stack(name="Stack", cards=(card,))
    store.save(previous)
    asset_id = uuid4()
    image_path = store.image_asset_path(card.id, asset_id)
    revision = card.active_revision.model_copy(
        update={"background": _generated_background(asset_id, image_path)}
    )
    changed = previous.model_copy(
        update={
            "cards": (
                card.model_copy(
                    update={
                        "revisions": (revision,),
                        "active_revision_id": revision.id,
                    }
                ),
            )
        }
    )
    injected = False

    def fail_after_replace(name: str) -> None:
        nonlocal injected
        if name == "manifest-replaced" and not injected:
            injected = True
            raise OSError("injected image durability failure")

    monkeypatch.setattr(stack_store_module, "_io_checkpoint", fail_after_replace)

    with pytest.raises(
        StackStoreTransactionError,
        match="image durability failure",
    ):
        store.store_image_asset_and_save(
            source,
            destination_card_id=card.id,
            destination_asset_id=asset_id,
            previous_stack=previous,
            changed_stack=changed,
        )

    assert store.load() == previous
    assert not store.asset_path(image_path).exists()


@pytest.mark.parametrize("checkpoint", ["manifest-file-fsynced", "manifest-replaced"])
def test_generate_reference_replacement_rolls_back_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, checkpoint: str
) -> None:
    source = tmp_path / "source.png"
    _write_png(source)
    store = StackStore(tmp_path / "References.hotcards")
    before = _stack_with_asset(store, source)
    store.create(before)
    reference, target = before.cards
    background = reference.active_revision.background
    assert background is not None
    private = tmp_path / "private"
    private.mkdir()
    snapshot = store.snapshot_image_asset(
        background.image_path,
        card_id=reference.id,
        asset_id=background.id,
        destination_directory=private,
    )
    asset_id = uuid4()
    image_path = store.image_asset_path(target.id, asset_id)
    target_revision = target.active_revision.model_copy(
        update={"background": _generated_background(asset_id, image_path)}
    )
    after = before.model_copy(
        update={
            "cards": (
                reference,
                target.model_copy(update={"revisions": (target_revision,)}),
            )
        }
    )
    replaced = False
    reference_path = store.asset_path(background.image_path)

    def replace_reference(name: str) -> None:
        nonlocal replaced
        if name == checkpoint and not replaced:
            replaced = True
            reference_path.rename(reference_path.with_suffix(".original"))
            _write_png(reference_path)

    monkeypatch.setattr(stack_store_module, "_io_checkpoint", replace_reference)
    try:
        with pytest.raises(StackStoreTransactionError, match="Generate Reference"):
            store.store_image_asset_and_save(
                source,
                destination_card_id=target.id,
                destination_asset_id=asset_id,
                previous_stack=before,
                changed_stack=after,
                expected_references=(ImageAssetExpectation(snapshot, reference.id, background.id),),
            )
        assert replaced
        assert store.load() == before
        assert not store.asset_path(image_path).exists()
        assert reference_path.exists()
    finally:
        assert snapshot.dispose()


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
