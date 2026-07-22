"""Atomic persistence for human-readable HyperGen directory bundles."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from uuid import UUID

from PIL import Image, UnidentifiedImageError
from pydantic import ValidationError

from hypergen.domain.models import Stack
from hypergen.storage.migrations import MigrationError, migrate_document

STACK_FILENAME = "stack.json"
ASSET_ROOT = PurePosixPath("assets/cards")


class StackStoreError(ValueError):
    """A stack bundle cannot be read or written safely."""


def _relative_asset_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts:
        raise StackStoreError(f"asset path must be relative to the bundle: {value!r}")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise StackStoreError(f"asset path contains unsafe traversal components: {value!r}")
    if not path.is_relative_to(ASSET_ROOT):
        raise StackStoreError(f"asset path must be below {ASSET_ROOT}: {value!r}")
    return path


def _image_asset_path(card_id: UUID, revision_id: UUID) -> PurePosixPath:
    return ASSET_ROOT / str(card_id) / f"image-{revision_id}.png"


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _mkdir_durable(path: Path) -> None:
    if path.is_dir():
        return
    existing_ancestor = path
    while not existing_ancestor.exists():
        existing_ancestor = existing_ancestor.parent
    path.mkdir(parents=True, exist_ok=True)
    current = path.resolve()
    anchor = existing_ancestor.resolve()
    while True:
        _fsync_directory(current)
        if current == anchor:
            break
        current = current.parent


class StackStore:
    """Read and write one `.hypergen` directory bundle."""

    def __init__(self, bundle_path: Path) -> None:
        self.bundle_path = bundle_path

    @property
    def stack_path(self) -> Path:
        return self.bundle_path / STACK_FILENAME

    def _resolved_asset(self, relative_path: PurePosixPath) -> Path:
        bundle_root = self.bundle_path.resolve()
        candidate = self.bundle_path.joinpath(*relative_path.parts).resolve()
        try:
            candidate.relative_to(bundle_root)
        except ValueError as error:
            raise StackStoreError(
                f"asset path resolves outside the stack bundle: {relative_path}"
            ) from error
        return candidate

    def asset_path(self, relative_path: str) -> Path:
        """Resolve one validated bundle-relative asset path."""
        return self._resolved_asset(_relative_asset_path(relative_path))

    def _validate_assets(self, stack: Stack) -> None:
        for card in stack.cards:
            for revision in card.image_revisions:
                relative_path = _relative_asset_path(revision.image_path)
                expected_path = _image_asset_path(card.id, revision.id)
                if relative_path != expected_path:
                    raise StackStoreError(
                        f"image asset path {relative_path} does not match its "
                        "card and revision IDs; "
                        f"expected {expected_path}"
                    )
                asset_path = self._resolved_asset(relative_path)
                if not asset_path.is_file():
                    raise StackStoreError(
                        f"stack references a missing image asset: {relative_path}"
                    )

    def load(self) -> Stack:
        """Load, migrate, validate, and verify a bundle document."""
        try:
            payload = json.loads(self.stack_path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise StackStoreError(f"stack document does not exist: {self.stack_path}") from error
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise StackStoreError(
                f"could not read stack document {self.stack_path}: {error}"
            ) from error
        if not isinstance(payload, dict):
            raise StackStoreError("stack document root must be a JSON object")
        try:
            migrated = migrate_document(payload)
            stack = Stack.model_validate_json(json.dumps(migrated))
        except (MigrationError, ValidationError) as error:
            raise StackStoreError(f"invalid stack document: {error}") from error
        self._validate_assets(stack)
        return stack

    def save(self, stack: Stack) -> None:
        """Atomically replace `stack.json` after validating all referenced assets."""
        try:
            validated = Stack.model_validate(stack.model_dump(mode="python", round_trip=True))
        except ValidationError as error:
            raise StackStoreError(f"stack is invalid and cannot be saved: {error}") from error
        self._validate_assets(validated)
        _mkdir_durable(self.bundle_path)
        payload = validated.model_dump_json(indent=2) + "\n"
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.bundle_path,
                prefix=".stack-",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                temporary.write(payload)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_path, self.stack_path)
            _fsync_directory(self.bundle_path)
        except OSError as error:
            raise StackStoreError(
                f"could not atomically save {self.stack_path}: {error}"
            ) from error
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def import_image(
        self,
        source_path: Path,
        *,
        card_id: UUID,
        revision_id: UUID,
    ) -> str:
        """Copy a validated PNG into its deterministic bundle-owned asset path."""
        relative_path = _image_asset_path(card_id, revision_id)
        destination = self._resolved_asset(relative_path)
        if destination.exists():
            raise StackStoreError(f"refusing to overwrite existing image asset: {relative_path}")

        _mkdir_durable(destination.parent)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=destination.parent,
                prefix=f".{destination.stem}-",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                with source_path.open("rb") as source:
                    shutil.copyfileobj(source, temporary)
                temporary.flush()
                os.fsync(temporary.fileno())
            try:
                with Image.open(temporary_path) as image:
                    if image.format != "PNG":
                        raise StackStoreError(f"image asset must be a PNG: {source_path}")
                    image.verify()
                with Image.open(temporary_path) as image:
                    image.load()
            except (OSError, UnidentifiedImageError) as error:
                raise StackStoreError(
                    f"could not decode image asset {source_path}: {error}"
                ) from error
            os.link(temporary_path, destination)
            bundle_root = self.bundle_path.resolve()
            current = destination.parent
            while True:
                _fsync_directory(current)
                if current == bundle_root:
                    break
                current = current.parent
            _fsync_directory(bundle_root.parent)
        except FileExistsError as error:
            raise StackStoreError(
                f"refusing to overwrite existing image asset: {relative_path}"
            ) from error
        except OSError as error:
            raise StackStoreError(f"could not import image asset {source_path}: {error}") from error
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        return relative_path.as_posix()

    def autosave_hook(self) -> Callable[[Stack], None]:
        """Return the synchronous save boundary for a later debouncer/controller."""
        return self.save

    def clone_to(self, destination: Path, stack: Stack) -> StackStore:
        """Create an independent bundle containing exactly the referenced assets."""
        if destination.exists():
            raise StackStoreError(f"refusing to overwrite existing bundle: {destination}")
        temporary_bundle: Path | None = None
        try:
            _mkdir_durable(destination.parent)
            temporary_bundle = Path(
                tempfile.mkdtemp(
                    prefix=f".{destination.name}-",
                    suffix=".tmp",
                    dir=destination.parent,
                )
            )
            temporary_store = StackStore(temporary_bundle)
            for card in stack.cards:
                for revision in card.image_revisions:
                    temporary_store.import_image(
                        self.asset_path(revision.image_path),
                        card_id=card.id,
                        revision_id=revision.id,
                    )
            temporary_store.save(stack)
            os.rename(temporary_bundle, destination)
            _fsync_directory(destination.parent)
        except (OSError, StackStoreError) as error:
            if temporary_bundle is not None:
                shutil.rmtree(temporary_bundle, ignore_errors=True)
            if isinstance(error, StackStoreError):
                raise
            raise StackStoreError(f"could not create bundle {destination}: {error}") from error
        return StackStore(destination)
