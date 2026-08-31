"""Atomic persistence for human-readable HotCards directory bundles."""

from __future__ import annotations

import errno
import json
import logging
import os
import shutil
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from uuid import UUID

from PIL import Image, UnidentifiedImageError
from pydantic import ValidationError

from hotcards.domain.models import CURRENT_SCHEMA_VERSION, Stack

STACK_FILENAME = "stack.json"
ASSET_ROOT = PurePosixPath("assets/cards")
logger = logging.getLogger(__name__)


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


def _image_asset_path(card_id: UUID, asset_id: UUID) -> PurePosixPath:
    return ASSET_ROOT / str(card_id) / f"image-{asset_id}.png"


def _fsync_directory(path: Path) -> None:
    try:
        directory_fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except PermissionError as error:
        if sys.platform == "darwin" and error.errno in {errno.EACCES, errno.EPERM}:
            logger.warning(
                "macOS denied directory fsync for %s; file data remains flushed",
                path,
            )
            return
        raise


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
    """Read and write one `.hotcards` directory bundle."""

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
            for revision in card.revisions:
                if revision.background is None:
                    continue
                relative_path = _relative_asset_path(revision.background.image_path)
                expected_path = _image_asset_path(card.id, revision.background.id)
                if relative_path != expected_path:
                    raise StackStoreError(
                        f"image asset path {relative_path} does not match its "
                        "card and asset IDs; "
                        f"expected {expected_path}"
                    )
                asset_path = self._resolved_asset(relative_path)
                if not asset_path.is_file():
                    raise StackStoreError(
                        f"stack references a missing image asset: {relative_path}"
                    )

    def load(self) -> Stack:
        """Load and validate one current-schema bundle document."""
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
        version = payload.get("schema_version")
        supported_versions = {CURRENT_SCHEMA_VERSION}
        if type(version) is not int or version not in supported_versions:
            raise StackStoreError(
                "invalid stack document: schema_version must be one of "
                + ", ".join(str(item) for item in sorted(supported_versions))
            )
        try:
            stack = Stack.model_validate_json(json.dumps(payload))
        except ValidationError as error:
            raise StackStoreError(f"invalid stack document: {error}") from error
        self._validate_assets(stack)
        return stack

    def save(self, stack: Stack) -> None:
        """Atomically replace `stack.json` after validating all referenced assets."""
        self._write_stack(stack, replace=True)

    def create(self, stack: Stack) -> None:
        """Atomically create `stack.json` without replacing an existing document."""
        self._write_stack(stack, replace=False)

    def _write_stack(self, stack: Stack, *, replace: bool) -> None:
        try:
            validated = Stack.model_validate(stack.model_dump(mode="python", round_trip=True))
        except ValidationError as error:
            raise StackStoreError(f"stack is invalid and cannot be saved: {error}") from error
        self._validate_assets(validated)
        payload = validated.model_dump_json(indent=2) + "\n"
        temporary_path: Path | None = None
        try:
            _mkdir_durable(self.bundle_path)
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
            if replace:
                os.replace(temporary_path, self.stack_path)
            else:
                os.link(temporary_path, self.stack_path)
            _fsync_directory(self.bundle_path)
        except FileExistsError as error:
            raise StackStoreError(
                f"refusing to overwrite existing stack document: {self.stack_path}"
            ) from error
        except OSError as error:
            raise StackStoreError(
                f"could not atomically save {self.stack_path}: {error}"
            ) from error
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def store_image_asset(
        self,
        source_path: Path,
        *,
        card_id: UUID,
        asset_id: UUID | None = None,
        revision_id: UUID | None = None,
    ) -> str:
        """Copy a validated PNG into its deterministic bundle-owned asset path."""
        resolved_asset_id = asset_id if asset_id is not None else revision_id
        if resolved_asset_id is None:
            raise StackStoreError("an image asset ID is required")
        relative_path = _image_asset_path(card_id, resolved_asset_id)
        destination = self._resolved_asset(relative_path)
        if destination.exists():
            raise StackStoreError(f"refusing to overwrite existing image asset: {relative_path}")

        temporary_path: Path | None = None
        try:
            _mkdir_durable(destination.parent)
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
            raise StackStoreError(f"could not store image asset {source_path}: {error}") from error
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        return relative_path.as_posix()

    def remove_image_asset_if_unreferenced(
        self,
        relative_path: str,
        *,
        card_id: UUID,
        asset_id: UUID,
        stack: Stack,
    ) -> bool:
        """Remove one exact bundle-owned image only when the document cannot reach it."""
        parsed_path = _relative_asset_path(relative_path)
        expected_path = _image_asset_path(card_id, asset_id)
        if parsed_path != expected_path:
            raise StackStoreError(
                f"image asset path {parsed_path} does not match its card and asset IDs"
            )
        documents = [stack]
        if self.stack_path.is_file():
            documents.append(self.load())
        if any(
            revision.background is not None
            and revision.background.image_path == relative_path
            for document in documents
            for card in document.cards
            for revision in card.revisions
        ):
            raise StackStoreError(
                f"refusing to remove referenced image asset: {relative_path}"
            )
        destination = self._resolved_asset(parsed_path)
        try:
            if not destination.exists():
                return False
            destination.unlink()
            _fsync_directory(destination.parent)
        except OSError as error:
            raise StackStoreError(
                f"could not remove image asset {relative_path}: {error}"
            ) from error
        return True

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
            copied_paths: set[str] = set()
            for card in stack.cards:
                for revision in card.revisions:
                    background = revision.background
                    if background is None or background.image_path in copied_paths:
                        continue
                    temporary_store.store_image_asset(
                        self.asset_path(background.image_path),
                        card_id=card.id,
                        asset_id=background.id,
                    )
                    copied_paths.add(background.image_path)
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
