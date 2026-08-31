"""Atomic persistence for human-readable HotCards directory bundles."""

from __future__ import annotations

import errno
import json
import logging
import os
import shutil
import stat
import sys
import tempfile
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from functools import wraps
from pathlib import Path, PurePosixPath
from threading import Lock, RLock
from typing import Concatenate
from uuid import UUID, uuid4

from PIL import Image, UnidentifiedImageError
from pydantic import ValidationError

from hotcards.domain.models import CURRENT_SCHEMA_VERSION, Stack

STACK_FILENAME = "stack.json"
ASSET_ROOT = PurePosixPath("assets/cards")
logger = logging.getLogger(__name__)
_BUNDLE_LOCKS_GUARD = Lock()
_BUNDLE_LOCKS: dict[str, RLock] = {}


class StackStoreError(ValueError):
    """A stack bundle cannot be read or written safely."""


class StackStoreTransactionError(StackStoreError):
    """A bundle transaction failed and may also have rollback errors."""

    def __init__(
        self,
        operation_error: Exception,
        rollback_errors: tuple[Exception, ...] = (),
        *,
        persisted_stack: Stack | None = None,
        observed_stack: Stack | None = None,
        durability_indeterminate: bool = False,
        owned_asset: StoredImageAsset | None = None,
    ) -> None:
        message = f"could not commit duplicate asset and stack document: {operation_error}"
        if rollback_errors:
            details = "; ".join(str(error) for error in rollback_errors)
            message += f"; rollback also failed: {details}"
        if durability_indeterminate:
            message += "; manifest durability remains indeterminate"
        super().__init__(message)
        self.operation_error = operation_error
        self.rollback_errors = rollback_errors
        self.persisted_stack = persisted_stack
        self.observed_stack = observed_stack
        self.durability_indeterminate = durability_indeterminate
        self.owned_asset = owned_asset


@dataclass(frozen=True, slots=True)
class StoredImageAsset:
    """Filesystem identity of one securely created bundle image."""

    relative_path: str
    device: int
    inode: int
    directory_device: int
    directory_inode: int


def _io_checkpoint(_name: str) -> None:
    """Fault-injection seam around durable transaction boundaries."""


def _bundle_mutation_lock(bundle_path: Path) -> RLock:
    key = os.path.abspath(os.fspath(bundle_path))
    with _BUNDLE_LOCKS_GUARD:
        return _BUNDLE_LOCKS.setdefault(key, RLock())


def _serialized_bundle_mutation[**P, R](
    operation: Callable[Concatenate[StackStore, P], R],
) -> Callable[Concatenate[StackStore, P], R]:
    @wraps(operation)
    def serialized(
        store: StackStore,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R:
        with _bundle_mutation_lock(store.bundle_path):
            return operation(store, *args, **kwargs)

    return serialized


def _secure_open_flags(*, directory: bool = False) -> int:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory_flag = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or (directory and directory_flag is None):
        raise StackStoreError(
            "secure duplicate asset operations require O_NOFOLLOW and O_DIRECTORY"
        )
    flags = os.O_RDONLY | nofollow
    if directory:
        flags |= directory_flag
    return flags


def _open_directory_at(parent_fd: int, name: str, *, create: bool = False) -> int:
    if "/" in name or name in {"", ".", ".."}:
        raise StackStoreError(f"invalid asset directory component: {name!r}")
    if create:
        try:
            os.mkdir(name, mode=0o700, dir_fd=parent_fd)
        except FileExistsError:
            pass
    try:
        directory_fd = os.open(
            name,
            _secure_open_flags(directory=True),
            dir_fd=parent_fd,
        )
    except OSError as error:
        raise StackStoreError(
            f"could not securely open asset directory {name!r}: {error}"
        ) from error
    if not stat.S_ISDIR(os.fstat(directory_fd).st_mode):
        os.close(directory_fd)
        raise StackStoreError(f"asset path component is not a directory: {name!r}")
    return directory_fd


def _read_all(fd: int) -> bytes:
    chunks: list[bytes] = []
    while chunk := os.read(fd, 1024 * 1024):
        chunks.append(chunk)
    return b"".join(chunks)


def _write_all(fd: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(fd, payload[offset:])
        if written <= 0:
            raise OSError("write returned no progress")
        offset += written


def _copy_all(source_fd: int, destination_fd: int) -> None:
    while chunk := os.read(source_fd, 1024 * 1024):
        _write_all(destination_fd, chunk)


def _validate_png_fd(fd: int, source_label: str) -> None:
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        with os.fdopen(os.dup(fd), "rb") as image_file:
            with Image.open(image_file) as image:
                if image.format != "PNG":
                    raise StackStoreError(
                        f"image asset must be a PNG: {source_label}"
                    )
                image.verify()
        os.lseek(fd, 0, os.SEEK_SET)
        with os.fdopen(os.dup(fd), "rb") as image_file:
            with Image.open(image_file) as image:
                image.load()
    except (OSError, UnidentifiedImageError) as error:
        raise StackStoreError(
            f"could not decode image asset {source_label}: {error}"
        ) from error


def _file_at_matches(
    parent_fd: int,
    name: str,
    *,
    device: int,
    inode: int,
) -> bool:
    try:
        candidate_fd = os.open(
            name,
            _secure_open_flags(),
            dir_fd=parent_fd,
        )
    except FileNotFoundError:
        return False
    except OSError:
        return False
    try:
        candidate_stat = os.fstat(candidate_fd)
        return (
            stat.S_ISREG(candidate_stat.st_mode)
            and candidate_stat.st_dev == device
            and candidate_stat.st_ino == inode
        )
    finally:
        os.close(candidate_fd)


def _open_matching_directory_at(
    parent_fd: int,
    name: str,
    *,
    device: int,
    inode: int,
) -> int | None:
    try:
        candidate_fd = os.open(
            name,
            _secure_open_flags(directory=True),
            dir_fd=parent_fd,
        )
    except FileNotFoundError:
        return None
    except OSError as error:
        if error.errno in {errno.ELOOP, errno.ENOTDIR}:
            return None
        raise StackStoreError(
            f"could not securely inspect asset directory {name!r}: {error}"
        ) from error
    candidate_stat = os.fstat(candidate_fd)
    if (
        candidate_stat.st_dev != device
        or candidate_stat.st_ino != inode
        or not stat.S_ISDIR(candidate_stat.st_mode)
    ):
        os.close(candidate_fd)
        return None
    return candidate_fd


def _quarantine_name(name: str, kind: str) -> str:
    return f".{name}.{kind}-{uuid4().hex}.tmp"


def _quarantine_owned_file_at(
    parent_fd: int,
    name: str,
    *,
    device: int,
    inode: int,
) -> bool:
    try:
        candidate_fd = os.open(
            name,
            _secure_open_flags(),
            dir_fd=parent_fd,
        )
    except FileNotFoundError:
        return False
    except OSError as error:
        raise StackStoreError(
            f"could not securely open owned image {name!r}: {error}"
        ) from error
    try:
        candidate_stat = os.fstat(candidate_fd)
        if (
            not stat.S_ISREG(candidate_stat.st_mode)
            or candidate_stat.st_dev != device
            or candidate_stat.st_ino != inode
        ):
            raise StackStoreError(
                f"refusing to remove image asset whose identity changed: {name}"
            )
        _io_checkpoint("owned-file-cleanup-prechecked")
        quarantine_name = _quarantine_name(name, "owned-file")
        try:
            os.rename(
                name,
                quarantine_name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
        except OSError as error:
            raise StackStoreError(
                f"could not quarantine owned image {name!r}: {error}"
            ) from error
        try:
            quarantine_fd = os.open(
                quarantine_name,
                _secure_open_flags(),
                dir_fd=parent_fd,
            )
        except OSError as error:
            raise StackStoreError(
                f"owned image quarantine {quarantine_name!r} could not be "
                f"verified and was preserved: {error}"
            ) from error
        try:
            quarantine_stat = os.fstat(quarantine_fd)
            if (
                not stat.S_ISREG(quarantine_stat.st_mode)
                or quarantine_stat.st_dev != device
                or quarantine_stat.st_ino != inode
            ):
                raise StackStoreError(
                    f"owned image quarantine {quarantine_name!r} has a foreign "
                    "identity and was preserved"
                )
        finally:
            os.close(quarantine_fd)
        try:
            os.unlink(quarantine_name, dir_fd=parent_fd)
            os.fsync(parent_fd)
        except OSError as error:
            raise StackStoreError(
                f"could not remove owned image quarantine {quarantine_name!r}: "
                f"{error}"
            ) from error
        return True
    finally:
        os.close(candidate_fd)


def _quarantine_owned_directory_if_empty(
    parent_fd: int,
    name: str,
    *,
    device: int,
    inode: int,
) -> bool:
    candidate_fd = _open_matching_directory_at(
        parent_fd,
        name,
        device=device,
        inode=inode,
    )
    if candidate_fd is None:
        try:
            os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return False
        except OSError as error:
            raise StackStoreError(
                f"could not inspect owned asset directory {name!r}: {error}"
            ) from error
        raise StackStoreError(
            f"refusing to remove asset directory whose identity changed: {name}"
        )
    try:
        if os.listdir(candidate_fd):
            return False
        _io_checkpoint("owned-directory-cleanup-prechecked")
        quarantine_name = _quarantine_name(name, "owned-directory")
        try:
            os.rename(
                name,
                quarantine_name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
        except OSError as error:
            raise StackStoreError(
                f"could not quarantine owned asset directory {name!r}: {error}"
            ) from error
        try:
            quarantine_fd = os.open(
                quarantine_name,
                _secure_open_flags(directory=True),
                dir_fd=parent_fd,
            )
        except OSError as error:
            raise StackStoreError(
                f"owned directory quarantine {quarantine_name!r} could not be "
                f"verified and was preserved: {error}"
            ) from error
        try:
            quarantine_stat = os.fstat(quarantine_fd)
            if (
                not stat.S_ISDIR(quarantine_stat.st_mode)
                or quarantine_stat.st_dev != device
                or quarantine_stat.st_ino != inode
                or os.listdir(quarantine_fd)
            ):
                raise StackStoreError(
                    f"owned directory quarantine {quarantine_name!r} changed "
                    "identity or contents and was preserved"
                )
        finally:
            os.close(quarantine_fd)
        try:
            os.rmdir(quarantine_name, dir_fd=parent_fd)
            os.fsync(parent_fd)
        except OSError as error:
            raise StackStoreError(
                f"could not remove owned directory quarantine "
                f"{quarantine_name!r}: {error}"
            ) from error
        return True
    finally:
        os.close(candidate_fd)


def _asset_path_matches(
    bundle_fd: int,
    *,
    card_id: UUID,
    filename: str,
    device: int,
    inode: int,
) -> bool:
    try:
        with ExitStack() as stack:
            assets_fd = _open_directory_at(bundle_fd, "assets")
            stack.callback(os.close, assets_fd)
            cards_fd = _open_directory_at(assets_fd, "cards")
            stack.callback(os.close, cards_fd)
            card_fd = _open_directory_at(cards_fd, str(card_id))
            stack.callback(os.close, card_fd)
            return _file_at_matches(
                card_fd,
                filename,
                device=device,
                inode=inode,
            )
    except StackStoreError:
        return False


def _temporary_manifest_name(prefix: str) -> str:
    return f"{prefix}{uuid4().hex}.tmp"


def _write_manifest_at(
    bundle_fd: int,
    payload: bytes,
    *,
    prefix: str,
    checkpoints: bool,
) -> str:
    name = _temporary_manifest_name(prefix)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | _secure_open_flags()
    manifest_fd = os.open(name, flags, 0o600, dir_fd=bundle_fd)
    try:
        _write_all(manifest_fd, payload)
        if checkpoints:
            _io_checkpoint("manifest-written")
        os.fsync(manifest_fd)
        if checkpoints:
            _io_checkpoint("manifest-file-fsynced")
    except Exception:
        os.close(manifest_fd)
        os.unlink(name, dir_fd=bundle_fd)
        raise
    os.close(manifest_fd)
    return name


def _replace_manifest_at(
    bundle_fd: int,
    temporary_name: str,
    *,
    checkpoints: bool,
    on_replaced: Callable[[], None] | None = None,
    on_directory_fsynced: Callable[[], None] | None = None,
) -> None:
    os.replace(
        temporary_name,
        STACK_FILENAME,
        src_dir_fd=bundle_fd,
        dst_dir_fd=bundle_fd,
    )
    if on_replaced is not None:
        on_replaced()
    if checkpoints:
        _io_checkpoint("manifest-replaced")
    os.fsync(bundle_fd)
    if on_directory_fsynced is not None:
        on_directory_fsynced()
    if checkpoints:
        _io_checkpoint("manifest-directory-fsynced")


def _recover_previous_manifest(
    bundle_fd: int,
    *,
    previous_payload: bytes,
    previous_stack: Stack,
    changed_stack: Stack,
    rollback_errors: list[Exception],
) -> tuple[bool, Stack | None]:
    previous_manifest_durable = False

    def attempt(prefix: str) -> bool:
        temporary_name: str | None = None
        directory_fsynced = False

        def mark_directory_fsynced() -> None:
            nonlocal directory_fsynced
            directory_fsynced = True

        try:
            temporary_name = _write_manifest_at(
                bundle_fd,
                previous_payload,
                prefix=prefix,
                checkpoints=False,
            )
            _replace_manifest_at(
                bundle_fd,
                temporary_name,
                checkpoints=False,
                on_directory_fsynced=mark_directory_fsynced,
            )
            temporary_name = None
        except Exception as error:
            rollback_errors.append(error)
            if temporary_name is not None:
                try:
                    os.unlink(temporary_name, dir_fd=bundle_fd)
                except FileNotFoundError:
                    pass
                except OSError as cleanup_error:
                    rollback_errors.append(cleanup_error)
        return directory_fsynced

    previous_manifest_durable = attempt(".stack-rollback-")
    if not previous_manifest_durable:
        previous_manifest_durable = attempt(".stack-stabilize-")

    observed_stack: Stack | None = None
    try:
        current_stack_fd = os.open(
            STACK_FILENAME,
            _secure_open_flags(),
            dir_fd=bundle_fd,
        )
        try:
            current_payload = _read_all(current_stack_fd)
        finally:
            os.close(current_stack_fd)
        current_stack = Stack.model_validate_json(current_payload)
        if current_stack == previous_stack:
            observed_stack = previous_stack
        elif current_stack == changed_stack:
            observed_stack = changed_stack
        else:
            rollback_errors.append(
                StackStoreError(
                    "observed stack document matches neither the previous nor "
                    "duplicate state"
                )
            )
    except (OSError, ValidationError) as error:
        rollback_errors.append(error)
    return previous_manifest_durable, observed_stack


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

    @staticmethod
    def image_asset_path(card_id: UUID, asset_id: UUID) -> str:
        """Return the deterministic bundle-relative path for one image asset."""
        return _image_asset_path(card_id, asset_id).as_posix()

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

    @_serialized_bundle_mutation
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

    @_serialized_bundle_mutation
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

    @_serialized_bundle_mutation
    def copy_image_asset_and_save(
        self,
        source_relative_path: str,
        *,
        source_card_id: UUID,
        source_asset_id: UUID,
        destination_card_id: UUID,
        destination_asset_id: UUID,
        previous_stack: Stack,
        changed_stack: Stack,
    ) -> StoredImageAsset:
        """Atomically own a secure image copy and its manifest transition."""
        source_path = _relative_asset_path(source_relative_path)
        expected_source_path = _image_asset_path(
            source_card_id,
            source_asset_id,
        )
        if source_path != expected_source_path:
            raise StackStoreError(
                f"image asset path {source_path} does not match its card and "
                f"asset IDs; expected {expected_source_path}"
            )
        destination_path = _image_asset_path(
            destination_card_id,
            destination_asset_id,
        )
        matching_backgrounds = [
            revision.background
            for card in changed_stack.cards
            if card.id == destination_card_id
            for revision in card.revisions
            if revision.background is not None
            and revision.background.id == destination_asset_id
        ]
        if (
            len(matching_backgrounds) != 1
            or matching_backgrounds[0].image_path != destination_path.as_posix()
        ):
            raise StackStoreError(
                "changed stack does not reference the reserved duplicate asset"
            )

        changed_payload = changed_stack.model_dump_json(indent=2).encode() + b"\n"
        rollback_errors: list[Exception] = []
        destination_owned = False
        destination_directory_created = False
        destination_directory_device: int | None = None
        destination_directory_inode: int | None = None
        destination_name = destination_path.name
        stored_asset: StoredImageAsset | None = None
        manifest_temporary_name: str | None = None

        with ExitStack() as stack:
            try:
                bundle_fd = os.open(
                    self.bundle_path,
                    _secure_open_flags(directory=True),
                )
            except OSError as error:
                raise StackStoreError(
                    f"could not securely open stack bundle {self.bundle_path}: {error}"
                ) from error
            stack.callback(os.close, bundle_fd)
            try:
                stack_fd = os.open(
                    STACK_FILENAME,
                    _secure_open_flags(),
                    dir_fd=bundle_fd,
                )
            except OSError as error:
                raise StackStoreError(
                    f"could not securely open {self.stack_path}: {error}"
                ) from error
            stack.callback(os.close, stack_fd)
            stack_stat = os.fstat(stack_fd)
            if not stat.S_ISREG(stack_stat.st_mode):
                raise StackStoreError("stack document is not a regular file")
            persisted_payload = _read_all(stack_fd)
            try:
                persisted_stack = Stack.model_validate_json(persisted_payload)
            except ValidationError as error:
                raise StackStoreError(
                    f"invalid current stack document: {error}"
                ) from error
            if persisted_stack != previous_stack:
                raise StackStoreError(
                    "current stack document changed before duplication"
                )
            previous_payload = persisted_payload

            assets_fd = _open_directory_at(bundle_fd, "assets")
            stack.callback(os.close, assets_fd)
            cards_fd = _open_directory_at(assets_fd, "cards")
            stack.callback(os.close, cards_fd)
            source_card_fd = _open_directory_at(cards_fd, str(source_card_id))
            stack.callback(os.close, source_card_fd)
            try:
                source_fd = os.open(
                    source_path.name,
                    _secure_open_flags(),
                    dir_fd=source_card_fd,
                )
            except OSError as error:
                raise StackStoreError(
                    f"could not securely open source image {source_path}: {error}"
                ) from error
            stack.callback(os.close, source_fd)
            if not stat.S_ISREG(os.fstat(source_fd).st_mode):
                raise StackStoreError(
                    f"source image is not a regular file: {source_path}"
                )
            _io_checkpoint("source-opened")

            try:
                os.mkdir(str(destination_card_id), mode=0o700, dir_fd=cards_fd)
                destination_directory_created = True
            except FileExistsError:
                pass
            destination_card_fd = _open_directory_at(
                cards_fd,
                str(destination_card_id),
            )
            stack.callback(os.close, destination_card_fd)
            destination_directory_stat = os.fstat(destination_card_fd)
            destination_directory_device = destination_directory_stat.st_dev
            destination_directory_inode = destination_directory_stat.st_ino
            _io_checkpoint("destination-directory-opened")

            try:
                destination_fd = os.open(
                    destination_name,
                    os.O_RDWR
                    | os.O_CREAT
                    | os.O_EXCL
                    | _secure_open_flags(),
                    0o600,
                    dir_fd=destination_card_fd,
                )
                destination_owned = True
            except OSError as error:
                raise StackStoreError(
                    f"could not create duplicate image {destination_path}: {error}"
                ) from error
            stack.callback(os.close, destination_fd)
            destination_stat = os.fstat(destination_fd)
            stored_asset = StoredImageAsset(
                relative_path=destination_path.as_posix(),
                device=destination_stat.st_dev,
                inode=destination_stat.st_ino,
                directory_device=destination_directory_stat.st_dev,
                directory_inode=destination_directory_stat.st_ino,
            )

            changed_manifest_replaced = False
            changed_manifest_durable = False

            def mark_changed_manifest_replaced() -> None:
                nonlocal changed_manifest_replaced
                changed_manifest_replaced = True

            def mark_changed_manifest_durable() -> None:
                nonlocal changed_manifest_durable
                changed_manifest_durable = True

            try:
                _io_checkpoint("destination-created")
                _copy_all(source_fd, destination_fd)
                os.fsync(destination_fd)
                _io_checkpoint("asset-file-fsynced")
                _validate_png_fd(destination_fd, source_path.as_posix())
                os.fsync(destination_card_fd)
                _io_checkpoint("asset-directory-fsynced")
                os.fsync(cards_fd)
                _io_checkpoint("cards-directory-fsynced")

                current_assets_fd = _open_directory_at(bundle_fd, "assets")
                stack.callback(os.close, current_assets_fd)
                current_cards_fd = _open_directory_at(current_assets_fd, "cards")
                stack.callback(os.close, current_cards_fd)
                current_source_card_fd = _open_directory_at(
                    current_cards_fd,
                    str(source_card_id),
                )
                stack.callback(os.close, current_source_card_fd)
                current_source_fd = os.open(
                    source_path.name,
                    _secure_open_flags(),
                    dir_fd=current_source_card_fd,
                )
                stack.callback(os.close, current_source_fd)
                source_stat = os.fstat(source_fd)
                current_source_stat = os.fstat(current_source_fd)
                if (
                    source_stat.st_dev,
                    source_stat.st_ino,
                ) != (
                    current_source_stat.st_dev,
                    current_source_stat.st_ino,
                ):
                    raise StackStoreError(
                        "source image namespace changed during the transaction"
                    )
                current_destination_card_fd = _open_directory_at(
                    current_cards_fd,
                    str(destination_card_id),
                )
                stack.callback(os.close, current_destination_card_fd)
                current_destination_fd = os.open(
                    destination_name,
                    _secure_open_flags(),
                    dir_fd=current_destination_card_fd,
                )
                stack.callback(os.close, current_destination_fd)
                current_destination_stat = os.fstat(current_destination_fd)
                if (
                    stored_asset.device,
                    stored_asset.inode,
                ) != (
                    current_destination_stat.st_dev,
                    current_destination_stat.st_ino,
                ):
                    raise StackStoreError(
                        "duplicate image namespace changed during the transaction"
                    )

                manifest_temporary_name = _write_manifest_at(
                    bundle_fd,
                    changed_payload,
                    prefix=".stack-duplicate-",
                    checkpoints=True,
                )
                _replace_manifest_at(
                    bundle_fd,
                    manifest_temporary_name,
                    checkpoints=True,
                    on_replaced=mark_changed_manifest_replaced,
                    on_directory_fsynced=mark_changed_manifest_durable,
                )
                manifest_temporary_name = None
                if not _asset_path_matches(
                    bundle_fd,
                    card_id=destination_card_id,
                    filename=destination_name,
                    device=stored_asset.device,
                    inode=stored_asset.inode,
                ) or not _asset_path_matches(
                    bundle_fd,
                    card_id=source_card_id,
                    filename=source_path.name,
                    device=source_stat.st_dev,
                    inode=source_stat.st_ino,
                ):
                    raise StackStoreError(
                        "image namespace changed before transaction completion"
                    )
                if not changed_manifest_durable:
                    raise StackStoreError(
                        "duplicate manifest durability was not established"
                    )
            except Exception as operation_error:
                if manifest_temporary_name is not None:
                    try:
                        os.unlink(manifest_temporary_name, dir_fd=bundle_fd)
                    except FileNotFoundError:
                        pass
                    except OSError as error:
                        rollback_errors.append(error)
                if changed_manifest_replaced:
                    previous_manifest_durable, observed_stack = (
                        _recover_previous_manifest(
                            bundle_fd,
                            previous_payload=previous_payload,
                            previous_stack=previous_stack,
                            changed_stack=changed_stack,
                            rollback_errors=rollback_errors,
                        )
                    )
                else:
                    previous_manifest_durable = True
                    observed_stack = previous_stack
                retained_asset = stored_asset if destination_owned else None
                if previous_manifest_durable:
                    if destination_owned and stored_asset is not None:
                        try:
                            _quarantine_owned_file_at(
                                destination_card_fd,
                                destination_name,
                                device=stored_asset.device,
                                inode=stored_asset.inode,
                            )
                            retained_asset = None
                        except StackStoreError as error:
                            rollback_errors.append(error)
                    if (
                        retained_asset is None
                        and destination_directory_created
                        and destination_directory_device is not None
                        and destination_directory_inode is not None
                    ):
                        try:
                            _io_checkpoint("destination-directory-cleanup")
                            _quarantine_owned_directory_if_empty(
                                cards_fd,
                                str(destination_card_id),
                                device=destination_directory_device,
                                inode=destination_directory_inode,
                            )
                        except StackStoreError as error:
                            rollback_errors.append(error)
                raise StackStoreTransactionError(
                    operation_error,
                    tuple(rollback_errors),
                    persisted_stack=(
                        previous_stack if previous_manifest_durable else None
                    ),
                    observed_stack=observed_stack,
                    durability_indeterminate=not previous_manifest_durable,
                    owned_asset=retained_asset,
                ) from operation_error

        assert stored_asset is not None
        return stored_asset

    @_serialized_bundle_mutation
    def save_stack_transaction(
        self,
        previous_stack: Stack,
        changed_stack: Stack,
    ) -> None:
        """Replace one manifest with durable rollback on any reported failure."""
        changed_payload = changed_stack.model_dump_json(indent=2).encode() + b"\n"
        rollback_errors: list[Exception] = []
        manifest_temporary_name: str | None = None
        try:
            bundle_fd = os.open(
                self.bundle_path,
                _secure_open_flags(directory=True),
            )
        except OSError as error:
            raise StackStoreError(
                f"could not securely open stack bundle {self.bundle_path}: {error}"
            ) from error
        try:
            try:
                stack_fd = os.open(
                    STACK_FILENAME,
                    _secure_open_flags(),
                    dir_fd=bundle_fd,
                )
            except OSError as error:
                raise StackStoreError(
                    f"could not securely open {self.stack_path}: {error}"
                ) from error
            try:
                if not stat.S_ISREG(os.fstat(stack_fd).st_mode):
                    raise StackStoreError("stack document is not a regular file")
                previous_payload = _read_all(stack_fd)
            finally:
                os.close(stack_fd)
            try:
                persisted_stack = Stack.model_validate_json(previous_payload)
            except ValidationError as error:
                raise StackStoreError(
                    f"invalid current stack document: {error}"
                ) from error
            if persisted_stack != previous_stack:
                raise StackStoreError(
                    "current stack document changed before duplication"
                )
            changed_manifest_replaced = False
            changed_manifest_durable = False

            def mark_changed_manifest_replaced() -> None:
                nonlocal changed_manifest_replaced
                changed_manifest_replaced = True

            def mark_changed_manifest_durable() -> None:
                nonlocal changed_manifest_durable
                changed_manifest_durable = True

            try:
                manifest_temporary_name = _write_manifest_at(
                    bundle_fd,
                    changed_payload,
                    prefix=".stack-duplicate-",
                    checkpoints=True,
                )
                _replace_manifest_at(
                    bundle_fd,
                    manifest_temporary_name,
                    checkpoints=True,
                    on_replaced=mark_changed_manifest_replaced,
                    on_directory_fsynced=mark_changed_manifest_durable,
                )
                manifest_temporary_name = None
                if not changed_manifest_durable:
                    raise StackStoreError(
                        "duplicate manifest durability was not established"
                    )
            except Exception as operation_error:
                if manifest_temporary_name is not None:
                    try:
                        os.unlink(manifest_temporary_name, dir_fd=bundle_fd)
                    except FileNotFoundError:
                        pass
                    except OSError as error:
                        rollback_errors.append(error)
                if changed_manifest_replaced:
                    previous_manifest_durable, observed_stack = (
                        _recover_previous_manifest(
                            bundle_fd,
                            previous_payload=previous_payload,
                            previous_stack=previous_stack,
                            changed_stack=changed_stack,
                            rollback_errors=rollback_errors,
                        )
                    )
                else:
                    previous_manifest_durable = True
                    observed_stack = previous_stack
                raise StackStoreTransactionError(
                    operation_error,
                    tuple(rollback_errors),
                    persisted_stack=(
                        previous_stack if previous_manifest_durable else None
                    ),
                    observed_stack=observed_stack,
                    durability_indeterminate=not previous_manifest_durable,
                ) from operation_error
        finally:
            os.close(bundle_fd)

    @_serialized_bundle_mutation
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

    def stored_image_asset(
        self,
        relative_path: str,
        *,
        card_id: UUID,
        asset_id: UUID,
    ) -> StoredImageAsset:
        """Return the securely opened filesystem identity of one owned image."""
        parsed_path = _relative_asset_path(relative_path)
        expected_path = _image_asset_path(card_id, asset_id)
        if parsed_path != expected_path:
            raise StackStoreError(
                f"image asset path {parsed_path} does not match its card and asset IDs"
            )
        with ExitStack() as stack:
            try:
                bundle_fd = os.open(
                    self.bundle_path,
                    _secure_open_flags(directory=True),
                )
            except OSError as error:
                raise StackStoreError(
                    f"could not securely open stack bundle {self.bundle_path}: {error}"
                ) from error
            stack.callback(os.close, bundle_fd)
            assets_fd = _open_directory_at(bundle_fd, "assets")
            stack.callback(os.close, assets_fd)
            cards_fd = _open_directory_at(assets_fd, "cards")
            stack.callback(os.close, cards_fd)
            card_fd = _open_directory_at(cards_fd, str(card_id))
            stack.callback(os.close, card_fd)
            try:
                image_fd = os.open(
                    parsed_path.name,
                    _secure_open_flags(),
                    dir_fd=card_fd,
                )
            except OSError as error:
                raise StackStoreError(
                    f"could not securely open owned image {parsed_path}: {error}"
                ) from error
            stack.callback(os.close, image_fd)
            image_stat = os.fstat(image_fd)
            if not stat.S_ISREG(image_stat.st_mode):
                raise StackStoreError(
                    f"owned image is not a regular file: {parsed_path}"
                )
            card_stat = os.fstat(card_fd)
            return StoredImageAsset(
                relative_path=relative_path,
                device=image_stat.st_dev,
                inode=image_stat.st_ino,
                directory_device=card_stat.st_dev,
                directory_inode=card_stat.st_ino,
            )

    @_serialized_bundle_mutation
    def remove_owned_image_asset_if_unreferenced(
        self,
        asset: StoredImageAsset,
        *,
        card_id: UUID,
        asset_id: UUID,
        stack: Stack,
    ) -> bool:
        """Remove only the exact owned inode after all document references leave."""
        parsed_path = _relative_asset_path(asset.relative_path)
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
            and revision.background.image_path == asset.relative_path
            for document in documents
            for card in document.cards
            for revision in card.revisions
        ):
            raise StackStoreError(
                f"refusing to remove referenced image asset: {asset.relative_path}"
            )
        with ExitStack() as descriptors:
            try:
                bundle_fd = os.open(
                    self.bundle_path,
                    _secure_open_flags(directory=True),
                )
            except OSError as error:
                raise StackStoreError(
                    f"could not securely open stack bundle {self.bundle_path}: {error}"
                ) from error
            descriptors.callback(os.close, bundle_fd)
            assets_fd = _open_directory_at(bundle_fd, "assets")
            descriptors.callback(os.close, assets_fd)
            cards_fd = _open_directory_at(assets_fd, "cards")
            descriptors.callback(os.close, cards_fd)
            card_fd = _open_directory_at(cards_fd, str(card_id))
            descriptors.callback(os.close, card_fd)
            removed = _quarantine_owned_file_at(
                card_fd,
                parsed_path.name,
                device=asset.device,
                inode=asset.inode,
            )
            directory_removed = _quarantine_owned_directory_if_empty(
                cards_fd,
                str(card_id),
                device=asset.directory_device,
                inode=asset.directory_inode,
            )
        return removed or directory_removed

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
