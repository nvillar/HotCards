"""Explicit schema-version migration dispatch for stack documents."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Any

from hypergen.domain.models import CURRENT_SCHEMA_VERSION

JsonObject = dict[str, Any]
Migration = Callable[[JsonObject], JsonObject]


class MigrationError(ValueError):
    """A stack document cannot be migrated to the supported schema."""


def _schema_version(document: JsonObject) -> int:
    if "schema_version" not in document:
        raise MigrationError("stack document is missing schema_version")
    version = document["schema_version"]
    if type(version) is not int:
        raise MigrationError("stack document schema_version must be an integer")
    return version


def migrate_document(
    document: JsonObject,
    *,
    target_version: int = CURRENT_SCHEMA_VERSION,
    migrations: Mapping[int, Migration] | None = None,
) -> JsonObject:
    """Return a migrated copy, applying one registered step per schema version."""
    migrated = deepcopy(document)
    version = _schema_version(migrated)
    if version > target_version:
        raise MigrationError(
            f"stack schema version {version} is newer than supported version {target_version}"
        )
    registry = migrations or {}
    while version < target_version:
        migration = registry.get(version)
        if migration is None:
            raise MigrationError(
                f"stack schema version {version} is unsupported; "
                f"no migration to version {version + 1} is registered"
            )
        migrated = migration(deepcopy(migrated))
        next_version = _schema_version(migrated)
        if next_version != version + 1:
            raise MigrationError(
                f"migration from version {version} must produce version {version + 1}, "
                f"not {next_version}"
            )
        version = next_version
    return migrated
