"""Tests for deliberate stack-schema migration dispatch."""

import pytest

from hypergen.storage.migrations import MigrationError, migrate_document


def test_current_document_passes_without_mutating_input() -> None:
    document = {"schema_version": 1, "name": "Castle"}

    migrated = migrate_document(document)

    assert migrated == document
    assert migrated is not document


def test_migration_dispatch_applies_one_versioned_step() -> None:
    document = {"schema_version": 0, "title": "Castle"}

    def migrate_v0(payload: dict[str, object]) -> dict[str, object]:
        payload["schema_version"] = 1
        payload["name"] = payload.pop("title")
        return payload

    assert migrate_document(document, migrations={0: migrate_v0}) == {
        "schema_version": 1,
        "name": "Castle",
    }
    assert document == {"schema_version": 0, "title": "Castle"}


@pytest.mark.parametrize(
    ("document", "message"),
    [
        ({}, "missing schema_version"),
        ({"schema_version": True}, "must be an integer"),
        ({"schema_version": 2}, "newer than supported"),
        ({"schema_version": 0}, "no migration"),
    ],
)
def test_missing_and_unsupported_versions_fail_clearly(
    document: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(MigrationError, match=message):
        migrate_document(document)


def test_migration_must_advance_exactly_one_version() -> None:
    with pytest.raises(MigrationError, match="must produce version 1"):
        migrate_document(
            {"schema_version": 0},
            migrations={0: lambda payload: {**payload, "schema_version": 2}},
        )
