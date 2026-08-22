"""Tests for deliberate stack-schema migration dispatch."""

from uuid import uuid4

import pytest

from hypergen.storage.migrations import MigrationError, migrate_document


def test_current_document_passes_without_mutating_input() -> None:
    document = {"schema_version": 2, "name": "Castle"}

    migrated = migrate_document(document)

    assert migrated == document
    assert migrated is not document


def test_migration_dispatch_applies_one_versioned_step() -> None:
    document = {"schema_version": 0, "title": "Castle"}

    def migrate_v0(payload: dict[str, object]) -> dict[str, object]:
        payload["schema_version"] = 1
        payload["name"] = payload.pop("title")
        return payload

    assert migrate_document(
        document,
        target_version=1,
        migrations={0: migrate_v0},
    ) == {
        "schema_version": 1,
        "name": "Castle",
    }
    assert document == {"schema_version": 0, "title": "Castle"}


@pytest.mark.parametrize(
    ("document", "message"),
    [
        ({}, "missing schema_version"),
        ({"schema_version": True}, "must be an integer"),
        ({"schema_version": 3}, "newer than supported"),
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
            target_version=1,
            migrations={0: lambda payload: {**payload, "schema_version": 2}},
        )


@pytest.mark.parametrize(
    "document",
    [
        {"schema_version": 1, "cards": []},
        {"schema_version": 1, "id": "not-a-uuid", "cards": []},
        {
            "schema_version": 1,
            "id": str(uuid4()),
            "cards": [{"name": "Missing ID"}],
        },
    ],
)
def test_malformed_v1_data_fails_at_the_migration_boundary(
    document: dict[str, object],
) -> None:
    with pytest.raises(MigrationError, match="invalid migration data"):
        migrate_document(document)


def test_v1_document_migrates_card_content_into_revisions() -> None:
    stack_id = uuid4()
    card_id = uuid4()
    revision_id = uuid4()
    document = {
        "schema_version": 1,
        "id": str(stack_id),
        "name": "Castle",
        "global_style": "Ink wash",
        "canvas": {"width": 1024, "height": 768},
        "run_overlay_mode": "hidden",
        "start_card_id": str(card_id),
        "cards": [
            {
                "id": str(card_id),
                "name": "Courtyard",
                "scene_description": "A moonlit courtyard",
                "interaction_description": "Open the gate",
                "card_style": None,
                "image_revisions": [
                    {
                        "id": str(revision_id),
                        "name": "Moonlit Gate",
                        "image_path": (
                            f"assets/cards/{card_id}/image-{revision_id}.png"
                        ),
                        "origin": "imported",
                        "source_filename": "gate.png",
                        "generation_metadata": None,
                        "hotspot_set": None,
                        "created_at": "2026-08-22T00:00:00Z",
                    }
                ],
                "active_revision_id": str(revision_id),
            }
        ],
    }

    migrated = migrate_document(document)

    assert migrated["schema_version"] == 2
    assert "global_style" not in migrated
    assert migrated["styles"][0]["name"] == "Default"
    assert migrated["styles"][0]["prompt"] == "Ink wash"
    card = migrated["cards"][0]
    assert "scene_description" not in card
    assert "interaction_description" not in card
    assert "image_revisions" not in card
    assert card["active_revision_id"] == str(revision_id)
    revision = card["revisions"][0]
    assert revision["description"] == "A moonlit courtyard"
    assert revision["style_id"] == migrated["styles"][0]["id"]
    assert revision["background"]["type"] == "imported"
    assert revision["background"]["id"] == str(revision_id)
    assert "name" not in revision
