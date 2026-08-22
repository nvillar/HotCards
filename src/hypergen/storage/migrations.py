"""Explicit schema-version migration dispatch for stack documents."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Any
from uuid import UUID, uuid5

from hypergen.domain.models import CURRENT_SCHEMA_VERSION

JsonObject = dict[str, Any]
Migration = Callable[[JsonObject], JsonObject]


class MigrationError(ValueError):
    """A stack document cannot be migrated to the supported schema."""


def _unique_style_name(preferred: str, used_names: set[str]) -> str:
    base = preferred.strip() or "Style"
    candidate = base
    suffix = 2
    while candidate.casefold() in used_names:
        candidate = f"{base} {suffix}"
        suffix += 1
    used_names.add(candidate.casefold())
    return candidate


def _migrate_hotspot_set(value: object) -> object:
    """Drop obsolete generation provenance while preserving interactions."""
    if not isinstance(value, dict):
        return value
    hotspot_set = deepcopy(value)
    hotspot_set.pop("generation_provenance", None)
    return hotspot_set


def _migrate_v1_to_v2(document: JsonObject) -> JsonObject:
    """Move card-wide authoring data into complete card revisions."""
    stack_id = UUID(str(document["id"]))
    styles: list[JsonObject] = []
    style_ids_by_prompt: dict[str, str] = {}
    style_names_by_prompt: dict[str, str] = {}
    used_style_names: set[str] = set()

    def style_id_for(prompt: str, preferred_name: str) -> str | None:
        if not isinstance(prompt, str):
            raise MigrationError("style prompts must be strings")
        if not prompt.strip():
            return None
        existing = style_ids_by_prompt.get(prompt)
        if existing is not None:
            return existing
        style_id = str(uuid5(stack_id, f"style:{prompt}"))
        name = _unique_style_name(preferred_name, used_style_names)
        styles.append({"id": style_id, "name": name, "prompt": prompt})
        style_ids_by_prompt[prompt] = style_id
        style_names_by_prompt[prompt] = name
        return style_id

    global_style = document.pop("global_style", "")
    if not isinstance(global_style, str):
        raise MigrationError("stack global_style must be a string")
    style_id_for(global_style, "Default")

    cards = document.get("cards", [])
    if not isinstance(cards, list):
        raise MigrationError("stack cards must be an array")
    migrated_cards: list[JsonObject] = []
    for card_value in cards:
        if not isinstance(card_value, dict):
            raise MigrationError("stack cards must be objects")
        card = deepcopy(card_value)
        card_id = UUID(str(card["id"]))
        card_name = str(card.get("name", "Card"))
        description = card.pop("scene_description", "")
        card.pop("interaction_description", None)
        card_style = card.pop("card_style", None)
        if card_style is not None and not isinstance(card_style, str):
            raise MigrationError("card card_style must be a string or null")
        effective_style = card_style if card_style is not None else global_style
        preferred_style_name = (
            f"{card_name} Style" if card_style is not None else "Default"
        )
        selected_style_id = style_id_for(
            effective_style,
            preferred_style_name,
        )

        old_revisions = card.pop("image_revisions", [])
        if not isinstance(old_revisions, list):
            raise MigrationError("card image_revisions must be an array")
        revisions: list[JsonObject] = []
        for old_revision_value in old_revisions:
            if not isinstance(old_revision_value, dict):
                raise MigrationError("image revisions must be objects")
            old_revision = deepcopy(old_revision_value)
            revision_id = str(old_revision["id"])
            origin = old_revision["origin"]
            background: JsonObject = {
                "id": revision_id,
                "type": origin,
                "image_path": old_revision["image_path"],
                "created_at": old_revision["created_at"],
            }
            if origin == "generated":
                metadata = deepcopy(old_revision["generation_metadata"])
                if not isinstance(metadata, dict):
                    raise MigrationError(
                        "generated revision metadata must be an object"
                    )
                inputs = metadata["inputs"]
                if not isinstance(inputs, dict):
                    raise MigrationError(
                        "generated revision metadata inputs must be an object"
                    )
                historical_style = (
                    inputs.get("card_style")
                    if inputs.get("card_style") is not None
                    else inputs.get("global_style", "")
                )
                historical_name = style_names_by_prompt.get(historical_style)
                metadata["inputs"] = {
                    "description": inputs.get("scene_description", ""),
                    "style_name": historical_name,
                    "style_prompt": historical_style,
                }
                background["generation_metadata"] = metadata
            elif origin == "imported":
                background["source_filename"] = old_revision.get("source_filename")
            else:
                raise MigrationError(f"unsupported image origin: {origin!r}")
            revisions.append(
                {
                    "id": revision_id,
                    "description": description,
                    "style_id": selected_style_id,
                    "background": background,
                    "hotspot_set": _migrate_hotspot_set(
                        old_revision.get("hotspot_set")
                    ),
                }
            )

        if not revisions:
            revision_id = str(uuid5(card_id, "revision:1"))
            revisions.append(
                {
                    "id": revision_id,
                    "description": description,
                    "style_id": selected_style_id,
                    "background": None,
                    "hotspot_set": None,
                }
            )
            active_revision_id = revision_id
        else:
            active_revision_id = card.get("active_revision_id")
        card["revisions"] = revisions
        card["active_revision_id"] = active_revision_id
        migrated_cards.append(card)

    document["styles"] = styles
    document["cards"] = migrated_cards
    document["schema_version"] = 2
    return document


def _migrate_v2_to_v3(document: JsonObject) -> JsonObject:
    """Remove provenance from the retired hotspot-generation workflow."""
    cards = document.get("cards", [])
    if not isinstance(cards, list):
        raise MigrationError("stack cards must be an array")
    for card in cards:
        if not isinstance(card, dict):
            raise MigrationError("stack cards must be objects")
        revisions = card.get("revisions", [])
        if not isinstance(revisions, list):
            raise MigrationError("card revisions must be an array")
        for revision in revisions:
            if not isinstance(revision, dict):
                raise MigrationError("card revisions must be objects")
            revision["hotspot_set"] = _migrate_hotspot_set(
                revision.get("hotspot_set")
            )
    document["schema_version"] = 3
    return document


DEFAULT_MIGRATIONS: dict[int, Migration] = {
    1: _migrate_v1_to_v2,
    2: _migrate_v2_to_v3,
}


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
    registry = DEFAULT_MIGRATIONS if migrations is None else migrations
    while version < target_version:
        migration = registry.get(version)
        if migration is None:
            raise MigrationError(
                f"stack schema version {version} is unsupported; "
                f"no migration to version {version + 1} is registered"
            )
        try:
            migrated = migration(deepcopy(migrated))
        except MigrationError:
            raise
        except (KeyError, TypeError, ValueError) as error:
            raise MigrationError(
                f"stack schema version {version} has invalid migration data: {error}"
            ) from error
        next_version = _schema_version(migrated)
        if next_version != version + 1:
            raise MigrationError(
                f"migration from version {version} must produce version {version + 1}, "
                f"not {next_version}"
            )
        version = next_version
    return migrated
