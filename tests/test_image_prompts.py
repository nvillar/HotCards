"""Tests for deterministic author-controlled image prompts."""

from uuid import uuid4

import pytest

from hypergen.domain.models import ImageGenerationInputs, ImageReferenceSnapshot
from hypergen.generation.image_prompts import compose_image_prompt


def test_prompt_uses_the_authored_description_without_rewriting() -> None:
    description = "A stone courtyard at dusk in detailed ink and watercolor."

    assert compose_image_prompt(ImageGenerationInputs(description=description)) == description


def test_prompt_uses_only_enriched_description_when_present() -> None:
    prompt = compose_image_prompt(
        ImageGenerationInputs(
            description="The hatch is open.",
            enriched_description=(
                "A closed circular space-station hatch in a monochrome corridor."
            ),
        )
    )

    assert prompt == (
        "A closed circular space-station hatch in a monochrome corridor."
    )
    assert "The hatch is open." not in prompt


def test_referenced_prompt_uses_only_effective_enriched_description() -> None:
    snapshot = ImageReferenceSnapshot(
        card_id=uuid4(),
        revision_id=uuid4(),
        background_id=uuid4(),
    )
    prompt = compose_image_prompt(
        ImageGenerationInputs(
            description="The hatch is open.",
            enriched_description=(
                "An open circular hatch reveals a monochrome station corridor."
            ),
            style_reference=snapshot,
        )
    )

    assert "SCENE\nAn open circular hatch reveals" in prompt
    assert "The hatch is open." not in prompt
    assert "REFERENCE IMAGE 1\nSTYLE" in prompt


def test_both_empty_description_sources_are_rejected() -> None:
    for description in ("", "   "):
        with pytest.raises(ValueError, match="Description"):
            compose_image_prompt(
                ImageGenerationInputs(
                    description=description,
                    enriched_description="   ",
                )
            )


def test_enriched_description_can_be_the_only_text_source() -> None:
    enriched = "A monochrome circular hatch opens into a station corridor."

    assert compose_image_prompt(
        ImageGenerationInputs(
            description="",
            enriched_description=enriched,
        )
    ) == enriched


def test_reference_roles_are_ordered_and_missing_slots_are_renumbered() -> None:
    identity = ImageReferenceSnapshot(
        card_id=uuid4(),
        revision_id=uuid4(),
        background_id=uuid4(),
    )
    setting = ImageReferenceSnapshot(
        card_id=uuid4(),
        revision_id=uuid4(),
        background_id=uuid4(),
    )

    prompt = compose_image_prompt(
        ImageGenerationInputs(
            description="A knight crossing a market square",
            subject_reference=identity,
            setting_reference=setting,
        )
    )

    assert prompt.index("REFERENCE IMAGE 1\nSUBJECT") < prompt.index("REFERENCE IMAGE 2\nSETTING")
    assert "REFERENCE IMAGE 3" not in prompt
    assert "SCENE AUTHORITY" in prompt
    assert "SCENE supplies actions, poses, object states" in prompt
    assert "Derive the recognizable appearance and identity" in prompt
    assert "Derive environment, architecture, materials, terrain" in prompt
    assert "Ignore" not in prompt
    assert "Do not" not in prompt
    assert "\nSTYLE\n" not in prompt
    assert prompt.endswith("SCENE\nA knight crossing a market square")


def test_one_reference_image_can_supply_multiple_roles() -> None:
    castle = ImageReferenceSnapshot(
        card_id=uuid4(),
        revision_id=uuid4(),
        background_id=uuid4(),
    )

    prompt = compose_image_prompt(
        ImageGenerationInputs(
            description="The outer gate of the same castle",
            subject_reference=castle,
            setting_reference=castle,
        )
    )

    assert prompt.count("REFERENCE IMAGE") == 1
    assert "REFERENCE IMAGE 1\nSUBJECT + SETTING" in prompt
    assert prompt.count("SCENE AUTHORITY") == 1
    assert "REFERENCE IMAGE 2" not in prompt
