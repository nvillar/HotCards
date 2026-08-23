"""Tests for deterministic author-controlled image prompts."""

from uuid import uuid4

import pytest

from hypergen.domain.models import ImageGenerationInputs, ImageReferenceSnapshot
from hypergen.generation.image_prompts import compose_image_prompt


def test_prompt_uses_the_authored_description_without_rewriting() -> None:
    description = "A stone courtyard at dusk in detailed ink and watercolor."

    assert compose_image_prompt(ImageGenerationInputs(description=description)) == description


def test_empty_or_whitespace_description_is_rejected() -> None:
    for description in ("", "   "):
        with pytest.raises(ValueError, match="Description"):
            compose_image_prompt(ImageGenerationInputs(description=description))


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
            identity_reference=identity,
            setting_reference=setting,
        )
    )

    assert prompt.index("REFERENCE IMAGE 1\nIDENTITY") < prompt.index("REFERENCE IMAGE 2\nSETTING")
    assert "REFERENCE IMAGE 3" not in prompt
    assert "SCENE AUTHORITY" in prompt
    assert "SCENE supplies actions, poses, object states, time, weather, camera, mood" in prompt
    assert "every property not assigned to a labeled reference role" in prompt
    assert "Derive the recognizable appearance and identity" in prompt
    assert "Derive environment, architecture, materials, terrain" in prompt
    assert "Ignore" not in prompt
    assert "Do not" not in prompt
    assert "\nVISUAL STYLE\n" not in prompt
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
            identity_reference=castle,
            setting_reference=castle,
        )
    )

    assert prompt.count("REFERENCE IMAGE") == 1
    assert "REFERENCE IMAGE 1\nIDENTITY + SETTING" in prompt
    assert prompt.count("SCENE AUTHORITY") == 1
    assert "REFERENCE IMAGE 2" not in prompt
