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

    assert prompt.startswith("An open circular hatch reveals")
    assert "The hatch is open." not in prompt
    assert "Use image 1 for the medium, linework, texture" in prompt
    assert "Preserve the described subjects, actions, poses" in prompt
    assert "SCENE AUTHORITY" not in prompt


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

    assert prompt.startswith("A knight crossing a market square")
    assert "A knight crossing a market square. Use image 1" in prompt
    assert prompt.index("Use image 1 for the subject's") < prompt.index(
        "Use image 2 for the environment"
    )
    assert "image 3" not in prompt
    assert "SCENE AUTHORITY" not in prompt
    assert "REFERENCE IMAGE" not in prompt
    assert "SUBJECT" not in prompt
    assert "SETTING" not in prompt
    assert "recognizable identity and appearance" in prompt
    assert "environment, architecture, intrinsic materials, terrain" in prompt
    assert "Ignore" not in prompt
    assert "Do not" not in prompt
    assert prompt.endswith(
        "Preserve the described subjects, actions, poses, object states, "
        "time, weather, viewpoint, framing, and composition."
    )


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

    assert prompt.count("Use image 1") == 1
    assert (
        "Use image 1 for the subject's recognizable identity and appearance, "
        "including its defining features, body, and clothing and the environment, "
        "architecture, intrinsic materials, terrain, and spatial character."
    ) in prompt
    assert "SCENE AUTHORITY" not in prompt
    assert "image 2" not in prompt


def test_reference_prompt_does_not_expose_internal_role_labels() -> None:
    style = ImageReferenceSnapshot(
        card_id=uuid4(),
        revision_id=uuid4(),
        background_id=uuid4(),
    )

    prompt = compose_image_prompt(
        ImageGenerationInputs(
            description=(
                "The computer fills the frame in a direct head-on view, "
                "showing its screen and top row of keys."
            ),
            style_reference=style,
        )
    )

    assert prompt.startswith("The computer fills the frame")
    assert "Use image 1 for the medium, linework, texture" in prompt
    assert all(
        label not in prompt
        for label in (
            "SCENE AUTHORITY",
            "REFERENCE IMAGE",
            "\nSCENE\n",
            "\nSTYLE\n",
        )
    )
