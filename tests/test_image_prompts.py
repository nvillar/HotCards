"""Tests for deterministic author-controlled image prompts."""

import pytest

from hypergen.domain.models import ImageGenerationInputs
from hypergen.generation.image_prompts import compose_image_prompt


def test_prompt_combines_description_and_style_in_stable_order() -> None:
    inputs = ImageGenerationInputs(
        description="A stone courtyard at dusk",
        style_name="Storybook",
        style_prompt="Ink and watercolor",
    )

    assert compose_image_prompt(inputs) == (
        "A stone courtyard at dusk\n\nInk and watercolor"
    )


def test_style_name_is_not_rendered_as_prompt_text() -> None:
    inputs = ImageGenerationInputs(
        description="A stone courtyard at dusk",
        style_name="Noir",
        style_prompt="Cool photographic shadows",
    )

    assert compose_image_prompt(inputs) == (
        "A stone courtyard at dusk\n\nCool photographic shadows"
    )


def test_empty_scene_and_style_are_rejected() -> None:
    with pytest.raises(ValueError, match="Description or Style"):
        compose_image_prompt(
            ImageGenerationInputs(description="", style_prompt="")
        )


def test_whitespace_only_scene_is_omitted_when_style_is_present() -> None:
    assert compose_image_prompt(
        ImageGenerationInputs(
            description="   ",
            style_prompt="Watercolor",
        )
    ) == "Watercolor"
