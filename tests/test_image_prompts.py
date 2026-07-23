"""Tests for deterministic author-controlled image prompts."""

import pytest

from hypergen.domain.models import ImageGenerationInputs
from hypergen.generation.image_prompts import compose_image_prompt


def test_prompt_combines_scene_and_global_style_in_stable_order() -> None:
    inputs = ImageGenerationInputs(
        scene_description="A stone courtyard at dusk",
        global_style="Ink and watercolor",
    )

    assert compose_image_prompt(inputs) == (
        "A stone courtyard at dusk\n\nInk and watercolor"
    )


def test_card_style_fully_replaces_global_style() -> None:
    inputs = ImageGenerationInputs(
        scene_description="A stone courtyard at dusk",
        global_style="Ink and watercolor",
        card_style="Cool photographic shadows",
    )

    assert compose_image_prompt(inputs) == (
        "A stone courtyard at dusk\n\nCool photographic shadows"
    )


def test_empty_scene_and_style_are_rejected() -> None:
    with pytest.raises(ValueError, match="Scene or Style"):
        compose_image_prompt(
            ImageGenerationInputs(scene_description="", global_style="")
        )


def test_whitespace_only_scene_is_omitted_when_style_is_present() -> None:
    assert compose_image_prompt(
        ImageGenerationInputs(
            scene_description="   ",
            global_style="Watercolor",
        )
    ) == "Watercolor"
